"""Cambridge FCE Public Dataset (CLC-FCE) — paragraph-level original/corrected pairs.

Source
------
The Cambridge Learner Corpus FCE Public Dataset, distributed by Cambridge ESOL
under a data-use licence.  Not redistributed with this repo; the user obtains
``fce-released-dataset.zip`` themselves and runs ``scripts/setup_fce.sh`` to
extract it under ``~/datasets/`` and symlink it into ``data/fce-released-dataset``.

Each XML script has the structure::

    <learner>
      <head>
        <text>
          <answer1|2>
            <coded_answer>
              <p>...text with <NS type="..."><i>err</i><c>fix</c></NS>...</p>
              ...
            </coded_answer>
          </answer1|2>
        </text>
      </head>
    </learner>

Each ``<NS>`` correction node has zero or one ``<i>`` (incorrect-as-written)
child and zero or one ``<c>`` (corrected) child.  Asymmetric edits are
faithfully reproduced: a deletion (only ``<i>``) keeps the text in the
*original* and drops it from the *corrected*; an insertion (only ``<c>``)
does the reverse.

Iteration unit is one paragraph.  Sentence segmentation is intentionally
avoided — paragraph context preserves alignment of cross-sentence errors,
and paragraph lengths fit comfortably in GPT-2's 1024-token window.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, List, Optional
from xml.etree import ElementTree as ET

import torch

from model_viz.core.input.text_input import TextInput
from model_viz.core.output.class_label import ClassLabelOutput
from model_viz.data.base import DatasetInfo


# Where the dataset lives relative to the repo root.  The setup script
# creates this as a symlink to wherever the user keeps the extracted corpus.
_DEFAULT_RELATIVE_PATH = Path("data") / "fce-released-dataset"


@dataclass(frozen=True)
class FCEPair:
    """One paragraph rendered as both the learner-original and the gold-corrected form.

    ``error_types`` is the (possibly empty) list of `NS type=` codes that
    appeared in this paragraph (e.g. ``["RN", "AGV", "S"]``), in document order.
    Useful for downstream slicing (e.g. "show me only spelling errors").
    """

    original: str
    corrected: str
    error_types: Tuple[str, ...]
    source_file: str  # path of the originating XML, relative to dataset root


class FCEUnavailable(RuntimeError):
    """Raised when FCE's on-disk location does not exist or is malformed."""


class FCEDataset:
    """Read-only corpus of FCE learner-original / gold-corrected paragraph pairs.

    Implements ``InputCapable`` (for typing free-form text into a visualizer
    using the dataset's whitespace fallback) and ``EvalCapable`` (for offline
    iteration over the corpus).  Does **not** implement ``TrainCapable`` — the
    training UI will simply not surface for this dataset.

    Iteration is lazy: ``iter_pairs()`` / ``iter_examples()`` yields one
    ``FCEPair`` at a time so the full 1244-file corpus is never materialized
    in memory.
    """

    name = "fce"
    input_type = TextInput
    output_type = ClassLabelOutput
    info = DatasetInfo(
        vocab_size=None,
        num_classes=None,
        description=(
            "Cambridge FCE Public Dataset — learner-original/gold-corrected "
            "paragraph pairs.  Inference-only; not for training."
        ),
    )

    def __init__(self, root: Optional[Path] = None) -> None:
        if root is None:
            # Resolve relative to the repo (this file is at
            # model_viz/data/fce.py, so go up two levels).
            repo_root = Path(__file__).resolve().parents[2]
            root = repo_root / _DEFAULT_RELATIVE_PATH
        self._root = Path(root)

        if not self._root.exists():
            raise FCEUnavailable(
                f"FCE dataset not found at {self._root}.\n"
                "Run scripts/setup_fce.sh to install it (you must obtain "
                "fce-released-dataset.zip from Cambridge yourself)."
            )
        if not (self._root / "dataset").is_dir():
            raise FCEUnavailable(
                f"FCE directory at {self._root} is missing the 'dataset/' subdir."
            )

    # ------------------------------------------------------------------
    # Corpus iteration (the actual point of this class).
    # ------------------------------------------------------------------

    def iter_pairs(self) -> Iterator[FCEPair]:
        """Yield ``FCEPair`` instances, one per paragraph, lazily across all XML files.

        Files are sorted so iteration order is deterministic between runs.
        """
        dataset_dir = self._root / "dataset"
        for xml_path in sorted(dataset_dir.rglob("*.xml")):
            try:
                tree = ET.parse(xml_path)
            except ET.ParseError:
                # A small handful of files have malformed XML; skip silently.
                continue
            root = tree.getroot()
            rel = str(xml_path.relative_to(self._root))
            for p in root.iter("p"):
                pair = _paragraph_to_pair(p, source_file=rel)
                if pair is not None:
                    yield pair

    def __iter__(self) -> Iterator[FCEPair]:
        return self.iter_pairs()

    # ------------------------------------------------------------------
    # EvalCapable
    # ------------------------------------------------------------------

    def iter_examples(self) -> Iterator[FCEPair]:
        """``EvalCapable`` entrypoint.  Aliases :meth:`iter_pairs` for the protocol."""
        return self.iter_pairs()

    # ------------------------------------------------------------------
    # InputCapable — bare-minimum plumbing so the dataset can be selected
    # in the sidebar.  The intended workflow is to load a model (e.g. GPT-2)
    # whose adapter brings its own tokenizer; this class's tokenizer is a
    # coarse whitespace fallback.
    # ------------------------------------------------------------------

    def make_input(self, raw: Any) -> TextInput:
        if not isinstance(raw, str):
            raise ValueError("FCEDataset expects a string input.")
        return TextInput(raw, tokenizer=_whitespace_encode)

    def probe_input(self) -> TextInput:
        return TextInput(
            "Dear Sir, I am writing with reference to your article.",
            tokenizer=_whitespace_encode,
        )

    def interpret_output(self, raw: Any) -> ClassLabelOutput:
        # Mirror the inference-only datasets: if the model produces (B, T, V),
        # show the top-1 next token at the last position as a stand-in.
        if not isinstance(raw, torch.Tensor):
            raise ValueError("FCEDataset expects a tensor output from the model.")
        t = raw.detach().float().cpu()
        if t.ndim == 3:
            logits = t[0, -1]
        elif t.ndim == 2:
            logits = t[0]
        else:
            logits = t
        probs = torch.softmax(logits, dim=-1).tolist()
        class_id = int(torch.argmax(logits).item())
        return ClassLabelOutput(class_id=class_id, probs=probs)

    def decode_token(self, token_id: int) -> str:
        return str(token_id)


# ----------------------------------------------------------------------
# Parsing helpers
# ----------------------------------------------------------------------


def _paragraph_to_pair(p: ET.Element, *, source_file: str) -> Optional[FCEPair]:
    """Convert a <p>...</p> element into an (original, corrected) pair.

    Returns ``None`` if both renderings are empty.
    """
    original_parts: List[str] = []
    corrected_parts: List[str] = []
    error_types: List[str] = []
    _render(p, original_parts, corrected_parts, error_types, prefer="i")
    original = _normalize_whitespace("".join(original_parts))
    corrected = _normalize_whitespace("".join(corrected_parts))
    if not original and not corrected:
        return None
    return FCEPair(
        original=original,
        corrected=corrected,
        error_types=tuple(error_types),
        source_file=source_file,
    )


def _render(
    elem: ET.Element,
    original_parts: List[str],
    corrected_parts: List[str],
    error_types: List[str],
    *,
    prefer: str,
) -> None:
    """Walk ``elem``, accumulating the two text renderings.

    ``prefer`` is unused at top level; it exists so that recursion into
    nested ``<NS>`` nodes preserves the choice "<i> goes to original, <c>
    goes to corrected".  ``<NS>`` may nest — a typo inside a re-written
    phrase, for example — and we resolve that case by treating the
    contents of <i>/<c> wrappers as their own little subtrees.
    """
    # Text immediately inside this element (before its first child).
    if elem.text:
        original_parts.append(elem.text)
        corrected_parts.append(elem.text)

    for child in elem:
        tag = child.tag
        if tag == "NS":
            etype = child.attrib.get("type", "")
            if etype:
                error_types.append(etype)
            i_node = child.find("i")
            c_node = child.find("c")
            # <i> (incorrect as written) -> original side.  Recurse so any
            # nested <NS> inside also resolves correctly.
            if i_node is not None:
                _emit_branch(i_node, original_parts, error_types, side="i")
            # <c> (gold-corrected) -> corrected side.
            if c_node is not None:
                _emit_branch(c_node, corrected_parts, error_types, side="c")
        else:
            # Non-NS child element (e.g. inline markup we don't recognize) —
            # treat it as a passthrough: render its text content on both sides.
            _render(child, original_parts, corrected_parts, error_types, prefer=prefer)

        # Tail text immediately after this child (and before the next).
        if child.tail:
            original_parts.append(child.tail)
            corrected_parts.append(child.tail)


def _emit_branch(
    branch: ET.Element,
    out_parts: List[str],
    error_types: List[str],
    *,
    side: str,
) -> None:
    """Emit one side (<i> or <c>) of an NS span into ``out_parts``.

    A branch may contain nested <NS> spans (e.g. "<i>" wraps another
    correction).  For that case the *outer* side determines which inner
    side we follow:
      - inside an <i>, nested NS contributes its own <i> (original-of-original).
      - inside a <c>, nested NS contributes its own <c> (corrected-of-corrected).
    """
    if branch.text:
        out_parts.append(branch.text)
    for sub in branch:
        if sub.tag == "NS":
            etype = sub.attrib.get("type", "")
            if etype:
                error_types.append(etype)
            inner = sub.find(side)
            if inner is not None:
                _emit_branch(inner, out_parts, error_types, side=side)
            # No matching inner side on this nested NS — silently skip (it
            # wasn't part of this rendering anyway).
        else:
            # Non-NS content (rare: stray inline markup).
            _emit_branch(sub, out_parts, error_types, side=side)
        if sub.tail:
            out_parts.append(sub.tail)


_WS_RE = re.compile(r"\s+")


def _normalize_whitespace(s: str) -> str:
    return _WS_RE.sub(" ", s).strip()


# ----------------------------------------------------------------------
# Whitespace fallback tokenizer.
# ----------------------------------------------------------------------


def _whitespace_encode(text: str) -> torch.Tensor:
    """Crude whitespace tokenizer using a hash-bucketed vocab.

    This is a stub — FCEDataset is intended for use with adapters that bring
    their own tokenizer (HF causal LMs).  It exists only so that selecting
    FCE in the sidebar with a non-HF adapter doesn't immediately crash.
    """
    toks = text.strip().split() or [""]
    # Bucket into a fixed 50k-token pseudo-vocab via Python's stable hash.
    ids = [(hash(t) & 0xFFFF) % 50000 for t in toks]
    return torch.tensor([ids], dtype=torch.long)


def _env_root() -> Optional[Path]:
    """Allow $FCE_PATH to override the in-repo symlink location."""
    p = os.environ.get("FCE_PATH")
    return Path(p) if p else None


def try_load() -> Optional[FCEDataset]:
    """Best-effort loader: returns the dataset if available, else ``None``.

    Used by ``main.py`` so the framework keeps running when FCE isn't
    installed.  Errors are swallowed and surfaced as ``None``; callers that
    need diagnostics should construct ``FCEDataset()`` directly.
    """
    try:
        return FCEDataset(root=_env_root())
    except FCEUnavailable:
        return None
