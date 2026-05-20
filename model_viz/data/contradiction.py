"""ContradictionPairsDataset — eval-only paired statements for the contradiction probe.

Each example is a :class:`ContradictionPairSpec` with three short sentences::

    a              — Fact 1 (the premise).
    b_contradict   — Fact 2 that contradicts ``a``.
    b_compatible   — Fact 2 that is consistent with ``a`` (matched control).

The matched control is critical: it isolates "contradiction" from the
confound of "two facts about the same subject", which has its own signature
in hidden-state space.

Loaded from ``data/contradiction/pairs.json``.  See that file for the
schema and the three categories (``syntactic_negation``,
``semantic_contradiction``, ``temporal_update``).

Capability surface: ``EvalCapable`` only.  No input plumbing — the
``ContradictionVisualizer`` already reaches into its loaded HF adapter for
the tokenizer and never asks the dataset to encode anything.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional

from model_viz.data.base import DatasetInfo


_DEFAULT_RELATIVE_PATH = Path("data") / "contradiction" / "pairs.json"


@dataclass(frozen=True)
class ContradictionPairSpec:
    category: str
    a: str
    b_contradict: str
    b_compatible: str


class ContradictionPairsUnavailable(RuntimeError):
    """Raised when the pairs JSON cannot be found or parsed."""


class ContradictionPairsDataset:
    """``EvalCapable`` corpus of (contradiction, matched-compatible) pair specs.

    Iteration is eager — the file is small (~50 entries) and read once at
    construction.  Re-reading on each ``iter_examples()`` call would only
    matter for a live-editable corpus, which this isn't.
    """

    name = "contradiction-pairs"
    info = DatasetInfo(
        vocab_size=None,
        num_classes=None,
        description=(
            "Paired (contradiction, matched-compatible) statements across three "
            "categories: syntactic_negation, semantic_contradiction, temporal_update."
        ),
    )

    def __init__(self, path: Optional[Path] = None) -> None:
        if path is None:
            repo_root = Path(__file__).resolve().parents[2]
            path = repo_root / _DEFAULT_RELATIVE_PATH
        self._path = Path(path)
        if not self._path.is_file():
            raise ContradictionPairsUnavailable(
                f"Contradiction pairs not found at {self._path}."
            )
        try:
            obj = json.loads(self._path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise ContradictionPairsUnavailable(
                f"Failed to parse {self._path}: {e}"
            ) from e
        raw_pairs = obj.get("pairs", [])
        self._pairs: List[ContradictionPairSpec] = [
            ContradictionPairSpec(
                category=str(p.get("category", "uncategorized")),
                a=str(p["a"]),
                b_contradict=str(p["b_contradict"]),
                b_compatible=str(p["b_compatible"]),
            )
            for p in raw_pairs
        ]

    @property
    def path(self) -> Path:
        return self._path

    def __len__(self) -> int:
        return len(self._pairs)

    # ------------------------------------------------------------------
    # EvalCapable
    # ------------------------------------------------------------------

    def iter_examples(self) -> Iterator[ContradictionPairSpec]:
        return iter(self._pairs)


def try_load() -> Optional[ContradictionPairsDataset]:
    """Best-effort loader.  Returns ``None`` if the JSON is missing."""
    try:
        return ContradictionPairsDataset()
    except ContradictionPairsUnavailable:
        return None
