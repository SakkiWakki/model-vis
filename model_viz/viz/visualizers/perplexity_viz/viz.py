"""PerplexityVisualizer: per-token perplexity over user-supplied sequences.

Compatible only with model outputs of shape ``(B, T, V)`` (per-position logits
over a vocab) — i.e. language-model style heads, not classifiers.  The
visualizer carries its own text input field; it runs ``adapter.forward`` on
the supplied sequence and renders perplexity per next-token prediction.

Multi-query comparison
----------------------
Each ``Score`` click appends a new ``QueryResult`` to ``self._queries`` and
re-renders every strip on a *shared* color scale.  This is the operationally
meaningful behavior for comparing variants of the same sentence — without a
shared scale, two sentences with different perplexity ranges look identical
because each strip auto-scales independently.  ``Clear`` empties the list.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from model_viz.core.layer import LayerLike
from model_viz.viz.visualizer_base import VisualizerBase
from model_viz.viz.visualizers.perplexity_viz.components.token_chips import TokenChipStrip


@dataclass
class QueryResult:
    """A single scored sentence retained for cross-query comparison."""

    text: str
    tokens: List[str]
    values: List[Optional[float]]
    geom_mean: Optional[float]


def _final_logits(outputs: object) -> Optional[torch.Tensor]:
    """Return a (B, T, V) tensor if outputs match that shape, else None."""
    if isinstance(outputs, torch.Tensor) and outputs.ndim == 3 and outputs.shape[-1] > 1:
        return outputs
    if isinstance(outputs, (tuple, list)) and outputs:
        first = outputs[0]
        if isinstance(first, torch.Tensor) and first.ndim == 3 and first.shape[-1] > 1:
            return first
    return None


class PerplexityVisualizer(VisualizerBase):
    display_name = "Perplexity"

    def __init__(
        self,
        layer: LayerLike,
        adapter: Optional[object] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        # Per-query state.  One ``QueryResult`` per ``Score`` click; cleared by
        # the Clear button.  ``_render_all`` rebuilds the strips on demand.
        self._queries: List[QueryResult] = []

        # Vertical container of stacked strips, inside a scroll area so many
        # queries don't overflow the panel.
        self._strips_scroll = QScrollArea()
        self._strips_scroll.setWidgetResizable(True)
        self._strips_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self._strips_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._strips_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        self._strips_container = QWidget()
        self._strips_layout = QVBoxLayout(self._strips_container)
        self._strips_layout.setContentsMargins(0, 0, 0, 0)
        self._strips_layout.setSpacing(8)
        self._strips_layout.addStretch(1)  # keep strips pinned to the top
        self._strips_scroll.setWidget(self._strips_container)

        self._input = QLineEdit()
        self._input.setPlaceholderText("Enter text to score…")
        self._run_btn = QPushButton("Score")
        self._clear_btn = QPushButton("Clear")
        self._status = QLabel("")
        super().__init__(layer=layer, adapter=adapter, parent=parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)
        layout.addWidget(self._strips_scroll, stretch=1)
        layout.addWidget(self._status)

        ctrl = QHBoxLayout()
        ctrl.setContentsMargins(0, 0, 0, 0)
        ctrl.addWidget(self._input, stretch=1)
        ctrl.addWidget(self._run_btn)
        ctrl.addWidget(self._clear_btn)
        layout.addLayout(ctrl)

        self._run_btn.clicked.connect(self._score)
        self._clear_btn.clicked.connect(self._clear)
        self._input.returnPressed.connect(self._score)

    # ------------------------------------------------------------------
    @classmethod
    def compatible_with(cls, layer: LayerLike) -> bool:
        # Only the model's final leaf can produce the next-token logits we need.
        if getattr(layer, "next_layer", None) is not None:
            return False
        # Live check: a forward pass has already cached (B, T, V) logits here.
        if _final_logits(layer.outputs()) is not None:
            return True
        # Structural check: the owning adapter declares it will produce
        # per-position logits at this leaf (causal-LM head).  Lets us surface
        # the visualizer before any forward pass has run.
        adapter = getattr(layer, "_adapter", None)
        return bool(getattr(adapter, "produces_per_position_logits", False))

    # ------------------------------------------------------------------
    def refresh(self) -> None:
        # Nothing to do on auto-refresh — perplexity needs an explicit input.
        pass

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _clear(self) -> None:
        self._queries.clear()
        self._render_all()
        self._status.setText("")

    def _score(self) -> None:
        adapter = self._adapter
        if adapter is None:
            self._status.setText("No adapter available.")
            return
        dataset = getattr(adapter, "_dataset", None)
        if dataset is None:
            self._status.setText("Adapter has no dataset for tokenization.")
            return

        text = self._input.text().strip()
        if not text:
            self._status.setText("Enter text to score.")
            return

        try:
            inp = dataset.make_input(text)
        except Exception as e:
            self._status.setText(f"Tokenization failed: {e}")
            return
        if inp is None:
            self._status.setText("Tokenization returned no input.")
            return

        try:
            adapter.forward(inp)
        except Exception as e:
            self._status.setText(f"Forward failed: {e}")
            return

        logits = _final_logits(self._layer.outputs())
        if logits is None:
            self._status.setText("Final layer did not produce per-token logits.")
            return

        try:
            token_ids = inp.to_tensor()
        except Exception as e:
            self._status.setText(f"Could not retrieve token IDs: {e}")
            return
        if token_ids is None:
            self._status.setText("Tokenizer produced no token ids.")
            return

        tokens, values = self._compute(token_ids, logits, dataset)
        if not tokens:
            self._status.setText("Empty sequence.")
            return

        # Sequence-level perplexity is the geometric mean of per-token
        # perplexity (equivalently exp of the mean NLL): one outlier no
        # longer dominates an otherwise-confident sentence the way an
        # arithmetic mean would.  This matches the standard definition of
        # perplexity-of-a-sequence used in the language-modeling literature.
        finite_nlls = [
            math.log(v) for v in values
            if v is not None and math.isfinite(v) and v > 0
        ]
        geom_mean: Optional[float] = (
            math.exp(sum(finite_nlls) / len(finite_nlls)) if finite_nlls else None
        )

        self._queries.append(
            QueryResult(text=text, tokens=tokens, values=values, geom_mean=geom_mean)
        )
        self._render_all()

        # Status line: total query count + last sentence + its geom-mean.
        bits: List[str] = [f"{len(self._queries)} queries", f"last: {text}"]
        if geom_mean is not None:
            bits.append(f"geom-mean perplexity {geom_mean:.3f}")
        # Surface truncation as a tail so it isn't swallowed silently.
        n_displayed = len(tokens)
        n_input = int(token_ids.shape[-1]) if token_ids.ndim >= 1 else n_displayed
        if n_displayed < n_input:
            bits.append(f"showing {n_displayed}/{n_input} tokens (model truncated)")
        self._status.setText(" • ".join(bits))

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render_all(self) -> None:
        """Tear down existing strips, recompute the shared scale, and rebuild.

        The shared scale is the min/max of *all* non-None values across every
        query.  This is what makes per-token chips comparable across queries —
        a verb-position chip in a "wrong" sentence renders visibly redder
        than the same position in a "correct" one only if they share a scale.
        """
        # Remove every widget from the strips layout, keeping the trailing
        # stretch in place so a single strip pins to the top.
        while self._strips_layout.count() > 1:
            item = self._strips_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

        if not self._queries:
            return

        # Compute global vmin/vmax over all finite, positive values.  Falling
        # back to ``None`` lets each strip use its own range if no shared
        # range can be determined (e.g. every value is None).
        all_values: List[float] = []
        for q in self._queries:
            for v in q.values:
                if v is not None and math.isfinite(v) and v > 0:
                    all_values.append(v)
        shared_vmin: Optional[float] = min(all_values) if all_values else None
        shared_vmax: Optional[float] = max(all_values) if all_values else None

        for q in self._queries:
            # Small label above each strip so stacked rows are identifiable.
            label = QLabel(self._strip_label_text(q))
            label.setStyleSheet(
                "color: rgba(220, 220, 220, 0.80); font-size: 10px; padding: 0 2px;"
            )
            label.setWordWrap(True)
            self._strips_layout.insertWidget(self._strips_layout.count() - 1, label)

            strip = TokenChipStrip()
            strip.set_tokens(q.tokens, q.values, vmin=shared_vmin, vmax=shared_vmax)
            self._strips_layout.insertWidget(self._strips_layout.count() - 1, strip)

    @staticmethod
    def _strip_label_text(q: QueryResult) -> str:
        if q.geom_mean is None:
            return q.text
        return f"{q.text}  ·  geom-mean {q.geom_mean:.3f}"

    # ------------------------------------------------------------------
    # Compute
    # ------------------------------------------------------------------

    def _compute(
        self,
        token_ids: torch.Tensor,
        logits: torch.Tensor,
        dataset: object,
    ) -> Tuple[List[str], List[Optional[float]]]:
        # token_ids: (B, T) or (T,); logits: (B, T, V).  Use batch 0.
        ids = token_ids.detach().cpu()
        if ids.ndim == 2:
            ids = ids[0]
        ids = ids.to(torch.long).tolist()

        L = logits.detach().float().cpu()
        if L.ndim == 3:
            L = L[0]   # (T, V)
        T, V = L.shape

        # The model's logits at position i predict the token at position i+1.
        # So perplexity for token at position i (i >= 1) uses logits row i-1.
        log_probs = torch.log_softmax(L, dim=-1)

        n = min(len(ids), T)
        decode = getattr(dataset, "decode_token", None)
        tokens: List[str] = []
        values: List[Optional[float]] = []
        for i in range(n):
            tok = decode(ids[i]) if callable(decode) else str(ids[i])
            tokens.append(tok)
            if i == 0:
                values.append(None)  # no preceding context
                continue
            tgt = ids[i]
            if tgt < 0 or tgt >= V:
                values.append(None)
                continue
            nll = -float(log_probs[i - 1, tgt].item())
            values.append(math.exp(nll))
        return tokens, values
