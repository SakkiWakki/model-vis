"""PerplexityVisualizer: per-token perplexity over a user-supplied sequence.

Compatible only with model outputs of shape ``(B, T, V)`` (per-position logits
over a vocab) — i.e. language-model style heads, not classifiers.  The
visualizer carries its own text input field; it runs ``adapter.forward`` on
the supplied sequence and renders perplexity per next-token prediction.
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from model_viz.core.layer import LayerLike
from model_viz.viz.visualizer_base import VisualizerBase
from model_viz.viz.visualizers.perplexity_viz.components.token_chips import TokenChipStrip


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
        self._strip = TokenChipStrip()
        self._input = QLineEdit()
        self._input.setPlaceholderText("Enter text to score…")
        self._run_btn = QPushButton("Score")
        self._status = QLabel("")
        super().__init__(layer=layer, adapter=adapter, parent=parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)
        layout.addWidget(self._strip, stretch=1)
        layout.addWidget(self._status)

        ctrl = QHBoxLayout()
        ctrl.setContentsMargins(0, 0, 0, 0)
        ctrl.addWidget(self._input, stretch=1)
        ctrl.addWidget(self._run_btn)
        layout.addLayout(ctrl)

        self._run_btn.clicked.connect(self._score)
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

        tokens, values = self._compute(token_ids, logits, dataset)
        if not tokens:
            self._status.setText("Empty sequence.")
            return
        self._strip.set_tokens(tokens, values)
        finite = [v for v in values if v is not None and math.isfinite(v)]
        if finite:
            avg = sum(finite) / len(finite)
            self._status.setText(f"{len(tokens)} tokens • mean perplexity {avg:.3f}")
        else:
            self._status.setText(f"{len(tokens)} tokens")

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
