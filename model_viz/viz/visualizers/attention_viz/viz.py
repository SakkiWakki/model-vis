"""AttentionVisualizer: displays per-head attention weight matrices as heatmaps."""
from __future__ import annotations

from typing import Optional

import torch
from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from model_viz.core.layer import LayerLike
from model_viz.viz.visualizer_base import VisualizerBase
from model_viz.viz.components.heatmap import HeatmapWidget


def _extract_attn(outputs: object) -> Optional[torch.Tensor]:
    """Find an attention-weight tensor inside a module's forward output.

    Handles nn.MultiheadAttention's ``(out, weights)``, HF attention's
    ``(hidden, attn_weights, past_kv)`` (where attn_weights may be at index 1
    or be ``None`` when SDPA elides them), and any object with an ``.attn``
    attribute.  An attention-weight tensor is identified by being 2D-4D with
    the last two dims equal (a square S x S attention matrix).
    """
    def _looks_like_attn(t: object) -> bool:
        if not isinstance(t, torch.Tensor):
            return False
        if t.ndim < 2 or t.ndim > 4:
            return False
        return t.shape[-1] == t.shape[-2]

    if isinstance(outputs, (tuple, list)):
        for item in outputs[1:]:  # skip the primary hidden-state output at index 0
            if _looks_like_attn(item):
                return item  # type: ignore[return-value]
    if hasattr(outputs, "attn"):
        a = outputs.attn
        if _looks_like_attn(a):
            return a  # type: ignore[return-value]
    return None


def _has_attn_weights(outputs: object) -> bool:
    return _extract_attn(outputs) is not None


class AttentionVisualizer(VisualizerBase):
    display_name = "Attention Weights"

    def __init__(
        self,
        layer: LayerLike,
        adapter: Optional[object] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        self._heatmaps: list[HeatmapWidget] = []
        self._head_combo: Optional[QComboBox] = None
        self._attn: Optional[torch.Tensor] = None
        self._attn_4d: Optional[torch.Tensor] = None
        super().__init__(layer=layer, adapter=adapter, parent=parent)

    @classmethod
    def compatible_with(cls, layer: LayerLike) -> bool:
        return "attn" in layer.name.lower() and _has_attn_weights(layer.outputs())

    def set_layer(self, layer: LayerLike) -> None:
        self._layer = layer

    def refresh(self) -> None:
        self._attn = _extract_attn(self._layer.outputs())
        if self._attn is None:
            self._attn_4d = None
            self._rebuild_ui()
            return
        # attn shape: (batch, heads, seq, seq) or (heads, seq, seq) or (seq, seq)
        t = self._attn.detach().float().cpu()
        if t.ndim == 2:
            t = t.unsqueeze(0).unsqueeze(0)  # (1,1,S,S)
        elif t.ndim == 3:
            t = t.unsqueeze(0)              # (1,H,S,S)
        self._attn_4d = t  # (B, H, S, S)
        self._rebuild_ui()

    # ------------------------------------------------------------------
    def _rebuild_ui(self) -> None:
        # Ensure we have exactly one layout on this widget, and reuse it.
        layout = self.layout()
        if layout is None:
            layout = QVBoxLayout()
            layout.setContentsMargins(4, 4, 4, 4)
            self.setLayout(layout)

        # Clear existing layout children (widgets only; no nested layouts used here).
        while layout.count():
            item = layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()

        if self._attn_4d is None:
            layout.addWidget(QLabel("No attention weights available."))
            return

        B, H, S, _ = self._attn_4d.shape
        batch_idx = 0  # always show first batch item

        # Head selector.
        ctrl = QWidget()
        ctrl_layout = QHBoxLayout(ctrl)
        ctrl_layout.setContentsMargins(0, 0, 0, 0)
        ctrl_layout.addWidget(QLabel("Head:"))
        self._head_combo = QComboBox()
        self._head_combo.addItem("All heads")
        for h in range(H):
            self._head_combo.addItem(f"Head {h}")
        self._head_combo.currentIndexChanged.connect(self._on_head_changed)
        ctrl_layout.addWidget(self._head_combo)
        ctrl_layout.addStretch(1)
        layout.addWidget(ctrl)

        # Scroll area for heatmaps.
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self._heatmap_container = QWidget()
        self._heatmap_layout = QVBoxLayout(self._heatmap_container)
        scroll.setWidget(self._heatmap_container)
        layout.addWidget(scroll)

        self._show_heads(batch_idx=batch_idx, head_idx=None)

    def _on_head_changed(self, index: int) -> None:
        if self._attn_4d is None:
            return
        head_idx = None if index == 0 else index - 1
        self._show_heads(batch_idx=0, head_idx=head_idx)

    def _show_heads(self, batch_idx: int, head_idx: Optional[int]) -> None:
        if self._attn_4d is None:
            return
        # Clear existing heatmaps.
        while self._heatmap_layout.count():
            item = self._heatmap_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._heatmaps.clear()

        B, H, S, _ = self._attn_4d.shape
        heads_to_show = [head_idx] if head_idx is not None else list(range(H))
        for h in heads_to_show:
            hw = HeatmapWidget(
                data=self._attn_4d[batch_idx, h],
                title=f"Head {h}",
            )
            hw.setMinimumHeight(200)
            self._heatmap_layout.addWidget(hw)
            self._heatmaps.append(hw)
