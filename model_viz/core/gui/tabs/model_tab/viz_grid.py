"""VizGrid: vertical stack of visualization panels with scroll.

Layout rules (matching the user's spec):
  1 panel  -> top half of the viewport (panel height = viewport / 2)
  2 panels -> each panel half the viewport
  3 panels -> each panel one-third of the viewport
  4+       -> each panel still one-third of the viewport; the rest scroll

Each panel's height is therefore a fraction of the *viewport* (not of the
inner content), so the visible-N-panel split stays consistent whether
there are 2 panels or 20.  Panels beyond what fits push the inner widget
past the viewport and the scroll bar appears.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Type

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QResizeEvent
from PyQt6.QtWidgets import QScrollArea, QVBoxLayout, QWidget

from model_viz.core.adapter import ModelAdapter
from model_viz.core.gui.tabs.model_tab.viz_panel import VizPanel
from model_viz.core.layer import LayerLike


class VizGrid(QWidget):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        outer.addWidget(self._scroll)

        self._inner = QWidget()
        self._scroll.setWidget(self._inner)

        self._inner_layout = QVBoxLayout(self._inner)
        self._inner_layout.setContentsMargins(8, 8, 8, 8)
        self._inner_layout.setSpacing(8)
        self._inner_layout.addStretch(1)  # keeps single panel pinned to top

        self._by_viz: Dict[Type, VizPanel] = {}
        self._order: List[Type] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_visualization(
        self,
        layer: LayerLike,
        viz_cls: Type,
        adapter: Optional[ModelAdapter] = None,
    ) -> None:
        panel = self._by_viz.get(viz_cls)
        if panel is None:
            panel = VizPanel(viz_cls=viz_cls, parent=self._inner)
            self._by_viz[viz_cls] = panel
            self._order.append(viz_cls)
            # Insert before the trailing stretch.
            self._inner_layout.insertWidget(self._inner_layout.count() - 1, panel)
            self._update_panel_sizes()
        panel.add_layer(layer, adapter=adapter)

    def refresh_all(self, adapter: ModelAdapter) -> None:
        for panel in self._by_viz.values():
            panel.refresh_all(adapter)

    # ------------------------------------------------------------------
    # Sizing
    # ------------------------------------------------------------------

    def resizeEvent(self, event: QResizeEvent) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._update_panel_sizes()

    def _update_panel_sizes(self) -> None:
        """Set each panel's height to viewport/2 (1 or 2 panels) or viewport/3 (3+).

        For 1 panel the trailing stretch fills the bottom half, giving the
        "panel takes the top half" visual the user wanted.
        """
        n = len(self._order)
        if n == 0:
            return
        viewport_h = self._scroll.viewport().height()
        if viewport_h <= 0:
            return
        # Account for the inter-panel spacing in the inner layout.
        spacing = self._inner_layout.spacing()
        margins = self._inner_layout.contentsMargins()
        vmargins = margins.top() + margins.bottom()

        denom = 2 if n <= 2 else 3
        # Distribute the visible viewport across `denom` slots (minus margins
        # + spacing between visible slots).
        usable = viewport_h - vmargins - spacing * max(denom - 1, 0)
        slot_h = max(usable // denom, 80)

        for cls in self._order:
            panel = self._by_viz[cls]
            panel.setFixedHeight(slot_h)
