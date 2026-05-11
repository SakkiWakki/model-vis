"""VizGrid: lays out visualization panels in a 3-row grid.

Design goals:
- At most 3 rows with equal vertical space (viewport height divided by 3).
- Add visualizations into cells; no duplicate viz types.
- If a viz type is added again for a different layer, it becomes a horizontal
  scroll inside the same cell.
"""
from __future__ import annotations

from typing import Dict, Optional, Type

from PyQt6.QtWidgets import QGridLayout, QWidget

from model_viz.core.adapter import ModelAdapter
from model_viz.core.gui.tabs.model_tab.viz_panel import VizPanel
from model_viz.core.layer import LayerLike


class VizGrid(QWidget):
    MAX_ROWS = 3

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._layout = QGridLayout(self)
        self._layout.setContentsMargins(8, 8, 8, 8)
        self._layout.setHorizontalSpacing(8)
        self._layout.setVerticalSpacing(8)

        for r in range(self.MAX_ROWS):
            self._layout.setRowStretch(r, 1)

        self._by_viz: Dict[Type, VizPanel] = {}
        self._order: list[Type] = []

    def add_visualization(
        self,
        layer: LayerLike,
        viz_cls: Type,
        adapter: Optional[ModelAdapter] = None,
    ) -> None:
        panel = self._by_viz.get(viz_cls)
        if panel is None:
            panel = VizPanel(viz_cls=viz_cls, parent=self)
            self._by_viz[viz_cls] = panel
            self._order.append(viz_cls)

            idx = len(self._order) - 1
            row = idx % self.MAX_ROWS
            col = idx // self.MAX_ROWS
            self._layout.addWidget(panel, row, col)
            self._layout.setColumnStretch(col, 1)

        panel.add_layer(layer, adapter=adapter)

    def refresh_all(self, adapter: ModelAdapter) -> None:
        for panel in self._by_viz.values():
            panel.refresh_all(adapter)
