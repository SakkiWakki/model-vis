"""ModelTab: a single model workspace tab containing a stacked visualization grid."""
from __future__ import annotations

from typing import Optional, Type

from PyQt6.QtWidgets import QVBoxLayout, QWidget

from model_viz.core.adapter import ModelAdapter
from model_viz.core.gui.tabs.model_tab.viz_grid import VizGrid
from model_viz.core.layer import LayerLike


class ModelTab(QWidget):
    def __init__(self, adapter: ModelAdapter, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.adapter = adapter

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._grid = VizGrid()
        layout.addWidget(self._grid)

    @property
    def title(self) -> str:
        return self.adapter.name

    def set_adapter(self, adapter: ModelAdapter) -> None:
        self.adapter = adapter
        self.refresh_all()

    def add_visualization(self, layer: LayerLike, viz_cls: Type) -> None:
        self._grid.add_visualization(layer, viz_cls, adapter=self.adapter)

    def refresh_all(self) -> None:
        self._grid.refresh_all(self.adapter)
