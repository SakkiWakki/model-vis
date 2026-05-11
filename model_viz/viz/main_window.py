"""MainWindow: QMainWindow that wires Sidebar ↔ TabArea; all signal routing lives here."""
from __future__ import annotations

from typing import Optional, Type

from PyQt6.QtWidgets import QHBoxLayout, QMainWindow, QWidget

from model_viz.core.gui.sidebar.sidebar import Sidebar
from model_viz.core.gui.tabs.tab_area import TabArea
from model_viz.core.gui.training.flyout import TrainingFlyout
from model_viz.core.layer import LayerLike
from model_viz.viz.visualizer_base import VisualizerBase


class MainWindow(QMainWindow):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("model_viz")
        self.resize(1200, 700)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._sidebar = Sidebar()
        self._tab_area = TabArea()
        self._flyout = TrainingFlyout(on_refresh_visuals=self._tab_area.refresh_all)

        layout.addWidget(self._sidebar)
        layout.addWidget(self._tab_area, stretch=1)
        layout.addWidget(self._flyout)

        self._sidebar.visualizer_requested.connect(self._open_viz)
        self._sidebar.model_selected.connect(self._on_model_selected)

    def _on_model_selected(self, adapter) -> None:
        self._tab_area.set_model(adapter)
        self._flyout.set_adapter(adapter)

    def _open_viz(self, layer: LayerLike, viz_cls: Type[VisualizerBase]) -> None:
        self._tab_area.add_visualization(layer, viz_cls)

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self._flyout.shutdown()
        super().closeEvent(event)
