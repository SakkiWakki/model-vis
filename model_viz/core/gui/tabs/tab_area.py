"""TabArea: QTabWidget host; manages per-model workspace tabs."""
from __future__ import annotations

from typing import Dict, Optional, Type, TYPE_CHECKING

from PyQt6.QtWidgets import QStackedWidget, QTabWidget, QVBoxLayout, QWidget

from model_viz.core.gui.tabs.empty_state import EmptyState
from model_viz.core.gui.tabs.model_tab.model_tab import ModelTab
from model_viz.core.adapter import ModelAdapter
from model_viz.core.layer import LayerLike
from model_viz.core.gui.event_bus import get_bus

if TYPE_CHECKING:
    from model_viz.viz.visualizer_base import VisualizerBase


class TabArea(QWidget):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._stack = QStackedWidget()
        layout.addWidget(self._stack)

        self._empty = EmptyState()
        self._tabs = QTabWidget()
        self._tabs.setTabsClosable(True)
        self._tabs.tabCloseRequested.connect(self._close_tab)

        self._stack.addWidget(self._empty)
        self._stack.addWidget(self._tabs)
        self._stack.setCurrentWidget(self._empty)

        self._model_tabs: Dict[str, ModelTab] = {}
        self._refresh_pending = False

        # Refresh on global model updates, but coalesce to avoid UI overload.
        get_bus().model_updated.connect(self._on_model_updated)

    def set_model(self, adapter: ModelAdapter) -> None:
        key = adapter.name
        tab = self._model_tabs.get(key)
        if tab is None:
            tab = ModelTab(adapter=adapter)
            self._model_tabs[key] = tab
            idx = self._tabs.addTab(tab, tab.title)
            self._tabs.setCurrentIndex(idx)
        else:
            tab.set_adapter(adapter)
            self._tabs.setCurrentWidget(tab)
        self._stack.setCurrentWidget(self._tabs)

    def add_visualization(self, layer: LayerLike, viz_cls: Type["VisualizerBase"]) -> None:
        tab = self.current_model_tab()
        if tab is None:
            return
        tab.add_visualization(layer, viz_cls)

    def refresh_all(self) -> None:
        tab = self.current_model_tab()
        if tab is not None:
            tab.refresh_all()

    def current_model_tab(self) -> Optional[ModelTab]:
        w = self._tabs.currentWidget()
        return w if isinstance(w, ModelTab) else None

    def _on_model_updated(self, payload: object) -> None:
        # Only refresh the current workspace, and coalesce to one refresh per event loop tick.
        if self._refresh_pending:
            return
        self._refresh_pending = True

        def do() -> None:
            self._refresh_pending = False
            self.refresh_all()

        # Use a single-shot timer to coalesce bursts.
        from PyQt6.QtCore import QTimer

        QTimer.singleShot(0, do)

    def _close_tab(self, index: int) -> None:
        widget = self._tabs.widget(index)
        if isinstance(widget, ModelTab):
            self._model_tabs.pop(widget.adapter.name, None)
        self._tabs.removeTab(index)
        if widget is not None:
            widget.deleteLater()
        if self._tabs.count() == 0:
            self._stack.setCurrentWidget(self._empty)
