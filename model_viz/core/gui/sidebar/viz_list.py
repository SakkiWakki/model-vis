"""VizList: shows visualizers supported by the current model, filtered by layer."""
from __future__ import annotations

from typing import List, Optional, Type

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QLabel, QListWidget, QListWidgetItem, QVBoxLayout, QWidget

from model_viz.core.adapter import ModelAdapter
from model_viz.core.layer import LayerLike
from model_viz.core import registry
from model_viz.core.gui.event_bus import get_bus


class VizList(QWidget):
    viz_chosen = pyqtSignal(object, object)  # emits (layer, viz_cls)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addWidget(QLabel("Visualizations:"))
        self._list = QListWidget()
        layout.addWidget(self._list)
        self._layer: Optional[LayerLike] = None
        self._adapter: Optional[ModelAdapter] = None
        self._supported: list[Type] = []
        self._viz_classes: list[Type] = []
        self._list.itemDoubleClicked.connect(self._on_double_clicked)
        self._list.itemClicked.connect(self._on_clicked)

        # Re-scan compatibility whenever the model's activation state changes.
        # Some visualizers (Perplexity, AttentionVisualizer) check live
        # outputs in ``compatible_with``; without this signal the list stays
        # stale until the user re-selects the adapter.
        get_bus().model_updated.connect(self._on_model_updated)

    def set_adapter(self, adapter: Optional[ModelAdapter]) -> None:
        """Set current model adapter and populate supported visualizers.

        Passing ``None`` clears the list — used during shutdown.
        """
        self._adapter = adapter
        self._layer = None
        self._rescan_supported()
        self._render()

    def set_layer(self, layer: LayerLike) -> None:
        self._layer = layer
        self._render()

    def _rescan_supported(self) -> None:
        if self._adapter is None:
            self._supported = []
            return
        self._supported = self._adapter.supported_visualizers(registry.get_visualizers())

    def _on_model_updated(self, _payload: object) -> None:
        if self._adapter is None:
            return
        self._rescan_supported()
        self._render()

    def _render(self) -> None:
        self._list.clear()
        if self._adapter is None:
            self._viz_classes = []
            return

        base = self._supported
        if self._layer is None:
            self._viz_classes = list(base)
        else:
            self._viz_classes = [cls for cls in base if cls.compatible_with(self._layer)]

        for cls in self._viz_classes:
            self._list.addItem(QListWidgetItem(cls.display_name))

    def _on_clicked(self, item: QListWidgetItem) -> None:
        self._emit_for_item(item)

    def _on_double_clicked(self, item: QListWidgetItem) -> None:
        self._emit_for_item(item)

    def _emit_for_item(self, item: QListWidgetItem) -> None:
        idx = self._list.row(item)
        if self._layer is not None and 0 <= idx < len(self._viz_classes):
            self.viz_chosen.emit(self._layer, self._viz_classes[idx])
