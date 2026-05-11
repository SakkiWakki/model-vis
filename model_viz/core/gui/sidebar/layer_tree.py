"""LayerTree: collapsible tree menu built from LayerLike.children()."""
from __future__ import annotations

from typing import List, Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QLabel,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from model_viz.core.layer import LayerLike


class LayerTree(QWidget):
    layer_selected = pyqtSignal(object)  # emits LayerLike

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        layout.addWidget(QLabel("Layers:"))

        self._tree = QTreeWidget()
        self._tree.setHeaderHidden(True)
        self._tree.setIndentation(14)
        self._tree.setUniformRowHeights(True)
        self._tree.setExpandsOnDoubleClick(True)
        self._tree.itemClicked.connect(self._on_item_clicked)
        layout.addWidget(self._tree, stretch=1)

    # ------------------------------------------------------------------
    def set_root(self, root_layers: List[LayerLike]) -> None:
        self._tree.clear()
        for layer in root_layers:
            self._tree.addTopLevelItem(self._build_item(layer))

        # Start with just the top level visible; user expands what they need.
        self._tree.expandToDepth(0)

    def _build_item(self, layer: LayerLike) -> QTreeWidgetItem:
        item = QTreeWidgetItem([layer.name])
        item.setData(0, int(Qt.ItemDataRole.UserRole), layer)
        for child in layer.children():
            item.addChild(self._build_item(child))
        # Hint to Qt whether this is expandable.
        if item.childCount() == 0:
            item.setChildIndicatorPolicy(QTreeWidgetItem.ChildIndicatorPolicy.DontShowIndicator)
        return item

    def _on_item_clicked(self, item: QTreeWidgetItem, _column: int) -> None:
        layer = item.data(0, int(Qt.ItemDataRole.UserRole))
        if layer is not None:
            self.layer_selected.emit(layer)
