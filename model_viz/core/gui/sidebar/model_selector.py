"""ModelSelector: dropdown that lists registered model factories."""
from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QWidget

from model_viz.core import registry


class ModelSelector(QWidget):
    model_selected = pyqtSignal(object)  # emits model_name (str)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._model_names: list[str] = []

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(QLabel("Model:"))
        self._combo = QComboBox()
        self._combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        layout.addWidget(self._combo, stretch=1)

        self._combo.currentIndexChanged.connect(self._on_index_changed)
        self.refresh()

    def refresh(self) -> None:
        """Reload model factories from the registry."""
        self._combo.blockSignals(True)
        self._combo.clear()
        self._model_names = list(registry.get_model_factories().keys())
        self._combo.addItem("— select a model —")
        for name in self._model_names:
            self._combo.addItem(name)
        self._combo.blockSignals(False)

    def _on_index_changed(self, index: int) -> None:
        if index <= 0:
            return
        name = self._model_names[index - 1]
        self.model_selected.emit(name)
