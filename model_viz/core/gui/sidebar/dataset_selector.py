"""DatasetSelector: dropdown that lists registered datasets."""
from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QWidget

from model_viz.core import registry


class DatasetSelector(QWidget):
    # Emits an opaque dataset object; consumers narrow via the capability
    # protocols in ``model_viz.data.base``.
    dataset_selected = pyqtSignal(object)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._datasets: list[object] = []

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel("Data:"))

        self._combo = QComboBox()
        self._combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        layout.addWidget(self._combo, stretch=1)

        self._combo.currentIndexChanged.connect(self._on_index_changed)
        self.refresh()

    def refresh(self) -> None:
        self._combo.blockSignals(True)
        self._combo.clear()
        self._datasets = list(registry.get_datasets().values())
        # The first entry is "no dataset" — a valid, supported state for
        # inference-only models (HF / Ollama loads bypass dataset selection).
        self._combo.addItem("— no dataset (inference only) —")
        for ds in self._datasets:
            self._combo.addItem(ds.name)
        self._combo.blockSignals(False)

    def _on_index_changed(self, index: int) -> None:
        if index <= 0:
            return
        ds = self._datasets[index - 1]
        self.dataset_selected.emit(ds)

