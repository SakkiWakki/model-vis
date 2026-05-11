"""TensorGrid: displays named tensors as a scrollable grid of labelled heatmaps."""
from __future__ import annotations

from typing import Dict, Optional

import torch
from PyQt6.QtWidgets import QLabel, QScrollArea, QVBoxLayout, QWidget

from model_viz.viz.components.heatmap import HeatmapWidget


class TensorGrid(QWidget):
    def __init__(
        self,
        tensors: Optional[Dict[str, torch.Tensor]] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        outer.addWidget(scroll)

        self._inner = QWidget()
        self._layout = QVBoxLayout(self._inner)
        scroll.setWidget(self._inner)

        if tensors:
            self.update_tensors(tensors)

    def update_tensors(self, tensors: Dict[str, torch.Tensor]) -> None:
        while self._layout.count():
            item = self._layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        for name, tensor in tensors.items():
            self._layout.addWidget(QLabel(f"<b>{name}</b>"))
            hw = HeatmapWidget(data=tensor, title=name)
            hw.setMinimumHeight(160)
            self._layout.addWidget(hw)
