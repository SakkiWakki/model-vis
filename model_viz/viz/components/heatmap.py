"""HeatmapWidget: renders a 2-D tensor as a colour-mapped heatmap using matplotlib."""
from __future__ import annotations

from typing import Optional

import torch
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtWidgets import QVBoxLayout, QWidget


class HeatmapWidget(QWidget):
    def __init__(
        self,
        data: Optional[torch.Tensor] = None,
        title: str = "",
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._fig = Figure(tight_layout=True)
        self._ax = self._fig.add_subplot(111)
        self._canvas = FigureCanvas(self._fig)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._canvas)
        self._title = title
        if data is not None:
            self.update_data(data)

    def update_data(self, data: torch.Tensor, title: str = "") -> None:
        arr = data.detach().float().cpu().numpy()
        if arr.ndim > 2:
            arr = arr.reshape(arr.shape[0], -1)
        # Clear the entire figure so we don't accumulate multiple colorbar axes.
        self._fig.clear()
        self._ax = self._fig.add_subplot(111)
        im = self._ax.imshow(arr, aspect="auto", cmap="viridis")
        self._fig.colorbar(im, ax=self._ax)
        self._ax.set_title(title or self._title)
        self._canvas.draw()
