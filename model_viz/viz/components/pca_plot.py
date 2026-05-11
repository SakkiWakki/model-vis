"""PCAPlot: projects a 2-D activation tensor to 2 PCA components and scatter-plots it."""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtWidgets import QVBoxLayout, QWidget


class PCAPlotWidget(QWidget):
    def __init__(
        self,
        data: Optional[torch.Tensor] = None,
        title: str = "PCA",
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
        # Flatten to (N, D).
        if arr.ndim == 1:
            arr = arr[None, :]
        elif arr.ndim > 2:
            arr = arr.reshape(-1, arr.shape[-1])

        if arr.shape[1] >= 2:
            arr = arr - arr.mean(axis=0)
            _, _, vt = np.linalg.svd(arr, full_matrices=False)
            proj = arr @ vt[:2].T
        else:
            proj = np.hstack([arr, np.zeros((arr.shape[0], 1))])

        self._ax.clear()
        self._ax.scatter(proj[:, 0], proj[:, 1], alpha=0.7, s=20)
        self._ax.set_title(title or self._title)
        self._ax.set_xlabel("PC1")
        self._ax.set_ylabel("PC2")
        self._canvas.draw()
