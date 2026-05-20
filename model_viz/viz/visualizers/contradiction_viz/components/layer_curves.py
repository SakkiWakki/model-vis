"""LayerCurvesWidget: per-layer line plot for cosine / probe-accuracy curves."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtWidgets import QVBoxLayout, QWidget


class LayerCurvesWidget(QWidget):
    def __init__(self, title: str = "", parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._fig = Figure(tight_layout=True)
        self._ax = self._fig.add_subplot(111)
        self._canvas = FigureCanvas(self._fig)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._canvas)
        self._title = title
        self.clear()

    def clear(self) -> None:
        self._ax.clear()
        self._fig.patch.set_alpha(0.0)
        self._ax.patch.set_alpha(0.0)
        self._ax.set_title(self._title, color="#cccccc", fontsize=10)
        self._canvas.draw()

    def set_curves(
        self,
        curves: Dict[str, Sequence[float]],
        *,
        ylabel: str = "",
        ylim: Optional[tuple] = None,
        hline: Optional[float] = None,
    ) -> None:
        self._ax.clear()
        for label, ys in curves.items():
            ys_arr = np.asarray(list(ys), dtype=float)
            xs = np.arange(len(ys_arr))
            self._ax.plot(xs, ys_arr, marker="o", markersize=3, label=label, linewidth=1.5)
        if hline is not None:
            self._ax.axhline(
                hline, linestyle="--", linewidth=1.0,
                color="#888888", alpha=0.7,
            )
        self._ax.set_xlabel("layer", color="#aaaaaa", fontsize=9)
        if ylabel:
            self._ax.set_ylabel(ylabel, color="#aaaaaa", fontsize=9)
        if ylim is not None:
            self._ax.set_ylim(*ylim)
        self._ax.tick_params(colors="#aaaaaa", labelsize=8)
        for spine in self._ax.spines.values():
            spine.set_color("#444444")
        self._ax.grid(True, alpha=0.15)
        if curves:
            leg = self._ax.legend(fontsize=8, framealpha=0.3)
            for text in leg.get_texts():
                text.set_color("#cccccc")
        self._fig.patch.set_alpha(0.0)
        self._ax.patch.set_alpha(0.0)
        self._ax.set_title(self._title, color="#cccccc", fontsize=10)
        self._canvas.draw()

    def save_png(self, path: Path, *, dpi: int = 150) -> None:
        """Write the current figure to ``path`` as a PNG.

        Uses an opaque white background so the figure is readable when opened
        outside the dark-themed app (the on-screen version is transparent so
        it blends with the Qt panel).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fig.savefig(
            str(path),
            dpi=dpi,
            facecolor="white",
            edgecolor="none",
            bbox_inches="tight",
        )
