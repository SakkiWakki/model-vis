"""Bipartite attention view: two parallel token rows with edges weighted by attention.

Used by ContradictionVisualizer to show, for a paired input
"Fact 1: ...  Fact 2: ...", how tokens of Fact 2 attend back to tokens of Fact 1.
Edges thinner / fainter = less attention.  Useful for inspecting whether the
contradicting token (e.g. "York" given a prior "Tokyo") is the one pulling
strong attention back to the source fact.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtWidgets import QVBoxLayout, QWidget


class BipartiteAttentionWidget(QWidget):
    """Render two rows of tokens with weighted edges between them."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._fig = Figure(tight_layout=True)
        self._ax = self._fig.add_subplot(111)
        self._canvas = FigureCanvas(self._fig)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._canvas)
        self.clear()

    def clear(self) -> None:
        self._ax.clear()
        self._ax.axis("off")
        self._canvas.draw()

    def set_data(
        self,
        top_tokens: Sequence[str],
        bottom_tokens: Sequence[str],
        weights: np.ndarray,
        title: str = "",
        threshold: float = 0.02,
    ) -> None:
        """Draw the bipartite graph.

        weights: (len(bottom_tokens), len(top_tokens)) — attention from each
        bottom token (Fact 2) back to each top token (Fact 1).  Edges below
        ``threshold`` are skipped to avoid hairball plots.
        """
        self._ax.clear()
        self._ax.axis("off")

        if not top_tokens or not bottom_tokens:
            self._canvas.draw()
            return

        # Layout: top row at y=1, bottom row at y=0.  X spread across [0, 1].
        nt, nb = len(top_tokens), len(bottom_tokens)
        xt = np.linspace(0.05, 0.95, nt)
        xb = np.linspace(0.05, 0.95, nb)

        # Edges first (so they sit under the labels).
        wmax = float(weights.max()) if weights.size else 0.0
        if wmax <= 0:
            wmax = 1.0
        for j in range(nb):
            for i in range(nt):
                w = float(weights[j, i])
                if w < threshold:
                    continue
                alpha = min(1.0, 0.15 + 0.85 * (w / wmax))
                lw = 0.4 + 3.0 * (w / wmax)
                self._ax.plot(
                    [xt[i], xb[j]],
                    [1.0, 0.0],
                    color="#4ea3ff",
                    alpha=alpha,
                    linewidth=lw,
                    solid_capstyle="round",
                )

        # Token labels.
        for i, tok in enumerate(top_tokens):
            self._ax.text(
                xt[i], 1.04, _clean(tok),
                ha="center", va="bottom", fontsize=9,
                color="#dddddd",
            )
        for j, tok in enumerate(bottom_tokens):
            self._ax.text(
                xb[j], -0.04, _clean(tok),
                ha="center", va="top", fontsize=9,
                color="#dddddd",
            )

        # Row labels on the left.
        self._ax.text(
            0.0, 1.0, "Fact 1", ha="right", va="center",
            fontsize=9, color="#aaaaaa", fontstyle="italic",
        )
        self._ax.text(
            0.0, 0.0, "Fact 2", ha="right", va="center",
            fontsize=9, color="#aaaaaa", fontstyle="italic",
        )

        self._ax.set_xlim(-0.05, 1.02)
        self._ax.set_ylim(-0.25, 1.25)
        if title:
            self._ax.set_title(title, color="#cccccc", fontsize=10)
        self._fig.patch.set_alpha(0.0)
        self._ax.patch.set_alpha(0.0)
        self._canvas.draw()


def _clean(tok: str) -> str:
    # GPT-2 / Qwen tokenizers prefix space tokens with a marker.  Replace for readability.
    return tok.replace("Ġ", "·").replace("Ġ", "·").replace(" ", "·") or "∅"
