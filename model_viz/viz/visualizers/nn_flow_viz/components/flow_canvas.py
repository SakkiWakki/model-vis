"""FlowCanvas: draws a simple feedforward NN with opacity based on "flow" magnitudes."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from PyQt6.QtCore import Qt, QRectF
from PyQt6.QtGui import QColor, QPainter, QPen, QBrush
from PyQt6.QtWidgets import QWidget


@dataclass(frozen=True)
class FlowSnapshot:
    layer_sizes: List[int]
    node_alpha: List[np.ndarray]  # per layer (n,)
    edge_alpha: List[np.ndarray]  # per connection (out, in)


class FlowCanvas(QWidget):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._snap: Optional[FlowSnapshot] = None
        self.setMinimumHeight(220)

    def set_snapshot(self, snap: Optional[FlowSnapshot]) -> None:
        self._snap = snap
        self.update()

    def paintEvent(self, event) -> None:  # type: ignore[override]
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.fillRect(self.rect(), self.palette().window())

        if self._snap is None:
            p.setPen(self.palette().text().color())
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No flow data available.")
            return

        sizes = self._snap.layer_sizes
        n_layers = len(sizes)
        if n_layers < 2:
            return

        margin_x = 22
        margin_y = 22
        w = max(1, self.width() - 2 * margin_x)
        h = max(1, self.height() - 2 * margin_y)

        xs = [margin_x + (i / (n_layers - 1)) * w for i in range(n_layers)]

        # Node positions
        y_positions: list[list[float]] = []
        for sz in sizes:
            if sz <= 1:
                y_positions.append([margin_y + h / 2])
                continue
            ys = [margin_y + (j / (sz - 1)) * h for j in range(sz)]
            y_positions.append(ys)

        # Draw edges (behind nodes)
        base_edge = QColor(0, 170, 255)
        for li in range(n_layers - 1):
            in_sz = sizes[li]
            out_sz = sizes[li + 1]
            A = self._snap.edge_alpha[li]
            # A shape: (out, in)
            for j in range(out_sz):
                y2 = y_positions[li + 1][j]
                for i in range(in_sz):
                    y1 = y_positions[li][i]
                    a = float(A[j, i])
                    if a <= 0.001:
                        continue
                    c = QColor(base_edge)
                    c.setAlphaF(min(1.0, max(0.0, a)))
                    pen = QPen(c)
                    pen.setWidthF(1.0 + 1.5 * a)
                    p.setPen(pen)
                    p.drawLine(int(xs[li]), int(y1), int(xs[li + 1]), int(y2))

        # Draw nodes
        base_node = QColor(50, 220, 200)
        node_r = 8.5
        for li in range(n_layers):
            al = self._snap.node_alpha[li]
            for i in range(sizes[li]):
                a = float(al[i])
                c = QColor(base_node)
                c.setAlphaF(min(1.0, max(0.08, a)))
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QBrush(c))
                x = xs[li]
                y = y_positions[li][i]
                p.drawEllipse(QRectF(x - node_r, y - node_r, node_r * 2, node_r * 2))

