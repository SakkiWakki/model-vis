"""ClassLabelOutput: a discrete class prediction with per-class probabilities."""
from __future__ import annotations

from typing import List, Optional, Sequence

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QProgressBar, QVBoxLayout, QWidget


class ClassLabelOutput:
    """A predicted class index plus an optional probability vector."""

    def __init__(
        self,
        class_id: int,
        probs: Optional[Sequence[float]] = None,
        labels: Optional[Sequence[str]] = None,
    ) -> None:
        self._class_id = int(class_id)
        self._probs: List[float] = [float(p) for p in probs] if probs is not None else []
        self._labels: List[str] = list(labels) if labels is not None else []

    @property
    def raw(self) -> int:
        return self._class_id

    @property
    def class_id(self) -> int:
        return self._class_id

    @property
    def probs(self) -> List[float]:
        return list(self._probs)

    @property
    def labels(self) -> List[str]:
        return list(self._labels)

    @classmethod
    def render_widget(cls, parent: Optional[QWidget] = None) -> "ClassLabelRenderer":
        return ClassLabelRenderer(parent=parent)


class ClassLabelRenderer(QWidget):
    """Shows the predicted class label and a bar per class probability."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(4)

        self._headline = QLabel("—")
        self._headline.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self._layout.addWidget(self._headline)

        self._bars_host = QWidget()
        self._bars_layout = QVBoxLayout(self._bars_host)
        self._bars_layout.setContentsMargins(0, 0, 0, 0)
        self._bars_layout.setSpacing(2)
        self._layout.addWidget(self._bars_host)

    def set_output(self, output: Optional[ClassLabelOutput]) -> None:
        # Clear existing bars.
        while self._bars_layout.count():
            item = self._bars_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

        if output is None:
            self._headline.setText("—")
            return

        labels = output.labels or [str(i) for i in range(max(len(output.probs), output.class_id + 1))]
        predicted_label = labels[output.class_id] if output.class_id < len(labels) else str(output.class_id)
        self._headline.setText(f"<b>Predicted:</b> {predicted_label}")

        for i, p in enumerate(output.probs):
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            name = labels[i] if i < len(labels) else str(i)
            tag = QLabel(name)
            tag.setFixedWidth(40)
            row_layout.addWidget(tag)
            bar = QProgressBar()
            bar.setRange(0, 1000)
            bar.setValue(int(max(0.0, min(1.0, p)) * 1000))
            bar.setFormat(f"{p:.3f}")
            row_layout.addWidget(bar, stretch=1)
            self._bars_layout.addWidget(row)
