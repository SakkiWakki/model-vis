"""TextInput: wraps a raw string and an optional tokeniser callable."""
from __future__ import annotations

from typing import Any, Callable, Optional, TYPE_CHECKING

import torch
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QLineEdit, QWidget

if TYPE_CHECKING:
    from model_viz.data.base import Dataset


class TextInput:
    """Holds a text string and encodes it via a provided tokeniser."""

    def __init__(
        self,
        text: str,
        tokenizer: Optional[Callable[[str], torch.Tensor]] = None,
    ) -> None:
        self._text = text
        self._tokenizer = tokenizer

    @property
    def raw(self) -> str:
        return self._text

    def to_tensor(self) -> torch.Tensor:
        if self._tokenizer is None:
            raise ValueError("TextInput requires a tokenizer to produce a tensor.")
        return self._tokenizer(self._text)

    @classmethod
    def editor_widget(
        cls, dataset: "Dataset", parent: Optional[QWidget] = None
    ) -> "TextInputEditor":
        return TextInputEditor(dataset=dataset, parent=parent)


class TextInputEditor(QWidget):
    """Single-line text editor that produces TextInput instances via the dataset."""

    def __init__(self, dataset: "Dataset", parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._dataset = dataset

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel("Text:"))
        self._edit = QLineEdit()
        # Seed from the dataset's probe input so the field is never empty.
        try:
            self._edit.setText(str(dataset.probe_input().raw))
        except Exception:
            pass
        layout.addWidget(self._edit, stretch=1)

    def current_value(self) -> Optional[TextInput]:
        text = self._edit.text()
        if not text:
            return None
        try:
            return self._dataset.make_input(text)  # type: ignore[return-value]
        except Exception:
            return None
