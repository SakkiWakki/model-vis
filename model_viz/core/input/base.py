"""InputBase protocol: converts raw user input into a model-ready tensor."""
from __future__ import annotations

from typing import Any, Optional, Protocol, runtime_checkable, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from PyQt6.QtWidgets import QWidget
    from model_viz.data.base import Dataset


@runtime_checkable
class InputBase(Protocol):
    @property
    def raw(self) -> Any:
        """The original user-supplied value before encoding."""
        ...

    def to_tensor(self) -> torch.Tensor:
        """Return the model-ready tensor representation."""
        ...

    @classmethod
    def editor_widget(
        cls, dataset: "Dataset", parent: Optional["QWidget"] = None
    ) -> "InputEditor":
        """Return a widget for building instances of this input type."""
        ...


@runtime_checkable
class InputEditor(Protocol):
    """A QWidget that edits an input value of a specific InputBase subclass."""

    def current_value(self) -> Optional[InputBase]:
        """Return the current input, or ``None`` if invalid/empty."""
        ...
