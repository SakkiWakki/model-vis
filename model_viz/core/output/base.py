"""OutputBase protocol: wraps a model's raw forward output for display."""
from __future__ import annotations

from typing import Any, Optional, Protocol, runtime_checkable, TYPE_CHECKING

if TYPE_CHECKING:
    from PyQt6.QtWidgets import QWidget


@runtime_checkable
class OutputBase(Protocol):
    """A typed wrapper around a model's forward output.

    Datasets produce these from raw model output via ``Dataset.interpret_output``.
    The UI uses ``render_widget`` to display the result.  Subclasses can be as
    rich as needed (image tensors for T2I, generated text, class labels, etc.).
    """

    @property
    def raw(self) -> Any:
        """The underlying value (tensor, string, etc.) for inspection."""
        ...

    @classmethod
    def render_widget(cls, parent: Optional["QWidget"] = None) -> "OutputRenderer":
        """Return a widget that can render instances of this output type."""
        ...


@runtime_checkable
class OutputRenderer(Protocol):
    """A QWidget that renders OutputBase instances via ``set_output``."""

    def set_output(self, output: Optional[OutputBase]) -> None:
        """Display the given output, or clear if ``None``."""
        ...
