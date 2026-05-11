"""VisualizerBase: abstract QWidget that all visualizers must subclass."""
from __future__ import annotations

from typing import Optional, TYPE_CHECKING

from PyQt6.QtWidgets import QWidget

from model_viz.core.layer import LayerLike

if TYPE_CHECKING:
    from model_viz.core.adapter import ModelAdapter


class VisualizerBase(QWidget):
    display_name: str = "Visualizer"

    def __init__(
        self,
        layer: LayerLike,
        adapter: Optional["ModelAdapter"] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._layer = layer
        self._adapter = adapter
        self.set_layer(layer)
        self.refresh()

    @classmethod
    def compatible_with(cls, layer: LayerLike) -> bool:
        """Return True if this visualizer can render the given layer."""
        return False

    def set_layer(self, layer: LayerLike) -> None:
        self._layer = layer

    def set_adapter(self, adapter: Optional["ModelAdapter"]) -> None:
        self._adapter = adapter

    def refresh(self) -> None:
        """Re-render from the layer's current activation state."""
