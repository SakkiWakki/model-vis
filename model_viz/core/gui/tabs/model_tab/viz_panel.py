"""VizPanel: one grid cell for a visualization type.

Holds one or more instances of the same visualization class (different layers)
in a horizontal scroll area.
"""
from __future__ import annotations

from typing import Dict, Optional, Type

from PyQt6.QtWidgets import QHBoxLayout, QLabel, QScrollArea, QVBoxLayout, QWidget

from model_viz.core.adapter import ModelAdapter
from model_viz.core.layer import LayerLike


class VizPanel(QWidget):
    def __init__(self, viz_cls: Type, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._viz_cls = viz_cls
        self._instances: Dict[str, QWidget] = {}  # key by layer.name

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(4)

        title = QLabel(f"<b>{getattr(viz_cls, 'display_name', viz_cls.__name__)}</b>")
        outer.addWidget(title)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        outer.addWidget(scroll, stretch=1)

        self._inner = QWidget()
        self._row = QHBoxLayout(self._inner)
        self._row.setContentsMargins(0, 0, 0, 0)
        self._row.setSpacing(8)
        self._row.addStretch(1)
        scroll.setWidget(self._inner)

    def add_layer(self, layer: LayerLike, adapter: Optional[ModelAdapter] = None) -> None:
        key = layer.name
        if key in self._instances:
            return
        try:
            w = self._viz_cls(layer=layer, adapter=adapter)
        except Exception as e:
            w = QLabel(f"Failed to build viz: {e}")
        # Keep each instance from growing arbitrarily wide; user can scroll horizontally.
        w.setMinimumWidth(260)

        # Insert before the stretch.
        self._row.insertWidget(self._row.count() - 1, w)
        self._instances[key] = w

    def refresh_all(self, adapter: ModelAdapter) -> None:
        layers_by_name = {layer.name: layer for layer in adapter.layers()}
        for layer_name, w in self._instances.items():
            layer = layers_by_name.get(layer_name)
            if layer is not None and hasattr(w, "set_layer"):
                try:
                    w.set_layer(layer)
                except Exception:
                    pass
            if hasattr(w, "set_adapter"):
                try:
                    w.set_adapter(adapter)
                except Exception:
                    pass
            if hasattr(w, "refresh"):
                try:
                    w.refresh()
                except Exception:
                    continue
