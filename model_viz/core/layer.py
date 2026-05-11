"""Defines the LayerLike protocol and concrete wrappers VisualizableLayer and LayerGroup."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

import threading
import torch.nn as nn


@runtime_checkable
class LayerLike(Protocol):
    name: str
    next_layer: Optional[LayerLike]
    prev_layer: Optional[LayerLike]

    @property
    def curr_layer(self) -> nn.Module: ...
    def weights(self) -> Dict[str, Any]: ...
    def inputs(self) -> Any: ...
    def outputs(self) -> Any: ...
    def children(self) -> List[LayerLike]: ...


class VisualizableLayer:
    """Wraps a single nn.Module leaf as a LayerLike."""

    def __init__(self, name: str, module: nn.Module) -> None:
        self.name = name
        self._module = module
        self._inputs: Any = None
        self._outputs: Any = None
        self._lock = threading.RLock()
        self.next_layer: Optional[LayerLike] = None
        self.prev_layer: Optional[LayerLike] = None

    @property
    def curr_layer(self) -> nn.Module:
        return self._module

    def weights(self) -> Dict[str, Any]:
        return {k: v.detach() for k, v in self._module.named_parameters(recurse=False)}

    def inputs(self) -> Any:
        with self._lock:
            return self._inputs

    def outputs(self) -> Any:
        with self._lock:
            return self._outputs

    def set_inputs(self, value: Any) -> None:
        with self._lock:
            self._inputs = value

    def set_outputs(self, value: Any) -> None:
        with self._lock:
            self._outputs = value

    def children(self) -> List[LayerLike]:
        return []

    def __repr__(self) -> str:
        return f"VisualizableLayer({self.name!r})"


class LayerGroup:
    """Groups LayerLike children under a common name; behaves as a LayerLike itself."""

    def __init__(self, name: str, sublayers: List[LayerLike]) -> None:
        self.name = name
        self._sublayers = sublayers
        self.next_layer: Optional[LayerLike] = None
        self.prev_layer: Optional[LayerLike] = None

    @property
    def curr_layer(self) -> nn.Module:
        # A group has no single module; return the first child's module as a proxy.
        if self._sublayers:
            return self._sublayers[0].curr_layer
        raise ValueError(f"LayerGroup {self.name!r} has no children")

    def weights(self) -> Dict[str, Any]:
        return {}

    def inputs(self) -> Any:
        return None

    def outputs(self) -> Any:
        return None

    def children(self) -> List[LayerLike]:
        return list(self._sublayers)

    def __repr__(self) -> str:
        return f"LayerGroup({self.name!r}, {len(self._sublayers)} children)"
