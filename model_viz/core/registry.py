"""Global registries for model adapters and visualizers.

This is the only file permitted to hold global mutable state.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Type, TYPE_CHECKING

if TYPE_CHECKING:
    from model_viz.core.adapter import ModelAdapter
    from model_viz.viz.visualizer_base import VisualizerBase

# Datasets are stored as opaque objects.  They are duck-typed against the
# capability protocols in ``model_viz.data.base`` (InputCapable, TrainCapable,
# EvalCapable) at the point of use, not at registration time.
_adapters: Dict[str, "ModelAdapter"] = {}
_visualizers: List[Type["VisualizerBase"]] = []
_datasets: Dict[str, object] = {}
_model_factories: Dict[str, Callable[[object], "ModelAdapter"]] = {}


def register_adapter(adapter: "ModelAdapter") -> None:
    _adapters[adapter.name] = adapter


def get_adapters() -> Dict[str, "ModelAdapter"]:
    return dict(_adapters)


def register_visualizer(cls: Type["VisualizerBase"]) -> None:
    if cls not in _visualizers:
        _visualizers.append(cls)


def get_visualizers() -> List[Type["VisualizerBase"]]:
    return list(_visualizers)


def register_dataset(dataset: object) -> None:
    name = getattr(dataset, "name", None)
    if not isinstance(name, str) or not name:
        raise ValueError(
            f"register_dataset expects an object with a non-empty ``name`` "
            f"attribute; got {dataset!r}"
        )
    _datasets[name] = dataset


def get_datasets() -> Dict[str, object]:
    return dict(_datasets)


def register_model_factory(name: str, factory: Callable[[object], "ModelAdapter"]) -> None:
    _model_factories[name] = factory


def get_model_factories() -> Dict[str, Callable[[object], "ModelAdapter"]]:
    return dict(_model_factories)
