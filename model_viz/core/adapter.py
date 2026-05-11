"""ModelAdapter protocol: the contract every model adapter must satisfy."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Tuple, Type, runtime_checkable, TYPE_CHECKING, Literal

from model_viz.core.input.base import InputBase
from model_viz.core.layer import LayerLike

if TYPE_CHECKING:
    from model_viz.viz.visualizer_base import VisualizerBase


@dataclass(frozen=True)
class HyperParamSpec:
    """Specification for a user-editable hyperparameter."""

    name: str
    kind: Literal["int", "float", "choice", "bool"]
    default: object
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    step: Optional[float] = None
    choices: Optional[List[object]] = None
    description: str = ""


@runtime_checkable
class ModelAdapter(Protocol):
    """Minimum interface every model adapter must satisfy.

    The training extension (``hyperparameters``, ``train_step``, etc.) lives on
    the narrower ``TrainableAdapter`` protocol.  Inference-only adapters (HF,
    llama.cpp, etc.) implement only this base protocol.
    """

    name: str
    accepted_inputs: Tuple[Type[InputBase], ...]

    # Child adapters, keyed by name, in insertion order.
    # layers() / groups() on a composite adapter should merge children's output.
    child_adapters: Dict[str, "ModelAdapter"]

    def layers(self) -> List[LayerLike]:
        """Return all leaf layers in forward order, prev/next linked."""
        ...

    def groups(self) -> List[LayerLike]:
        """Return top-level LayerLike items (groups or leaves) for the sidebar."""
        ...

    def forward(self, inputs: Any) -> Any:
        """Run a forward pass and populate each layer's activation store."""
        ...

    def supported_visualizers(
        self, visualizers: List[Type["VisualizerBase"]]
    ) -> List[Type["VisualizerBase"]]:
        """Return the subset of visualizers compatible with at least one layer."""
        ...


@runtime_checkable
class TrainableAdapter(ModelAdapter, Protocol):
    """Narrower protocol for adapters that implement the training extension.

    Kept for backward compatibility with TrainingController / TrainingFlyout
    which use isinstance(adapter, TrainableAdapter) to gate the training UI.
    All methods are already declared on ModelAdapter; this protocol simply
    asserts they are meaningfully implemented.
    """

    def hyperparameters(self) -> Dict[str, HyperParamSpec]: ...
    def apply_hyperparameters(self, values: Dict[str, object]) -> None: ...
    def reset_training(self) -> None: ...
    def train_step(self) -> Dict[str, float]: ...
    def probe(self) -> None: ...
