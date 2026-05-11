"""Dataset protocol for training + visualization presets."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Tuple, Type, runtime_checkable

import torch

from model_viz.core.input.base import InputBase
from model_viz.core.output.base import OutputBase


@dataclass(frozen=True)
class DatasetInfo:
    vocab_size: int | None = None
    num_classes: int | None = None
    description: str = ""


@runtime_checkable
class Dataset(Protocol):
    """A dataset/preset that can train a model and provide visualization probes."""

    name: str
    input_type: Type[InputBase]
    output_type: Type[OutputBase]
    info: DatasetInfo

    def make_input(self, raw: Any) -> InputBase:
        """Wrap a raw user value into an InputBase instance."""
        ...

    def probe_input(self) -> InputBase:
        """A stable input used to refresh activations while training."""
        ...

    def batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return a training batch (xs, ys)."""
        ...

    def interpret_output(self, raw: Any) -> OutputBase:
        """Wrap the model's raw forward output into a displayable OutputBase."""
        ...
