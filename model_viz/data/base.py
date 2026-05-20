"""Composable dataset capabilities.

A "dataset" in this codebase is just an object that opts into one or more of
the capability ``Protocol``s defined here.  There is intentionally no
``Dataset`` superclass to inherit from — concrete classes simply implement the
methods that match what they can do, and consumers check at the use site via
``isinstance(obj, InputCapable)`` / ``isinstance(obj, TrainCapable)`` /
``isinstance(obj, EvalCapable)``.

Why a capability split
----------------------
The original ``Dataset`` lumped training (``batch``), input plumbing
(``make_input`` / ``probe_input`` / ``interpret_output``), and metadata into
one interface.  Inference-only corpora (FCE, HF text, contradiction pairs)
had to fake a training method that just raised ``NotImplementedError``, and
there was no way to declare a dataset that only supports offline iteration
(e.g. "give me the next FCE paragraph pair").

Capabilities
------------
- :class:`InputCapable` — produce model inputs from raw values; required by
  any flow that wants a forward pass with user-typed text.
- :class:`TrainCapable` — produce supervised ``(xs, ys)`` batches; required
  by the training UI and trainable adapters.
- :class:`EvalCapable` — iterate structured examples for offline / probe
  experiments (FCE paragraph pairs, contradiction-pair specs, etc.).  The
  element type is dataset-specific; consumers narrow it themselves.

A dataset can mix and match — XOR is ``InputCapable + TrainCapable``, FCE is
``InputCapable + EvalCapable``, ContradictionPairs is ``EvalCapable`` only.

All capability ``Protocol``s share the same identity surface (``name`` +
``info``).  Two helper ``Protocol``s with both flags set, ``TrainableDataset``
and ``EvalDataset``, exist purely so type annotations on call sites stay
short.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, Protocol, Tuple, Type, runtime_checkable

import torch

from model_viz.core.input.base import InputBase
from model_viz.core.output.base import OutputBase


@dataclass(frozen=True)
class DatasetInfo:
    vocab_size: int | None = None
    num_classes: int | None = None
    description: str = ""


# ----------------------------------------------------------------------
# Capability protocols.  Each carries the shared identity surface so any one
# of them is enough to register / display the dataset in the UI.
# ----------------------------------------------------------------------


@runtime_checkable
class InputCapable(Protocol):
    """Can turn raw values into model inputs and interpret model outputs."""

    name: str
    info: DatasetInfo
    input_type: Type[InputBase]
    output_type: Type[OutputBase]

    def make_input(self, raw: Any) -> InputBase: ...
    def probe_input(self) -> InputBase: ...
    def interpret_output(self, raw: Any) -> OutputBase: ...


@runtime_checkable
class TrainCapable(Protocol):
    """Can produce supervised training batches.

    The narrower ``input_type`` field is still required because the training
    loop needs to know what kind of inputs it is producing.
    """

    name: str
    info: DatasetInfo
    input_type: Type[InputBase]

    def batch(self) -> Tuple[torch.Tensor, torch.Tensor]: ...


@runtime_checkable
class EvalCapable(Protocol):
    """Can iterate structured examples for offline / probe experiments.

    The yielded element type is dataset-specific (e.g. ``FCEPair``,
    ``ContradictionPairSpec``).  Consumers should narrow the type themselves
    at the use site.  ``__len__`` is optional — implementers that know the
    length up front are encouraged to expose it for progress bars.
    """

    name: str
    info: DatasetInfo

    def iter_examples(self) -> Iterator[Any]: ...


# ----------------------------------------------------------------------
# Convenience composites for annotations only.  These add no new methods.
# ----------------------------------------------------------------------


@runtime_checkable
class TrainableDataset(InputCapable, TrainCapable, Protocol):
    """Shorthand annotation for the common train+input combination."""

    ...


@runtime_checkable
class EvalDataset(InputCapable, EvalCapable, Protocol):
    """Shorthand annotation for the common eval+input combination."""

    ...
