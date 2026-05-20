"""MLPAdapter: trainable MLP with dataset integration."""
from __future__ import annotations

from typing import Any, Dict, List, Tuple, Type, TYPE_CHECKING

import torch
import torch.nn as nn

from model_viz.core.adapter import HyperParamSpec
from model_viz.core.input.base import InputBase
from model_viz.core.module_adapter import ModuleAdapter
from model_viz.data.base import TrainableDataset
from model_viz.models.mlp import MLPConfig, build_mlp

if TYPE_CHECKING:
    from model_viz.viz.visualizer_base import VisualizerBase


class MLPAdapter(ModuleAdapter):
    """MLP trained on a Dataset that yields integer token tensors.

    Inputs are one-hot encoded and flattened to a vector before being passed
    to the network.  Compose this adapter as a child of a larger adapter to
    reuse its layer tree without redeclaring visualizations.
    """

    def __init__(self, name: str, dataset: TrainableDataset) -> None:
        self.name = name
        self._dataset = dataset
        self.accepted_inputs: Tuple[Type[InputBase], ...] = (dataset.input_type,)  # type: ignore[assignment]
        self.child_adapters: Dict[str, Any] = {}
        self._hooks: List[Any] = []

        self._hp: Dict[str, object] = {
            "hidden1": 8,
            "hidden2": 8,
            "lr": 1e-2,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
        }

        self._model: nn.Module
        self._optim: torch.optim.Optimizer
        self._loss_fn = nn.CrossEntropyLoss()
        self._step = 0
        self._rebuild()

    # ------------------------------------------------------------------
    # Training extension
    # ------------------------------------------------------------------

    def hyperparameters(self) -> Dict[str, HyperParamSpec]:
        return {
            "hidden1": HyperParamSpec(
                name="hidden1", kind="int", default=self._hp["hidden1"],
                minimum=2, maximum=64, step=1, description="Hidden layer 1 width",
            ),
            "hidden2": HyperParamSpec(
                name="hidden2", kind="int", default=self._hp["hidden2"],
                minimum=0, maximum=64, step=1, description="Hidden layer 2 width (0 disables)",
            ),
            "lr": HyperParamSpec(
                name="lr", kind="float", default=self._hp["lr"],
                minimum=1e-6, maximum=1.0, step=1e-3, description="Learning rate",
            ),
            "device": HyperParamSpec(
                name="device", kind="choice", default=self._hp["device"],
                choices=["cpu"] + (["cuda"] if torch.cuda.is_available() else []),
                description="Device",
            ),
        }

    def apply_hyperparameters(self, values: Dict[str, object]) -> None:
        for k in list(self._hp.keys()):
            if k in values and values[k] is not None:
                self._hp[k] = values[k]
        self._rebuild()

    def reset_training(self) -> None:
        self._step = 0
        self._rebuild()

    def train_step(self) -> Dict[str, float]:
        self._model.train()
        xs, ys = self._dataset.batch()
        # MLP is a classifier head: collapse per-position targets to a single
        # label per example by taking the last non-ignored position.  For XOR
        # this lifts the XOR-result target at position 1.
        if ys.ndim == 2:
            ys = self._last_valid_target(ys)
        xs = self._vectorize(xs)
        device = next(self._model.parameters()).device
        xs, ys = xs.to(device), ys.to(device)
        logits = self.forward(xs, no_grad=False)
        loss = self._loss_fn(logits, ys)
        self._optim.zero_grad(set_to_none=True)
        loss.backward()
        self._optim.step()
        self._step += 1
        with torch.no_grad():
            acc = (logits.argmax(dim=-1) == ys).float().mean().item()
        return {"step": float(self._step), "loss": float(loss.item()), "acc": float(acc)}

    @staticmethod
    def _last_valid_target(ys: torch.Tensor) -> torch.Tensor:
        # ys: (B, T) with -100 for ignored positions.  Return (B,) using the
        # last column that is not -100 per row.  Falls back to last column.
        mask = ys != -100
        if not mask.any():
            return ys[:, -1]
        # Find the largest index per row where mask is True.
        idxs = torch.arange(ys.shape[1], device=ys.device)
        masked_idxs = torch.where(mask, idxs.unsqueeze(0), torch.full_like(ys, -1))
        last = masked_idxs.max(dim=1).values
        last = torch.where(last < 0, torch.tensor(ys.shape[1] - 1, device=ys.device), last)
        return ys.gather(1, last.unsqueeze(1)).squeeze(1)

    def probe(self) -> None:
        inp = self._dataset.probe_input()
        with torch.no_grad():
            t = inp.to_tensor()  # type: ignore[attr-defined]
            x = self._vectorize(t).to(next(self._model.parameters()).device)
            self.forward(x, no_grad=True)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _vectorize(self, token_ids: torch.Tensor) -> torch.Tensor:
        vocab = int(self._dataset.info.vocab_size or 2)
        oh = torch.nn.functional.one_hot(token_ids.to(torch.long), num_classes=vocab).float()
        return oh.view(oh.shape[0], -1)

    def _rebuild(self) -> None:
        vocab = int(self._dataset.info.vocab_size or 2)
        num_classes = int(self._dataset.info.num_classes or 2)
        xs, _ = self._dataset.batch()
        seq_len = int(xs.shape[1]) if xs.ndim >= 2 else 1

        h1 = int(self._hp["hidden1"])
        h2 = int(self._hp["hidden2"])
        hidden = [h1] + ([h2] if h2 > 0 else [])
        self._model = build_mlp(MLPConfig(
            input_dim=vocab * seq_len, hidden_dims=hidden, num_classes=num_classes,
        ))

        dev = str(self._hp.get("device", "cpu"))
        if dev == "cuda" and not torch.cuda.is_available():
            dev = "cpu"
            self._hp["device"] = dev
        self._model.to(torch.device(dev))

        self._build_layers()
        self._optim = torch.optim.Adam(self._model.parameters(), lr=float(self._hp["lr"]))
        self.probe()
