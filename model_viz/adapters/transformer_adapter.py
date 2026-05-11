"""TransformerAdapter: trainable transformer with composable child adapters.

Child adapters
--------------
Pass ``children`` to compose existing adapters (e.g. MLPAdapters for each
FFN block) rather than walking the nn.Module directly.  Each child's layer
tree is merged in insertion order, so visualizations defined for a child are
reused here automatically.

    ffn0 = MLPAdapter("ffn.0", dataset)
    model = TransformerAdapter("transformer", dataset, children={"ffn.0": ffn0})

Without ``children`` the adapter falls back to walking ``model.named_children()``
recursively, which is the original behaviour.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Type

import torch
import torch.nn as nn

from model_viz.core.adapter import HyperParamSpec
from model_viz.core.input.base import InputBase
from model_viz.core.input.text_input import TextInput
from model_viz.core.module_adapter import ModuleAdapter, ModuleChildAdapter
from model_viz.data.base import Dataset
from model_viz.models.transformer import TransformerConfig, build_transformer


class TransformerAdapter(ModuleAdapter):
    """Trainable transformer adapter parameterized by a Dataset.

    Inherits layer walking, hook capture, and visualizer scanning from
    ModuleAdapter.  Pass child adapters to compose reusable sub-visualizations
    (e.g. an MLPAdapter for each feedforward block) without redefining them.
    """

    accepted_inputs: Tuple[Type[InputBase], ...]

    def __init__(
        self,
        name: str,
        dataset: Dataset,
        *,
        children: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.name = name
        self._dataset = dataset
        self.accepted_inputs = (dataset.input_type,)  # type: ignore[assignment]
        self.child_adapters: Dict[str, Any] = dict(children) if children else {}
        self._hooks: List[Any] = []

        self._hp: Dict[str, object] = {
            "d_model": 32,
            "nhead": 2,
            "num_layers": 2,
            "ff_mult": 2,
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
            "d_model": HyperParamSpec(
                name="d_model", kind="int", default=self._hp["d_model"],
                minimum=8, maximum=512, step=8, description="Model width (d_model)",
            ),
            "nhead": HyperParamSpec(
                name="nhead", kind="int", default=self._hp["nhead"],
                minimum=1, maximum=16, step=1, description="Attention heads",
            ),
            "num_layers": HyperParamSpec(
                name="num_layers", kind="int", default=self._hp["num_layers"],
                minimum=1, maximum=24, step=1, description="Transformer blocks",
            ),
            "ff_mult": HyperParamSpec(
                name="ff_mult", kind="int", default=self._hp["ff_mult"],
                minimum=1, maximum=8, step=1, description="Feedforward multiplier",
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
        device = next(self._model.parameters()).device
        xs, ys = xs.to(device), ys.to(device)
        logits = self.forward(xs, no_grad=False)
        # Per-position loss: flatten (B, T, V) and (B, T) so ignore_index works
        # uniformly across positions.  ys may already be flat for classifier-style
        # datasets — in that case we fall back to the original shape.
        if logits.ndim == 3 and ys.ndim == 2:
            B, T, V = logits.shape
            loss = self._loss_fn(logits.reshape(B * T, V), ys.reshape(B * T))
            with torch.no_grad():
                pred = logits.argmax(dim=-1)
                mask = ys != -100
                acc = ((pred == ys) & mask).float().sum().item() / max(int(mask.sum().item()), 1)
        else:
            loss = self._loss_fn(logits, ys)
            with torch.no_grad():
                acc = (logits.argmax(dim=-1) == ys).float().mean().item()
        self._optim.zero_grad(set_to_none=True)
        loss.backward()
        self._optim.step()
        self._step += 1
        return {"step": float(self._step), "loss": float(loss.item()), "acc": float(acc)}

    def probe(self) -> None:
        inp = self._dataset.probe_input()
        if isinstance(inp, TextInput):
            self.forward(inp)
        else:
            self.forward(inp)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _rebuild(self) -> None:
        vocab_size = int(self._dataset.info.vocab_size or 2)
        cfg = TransformerConfig(
            vocab_size=vocab_size,
            d_model=int(self._hp["d_model"]),
            nhead=int(self._hp["nhead"]),
            num_layers=int(self._hp["num_layers"]),
            ff_mult=int(self._hp["ff_mult"]),
        )
        self._model = build_transformer(cfg)
        dev = str(self._hp.get("device", "cpu"))
        if dev == "cuda" and not torch.cuda.is_available():
            dev = "cpu"
            self._hp["device"] = dev
        self._model.to(torch.device(dev))

        # Auto-wrap each ff block as a child adapter so MLP visualizations apply.
        self.child_adapters = {
            f"blocks.{i}.ff": ModuleChildAdapter(f"blocks.{i}.ff", block.ff)
            for i, block in enumerate(self._model.blocks)  # type: ignore[attr-defined]
        }
        self._build_layers()
        self._optim = torch.optim.Adam(self._model.parameters(), lr=float(self._hp["lr"]))
        self.probe()
