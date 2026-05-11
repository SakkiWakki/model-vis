"""HFCausalLMAdapter: wraps a HuggingFace causal LM for inference + visualization.

Reuses ``ModuleAdapter`` for layer walking + hook capture.  Forward is customized
to (a) request ``output_attentions=True`` so attention modules emit real weights,
and (b) return the ``logits`` tensor (not the HF ModelOutput wrapper).

Training methods are not implemented — HF causal LMs are used for inference here.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Type, TYPE_CHECKING

import torch
import torch.nn as nn

from model_viz.core.input.base import InputBase
from model_viz.core.input.text_input import TextInput
from model_viz.core.module_adapter import ModuleAdapter
from model_viz.data.hf.text import HFTextDataset

if TYPE_CHECKING:
    from model_viz.viz.visualizer_base import VisualizerBase


class HFCausalLMAdapter(ModuleAdapter):
    """Adapter for an already-loaded HF AutoModelForCausalLM."""

    # Structural guarantee: causal LMs produce (B, T, V) logits at the final
    # leaf.  Visualizers that key on this shape (e.g. Perplexity) can surface
    # themselves before any forward pass has cached activations.
    produces_per_position_logits = True

    def __init__(
        self,
        name: str,
        model: nn.Module,
        tokenizer: object,
    ) -> None:
        self.name = name
        self._model = model
        self._dataset = HFTextDataset(tokenizer=tokenizer, label=name)
        self.accepted_inputs: Tuple[Type[InputBase], ...] = (TextInput,)
        self.child_adapters: Dict[str, Any] = {}
        self._hooks: List[Any] = []
        self._capture_attentions = True
        self._build_layers()

    # ------------------------------------------------------------------
    # Override forward to pass ``output_attentions=True`` and unwrap logits.
    # ------------------------------------------------------------------

    def forward(self, inputs: Any, *, no_grad: bool = True) -> Any:
        if self._model is None:
            raise RuntimeError(f"Adapter {self.name!r} has no model to run forward on.")

        self._remove_hooks()
        leaf_map: Dict[nn.Module, Any] = {vl.curr_layer: vl for vl in self._leaf_layers}

        def _detach(x: Any) -> Any:
            if isinstance(x, torch.Tensor):
                return x.detach()
            if isinstance(x, (tuple, list)):
                return type(x)(_detach(v) for v in x)
            return x

        def make_hook(vl):
            def hook(__, input, output):
                vl.set_inputs(_detach(input))
                vl.set_outputs(_detach(output))
            return hook

        for module, vl in leaf_map.items():
            self._hooks.append(module.register_forward_hook(make_hook(vl)))

        def run() -> Any:
            if isinstance(inputs, TextInput):
                tensor = inputs.to_tensor()
            else:
                tensor = inputs
            if isinstance(tensor, torch.Tensor):
                tensor = tensor.to(next(self._model.parameters()).device)
            out = self._model(
                input_ids=tensor,
                output_attentions=self._capture_attentions,
                use_cache=False,
                return_dict=True,
            )
            return out

        try:
            if no_grad:
                with torch.no_grad():
                    raw = run()
            else:
                raw = run()
        finally:
            self._remove_hooks()

        # Return the logits tensor so downstream consumers (perplexity viz,
        # interpret_output) see (B, T, V) directly.
        return getattr(raw, "logits", raw)
