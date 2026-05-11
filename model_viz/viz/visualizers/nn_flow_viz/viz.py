"""NNFlowVisualizer: traditional input/hidden/output network view with flow opacity."""
from __future__ import annotations

from typing import List, Optional, Tuple, Type

import numpy as np
import torch
import torch.nn as nn
from PyQt6.QtWidgets import QVBoxLayout, QWidget

from model_viz.core.layer import LayerLike, VisualizableLayer
from model_viz.viz.visualizer_base import VisualizerBase
from model_viz.viz.visualizers.nn_flow_viz.components.flow_canvas import FlowCanvas, FlowSnapshot


def _leaves(layer: LayerLike) -> List[VisualizableLayer]:
    """Recursively collect VisualizableLayer leaves from a layer or group."""
    kids = layer.children()
    if kids:
        out: List[VisualizableLayer] = []
        for k in kids:
            out.extend(_leaves(k))
        return out
    return [layer] if isinstance(layer, VisualizableLayer) else []


def _collect_linear_layers(layer: LayerLike) -> List[VisualizableLayer]:
    """Return all Linear-backed leaves within the given layer/group, in order."""
    return [l for l in _leaves(layer) if isinstance(l.curr_layer, nn.Linear)]


def _tensor_from_hook_input(inp: object) -> Optional[torch.Tensor]:
    # Forward hooks receive `input` as a tuple of args.
    if isinstance(inp, tuple) and len(inp) >= 1 and isinstance(inp[0], torch.Tensor):
        return inp[0]
    if isinstance(inp, torch.Tensor):
        return inp
    return None


class NNFlowVisualizer(VisualizerBase):
    display_name = "NN Flow"

    def __init__(
        self,
        layer: LayerLike,
        adapter: Optional[object] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        self._canvas = FlowCanvas()
        super().__init__(layer=layer, adapter=adapter, parent=parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.addWidget(self._canvas)

    @classmethod
    def compatible_with(cls, layer: LayerLike) -> bool:
        linears = _collect_linear_layers(layer)
        return 2 <= len(linears) <= 6

    def set_layer(self, layer: LayerLike) -> None:
        self._layer = layer

    def refresh(self) -> None:
        linears = _collect_linear_layers(self._layer)
        snap = self._compute_snapshot(linears)
        self._canvas.set_snapshot(snap)

    # ------------------------------------------------------------------
    def _compute_snapshot(self, linears: List[VisualizableLayer]) -> Optional[FlowSnapshot]:
        if len(linears) < 2:
            return None

        # For each linear layer, we use its hook-captured input tensor as the "incoming activations".
        inputs: list[np.ndarray] = []
        weights: list[np.ndarray] = []
        for l in linears:
            mod = l.curr_layer
            assert isinstance(mod, nn.Linear)
            w = l.weights().get("weight")
            if not isinstance(w, torch.Tensor):
                return None
            inp_t = _tensor_from_hook_input(l.inputs())
            if inp_t is None:
                return None
            x = inp_t.detach().float().cpu()
            # Reduce to a single (in,) vector matching the linear's in_features.
            # Transformer ff inputs are (B, S, D); MLP inputs are (B, D_flat).
            # Take batch 0 and (if present) sequence position 0.
            while x.ndim > 1:
                x = x[0]
            if x.shape[-1] != mod.in_features:
                # Last resort: flatten any remaining leading dims and slice.
                x = x.reshape(-1)[: mod.in_features]
            inputs.append(x.numpy())
            weights.append(w.detach().float().cpu().numpy())  # (out,in)

        # layer_sizes includes input layer size and each linear out_features
        layer_sizes: list[int] = [weights[0].shape[1]] + [w.shape[0] for w in weights]

        # Compute node alpha per layer.
        node_alpha: list[np.ndarray] = []
        # Input layer alpha: abs(input)
        a0 = np.abs(inputs[0])
        node_alpha.append(_normalize(a0))

        # Hidden/output layers: flow into each output unit = sum_i |x_i * w_ji|
        for li, w in enumerate(weights):
            x = inputs[li]
            flow_out = np.sum(np.abs(w * x[None, :]), axis=1)  # (out,)
            node_alpha.append(_normalize(flow_out))

        # Edge alpha per connection.
        edge_alpha: list[np.ndarray] = []
        for li, w in enumerate(weights):
            x = inputs[li]
            edge = np.abs(w * x[None, :])  # (out,in)
            edge_alpha.append(_normalize(edge))

        return FlowSnapshot(layer_sizes=layer_sizes, node_alpha=node_alpha, edge_alpha=edge_alpha)


def _normalize(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    m = float(a.max()) if a.size else 0.0
    if m <= 1e-12:
        return np.zeros_like(a, dtype=np.float32)
    return (a / m).clip(0.0, 1.0)

