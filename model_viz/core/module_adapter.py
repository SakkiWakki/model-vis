"""ModuleAdapter: shared layer-walking and hook machinery for nn.Module-based adapters.

Neither MLPAdapter nor TransformerAdapter should import each other.  Both
inherit from this base to get identical layer-tree building, forward-hook
capture, and visualizer-compatibility scanning without any cross-dependency.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Type, TYPE_CHECKING

import torch
import torch.nn as nn

from model_viz.core.input.text_input import TextInput
from model_viz.core.layer import LayerGroup, LayerLike, VisualizableLayer

if TYPE_CHECKING:
    from model_viz.core.adapter import ModelAdapter
    from model_viz.viz.visualizer_base import VisualizerBase


class ModuleAdapter:
    """Base for adapters that wrap an nn.Module.

    Subclasses set ``self._model``, ``self.child_adapters``, and ``self._hooks``
    before calling ``_build_layers()``.

    ``child_adapters`` maps a dotted module path (e.g. ``"blocks.0.ff"``) to a
    child adapter.  During module walking, whenever the current prefix matches a
    key in ``child_adapters``, that child's layer tree is spliced in instead of
    recursing into the sub-module.  This lets a TransformerAdapter reuse the
    MLPAdapter's visualization for each feedforward block automatically.
    """

    name: str
    child_adapters: Dict[str, "ModelAdapter"]
    _model: Optional[nn.Module]

    # ------------------------------------------------------------------
    # Layer tree
    # ------------------------------------------------------------------

    def _build_layers(self) -> None:
        self._groups: List[LayerLike] = []
        if self._model is not None:
            self._build_from_module()

        self._leaf_layers: List[VisualizableLayer] = self._collect_leaves(self._groups)
        for i, leaf in enumerate(self._leaf_layers):
            leaf.prev_layer = self._leaf_layers[i - 1] if i > 0 else None
            leaf.next_layer = (
                self._leaf_layers[i + 1] if i < len(self._leaf_layers) - 1 else None
            )
            # Back-ref so visualizers can query adapter-level capabilities
            # (e.g. produces_per_position_logits) before any forward has run.
            try:
                leaf._adapter = self  # type: ignore[attr-defined]
            except Exception:
                pass

    def _build_from_module(self) -> None:
        assert self._model is not None
        for top_name, top_module in self._model.named_children():
            kids = self._build_subtree(top_name, top_module)
            if len(kids) == 1 and not list(top_module.children()):
                self._groups.append(kids[0])
            else:
                self._groups.append(LayerGroup(name=top_name, sublayers=kids))

    def _build_subtree(
        self, prefix: str, module: nn.Module
    ) -> List[VisualizableLayer | LayerGroup]:
        # If a child adapter is registered for this exact path, splice its tree in.
        child = self.child_adapters.get(prefix)
        if child is not None:
            child_groups = child.groups()
            if len(child_groups) == 1:
                g = child_groups[0]
                # Re-wrap under the expected prefix name if the child used a different one.
                return [g if g.name == prefix else LayerGroup(name=prefix, sublayers=list(child.layers()))]
            return [LayerGroup(name=prefix, sublayers=child_groups)]

        # Treat any attention-like module as a leaf so we capture its full
        # output (which carries attention weights as a tuple) rather than
        # descending into q_proj/k_proj/v_proj/o_proj.  This covers
        # nn.MultiheadAttention as well as HF attention classes
        # (LlamaAttention, Qwen2Attention, ...).
        if isinstance(module, nn.MultiheadAttention) or type(module).__name__.endswith("Attention"):
            return [VisualizableLayer(name=prefix, module=module)]
        children = list(module.named_children())
        if not children:
            return [VisualizableLayer(name=prefix, module=module)]
        result: List[LayerLike] = []
        for child_name, child_module in children:
            full_name = f"{prefix}.{child_name}"
            subtree = self._build_subtree(full_name, child_module)
            if len(subtree) == 1:
                result.extend(subtree)
            else:
                result.append(LayerGroup(name=full_name, sublayers=subtree))
        return result

    def _collect_leaves(self, layers: List[LayerLike]) -> List[VisualizableLayer]:
        out: List[VisualizableLayer] = []
        for layer in layers:
            kids = layer.children()
            if kids:
                out.extend(self._collect_leaves(kids))
            elif isinstance(layer, VisualizableLayer):
                out.append(layer)
        return out

    # ------------------------------------------------------------------
    # ModelAdapter interface
    # ------------------------------------------------------------------

    def layers(self) -> List[LayerLike]:
        return list(self._leaf_layers)

    def groups(self) -> List[LayerLike]:
        return list(self._groups)

    def forward(self, inputs: Any, *, no_grad: bool = True) -> Any:
        if self._model is None:
            raise RuntimeError(f"Adapter {self.name!r} has no model to run forward on.")

        self._remove_hooks()
        leaf_map: Dict[nn.Module, VisualizableLayer] = {
            vl.curr_layer: vl for vl in self._leaf_layers
        }

        def _detach(x: Any) -> Any:
            if isinstance(x, torch.Tensor):
                return x.detach()
            if isinstance(x, (tuple, list)):
                return type(x)(_detach(v) for v in x)
            return x

        def make_hook(vl: VisualizableLayer):
            def hook(__, input, output):
                vl.set_inputs(_detach(input))
                vl.set_outputs(_detach(output))
            return hook

        self._hooks: List[Any] = getattr(self, "_hooks", [])
        for module, vl in leaf_map.items():
            self._hooks.append(module.register_forward_hook(make_hook(vl)))

        def run() -> Any:
            if isinstance(inputs, TextInput):
                tensor = inputs.to_tensor()
            else:
                tensor = inputs
            try:
                device = next(self._model.parameters()).device  # type: ignore[union-attr]
                if isinstance(tensor, torch.Tensor):
                    tensor = tensor.to(device)
            except StopIteration:
                pass
            return self._model(tensor)  # type: ignore[misc]

        if no_grad:
            with torch.no_grad():
                out = run()
        else:
            out = run()

        self._remove_hooks()
        return out

    def supported_visualizers(
        self, visualizers: List[Type["VisualizerBase"]]
    ) -> List[Type["VisualizerBase"]]:
        # Check both leaves and groups so multi-layer visualizers (e.g. NNFlow)
        # that require a subtree of layers can signal compatibility against a group.
        candidates: List[LayerLike] = list(self._leaf_layers) + list(self._all_groups())
        supported: List[Type["VisualizerBase"]] = []
        for viz_cls in visualizers:
            for layer in candidates:
                try:
                    if viz_cls.compatible_with(layer):
                        supported.append(viz_cls)
                        break
                except Exception:
                    continue
        return supported

    def _all_groups(self) -> List["LayerLike"]:
        """Return all LayerGroup nodes in the tree (depth-first)."""
        from model_viz.core.layer import LayerGroup
        out: List[LayerLike] = []
        def walk(layers: List[LayerLike]) -> None:
            for layer in layers:
                kids = layer.children()
                if kids:
                    out.append(layer)
                    walk(kids)
        walk(self._groups)
        return out

    def _remove_hooks(self) -> None:
        hooks: List[Any] = getattr(self, "_hooks", [])
        for h in hooks:
            h.remove()
        if hasattr(self, "_hooks"):
            self._hooks.clear()


class ModuleChildAdapter(ModuleAdapter):
    """Wraps a bare nn.Module as a read-only child adapter.

    Used by TransformerAdapter to represent sub-modules (e.g. each ``ff``
    block) as composable adapters whose layer tree can be spliced into the
    parent's tree, enabling visualizations defined for MLPAdapter to apply
    to those sub-modules automatically.

    The ``name`` is used as the root prefix for all child layer names, so
    leaves get fully-qualified names like ``blocks.0.ff.0`` rather than bare
    ``0``.
    """

    def __init__(self, name: str, module: nn.Module) -> None:
        self.name = name
        self._model: Optional[nn.Module] = module
        self.child_adapters: Dict[str, ModuleAdapter] = {}
        self._hooks: List[Any] = []
        self._build_layers()

    def _build_from_module(self) -> None:
        assert self._model is not None
        # Walk children under the adapter's own name as the root prefix so
        # leaf names are fully qualified (e.g. "blocks.0.ff.0" not "0").
        kids = self._build_subtree(self.name, self._model)
        if len(kids) == 1 and not list(self._model.children()):
            self._groups.extend(kids)
        else:
            self._groups.append(LayerGroup(name=self.name, sublayers=kids))
