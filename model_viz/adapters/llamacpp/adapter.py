"""LlamaCppAdapter: wraps a llama.cpp model for inference + perplexity viz.

Unlike the HF/transformers path, this adapter does not own an ``nn.Module`` —
weights stay quantized inside the llama.cpp runtime.  As a consequence:

- ``layers()`` and ``groups()`` expose exactly one pseudo-leaf named ``output``.
  Its ``outputs()`` returns the most recent ``(1, T, V)`` per-position logits
  tensor after each ``forward`` call.  This is enough for the perplexity
  visualizer (which keys on per-position logits at the model's final leaf).
- Attention/NN-Flow visualizers naturally show as not-compatible because
  there are no module-shaped activations to read.
- Training methods are not implemented.

Why this works: visualizers only ever see the data we hand them through the
``LayerLike`` interface.  As long as our pseudo-leaf returns the right tensor
from ``outputs()``, the perplexity viz cannot tell whether the model behind
it is a PyTorch module or a llama.cpp runtime.
"""
from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional, Tuple, Type, TYPE_CHECKING

import torch

from model_viz.core.input.base import InputBase
from model_viz.core.input.text_input import TextInput
from model_viz.core.layer import LayerLike
from model_viz.data.llamacpp.text import LlamaCppTextDataset

if TYPE_CHECKING:
    from model_viz.viz.visualizer_base import VisualizerBase


class _OutputPseudoLeaf:
    """Minimal LayerLike that just holds a (1, T, V) logits tensor."""

    def __init__(self, name: str = "output") -> None:
        self.name = name
        self.next_layer: Optional[LayerLike] = None
        self.prev_layer: Optional[LayerLike] = None
        self._inputs: Any = None
        self._outputs: Any = None
        self._lock = threading.RLock()

    @property
    def curr_layer(self) -> Any:
        # No backing nn.Module — return self so any caller asking for it
        # gets something with sensible attributes rather than crashing.
        return self

    def weights(self) -> Dict[str, Any]:
        return {}

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
        return f"_OutputPseudoLeaf({self.name!r})"


class LlamaCppAdapter:
    """Inference-only adapter for a llama.cpp GGUF runtime."""

    # Causal LM via llama.cpp: the output pseudo-leaf will carry (B, T, V)
    # logits after forward.  Surfaced as a structural guarantee so the
    # Perplexity visualizer appears before the first forward pass.
    produces_per_position_logits = True

    def __init__(self, name: str, llama: object) -> None:
        self.name = name
        self._llama = llama
        self._dataset = LlamaCppTextDataset(llama=llama, label=name)
        self.accepted_inputs: Tuple[Type[InputBase], ...] = (TextInput,)
        self.child_adapters: Dict[str, Any] = {}

        self._output_leaf = _OutputPseudoLeaf(name="output")
        # Back-ref so visualizers can query adapter-level capabilities.
        self._output_leaf._adapter = self  # type: ignore[attr-defined]

    def close(self) -> None:
        """Explicitly release the underlying llama.cpp runtime.

        Call this *before* the Python interpreter starts tearing down, so the
        ``Llama`` instance's C++ destructor runs while libllama.so and the
        CUDA context are still alive.  Letting it die at atexit is a known
        segfault path on app exit.
        """
        llama = getattr(self, "_llama", None)
        if llama is None:
            return
        # llama-cpp-python exposes close() in recent versions; fall back to
        # dropping the reference and forcing a GC otherwise.
        try:
            if hasattr(llama, "close"):
                llama.close()
        except Exception:
            pass
        self._llama = None
        # The dataset also holds a reference; clear it so nothing keeps the
        # native runtime alive past this point.
        if getattr(self, "_dataset", None) is not None:
            try:
                self._dataset._llama = None  # type: ignore[attr-defined]
            except Exception:
                pass

    def __del__(self) -> None:
        # Best-effort: at interpreter teardown the modules we depend on may
        # already be partially torn down, so swallow anything that happens.
        try:
            self.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # ModelAdapter
    # ------------------------------------------------------------------

    def layers(self) -> List[LayerLike]:
        return [self._output_leaf]

    def groups(self) -> List[LayerLike]:
        return [self._output_leaf]

    def forward(self, inputs: Any, *, no_grad: bool = True) -> Any:
        """Run a forward pass through llama.cpp and return (1, T, V) logits.

        Side-effect: stores the logits on the ``output`` pseudo-leaf so the
        perplexity visualizer can find them via ``layer.outputs()``.
        """
        # Resolve to a token-id tensor of shape (1, T).
        if isinstance(inputs, TextInput):
            ids_t = inputs.to_tensor()
        elif isinstance(inputs, torch.Tensor):
            ids_t = inputs
        else:
            raise ValueError(f"Unsupported input type for LlamaCppAdapter: {type(inputs).__name__}")

        if ids_t.ndim == 2:
            ids = ids_t[0].tolist()
        elif ids_t.ndim == 1:
            ids = ids_t.tolist()
        else:
            raise ValueError(f"Expected (B, T) or (T,) token ids; got shape {tuple(ids_t.shape)}.")
        ids = [int(t) for t in ids]

        # Reset KV state and feed the full prompt; ``logits_all=True`` ensures
        # per-position logits are retained.
        self._llama.reset()  # type: ignore[attr-defined]
        self._llama.eval(ids)  # type: ignore[attr-defined]

        # eval_logits is a sequence of per-position logit vectors.  Stack to (T, V).
        per_pos = self._llama.eval_logits  # type: ignore[attr-defined]
        # Some bindings return a deque/list of lists; coerce uniformly.
        logits = torch.tensor(list(per_pos), dtype=torch.float32).unsqueeze(0)  # (1, T, V)

        self._output_leaf.set_inputs((ids_t,))
        self._output_leaf.set_outputs(logits)
        return logits

    def supported_visualizers(
        self, visualizers: List[Type["VisualizerBase"]]
    ) -> List[Type["VisualizerBase"]]:
        supported: List[Type["VisualizerBase"]] = []
        for viz_cls in visualizers:
            try:
                if viz_cls.compatible_with(self._output_leaf):
                    supported.append(viz_cls)
            except Exception:
                continue
        return supported
