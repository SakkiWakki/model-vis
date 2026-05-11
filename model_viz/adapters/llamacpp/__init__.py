"""llama.cpp adapters package.

A separate path from the HF/transformers loaders: keeps GGUF weights quantized
in-place via ``llama-cpp-python``.  Only the perplexity visualizer is supported
here — there is no PyTorch nn.Module to introspect.
"""
from model_viz.adapters.llamacpp.adapter import LlamaCppAdapter
from model_viz.adapters.llamacpp.loader import LlamaCppBundle, load_llamacpp

__all__ = ["LlamaCppAdapter", "LlamaCppBundle", "load_llamacpp"]
