"""Model definitions / factories.

These are *not* adapters. A model definition knows how to build (or load) a model
module and any small helpers (tokenizer, training data) needed by adapters/presets.
"""

from model_viz.models.transformer import TransformerConfig, build_transformer

__all__ = ["TransformerConfig", "build_transformer"]
