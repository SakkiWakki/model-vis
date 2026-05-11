"""Generic transformer model definition.

This is a small, self-contained transformer-like model meant for visualization.
It is *not* tied to XOR; presets (like XOR) provide tokenizers/datasets and pick
hyperparameters, while this module only defines the model architecture.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class TransformerConfig:
    vocab_size: int
    d_model: int = 32
    nhead: int = 2
    num_layers: int = 2
    ff_mult: int = 2


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int, ff_mult: int) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_mult * d_model),
            nn.ReLU(),
            nn.Linear(ff_mult * d_model, d_model),
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Attention weights are returned by self.attn; the adapter captures those
        # via forward hooks on the `attn` module itself.
        attn_out, _attn_w = self.attn(
            x, x, x, need_weights=True, average_attn_weights=False
        )
        x = self.ln1(x + attn_out)
        x = self.ln2(x + self.ff(x))
        return x


class TinyTransformerLM(nn.Module):
    """Token embedding + N self-attn blocks + per-position vocab projection.

    Output shape is ``(B, T, V)`` — logits over the vocab at every position,
    suitable for next-token prediction and the per-token perplexity visualizer.
    """

    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(
            [TransformerBlock(cfg.d_model, cfg.nhead, cfg.ff_mult) for _ in range(cfg.num_layers)]
        )
        self.classifier = nn.Linear(cfg.d_model, cfg.vocab_size)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(token_ids)  # (B,S,D)
        for b in self.blocks:
            x = b(x)
        return self.classifier(x)   # (B,S,V)


def build_transformer(cfg: TransformerConfig) -> nn.Module:
    return TinyTransformerLM(cfg)

