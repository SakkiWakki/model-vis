"""Generic MLP classifier model.

This is meant for "traditional NN" visualizations (input -> hidden -> output).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn


@dataclass(frozen=True)
class MLPConfig:
    input_dim: int
    hidden_dims: Sequence[int] = (8, 8)
    num_classes: int = 2


class MLPClassifier(nn.Module):
    def __init__(self, cfg: MLPConfig) -> None:
        super().__init__()
        dims = [cfg.input_dim, *list(cfg.hidden_dims), cfg.num_classes]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i != len(dims) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_mlp(cfg: MLPConfig) -> nn.Module:
    return MLPClassifier(cfg)

