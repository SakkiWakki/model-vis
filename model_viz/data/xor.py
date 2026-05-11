"""XOR dataset preset (text).

The task is reframed as next-token prediction so per-position visualizations
(e.g. perplexity) make sense.  The input is a 2-token sequence ``[a, b]`` and
the supervised target is ``[-100, a XOR b]`` — only position 1 carries a real
target; position 0 is ignored.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import torch

from model_viz.core.input.text_input import TextInput
from model_viz.core.output.class_label import ClassLabelOutput
from model_viz.data.base import Dataset, DatasetInfo


IGNORE_INDEX = -100


@dataclass(frozen=True)
class XORVocab:
    stoi: Dict[str, int]
    itos: List[str]

    @classmethod
    def default(cls) -> "XORVocab":
        tokens = ["0", "1"]
        stoi = {t: i for i, t in enumerate(tokens)}
        return cls(stoi=stoi, itos=tokens)

    def encode_pair(self, text: str) -> torch.Tensor:
        parts = [p for p in text.strip().split() if p]
        if len(parts) != 2:
            raise ValueError("Expected exactly two tokens, e.g. '0 1'.")
        ids = [self.stoi[p] for p in parts]
        return torch.tensor([ids], dtype=torch.long)  # (B=1, S=2)


class XORDataset(Dataset):
    name = "xor"
    input_type = TextInput
    output_type = ClassLabelOutput
    info = DatasetInfo(vocab_size=2, num_classes=2, description="XOR over two bits: '0 1' -> 1")

    def __init__(self) -> None:
        self._vocab = XORVocab.default()

    @property
    def vocab(self) -> XORVocab:
        return self._vocab

    def make_input(self, raw: Any) -> TextInput:
        if not isinstance(raw, str):
            raise ValueError("XOR expects a string like '0 1'.")
        return TextInput(raw, tokenizer=self._vocab.encode_pair)

    def probe_input(self) -> TextInput:
        return TextInput("0 1", tokenizer=self._vocab.encode_pair)

    def batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        # Inputs: pairs of bits.  Targets: per-position, with the XOR result at
        # position 1 and an ignore index at position 0.
        texts = ["0 0", "0 1", "1 0", "1 1"]
        xor_results = [0, 1, 1, 0]
        xs = torch.cat([self._vocab.encode_pair(t) for t in texts], dim=0)  # (4,2)
        ys = torch.full_like(xs, IGNORE_INDEX)
        ys[:, 1] = torch.tensor(xor_results, dtype=torch.long)
        return xs, ys

    def decode_token(self, token_id: int) -> str:
        if 0 <= token_id < len(self._vocab.itos):
            return self._vocab.itos[token_id]
        return str(token_id)

    def interpret_output(self, raw: Any) -> ClassLabelOutput:
        # Accepts either per-position logits (B, T, V) — read the last position,
        # which under this task carries the XOR prediction — or a flat (B, V) /
        # (V,) tensor for adapters that produce a single classification.
        if not isinstance(raw, torch.Tensor):
            raise ValueError("XOR expects a tensor output from the model.")
        t = raw.detach().float().cpu()
        if t.ndim == 3:
            logits = t[0, -1]
        elif t.ndim == 2:
            logits = t[0]
        else:
            logits = t
        probs = torch.softmax(logits, dim=-1).tolist()
        class_id = int(torch.argmax(logits).item())
        return ClassLabelOutput(class_id=class_id, probs=probs, labels=["0", "1"])

