"""HFTextDataset: free-form text input paired with a HuggingFace tokenizer.

Designed for inference/inspection of a loaded HF causal LM.  There is no
training corpus — ``batch`` raises if called.  ``interpret_output`` reads the
last position's logits and returns top-k next-token predictions wrapped in a
ClassLabelOutput, which the existing output renderer can display.
"""
from __future__ import annotations

from typing import Any, Tuple

import torch

from model_viz.core.input.text_input import TextInput
from model_viz.core.output.class_label import ClassLabelOutput
from model_viz.data.base import DatasetInfo


class HFTextDataset:
    name: str
    input_type = TextInput
    output_type = ClassLabelOutput

    def __init__(self, tokenizer: object, label: str) -> None:
        self._tokenizer = tokenizer
        self.name = f"hf-text[{label}]"
        vocab_size = int(getattr(tokenizer, "vocab_size", 0)) or len(getattr(tokenizer, "get_vocab", lambda: {})())
        self.info = DatasetInfo(
            vocab_size=vocab_size or None,
            num_classes=None,
            description=f"Free-form text via HF tokenizer ({label}).",
        )

    # ---- input -----------------------------------------------------
    def _encode(self, text: str) -> torch.Tensor:
        enc = self._tokenizer(text, return_tensors="pt", add_special_tokens=False)
        ids = enc["input_ids"]
        return ids  # (1, T)

    def make_input(self, raw: Any) -> TextInput:
        if not isinstance(raw, str):
            raise ValueError("HFTextDataset expects a string input.")
        return TextInput(raw, tokenizer=self._encode)

    def probe_input(self) -> TextInput:
        return TextInput("The quick brown fox", tokenizer=self._encode)

    # ---- training (unsupported) ------------------------------------
    def batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("HFTextDataset has no training corpus.")

    # ---- output ----------------------------------------------------
    def interpret_output(self, raw: Any) -> ClassLabelOutput:
        # Expect (B, T, V) logits; show top-k next-token probs at the last position.
        if not isinstance(raw, torch.Tensor):
            # Some HF models return ModelOutput objects.
            raw = getattr(raw, "logits", None)
            if not isinstance(raw, torch.Tensor):
                raise ValueError("HF model output had no logits tensor.")
        t = raw.detach().float().cpu()
        if t.ndim != 3:
            raise ValueError(f"Expected (B, T, V) logits, got shape {tuple(t.shape)}.")
        logits = t[0, -1]
        probs = torch.softmax(logits, dim=-1)
        k = min(5, probs.shape[-1])
        topk = torch.topk(probs, k=k)
        labels = [self._render_token(int(i)) for i in topk.indices.tolist()]
        return ClassLabelOutput(
            class_id=0,  # always show the top-1 as "predicted"
            probs=topk.values.tolist(),
            labels=labels,
        )

    def decode_token(self, token_id: int) -> str:
        return self._render_token(int(token_id))

    def _render_token(self, token_id: int) -> str:
        try:
            s = self._tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
        except Exception:
            s = str(token_id)
        # Make whitespace visible.
        return s.replace(" ", "·") or "∅"
