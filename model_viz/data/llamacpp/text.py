"""LlamaCppTextDataset: free-form text input paired with a llama.cpp tokenizer.

Mirrors HFTextDataset but uses the GGUF-embedded tokenizer exposed by
``llama_cpp.Llama.tokenize`` / ``detokenize``.  Implements ``InputCapable``
only (no training corpus).  ``interpret_output`` reads the last position's
logits and returns top-k next-token predictions.
"""
from __future__ import annotations

from typing import Any

import torch

from model_viz.core.input.text_input import TextInput
from model_viz.core.output.class_label import ClassLabelOutput
from model_viz.data.base import DatasetInfo


class LlamaCppTextDataset:
    name: str
    input_type = TextInput
    output_type = ClassLabelOutput

    def __init__(self, llama: object, label: str) -> None:
        self._llama = llama
        self.name = f"llamacpp-text[{label}]"
        n_vocab = int(llama.n_vocab())  # type: ignore[attr-defined]
        self.info = DatasetInfo(
            vocab_size=n_vocab,
            num_classes=None,
            description=f"Free-form text via llama.cpp tokenizer ({label}).",
        )

    # ---- input -----------------------------------------------------
    def _encode(self, text: str) -> torch.Tensor:
        # llama.cpp tokenize expects bytes; add_bos=True puts the BOS token in front.
        toks = self._llama.tokenize(text.encode("utf-8"), add_bos=True)  # type: ignore[attr-defined]
        ids = list(toks)
        return torch.tensor([ids], dtype=torch.long)  # (1, T)

    def make_input(self, raw: Any) -> TextInput:
        if not isinstance(raw, str):
            raise ValueError("LlamaCppTextDataset expects a string input.")
        return TextInput(raw, tokenizer=self._encode)

    def probe_input(self) -> TextInput:
        return TextInput("The quick brown fox", tokenizer=self._encode)

    # ---- output ----------------------------------------------------
    def interpret_output(self, raw: Any) -> ClassLabelOutput:
        if not isinstance(raw, torch.Tensor):
            raise ValueError("Expected a torch tensor of logits.")
        t = raw.detach().float().cpu()
        if t.ndim != 3:
            raise ValueError(f"Expected (B, T, V) logits, got shape {tuple(t.shape)}.")
        logits = t[0, -1]
        probs = torch.softmax(logits, dim=-1)
        k = min(5, probs.shape[-1])
        topk = torch.topk(probs, k=k)
        labels = [self._render_token(int(i)) for i in topk.indices.tolist()]
        return ClassLabelOutput(
            class_id=0,
            probs=topk.values.tolist(),
            labels=labels,
        )

    def decode_token(self, token_id: int) -> str:
        return self._render_token(int(token_id))

    def _render_token(self, token_id: int) -> str:
        try:
            s = self._llama.detokenize([int(token_id)]).decode("utf-8", errors="replace")  # type: ignore[attr-defined]
        except Exception:
            s = str(token_id)
        return s.replace(" ", "·") or "∅"
