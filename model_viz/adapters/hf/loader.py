"""Load a HuggingFace causal LM from a model id, local path, or GGUF blob.

Three sources are supported:

1. **HF Hub id** (``"Qwen/Qwen2.5-0.5B-Instruct"``) — downloaded via ``transformers``.
2. **Local HF-format directory** containing ``config.json`` + weights + tokenizer.
3. **GGUF file** (any path that looks like a single file — e.g. an Ollama blob).
   The tokenizer and config are read out of the GGUF metadata; weights are
   dequantized into the resulting ``nn.Module``.  The dequantized model lives
   in RAM at the requested dtype, so a 24 GB Q4 model becomes ~50-70 GB.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class HFModelBundle:
    model: nn.Module
    tokenizer: object
    name: str  # human-readable label


def _looks_like_local_path(name_or_path: str) -> bool:
    if name_or_path.startswith(("/", "./", "../", "~")):
        return True
    if os.path.exists(os.path.expanduser(name_or_path)):
        return True
    return False


def _looks_like_gguf(path: str) -> bool:
    """File-like and either has a .gguf extension or starts with the GGUF magic."""
    if not os.path.isfile(path):
        return False
    if path.lower().endswith(".gguf"):
        return True
    try:
        with open(path, "rb") as f:
            return f.read(4) == b"GGUF"
    except OSError:
        return False


def load_hf_causal_lm(
    name_or_path: str,
    *,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> HFModelBundle:
    """Load a causal LM from a HF id, local directory, or GGUF file."""
    # Import lazily so the rest of the app doesn't need transformers at import time.
    from transformers import AutoModelForCausalLM, AutoTokenizer

    is_local = _looks_like_local_path(name_or_path)
    source = os.path.expanduser(name_or_path) if is_local else name_or_path

    if is_local and os.path.isfile(source):
        if not _looks_like_gguf(source):
            raise ValueError(
                f"Local file is not a recognized GGUF: {source!r}.  "
                "Point at a model directory or a .gguf / Ollama blob file."
            )
        return _load_from_gguf(source, device=device, dtype=dtype, label=name_or_path)

    if is_local and not os.path.isdir(source):
        raise ValueError(
            f"Local path is neither a directory nor a GGUF file: {source!r}"
        )

    # HF Hub id or local HF-format directory.
    tokenizer = AutoTokenizer.from_pretrained(source)
    model = AutoModelForCausalLM.from_pretrained(
        source,
        dtype=dtype,
        attn_implementation="eager",
    )
    model.to(torch.device(device))
    model.eval()
    return HFModelBundle(model=model, tokenizer=tokenizer, name=name_or_path)


def _load_from_gguf(
    gguf_path: str,
    *,
    device: str,
    dtype: torch.dtype,
    label: str,
) -> HFModelBundle:
    """Load a GGUF file by dequantizing it into a HF causal LM."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # `from_pretrained(None, gguf_file=...)` is the documented path for loading
    # a bare GGUF.  Both tokenizer and model read metadata from the same file.
    tokenizer = AutoTokenizer.from_pretrained(None, gguf_file=gguf_path)
    model = AutoModelForCausalLM.from_pretrained(
        None,
        gguf_file=gguf_path,
        dtype=dtype,
        attn_implementation="eager",
    )
    model.to(torch.device(device))
    model.eval()
    return HFModelBundle(model=model, tokenizer=tokenizer, name=label)
