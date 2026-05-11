"""Discover GGUF models stored locally by Ollama.

Ollama's on-disk layout (Linux defaults: ``~/.ollama`` or ``/var/lib/ollama``)::

    <root>/manifests/<registry>/<owner>/<name>/<tag>     # JSON manifest
    <root>/blobs/sha256-<digest>                          # content-addressed blobs

The manifest is a Docker-style image manifest pointing at one or more layers.
The layer whose ``mediaType`` is ``application/vnd.ollama.image.model`` is the
GGUF weights blob.  The ``config`` digest points to a JSON blob containing
metadata like ``model_family``, ``model_type`` (parameter count string), and
``file_type`` (quantization).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


# Default Ollama storage roots, in priority order.  First one that exists wins.
_DEFAULT_ROOTS = (
    Path.home() / ".ollama",
    Path("/var/lib/ollama"),
    Path("/usr/share/ollama"),
)


@dataclass(frozen=True)
class OllamaModel:
    """A single Ollama model on disk, identified by manifest path."""

    full_name: str            # "huihui_ai/Qwen3.6-abliterated:35b"
    manifest_path: Path
    blob_path: Path           # GGUF weights blob, ready to feed to a GGUF loader
    blob_size: int            # bytes on disk (still quantized)
    family: Optional[str]     # e.g. "qwen35", "llama"
    parameters: Optional[str] # e.g. "26.9B"
    quantization: Optional[str]  # e.g. "Q6_K", "Q4_K_M"

    @property
    def label(self) -> str:
        bits = [self.full_name]
        if self.parameters:
            bits.append(self.parameters)
        if self.quantization:
            bits.append(self.quantization)
        return " · ".join(bits)

    @property
    def quantized_gb(self) -> float:
        return self.blob_size / (1024 ** 3)

    def estimated_dequantized_gb(self, dtype_bytes: int = 2) -> Optional[float]:
        """Rough estimate of memory needed once dequantized to fp16/bf16/fp32.

        Parses ``parameters`` (e.g. "26.9B") and multiplies by bytes per param.
        Returns ``None`` if the parameter count is unparseable.
        """
        if not self.parameters:
            return None
        s = self.parameters.strip().upper()
        try:
            if s.endswith("B"):
                n = float(s[:-1]) * 1e9
            elif s.endswith("M"):
                n = float(s[:-1]) * 1e6
            else:
                n = float(s)
        except ValueError:
            return None
        return (n * dtype_bytes) / (1024 ** 3)


def default_ollama_root() -> Optional[Path]:
    """Return the first Ollama storage root that exists, or None."""
    for root in _DEFAULT_ROOTS:
        if root.is_dir() and (root / "manifests").is_dir():
            return root
    return None


def scan_models(root: Optional[Path] = None) -> List[OllamaModel]:
    """Enumerate Ollama models under ``root`` (or the default).  Empty if none."""
    if root is None:
        root = default_ollama_root()
    if root is None:
        return []

    manifests_dir = root / "manifests"
    blobs_dir = root / "blobs"
    if not manifests_dir.is_dir() or not blobs_dir.is_dir():
        return []

    out: List[OllamaModel] = []
    for path in manifests_dir.rglob("*"):
        if not path.is_file():
            continue
        try:
            model = _read_manifest(path, manifests_dir, blobs_dir)
        except Exception:
            # Skip anything that doesn't parse — don't bring down the picker.
            continue
        if model is not None:
            out.append(model)
    out.sort(key=lambda m: m.full_name.lower())
    return out


def _read_manifest(
    manifest_path: Path, manifests_dir: Path, blobs_dir: Path
) -> Optional[OllamaModel]:
    rel = manifest_path.relative_to(manifests_dir).parts
    # rel = (registry, owner, name, tag) typically; some library models drop owner.
    if len(rel) < 3:
        return None
    tag = rel[-1]
    name_parts = list(rel[1:-1])
    name = "/".join(name_parts) if name_parts else rel[-2]
    # Strip the "library/" prefix for the canonical ollama-style name.
    if name_parts and name_parts[0] == "library":
        name = "/".join(name_parts[1:]) if len(name_parts) > 1 else name_parts[0]
    full_name = f"{name}:{tag}"

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    # Find the GGUF blob layer.
    layers = manifest.get("layers", [])
    blob_digest: Optional[str] = None
    blob_size: int = 0
    for layer in layers:
        if layer.get("mediaType") == "application/vnd.ollama.image.model":
            digest = layer.get("digest", "")
            if digest.startswith("sha256:"):
                blob_digest = digest.split(":", 1)[1]
                blob_size = int(layer.get("size", 0))
            break
    if blob_digest is None:
        return None
    blob_path = blobs_dir / f"sha256-{blob_digest}"
    if not blob_path.is_file():
        return None

    # Pull family / params / quant from the config blob.
    family = parameters = quantization = None
    cfg_digest = manifest.get("config", {}).get("digest", "")
    if cfg_digest.startswith("sha256:"):
        cfg_path = blobs_dir / f"sha256-{cfg_digest.split(':', 1)[1]}"
        if cfg_path.is_file():
            try:
                with open(cfg_path, "r") as f:
                    cfg = json.load(f)
                family = cfg.get("model_family")
                parameters = cfg.get("model_type")
                quantization = cfg.get("file_type")
            except Exception:
                pass

    return OllamaModel(
        full_name=full_name,
        manifest_path=manifest_path,
        blob_path=blob_path,
        blob_size=blob_size,
        family=family,
        parameters=parameters,
        quantization=quantization,
    )
