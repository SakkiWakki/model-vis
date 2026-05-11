"""Load a GGUF file via llama-cpp-python, keeping weights quantized in place.

If a system llama.cpp install is found (the Arch chaotic-aur ``llama-cpp-cuda-git``
package, installed at ``/opt/llama-cpp``), prefer its shared libraries so GPU
offload works without needing to rebuild llama-cpp-python locally.  The bundled
CPU-only libs that ship with the Python wheel are used as a fallback.
"""
from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class LlamaCppBundle:
    llama: object         # llama_cpp.Llama instance
    name: str             # human-readable label
    n_vocab: int
    n_ctx: int


# System install locations to check for a CUDA-enabled libllama.so.  The Arch
# ``llama-cpp-cuda-git`` package installs to /opt/llama-cpp/lib.
_SYSTEM_LIB_DIRS = (
    "/opt/llama-cpp/lib",
    "/usr/local/llama-cpp/lib",
    "/usr/lib",
)

# The order matters: each entry's symbols must already be resolved when later
# entries are loaded, so this is rough topological order (base → leaves → root).
_PRELOAD_ORDER = (
    "libggml-base.so",
    "libggml-cpu.so",
    "libggml-blas.so",
    "libggml-cuda.so",
    "libggml-rpc.so",
    "libggml-vulkan.so",
    "libggml.so",
)


_SYSTEM_LIB_LOADED = False


def _try_use_system_libs() -> bool:
    """If a system llama.cpp install is found, preload its libs and point the
    Python binding at them via ``LLAMA_CPP_LIB_PATH``.  Idempotent.

    Returns True if system libs are now in use, False if we fell back to the
    bundled libs that ship with ``llama-cpp-python``.
    """
    global _SYSTEM_LIB_LOADED
    if _SYSTEM_LIB_LOADED:
        return True

    for lib_dir in _SYSTEM_LIB_DIRS:
        candidate = Path(lib_dir) / "libllama.so"
        if not candidate.is_file():
            continue
        try:
            # Preload deps with RTLD_GLOBAL so subsequent dlopens find their
            # symbols in memory rather than needing LD_LIBRARY_PATH.
            for name in _PRELOAD_ORDER:
                p = Path(lib_dir) / name
                if p.is_file():
                    ctypes.CDLL(str(p), mode=ctypes.RTLD_GLOBAL)
            os.environ["LLAMA_CPP_LIB_PATH"] = lib_dir
            _SYSTEM_LIB_LOADED = True
            return True
        except OSError:
            # Bad version mismatch or missing symbol — try the next dir.
            continue
    return False


def load_llamacpp(
    gguf_path: str,
    *,
    n_ctx: int = 4096,
    n_threads: Optional[int] = None,
    n_gpu_layers: int = 0,
    label: Optional[str] = None,
) -> LlamaCppBundle:
    """Load a GGUF via llama-cpp-python.

    ``logits_all=True`` is required so per-position logits are retained for
    the perplexity visualizer.  ``verbose=False`` keeps stderr quiet.
    """
    # Try to wire up the system CUDA-enabled libllama.so before importing
    # llama_cpp, so its bundled CPU-only lib is bypassed.
    _try_use_system_libs()

    # Import lazily so the rest of the app doesn't pay the import cost.
    from llama_cpp import Llama

    # Verbose during load so any underlying llama.cpp error message reaches
    # the terminal — silent failures behind ``verbose=False`` make diagnosis
    # impossible (the Python wrapper only re-raises a generic ValueError).
    llama = Llama(
        model_path=gguf_path,
        n_ctx=n_ctx,
        n_threads=n_threads,
        n_gpu_layers=n_gpu_layers,
        logits_all=True,
        verbose=True,
    )
    return LlamaCppBundle(
        llama=llama,
        name=label or gguf_path,
        n_vocab=int(llama.n_vocab()),
        n_ctx=n_ctx,
    )
