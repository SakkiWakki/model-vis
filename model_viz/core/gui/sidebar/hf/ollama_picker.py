"""OllamaPickerDialog: pick a locally-stored Ollama model and load it.

Two load paths:

- **Dequantized via transformers (default).**  Reads the GGUF, dequantizes to
  fp16/fp32 PyTorch weights, wraps as an HF causal LM.  Full visualizer support
  (attention, NN-flow, perplexity, …) but memory blows up to model_params *
  dtype_bytes (≈70 GB for a 35B model at bf16).

- **Quantized via llama.cpp (checkbox).**  Loads the GGUF in-place through
  llama-cpp-python.  Stays quantized (no memory blow-up), but only the
  perplexity visualizer works — there is no PyTorch nn.Module to introspect.
"""
from __future__ import annotations

from typing import List, Optional

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from model_viz.adapters.hf.ollama import OllamaModel, scan_models


class _LoadWorker(QThread):
    """Loads the GGUF on a background thread; emits the resulting bundle.

    The bundle type depends on the load mode:
    - ``use_llamacpp=False``: ``HFModelBundle`` (dequantized nn.Module).
    - ``use_llamacpp=True``:  ``LlamaCppBundle`` (quantized llama.cpp runtime).
    """

    loaded = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(
        self,
        blob_path: str,
        label: str,
        device: str,
        use_llamacpp: bool,
        n_gpu_layers: int,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._blob_path = blob_path
        self._label = label
        self._device = device
        self._use_llamacpp = use_llamacpp
        self._n_gpu_layers = n_gpu_layers

    def run(self) -> None:
        try:
            if self._use_llamacpp:
                from model_viz.adapters.llamacpp.loader import load_llamacpp
                bundle = load_llamacpp(
                    self._blob_path,
                    label=self._label,
                    n_gpu_layers=self._n_gpu_layers,
                )
            else:
                from model_viz.adapters.hf.loader import load_hf_causal_lm
                bundle = load_hf_causal_lm(self._blob_path, device=self._device)
                bundle.name = self._label
            self.loaded.emit(bundle)
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}")


class OllamaPickerDialog(QDialog):
    """Lists locally-stored Ollama models for loading via HF or llama.cpp."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Load Ollama model")
        self.setModal(True)
        self.resize(560, 460)

        self._models: List[OllamaModel] = scan_models()
        self._bundle = None
        self._worker: Optional[_LoadWorker] = None

        layout = QVBoxLayout(self)

        if not self._models:
            layout.addWidget(QLabel(
                "No Ollama models found.\n\n"
                "Checked ~/.ollama, /var/lib/ollama, and /usr/share/ollama.\n"
                "If Ollama is installed under a different path, this picker won't see it."
            ))
        else:
            layout.addWidget(QLabel("Pick a model:"))
            self._list = QListWidget()
            for m in self._models:
                line = m.label
                deq = m.estimated_dequantized_gb()
                tail = f"quantized {m.quantized_gb:.1f} GB"
                if deq is not None:
                    tail += f" · ~{deq:.0f} GB dequantized"
                line = f"{line}\n    {tail}"
                self._list.addItem(QListWidgetItem(line))
            self._list.setCurrentRow(0)
            layout.addWidget(self._list, stretch=1)

        # Quantization-mode checkbox.
        self._llamacpp_chk = QCheckBox(
            "Run quantized via llama.cpp (no dequantization; only perplexity visualizer)"
        )
        self._llamacpp_chk.toggled.connect(self._on_mode_toggled)
        layout.addWidget(self._llamacpp_chk)

        dev_row = QHBoxLayout()
        dev_row.addWidget(QLabel("Device:"))
        self._device = QComboBox()
        import torch
        self._device.addItem("cpu")
        if torch.cuda.is_available():
            self._device.addItem("cuda")
        dev_row.addWidget(self._device)
        dev_row.addStretch(1)
        layout.addLayout(dev_row)

        # GPU offload row (only relevant in llama.cpp mode).
        gpu_row = QHBoxLayout()
        self._gpu_label = QLabel("GPU layers:")
        gpu_row.addWidget(self._gpu_label)
        self._gpu_layers = QSpinBox()
        self._gpu_layers.setMinimum(-1)
        self._gpu_layers.setMaximum(1000)
        self._gpu_layers.setValue(-1)  # -1 = offload all layers to GPU
        self._gpu_layers.setToolTip(
            "Layers to offload to GPU.  -1 = all (full GPU), 0 = none (CPU only), "
            "positive N = first N layers on GPU, rest on CPU.\n\n"
            "Requires llama-cpp-python built with CUDA support."
        )
        gpu_row.addWidget(self._gpu_layers)
        gpu_row.addStretch(1)
        layout.addLayout(gpu_row)
        self._gpu_label.setVisible(False)
        self._gpu_layers.setVisible(False)

        self._status = QLabel("")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

        self._progress = QProgressBar()
        self._progress.setRange(0, 0)
        self._progress.setVisible(False)
        layout.addWidget(self._progress)

        self._buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self._buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Load")
        self._buttons.accepted.connect(self._on_load)
        self._buttons.rejected.connect(self.reject)
        self._buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(bool(self._models))
        layout.addWidget(self._buttons)

    @property
    def bundle(self):
        return self._bundle

    # ------------------------------------------------------------------
    def _on_mode_toggled(self, on: bool) -> None:
        # HF mode uses the Device dropdown; llama.cpp mode uses GPU layers instead.
        self._device.setEnabled(not on)
        self._gpu_label.setVisible(on)
        self._gpu_layers.setVisible(on)
        if on:
            self._status.setText(
                "llama.cpp mode: weights stay quantized.  Only the perplexity "
                "visualizer will be available.  GPU offload requires "
                "llama-cpp-python built with CUDA."
            )
        else:
            self._status.setText("")

    def _on_load(self) -> None:
        idx = self._list.currentRow() if self._models else -1
        if idx < 0 or idx >= len(self._models):
            return
        model = self._models[idx]
        use_llamacpp = self._llamacpp_chk.isChecked()

        # If the user asked for GPU offload but the installed llama-cpp-python
        # wasn't built with CUDA, warn loudly so they don't silently end up on CPU.
        if use_llamacpp and self._gpu_layers.value() != 0 and not _llamacpp_supports_gpu():
            if not self._confirm_no_gpu():
                return

        # Only prompt when memory is actually tight — small models don't need it.
        if self._memory_is_tight(model, use_llamacpp):
            if not self._confirm_size(model, use_llamacpp):
                return

        self._set_busy(True)
        self._status.setText(f"Loading {model.full_name} (this may take a while)…")

        self._worker = _LoadWorker(
            blob_path=str(model.blob_path),
            label=model.full_name,
            device=self._device.currentText(),
            use_llamacpp=use_llamacpp,
            n_gpu_layers=int(self._gpu_layers.value()),
            parent=self,
        )
        self._worker.loaded.connect(self._on_loaded)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    def _memory_is_tight(self, model: OllamaModel, use_llamacpp: bool) -> bool:
        """Return True if the load is plausibly close to running out of memory.

        Used to gate the confirmation popup so tiny models don't trigger it.
        Headroom = 85% — anything below that footprint is considered safe.
        """
        HEADROOM = 0.85

        if use_llamacpp:
            footprint_gb = model.quantized_gb
            gpu_layers = self._gpu_layers.value()
            on_gpu = gpu_layers != 0  # -1 = all, positive = partial
        else:
            deq = model.estimated_dequantized_gb()
            if deq is None:
                # No estimate — be conservative and prompt.
                return True
            footprint_gb = deq
            on_gpu = self._device.currentText() == "cuda"

        if on_gpu:
            vram_gb = _vram_total_gb()
            if vram_gb is not None and footprint_gb > vram_gb * HEADROOM:
                return True
            # If we couldn't read VRAM, fall through and also check RAM
            # (offload requires staging through host memory).

        ram_gb = _available_ram_gb()
        if ram_gb is not None and footprint_gb > ram_gb * HEADROOM:
            return True
        return False

    def _confirm_size(self, model: OllamaModel, use_llamacpp: bool) -> bool:
        ram_gb = _available_ram_gb()
        if use_llamacpp:
            msg = (
                f"<b>{model.full_name}</b><br><br>"
                f"On disk (quantized): {model.quantized_gb:.1f} GB<br>"
                "Memory footprint: roughly on-disk size (weights stay quantized).<br>"
            )
            if ram_gb is not None:
                msg += f"Available RAM: ~{ram_gb:.0f} GB<br>"
            msg += "<br>Proceed?"
        else:
            deq_gb = model.estimated_dequantized_gb()
            if deq_gb is None:
                return True  # No estimate available — let it through.
            msg = (
                f"<b>{model.full_name}</b><br><br>"
                f"On disk (quantized): {model.quantized_gb:.1f} GB<br>"
                f"Estimated dequantized size: <b>~{deq_gb:.0f} GB</b> at bf16/fp16<br>"
            )
            if ram_gb is not None:
                msg += f"Available RAM: ~{ram_gb:.0f} GB<br>"
            msg += (
                "<br>Loading dequantizes weights into RAM (and onto the chosen "
                "device).  If this exceeds memory, the load will OOM.  "
                "Tip: check 'Run quantized via llama.cpp' above to avoid this.<br><br>"
                "Proceed anyway?"
            )
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle("Confirm GGUF load")
        box.setTextFormat(Qt.TextFormat.RichText)
        box.setText(msg)
        box.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        box.setDefaultButton(QMessageBox.StandardButton.No)
        return box.exec() == QMessageBox.StandardButton.Yes

    def _set_busy(self, busy: bool) -> None:
        self._progress.setVisible(busy)
        self._buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(not busy)
        if hasattr(self, "_list"):
            self._list.setEnabled(not busy)
        self._device.setEnabled(not busy and not self._llamacpp_chk.isChecked())
        self._gpu_layers.setEnabled(not busy)
        self._llamacpp_chk.setEnabled(not busy)

    def _on_loaded(self, bundle: object) -> None:
        self._bundle = bundle
        self.accept()

    def _on_failed(self, message: str) -> None:
        self._set_busy(False)
        self._status.setText(f"Load failed: {message}")

    def _confirm_no_gpu(self) -> bool:
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle("llama-cpp-python has no GPU support")
        box.setTextFormat(Qt.TextFormat.RichText)
        box.setText(
            "<b>GPU offload requested but not available.</b><br><br>"
            "The installed <code>llama-cpp-python</code> was not built with CUDA "
            "support, so layers will run on CPU regardless of the GPU-layers "
            "setting.<br><br>"
            "To enable GPU offload, rebuild with:<br>"
            "<pre>CMAKE_ARGS=\"-DGGML_CUDA=on\" \\<br>"
            "  pip install --force-reinstall --no-cache-dir llama-cpp-python</pre>"
            "Proceed anyway (CPU only)?"
        )
        box.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        box.setDefaultButton(QMessageBox.StandardButton.No)
        return box.exec() == QMessageBox.StandardButton.Yes


def _available_ram_gb() -> Optional[float]:
    try:
        import psutil
        return psutil.virtual_memory().available / (1024 ** 3)
    except Exception:
        return None


def _vram_total_gb() -> Optional[float]:
    """Total VRAM on the first CUDA device, or None if no GPU / detection fails."""
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        props = torch.cuda.get_device_properties(0)
        return props.total_memory / (1024 ** 3)
    except Exception:
        return None


def _llamacpp_supports_gpu() -> bool:
    """Return True if llama.cpp (system or bundled) supports GPU offload."""
    try:
        # Prefer the system CUDA-enabled libs if present; this also matches
        # what load_llamacpp will use, so the check is consistent.
        from model_viz.adapters.llamacpp.loader import _try_use_system_libs
        _try_use_system_libs()
        import llama_cpp
        return bool(llama_cpp.llama_supports_gpu_offload())
    except Exception:
        return False
