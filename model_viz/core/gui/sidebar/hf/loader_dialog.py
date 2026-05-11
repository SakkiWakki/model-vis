"""HFLoaderDialog: prompts for a HF model id or local path and loads it."""
from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class _LoadWorker(QThread):
    """Loads the HF model on a background thread to keep the UI responsive."""

    loaded = pyqtSignal(object)  # HFModelBundle
    failed = pyqtSignal(str)

    def __init__(self, name_or_path: str, device: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._name_or_path = name_or_path
        self._device = device

    def run(self) -> None:
        try:
            from model_viz.adapters.hf.loader import load_hf_causal_lm
            bundle = load_hf_causal_lm(self._name_or_path, device=self._device)
            self.loaded.emit(bundle)
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}")


class HFLoaderDialog(QDialog):
    """Modal dialog for entering a HF model id (or local path) and loading it."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Load HuggingFace model")
        self.setModal(True)
        self.resize(480, 180)

        self._bundle = None  # HFModelBundle once loaded
        self._worker: Optional[_LoadWorker] = None

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Model id (e.g. 'Qwen/Qwen2.5-0.5B-Instruct') or local directory:"))

        row = QHBoxLayout()
        self._edit = QLineEdit()
        self._edit.setPlaceholderText("Qwen/Qwen2.5-0.5B-Instruct")
        row.addWidget(self._edit, stretch=1)
        self._browse = QPushButton("Browse…")
        self._browse.clicked.connect(self._on_browse)
        row.addWidget(self._browse)
        layout.addLayout(row)

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

        self._status = QLabel("")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

        self._progress = QProgressBar()
        self._progress.setRange(0, 0)  # indeterminate
        self._progress.setVisible(False)
        layout.addWidget(self._progress)

        self._buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self._buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Load")
        self._buttons.accepted.connect(self._on_load)
        self._buttons.rejected.connect(self.reject)
        layout.addWidget(self._buttons)

    @property
    def bundle(self):
        return self._bundle

    def _on_browse(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Pick a model directory")
        if path:
            self._edit.setText(path)

    def _on_load(self) -> None:
        text = self._edit.text().strip()
        if not text:
            self._status.setText("Enter a model id or path.")
            return
        self._buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(False)
        self._edit.setEnabled(False)
        self._browse.setEnabled(False)
        self._device.setEnabled(False)
        self._progress.setVisible(True)
        self._status.setText(f"Loading {text}…")

        self._worker = _LoadWorker(text, self._device.currentText(), parent=self)
        self._worker.loaded.connect(self._on_loaded)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    def _on_loaded(self, bundle: object) -> None:
        self._bundle = bundle
        self.accept()

    def _on_failed(self, message: str) -> None:
        self._progress.setVisible(False)
        self._buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(True)
        self._edit.setEnabled(True)
        self._browse.setEnabled(True)
        self._device.setEnabled(True)
        self._status.setText(f"Load failed: {message}")
