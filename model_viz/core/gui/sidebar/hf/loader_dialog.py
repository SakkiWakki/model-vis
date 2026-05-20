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
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from model_viz.core.gui.sidebar.hf.recents import add_recent, load_recents


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
        # Editable combo: doubles as a free-form text field (when typing) and
        # a recent-models dropdown (when clicking the arrow).  The blank first
        # entry keeps the field empty on open so the placeholder shows and a
        # recent isn't auto-loaded; users still see their history one click away.
        self._edit = QComboBox()
        self._edit.setEditable(True)
        self._edit.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        line = self._edit.lineEdit()
        if line is not None:
            line.setPlaceholderText("Qwen/Qwen2.5-0.5B-Instruct")
            line.setClearButtonEnabled(True)
        self._populate_recents()
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

    def _populate_recents(self) -> None:
        """Reload the combo from the persisted recents file.

        A blank entry sits at index 0 so the dialog opens empty (placeholder
        visible, nothing auto-selected).  Selecting a recent populates the
        editable text field via Qt's default combo behavior.
        """
        self._edit.blockSignals(True)
        self._edit.clear()
        self._edit.addItem("")  # blank "no selection" row
        for entry in load_recents():
            self._edit.addItem(entry)
        self._edit.setCurrentIndex(0)
        line = self._edit.lineEdit()
        if line is not None:
            line.clear()
        self._edit.blockSignals(False)

    def _current_text(self) -> str:
        line = self._edit.lineEdit()
        if line is not None:
            return line.text().strip()
        return self._edit.currentText().strip()

    def _on_browse(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Pick a model directory")
        if path:
            line = self._edit.lineEdit()
            if line is not None:
                line.setText(path)
            else:
                self._edit.setCurrentText(path)

    def _on_load(self) -> None:
        text = self._current_text()
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
        # Record the successfully-loaded entry so it shows up first in the
        # dropdown next time.  Failures are not recorded — only working
        # ids/paths earn a slot in the recents list.
        add_recent(self._current_text())
        self.accept()

    def _on_failed(self, message: str) -> None:
        self._progress.setVisible(False)
        self._buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(True)
        self._edit.setEnabled(True)
        self._browse.setEnabled(True)
        self._device.setEnabled(True)
        self._status.setText(f"Load failed: {message}")
