"""MainWindow: QMainWindow that wires Sidebar ↔ TabArea; all signal routing lives here."""
from __future__ import annotations

from typing import Optional, Type

from PyQt6.QtWidgets import QHBoxLayout, QMainWindow, QWidget

from model_viz.core.gui.sidebar.sidebar import Sidebar
from model_viz.core.gui.tabs.tab_area import TabArea
from model_viz.core.gui.training.flyout import TrainingFlyout
from model_viz.core.layer import LayerLike
from model_viz.viz.visualizer_base import VisualizerBase


class MainWindow(QMainWindow):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("model_viz")
        self.resize(1200, 700)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._sidebar = Sidebar()
        self._tab_area = TabArea()
        self._flyout = TrainingFlyout(on_refresh_visuals=self._tab_area.refresh_all)

        layout.addWidget(self._sidebar)
        layout.addWidget(self._tab_area, stretch=1)
        layout.addWidget(self._flyout)

        self._sidebar.visualizer_requested.connect(self._open_viz)
        self._sidebar.model_selected.connect(self._on_model_selected)

    def _on_model_selected(self, adapter) -> None:
        # Explicitly close the previous adapter so any native runtime (llama.cpp
        # Llama) releases its resources while the app is still healthy, rather
        # than at GC time when the C library may already be torn down.
        prev = getattr(self, "_current_adapter", None)
        if prev is not None and prev is not adapter:
            self._close_adapter(prev)
        self._current_adapter = adapter
        self._tab_area.set_model(adapter)
        self._flyout.set_adapter(adapter)

    def _open_viz(self, layer: LayerLike, viz_cls: Type[VisualizerBase]) -> None:
        self._tab_area.add_visualization(layer, viz_cls)

    def _close_adapter(self, adapter) -> None:
        """Best-effort: call ``adapter.close()`` if it exists."""
        try:
            close = getattr(adapter, "close", None)
            if callable(close):
                close()
        except Exception:
            pass

    def closeEvent(self, event) -> None:  # type: ignore[override]
        # Order matters: stop training threads before any C-backed model
        # (llama.cpp Llama, HF nn.Module) loses its references, otherwise the
        # background worker can touch freed Qt objects on its way out.
        self._flyout.shutdown()

        # Force-drop adapter references so any native runtime gets released
        # while the Qt event loop is still alive and Python is in a healthy
        # state.  Otherwise the Llama / nn.Module C++ destructors fire at
        # interpreter atexit, when libllama.so / CUDA contexts have already
        # torn down — a common segfault path on app exit.
        if getattr(self, "_current_adapter", None) is not None:
            self._close_adapter(self._current_adapter)
            self._current_adapter = None
        try:
            self._tab_area.set_model(None)  # type: ignore[arg-type]
        except Exception:
            pass
        try:
            self._flyout.set_adapter(None)  # type: ignore[arg-type]
        except Exception:
            pass
        try:
            self._sidebar.viz_list.set_adapter(None)  # type: ignore[arg-type]
        except Exception:
            pass

        import gc
        gc.collect()
        super().closeEvent(event)
