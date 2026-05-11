"""TrainingController: background training runner + UI-safe snapshot polling."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import threading

from PyQt6.QtCore import QObject, QThread, pyqtSignal, pyqtSlot

from model_viz.core.adapter import ModelAdapter, TrainableAdapter
from model_viz.core.input.base import InputBase


@dataclass(frozen=True)
class TrainSnapshot:
    metrics: Dict[str, float]
    step_kind: str  # "train" | "step" | "probe" | "reset"


class _TrainWorker(QObject):
    stopped = pyqtSignal()

    def __init__(self, adapter: TrainableAdapter, steps_per_loop: int) -> None:
        super().__init__()
        self._adapter = adapter
        self._steps_per_loop = max(1, int(steps_per_loop))
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._latest: Optional[TrainSnapshot] = None
        self._dirty = False

    @pyqtSlot()
    def start(self) -> None:
        self._stop_event.clear()
        while not self._stop_event.is_set():
            last: Dict[str, float] = {}
            for _ in range(self._steps_per_loop):
                if self._stop_event.is_set():
                    break
                last = self._adapter.train_step()
            self._adapter.probe()
            if last:
                with self._lock:
                    self._latest = TrainSnapshot(metrics=dict(last), step_kind="train")
                    self._dirty = True
            QThread.msleep(1)
        self.stopped.emit()

    def stop(self) -> None:
        self._stop_event.set()

    def take_latest(self) -> Optional[TrainSnapshot]:
        with self._lock:
            if not self._dirty or self._latest is None:
                return None
            self._dirty = False
            return self._latest


class TrainingController(QObject):
    """Owns a background training thread and exposes a polling API for the UI."""

    updated = pyqtSignal(object)  # TrainSnapshot
    running_changed = pyqtSignal(bool)

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._thread: Optional[QThread] = None
        self._worker: Optional[_TrainWorker] = None
        self._adapter: Optional[ModelAdapter] = None
        self._steps_per_loop = 1

    def set_adapter(self, adapter: Optional[ModelAdapter]) -> None:
        self.stop(wait=True)
        self._adapter = adapter

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.isRunning()

    def start(self) -> None:
        if self._adapter is None or not isinstance(self._adapter, TrainableAdapter) or self.is_running():
            return
        self._thread = QThread()
        self._worker = _TrainWorker(adapter=self._adapter, steps_per_loop=self._steps_per_loop)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.start)
        self._worker.stopped.connect(self._thread.quit)
        self._thread.finished.connect(self._on_finished)
        self._thread.start()
        self.running_changed.emit(True)

    def stop(self, wait: bool = False, timeout_ms: Optional[int] = 2000) -> None:
        if self._worker is not None:
            self._worker.stop()
        if wait and self._thread is not None:
            if timeout_ms is None:
                self._thread.wait()
            else:
                self._thread.wait(timeout_ms)
            if not self._thread.isRunning():
                self._on_finished()
        elif self._thread is not None and not self._thread.isRunning():
            self._on_finished()
        self.running_changed.emit(False)

    def shutdown(self) -> None:
        """Stop training synchronously before Qt destroys this controller.

        Must wait for the worker thread to finish — letting Qt tear down the
        QThread while it's still running causes a segfault on app exit.
        """
        self.stop(wait=True, timeout_ms=5000)
        # Drop the adapter reference so any backing native runtime (Llama,
        # torch model) can be GC'd while the Qt event loop is still alive.
        self._adapter = None

    def poll(self) -> None:
        if self._worker is None:
            return
        snap = self._worker.take_latest()
        if snap is not None:
            self.updated.emit(snap)

    def probe_with(self, inp: InputBase) -> object:
        """Run a one-shot forward pass with a user-provided input.

        Stops training if running, runs ``adapter.forward(inp)`` once, emits a
        ``probe`` snapshot so visualizations refresh, and returns the raw
        forward output (the caller wraps it via ``dataset.interpret_output``).
        Subsequent training resumes its normal ``dataset.probe_input()`` probe.
        """
        if self._adapter is None:
            return None
        self.stop(wait=True)
        raw = self._adapter.forward(inp)
        self.updated.emit(TrainSnapshot(metrics={}, step_kind="probe"))
        return raw

    def step_once(self) -> Optional[TrainSnapshot]:
        if not isinstance(self._adapter, TrainableAdapter):
            return None
        m = self._adapter.train_step()
        self._adapter.probe()
        snap = TrainSnapshot(metrics=dict(m), step_kind="step")
        self.updated.emit(snap)
        return snap

    def reset(self) -> None:
        if not isinstance(self._adapter, TrainableAdapter):
            return
        self.stop(wait=True)
        self._adapter.reset_training()
        self._adapter.probe()
        self.updated.emit(TrainSnapshot(metrics={}, step_kind="reset"))

    def apply_and_reset(self, values: Dict[str, object]) -> None:
        if not isinstance(self._adapter, TrainableAdapter):
            return
        self.stop(wait=True)
        self._adapter.apply_hyperparameters(values)
        # Apply implies reset, per UX.
        self._adapter.reset_training()
        self._adapter.probe()
        self.updated.emit(TrainSnapshot(metrics={}, step_kind="probe"))

    def _on_finished(self) -> None:
        if self._thread is not None:
            self._thread.deleteLater()
        if self._worker is not None:
            self._worker.deleteLater()
        self._thread = None
        self._worker = None
        self.running_changed.emit(False)
