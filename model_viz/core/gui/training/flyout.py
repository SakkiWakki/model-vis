"""TrainingFlyout: slim control bar + collapsible settings panel.

UI-only: training execution lives in TrainingController.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QSizePolicy,
    QApplication,
)

from model_viz.core.adapter import HyperParamSpec, ModelAdapter, TrainableAdapter
from model_viz.core.gui.event_bus import get_bus
from model_viz.core.gui.training.controller import TrainingController, TrainSnapshot
from model_viz.core.input.base import InputEditor


def _dataset_of(adapter: Optional[ModelAdapter]):
    """Best-effort accessor: concrete adapters keep their dataset as `_dataset`."""
    return getattr(adapter, "_dataset", None)


class TrainingFlyout(QWidget):
    def __init__(
        self,
        adapter: Optional[ModelAdapter] = None,
        on_refresh_visuals: Optional[Callable[[], None]] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._adapter: Optional[ModelAdapter] = None
        self._on_refresh_visuals = on_refresh_visuals or (lambda: None)

        self._controller = TrainingController(self)
        self._controller.updated.connect(self._on_updated)
        self._controller.running_changed.connect(self._on_running_changed)

        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(100)  # 10 Hz UI updates while training
        self._poll_timer.timeout.connect(self._controller.poll)
        app = QApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self.shutdown)

        self._param_widgets: Dict[str, QWidget] = {}

        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Control bar (always visible)
        self._bar = QWidget()
        bar_layout = QVBoxLayout(self._bar)
        bar_layout.setContentsMargins(6, 6, 6, 6)
        bar_layout.setSpacing(6)

        self._train_btn = QPushButton("Train")
        self._train_btn.setCheckable(True)
        self._step_btn = QPushButton("Step")

        bar_layout.addWidget(self._train_btn)
        bar_layout.addWidget(self._step_btn)

        bar_layout.addSpacing(8)

        self._toggle = QToolButton()
        self._toggle.setText("Settings")
        self._toggle.setCheckable(True)
        self._toggle.setChecked(True)
        bar_layout.addWidget(self._toggle)

        bar_layout.addStretch(1)

        self._status = QLabel("")
        self._status.setWordWrap(True)
        self._status.setAlignment(Qt.AlignmentFlag.AlignTop)
        bar_layout.addWidget(self._status)

        self._bar.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        outer.addWidget(self._bar)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.VLine)
        outer.addWidget(sep)

        # Settings panel (collapsible)
        self._panel = QScrollArea()
        self._panel.setWidgetResizable(True)
        self._panel.setMinimumWidth(260)
        self._panel.setMaximumWidth(360)
        self._panel_inner = QWidget()
        self._panel.setWidget(self._panel_inner)

        panel_layout = QVBoxLayout(self._panel_inner)
        panel_layout.setContentsMargins(10, 10, 10, 10)
        panel_layout.setSpacing(8)

        self._title = QLabel("<b>Settings</b>")
        panel_layout.addWidget(self._title)

        self._hp_form_host = QWidget()
        self._hp_form = QFormLayout(self._hp_form_host)
        panel_layout.addWidget(self._hp_form_host)

        # Settings-only actions
        self._reset_btn = QPushButton("Reset")
        self._apply_btn = QPushButton("Apply (Reset)")
        panel_layout.addWidget(self._reset_btn)
        panel_layout.addWidget(self._apply_btn)

        # Input / output sections (one-shot inference with a user-supplied input).
        self._io_sep = QFrame()
        self._io_sep.setFrameShape(QFrame.Shape.HLine)
        panel_layout.addWidget(self._io_sep)

        panel_layout.addWidget(QLabel("<b>Input</b>"))
        self._input_host = QWidget()
        self._input_host_layout = QVBoxLayout(self._input_host)
        self._input_host_layout.setContentsMargins(0, 0, 0, 0)
        panel_layout.addWidget(self._input_host)
        self._input_editor: Optional[InputEditor] = None

        self._run_btn = QPushButton("Run forward")
        panel_layout.addWidget(self._run_btn)

        panel_layout.addWidget(QLabel("<b>Output</b>"))
        self._output_host = QWidget()
        self._output_host_layout = QVBoxLayout(self._output_host)
        self._output_host_layout.setContentsMargins(0, 0, 0, 0)
        panel_layout.addWidget(self._output_host)
        self._output_renderer: Optional[QWidget] = None

        panel_layout.addStretch(1)

        outer.addWidget(self._panel)

        # Wiring
        self._train_btn.toggled.connect(self._on_train_toggled)
        self._step_btn.clicked.connect(self._on_step)
        self._toggle.toggled.connect(self._panel.setVisible)
        self._reset_btn.clicked.connect(self._on_reset)
        self._apply_btn.clicked.connect(self._on_apply)
        self._run_btn.clicked.connect(self._on_run_forward)

        self._panel.setVisible(True)
        self._freeze_bar_width()

        if adapter is not None:
            self.set_adapter(adapter)
        else:
            self._set_enabled(False)

    # ------------------------------------------------------------------
    def set_adapter(self, adapter: Optional[ModelAdapter]) -> None:
        self._adapter = adapter
        self._train_btn.setChecked(False)
        self._poll_timer.stop()
        # Controller needs the adapter even for non-trainable models so that
        # probe_with() (one-shot inference) can run.  Passing None clears it.
        self._controller.set_adapter(adapter)
        if adapter is None:
            self._title.setText("")
            self._set_enabled(False)
            self._status.setText("")
            return
        self._title.setText(f"<b>{adapter.name}</b>")
        self._rebuild_hparams()
        self._rebuild_io()
        trainable = isinstance(adapter, TrainableAdapter)
        self._set_enabled(trainable)
        self._status.setText("")

    def _set_enabled(self, trainable: bool) -> None:
        self._train_btn.setEnabled(trainable)
        self._step_btn.setEnabled(trainable)
        self._reset_btn.setEnabled(trainable)
        self._apply_btn.setEnabled(trainable)
        # Settings toggle and Run forward stay available for any adapter that
        # has an input editor — needed for HF/inference-only models.
        self._toggle.setEnabled(True)
        self._run_btn.setEnabled(self._input_editor is not None)

    def _on_running_changed(self, running: bool) -> None:
        # Keep toggle state consistent if controller stops unexpectedly.
        if not running:
            self._train_btn.blockSignals(True)
            self._train_btn.setChecked(False)
            self._train_btn.blockSignals(False)
            self._poll_timer.stop()

    def _on_train_toggled(self, on: bool) -> None:
        if on:
            self._controller.start()
            self._poll_timer.start()
        else:
            self._poll_timer.stop()
            self._controller.stop()
        # Update visualizations on any control action
        get_bus().model_updated.emit({"adapter": self._adapter, "metrics": {}, "reason": "control"})

    def _on_step(self) -> None:
        self._train_btn.setChecked(False)
        snap = self._controller.step_once()
        if snap is not None:
            self._render_metrics(snap.metrics)
        get_bus().model_updated.emit({"adapter": self._adapter, "metrics": {}, "reason": "step"})

    def _on_reset(self) -> None:
        self._train_btn.setChecked(False)
        self._controller.reset()
        self._status.setText("Training reset.")
        get_bus().model_updated.emit({"adapter": self._adapter, "metrics": {}, "reason": "reset"})

    def _on_apply(self) -> None:
        t = self._adapter if isinstance(self._adapter, TrainableAdapter) else None
        if t is None:
            return
        self._train_btn.setChecked(False)
        values: Dict[str, object] = {
            name: self._read_widget_value(w) for name, w in self._param_widgets.items()
        }
        self._controller.apply_and_reset(values)
        self._status.setText("Applied hyperparameters and reset.")
        get_bus().model_updated.emit({"adapter": self._adapter, "metrics": {}, "reason": "probe"})

    def _on_run_forward(self) -> None:
        if self._adapter is None or self._input_editor is None:
            return
        dataset = _dataset_of(self._adapter)
        if dataset is None:
            return
        inp = self._input_editor.current_value()
        if inp is None:
            self._status.setText("Invalid input.")
            return
        self._train_btn.setChecked(False)
        raw = self._controller.probe_with(inp)
        if raw is None:
            return
        try:
            output = dataset.interpret_output(raw)
        except Exception as e:
            self._status.setText(f"Output error: {e}")
            return
        if self._output_renderer is not None and hasattr(self._output_renderer, "set_output"):
            self._output_renderer.set_output(output)  # type: ignore[attr-defined]
        get_bus().model_updated.emit({"adapter": self._adapter, "metrics": {}, "reason": "probe"})

    # ------------------------------------------------------------------
    def _on_updated(self, snap_obj: object) -> None:
        if isinstance(snap_obj, TrainSnapshot):
            self._render_metrics(snap_obj.metrics)
        # Broadcast and refresh
        get_bus().model_updated.emit({"adapter": self._adapter, "metrics": getattr(snap_obj, "metrics", {}), "reason": "train"})
        self._on_refresh_visuals()

    def _render_metrics(self, metrics: Dict[str, float]) -> None:
        if metrics:
            self._status.setText(", ".join(f"{k}: {v:.4g}" for k, v in metrics.items()))

    # ------------------------------------------------------------------
    def _freeze_bar_width(self) -> None:
        controls: list[QWidget] = [self._train_btn, self._step_btn, self._toggle]
        w = max(c.sizeHint().width() for c in controls) + 2
        self._bar.setFixedWidth(w + 12)
        for c in controls:
            c.setFixedWidth(w)
        self._status.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
        self._status.setFixedWidth(w)

    def _clear_form(self) -> None:
        while self._hp_form.rowCount():
            self._hp_form.removeRow(0)
        self._param_widgets.clear()

    def _rebuild_io(self) -> None:
        # Clear current input editor.
        if self._input_editor is not None:
            w = self._input_editor  # type: ignore[assignment]
            self._input_host_layout.removeWidget(w)  # type: ignore[arg-type]
            w.setParent(None)  # type: ignore[attr-defined]
            w.deleteLater()  # type: ignore[attr-defined]
            self._input_editor = None

        # Clear current output renderer.
        if self._output_renderer is not None:
            self._output_host_layout.removeWidget(self._output_renderer)
            self._output_renderer.setParent(None)
            self._output_renderer.deleteLater()
            self._output_renderer = None

        dataset = _dataset_of(self._adapter)
        if dataset is None:
            return

        # Build the editor for this dataset's input type.
        try:
            editor = dataset.input_type.editor_widget(dataset, parent=self._input_host)
        except Exception:
            editor = None
        if editor is not None:
            self._input_editor = editor
            self._input_host_layout.addWidget(editor)  # type: ignore[arg-type]

        # Build the renderer for this dataset's output type.
        try:
            renderer = dataset.output_type.render_widget(parent=self._output_host)
        except Exception:
            renderer = None
        if renderer is not None:
            self._output_renderer = renderer
            self._output_host_layout.addWidget(renderer)

    def _rebuild_hparams(self) -> None:
        self._clear_form()
        t = self._adapter if isinstance(self._adapter, TrainableAdapter) else None
        if t is None:
            self._hp_form.addRow(QLabel("Hyperparameters"), QLabel("not supported"))
            return
        for name, spec in t.hyperparameters().items():
            w = self._make_param_widget(spec)
            self._param_widgets[name] = w
            label = spec.description or name
            self._hp_form.addRow(QLabel(label), w)

    def _make_param_widget(self, spec: HyperParamSpec) -> QWidget:
        if spec.kind == "int":
            w = QSpinBox()
            if spec.minimum is not None:
                w.setMinimum(int(spec.minimum))
            if spec.maximum is not None:
                w.setMaximum(int(spec.maximum))
            if spec.step is not None:
                w.setSingleStep(int(spec.step))
            w.setValue(int(spec.default))
            return w
        if spec.kind == "float":
            w = QDoubleSpinBox()
            w.setDecimals(6)
            if spec.minimum is not None:
                w.setMinimum(float(spec.minimum))
            if spec.maximum is not None:
                w.setMaximum(float(spec.maximum))
            if spec.step is not None:
                w.setSingleStep(float(spec.step))
            w.setValue(float(spec.default))
            return w
        if spec.kind == "bool":
            w = QCheckBox()
            w.setChecked(bool(spec.default))
            return w
        if spec.kind == "choice":
            w = QComboBox()
            choices = spec.choices or []
            for c in choices:
                w.addItem(str(c), c)
            for i in range(w.count()):
                if w.itemData(i) == spec.default:
                    w.setCurrentIndex(i)
                    break
            return w
        return QLabel(str(spec.default))

    def _read_widget_value(self, w: QWidget) -> object:
        if isinstance(w, QSpinBox):
            return int(w.value())
        if isinstance(w, QDoubleSpinBox):
            return float(w.value())
        if isinstance(w, QCheckBox):
            return bool(w.isChecked())
        if isinstance(w, QComboBox):
            return w.currentData()
        if isinstance(w, QLabel):
            return w.text()
        return None

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self._poll_timer.stop()
        self._controller.shutdown()
        super().closeEvent(event)

    def shutdown(self) -> None:
        self._poll_timer.stop()
        self._controller.shutdown()
