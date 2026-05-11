"""model_viz entry point.

This wires up:
1. Example model adapters (from `examples/`)
2. Built-in visualizers (from `model_viz/viz/visualizers/`)
3. The Qt application window
"""
from __future__ import annotations

import argparse
import sys
import warnings

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication

from model_viz.core import registry
from model_viz.data.xor import XORDataset
from model_viz.data.fce import try_load as try_load_fce
from model_viz.adapters.transformer_adapter import TransformerAdapter
from model_viz.adapters.mlp_adapter import MLPAdapter
from model_viz.viz.main_window import MainWindow
from model_viz.viz.visualizers.attention_viz.viz import AttentionVisualizer
from model_viz.viz.visualizers.nn_flow_viz.viz import NNFlowVisualizer
from model_viz.viz.visualizers.perplexity_viz.viz import PerplexityVisualizer


def _register_all() -> None:
    # Datasets
    registry.register_dataset(XORDataset())

    # Optional: the Cambridge FCE corpus (licensed; installed via
    # scripts/setup_fce.sh).  Absence is a warning, never a crash.
    fce = try_load_fce()
    if fce is not None:
        registry.register_dataset(fce)
    else:
        warnings.warn(
            "FCE dataset not installed; skipping registration. "
            "Run scripts/setup_fce.sh to enable it.",
            stacklevel=2,
        )

    # Model factories
    registry.register_model_factory(
        "Transformer",
        lambda ds: TransformerAdapter(name=f"transformer[{ds.name}]", dataset=ds),
    )
    registry.register_model_factory(
        "MLP",
        lambda ds: MLPAdapter(name=f"mlp[{ds.name}]", dataset=ds),
    )

    # Visualizers
    registry.register_visualizer(AttentionVisualizer)
    registry.register_visualizer(NNFlowVisualizer)
    registry.register_visualizer(PerplexityVisualizer)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="model_viz")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a minimal startup smoke test and exit quickly.",
    )
    args = parser.parse_args(argv)

    _register_all()

    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()

    if args.smoke:
        # Give Qt a moment to construct widgets, then exit.
        QTimer.singleShot(150, app.quit)

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
