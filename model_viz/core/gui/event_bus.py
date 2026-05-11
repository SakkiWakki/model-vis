"""Event bus for cross-widget updates.

Qt widgets must be updated on the UI thread. This bus provides a single place to
broadcast "model state changed" style events (e.g. training progressed) without
tight coupling between the training controls and specific visualizers/tabs.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from PyQt6.QtCore import QObject, pyqtSignal


class EventBus(QObject):
    """Global UI event bus (UI thread)."""

    # Emitted when a model has advanced training/probe and activations may have changed.
    # payload: {"adapter": adapter, "metrics": dict, "reason": "train"|"step"|"probe"}
    model_updated = pyqtSignal(object)


_BUS: Optional[EventBus] = None


def get_bus() -> EventBus:
    global _BUS
    if _BUS is None:
        _BUS = EventBus()
    return _BUS

