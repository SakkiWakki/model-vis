"""Horizontal strip of token chips colored by per-token perplexity."""
from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)


def _color_for(value: float, vmin: float, vmax: float) -> Tuple[int, int, int]:
    """Interpolate green -> yellow -> red across [log(vmin), log(vmax)].

    Perplexity is exp(NLL); the perceptually meaningful axis is therefore the
    log of it.  Linear interpolation in perplexity space collapses everything
    except the largest outlier to the green end whenever a sequence contains
    even one highly-surprising token (and perplexity routinely spans many
    orders of magnitude).  Log interpolation keeps downstream effects
    (e.g. elevated perplexity on tokens following an error) visible.
    """
    # Guard against non-positive values and zero-width ranges.
    if not (vmin > 0 and vmax > 0):
        t = 0.0
    elif vmax <= vmin:
        t = 0.0
    else:
        t = (math.log(value) - math.log(vmin)) / (math.log(vmax) - math.log(vmin))
        t = max(0.0, min(1.0, t))
    # Green (60, 160, 90) -> Yellow (220, 180, 60) -> Red (200, 70, 70)
    if t < 0.5:
        k = t / 0.5
        r = int(60 + (220 - 60) * k)
        g = int(160 + (180 - 160) * k)
        b = int(90 + (60 - 90) * k)
    else:
        k = (t - 0.5) / 0.5
        r = int(220 + (200 - 220) * k)
        g = int(180 + (70 - 180) * k)
        b = int(60 + (70 - 60) * k)
    return r, g, b


class TokenChipStrip(QWidget):
    """Scroll-area of token chips with background color keyed to a scalar."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        outer.addWidget(self._scroll)

        self._inner = QWidget()
        self._row = QHBoxLayout(self._inner)
        self._row.setContentsMargins(6, 6, 6, 6)
        self._row.setSpacing(4)
        self._row.addStretch(1)
        self._scroll.setWidget(self._inner)

    def set_tokens(
        self,
        tokens: Sequence[str],
        values: Sequence[Optional[float]],
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
    ) -> None:
        """Render one chip per token; values[i] is the scalar for tokens[i].

        ``values[i]`` may be ``None`` to render the chip in a neutral color
        (e.g. the first token, for which we have no preceding context).

        ``vmin`` / ``vmax`` pin the color scale to an externally-supplied
        range so multiple strips can be visually compared on the same scale.
        When either is ``None``, the missing bound is derived from this
        strip's own values, preserving single-strip behavior.
        """
        # Clear existing chips (keep the trailing stretch).
        while self._row.count() > 1:
            item = self._row.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

        numeric = [v for v in values if v is not None]
        if vmin is None:
            vmin = min(numeric) if numeric else 0.0
        if vmax is None:
            vmax = max(numeric) if numeric else 1.0

        for tok, val in zip(tokens, values):
            cell = QWidget()
            col = QVBoxLayout(cell)
            col.setContentsMargins(0, 0, 0, 0)
            col.setSpacing(2)

            chip = QLabel(tok)
            chip.setAlignment(Qt.AlignmentFlag.AlignCenter)
            chip.setContentsMargins(8, 4, 8, 4)
            if val is None:
                bg = "rgb(80, 80, 90)"
                tooltip = f"{tok}\nno preceding context"
                value_text = "—"
            else:
                r, g, b = _color_for(val, vmin, vmax)
                bg = f"rgb({r}, {g}, {b})"
                tooltip = f"{tok}\nperplexity: {val:.3f}"
                # Compact, readable formatting across a wide value range.
                if val >= 100:
                    value_text = f"{val:.0f}"
                elif val >= 10:
                    value_text = f"{val:.1f}"
                else:
                    value_text = f"{val:.2f}"
            chip.setStyleSheet(
                f"background: {bg}; color: white; border-radius: 6px; padding: 4px 8px; font-weight: 600;"
            )
            chip.setToolTip(tooltip)

            value_label = QLabel(value_text)
            value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            value_label.setStyleSheet("color: rgba(220, 220, 220, 0.85); font-size: 10px;")
            value_label.setToolTip(tooltip)

            col.addWidget(chip)
            col.addWidget(value_label)
            self._row.insertWidget(self._row.count() - 1, cell)
