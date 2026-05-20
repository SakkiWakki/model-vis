"""ContradictionVisualizer: probe per-layer hidden-state geometry of contradiction.

Background
----------
The first experiment (`contradiction_probe.py`, separate repo) embedded sentences
independently with nomic-embed-text-v1.5 and looked for a single contradiction
direction.  Result was firmly negative — embedding sentences in isolation throws
away the very interaction that makes two statements contradictory.

This visualizer takes the opposite approach: feed both statements as a single
sequence ("Fact 1: {a}. Fact 2: {b}.") to a causal LM, then read the model's
hidden state at the last token of each transformer layer.  The signal we are
looking for is the difference between

    h(contradiction pair) - h(matched compatible pair)

where the matched compatible pair shares the surface form ("Fact 1: ... Fact 2: ...
about the same subject") but doesn't conflict.  If contradiction has a stable
linear representation at some layer, those difference vectors should cluster.

Single-pair mode
----------------
For one (a, b) pair the visualizer also draws a bipartite premise/hypothesis
graph: tokens of Fact 1 on top, tokens of Fact 2 on the bottom, with edges
weighted by attention from each Fact-2 token back to each Fact-1 token at a
user-selectable layer/head.  Inspired by Liu et al. 2018 (EMNLP demos).

Compatibility
-------------
Only attaches to the model's final leaf when the adapter advertises
``produces_per_position_logits`` (the HF causal LM head).  The visualizer
reaches into ``adapter._model`` directly to request hidden states + attentions
in one shot, so we don't depend on the adapter capturing them via hooks.

Dataset
-------
Loads pairs from ``data/contradiction/pairs.json`` by default.  Each entry has
``a`` (Fact 1), ``b_contradict`` (Fact 2 that conflicts with a), and
``b_compatible`` (Fact 2 that doesn't), plus a ``category`` field
(syntactic_negation / semantic_contradiction / temporal_update).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from model_viz.core import registry
from model_viz.core.layer import LayerLike
from model_viz.data.base import EvalCapable
from model_viz.data.contradiction import ContradictionPairSpec
from model_viz.viz.visualizer_base import VisualizerBase
from model_viz.viz.visualizers.contradiction_viz.components.bipartite_attn import (
    BipartiteAttentionWidget,
)
from model_viz.viz.visualizers.contradiction_viz.components.layer_curves import (
    LayerCurvesWidget,
)


_REPO_ROOT = Path(__file__).resolve().parents[5]
_DEFAULT_PAIRS = _REPO_ROOT / "data" / "contradiction" / "pairs.json"
_RESULTS_DIR = _REPO_ROOT / "data" / "contradiction" / "results"


# ----------------------------------------------------------------------
# Pair templating
# ----------------------------------------------------------------------


def _build_text(a: str, b: str) -> str:
    """Single canonical paired-input template.  Both statements share context."""
    a = a.strip().rstrip(".")
    b = b.strip().rstrip(".")
    return f"Fact 1: {a}. Fact 2: {b}."


# ----------------------------------------------------------------------
# Captured tensors for one forward pass
# ----------------------------------------------------------------------


@dataclass
class PairCapture:
    """All numbers we need from one forward pass over a paired input."""

    text: str
    token_ids: List[int]
    tokens: List[str]
    # hidden states at the last position, one row per layer: shape (n_layers+1, hidden)
    last_token_states: np.ndarray
    # full attentions if captured, otherwise None.  Stored as a list of (H, S, S) arrays.
    attentions: Optional[List[np.ndarray]] = None


# ----------------------------------------------------------------------
# Dataset row: one (contradiction, matched-compatible) pair
# ----------------------------------------------------------------------


@dataclass
class PairResult:
    category: str
    a: str
    b_contradict: str
    b_compatible: str
    cap_contradict: Optional[PairCapture] = None
    cap_compatible: Optional[PairCapture] = None


@dataclass
class DatasetResult:
    pairs: List[PairResult] = field(default_factory=list)
    # cosine_within[category] -> per-layer mean within-category cosine of d_i
    cosine_within: Dict[str, List[float]] = field(default_factory=dict)
    # cross_category[pair_label "catA|catB"] -> per-layer cosine of mean directions
    cross_category: Dict[str, List[float]] = field(default_factory=dict)
    # probe_acc[category] -> per-layer logistic-probe accuracy (LOO on that category)
    probe_acc: Dict[str, List[float]] = field(default_factory=dict)
    probe_acc_all: List[float] = field(default_factory=list)
    n_layers: int = 0


# ----------------------------------------------------------------------
# Forward-pass helper
# ----------------------------------------------------------------------


@torch.no_grad()
def _capture(
    text: str,
    *,
    model: Any,
    tokenizer: Any,
    want_attentions: bool,
) -> Optional[PairCapture]:
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"]
    if input_ids.numel() < 2:
        return None
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)

    out = model(
        input_ids=input_ids,
        output_hidden_states=True,
        output_attentions=want_attentions,
        use_cache=False,
        return_dict=True,
    )

    hs = out.hidden_states  # tuple of (1, T, hidden), length n_layers + 1
    last = np.stack(
        [h[0, -1, :].detach().float().cpu().numpy() for h in hs],
        axis=0,
    )  # (n_layers+1, hidden)

    attns: Optional[List[np.ndarray]] = None
    if want_attentions and getattr(out, "attentions", None) is not None:
        attns = []
        for a in out.attentions:
            # a: (1, H, S, S)
            if a is None:
                continue
            attns.append(a[0].detach().float().cpu().numpy())

    ids = input_ids[0].detach().cpu().tolist()
    tokens = [
        _decode_one(tokenizer, tid) for tid in ids
    ]
    return PairCapture(
        text=text,
        token_ids=ids,
        tokens=tokens,
        last_token_states=last,
        attentions=attns,
    )


def _decode_one(tokenizer: Any, token_id: int) -> str:
    try:
        s = tokenizer.decode([int(token_id)], clean_up_tokenization_spaces=False)
    except Exception:
        s = str(token_id)
    return s


# ----------------------------------------------------------------------
# Analysis (geometry + probe)
# ----------------------------------------------------------------------


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _within_category_cosine(diffs: np.ndarray) -> float:
    """Mean pairwise cosine among rows of ``diffs`` (n, D)."""
    n = diffs.shape[0]
    if n < 2:
        return 0.0
    # Normalize once and use Gram trick.
    norms = np.linalg.norm(diffs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    nd = diffs / norms
    gram = nd @ nd.T  # (n, n)
    # Exclude self-pairs from the mean.
    mask = ~np.eye(n, dtype=bool)
    return float(gram[mask].mean())


def _logistic_probe_loo(X: np.ndarray, y: np.ndarray) -> float:
    """Leave-one-out logistic-regression accuracy.

    Tiny dataset → LOO is feasible and avoids the variance of a single split.
    Uses sklearn if available (handles regularization sanely); otherwise falls
    back to a plain numpy gradient solver.
    """
    n = X.shape[0]
    if n < 4 or len(np.unique(y)) < 2:
        return float("nan")
    try:
        from sklearn.linear_model import LogisticRegression
        correct = 0
        for i in range(n):
            mask = np.arange(n) != i
            clf = LogisticRegression(
                C=1.0, max_iter=200, solver="lbfgs", n_jobs=1,
            )
            clf.fit(X[mask], y[mask])
            correct += int(clf.predict(X[i:i+1])[0] == y[i])
        return correct / n
    except Exception:
        return _logistic_probe_loo_numpy(X, y)


def _logistic_probe_loo_numpy(X: np.ndarray, y: np.ndarray) -> float:
    n, d = X.shape
    correct = 0
    for i in range(n):
        mask = np.arange(n) != i
        Xtr = X[mask]
        ytr = y[mask].astype(np.float32)
        # tiny ridge logistic via batch gradient
        w = np.zeros(d, dtype=np.float32)
        b = 0.0
        lr = 0.05
        for _ in range(200):
            z = Xtr @ w + b
            p = 1.0 / (1.0 + np.exp(-z))
            g_w = Xtr.T @ (p - ytr) / len(ytr) + 1e-3 * w
            g_b = float((p - ytr).mean())
            w -= lr * g_w
            b -= lr * g_b
        pred = (X[i] @ w + b) > 0.0
        correct += int(pred == bool(y[i]))
    return correct / n


# ----------------------------------------------------------------------
# Visualizer
# ----------------------------------------------------------------------


class ContradictionVisualizer(VisualizerBase):
    display_name = "Contradiction Probe"

    def __init__(
        self,
        layer: LayerLike,
        adapter: Optional[object] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        # --- Live captures from the most recent single-pair run.
        self._cap_a: Optional[PairCapture] = None
        self._cap_b_contradict: Optional[PairCapture] = None
        self._cap_b_compatible: Optional[PairCapture] = None
        self._single_cap: Optional[PairCapture] = None
        self._dataset_result: Optional[DatasetResult] = None
        self._layer_picker: Optional[QComboBox] = None
        self._head_picker: Optional[QComboBox] = None

        # --- Build widgets BEFORE super().__init__ runs refresh().
        self._fact1 = QLineEdit()
        self._fact1.setPlaceholderText("Fact 1 (e.g. Yucky lives in Tokyo.)")
        self._fact2 = QLineEdit()
        self._fact2.setPlaceholderText("Fact 2 (e.g. Yucky lives in New York.)")
        self._run_pair_btn = QPushButton("Run pair")
        self._run_dataset_btn = QPushButton("Run dataset")
        self._save_btn = QPushButton("Save")
        self._status = QLabel("")
        self._status.setWordWrap(True)
        self._status.setStyleSheet("color: #cccccc; font-size: 10px;")
        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setVisible(False)

        # Single-pair tab
        self._bipartite = BipartiteAttentionWidget()
        self._layer_picker = QComboBox()
        self._head_picker = QComboBox()
        self._layer_picker.currentIndexChanged.connect(self._refresh_bipartite)
        self._head_picker.currentIndexChanged.connect(self._refresh_bipartite)
        self._pair_info = QLabel("")
        self._pair_info.setWordWrap(True)
        self._pair_info.setStyleSheet("color: #cccccc; font-size: 10px;")

        # Dataset tab — three line plots stacked
        self._curve_within = LayerCurvesWidget(
            title="Within-category cosine (h_contradict − h_compatible)"
        )
        self._curve_cross = LayerCurvesWidget(
            title="Cross-category direction cosine"
        )
        self._curve_probe = LayerCurvesWidget(
            title="Linear-probe accuracy (LOO)"
        )

        super().__init__(layer=layer, adapter=adapter, parent=parent)

        # --- Top-level layout
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(4)

        # Input row
        in_row = QHBoxLayout()
        in_row.addWidget(self._fact1, stretch=1)
        in_row.addWidget(self._fact2, stretch=1)
        in_row.addWidget(self._run_pair_btn)
        in_row.addWidget(self._run_dataset_btn)
        in_row.addWidget(self._save_btn)
        root.addLayout(in_row)
        root.addWidget(self._progress)
        root.addWidget(self._status)

        # Tabs
        tabs = QTabWidget()

        # ---- single-pair tab
        single = QWidget()
        single_layout = QVBoxLayout(single)
        single_layout.setContentsMargins(0, 0, 0, 0)
        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("Layer:"))
        ctrl.addWidget(self._layer_picker)
        ctrl.addWidget(QLabel("Head:"))
        ctrl.addWidget(self._head_picker)
        ctrl.addStretch(1)
        single_layout.addLayout(ctrl)
        single_layout.addWidget(self._bipartite, stretch=1)
        single_layout.addWidget(self._pair_info)
        tabs.addTab(single, "Single pair")

        # ---- dataset tab
        ds = QSplitter(Qt.Orientation.Vertical)
        ds.addWidget(self._curve_within)
        ds.addWidget(self._curve_cross)
        ds.addWidget(self._curve_probe)
        ds.setSizes([220, 220, 220])
        tabs.addTab(ds, "Dataset")

        root.addWidget(tabs, stretch=1)

        # Wire actions
        self._run_pair_btn.clicked.connect(self._on_run_pair)
        self._run_dataset_btn.clicked.connect(self._on_run_dataset)
        self._save_btn.clicked.connect(self._on_save)
        self._fact1.returnPressed.connect(self._on_run_pair)
        self._fact2.returnPressed.connect(self._on_run_pair)

    # ------------------------------------------------------------------
    @classmethod
    def compatible_with(cls, layer: LayerLike) -> bool:
        # Only the final leaf of a causal LM — same gate as PerplexityVisualizer.
        if getattr(layer, "next_layer", None) is not None:
            return False
        adapter = getattr(layer, "_adapter", None)
        return bool(getattr(adapter, "produces_per_position_logits", False))

    # ------------------------------------------------------------------
    def refresh(self) -> None:
        # The visualizer is input-driven; nothing to do on automatic refresh.
        pass

    # ------------------------------------------------------------------
    # Helpers to reach into the adapter without modifying it
    # ------------------------------------------------------------------

    def _model_and_tokenizer(self) -> Optional[Tuple[Any, Any]]:
        adapter = self._adapter
        if adapter is None:
            return None
        model = getattr(adapter, "_model", None)
        dataset = getattr(adapter, "_dataset", None)
        tokenizer = getattr(dataset, "_tokenizer", None) if dataset is not None else None
        if model is None or tokenizer is None:
            return None
        return model, tokenizer

    # ------------------------------------------------------------------
    # Single-pair flow
    # ------------------------------------------------------------------

    def _on_run_pair(self) -> None:
        mt = self._model_and_tokenizer()
        if mt is None:
            self._status.setText("No HF causal LM loaded — pick a model from the sidebar.")
            return
        model, tokenizer = mt
        a = self._fact1.text().strip()
        b = self._fact2.text().strip()
        if not a or not b:
            self._status.setText("Enter both Fact 1 and Fact 2.")
            return
        text = _build_text(a, b)
        try:
            cap = _capture(text, model=model, tokenizer=tokenizer, want_attentions=True)
        except Exception as e:
            self._status.setText(f"Forward failed: {e}")
            return
        if cap is None:
            self._status.setText("Tokenization too short.")
            return
        self._single_cap = cap
        self._populate_layer_head_pickers()
        self._refresh_bipartite()
        self._status.setText(
            f"Captured {cap.last_token_states.shape[0]} layers, "
            f"{len(cap.tokens)} tokens.  Use the dataset tab to run the full probe."
        )

    def _populate_layer_head_pickers(self) -> None:
        if self._single_cap is None or self._single_cap.attentions is None:
            return
        attns = self._single_cap.attentions
        n_layers = len(attns)
        n_heads = attns[0].shape[0] if n_layers else 0

        # Block signals to avoid re-rendering on every addItem.
        self._layer_picker.blockSignals(True)
        self._head_picker.blockSignals(True)
        self._layer_picker.clear()
        for i in range(n_layers):
            self._layer_picker.addItem(f"Layer {i}")
        # Default to a middle layer where attention is typically most informative.
        self._layer_picker.setCurrentIndex(min(n_layers - 1, n_layers // 2))

        self._head_picker.clear()
        self._head_picker.addItem("Mean of heads")
        for h in range(n_heads):
            self._head_picker.addItem(f"Head {h}")
        self._head_picker.setCurrentIndex(0)
        self._layer_picker.blockSignals(False)
        self._head_picker.blockSignals(False)

    def _refresh_bipartite(self) -> None:
        cap = self._single_cap
        if cap is None or cap.attentions is None or not cap.attentions:
            self._bipartite.clear()
            return
        L = self._layer_picker.currentIndex()
        if L < 0 or L >= len(cap.attentions):
            return
        attn = cap.attentions[L]  # (H, S, S)
        h_idx = self._head_picker.currentIndex()
        if h_idx <= 0:
            A = attn.mean(axis=0)  # (S, S)
            head_label = "mean heads"
        else:
            A = attn[h_idx - 1]
            head_label = f"head {h_idx - 1}"

        # Split tokens into Fact 1 and Fact 2 chunks.
        split = _find_fact2_start(cap.tokens)
        if split is None or split <= 0 or split >= len(cap.tokens):
            self._bipartite.clear()
            self._pair_info.setText("Could not locate 'Fact 2:' marker in tokens.")
            return
        top_tokens = cap.tokens[:split]
        bottom_tokens = cap.tokens[split:]

        # Bottom→top attention block: rows are Fact-2 token positions, cols are Fact-1 positions.
        weights = A[split:, :split]  # (n_bottom, n_top)
        self._bipartite.set_data(
            top_tokens=top_tokens,
            bottom_tokens=bottom_tokens,
            weights=weights,
            title=f"Layer {L} · {head_label}  (Fact 2 → Fact 1 attention)",
        )
        self._pair_info.setText(
            f"{len(top_tokens)} top tokens · {len(bottom_tokens)} bottom tokens · "
            f"max attn = {float(weights.max()):.3f}"
        )

    # ------------------------------------------------------------------
    # Dataset flow
    # ------------------------------------------------------------------

    def _load_pairs(self) -> List[ContradictionPairSpec]:
        """Find the first registered ``EvalCapable`` dataset that yields
        :class:`ContradictionPairSpec` examples and return them as a list.

        Falls back to reading ``data/contradiction/pairs.json`` directly if
        no matching dataset has been registered (so the visualizer still
        works in setups where ``main.py`` did not register the dataset).
        """
        for ds in registry.get_datasets().values():
            if not isinstance(ds, EvalCapable):
                continue
            try:
                examples = list(ds.iter_examples())
            except Exception:
                continue
            if examples and isinstance(examples[0], ContradictionPairSpec):
                return examples  # type: ignore[return-value]

        # Fallback: read the JSON directly.
        if not _DEFAULT_PAIRS.is_file():
            raise FileNotFoundError(
                "No contradiction-pairs EvalCapable dataset is registered and "
                f"no fallback file exists at {_DEFAULT_PAIRS}."
            )
        with _DEFAULT_PAIRS.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        return [
            ContradictionPairSpec(
                category=str(p.get("category", "uncategorized")),
                a=str(p["a"]),
                b_contradict=str(p["b_contradict"]),
                b_compatible=str(p["b_compatible"]),
            )
            for p in obj.get("pairs", [])
        ]

    def _on_run_dataset(self) -> None:
        mt = self._model_and_tokenizer()
        if mt is None:
            self._status.setText("No HF causal LM loaded — pick a model from the sidebar.")
            return
        model, tokenizer = mt
        try:
            pair_specs = self._load_pairs()
        except Exception as e:
            self._status.setText(f"Could not load pairs: {e}")
            return
        if not pair_specs:
            self._status.setText("Pair file is empty.")
            return

        self._progress.setVisible(True)
        self._progress.setRange(0, len(pair_specs))
        self._progress.setValue(0)
        self._status.setText("Running…")

        results = DatasetResult()
        # The total forward count is 2 per spec (contradict + compatible).
        for idx, spec in enumerate(pair_specs):
            a = spec.a
            b_c = spec.b_contradict
            b_k = spec.b_compatible
            cat = spec.category
            text_c = _build_text(a, b_c)
            text_k = _build_text(a, b_k)
            try:
                cap_c = _capture(text_c, model=model, tokenizer=tokenizer, want_attentions=False)
                cap_k = _capture(text_k, model=model, tokenizer=tokenizer, want_attentions=False)
            except Exception as e:
                self._status.setText(f"Forward failed on pair {idx}: {e}")
                self._progress.setVisible(False)
                return
            if cap_c is None or cap_k is None:
                continue
            results.pairs.append(PairResult(
                category=cat, a=a, b_contradict=b_c, b_compatible=b_k,
                cap_contradict=cap_c, cap_compatible=cap_k,
            ))
            self._progress.setValue(idx + 1)
            # Keep Qt responsive.
            from PyQt6.QtWidgets import QApplication
            QApplication.processEvents()

        if not results.pairs:
            self._progress.setVisible(False)
            self._status.setText("No pairs captured.")
            return

        # All captures share n_layers+1 rows (one per layer index).
        n_layer_rows = results.pairs[0].cap_contradict.last_token_states.shape[0]
        results.n_layers = n_layer_rows

        self._analyze(results)
        self._dataset_result = results
        self._render_dataset_curves(results)
        self._progress.setVisible(False)
        cats = sorted({p.category for p in results.pairs})
        self._status.setText(
            f"Done. {len(results.pairs)} pairs across categories: {', '.join(cats)}."
        )

    def _analyze(self, results: DatasetResult) -> None:
        n_layer_rows = results.n_layers
        # Group pairs by category
        by_cat: Dict[str, List[PairResult]] = {}
        for p in results.pairs:
            by_cat.setdefault(p.category, []).append(p)

        # Per layer: compute difference vectors per category.
        # diffs_by_cat[L][cat] -> ndarray of shape (n_pairs_in_cat, hidden_dim)
        # We'll collect per-category, per-layer arrays once.
        diffs: Dict[str, np.ndarray] = {}  # cat -> (n_pairs, n_layers, hidden)
        for cat, plist in by_cat.items():
            stacks = []
            for p in plist:
                # h(contradiction) - h(matched_compatible) at each layer
                d = p.cap_contradict.last_token_states - p.cap_compatible.last_token_states
                stacks.append(d)
            diffs[cat] = np.stack(stacks, axis=0)  # (n, L, D)

        # Within-category cosine per layer
        for cat, arr in diffs.items():
            ys = []
            for L in range(n_layer_rows):
                ys.append(_within_category_cosine(arr[:, L, :]))
            results.cosine_within[cat] = ys

        # Cross-category direction cosine: cosine between mean diff vectors of two categories at each layer.
        cats_sorted = sorted(diffs.keys())
        for i in range(len(cats_sorted)):
            for j in range(i + 1, len(cats_sorted)):
                ci, cj = cats_sorted[i], cats_sorted[j]
                ys = []
                for L in range(n_layer_rows):
                    mi = diffs[ci][:, L, :].mean(axis=0)
                    mj = diffs[cj][:, L, :].mean(axis=0)
                    ys.append(_cosine(mi, mj))
                results.cross_category[f"{ci} ↔ {cj}"] = ys

        # Linear probe: per layer, leave-one-out on (h_contradict, h_compatible).
        # Per category and across all categories combined.
        for cat, plist in by_cat.items():
            ys = []
            for L in range(n_layer_rows):
                X_c = np.stack([p.cap_contradict.last_token_states[L] for p in plist], axis=0)
                X_k = np.stack([p.cap_compatible.last_token_states[L] for p in plist], axis=0)
                X = np.concatenate([X_c, X_k], axis=0)
                y = np.concatenate([
                    np.ones(len(plist), dtype=np.int64),
                    np.zeros(len(plist), dtype=np.int64),
                ])
                ys.append(_logistic_probe_loo(X, y))
            results.probe_acc[cat] = ys

        ys_all = []
        for L in range(n_layer_rows):
            X_c = np.stack([p.cap_contradict.last_token_states[L] for p in results.pairs], axis=0)
            X_k = np.stack([p.cap_compatible.last_token_states[L] for p in results.pairs], axis=0)
            X = np.concatenate([X_c, X_k], axis=0)
            y = np.concatenate([
                np.ones(len(results.pairs), dtype=np.int64),
                np.zeros(len(results.pairs), dtype=np.int64),
            ])
            ys_all.append(_logistic_probe_loo(X, y))
        results.probe_acc_all = ys_all

    def _render_dataset_curves(self, r: DatasetResult) -> None:
        self._curve_within.set_curves(
            r.cosine_within,
            ylabel="mean cosine",
            ylim=(-0.2, 1.0),
            hline=0.0,
        )
        self._curve_cross.set_curves(
            r.cross_category,
            ylabel="cosine of means",
            ylim=(-1.0, 1.0),
            hline=0.0,
        )
        probe = dict(r.probe_acc)
        probe["all"] = r.probe_acc_all
        self._curve_probe.set_curves(
            probe,
            ylabel="LOO accuracy",
            ylim=(0.0, 1.05),
            hline=0.5,
        )

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def _on_save(self) -> None:
        r = self._dataset_result
        if r is None:
            self._status.setText("Nothing to save — run the dataset first.")
            return
        # Drop each save into a timestamped subdir so successive runs are kept
        # side-by-side rather than overwriting.  Sortable lexicographically.
        from datetime import datetime
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = _RESULTS_DIR / stamp
        run_dir.mkdir(parents=True, exist_ok=True)

        # JSON summary (curves + pair metadata; no big tensors)
        summary = {
            "n_layers": r.n_layers,
            "cosine_within": r.cosine_within,
            "cross_category": r.cross_category,
            "probe_acc": r.probe_acc,
            "probe_acc_all": r.probe_acc_all,
            "pairs": [
                {
                    "category": p.category,
                    "a": p.a,
                    "b_contradict": p.b_contradict,
                    "b_compatible": p.b_compatible,
                }
                for p in r.pairs
            ],
        }
        (run_dir / "summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )

        # NPZ with the raw last-token states for each pair (so model-vis can re-render
        # without re-running the model).  One array per pair, keyed by index.
        npz_payload: Dict[str, np.ndarray] = {}
        for i, p in enumerate(r.pairs):
            npz_payload[f"pair_{i}_contradict"] = p.cap_contradict.last_token_states
            npz_payload[f"pair_{i}_compatible"] = p.cap_compatible.last_token_states
        np.savez_compressed(run_dir / "hidden_states.npz", **npz_payload)

        # The three per-layer plots as PNGs, so the experiment artifact is
        # self-contained (no need to relaunch the GUI to read the figures).
        try:
            self._curve_within.save_png(run_dir / "cosine_within.png")
            self._curve_cross.save_png(run_dir / "cross_category.png")
            self._curve_probe.save_png(run_dir / "probe_accuracy.png")
        except Exception as e:
            self._status.setText(
                f"Saved JSON + hidden states to {run_dir}, "
                f"but PNG export failed: {e}"
            )
            return

        self._status.setText(f"Saved to {run_dir}")


# ----------------------------------------------------------------------
# Token-split heuristic
# ----------------------------------------------------------------------


def _find_fact2_start(tokens: List[str]) -> Optional[int]:
    """Locate the token index where 'Fact 2' starts.

    We look for the substring "Fact" then "2" appearing in adjacent tokens.
    Falls back to splitting on the first '.' if the marker isn't found.
    """
    joined = ""
    offsets: List[int] = []
    for tok in tokens:
        offsets.append(len(joined))
        joined += tok
    needle = "Fact 2"
    idx = joined.find(needle)
    if idx >= 0:
        # The first token whose own start position lies at or after `idx` is the
        # one that begins the "Fact 2" run.  Strip any leading whitespace from
        # the token so that " Fact" with offset N-1 still maps onto needle at N.
        for i, off in enumerate(offsets):
            tok_start = off + (len(tokens[i]) - len(tokens[i].lstrip()))
            if tok_start >= idx:
                return i
    # Fallback: split on the first sentence boundary.
    for i, tok in enumerate(tokens):
        if "." in tok and i + 1 < len(tokens):
            return i + 1
    return None
