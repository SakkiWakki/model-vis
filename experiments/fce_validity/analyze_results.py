"""Analyze a perplexity_pairs.csv produced by ``run_experiment.py``.

Reads the CSV alone — no model load, no dataset re-iteration — and produces:

  * results/summary.txt        — plain-text headline statistics
  * results/fig_scatter.png    — log PP corrected vs. original
  * results/fig_loghist.png    — histogram of log perplexity ratios
  * results/fig_by_error.png   — median ratio per error type, sorted

Statistics
----------
  * Wilcoxon signed-rank (one-sided, alternative=greater) on log-PP.
  * Median perplexity ratio with 95% bootstrap CI (10k resamples, seed 42).
  * Proportion of pairs where correction reduced perplexity, binomial
    test against 50%.
  * Spearman correlation between length difference and log-ratio
    (a length-confound diagnostic).
  * Stratified analysis: same three headline stats per error type code
    that appears in at least 100 pairs, sorted by median ratio descending.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # headless-safe; we save PNGs, no GUI window.
import matplotlib.pyplot as plt
from scipy.stats import binomtest, spearmanr, wilcoxon


_REPO_ROOT = Path(__file__).resolve().parents[2]
SEED = 42
BOOTSTRAP_N = 10_000
MIN_PAIRS_FOR_STRATUM = 100


# ----------------------------------------------------------------------
# Stats helpers
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class HeadlineStats:
    n: int
    median_ratio: float
    ci_low: float
    ci_high: float
    proportion_reduced: float
    binom_p: float
    wilcoxon_W: float
    wilcoxon_p: float

    def as_text(self, label: str = "All pairs") -> str:
        return (
            f"{label}: n={self.n}, "
            f"median ratio={self.median_ratio:.3f} (95% CI {self.ci_low:.3f}-{self.ci_high:.3f}), "
            f"prop reduced={self.proportion_reduced * 100:.1f}% "
            f"(binomial p={self.binom_p:.2e}), "
            f"Wilcoxon W={self.wilcoxon_W:.1f} p={self.wilcoxon_p:.2e}"
        )


def _bootstrap_median_ci(
    ratios: np.ndarray, *, n_boot: int = BOOTSTRAP_N, seed: int = SEED
) -> Tuple[float, float, float]:
    """Return ``(median, lower_ci, upper_ci)`` for a 95% percentile bootstrap."""
    if ratios.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    # Resample indices once in a (n_boot, n) block — faster than a Python loop.
    n = ratios.size
    idx = rng.integers(0, n, size=(n_boot, n))
    medians = np.median(ratios[idx], axis=1)
    lo, hi = np.percentile(medians, [2.5, 97.5])
    return float(np.median(ratios)), float(lo), float(hi)


def _headline_stats(df: pd.DataFrame) -> HeadlineStats:
    """Compute the three headline stats for a (sub-)set of pair rows."""
    ratios = (df["pp_original"] / df["pp_corrected"]).to_numpy()
    median, lo, hi = _bootstrap_median_ci(ratios)

    reduced = int((df["pp_original"] > df["pp_corrected"]).sum())
    n = len(df)
    if n > 0:
        binom = binomtest(reduced, n, p=0.5, alternative="greater")
        binom_p = float(binom.pvalue)
        prop = reduced / n
    else:
        binom_p = float("nan")
        prop = float("nan")

    # Wilcoxon: only defined when there's at least one nonzero difference.
    diffs = df["log_pp_original"].to_numpy() - df["log_pp_corrected"].to_numpy()
    if n >= 1 and np.any(diffs != 0):
        wresult = wilcoxon(
            df["log_pp_original"].to_numpy(),
            df["log_pp_corrected"].to_numpy(),
            alternative="greater",
            zero_method="wilcox",
        )
        W = float(wresult.statistic)
        Wp = float(wresult.pvalue)
    else:
        W = float("nan")
        Wp = float("nan")

    return HeadlineStats(
        n=n,
        median_ratio=median,
        ci_low=lo,
        ci_high=hi,
        proportion_reduced=prop,
        binom_p=binom_p,
        wilcoxon_W=W,
        wilcoxon_p=Wp,
    )


def _explode_error_types(df: pd.DataFrame) -> pd.DataFrame:
    """Long-form (one row per (pair_id, error_type)) for stratified analysis.

    Rows whose ``error_types`` field is empty contribute no entries.  A pair
    with three error types contributes three rows (the same pair counted in
    three strata).
    """
    series = df["error_types"].fillna("")
    long_rows: List[dict] = []
    for idx, raw in series.items():
        if not raw:
            continue
        for code in (c.strip() for c in str(raw).split(",")):
            if code:
                long_rows.append({"row_idx": idx, "error_type": code})
    return pd.DataFrame(long_rows)


def _stratified_by_error(
    df: pd.DataFrame, *, min_pairs: int = MIN_PAIRS_FOR_STRATUM
) -> pd.DataFrame:
    """Compute headline stats per error type code, filtered to dense strata."""
    long_df = _explode_error_types(df)
    if long_df.empty:
        return pd.DataFrame(
            columns=[
                "error_type", "n",
                "median_ratio", "ci_low", "ci_high",
                "proportion_reduced", "binom_p",
                "wilcoxon_W", "wilcoxon_p",
            ]
        )
    rows: List[dict] = []
    counts = long_df["error_type"].value_counts()
    for code, n in counts.items():
        if n < min_pairs:
            continue
        sub_idx = long_df.loc[long_df["error_type"] == code, "row_idx"].unique()
        sub = df.loc[sub_idx]
        s = _headline_stats(sub)
        rows.append({
            "error_type": code,
            "n": s.n,
            "median_ratio": s.median_ratio,
            "ci_low": s.ci_low,
            "ci_high": s.ci_high,
            "proportion_reduced": s.proportion_reduced,
            "binom_p": s.binom_p,
            "wilcoxon_W": s.wilcoxon_W,
            "wilcoxon_p": s.wilcoxon_p,
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("median_ratio", ascending=False, kind="mergesort").reset_index(drop=True)
    return out


# ----------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------


def _fig_scatter(df: pd.DataFrame, out_path: Path) -> None:
    x = df["log_pp_corrected"].to_numpy()
    y = df["log_pp_original"].to_numpy()
    prop_above = float((y > x).mean()) if len(y) else float("nan")
    lim_lo = float(min(np.min(x), np.min(y))) if len(x) else 0.0
    lim_hi = float(max(np.max(x), np.max(y))) if len(x) else 1.0

    fig, ax = plt.subplots(figsize=(6.0, 6.0))
    ax.scatter(x, y, alpha=0.3, s=12, color="#3a6ea8", edgecolors="none")
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], color="black", linewidth=1, linestyle="--")
    ax.set_xlim(lim_lo, lim_hi)
    ax.set_ylim(lim_lo, lim_hi)
    ax.set_xlabel("log perplexity (corrected)", fontsize=13)
    ax.set_ylabel("log perplexity (original)", fontsize=13)
    ax.set_title(
        f"Per-pair log perplexity\n"
        f"({prop_above * 100:.1f}% of pairs above diagonal; n={len(df)})",
        fontsize=13,
    )
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _fig_loghist(df: pd.DataFrame, out_path: Path) -> None:
    r = df["log_ratio"].to_numpy()
    if len(r) == 0:
        # Empty placeholder so downstream code doesn't blow up.
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.set_title("Log perplexity ratio (no data)")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.hist(r, bins=60, color="#3a6ea8", edgecolor="white", alpha=0.85)
    ax.axvline(0.0, color="black", linewidth=1, linestyle="--")
    ax.set_xlabel("log(original perplexity / corrected perplexity)", fontsize=13)
    ax.set_ylabel("count", fontsize=13)
    ax.set_title(
        f"Distribution of log perplexity ratios (n={len(df)})\n"
        f"Right-skew = corrections reduce perplexity on average",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _fig_by_error(stratified: pd.DataFrame, out_path: Path) -> None:
    if stratified.empty:
        # Empty placeholder image, but still a valid PNG.
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.set_title("No error-type strata with sufficient N")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return
    codes = stratified["error_type"].tolist()
    medians = stratified["median_ratio"].to_numpy()
    lows = stratified["ci_low"].to_numpy()
    highs = stratified["ci_high"].to_numpy()
    err_lo = medians - lows
    err_hi = highs - medians

    # One panel; horizontal bar chart so labels are readable for many codes.
    fig_h = max(3.2, 0.32 * len(codes) + 1.5)
    fig, ax = plt.subplots(figsize=(8.0, fig_h))
    y = np.arange(len(codes))
    ax.barh(
        y, medians,
        xerr=np.vstack([err_lo, err_hi]),
        color="#3a6ea8", alpha=0.85,
        ecolor="black", capsize=2,
    )
    ax.axvline(1.0, color="black", linewidth=1, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels(codes)
    ax.invert_yaxis()
    ax.set_xlabel("median perplexity ratio (original / corrected)", fontsize=13)
    ax.set_title(
        f"Per-error-type median ratio (n ≥ {MIN_PAIRS_FOR_STRATUM} pairs)\n"
        "95% bootstrap CI; values > 1 mean correction reduced PP",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------
# Summary file
# ----------------------------------------------------------------------


def _format_summary(
    *,
    n_pairs: int,
    metadata: dict,
    headline: HeadlineStats,
    spearman_rho: float,
    spearman_p: float,
    stratified: pd.DataFrame,
    min_corrections: int = 0,
    n_dropped_zero_corrections: int = 0,
) -> str:
    lines: List[str] = []
    lines.append("FCE Perplexity Validity Experiment")
    lines.append("=" * 34)
    lines.append("")
    if min_corrections > 0:
        lines.append(
            f"Subset: pairs with >= {min_corrections} correction(s) "
            f"(filtered out {n_dropped_zero_corrections} no-correction pairs)"
        )
    else:
        lines.append("Subset: all pairs (including no-correction pairs)")
    lines.append(f"N pairs analyzed: {n_pairs}")
    model = metadata.get("model", "?")
    lines.append(f"Model: {model}")
    lines.append(f"Tokenizer: {metadata.get('tokenizer', model)}")
    lines.append(f"Run date: {metadata.get('timestamp', '?')}")
    lines.append(f"Analysis date: {datetime.now(timezone.utc).isoformat()}")
    if metadata.get("git_commit"):
        lines.append(f"Git commit: {metadata['git_commit']}")
    lines.append("")
    lines.append("Headline statistics:")
    lines.append(
        f"- Median perplexity ratio (original / corrected): {headline.median_ratio:.3f}"
    )
    lines.append(
        f"  (95% bootstrap CI: {headline.ci_low:.3f} - {headline.ci_high:.3f}; "
        f"{BOOTSTRAP_N} resamples, seed {SEED})"
    )
    lines.append(
        f"- Proportion of pairs where correction reduces PP: "
        f"{headline.proportion_reduced * 100:.1f}% "
        f"(binomial test against 50%: p = {headline.binom_p:.2e})"
    )
    lines.append(
        f"- Wilcoxon signed-rank test on log PP (one-sided, original > corrected): "
        f"W = {headline.wilcoxon_W:.1f}, p = {headline.wilcoxon_p:.2e}"
    )
    lines.append("")
    lines.append("Length confound check:")
    lines.append(
        f"- Spearman correlation between (tokens_original - tokens_corrected) "
        f"and log_ratio: r = {spearman_rho:.3f}, p = {spearman_p:.2e}"
    )
    lines.append("")
    lines.append(f"Stratified by error type (n >= {MIN_PAIRS_FOR_STRATUM}):")
    if stratified.empty:
        lines.append("  (no error-type strata meet the minimum-N threshold)")
    else:
        # Pretty-print as a fixed-width table.
        header = (
            f"  {'code':<8} {'n':>5} "
            f"{'median':>9} {'ci_low':>9} {'ci_high':>9} "
            f"{'reduced%':>9} {'binom_p':>11} {'wilcox_p':>11}"
        )
        lines.append(header)
        lines.append("  " + "-" * (len(header) - 2))
        for _, row in stratified.iterrows():
            lines.append(
                f"  {row['error_type']:<8} {int(row['n']):>5} "
                f"{row['median_ratio']:>9.3f} {row['ci_low']:>9.3f} {row['ci_high']:>9.3f} "
                f"{row['proportion_reduced'] * 100:>8.1f}% "
                f"{row['binom_p']:>11.2e} {row['wilcoxon_p']:>11.2e}"
            )
    lines.append("")
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------
# Entrypoint
# ----------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Statistics + figures from a perplexity_pairs.csv."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=_REPO_ROOT / "experiments" / "fce_validity" / "results" / "perplexity_pairs.csv",
        help="Path to the CSV produced by run_experiment.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write summary.txt + PNGs (default: alongside the CSV).",
    )
    parser.add_argument(
        "--min-corrections",
        type=int,
        default=0,
        help=(
            "Restrict the analysis to pairs whose n_corrections is >= this value. "
            "Set to 1 to drop no-correction pairs (where original == corrected and "
            "ratio is trivially 1.0, biasing the median downward)."
        ),
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="",
        help=(
            "Optional filename suffix for outputs (e.g. '_corrected_only').  "
            "Without it, repeated runs overwrite the same summary.txt / PNGs."
        ),
    )
    args = parser.parse_args(argv)

    if not args.input.is_file():
        print(f"[analyze_results] no such file: {args.input}", file=sys.stderr)
        return 1
    out_dir = args.output_dir or args.input.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)
    print(f"[analyze_results] loaded {len(df)} rows from {args.input}")

    # Drop rows with non-finite values defensively (the runner already filters,
    # but a manually-edited CSV could slip them through).
    needed = ["pp_original", "pp_corrected", "log_pp_original", "log_pp_corrected", "log_ratio"]
    before = len(df)
    df = df.dropna(subset=needed)
    df = df[(df["pp_original"] > 0) & (df["pp_corrected"] > 0)]
    if len(df) != before:
        print(f"[analyze_results] dropped {before - len(df)} rows with invalid PP values")

    # Optional filter: --min-corrections drops no-correction pairs (where the
    # learner-original and gold-corrected texts are identical, so ratio = 1).
    # Including them biases the median ratio toward 1; for the central claim
    # we usually want --min-corrections 1.
    n_dropped_zero = 0
    if args.min_corrections > 0:
        before_filter = len(df)
        df = df[df["n_corrections"].fillna(0).astype(int) >= args.min_corrections]
        n_dropped_zero = before_filter - len(df)
        print(
            f"[analyze_results] --min-corrections={args.min_corrections}: "
            f"kept {len(df)} pairs, dropped {n_dropped_zero}"
        )

    # Reproducibility metadata, if present.
    meta_path = args.input.parent / "run_metadata.json"
    metadata: dict = {}
    if meta_path.is_file():
        try:
            metadata = json.loads(meta_path.read_text())
        except Exception:
            metadata = {}

    # Headline.
    headline = _headline_stats(df)
    print("[analyze_results] " + headline.as_text("Headline"))

    # Length-confound: Spearman on (len_o - len_c) vs log_ratio.
    if len(df) >= 3:
        len_diff = (df["tokens_original"].astype(int) - df["tokens_corrected"].astype(int)).to_numpy()
        log_ratio = df["log_ratio"].to_numpy()
        sp = spearmanr(len_diff, log_ratio)
        rho = float(sp.statistic if hasattr(sp, "statistic") else sp.correlation)
        p = float(sp.pvalue)
    else:
        rho = float("nan")
        p = float("nan")
    print(f"[analyze_results] length-confound: Spearman r={rho:.3f} p={p:.2e}")

    # Stratified by error type.
    stratified = _stratified_by_error(df)
    print(f"[analyze_results] {len(stratified)} error-type strata with n >= {MIN_PAIRS_FOR_STRATUM}")

    # Figures.
    suffix = args.suffix
    _fig_scatter(df, out_dir / f"fig_scatter{suffix}.png")
    _fig_loghist(df, out_dir / f"fig_loghist{suffix}.png")
    _fig_by_error(stratified, out_dir / f"fig_by_error{suffix}.png")
    print(f"[analyze_results] wrote figures to {out_dir}")

    # Save the stratified table as a CSV too so downstream tools / papers
    # don't have to scrape the summary text.
    strat_csv = out_dir / f"stratified_by_error{suffix}.csv"
    stratified.to_csv(strat_csv, index=False)
    print(f"[analyze_results] wrote {strat_csv}")

    # Summary.
    summary = _format_summary(
        n_pairs=len(df),
        metadata=metadata,
        headline=headline,
        spearman_rho=rho,
        spearman_p=p,
        stratified=stratified,
        min_corrections=args.min_corrections,
        n_dropped_zero_corrections=n_dropped_zero,
    )
    summary_path = out_dir / f"summary{suffix}.txt"
    summary_path.write_text(summary, encoding="utf-8")
    print(f"[analyze_results] wrote {summary_path}")
    print()
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
