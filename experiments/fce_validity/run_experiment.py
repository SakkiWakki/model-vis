"""Score every FCE (original, corrected) paragraph pair with GPT-2 perplexity.

Produces a single CSV that is the canonical artifact of the experiment.  All
subsequent analysis (``analyze_results.py``) reads from this CSV alone, so
re-running statistics or remaking plots does not require re-running the slow
perplexity computation.

Perplexity definition
---------------------
The math mirrors ``model_viz/viz/visualizers/perplexity_viz/viz.py`` exactly:
for a sequence of token ids ``t_0, t_1, ..., t_{T-1}`` and per-position logits
``L[i] = model(t)[i]`` of shape ``(V,)``::

    log_probs[i] = log_softmax(L[i])
    nll_i       = -log_probs[i-1, t_i]   for i in 1..T-1
    perplexity  = exp(mean(nll_i))

Position 0 has no preceding context and is excluded from the average.  This
is the geometric-mean form of sequence perplexity (i.e. ``exp`` of the mean
NLL across scored positions).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, List, Optional, Set, Tuple

import torch
from tqdm import tqdm

# Make the repo root importable when this script is run as a file.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from model_viz.data.fce import FCEDataset, FCEPair  # noqa: E402


CSV_COLUMNS: Tuple[str, ...] = (
    "pair_id",
    "original_text",
    "corrected_text",
    "pp_original",
    "pp_corrected",
    "log_pp_original",
    "log_pp_corrected",
    "log_ratio",
    "tokens_original",
    "tokens_corrected",
    "n_corrections",
    "error_types",
)


# ----------------------------------------------------------------------
# Perplexity
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class Scored:
    """Result of scoring one text under the LM."""

    perplexity: float        # exp(mean NLL); inf if no valid scoring positions
    n_tokens: int            # total tokens fed to the model
    n_scored: int            # positions with a valid (i>=1, in-vocab) NLL


@torch.no_grad()
def score_text(text: str, *, model, tokenizer, device: torch.device) -> Optional[Scored]:
    """Compute geometric-mean perplexity of ``text`` under ``model``.

    Returns ``None`` if the text cannot be scored (empty, single-token, or a
    tokenization failure).  Matches the ``_compute`` math in the visualizer.
    """
    if not text or not text.strip():
        return None
    try:
        enc = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    except Exception:
        return None
    input_ids = enc["input_ids"]
    if input_ids.numel() < 2:
        # Need at least two tokens — position 0 is unscored.
        return None
    input_ids = input_ids.to(device)
    out = model(input_ids=input_ids)
    logits = out.logits if hasattr(out, "logits") else out[0]  # (1, T, V)

    L = logits[0].float()                          # (T, V)
    log_probs = torch.log_softmax(L, dim=-1)       # (T, V)
    ids = input_ids[0].to(torch.long)              # (T,)
    T_len, V = L.shape

    # For each position i >= 1, predict ids[i] from log_probs[i-1].
    # Gather along the vocab dim with the target ids shifted by one.
    targets = ids[1:]                               # (T-1,)
    preds = log_probs[:-1]                          # (T-1, V)
    valid = (targets >= 0) & (targets < V)
    if not valid.any():
        return Scored(perplexity=float("inf"), n_tokens=int(T_len), n_scored=0)
    safe_targets = torch.where(valid, targets, torch.zeros_like(targets))
    chosen = preds.gather(dim=-1, index=safe_targets.unsqueeze(-1)).squeeze(-1)  # (T-1,)
    nll = -chosen
    nll = nll[valid]
    if nll.numel() == 0:
        return Scored(perplexity=float("inf"), n_tokens=int(T_len), n_scored=0)
    mean_nll = float(nll.mean().item())
    return Scored(
        perplexity=math.exp(mean_nll),
        n_tokens=int(T_len),
        n_scored=int(nll.numel()),
    )


# ----------------------------------------------------------------------
# Resumable CSV streaming
# ----------------------------------------------------------------------


def _already_processed_ids(csv_path: Path) -> Set[str]:
    """Return the set of ``pair_id`` values already present in an existing CSV.

    If the file doesn't exist, returns an empty set.  Used by ``--resume``.
    """
    if not csv_path.is_file():
        return set()
    ids: Set[str] = set()
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row.get("pair_id")
            if pid:
                ids.add(pid)
    return ids


def _open_csv(
    csv_path: Path, append: bool
) -> Tuple[csv.DictWriter, "object"]:
    """Open the CSV for writing, returning ``(writer, file_handle)``.

    Writes the header if the file is new or being recreated.
    """
    mode = "a" if append else "w"
    new_file = (mode == "w") or not csv_path.exists() or csv_path.stat().st_size == 0
    fh = csv_path.open(mode, newline="", encoding="utf-8")
    writer = csv.DictWriter(fh, fieldnames=list(CSV_COLUMNS))
    if new_file:
        writer.writeheader()
    return writer, fh


# ----------------------------------------------------------------------
# Dataset iteration with stable pair ids
# ----------------------------------------------------------------------


def _iter_indexed_pairs(dataset: FCEDataset) -> Iterator[Tuple[str, FCEPair]]:
    """Yield ``(pair_id, pair)``.

    ``pair_id`` is ``"<source_file>#p<paragraph_index>"`` where
    ``paragraph_index`` counts from 0 within the source file, in document
    order.  Stable across runs as long as the corpus on disk is unchanged.
    """
    last_file: Optional[str] = None
    idx_within_file = -1
    for pair in dataset.iter_pairs():
        if pair.source_file != last_file:
            last_file = pair.source_file
            idx_within_file = 0
        else:
            idx_within_file += 1
        yield (f"{pair.source_file}#p{idx_within_file}", pair)


# ----------------------------------------------------------------------
# Top-level runner
# ----------------------------------------------------------------------


def _resolve_device(arg: str) -> torch.device:
    if arg == "cpu":
        return torch.device("cpu")
    if arg == "cuda":
        return torch.device("cuda")
    # auto
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _git_commit_hash() -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(_REPO_ROOT),
            capture_output=True,
            text=True,
            check=False,
            timeout=2,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Score (original, corrected) FCE pairs with GPT-2 perplexity."
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=None,
        help="Path to the FCE dataset root (default: repo's data/fce-released-dataset symlink).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt2",
        help="HF model name or local path (default: gpt2).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_REPO_ROOT / "experiments" / "fce_validity" / "results" / "perplexity_pairs.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N pairs (for smoke testing).",
    )
    parser.add_argument(
        "--device",
        choices=("cuda", "cpu", "auto"),
        default="auto",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip pair_ids already present in the output CSV.",
    )
    parser.add_argument(
        "--flush-every",
        type=int,
        default=50,
        help="Flush the CSV to disk every N rows so partial runs are recoverable.",
    )
    args = parser.parse_args()

    device = _resolve_device(args.device)
    print(f"[run_experiment] device={device}")

    # ---- Load model + tokenizer once. ----
    from transformers import AutoModelForCausalLM, AutoTokenizer  # heavy import

    print(f"[run_experiment] loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, attn_implementation="eager"
    )
    model.to(device)
    model.eval()

    # ---- Open the dataset. ----
    dataset = FCEDataset(root=args.dataset_path) if args.dataset_path else FCEDataset()
    print(f"[run_experiment] FCE dataset root: {dataset._root}")

    skip_ids = _already_processed_ids(args.output) if args.resume else set()
    if skip_ids:
        print(f"[run_experiment] --resume: skipping {len(skip_ids)} already-scored pair ids")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer, fh = _open_csv(args.output, append=args.resume)

    # ---- Iterate + score. ----
    processed = 0
    skipped_empty = 0
    skipped_tokenize = 0
    started = time.time()
    try:
        bar = tqdm(_iter_indexed_pairs(dataset), unit="pair")
        for pair_id, pair in bar:
            if args.limit is not None and processed >= args.limit:
                break
            if pair_id in skip_ids:
                continue
            if not pair.original.strip() or not pair.corrected.strip():
                skipped_empty += 1
                continue

            scored_o = score_text(pair.original, model=model, tokenizer=tokenizer, device=device)
            scored_c = score_text(pair.corrected, model=model, tokenizer=tokenizer, device=device)
            if scored_o is None or scored_c is None:
                skipped_tokenize += 1
                continue
            if not (math.isfinite(scored_o.perplexity) and math.isfinite(scored_c.perplexity)):
                skipped_tokenize += 1
                continue

            log_o = math.log(scored_o.perplexity)
            log_c = math.log(scored_c.perplexity)

            row = {
                "pair_id": pair_id,
                "original_text": pair.original,
                "corrected_text": pair.corrected,
                "pp_original": f"{scored_o.perplexity:.6f}",
                "pp_corrected": f"{scored_c.perplexity:.6f}",
                "log_pp_original": f"{log_o:.6f}",
                "log_pp_corrected": f"{log_c:.6f}",
                "log_ratio": f"{(log_o - log_c):.6f}",
                "tokens_original": scored_o.n_tokens,
                "tokens_corrected": scored_c.n_tokens,
                "n_corrections": len(pair.error_types),
                "error_types": ",".join(pair.error_types),
            }
            writer.writerow(row)
            processed += 1
            if processed % args.flush_every == 0:
                fh.flush()

            bar.set_postfix(
                processed=processed,
                skipped_empty=skipped_empty,
                skipped_tok=skipped_tokenize,
            )
    finally:
        fh.flush()
        fh.close()

    elapsed = time.time() - started
    print(
        f"[run_experiment] done: processed={processed} "
        f"skipped_empty={skipped_empty} skipped_tokenize={skipped_tokenize} "
        f"elapsed={elapsed:.1f}s"
    )

    # ---- Sidecar reproducibility metadata. ----
    metadata = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "tokenizer": args.model,
        "device": str(device),
        "dataset_root": str(dataset._root),
        "output_csv": str(args.output),
        "pair_count": processed,
        "skipped_empty": skipped_empty,
        "skipped_tokenize": skipped_tokenize,
        "limit": args.limit,
        "resume": bool(args.resume),
        "git_commit": _git_commit_hash(),
        "torch_version": torch.__version__,
        "elapsed_seconds": elapsed,
    }
    metadata_path = args.output.parent / "run_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"[run_experiment] wrote {metadata_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
