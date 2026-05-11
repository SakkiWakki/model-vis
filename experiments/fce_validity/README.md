# FCE perplexity-validity experiment

## Question

Does language-model perplexity track "grammaticality" in L2 English production?

This experiment uses the Cambridge FCE Public Dataset — paragraphs of
learner writing with inline gold-corrected versions of every error — to
test the methodological claim that runs through this codebase: that the
perplexity assigned by a base language model is a useful proxy for how
"surprised" the model is by an ungrammatical construction, and therefore
a useful signal of grammaticality.

Concretely: for each (learner-original, gold-corrected) paragraph pair,
we ask whether the corrected version has lower perplexity under GPT-2.
If the perplexity-as-grammaticality framing has empirical content, the
answer should be yes — robustly, and in a way that varies by error type.

## Method

1. **Dataset.** The CLC-FCE corpus is parsed paragraph-by-paragraph by
   `model_viz.data.fce.FCEDataset`.  Each `<p>` element in the source
   XML yields one `FCEPair` containing the learner-original text, the
   gold-corrected text (with `<NS>` correction spans materialized into
   their `<i>` and `<c>` content respectively), and the list of error
   type codes (`RN`, `AGV`, `S`, …) that appeared inline.
2. **Scoring.** Each paragraph (both versions) is tokenized with the
   GPT-2 BPE tokenizer (no special tokens) and fed to GPT-2 small
   (124M).  Per-position log-probabilities are computed via
   `log_softmax`; per-token NLL is `-log_probs[i-1, target_i]` for
   `i >= 1`; sequence perplexity is `exp(mean(NLL))`.  This is the
   geometric-mean form, matching `_compute` in the existing
   perplexity visualizer exactly so any number that comes out of this
   script is the same number the GUI would display for the same text.
3. **Output.** A single CSV (`results/perplexity_pairs.csv`) plus a
   reproducibility sidecar (`results/run_metadata.json`).  Each row is
   one pair: original/corrected text, perplexity of each, log values,
   log ratio, token counts, number of corrections, error types.
4. **Analysis** (`analyze_results.py`) reads the CSV alone and
   produces:
    - Three headline statistics: median PP ratio (with 95% bootstrap
      CI), proportion of pairs where correction reduced PP (binomial
      test against 50%), and a one-sided Wilcoxon signed-rank test
      on the log perplexities.
    - A length-confound check: Spearman correlation between
      `tokens_original - tokens_corrected` and `log_ratio`.  Large
      magnitudes would mean the headline result is partially explained
      by length changes between original and corrected.
    - A stratified table: the same three headline statistics restricted
      to pairs containing each error type code (filtered to codes
      occurring in ≥ 100 pairs).  Sorted by median ratio descending.
    - Three publication-grade figures: a per-pair scatter, a log-ratio
      histogram, and a per-error-type bar chart with CIs.
    - A plain-text `summary.txt` suitable for direct quotation.

## Results

*To be filled in after the experiment runs.*

Expected pattern (the methodological prediction): the median ratio is
greater than 1, the proportion is comfortably above 50%, both
statistical tests are significant at conventional thresholds, and
morphological / syntactic error categories (`AGV` subject-verb
agreement, `TV` verb tense, etc.) show larger ratios than purely
lexical-substitution categories (`RN` noun replacement, `RV` verb
replacement, etc.).  If the data contradicts that prediction, that is
itself the publishable finding — see the "interpretation guardrails"
section.

## Reproducibility

Install the dataset once:

```bash
./scripts/setup_fce.sh
```

Run the full experiment + analysis:

```bash
python experiments/fce_validity/run_experiment.py
python experiments/fce_validity/analyze_results.py
```

A smoke run on a CPU finishes in roughly a minute:

```bash
python experiments/fce_validity/run_experiment.py --limit 50
python experiments/fce_validity/analyze_results.py
```

For the full corpus (~5–8k paragraphs after empty/single-token
filtering) on an RTX 5090, expect a few minutes for the forward passes
and another second or two for the statistics.  On CPU, expect 30+
minutes.

### Determinism

- Random seed `42` is fixed for the bootstrap.
- GPT-2 is loaded in `eval()` mode with `attn_implementation="eager"`
  (matches what the GUI uses); inference is `@torch.no_grad`.
- The CSV is the canonical artifact.  `analyze_results.py` reads only
  that CSV, so re-running statistics costs near-zero time.
- `results/run_metadata.json` records timestamp, model name, device,
  torch version, total pair count, and git commit hash.

### Dependencies

Everything is already in the project venv: `torch`, `transformers`,
`scipy`, `numpy`, `pandas`, `matplotlib`, `tqdm`.  No new third-party
deps are introduced by this experiment.

## Interpretation guardrails

- **The length confound matters.**  If `|spearman_r|` is large
  (say > 0.4), the headline ratio is mixing perplexity with length
  effects and a regression-based control would be appropriate.
- **Effect size per error type is the central claim.**  A uniform
  shift across all error categories is consistent with "any
  intervention reduces perplexity"; the methodological argument is
  specifically that grammar-shaped errors should show the effect more
  strongly than lexical substitutions.
- **Don't engineer the analysis toward the predicted outcome.**  If
  the stratified table puts spelling and lexical-replacement errors
  at the top, that's a substantive finding (probably about how
  surprising the *specific learner spelling/word choice* is, rather
  than about grammaticality per se) and should be reported as such.

## Files

- `run_experiment.py` — scoring runner.  Writes the CSV.
- `analyze_results.py` — statistics + figures.  Reads the CSV.
- `results/` — outputs go here.  Gitignored except `.gitkeep`.
- `results/perplexity_pairs.csv` — canonical artifact.
- `results/run_metadata.json` — reproducibility sidecar.
- `results/summary.txt` — quotable headline statistics.
- `results/stratified_by_error.csv` — same table as the bar chart.
- `results/fig_scatter.png`, `fig_loghist.png`, `fig_by_error.png` —
  the three figures.
