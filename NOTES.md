# model-vis notes

## Optional datasets

### Cambridge FCE (CLC-FCE Public Dataset)

Required for the perplexity-validity experiment that compares per-sentence
perplexity for learner-original vs. gold-corrected paragraphs from the
Cambridge FCE corpus.  Not committed to this repo — the corpus is licensed
through Cambridge ESOL.

Setup:

1. Obtain `fce-released-dataset.zip` from Cambridge.  Drop it in `~/Downloads`
   (or set `FCE_ARCHIVE=/path/to/fce-released-dataset.zip`).
2. Run:

   ```bash
   ./scripts/setup_fce.sh
   ```

   This extracts the archive to `~/datasets/fce-released-dataset` (override
   with `FCE_PATH=/your/path`) and symlinks it under `data/`.  The symlink
   and any files inside it are gitignored.
3. The dataset shows up in the sidebar's Data dropdown as `fce` and exposes
   `FCEDataset.iter_pairs() -> Iterator[FCEPair]` for offline experiment
   scripts.  `FCEPair` carries `original`, `corrected`, `error_types`, and
   the source XML path.

If the dataset isn't installed, `main.py` prints a warning and continues
with just the XOR dataset registered.
