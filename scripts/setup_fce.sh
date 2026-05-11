#!/usr/bin/env bash
# Set up the Cambridge FCE Public Dataset for use by the model-vis framework.
#
# Behavior:
#   - Resolves the dataset location to "$FCE_PATH" if set, else
#     "$HOME/datasets/fce-released-dataset".
#   - If that directory doesn't exist, looks for a downloaded archive
#     (fce-released-dataset.zip) in "$FCE_ARCHIVE" or in ~/Downloads,
#     and extracts it.  Cambridge does not redistribute the archive freely
#     — the user must obtain it from CLC and place it somewhere reachable.
#   - Creates a symlink at "<repo>/data/fce-released-dataset" pointing at
#     the resolved directory.
#   - Idempotent: re-running with everything in place is a no-op.
#
# The dataset is NOT committed to this repo (gitignored under data/).
set -euo pipefail

# Resolve the repo root from this script's location.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

FCE_PATH="${FCE_PATH:-$HOME/datasets/fce-released-dataset}"
SYMLINK="$REPO_ROOT/data/fce-released-dataset"

mkdir -p "$(dirname "$FCE_PATH")"
mkdir -p "$REPO_ROOT/data"

# Locate the archive if the target directory doesn't exist yet.
if [ ! -d "$FCE_PATH" ]; then
    ARCHIVE="${FCE_ARCHIVE:-}"
    if [ -z "$ARCHIVE" ]; then
        for candidate in \
            "$HOME/Downloads/fce-released-dataset.zip" \
            "$HOME/Downloads/FCE-released-dataset.zip" \
            "$HOME/Downloads/fce.zip"; do
            if [ -f "$candidate" ]; then
                ARCHIVE="$candidate"
                break
            fi
        done
    fi

    if [ -z "$ARCHIVE" ] || [ ! -f "$ARCHIVE" ]; then
        echo "error: could not find the FCE archive." >&2
        echo "  Looked in:" >&2
        echo "    \$FCE_ARCHIVE (=${FCE_ARCHIVE:-<unset>})" >&2
        echo "    ~/Downloads/fce-released-dataset.zip" >&2
        echo "" >&2
        echo "  Obtain fce-released-dataset.zip from Cambridge Learner Corpus" >&2
        echo "  and either place it in ~/Downloads or set FCE_ARCHIVE=/path/to/it" >&2
        exit 1
    fi

    echo "Extracting $ARCHIVE -> $FCE_PATH ..."
    TMPDIR="$(mktemp -d)"
    trap 'rm -rf "$TMPDIR"' EXIT
    unzip -q "$ARCHIVE" -d "$TMPDIR"

    # The archive root is "fce-released-dataset/"; move it into place.
    if [ -d "$TMPDIR/fce-released-dataset" ]; then
        mv "$TMPDIR/fce-released-dataset" "$FCE_PATH"
    else
        # Some custom archives may already have stripped the top dir.
        mv "$TMPDIR" "$FCE_PATH"
        trap - EXIT
    fi
fi

# Create or refresh the symlink.
if [ -L "$SYMLINK" ]; then
    EXISTING="$(readlink -f "$SYMLINK" || true)"
    RESOLVED="$(readlink -f "$FCE_PATH")"
    if [ "$EXISTING" != "$RESOLVED" ]; then
        echo "Refreshing symlink (was -> $EXISTING)..."
        rm "$SYMLINK"
        ln -s "$FCE_PATH" "$SYMLINK"
    fi
elif [ -e "$SYMLINK" ]; then
    echo "error: $SYMLINK exists and is not a symlink; refusing to overwrite." >&2
    exit 1
else
    ln -s "$FCE_PATH" "$SYMLINK"
fi

# Verify the symlink resolves to a directory containing the dataset.
if [ ! -d "$SYMLINK/dataset" ]; then
    echo "error: $SYMLINK -> $(readlink -f "$SYMLINK") does not contain a 'dataset' subdir." >&2
    exit 1
fi

# Friendly summary.
N_FILES="$(find -L "$SYMLINK/dataset" -name '*.xml' -type f 2>/dev/null | wc -l)"
echo
echo "FCE dataset ready."
echo "  Symlink:  $SYMLINK"
echo "  Resolves: $(readlink -f "$SYMLINK")"
echo "  XML docs: $N_FILES"
