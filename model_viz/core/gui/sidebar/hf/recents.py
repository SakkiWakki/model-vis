"""Persistent list of recently-loaded HuggingFace model ids / paths.

Stored as plain JSON under the user's config dir so it survives across
launches without polluting the repo.  The file is tiny (a list of strings),
read on every dialog open and rewritten after a successful load.

Locations
---------
``$XDG_CONFIG_HOME/model-vis/hf_recents.json`` when ``XDG_CONFIG_HOME`` is set,
otherwise ``~/.config/model-vis/hf_recents.json``.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List


_MAX_RECENTS = 12


def _config_dir() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME")
    if base:
        return Path(base) / "model-vis"
    return Path.home() / ".config" / "model-vis"


def _recents_path() -> Path:
    return _config_dir() / "hf_recents.json"


def load_recents() -> List[str]:
    """Return the saved list, or an empty list if the file is missing/malformed."""
    path = _recents_path()
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(data, list):
        return []
    # Defensive: drop anything that isn't a non-empty string.
    return [s for s in data if isinstance(s, str) and s.strip()]


def add_recent(entry: str) -> List[str]:
    """Push ``entry`` to the front of the recents list and persist.

    Existing duplicates are removed so the same id never appears twice.
    Returns the new list (also written to disk).  Failures to write are
    swallowed — losing a recents entry should never break the loader.
    """
    entry = entry.strip()
    if not entry:
        return load_recents()
    items = [s for s in load_recents() if s != entry]
    items.insert(0, entry)
    items = items[:_MAX_RECENTS]
    try:
        path = _recents_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(items, indent=2), encoding="utf-8")
    except OSError:
        pass
    return items
