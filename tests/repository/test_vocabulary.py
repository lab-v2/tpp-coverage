"""Banned words stay out of every file of the tree (file names included)."""

from __future__ import annotations

import re
from pathlib import Path

from conftest import REPO, tree_files, tree_text

BANNED = ("triptych",)  # say "three-panel figure"


def test_no_banned_word_in_names_or_text():
    pattern = re.compile("|".join(BANNED), re.I)
    hits = []
    for path in tree_files({"uv.lock", Path(__file__).name}):
        name = str(path.relative_to(REPO))
        if pattern.search(name):
            hits.append(name)
            continue
        text = tree_text(path)
        if text is not None:
            hits += [f"{name}:{i}" for i, line in enumerate(text.splitlines(), 1)
                     if pattern.search(line)]
    assert not hits, hits
