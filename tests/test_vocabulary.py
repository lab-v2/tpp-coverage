"""Banned words stay out of every tracked file (file names included)."""

from __future__ import annotations

import re
import subprocess

from conftest import REPO

BANNED = ("triptych",)  # say "three-panel figure"


def _tracked() -> list[str]:
    out = subprocess.run(["git", "ls-files"], cwd=REPO, capture_output=True, text=True, check=True)
    return [f for f in out.stdout.splitlines() if f != "uv.lock" and f != "tests/test_vocabulary.py"]


def test_no_banned_word_in_names_or_text():
    pattern = re.compile("|".join(BANNED), re.I)
    hits = []
    for name in _tracked():
        if pattern.search(name):
            hits.append(name)
            continue
        try:
            text = (REPO / name).read_text(errors="strict")
        except (UnicodeDecodeError, FileNotFoundError):
            continue
        hits += [f"{name}:{i}" for i, line in enumerate(text.splitlines(), 1) if pattern.search(line)]
    assert not hits, hits
