"""No private hosts, users, paths, cluster defaults or review-process residue in the tree."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from conftest import REPO, tree_files, tree_text

SKIP_FILES = {"uv.lock", Path(__file__).name}

FORBIDDEN = {
    "machine names": r"\b(godel|sanders|gottfried\d*|lcs-pashakar[\w.-]*|pashakar)\b",
    "home or allocation paths": r"(/home/[a-z]|/work/nvme|/data/jkrichel|/scratch/jkrichel)",
    "cluster user names": r"\b(areid\d*|jkricheli?|bhfk|bfui)\b",
    "cluster defaults": r"\b(ghx4|gh137|gh059|symbolic_regression_hello_world)\b",
    "review process": r"(\breviewers?\b|\bktkD\b|\bnK5Y\b|\bPnLF\b|\bQNJP\b|oVM1ErNilo|openreview\.net|rebuttal)",
    "anonymization": r"(anonymous\.4open|4open\.science)",
    "em dash": "\u2014",
}


FILES = tree_files(SKIP_FILES)


def test_tree_is_scanned():
    assert len(FILES) > 80


@pytest.mark.parametrize("label", sorted(FORBIDDEN))
def test_no_forbidden_strings(label):
    pattern = re.compile(FORBIDDEN[label], re.IGNORECASE)
    hits = []
    for path in FILES:
        text = tree_text(path)
        if text is None:
            continue
        for n, line in enumerate(text.splitlines(), 1):
            if pattern.search(line):
                hits.append(f"{path.relative_to(REPO)}:{n}: {line.strip()[:120]}")
    assert not hits, f"{label}:\n" + "\n".join(hits[:50])


def test_pattern_catches_a_planted_leak():
    # The scanner must be able to fail: each pattern matches a known-bad sample.
    samples = {
        "machine names": "ssh Sanders", "home or allocation paths": "/home/someone/x",
        "cluster user names": "user areid1", "cluster defaults": "--partition ghx4",
        "review process": "Reviewer ktkD", "anonymization": "anonymous.4open.science/r/x",
        "em dash": "a \u2014 b",
    }
    for label, sample in samples.items():
        assert re.search(FORBIDDEN[label], sample, re.IGNORECASE), label
