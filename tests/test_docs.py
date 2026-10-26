"""Every relative link in the Markdown docs points at an existing file and heading."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from conftest import REPO

DOCS = [p for p in REPO.rglob("*.md")
        if not {".venv", ".git", "work", ".pytest_cache"} & set(p.relative_to(REPO).parts)]
LINK = re.compile(r"\]\(([^)\s]+)\)")


def _slug(heading: str) -> str:
    text = re.sub(r"[^\w\- ]", "", heading.strip().lower())
    return text.replace(" ", "-")


def _anchors(path: Path) -> set[str]:
    return {_slug(m.group(1)) for m in re.finditer(r"^#+\s+(.+)$", path.read_text(), re.M)}


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: str(p.relative_to(REPO)))
def test_relative_links_resolve(doc: Path):
    broken = []
    for target in LINK.findall(doc.read_text()):
        if re.match(r"(https?|mailto):", target):
            continue
        file_part, _, anchor = target.partition("#")
        dest = (doc.parent / file_part).resolve() if file_part else doc
        if not dest.exists():
            broken.append(target)
        elif anchor and dest.suffix == ".md" and anchor not in _anchors(dest):
            broken.append(target)
    assert not broken, broken


def test_docs_are_found():
    names = {p.name for p in DOCS}
    assert {"README.md", "TRAINING.md", "ANALYSIS.md", "DOWNLOADING.md", "CRITERIA.md"} <= names
