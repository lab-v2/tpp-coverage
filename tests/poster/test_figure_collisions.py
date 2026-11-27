"""Gate for `figure_collisions.py`: it must flag planted collisions and pass a clean page."""

from __future__ import annotations

import sys
from pathlib import Path

import pymupdf

sys.path.insert(0, str(Path(__file__).parent))
from figure_collisions import problems  # noqa: E402


def _pdf(tmp_path: Path, name: str, draw) -> str:
    doc = pymupdf.open()
    page = doc.new_page(width=300, height=200)
    draw(page)
    path = tmp_path / name
    doc.save(path)
    return str(path)


def test_clean_page_passes(tmp_path: Path) -> None:
    def draw(page):
        page.insert_text((20, 50), "Loss", fontsize=12)
        page.insert_text((20, 120), "TPP", fontsize=12)
        page.draw_line((150, 10), (150, 190))

    assert problems(_pdf(tmp_path, "clean.pdf", draw)) == []


def test_touching_labels_fail(tmp_path: Path) -> None:
    def draw(page):
        page.insert_text((20, 50), "17.0", fontsize=12)
        page.insert_text((46, 54), "C", fontsize=12, rotate=90)

    assert any("touches" in p for p in problems(_pdf(tmp_path, "touch.pdf", draw)))


def test_line_through_label_fails(tmp_path: Path) -> None:
    def crossed(page):
        page.insert_text((20, 50), "16.5", fontsize=12)
        page.draw_line((10, 45), (90, 45))

    assert any("crossed" in p for p in problems(_pdf(tmp_path, "cross.pdf", crossed)))
