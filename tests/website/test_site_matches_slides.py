"""The project page follows the talk: one section per slide, in the deck's order, citing what it cites.

The owner asked for the website to match the talk slides. The deck (`slides/*.tex`) is the source;
the page may add only the sections a project page needs and a talk does not (summary, abstract,
BibTeX), and every work the page lists must be one the deck cites.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DECK = (REPO / "slides" / "tpp_neurips2026_slides.tex").read_text()
PAGE = (REPO / "website" / "index.html").read_text()
PAGE_ONLY = {"Summary", "Abstract", "BibTeX"}


def frame_titles(tex: str) -> list[str]:
    return re.findall(r"\\begin\{frame\}(?:\[[^]]*\])?\{([^}]*)\}", tex)


def page_sections(html: str) -> list[str]:
    return [t for t in re.findall(r'<h2 class="title is-3">([^<]*)</h2>', html) if t not in PAGE_ONLY]


def test_one_section_per_slide_in_order() -> None:
    assert page_sections(PAGE) == frame_titles(DECK)


def test_page_lists_only_works_the_deck_cites() -> None:
    start = DECK.index(r"\begin{frame}{References}")
    deck_refs = DECK[start : DECK.index(r"\end{frame}", start)]
    n_deck = len(re.findall(r"\\item\[\{\[\d+\]\}\]", deck_refs))
    page_refs = re.findall(r'<li id="ref-([^"]+)">', PAGE)
    assert len(page_refs) == n_deck, (page_refs, n_deck)


def test_every_page_citation_resolves() -> None:
    ids = set(re.findall(r'<li id="ref-([^"]+)">', PAGE))
    assert set(re.findall(r'href="#ref-([^"]+)"', PAGE)) <= ids


def test_page_shows_only_figures_the_deck_shows() -> None:
    deck_figs = {Path(f).stem for f in re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]*)\}", DECK)}
    page_figs = {Path(f).stem for f in re.findall(r'<img src="static/images/([^"]+)"', PAGE)}
    assert page_figs <= deck_figs, page_figs - deck_figs
