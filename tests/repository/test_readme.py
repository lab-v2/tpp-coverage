"""The README's images, code-structure diagram and citation match the repository and the paper."""

from __future__ import annotations

import re

import pytest
import yaml

from conftest import REPO

README = (REPO / "README.md").read_text()
DIAGRAM = (REPO / "images" / "pipeline" / "pipeline.mmd").read_text()

TITLE = "Tokens-per-Parameter Coverage Is Critical for Robust LLM Scaling Law Extrapolation"
AUTHORS = [
    "Joshua Shay Kricheli",
    "Alexander Lawrence Reid",
    "Soumajyoti Sarkar",
    "Venkata Gandikota",
    "Paulo Shakarian",
]
ARXIV_ID = "2605.08541"


def _block(lang: str) -> str:
    blocks = re.findall(rf"^```{lang}\n(.*?)^```", README, re.M | re.S)
    assert len(blocks) == 1, f"expected one ```{lang} block, found {len(blocks)}"
    return blocks[0]


def _bib_field(bib: str, name: str) -> str:
    match = re.search(rf"^\s*{name}\s*=\s*\{{(.*)\}},?\s*$", bib, re.M)
    assert match, f"BibTeX entry has no {name} field"
    return match.group(1)


@pytest.mark.parametrize("src", re.findall(r'(?:src|srcset)="([^"]+)"', README))
def test_local_images_exist_and_are_images(src: str) -> None:
    if src.startswith("http"):
        return
    path = REPO / src
    assert path.is_file(), src
    head = path.read_bytes()[:256]
    assert head.startswith(b"\x89PNG") or b"<svg" in head or head.lstrip().startswith(b"<?xml"), src


def test_logos_are_neurips_the_ai_institute_and_leibniz_lab() -> None:
    """Exactly the three logos every asset carries; the dark-text ones swap for a light copy."""
    logos = re.findall(r'<img alt="[^"]*" src="images/([^"/]+)"', README)  # not the diagram
    assert sorted(logos) == ["leibniz_lab.png", "neurips.svg", "syracuse_ai_institute.svg"], logos
    for stem in ("neurips", "syracuse_ai_institute"):
        assert f'media="(prefers-color-scheme: dark)" srcset="images/{stem}_dark.svg"' in README
        assert (REPO / "images" / f"{stem}_dark.svg").is_file(), stem


def test_diagram_labels_are_plain_words() -> None:
    diagram = DIAGRAM
    assert re.search(r"^flowchart\b", diagram, re.M)
    labels = re.findall(r'\(\[?"([^"]+)"\]?\)|\["([^"]+)"\]', diagram)
    labels = [a or b for a, b in labels]
    assert len(labels) >= 8, labels
    for label in labels:
        assert "_" not in label and "/" not in label.replace("<br/>", ""), label
        assert not re.search(r"[\U0001F300-\U0001FAFF\u2600-\u27BF]", label), label


def test_diagram_edges_only_join_declared_nodes() -> None:
    diagram = DIAGRAM
    declared = set(re.findall(r"\b(\w+)[\[(]\S", diagram))
    used = set(re.findall(r"(\w+)(?:\[[^\]]*\]|\(\[[^\]]*\]\))?\s*-\.?->", diagram))
    used |= set(re.findall(r"(?:-\.?->|==>|\.->)\s*(\w+)", diagram))
    used |= set(re.findall(r"^\s*(\w+)(?:\(\[.*?\]\)|\[\(.*?\)\])?\s*(?:==>|-\.)", diagram, re.M))
    assert len(used) >= 6, used
    assert used <= declared, used - declared


def test_bibtex_matches_the_arxiv_paper() -> None:
    bib = _block("bibtex")
    assert bib.count("{") == bib.count("}")
    assert _bib_field(bib, "title").replace("{LLM}", "LLM") == TITLE
    authors = [" ".join(reversed(a.split(", "))) for a in _bib_field(bib, "author").split(" and ")]
    assert authors == AUTHORS
    assert _bib_field(bib, "eprint") == ARXIV_ID
    assert _bib_field(bib, "url") == f"https://arxiv.org/abs/{ARXIV_ID}"


def test_byline_lists_the_authors_in_order() -> None:
    byline = re.search(r"<p align=\"center\">\n(Joshua.*?)<br/>", README).group(1)
    assert [a.strip(" *") for a in byline.split(",")] == AUTHORS


def test_citation_cff_agrees_with_the_readme() -> None:
    cff = yaml.safe_load((REPO / "CITATION.cff").read_text())
    assert cff["title"] == TITLE
    for people in (cff["authors"], cff["preferred-citation"]["authors"]):
        assert [f"{a['given-names']} {a['family-names']}" for a in people] == AUTHORS
    assert f"arXiv:{ARXIV_ID}" in {i["value"] for i in cff["identifiers"]}


def test_diagram_is_an_image_rendered_from_its_source() -> None:
    """GitHub's mobile app shows a Mermaid block as source, so the README embeds the rendered PNG."""
    import hashlib

    assert "```mermaid" not in README, "embed images/pipeline/pipeline.png, not a Mermaid block"
    assert 'src="images/pipeline/pipeline.png"' in README
    stamp = (REPO / "images" / "pipeline" / "pipeline.png.sha256").read_text().strip()
    assert stamp == hashlib.sha256(DIAGRAM.encode()).hexdigest(), "re-run images/pipeline/render.sh"


def badges_without_logo(readme: str) -> list[str]:
    """Each shields.io badge's label that carries no logo (owner, 2026-11-26: every badge has one)."""
    out = []
    for src in re.findall(r'src="(https://img\.shields\.io/[^"]+)"', readme):
        if "/endpoint?" in src:  # the badge's own JSON sets its logo (Ruff's does)
            continue
        if not re.search(r"[?&](amp;)?logo=", src):
            out.append(src.split("/badge/")[-1].split("-")[0])
    return out


def test_every_badge_has_a_logo() -> None:
    assert re.findall(r'src="https://img\.shields\.io/', README), "no badges found"
    assert badges_without_logo(README) == []


def test_a_badge_without_a_logo_is_caught() -> None:
    planted = README.replace("?logo=python&amp;logoColor=white", "", 1)
    assert planted != README, "planted change no longer applies"
    assert badges_without_logo(planted) == ["python"]
