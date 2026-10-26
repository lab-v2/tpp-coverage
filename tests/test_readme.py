"""The README's images, code-structure diagram and citation match the repository and the paper."""

from __future__ import annotations

import re

import pytest
import yaml

from conftest import REPO

README = (REPO / "README.md").read_text()

TITLE = "Tokens-per-Parameter Coverage Is Critical for Robust LLM Scaling Law Extrapolation"
AUTHORS = ["Joshua Shay Kricheli", "Alexander Lawrence Reid", "Soumajyoti Sarkar",
           "Venkata Gandikota", "Paulo Shakarian"]
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
def test_local_images_exist_and_are_images(src: str):
    if src.startswith("http"):
        return
    path = REPO / src
    assert path.is_file(), src
    head = path.read_bytes()[:256]
    assert head.startswith(b"\x89PNG") or b"<svg" in head or head.lstrip().startswith(b"<?xml"), src


def test_logos_of_the_three_institutions():
    for alt in ("Syracuse University", "College of Engineering and Computer Science", "Leibniz Lab"):
        assert re.search(rf'<img alt="[^"]*{alt}[^"]*" src="images/', README), alt


def test_diagram_names_real_folders():
    diagram = _block("mermaid")
    assert re.search(r"^flowchart\b", diagram, re.M)
    folders = re.findall(r"<br/>(\w+)/\"", diagram)
    assert len(folders) >= 5, folders
    for folder in folders:
        hits = [p for p in REPO.rglob(folder) if p.is_dir() and ".venv" not in p.parts]
        assert hits, f"diagram names {folder}/, which is not in the repository"


def test_diagram_edges_only_join_declared_nodes():
    diagram = _block("mermaid")
    declared = set(re.findall(r"\b(\w+)[\[(]\S", diagram))
    used = set(re.findall(r"(\w+)(?:\[[^\]]*\]|\(\[[^\]]*\]\))?\s*-\.?->", diagram))
    used |= set(re.findall(r"(?:-\.?->|==>|\.->)\s*(\w+)", diagram))
    used |= set(re.findall(r"^\s*(\w+)(?:\(\[.*?\]\)|\[\(.*?\)\])?\s*(?:==>|-\.)", diagram, re.M))
    assert len(used) >= 6, used
    assert used <= declared, used - declared


def test_bibtex_matches_the_arxiv_paper():
    bib = _block("bibtex")
    assert bib.count("{") == bib.count("}")
    assert _bib_field(bib, "title").replace("{LLM}", "LLM") == TITLE
    authors = [" ".join(reversed(a.split(", "))) for a in _bib_field(bib, "author").split(" and ")]
    assert authors == AUTHORS
    assert _bib_field(bib, "eprint") == ARXIV_ID
    assert _bib_field(bib, "url") == f"https://arxiv.org/abs/{ARXIV_ID}"


def test_byline_lists_the_authors_in_order():
    byline = re.search(r"<p align=\"center\">\n(Joshua.*?)<br/>", README).group(1)
    assert [a.strip(" *") for a in byline.split(",")] == AUTHORS


def test_citation_cff_agrees_with_the_readme():
    cff = yaml.safe_load((REPO / "CITATION.cff").read_text())
    assert cff["title"] == TITLE
    for people in (cff["authors"], cff["preferred-citation"]["authors"]):
        assert [f"{a['given-names']} {a['family-names']}" for a in people] == AUTHORS
    assert f"arXiv:{ARXIV_ID}" in {i["value"] for i in cff["identifiers"]}
