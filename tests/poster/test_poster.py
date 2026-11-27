"""The NeurIPS 2026 poster (`poster/`): print size, fonts, logos, QR code, title, authors, dashes.

Each check is a function of its input (a pdfinfo or pdffonts report, a LaTeX source, a decoded QR
payload) so that `test_planted_*` can feed it a known-bad input and prove it refuses it. The checks
read the committed PDF; rebuild it with `make -C poster` after a source change.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent))
import test_grid as grid  # noqa: E402  (a module import: its tests are not collected twice)

REPO = Path(__file__).resolve().parents[2]
POSTER = REPO / "poster"
PDF = POSTER / "tpp_neurips2026_poster.pdf"
TEX = POSTER / "tpp_neurips2026_poster.tex"
URL = "https://lab-v2.github.io/tpp-coverage/"
PAGE_PT = (3968.5, 2551.18)  # 140 x 90 cm landscape, the NeurIPS maximum
LOGOS = {"neurips", "syracuse_ai_institute", "leibniz_lab"}
EM_DASH = re.compile("\u2014|---")
MAX_HEADER_FRACTION = 0.17  # header rule above 17% of the height (reference posters: 12-17%)
MAX_COLUMN_SPREAD_CM = 1.0  # the three column bottoms within 1 cm
PT_PER_CM = 72 / 2.54
# The owner asked for a poster that looks like LaTeX: every font is Computer / Latin Modern.
ALLOWED_FONT = re.compile(
    r"^(CM(MI|MIB|SY|EX|R|BX|SS)[0-9]+|Cm(mi|r|sy|ex)[0-9]+|MSBM10|MSAM10|LMRoman|LMSans|LMMath|"
    r"LatinModernMath)"
)
MATH_FONT = re.compile(r"LatinModernMath|LMMath|CMMI|Cmmi")
BLOCK_TITLE_PT = 41.2  # fallback block-title height; normally read off the TL;DR title
# Gaps run from a block's last text line to the next block title, so a tinted block's inner padding
# (about 0.4 cm) and a figure's bottom margin are inside them; 0.55 cm around the median allows that
# and nothing more (a 0.9 cm \vspace pad fails).
MAX_GAP_DEVIATION_CM = 0.4  # all gaps within a 0.8 cm band
MAX_WORDS = 800  # the reference posters carry 500-800 words; raised from 750 in round 7


def _run(*cmd: str) -> str:
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout


# --- the checks -----------------------------------------------------------------------------------


def page_size_problems(pdfinfo: str) -> list[str]:
    pages = re.search(r"^Pages:\s+(\d+)", pdfinfo, re.M)
    size = re.search(r"^Page size:\s+([\d.]+) x ([\d.]+) pts", pdfinfo, re.M)
    if not pages or not size:
        return ["pdfinfo report has no page count or size"]
    problems = [] if pages[1] == "1" else [f"{pages[1]} pages, not 1"]
    w, h = float(size[1]), float(size[2])
    if abs(w - PAGE_PT[0]) > 1 or abs(h - PAGE_PT[1]) > 1:
        problems.append(f"page is {w} x {h} pt, not 140 x 90 cm landscape {PAGE_PT}")
    return problems


def font_problems(pdffonts: str) -> list[str]:
    rows = pdffonts.splitlines()[2:]
    if not rows:
        return ["no fonts listed"]
    problems = []
    for row in rows:
        if "Type 3" in row:
            problems.append(f"Type 3 font: {row.split()[0]}")
        # Columns after the type and encoding: emb sub uni object-id generation.
        if row.split()[-5] != "yes":
            problems.append(f"not embedded: {row.split()[0]}")
    return problems


def logo_problems(tex: str) -> list[str]:
    """The images taken from ../images/ (the repo's logos) must be exactly the three logos."""
    used = {
        Path(name).stem for name in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex)
    }
    available = {p.stem for p in (REPO / "images").iterdir() if p.is_file()}
    logos = {u for u in used if u in available or "syracuse" in u or "logo" in u}
    problems = []
    if logos != LOGOS:
        problems.append(f"logos used {sorted(logos)}, expected {sorted(LOGOS)}")
    return problems


def title_author_problems(tex: str, cff: dict) -> list[str]:
    title = re.search(r"\\title\{(.+?)\}\n", tex, re.S)
    author = re.search(r"\\author\{(.+?)\}\n\\institute", tex, re.S)
    if not title or not author:
        return ["no \\title or \\author"]
    got_title = " ".join(title[1].replace("\\\\", " ").split())
    problems = [] if got_title == cff["title"] else [f"title {got_title!r} != {cff['title']!r}"]
    plain = re.sub(r"\\textsuperscript\{[^}]*\}|\\quad|\\\\", "|", author[1])
    got = [" ".join(n.split()) for n in plain.split("|") if n.strip()]
    want = [f"{a['given-names']} {a['family-names']}" for a in cff["authors"]]
    if got != want:
        problems.append(f"authors {got} != {want}")
    return problems


def em_dash_problems(text: str) -> list[str]:
    return [
        f"em dash on line {i}"
        for i, line in enumerate(text.splitlines(), 1)
        if EM_DASH.search(line)
    ]


def math_font_problems(pdffonts: str, need_math: bool = True) -> list[str]:
    """Fonts outside Lato, Raleway and CM/LM; and, for the poster, no genuine math font at all."""
    names = [row.split()[0].split("+", 1)[-1] for row in pdffonts.splitlines()[2:] if row.strip()]
    problems = [
        f"font {n} is not Lato, Raleway or CM/LM" for n in names if not ALLOWED_FONT.match(n)
    ]
    if need_math and not any(MATH_FONT.search(n) for n in names):
        problems.append("no LaTeX math font embedded")
    return problems


def heading_math_problems(tex: str) -> list[str]:
    """No $...$ inside a block title or a \\statement{} run-in head (bold math reads as a defect)."""
    heads = re.findall(
        r"\\begin\{(?:alert)?(?:block|card)\}(?:\[[^\]]*\])?\{((?:[^{}]|\{[^{}]*\})*)\}", tex
    )
    heads += re.findall(r"\\statement\{((?:[^{}]|\{[^{}]*\})*)\}", tex)
    return [f"math in heading: {h}" for h in heads if "$" in h]


def word_count_problems(text: str) -> list[str]:
    """Words as pdftotext prints them (whitespace-separated), header and footer included."""
    n = len(text.split())
    return [] if n <= MAX_WORDS else [f"{n} words, more than {MAX_WORDS}"]


def bbox_words(pdf: Path) -> list[tuple[float, float, float, float, str]]:
    """Every word pdftotext finds, as (xMin, yMin, xMax, yMax, text) in points from the top left."""
    html = _run("pdftotext", "-bbox", str(pdf), "-")
    pat = r'xMin="([\d.]+)" yMin="([\d.]+)" xMax="([\d.]+)" yMax="([\d.]+)">([^<]*)<'
    return [(float(a), float(b), float(c), float(d), w) for a, b, c, d, w in re.findall(pat, html)]


# --- layout: textpos cards on the 3-column grid (tests/poster/test_grid.py measures the grid) ------

ZONE_GAP_CM = 2.00  # header rule to the first cards, last cards to the footer rule


def zone_gap_problems(cs: list, rs: list, tex: str, tol: float = 0.10) -> list[str]:
    """No \\vspace anywhere, and the body zone sits 2.00 cm from both rules: header rule bottom to
    every column's first card, every column's last card to the footer rule top."""
    problems = ["a \\vspace in the source"] if "\\vspace" in tex else []
    if len(rs) != 2:
        return problems + [f"rules {rs}"]
    head, foot = sorted(rs, key=lambda r: r[1])
    for key in {(c[0], c[2]) for c in cs}:
        col = sorted(c for c in cs if (c[0], c[2]) == key)
        if abs(col[0][1] - head[3] - ZONE_GAP_CM) > tol:
            problems.append(f"column {key}: {col[0][1] - head[3]:.2f} cm under the header rule")
        if abs(foot[1] - col[-1][3] - ZONE_GAP_CM) > tol:
            problems.append(f"column {key}: {foot[1] - col[-1][3]:.2f} cm above the footer rule")
    return problems


def header_fraction(png: Path) -> float:
    """Height of the header rule (the first full-width row of the theme's dark blue) over the page height."""
    script = (
        "import sys; from PIL import Image\n"
        "im = Image.open(sys.argv[1]).convert('RGB'); w, h = im.size; px = im.load()\n"
        "near = lambda p: abs(p[0] - 39) < 20 and abs(p[1] - 60) < 20 and abs(p[2] - 117) < 20\n"
        "rows = [y for y in range(h // 20, h // 2) if sum(near(px[x, y]) for x in range(0, w, 8)) > 0.8 * w / 8]\n"
        "print(rows[0] / h if rows else 1.0)"
    )
    out = _run(
        "uv",
        "run",
        "--quiet",
        "--no-project",
        "--with",
        "pillow",
        "python",
        "-I",
        "-c",
        script,
        str(png),
    )
    return float(out)


def decode_qr(png: Path) -> list[str]:
    script = (
        "import sys, zxingcpp; from PIL import Image; "
        "print('\\n'.join(r.text for r in zxingcpp.read_barcodes(Image.open(sys.argv[1]))))"
    )
    out = _run("uv", "run", "--quiet", "--no-project", "--with", "zxing-cpp", "--with", "pillow",
               "python", "-I", "-c", script, str(png))  # fmt: skip
    return [line for line in out.splitlines() if line]


# --- the poster passes ----------------------------------------------------------------------------


def test_page_is_140_by_90_landscape() -> None:
    assert page_size_problems(_run("pdfinfo", str(PDF))) == []


def test_fonts_embedded_and_none_type3() -> None:
    assert font_problems(_run("pdffonts", str(PDF))) == []


def test_exactly_the_three_logos() -> None:
    assert logo_problems(TEX.read_text()) == []


def test_title_and_authors_match_citation_cff() -> None:
    cff = yaml.safe_load((REPO / "CITATION.cff").read_text())
    assert title_author_problems(TEX.read_text(), cff) == []


def test_no_em_dash_in_sources() -> None:
    for tex in POSTER.glob("*.tex"):
        assert em_dash_problems(tex.read_text()) == [], tex.name


def test_latex_fonts_only() -> None:
    assert math_font_problems(_run("pdffonts", str(PDF))) == []
    for fig in sorted((POSTER / "figures").glob("*.pdf")):
        assert math_font_problems(_run("pdffonts", str(fig)), need_math=False) == [], fig.name


def poster_words() -> list:
    """The poster's own words: the text inside the paper's figures (embedded TrueType) is the
    figures' own, so it is left out: the figure card from its title ("Kaplan", title size) to its
    caption ("Paper")."""
    words = bbox_words(PDF)
    title = [w for w in words if w[4] == "Kaplan" and w[3] - w[1] > 0.8 * BLOCK_TITLE_PT]
    caption = [w for w in words if w[4] == "Paper"]
    if not title or not caption:
        return words
    left, right = grid.EDGES[2] * PT_PER_CM, grid.EDGES[3] * PT_PER_CM
    top, bottom = title[0][3], caption[0][1]
    return [w for w in words if not (left <= w[0] < right and top < w[1] and w[3] < bottom)]


def footer_rule_top(pdf: Path = PDF) -> float:
    """Top (pt) of the footer rule: the lowest thin wide bar in the theme's dark blue."""
    import pymupdf

    page = pymupdf.open(pdf)[0]
    bars = [
        d["rect"]
        for d in page.get_drawings()
        if d["rect"].width > 0.9 * page.rect.width
        and d["rect"].height < 0.01 * page.rect.height
        and d.get("fill") not in (None, (1.0, 1.0, 1.0))
    ]
    assert len(bars) >= 2, "no footer rule found"
    return max(bars, key=lambda r: r.y0).y0


def body_word_problems(words: list, footer_top: float) -> list[str]:
    """At most MAX_WORDS above the footer rule; the footer band (links, footnotes, funding and
    references: attribution) is not counted."""
    return word_count_problems(" ".join(w[4] for w in words if w[3] <= footer_top))


def test_at_most_800_words() -> None:
    assert body_word_problems(poster_words(), footer_rule_top()) == []


def test_column_bottoms_on_one_line() -> None:
    # The three columns end on one line (the last card of each stretches to the body bottom).
    assert grid.bottom_problems(grid.cards()) == []


def test_zone_gap_bounded() -> None:
    assert zone_gap_problems(grid.cards(), grid.rules(), TEX.read_text()) == []


def test_block_gaps_are_the_theme_gap() -> None:
    assert grid.gap_problems(grid.cards()) == []


def test_block_titles_one_line() -> None:
    """Replaces the old title-rule check (the cards have no title rules): every card opens with
    its title on one line."""
    cs, lines = grid.cards(), grid.text_lines()
    assert grid.title_wrap_problems(cs, lines) == []
    firsts = [min((ln for ln in lines if grid.inside(ln, c)), key=lambda ln: ln[1]) for c in cs]
    assert all(abs(ln[4] - 43.8) < 0.5 for ln in firsts), [ln[6] for ln in firsts]


def test_header_under_17_percent(tmp_path: Path) -> None:
    _run("pdftoppm", "-png", "-r", "40", "-singlefile", str(PDF), str(tmp_path / "poster"))
    assert header_fraction(tmp_path / "poster.png") < MAX_HEADER_FRACTION


def test_qr_code_decodes_to_project_page(tmp_path: Path) -> None:
    _run("pdftoppm", "-png", "-r", "72", "-singlefile", str(PDF), str(tmp_path / "poster"))
    assert decode_qr(tmp_path / "poster.png") == [URL]


# --- each check refuses a planted defect ----------------------------------------------------------


def test_planted_wrong_page_size() -> None:
    assert page_size_problems("Pages:          1\nPage size:      3370.39 x 2383.94 pts (A0)\n")
    assert page_size_problems("Pages:          1\nPage size:      2551.18 x 3968.5 pts\n")
    assert page_size_problems("Pages:          2\nPage size:      3968.5 x 2551.18 pts\n")


def test_planted_type3_and_unembedded_fonts() -> None:
    head = "name type encoding emb sub uni object ID\n---- ---- ---- --- --- --- ---------\n"
    assert font_problems(head + "[none] Type 3 Custom yes no no 12 0\n")
    assert font_problems(head + "Helvetica Type 1 Standard no no no 7 0\n")


def test_planted_logos() -> None:
    tex = TEX.read_text()
    assert logo_problems(tex.replace("{leibniz_lab.png}", "{syracuse_university.pdf}"))
    assert logo_problems(tex.replace("{neurips.pdf}", "{neurips_dark.svg}"))


def test_planted_title_and_authors() -> None:
    cff = yaml.safe_load((REPO / "CITATION.cff").read_text())
    tex = TEX.read_text()
    assert title_author_problems(tex.replace("Joshua Shay Kricheli", "Joshua Krichel"), cff)
    assert title_author_problems(tex.replace("Robust LLM", "Robust"), cff)
    swapped = tex.replace("Venkata Gandikota", "@@").replace("Paulo Shakarian", "Venkata Gandikota")
    assert title_author_problems(swapped.replace("@@", "Paulo Shakarian"), cff)


def test_planted_em_dash() -> None:
    assert em_dash_problems("a \u2014 b")
    assert em_dash_problems("a --- b")
    assert not em_dash_problems("5M-77M and 0.18--0.80")


@pytest.mark.parametrize(
    "payload", ["https://lab-v2.github.io/tpp-coverage", "https://example.org/"]
)
def test_planted_wrong_qr(tmp_path: Path, payload: str) -> None:
    png = tmp_path / "qr.png"
    _run("uvx", "--quiet", "segno", "--scale", "4", "--output", str(png), payload)
    assert decode_qr(png) == [payload] != [URL]


def test_planted_too_many_words() -> None:
    text = " ".join(w[4] for w in poster_words())
    assert word_count_problems(text + " extra" * (MAX_WORDS + 1 - len(text.split())))
    assert not word_count_problems("word " * MAX_WORDS)
    # The footer band may run past 800; the body may not.
    top = 2000.0
    footer = [(0, top + 10, 10, top + 20, "ref")] * (MAX_WORDS + 50)
    body = [(0, 100, 10, 110, "word")] * 100
    assert not body_word_problems(body + footer, top)
    assert body_word_problems(body * 9, top)


def test_planted_column_spread() -> None:
    cs = grid.cards()
    short = [
        (c[0], c[1], c[2], c[3] - 2.0) if c[3] == max(x[3] for x in cs) and c[0] == 3.0 else c
        for c in cs
    ]
    assert grid.bottom_problems(short)


def test_planted_zone_gaps() -> None:
    cs, rs = grid.cards(), grid.rules()
    assert zone_gap_problems(cs, rs, TEX.read_text() + "\\vspace{1cm}")
    lowered = [(c[0], c[1] + 0.9, c[2], c[3]) if c[1] == min(x[1] for x in cs) else c for c in cs]
    assert zone_gap_problems(lowered, rs, TEX.read_text())


def test_planted_tall_header(tmp_path: Path) -> None:
    script = (
        "import sys; from PIL import Image, ImageDraw\n"
        "im = Image.new('RGB', (400, 283), 'white'); d = ImageDraw.Draw(im)\n"
        "d.rectangle([0, 60, 399, 62], fill=(39, 60, 117)); im.save(sys.argv[1])"
    )
    png = tmp_path / "tall.png"
    _run(
        "uv",
        "run",
        "--quiet",
        "--no-project",
        "--with",
        "pillow",
        "python",
        "-I",
        "-c",
        script,
        str(png),
    )
    assert header_fraction(png) > MAX_HEADER_FRACTION


def test_planted_math_fonts() -> None:
    head = "name type encoding emb sub uni object ID\n---- ---- ---- --- --- --- ---------\n"
    lm = "ABCDEF+LMRoman10-Regular-Identity-H CID Type 0C Identity-H yes yes yes 5 0\n"
    lato = "ABCDEF+Lato-Regular CID TrueType Identity-H yes yes yes 5 0\n"
    math = "ABCDEF+LatinModernMath-Regular-Identity-H CID Type 0C Identity-H yes yes yes 6 0\n"
    assert not math_font_problems(head + lm + math)
    assert math_font_problems(head + lm)  # no math font
    assert math_font_problems(head + lato + math)  # a non-LaTeX text font
    assert math_font_problems(
        head + lm + math + "ABCDEF+FiraMath-Regular CID Type 0C Identity-H yes yes yes 7 0\n"
    )
    assert math_font_problems(
        head + "ABCDEF+DejaVuSans CID TrueType Identity-H yes yes yes 8 0\n", need_math=False
    )


def test_planted_padded_gap() -> None:
    cs = sorted(grid.cards())
    padded = [cs[0], (cs[1][0], cs[1][1] + 0.9, cs[1][2], cs[1][3])] + cs[2:]
    assert grid.gap_problems(padded)  # a 0.9 cm "column balance" pad


def test_no_math_in_bold_heads() -> None:
    assert heading_math_problems(TEX.read_text()) == []


def test_planted_heading_math() -> None:
    tex = TEX.read_text()
    assert heading_math_problems(
        tex.replace("Corollary 2 (Chinchilla)}", "Corollary 2 ($n \\geq 5$)}")
    )
    assert heading_math_problems("\\begin{block}{Two designs in the $(N, D)$ plane}")


# --- figure shape, figure text inside the canvas, header centring --------------------------------

MAX_LOGO_OFFSET_CM = 0.5


def logo_offsets(png: Path, words: list) -> tuple[float, float]:
    """Vertical offsets (cm) of the left and right logo groups' ink centres from the title block's centre."""
    script = (
        "import sys, json; from PIL import Image\n"
        "im = Image.open(sys.argv[1]).convert('L'); px = im.load(); a = json.loads(sys.argv[2])\n"
        "rule = next(y for y in range(30, im.height // 2) if sum(px[x, y] < 150 for x in range(0, im.width, 4)) > 0.2 * im.width)\n"
        "def ink(xa, xb):\n"
        "    ys = [y for y in range(rule - 5) if any(px[x, y] < 200 for x in range(int(xa), int(xb), 2))]\n"
        "    return (min(ys) + max(ys)) / 2\n"
        "print(json.dumps([ink(0, a[0] - 20), ink(a[1] + 20, im.width), rule]))"
    )
    width = PAGE_PT[0]
    import json

    head = [
        w for w in words if w[3] < 0.17 * PAGE_PT[1] and 0.15 * width < w[0] and w[2] < 0.85 * width
    ]
    x0, x1 = min(w[0] for w in head), max(w[2] for w in head)
    out = _run(
        "uv",
        "run",
        "--quiet",
        "--no-project",
        "--with",
        "pillow",
        "python",
        "-I",
        "-c",
        script,
        str(png),
        json.dumps([x0, x1]),
    )
    left, right, _ = json.loads(out)
    centre = (min(w[1] for w in head) + max(w[3] for w in head)) / 2
    return (left - centre) / PT_PER_CM, (right - centre) / PT_PER_CM


def test_logo_groups_centred_on_title(tmp_path: Path) -> None:
    _run("pdftoppm", "-png", "-r", "72", "-singlefile", str(PDF), str(tmp_path / "p"))
    left, right = logo_offsets(tmp_path / "p.png", bbox_words(PDF))
    assert abs(left) <= MAX_LOGO_OFFSET_CM and abs(right) <= MAX_LOGO_OFFSET_CM, (left, right)


def test_planted_high_logo(tmp_path: Path) -> None:
    _run("pdftoppm", "-png", "-r", "72", "-singlefile", str(PDF), str(tmp_path / "p"))
    script = (
        "import sys; from PIL import Image\n"
        "im = Image.open(sys.argv[1]).convert('RGB'); w = im.width\n"
        "box = (0, 0, int(0.19 * w), int(0.13 * im.height)); crop = im.crop(box)\n"
        "im.paste((255, 255, 255), box); im.paste(crop, (0, -60)); im.save(sys.argv[1])"
    )
    _run(
        "uv",
        "run",
        "--quiet",
        "--no-project",
        "--with",
        "pillow",
        "python",
        "-I",
        "-c",
        script,
        str(tmp_path / "p.png"),
    )
    left, _ = logo_offsets(tmp_path / "p.png", bbox_words(PDF))
    assert abs(left) > MAX_LOGO_OFFSET_CM


# --- the paper's figures, copied byte for byte; no overlapping text -------------------------------

# The two figures the owner chose: the paper's Figure 1 (a) and (c), rendered poster-sized by
# figure1_panels.py (FIG1_STYLE=poster; owner's decision of 2026-11-27). Pinned by content.
PAPER_FIGURES = {
    "01_surface_Red_Pajama_Kaplan_First.pdf": "959c713c55b2ea9214bede699f850b08fecd371393ff0facdc5e1f7dd3d86aa0",
    "01_isoflop_Red_Pajama_Kaplan_First.pdf": "fced6e74ed34332be843368792d9269cb35d1aa0128f03de8d1da821e8ecae53",
}
MIN_OVERLAP = 0.15  # share of the smaller word box that another box may not cover


def figure_problems(folder: Path, tex: str) -> list[str]:
    import hashlib

    files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.glob("*.pdf")}
    used = set(re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{(01_[^}]+)\}", tex))
    problems = [f"{n}: not the paper's file" for n, h in files.items() if PAPER_FIGURES.get(n) != h]
    if set(files) != set(PAPER_FIGURES):
        problems.append(f"figures/ holds {sorted(files)}, expected {sorted(PAPER_FIGURES)}")
    if used != set(PAPER_FIGURES):
        problems.append(f"the poster shows {sorted(used)}")
    if "tikzpicture" in tex:
        problems.append("a TikZ figure: the poster shows only the paper's figures")
    return problems


def overlap_problems(words: list) -> list[str]:
    """Pairs of word boxes (pdftotext) that cover each other: overlapping text, legends or labels."""
    # pdftotext boxes run from the font's ascender to its descender; keep the upper 75% (the ink of
    # most glyphs) and skip the extensible math braces, whose box spans the indices they group.
    boxes = sorted(
        (
            (w[0], w[1], w[2], w[3] - 0.25 * (w[3] - w[1]), w[4])
            for w in words
            if "\u23df" not in w[4]
        ),
        key=lambda w: w[0],
    )
    problems = []
    for i, a in enumerate(boxes):
        for b in boxes[i + 1 :]:
            if b[0] >= a[2]:
                break
            ix = min(a[2], b[2]) - max(a[0], b[0])
            iy = min(a[3], b[3]) - max(a[1], b[1])
            if ix <= 0 or iy <= 0:
                continue
            small = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
            if small > 0 and ix * iy / small > MIN_OVERLAP:
                problems.append(f"{a[4]!r} and {b[4]!r} overlap at {[round(v) for v in a[:2]]}")
    return problems


def test_figures_are_the_papers() -> None:
    assert figure_problems(POSTER / "figures", TEX.read_text()) == []


def test_planted_made_up_figure(tmp_path: Path) -> None:
    for name in PAPER_FIGURES:
        (tmp_path / name).write_bytes((POSTER / "figures" / name).read_bytes())
    assert not figure_problems(tmp_path, TEX.read_text())
    (tmp_path / "01_isoflop_Red_Pajama_Kaplan_First.pdf").write_bytes(b"%PDF-1.4 redrawn")
    assert figure_problems(tmp_path, TEX.read_text())
    assert figure_problems(
        POSTER / "figures", TEX.read_text() + "\\begin{tikzpicture}\\end{tikzpicture}"
    )


def test_no_overlapping_text() -> None:
    assert overlap_problems(bbox_words(PDF)) == []


def test_planted_overlapping_text() -> None:
    words = bbox_words(PDF)
    a = words[len(words) // 2]
    assert overlap_problems(words + [(a[0] + 2, a[1] + 2, a[2] + 2, a[3] + 2, "planted")])


# --- figure size and one body size (round 10; rule changed in makeover M2) ------------------------

# The figures are rendered at their printed size (41.733 x 29.80 cm, figure1_panels.py
# FIG1_STYLE=poster) and set at scale 1.00 +- 1%, so their text prints at the size it was drawn:
# every span on the poster's type scale (ticks 23.9 pt, labels and legends 29.9 pt), math scripts
# at 0.7 or 0.5 of a scale size.
FIG_SCALE_TOL = 0.01
TYPE_SCALE_PT = (109.6, 89.7, 43.8, 29.9, 23.9)
SCRIPT_RATIOS = (1.0, 0.7, 0.5)
BODY_PT, CAPTION_PT = 29.9, 23.9


def figure_scale_problems(tex: str, folder: Path) -> list[str]:
    problems = []
    found = re.findall(r"\\includegraphics\[width=([\d.]+)cm\]\{(01_[^}]+)\}", tex)
    if len(found) != len(PAPER_FIGURES):
        return [f"{len(found)} figures with an absolute width, expected {len(PAPER_FIGURES)}"]
    for width_cm, name in found:
        m = re.search(r"^Page size:\s+([\d.]+) x", _run("pdfinfo", str(folder / name)), re.M)
        natural_cm = float(m[1]) / PT_PER_CM
        if abs(float(width_cm) / natural_cm - 1) > FIG_SCALE_TOL:
            problems.append(f"{name}: set at {width_cm} cm, rendered at {natural_cm:.3f} cm")
    return problems


def body_size_problems(lines: list) -> list[str]:
    """Running text (lines of at least 4 words in Latin Modern) is body (29.9) or caption (23.9) size."""
    return [
        f"{ln[6][:30]!r} at {ln[4]:.1f} pt"
        for ln in lines
        if len(ln[6].split()) >= 4
        and not ln[5].startswith("Cm")
        and min(abs(ln[4] - BODY_PT), abs(ln[4] - CAPTION_PT)) > 0.5
        and ln[4] < 40  # titles, the header and the hero numbers are not running text
    ]


def test_figures_set_large() -> None:
    assert figure_scale_problems(TEX.read_text(), POSTER / "figures") == []


def test_planted_small_figure() -> None:
    tex = TEX.read_text().replace("width=41.733cm]{01_surface", "width=30cm]{01_surface")
    assert figure_scale_problems(tex, POSTER / "figures")


def test_one_body_size() -> None:
    assert body_size_problems(grid.text_lines()) == []


def test_planted_off_size_text() -> None:
    lines = grid.text_lines()
    i = next(k for k, ln in enumerate(lines) if ln[6].startswith("Fixed"))
    bigger = lines[:i] + [(*lines[i][:4], 34.5, *lines[i][5:])] + lines[i + 1 :]
    assert body_size_problems(bigger)


# The poster's figures are poster-sized renders of the paper's panels (figure1_panels.py,
# FIG1_STYLE=poster), not byte copies; tests/tpp_analysis/visualizations/
# test_characterize_figure1_panels.py::test_figure1_poster_style_draws_the_paper_data proves they
# draw exactly the paper render's data.


def off_scale_text(lines: list, scale: float, tol: float = 0.5) -> list[tuple[float, str]]:
    """Spans whose printed size (size x scale) is neither a type-scale size nor a math script of one."""
    allowed = [f * t for t in TYPE_SCALE_PT for f in SCRIPT_RATIOS]
    return [
        (size * scale, text)
        for spans in lines
        for size, text, _ in spans
        if min(abs(size * scale - a) for a in allowed) > tol
    ]


def figure_text_problems(tex: str, folder: Path) -> list[str]:
    import pymupdf

    problems = []
    for width_cm, name in re.findall(r"\\includegraphics\[width=([\d.]+)cm\]\{(01_[^}]+)\}", tex):
        page = pymupdf.open(folder / name)[0]
        scale = float(width_cm) * PT_PER_CM / page.rect.width
        lines = [
            [
                (sp["size"], sp["text"].strip(), sp["bbox"])
                for sp in line["spans"]
                if sp["text"].strip()
            ]
            for block in page.get_text("dict")["blocks"]
            for line in block.get("lines", [])
        ]
        if not any(lines):
            problems.append(f"{name}: no text found")
            continue
        problems += [f"{name}: {t!r} prints {pt:.1f} pt" for pt, t in off_scale_text(lines, scale)]
    return problems


def test_planted_lone_small_tick_label() -> None:
    # An 18 pt tick label is off the scale; 23.9 pt ticks and their 16.7 pt (0.7) exponents are on it.
    assert off_scale_text([[(18.0, "4", (0, 0, 9, 18))]], 1.0)
    assert not off_scale_text([[(23.9, "10", (0, 0, 26, 26)), (16.7, "7", (27, 0, 36, 18))]], 1.0)


def test_figure_text_readable() -> None:
    assert figure_text_problems(TEX.read_text(), POSTER / "figures") == []


def test_planted_unreadable_figure_text() -> None:
    tex = TEX.read_text().replace("width=41.733cm]{01_isoflop", "width=33cm]{01_isoflop")
    assert figure_text_problems(tex, POSTER / "figures")


# --- the build PNGs are renders of the current PDF ------------------------------------------------


def png_staleness(pdf: Path, png: Path) -> list[str]:
    """The PDF hash recorded when `make` rendered `png` must be the current PDF's."""
    import hashlib

    record = png.with_name(png.name + ".pdf-sha256")
    if not png.exists():
        return []  # nothing rendered yet (a clean checkout); `make` renders it with the PDF
    if not record.exists():
        return [f"{png.name}: no render-time PDF hash"]
    if record.read_text().strip() != hashlib.sha256(pdf.read_bytes()).hexdigest():
        return [f"{png.name}: rendered from an older PDF"]
    return []


def test_virtual_png_matches_pdf() -> None:
    assert png_staleness(PDF, POSTER / "build" / "tpp_neurips2026_poster_virtual.png") == []


def test_planted_stale_png(tmp_path: Path) -> None:
    pdf, png = tmp_path / "p.pdf", tmp_path / "v.png"
    pdf.write_bytes(b"new")
    png.write_bytes(b"png")
    (tmp_path / "v.png.pdf-sha256").write_text("0" * 64)
    assert png_staleness(pdf, png)


# --- the page holds everything: no overfull \vbox in the build log --------------------------------


def overfull_problems(log: str) -> list[str]:
    return [line.strip() for line in log.splitlines() if line.startswith("Overfull \\vbox")]


STAMP = POSTER / "build_stamp.txt"  # written by `make`, committed: PDF sha256 and overfull count


def stamp_problems(stamp: str, pdf_sha: str) -> list[str]:
    fields = dict(line.split(" ", 1) for line in stamp.splitlines() if " " in line)
    problems = []
    if fields.get("pdf_sha256") != pdf_sha:
        problems.append("build_stamp.txt is not from the committed PDF (run make)")
    if fields.get("overfull_vbox", "").strip() != "0":
        problems.append(f"{fields.get('overfull_vbox')} overfull vbox in the build")
    return problems


def test_no_overfull_vbox() -> None:
    import hashlib

    assert stamp_problems(STAMP.read_text(), hashlib.sha256(PDF.read_bytes()).hexdigest()) == []
    log = POSTER / "build" / "tpp_neurips2026_poster.log"
    if log.exists():  # a local build: its log is checked too
        assert overfull_problems(log.read_text(errors="replace")) == []


# NeurIPS limits: virtual poster at most 5120 x 2880 px and 10 MB; thumbnail at most 320 x 256 px
# and 5 MB. Read from the committed stamp (CI has no build), and from build/ when it exists.
PNG_LIMITS = {"virtual_png": (5120, 2880, 10_000_000), "thumbnail_png": (320, 256, 5_000_000)}


def png_limit_problems(stamp: str) -> list[str]:
    fields = {line.split()[0]: line.split()[1:] for line in stamp.splitlines() if line.split()}
    problems = []
    for key, (max_w, max_h, max_bytes) in PNG_LIMITS.items():
        if len(fields.get(key, [])) != 2:
            problems.append(f"{key}: not in build_stamp.txt (run make)")
            continue
        size, dims = fields[key]
        w, h = (int(v) for v in dims.split("x"))
        if int(size) > max_bytes or w > max_w or h > max_h:
            problems.append(
                f"{key}: {w} x {h} px, {int(size)} bytes over {max_w} x {max_h}, {max_bytes}"
            )
    return problems


def test_pngs_within_neurips_limits() -> None:
    assert png_limit_problems(STAMP.read_text()) == []
    for key, name in (("virtual_png", "virtual"), ("thumbnail_png", "thumbnail")):
        png = POSTER / "build" / f"tpp_neurips2026_poster_{name}.png"
        if png.exists():  # a local build: the files themselves
            max_w, max_h, max_bytes = PNG_LIMITS[key]
            w, h = _png_size(png)
            assert png.stat().st_size <= max_bytes and w <= max_w and h <= max_h, (key, w, h)


def _png_size(png: Path) -> tuple[int, int]:
    import struct

    head = png.read_bytes()[16:24]  # the IHDR width and height
    return struct.unpack(">II", head)


def test_planted_png_over_limits() -> None:
    assert png_limit_problems("virtual_png 12000000 4480x2880\nthumbnail_png 1000 320x206\n")
    assert png_limit_problems("virtual_png 1000 4480x2880\nthumbnail_png 1000 400x258\n")
    assert png_limit_problems("virtual_png 1000 4480x2880\n")
    assert not png_limit_problems("virtual_png 1000 4480x2880\nthumbnail_png 1000 320x206\n")


def test_planted_bad_stamp() -> None:
    assert stamp_problems("pdf_sha256 abc\noverfull_vbox 0\n", "def")
    assert stamp_problems("pdf_sha256 abc\noverfull_vbox 2\n", "abc")
    assert not stamp_problems("pdf_sha256 abc\noverfull_vbox 0\n", "abc")


def test_planted_overfull_vbox() -> None:
    assert overfull_problems("x\nOverfull \\vbox (48.74733pt too high) detected at line 244\n")
    assert not overfull_problems("Overfull \\hbox (1.2pt too wide) in paragraph\n")


# --- trim-safe: no ink within 1 cm of an edge, footer text at least 1 cm above the bottom ---------

TRIM_CM = 1.0


def edge_ink_problems(pdf: Path) -> list[str]:
    """Drawn paths (other than white), images and text closer than TRIM_CM to any page edge."""
    import pymupdf

    page = pymupdf.open(pdf)[0]
    w, h, m = page.rect.width, page.rect.height, TRIM_CM * PT_PER_CM
    white = lambda c: c is None or all(v > 0.99 for v in c)  # noqa: E731
    boxes = [
        ("path", d["rect"])
        for d in page.get_drawings()
        if not (white(d.get("fill")) and white(d.get("color")))
    ]
    boxes += [("image", page.get_image_bbox(im)) for im in page.get_images(full=True)]
    boxes += [("text " + wd[4], pymupdf.Rect(wd[:4])) for wd in page.get_text("words")]
    return [
        f"{kind} at {[round(v / PT_PER_CM, 2) for v in r]} cm"
        for kind, r in boxes
        if r.x0 < m or r.y0 < m or r.x1 > w - m or r.y1 > h - m
    ]


def test_no_ink_within_1_cm_of_the_edges() -> None:
    assert edge_ink_problems(PDF) == []


def test_planted_full_bleed_rule(tmp_path: Path) -> None:
    import pymupdf

    doc = pymupdf.open()
    page = doc.new_page(width=PAGE_PT[0], height=PAGE_PT[1])
    page.draw_rect(pymupdf.Rect(0, 370, PAGE_PT[0], 376), color=None, fill=(0.1, 0.2, 0.4))
    page.insert_text((200, PAGE_PT[1] - 10), "footer", fontsize=20)
    doc.save(tmp_path / "bleed.pdf")
    found = edge_ink_problems(tmp_path / "bleed.pdf")
    assert any(f.startswith("path") for f in found) and any(f.startswith("text") for f in found)
