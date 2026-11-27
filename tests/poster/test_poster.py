"""The NeurIPS 2026 poster (`poster/`): print size, fonts, logos, QR code, title, authors, dashes.

Each check is a function of its input (a pdfinfo or pdffonts report, a LaTeX source, a decoded QR
payload) so that `test_planted_*` can feed it a known-bad input and prove it refuses it. The checks
read the committed PDF; rebuild it with `make -C poster` after a source change.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest
import yaml

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
    heads = re.findall(r"\\begin\{(?:alert)?block\}\{((?:[^{}]|\{[^{}]*\})*)\}", tex)
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


# --- layout: a top zone of three text columns, then the evidence row across the full width -------
# Cells 0-2 are the top-zone columns (widths read off the source, as multiples of \colwidth); cell
# 3 is the evidence row's banner and cell 4 its figure block.
SEP_FRACTION, COL_FRACTION = (
    0.02425,
    0.3,
)  # \sepwidth and \colwidth, as fractions of the paper width
ROW_MARK = "% ==== the evidence row"


def cell_edges(tex: str) -> list[tuple[float, float]]:
    width = PAGE_PT[0]
    sep, col = SEP_FRACTION * width, COL_FRACTION * width
    top = tex[: tex.index(ROW_MARK)] if ROW_MARK in tex else tex
    fractions = [float(f or 1) for f in re.findall(r"\\begin\{column\}\{([\d.]*)\\colwidth\}", top)]
    edges, x = [], sep
    for f in fractions:
        edges.append((x, x + f * col))
        x += f * col + sep
    row = 3 * col + 2 * sep
    banner = (
        float(re.search(r"\\begin\{column\}\{([\d.]+)\\linewidth\}", tex[tex.index(ROW_MARK) :])[1])
        if ROW_MARK in tex
        else 0.27
    )
    edges += [(sep, sep + banner * row), (sep + banner * row, sep + row)]
    return edges


def row_top(words: list) -> float:
    """yMin of the evidence row's figure-block title (its first word, "Kaplan", at title size)."""
    tops = [w[1] for w in words if w[4] == "Kaplan" and w[3] - w[1] > 0.8 * BLOCK_TITLE_PT]
    return min(tops) if tops else PAGE_PT[1]


def cell_of(w, edges: list, top: float) -> int:
    """The cell a word sits in: a top-zone column above the evidence row, a row cell below it."""
    cells = range(3, 5) if w[1] >= top - 1 else range(3)
    return min(
        cells,
        key=lambda c: 0 if edges[c][0] - 5 <= w[0] < edges[c][1] else 1 + abs(w[0] - edges[c][0]),
    )


def column_spread_problems(words: list, width: float, height: float) -> list[str]:
    """The lowest word of each top-zone column, above the evidence row: within MAX_COLUMN_SPREAD_CM."""
    edges, top = cell_edges(TEX.read_text()), row_top(words)
    body = [w for w in words if height * 0.2 < w[3] < top - 5]
    bottoms = []
    for c in range(3):
        col = [w[3] for w in body if cell_of(w, edges, top) == c]
        if not col:
            return [f"column {c + 1} has no text"]
        bottoms.append(max(col))
    spread = (max(bottoms) - min(bottoms)) / PT_PER_CM
    ok = spread < MAX_COLUMN_SPREAD_CM
    return [] if ok else [f"column bottoms {[round(b / PT_PER_CM, 1) for b in bottoms]} cm"]


def block_gaps(words: list, width: float) -> list[list[float]]:
    """Per top-zone column, the gaps (cm) between each block title and the text line above it."""
    edges, row = cell_edges(TEX.read_text()), row_top(words)
    gaps = []
    for c in range(3):
        tops = sorted({y0 for cc, y0, _ in block_title_tops(words, TEX.read_text()) if cc == c})
        col_gaps = []
        for top in tops[1:]:
            above = [w[3] for w in words if cell_of(w, edges, row) == c and w[3] < top - 1]
            col_gaps.append((top - max(above)) / PT_PER_CM)
        gaps.append(col_gaps)
    return gaps


def zone_gap_cm(words: list) -> float:
    """From the lowest top-zone text to the evidence row's titles (cm)."""
    top = row_top(words)
    return (top - max(w[3] for w in words if PAGE_PT[1] * 0.2 < w[3] < top - 1)) / PT_PER_CM


def block_title_tops(words: list, tex: str) -> list[tuple[int, float, float]]:
    """(column, yMin, yMax) of each block title: words at the title size whose text starts a title in the source."""
    heads = re.findall(r"\\begin\{(?:alert)?block\}\{((?:[^{}]|\{[^{}]*\})*)\}", tex)
    firsts = {}
    for h in heads:
        plain = re.sub(r"\\(CO|NC)\{\}", r"\1", h)
        plain = re.sub(r"[\\{}$]", " ", plain).split()
        # A title is a row holding exactly its words (the TL;DR repeats one title's words mid-line).
        if plain:
            firsts.setdefault(plain[0], []).append(plain)
    edges, top = cell_edges(tex), row_top(words)
    # The title size is read off the first block's title ("TL;DR"), so a body-scale change follows.
    sizes = [w[3] - w[1] for w in words if w[4] == "TL;DR"]
    title_pt = sizes[0] if sizes else BLOCK_TITLE_PT
    out = set()
    for w in words:
        if w[4] not in firsts or abs((w[3] - w[1]) - title_pt) >= 0.3:
            continue
        want = firsts[w[4]]
        # The words on this line that run on from w with normal word spacing (a title may span the
        # band across columns 2 and 3; another column's text on the same line starts a new run).
        line = sorted(
            (v for v in words if abs(v[1] - w[1]) < 1 and v[0] >= w[0]), key=lambda v: v[0]
        )
        row = [line[0]]
        for v in line[1:]:
            if v[0] - row[-1][2] > title_pt:
                break
            row.append(v)
        left = [v for v in words if abs(v[1] - w[1]) < 1 and 0 < w[0] - v[2] <= title_pt]
        if not left and [v[4] for v in row] in want:
            out.add((cell_of(w, edges, top), round(w[1], 1), round(w[3], 1)))
    return sorted(out)


def rule_coverage(png: Path, titles: list[tuple[int, float, float]], scale: float) -> list[float]:
    """For each title, the largest share of its column's width covered by dark pixels on one row just below it."""
    script = (
        "import sys, json; from PIL import Image\n"
        "im = Image.open(sys.argv[1]).convert('L'); px = im.load(); t = json.loads(sys.argv[2]); s = float(sys.argv[3])\n"
        "out = []\n"
        "for a, b, y0, y1 in t:\n"
        "    xa, xb = int((a + 0.075 * (b - a)) * s), int((b - 0.075 * (b - a)) * s)\n"
        "    best = max(sum(px[x, y] < 120 for x in range(xa, xb)) for y in range(int(y1 * s), int((y1 + 45) * s)))\n"
        "    out.append(best / (xb - xa))\n"
        "print(json.dumps(out))"
    )
    import json

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
        json.dumps([(*cell_edges(TEX.read_text())[c], y0, y1) for c, y0, y1 in titles]),
        str(scale),
    )
    return json.loads(out)


def gap_problems(gaps: list[list[float]]) -> list[str]:
    flat = sorted(g for col in gaps for g in col)
    if len(flat) < 6:
        return [f"only {len(flat)} block gaps found"]
    # Centre of the range, not the median: a gap may end on text or on a tinted box (about 0.4 cm
    # apart), so the honest gaps form a band; a padding \vspace sits outside it.
    centre = (flat[0] + flat[-1]) / 2 if flat[-1] - flat[0] <= 2 * MAX_GAP_DEVIATION_CM else None
    if centre is None:
        return [
            f"block gaps span {flat[0]:.2f}-{flat[-1]:.2f} cm, more than {2 * MAX_GAP_DEVIATION_CM} cm"
        ]
    return []


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
    """The poster's own words: the text inside the paper's figures (embedded TrueType since the
    figures were re-rendered) is the figures' own, so it is left out. The figure band runs from the
    band's block title to the caption line ("Left:")."""
    words = bbox_words(PDF)
    left = cell_edges(TEX.read_text())[4][0]
    title = [w for w in words if w[4] == "Kaplan" and w[3] - w[1] > 0.8 * BLOCK_TITLE_PT]
    caption = [w for w in words if w[4] == "Left:"]
    if not title or not caption:
        return words
    top, bottom = title[0][3], caption[0][1]
    return [w for w in words if not (w[0] >= left - 5 and top < w[1] and w[3] < bottom)]


def test_at_most_800_words() -> None:
    assert word_count_problems(" ".join(w[4] for w in poster_words())) == []


def test_column_bottoms_within_1_cm() -> None:
    # The three top-zone columns end on one line above the evidence row.
    assert column_spread_problems(bbox_words(PDF), *PAGE_PT) == []


MAX_ZONE_GAP_CM = (
    3.5  # lowest top-zone text to the evidence row's titles; 3.14 cm measured at 140 x 90
)


def test_zone_gap_bounded() -> None:
    # No \\vspace between the zones: the gap is the columns' and blocks' own skip, bounded.
    tex = TEX.read_text()
    between = tex[tex.rindex("\\end{columns}", 0, tex.index(ROW_MARK)) : tex.index(ROW_MARK)]
    assert "vspace" not in between
    gap = zone_gap_cm(bbox_words(PDF))
    assert MAX_GAP_DEVIATION_CM < gap <= MAX_ZONE_GAP_CM, gap


def test_block_gaps_are_the_theme_gap() -> None:
    assert gap_problems(block_gaps(bbox_words(PDF), PAGE_PT[0])) == []


def test_block_title_rules_are_unbroken(tmp_path: Path) -> None:
    """Nothing (a figure's white margin, a legend) covers a block title's rule: >= 95% of the column."""
    titles = block_title_tops(bbox_words(PDF), TEX.read_text())
    assert len(titles) == len(re.findall(r"\\begin\{(?:alert)?block\}", TEX.read_text()))
    _run("pdftoppm", "-png", "-r", "72", "-singlefile", str(PDF), str(tmp_path / "p"))
    cover = rule_coverage(tmp_path / "p.png", titles, 1.0)
    assert min(cover) >= 0.95, list(zip(titles, cover))


def test_planted_broken_rule(tmp_path: Path) -> None:
    titles = block_title_tops(bbox_words(PDF), TEX.read_text())
    _run("pdftoppm", "-png", "-r", "72", "-singlefile", str(PDF), str(tmp_path / "p"))
    script = (
        "import sys, json; from PIL import Image, ImageDraw\n"
        "im = Image.open(sys.argv[1]).convert('RGB'); c, y0, y1 = json.loads(sys.argv[2])\n"
        "a, b = json.loads(sys.argv[3]); ImageDraw.Draw(im).rectangle([a + 0.3 * (b - a), y1, a + 0.5 * (b - a), y1 + 45], fill='white')\n"
        "im.save(sys.argv[1])"
    )
    import json

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
        json.dumps(titles[1]),
        json.dumps(cell_edges(TEX.read_text())[titles[1][0]]),
    )
    assert rule_coverage(tmp_path / "p.png", [titles[1]], 1.0)[0] < 0.95


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


def test_planted_column_spread() -> None:
    words = bbox_words(PDF)
    w, h = PAGE_PT
    edges, top = cell_edges(TEX.read_text()), row_top(words)
    col1 = [x for x in words if cell_of(x, edges, top) == 0 and h * 0.2 < x[3] < top - 5]
    cut = max(x[3] for x in col1) - 2 * PT_PER_CM  # column 1 ends 2 cm higher
    others = [x for x in words if not (x in col1 and x[3] > cut)]
    assert column_spread_problems(others, w, h)


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
    gaps = block_gaps(bbox_words(PDF), PAGE_PT[0])
    padded = [list(col) for col in gaps]
    padded[0][1] += 0.9  # a \vspace{0.9cm} "column balance" pad, as round 4 had
    assert gap_problems(padded)
    assert gap_problems([[1.4], [1.4], []])


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
        "box = (0, 0, int(0.19 * w), int(0.155 * im.height)); crop = im.crop(box)\n"
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

# The two figures the owner chose: the paper's Figure 1 (a) and (c), as `src/updated_figures/` of the
# paper's LaTeX source ships them (`\includegraphics` in its empirical section). Pinned by content.
PAPER_FIGURES = {
    "01_surface_Red_Pajama_Kaplan_First.pdf": "ea6ee430fe4a25fa246ebce828d35d15799801fe39211e83931353815cd0eeca",
    "01_isoflop_Red_Pajama_Kaplan_First.pdf": "ce974e0d2f6af0263d8e8f8308b8384613584882486ac9992a48eb15c9b4529e",
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


# --- figure size and one body size (round 10) ------------------------------------------------------

# Each paper figure is set at least this many times its natural width (the figures are rendered at
# the paper's print size, so the poster scales them up), and every word inside them prints at least
# MIN_FIG_TEXT_PT tall (pdftotext height times the scale; the isoFLOP figure's exponent digits are
# the smallest).
MIN_FIG_SCALE = 0.75
MIN_FIG_TEXT_PT = 24.0
CAPTION_MIN_PT = 26.0  # pdftotext height of caption text
TEXT_WORD = re.compile(r"^[A-Za-z][A-Za-z,.;:'()-]*$")


def figure_scale_problems(tex: str, folder: Path) -> list[str]:
    problems = []
    found = re.findall(r"\\includegraphics\[width=([\d.]+)cm\]\{(01_[^}]+)\}", tex)
    if len(found) != len(PAPER_FIGURES):
        return [f"{len(found)} figures with an absolute width, expected {len(PAPER_FIGURES)}"]
    for width_cm, name in found:
        m = re.search(r"^Page size:\s+([\d.]+) x", _run("pdfinfo", str(folder / name)), re.M)
        natural_cm = float(m[1]) / PT_PER_CM
        if float(width_cm) < MIN_FIG_SCALE * natural_cm:
            problems.append(f"{name}: {width_cm} cm < {MIN_FIG_SCALE} x {natural_cm:.1f} cm")
    return problems


def header_rule_bottom(pdf: Path = PDF) -> float:
    """Bottom (pt) of the header rule: the thin full-width filled bar in the theme's dark blue."""
    import pymupdf

    page = pymupdf.open(pdf)[0]
    bars = [
        d["rect"]
        for d in page.get_drawings()
        if d["rect"].width > 0.95 * page.rect.width
        and d["rect"].height < 0.01 * page.rect.height
        and d.get("fill") not in (None, (1.0, 1.0, 1.0))
    ]
    assert bars, "no header rule found"
    return min(bars, key=lambda r: r.y0).y1


def body_size_problems(words: list, tex: str) -> list[str]:
    """Rows of running text (at least 4 words, half of them plain words) are body size or caption size.

    Block titles, the header, the footer, the banner numbers and display math are not running text.
    """
    width, height = PAGE_PT
    titles = {round(y0) for _, y0, _ in block_title_tops(words, tex)}
    edges, top = cell_edges(tex), row_top(words)
    header = header_rule_bottom() + 0.5 * PT_PER_CM  # running text starts below the header rule
    rows: dict[tuple[int, int], list] = {}
    for w in words:
        if header < w[1] and w[3] < 0.955 * height:
            rows.setdefault((cell_of(w, edges, top), round(w[1])), []).append(w)
    medians = []
    for (col, y), row in rows.items():
        plain = [w for w in row if TEXT_WORD.match(w[4])]
        if len(row) < 4 or 2 * len(plain) < len(row) or any(abs(y - t) <= 1 for t in titles):
            continue
        hs = sorted(w[3] - w[1] for w in row)
        medians.append(((col, y), hs[len(hs) // 2], row[0][4]))
    if not medians:
        return ["no running text found"]
    from collections import Counter

    body = Counter(round(m, 1) for _, m, _ in medians).most_common(1)[0][0]
    return [
        f"row at {pos} ({first!r}...) is {m:.1f} pt; body {body:.1f}, captions {CAPTION_MIN_PT}-{body:.1f}"
        for pos, m, first in medians
        if abs(m - body) > 0.6 and not (CAPTION_MIN_PT <= m < body)
    ]


def test_figures_set_large() -> None:
    assert figure_scale_problems(TEX.read_text(), POSTER / "figures") == []


def test_planted_small_figure() -> None:
    tex = TEX.read_text().replace("width=42.0cm]{01_surface", "width=15cm]{01_surface")
    assert figure_scale_problems(tex, POSTER / "figures")


def test_one_body_size() -> None:
    assert body_size_problems(poster_words(), TEX.read_text()) == []


def test_planted_off_size_text() -> None:
    words = poster_words()
    tex = TEX.read_text()
    y = next(round(w[1]) for w in words if w[4] == "Train")  # the recipe's step 2
    bigger = [
        (a, b, c, b + 1.15 * (d - b), t) if round(b) == y else (a, b, c, d, t)
        for a, b, c, d, t in words
    ]
    assert body_size_problems(bigger, tex)
    # The TL;DR's first line, just under the header rule, set at 40 pt.
    y = next(round(w[1]) for w in words if w[4] == "Fixed")
    tldr = [(a, b, c, b + 40.0, t) if round(b) == y else (a, b, c, d, t) for a, b, c, d, t in words]
    assert body_size_problems(tldr, tex)


def paper_copy_problems(poster_dir: Path, paper_dir: Path) -> list[str]:
    import hashlib

    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()  # noqa: E731
    return [
        f"{name}: poster copy differs from the paper's"
        for name in PAPER_FIGURES
        if not (paper_dir / name).is_file() or digest(poster_dir / name) != digest(paper_dir / name)
    ]


def test_figures_equal_the_paper_repos() -> None:
    """With TPP_PAPER_FIGURES=<the paper source's src/updated_figures>, the copies are byte-identical."""
    import os

    paper = os.environ.get("TPP_PAPER_FIGURES")
    if not paper:
        pytest.skip("TPP_PAPER_FIGURES not set (the paper's LaTeX source is a separate repository)")
    assert paper_copy_problems(POSTER / "figures", Path(paper)) == []


def test_planted_paper_copy_mismatch(tmp_path: Path) -> None:
    for name in PAPER_FIGURES:
        (tmp_path / name).write_bytes((POSTER / "figures" / name).read_bytes())
    assert not paper_copy_problems(POSTER / "figures", tmp_path)
    (tmp_path / "01_surface_Red_Pajama_Kaplan_First.pdf").write_bytes(b"%PDF old render")
    assert paper_copy_problems(POSTER / "figures", tmp_path)


def figure_text_problems(tex: str, folder: Path) -> list[str]:
    """Every word inside each paper figure, at the poster's scale, prints at least MIN_FIG_TEXT_PT."""
    problems = []
    for width_cm, name in re.findall(r"\\includegraphics\[width=([\d.]+)cm\]\{(01_[^}]+)\}", tex):
        m = re.search(r"^Page size:\s+([\d.]+) x", _run("pdfinfo", str(folder / name)), re.M)
        scale = float(width_cm) * PT_PER_CM / float(m[1])
        words = bbox_words(folder / name)
        if not words:
            problems.append(f"{name}: no text found")
            continue
        h, word = min((w[3] - w[1], w[4]) for w in words)
        if h * scale < MIN_FIG_TEXT_PT:
            problems.append(f"{name}: {word!r} prints {h * scale:.1f} pt")
    return problems


def test_figure_text_readable() -> None:
    assert figure_text_problems(TEX.read_text(), POSTER / "figures") == []


def test_planted_unreadable_figure_text() -> None:
    tex = TEX.read_text().replace("width=32.5cm]{01_isoflop", "width=12cm]{01_isoflop")
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


def test_planted_bad_stamp() -> None:
    assert stamp_problems("pdf_sha256 abc\noverfull_vbox 0\n", "def")
    assert stamp_problems("pdf_sha256 abc\noverfull_vbox 2\n", "abc")
    assert not stamp_problems("pdf_sha256 abc\noverfull_vbox 0\n", "abc")


def test_planted_overfull_vbox() -> None:
    assert overfull_problems("x\nOverfull \\vbox (48.74733pt too high) detected at line 244\n")
    assert not overfull_problems("Overfull \\hbox (1.2pt too wide) in paragraph\n")
