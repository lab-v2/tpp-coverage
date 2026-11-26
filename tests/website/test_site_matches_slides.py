"""The project page follows the talk: one section per slide, in the deck's order, saying what it says.

The owner asked for the website to match the talk slides. The deck (`slides/*.tex`) is the source;
the page may add only the sections a project page needs and a talk does not (summary, abstract,
BibTeX), and those may state only numbers and formulas the deck states.

Per slide, the page's bullets pair one to one with the deck's `\\item`s, and each pair must carry
the same facts: every number, every formula (LaTeX on both sides, compared without spacing and
bold), every cited work (resolved to its first author) and every name (laws, corpora,
precisions, designs). The prose around them may differ. The references are matched by authors,
title, year and arXiv id; the header's authors by name. `PLANTED` holds edits that
the earlier, count-only version of this file let through; each must fail here.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
DECK = (REPO / "slides" / "tpp_neurips2026_slides.tex").read_text()
PAGE = (REPO / "website" / "index.html").read_text()
PAGE_ONLY = {"Summary", "Abstract", "BibTeX"}
REFERENCES = "References"

# Names counted as facts besides the automatic ones (a capital after the first letter, or a
# digit: RedPajama, FP32, CO, Gauss-Newton). These have a leading capital only.
NAMES = {"Chinchilla", "Kaplan", "Jacobian", "Hessian", "Gram", "Wikipedia", "Cosmopedia"}
NUMBER_WORDS = dict(
    zip("one two three four five six seven eight nine ten".split(), map(str, range(1, 11)))
)
EM_DASH = re.compile("\u2014|&mdash;|&#8212;|&#x2014;", re.IGNORECASE)
WRONG_NAME = re.compile(
    r"(?<!Kricheli, )Joshua(?!\s+Shay\s+Kricheli\b)\s+\S+|Kricheli,\s*Joshua(?!\s+Shay\b)|\bKrichel\b"
)
ARXIV_ID = re.compile(r"\b(\d{4}\.\d{4,5})\b")
YEAR = re.compile(r"\b(19[5-9]\d|20[0-3]\d)\b")
CITE = re.compile(r"@@cite:([^@]+)@@")
NUMBER = re.compile(r"(?<![\w.])\d+(?:,\d{3})*(?:\.\d+)?(?:%|[KMB](?![a-z]))?")
MATH = re.compile(r"(\\\$)|\$\$(.+?)\$\$|\\\[(.+?)\\\]|\$(.+?)\$", re.DOTALL)


# --- LaTeX and math ------------------------------------------------------------------------------


def _group(s: str, i: int) -> int:
    """Index just past the brace group that opens at ``s[i] == '{'``."""
    depth = 0
    for j in range(i, len(s)):
        depth += {"{": 1, "}": -1}.get(s[j], 0)
        if depth == 0:
            return j + 1
    raise ValueError(f"unbalanced braces at {i}: {s[i : i + 40]!r}")


def drop_command(s: str, name: str, keep_arg: int | None = None) -> str:
    """Remove ``\\name[opt]{a}{b}...``; with ``keep_arg``, keep that argument's text."""
    out, i = [], 0
    pat = re.compile(r"\\" + name + r"(?![a-zA-Z])\*?")
    while m := pat.search(s, i):
        out.append(s[i : m.start()])
        j = m.end()
        while j < len(s) and s[j] == "[":
            j = s.index("]", j) + 1
        args = []
        while j < len(s) and s[j] == "{":
            k = _group(s, j)
            args.append(s[j + 1 : k - 1])
            j = k
        if keep_arg is not None and len(args) > keep_arg:
            out.append(args[keep_arg])
        i = j
    out.append(s[i:])
    return "".join(out)


def split_math(s: str) -> tuple[str, list[str]]:
    """Prose with the math taken out (``\\$`` kept as a literal), and the math segments."""
    maths: list[str] = []

    def take(m: re.Match) -> str:
        if m.group(1):
            return m.group(1)
        maths.append(next(g for g in m.groups()[1:] if g is not None))
        return " "

    return MATH.sub(take, s), maths


def _unwrap(m: str) -> str:
    for cmd in (r"\boldsymbol", r"\mathbf", r"\bm"):
        if m.startswith(cmd + "{") and _group(m, len(cmd)) == len(m):
            return _unwrap(m[len(cmd) + 1 : -1])
    return m


def norm_math(m: str) -> list[str]:
    """One math segment as comparable pieces: split at ``\\qquad``, spacing and bold dropped."""
    out = []
    for piece in m.split(r"\qquad"):
        p = re.sub(r"\\[,;:! ]|\\quad(?![a-zA-Z])|~|\s+", "", piece)
        p = _unwrap(p.rstrip(".,;:")).replace(r"\%", "%")
        if p:
            out.append(p)
    return out


def facts(text: str, word_numbers: bool = True) -> Counter:
    """Numbers, formulas, citations and names of one block (prose with ``$math$``)."""
    prose, maths = split_math(text)
    found: Counter = Counter()
    for m in maths:
        for p in norm_math(m):
            numeric = re.fullmatch(r"\d+(?:,\d{3})*(?:\.\d+)?%?", p)
            found["number" if numeric else "math", p] += 1
    found.update(("cite", c) for c in CITE.findall(prose))
    prose = CITE.sub(" ", prose).replace(r"\$", " ")
    if word_numbers:
        words = r"\b(" + "|".join(NUMBER_WORDS) + r")\b"
        prose = re.sub(words, lambda m: NUMBER_WORDS[m[1]], prose)
    found.update(("number", n) for n in NUMBER.findall(prose))
    for tok in re.findall(r"[A-Za-z][\w-]*", prose):
        if tok in NAMES or re.search(r"[A-Z\d]", tok[1:]):
            found["name", tok] += 1
    return found


# --- the deck ------------------------------------------------------------------------------------


@dataclass
class Reference:
    surnames: list[str]
    title: str
    year: str
    arxiv: set[str] = field(default_factory=set)


def parse_reference(text: str) -> Reference:
    """Authors, title, year and arXiv ids of one reference (deck or page)."""
    text = re.sub(r"\s+", " ", text.replace("~", " ")).strip()
    m = re.match(r"(.*?(?:et al\.|[a-z]\.))\s+(.*?[a-z)][.?])(?:\s|$)", text)
    assert m, f"unparsed reference: {text!r}"
    surnames = re.findall(r"(?:[A-Z]\.\s?)+([A-Z][\w-]+)", m[1])
    years = YEAR.findall(ARXIV_ID.sub(" ", text[m.end() :]))
    assert surnames and years, f"no author or year in reference: {text!r}"
    return Reference(surnames, m[2].rstrip(".?").strip(), years[0], set(ARXIV_ID.findall(text)))


@dataclass
class Frame:
    title: str
    items: list[str]
    rest: str


def tex_prose(s: str) -> str:
    """The non-math LaTeX markup of ``s`` reduced to plain text; math stays in ``$...$``."""
    s = re.sub(r"(?<!\\)%.*", "", s)
    s = re.sub(r"\\\\(\[[^]]*\])?", " ", s)
    for cmd in ("includegraphics", "setlength", "end"):
        s = drop_command(s, cmd)
    s = re.sub(r"\\begin\{(columns|column|tabular)\}(\[[^]]*\])?(\{[^}]*\})?", " ", s)
    s = re.sub(r"\\begin\{\w+\}", " ", s)
    s = drop_command(s, "href", keep_arg=1)
    for cmd in ("texttt", "emph", "textbf", "textit"):
        s = drop_command(s, cmd, keep_arg=0)
    s = re.sub(r"\(slides? \d+\)", " ", s)
    s = re.sub(
        r"\\(Large|large|scriptsize|tiny|small|boldmath|centering|raggedright|par|medskip"
        r"|bigskip|smallskip|toprule|midrule|bottomrule|quad|hfill|vfill)(?![a-zA-Z])",
        " ",
        s,
    )
    s = s.replace("&", " ").replace("~", " ").replace("\\ ", " ")
    # Prose braces go; the ones inside math stay.
    out, last = [], 0
    for m in MATH.finditer(s):
        out += [re.sub(r"[{}]", " ", s[last : m.start()]), m.group(0)]
        last = m.end()
    return "".join(out) + re.sub(r"[{}]", " ", s[last:])


def frame_bodies(tex: str) -> dict[str, str]:
    pat = r"\\begin\{frame\}(?:\[[^]]*\])?\{([^}]*)\}(.*?)\\end\{frame\}"
    return dict(re.findall(pat, tex, re.S))


def frame_titles(tex: str) -> list[str]:
    return re.findall(r"\\begin\{frame\}(?:\[[^]]*\])?\{([^}]*)\}", tex)


def deck_references(tex: str) -> list[Reference]:
    entries = re.split(r"\\item\[\{\[\d+\]\}\]", frame_bodies(tex)[REFERENCES])[1:]
    return [parse_reference(re.sub(r"\\end\{.*", "", e, flags=re.S)) for e in entries]


def deck_frames(tex: str) -> list[Frame]:
    """Every titled slide but the references: its ``\\item``s and the rest of its text."""
    refs = deck_references(tex)

    def cite(m: re.Match) -> str:
        nums = re.findall(r"\d+", m[1])
        return " ".join(f"@@cite:{refs[int(n) - 1].surnames[0]}@@" for n in nums)

    frames = []
    for title, body in frame_bodies(tex).items():
        if title == REFERENCES:
            continue
        body = re.sub(r"\[(\d+(?:,\s*~?\d+)*)\]", cite, drop_command(body, "slidecite"))
        items: list[str] = []
        rest = body
        for m in re.finditer(r"\\begin\{itemize\}(.*?)\\end\{itemize\}", body, re.S):
            items += [tex_prose(i) for i in re.split(r"\\item(?![a-zA-Z])", m[1])[1:]]
            rest = rest.replace(m[0], " ")
        frames.append(Frame(title, items, tex_prose(rest)))
    return frames


def deck_authors(tex: str) -> list[str]:
    i = tex.index(r"\author{") + len(r"\author")
    raw = drop_command(tex[i + 1 : _group(tex, i) - 1], "textsuperscript")
    return [a.strip() for a in re.split(r"\\quad|\\\\", raw) if a.strip()]


# --- the page ------------------------------------------------------------------------------------


@dataclass
class Section:
    title: str
    items: list[str] = field(default_factory=list)
    texts: list[str] = field(default_factory=list)
    refs: list[tuple[str, str]] = field(default_factory=list)  # (id, text) of each <li id=ref-*>


class PageParser(HTMLParser):
    """The sections under each h2 or h3: bullets, other text, and citations as markers."""

    BLOCKS = {"li", "p", "figcaption", "th", "td"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.sections: list[Section] = []
        self.authors: list[str] = []
        self.cites: list[tuple[str, str]] = []  # (target id, link text)
        self.imgs: list[str] = []
        self.links: list[str] = []
        self._heading: list[str] | None = None
        self._block: tuple[str, list[str]] | None = None
        self._ref_id: str | None = None
        self._cite: tuple[str, list[str]] | None = None
        self._author: list[str] | None = None
        self._sup = self._pre = 0

    def handle_starttag(self, tag: str, attrs: list) -> None:
        a = {k: v or "" for k, v in attrs}
        href = a.get("href", "")
        if tag in ("h2", "h3"):
            self._heading = []
        elif tag == "img":
            self.imgs.append(a.get("src", ""))
        elif tag == "a" and href.startswith("#ref-"):
            self._cite = (href[5:], [])
        elif tag == "a":
            self.links.append(href)
        elif tag == "sup":
            self._sup += 1
        elif tag == "pre":
            self._pre += 1
        elif tag == "span" and "author-block" in a.get("class", ""):
            self._author = []
        if tag in self.BLOCKS and self._block is None and self.sections and not self._pre:
            self._block = (tag, [])
            if tag == "li" and a.get("id", "").startswith("ref-"):
                self._ref_id = a["id"][4:]

    def handle_endtag(self, tag: str) -> None:
        if tag in ("h2", "h3") and self._heading is not None:
            self.sections.append(Section(re.sub(r"\s+", " ", "".join(self._heading)).strip()))
            self._heading = None
        elif tag == "a" and self._cite is not None:
            target, words = self._cite
            self.cites.append((target, "".join(words)))
            self._cite = None
            self._emit(f" @@cite:{target}@@ ")
        elif tag == "sup":
            self._sup -= 1
        elif tag == "pre":
            self._pre -= 1
        elif tag == "span" and self._author is not None:
            self.authors.append("".join(self._author).strip().rstrip(",").strip())
            self._author = None
        if self._block is not None and tag == self._block[0]:
            kind, words = self._block
            sec, text = self.sections[-1], "".join(words)
            if self._ref_id is not None:
                sec.refs.append((self._ref_id, text))
                self._ref_id = None
            else:
                (sec.items if kind == "li" else sec.texts).append(text)
            self._block = None

    def _emit(self, data: str) -> None:
        if self._block is not None:
            self._block[1].append(data)

    def handle_data(self, data: str) -> None:
        if self._heading is not None:
            self._heading.append(data)
        if self._author is not None and not self._sup:
            self._author.append(data)
        if self._cite is not None:
            self._cite[1].append(data)
        else:
            self._emit(data)


def parse_page(page: str) -> PageParser:
    p = PageParser()
    p.feed(page)
    p.close()
    return p


# --- the checks ----------------------------------------------------------------------------------


def _diff(where: str, deck: Counter, page: Counter) -> list[str]:
    if deck == page:
        return []
    return [f"{where}: deck only {dict(deck - page)}, page only {dict(page - deck)}"]


def _site_files(repo: Path) -> set[str]:
    """Site paths, besides website/, that pages.yml copies from files the repo tracks."""
    wf = (repo / ".github" / "workflows" / "pages.yml").read_text()
    out = set()
    for m in re.finditer(r"^\s*cp ((?:[^\s-]\S* )+)_site/(\S+?)/?$", wf, re.M):
        out |= {f"{m[2]}/{Path(s).name}" for s in m[1].split() if (repo / s).is_file()}
    return out


def problems(deck: str, page: str, repo: Path = REPO) -> list[str]:
    """Every way the page departs from the deck; empty when it follows it."""
    out: list[str] = []
    for label, text in (("deck", deck), ("page", page)):
        if EM_DASH.search(text):
            out.append(f"{label}: em dash")
        out += [f"{label}: author name {bad!r}" for bad in WRONG_NAME.findall(text)]
    p = parse_page(page)

    # The header's and the BibTeX's authors are the deck's.
    authors = deck_authors(deck)
    if p.authors != authors:
        out.append(f"page authors {p.authors} != deck authors {authors}")
    bib = re.search(r"author\s*=\s*\{(.*?)\},?\n", page)
    bib_authors = [" ".join(reversed(a.split(", "))) for a in bib[1].split(" and ")] if bib else []
    if bib_authors != authors:
        out.append(f"BibTeX authors {bib_authors} != deck authors {authors}")

    # The paper's arXiv id is CITATION.cff's, and every arXiv link reads the id it opens.
    cff = set(ARXIV_ID.findall((repo / "CITATION.cff").read_text()))
    paper = set(re.findall(r'arxiv\.org/abs/(\d{4}\.\d{4,5})">arXiv<', page))
    paper |= set(re.findall(r"note\s*=\s*\{arXiv:(\d{4}\.\d{4,5})\}", page))
    if paper != cff or not cff:
        out.append(f"paper arXiv ids on the page {paper} != CITATION.cff {cff}")
    for href, text in re.findall(r'href="https://arxiv\.org/abs/([^"]+)">arXiv:([^<]+)<', page):
        if href != text:
            out.append(f"arXiv link to {href} reads arXiv:{text}")

    # Sections in the deck's order; every h2 and h3, tags stripped.
    titles = [s.title for s in p.sections if s.title not in PAGE_ONLY]
    if titles != frame_titles(deck):
        out.append(f"page sections {titles} != deck frames {frame_titles(deck)}")

    # References: the same works in the same order, by authors, title, year and arXiv id.
    deck_refs = deck_references(deck)
    entries = [r for s in p.sections if s.title == REFERENCES for r in s.refs]
    page_refs = {rid: parse_reference(t) for rid, t in entries}
    if len(entries) != len(deck_refs):
        out.append(f"{len(entries)} references on the page, {len(deck_refs)} in the deck")
    for n, (d, (rid, _)) in enumerate(zip(deck_refs, entries), 1):
        q = page_refs[rid]
        if (d.surnames, d.title, d.year) != (q.surnames, q.title, q.year) or not d.arxiv <= q.arxiv:
            out.append(f"reference [{n}]: deck {d} != page {q}")
    for target, text in p.cites:
        ref = page_refs.get(target)
        if ref is None:
            out.append(f"citation #ref-{target} has no reference")
        elif ref.surnames[0] not in text or ref.year not in text:
            out.append(f"citation {text!r} links to #ref-{target} ({ref.surnames[0]}, {ref.year})")

    def page_facts(text: str, word_numbers: bool = True) -> Counter:
        def first(m: re.Match) -> str:
            ref = page_refs.get(m[1])
            return f"@@cite:{ref.surnames[0] if ref else '#' + m[1]}@@"

        return facts(CITE.sub(first, text), word_numbers)

    # Each slide: bullets pair one to one; the rest of the slide carries the same facts.
    sections = {s.title: s for s in p.sections}
    deck_all: Counter = Counter()
    for f in deck_frames(deck):
        deck_all += sum(map(facts, f.items), facts(f.rest))
        s = sections.get(f.title)
        if s is None:
            continue
        if len(s.items) != len(f.items):
            out.append(f"{f.title}: {len(s.items)} bullets on the page, {len(f.items)} in the deck")
        for n, (di, pi) in enumerate(zip(f.items, s.items), 1):
            out += _diff(f"{f.title}, bullet {n}", facts(di), page_facts(pi))
        page_rest = sum(map(page_facts, s.texts), Counter())
        out += _diff(f"{f.title}, text", facts(f.rest), page_rest)

    # The page-only sections state only digits and formulas the deck states.
    for title in ("Summary", "Abstract"):
        s = sections.get(title, Section(title))
        for block in s.items + s.texts:
            for kind, value in page_facts(block, word_numbers=False):
                if kind in ("number", "math") and (kind, value) not in deck_all:
                    out.append(f"{title}: {kind} {value!r} is not in the deck")

    # Every image and local link resolves on the published site.
    site = _site_files(repo)
    for src in p.imgs + [h for h in p.links if not re.match(r"(https?:|mailto:|#)", h)]:
        if not ((repo / "website" / src).is_file() or src in site):
            out.append(f"{src} does not exist on the site")
    return out


# --- tests ---------------------------------------------------------------------------------------


def test_page_follows_the_deck() -> None:
    assert problems(DECK, PAGE) == []


def test_every_slide_and_section_is_read() -> None:
    p = parse_page(PAGE)
    frames = deck_frames(DECK)
    assert len(frames) == 14 and sum(len(f.items) for f in frames) == 44
    assert sum(len(s.items) for s in p.sections if s.title not in PAGE_ONLY) == 44
    assert len(deck_references(DECK)) == 15 and len(p.cites) == 25
    assert p.authors == deck_authors(DECK) and len(p.authors) == 5
    assert {"images/neurips.svg", "slides/tpp_neurips2026_slides.pdf"} <= _site_files(REPO)


def test_page_shows_only_figures_the_deck_shows() -> None:
    deck_figs = {
        Path(f).stem for f in re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]*)\}", DECK)
    }
    page_figs = {Path(f).stem for f in re.findall(r'<img src="static/images/([^"]+)"', PAGE)}
    assert page_figs <= deck_figs, page_figs - deck_figs


def test_heading_markup_is_stripped() -> None:
    old = '<h2 class="title is-3">Gauss-Newton</h2>'
    assert problems(DECK, PAGE.replace(old, "<h2>Gauss-<em>Newton</em></h2>")) == []


# (what, old text on the page, new text): each a change the count-only test let through.
PLANTED = [
    ("win rate in the table", "<strong>97.3%</strong>", "<strong>91.3%</strong>"),
    ("win rate in the summary", "win 97.3% of", "win 91.3% of"),
    ("win rate in the abstract", "with a 97.3% win rate", "with a 91.3% win rate"),
    ("model count", "1,900 models", "900 models"),
    (
        "main result rate",
        r"\boldsymbol{\kappa_{A,B} = \Theta\!\left(\dfrac{1}{\varepsilon^{2}}\right)}",
        r"\boldsymbol{\kappa_{A,B} = \Theta\!\left(\dfrac{1}{\varepsilon}\right)}",
    ),
    (
        "Kaplan citation target",
        r'$\alpha_D \approx 0.095$ (<a href="#ref-kaplan">',
        r'$\alpha_D \approx 0.095$ (<a href="#ref-hoffmann">',
    ),
    (
        "fake reference",
        "</ol>",
        '<li id="ref-fake">A. Nobody et al. A fake paper. arXiv:2601.00001, 2026.</li></ol>',
    ),
    (
        "reference swapped for a fake, same count",
        "J. Hestness et al. Deep learning scaling is predictable, empirically.",
        "J. Nobody et al. A fake paper on scaling.",
    ),
    ("deleted bullet", "<li>NC bars are tight.</li>", ""),
    ("truncated author", "Joshua Shay Kricheli<sup>", "Joshua Kricheli<sup>"),
    ("truncated author, Unix name", "Joshua Shay Kricheli<sup>", "Joshua Krichel<sup>"),
    ("truncated BibTeX author", "Kricheli, Joshua Shay and", "Kricheli, Joshua and"),
    ("caption corpus", "on RedPajama (<a", "on C4 (<a"),
    ("caption corpus, mixed case", "on RedPajama (<a", "on SlimPajama (<a"),
    ("paper arXiv id", "abs/2605.08541", "abs/2605.08514"),
    ("reference arXiv id", "2405.21015", "2405.21051"),
    ("reference arXiv link text", "arXiv:2411.04330</a>", "arXiv:2411.04303</a>"),
    ("em dash", "Gauss-Newton drops", "Gauss-Newton \u2014 drops"),
    ("em dash entity", "NC bars are tight.", "NC bars &mdash; are tight."),
    ("missing image", "01_convergence_Red_Pajama_Kaplan_First.png", "01_convergence_x.png"),
    (
        "extra h3 heading",
        '<h2 class="title is-3">Gauss-Newton</h2>',
        "<h3>Step</h3><h2>Gauss-Newton</h2>",
    ),
    ("number word", "four laws, five corpora", "four laws, six corpora"),
    ("formula in a bullet", r"$k \approx 20$", r"$k \approx 25$"),
]


@pytest.mark.parametrize("what,old,new", PLANTED, ids=[p[0] for p in PLANTED])
def test_planted_changes_fail(what: str, old: str, new: str) -> None:
    assert old in PAGE, f"planted change {what!r} no longer applies"
    assert problems(DECK, PAGE.replace(old, new, 1)), what
