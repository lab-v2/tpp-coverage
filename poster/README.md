# poster

The NeurIPS 2026 poster, 140 x 90 cm landscape (the NeurIPS maximum),
built with the [gemini](https://github.com/anishathalye/gemini) beamerposter theme (vendored in
`theme/` with its MIT licence; one local change, the inset rules). Every text, title and headings included, is set in Latin Modern Roman and
the mathematics in Latin Modern Math, as LaTeX sets them; every font is embedded (no Type 3).

`make` converts the SVG logos in `../images/` to PDF, draws the QR code of the project page, and
builds with `latexmk -xelatex` into `build/`; only `tpp_neurips2026_poster.pdf` is kept next to the
source. Every `make` (and `make png`) re-renders the virtual poster (`build/tpp_neurips2026_poster_virtual.png`, 4480 x
2880 px) and the thumbnail (`build/tpp_neurips2026_poster_thumbnail.png`, 320 x 206 px) within the
NeurIPS limits, and records the PDF's sha256 beside the PNG so `tests/poster/` fails on a PNG rendered from an older PDF. Needs `latexmk`, `xelatex`, poppler's `pdftoppm` and `uv` (for `cairosvg` and `segno`).

The poster shows only the paper's own figures, Figure 1 (a), the loss surface, and Figure 1 (c), the
induced isoFLOP curves (Kaplan law on RedPajama, first epoch). Owner's decision of 2026-11-27: the
poster gets a poster-sized render of these panels from the paper's own code and the same released
data, instead of byte copies of the paper's files (whose text, set for print, came out at 66-83 pt
on the poster). `src/tpp_analysis/visualizations/figure1_panels.py` with `FIG1_STYLE=poster` multiplies
every text size and point-sized gap by `FIG1_TEXT_SCALE` (legends by 1.45 times that); the paper style
is unchanged (its PDFs' content streams are identical). `figures/` holds:

| Figure | Rendered with | Set at | Its text on the poster (main, by font size) | Legend |
|---|---|---|---|---|
| `01_surface_...pdf` | `FIG1_STYLE=poster FIG1_TEXT_SCALE=0.31` | 44 cm | 27.2-32.0 pt | one column, right of the 3-D box |
| `01_isoflop_...pdf` | `FIG1_STYLE=poster FIG1_TEXT_SCALE=0.33` | 45 cm | 26.0-29.8 pt | one column, right of the colorbar |

Each figure is centred in its half of the figure block; the caption starts at the block's left edge.
The poster renders are made by `src/tpp_analysis/visualizations/figure1_panels.py` (poster-only
branches: text and gaps times `FIG1_TEXT_SCALE`, legends beside the plots) and kept in `figures/`,
pinned by SHA-256 in `tests/poster/test_poster.py`; the owner approved this on 2026-11-27.

Regenerate with `TPP_DATA_ROOT=<released data root> FIG1_STYLE=poster FIG1_TEXT_SCALE=<scale> uv run python
src/tpp_analysis/visualizations/figure1_panels.py` and copy the panel from `full_coverage_results/figures/`.
`tests/tpp_analysis/visualizations/test_characterize_figure1_panels.py::test_figure1_poster_style_draws_the_paper_data`
proves the poster style draws exactly the paper render's data (lines, collections, limits, labels, legends). The QR code of the project page
sits in the header, beside the NeurIPS logo. `tests/poster/` checks the page size, the fonts (Latin
Modern or Computer Modern only), the three logos, the QR code, the title and authors against
`CITATION.cff`, the absence of em dashes, at most 800 words above the footer rule (the footer band is attribution), a header under 17% of the height, the
top-zone column bottoms within 1 cm, block gaps within a 0.8 cm band, the gap between the zones at most 3.5 cm with no `\vspace`, no overfull `\vbox` (the committed `build_stamp.txt` holds the PDF sha256 and overfull count that `make` wrote), no math in bold heads, block title rules
unbroken, logo groups centred on the title block within 0.5 cm, the figures being the pinned poster renders (by
SHA-256), set at least 0.75 times their natural width (rendered at paper print size), every figure word printing at least 24 pt, and no TikZ figure, one body text size (captions at
least 26 pt), no overlapping text, figure text no larger than the block titles; the virtual PNG must be a render of the current PDF.

## Page size: A0 against 140 x 90 cm

The owner chose 140 x 90 cm, the NeurIPS maximum, on 2026-11-27, because the page height is the
binding constraint. The vertical budget of the current build (measured with `pdftotext -bbox`):

| Band | From (cm) | To (cm) |
|---|---|---|
| Header, to the bottom of its rule | 0 | 13.4 |
| Top zone: first block title to the lowest column text (46.2 / 46.3 / 46.7) | 15.4 | 46.7 |
| Gap between the zones | 46.7 | 49.4 |
| Evidence row: its titles to the caption | 49.4 | 85.8 |
| Footer | 88.3 | 90.0 |

The figures are set by equal plot-box height: 41.2 cm (surface) and 33 cm (isoFLOP) give 3-D box and
isoFLOP axes both about 18.8 cm tall (399 / 873 and 499 / 874 of their canvases). Their text prints
up to 82.9 pt (Fig. 1a, "Loss") and 66.4 pt (Fig. 1c) against 34.5 pt block titles; the gate for that
is a strict xfail until the paper figures change.

The binding constraint is the page height: all the text sits in the top zone, and the evidence row
takes what is left. Measured with xelatex (body 28.8 pt, scale 1.16):

| Page | Layout | Figures (surface / isoFLOP) | Result |
|---|---|---|---|
| A0, 118.9 x 84.1 cm | band across columns 2-3 (round 15) | 30 / 24.5 cm | fits |
| A0 | evidence row, equal columns | 36 / 29 cm | overfull by 266.5 pt |
| A0 | evidence row, equal columns | 27 / 21 cm | overfull by 48.7 pt |
| 140 x 90 cm | evidence row, equal columns | 36 / 29 cm | overfull by 98.6 pt |
| 140 x 90 cm | evidence row, equal columns | 33 / 26 cm | overfull by 14.8 pt; columns end at 51.9 / 54.2 / 47.0 cm |
| 140 x 90 cm | evidence row, columns 0.80 / 1.50 / 0.70 | 34 / 26.5 cm | overfull by 11.0 pt |
| 140 x 90 cm | evidence row, columns 0.80 / 1.50 / 0.70 | **33.5 / 26 cm** | **fits; columns end at 52.3 / 52.7 / 53.0 cm** (round 17) |
| 140 x 90 cm | the same, body 30 pt (scale 1.21) | 33.5 / 26 cm | overfull by 67.2 pt (not used) |
| 140 x 90 cm | Takeaways in the banner cell, columns 0.67 / 1.36 / 0.98 (round 18) | **42 / 32.5 cm** | **fits; columns end at 46.2 / 46.3 / 46.7 cm** (used) |
| 140 x 90 cm | the same, body 30 pt (scale 1.21) | 42 / 32.5 cm | overfull by 85.9 pt (not used) |

At 41.2 / 33 cm the surface's 3-D box prints about 18.8 cm wide and the isoFLOP axes about 16.7 cm
(axes share of the natural canvas width: 399 / 873 and 442 / 874 pt).

## Print-ready build

The footer carries the paper's funding sentence verbatim (`src/ack.tex`: "This research was funded by
ARO grant W911NF-24-1-0007.") and a one-line reference list for the works the poster names (Kaplan et
al. 2020, Droppo and Elibol 2021, Hoffmann et al. 2022, Muennighoff et al. 2025, Bhagia et al. 2025,
Magnusson et al. 2025, from the paper's `.bib`). Proofread on 2026-11-27 against the paper's `src/neurips2026_conference.tex`: title,
the five authors in the OpenReview order, the equal-contribution and Amazon marks, both affiliations,
and all 50 numbers in the poster source (each found verbatim in the paper's `.tex` sources).

Trim-safe without bleed: the header and footer rules stop 1.5 cm inside each edge (a local change to
the vendored `theme/beamerthemegemini.sty`, marked in it), and `tests/poster/` fails on any ink, image or
text within 1.0 cm of an edge (measured text margins: left 2.44, right 1.36, top 1.36, bottom 1.10 cm).

Final PDF (round 24): `tpp_neurips2026_poster.pdf`, sha256
`74dee81eb6002fb03c03507650900eb2e1a14b0b251bf8433808e3c123000317`, 749 words above the footer rule (the
word gate's count; the footer band, links, footnotes, funding and references, adds 59); virtual PNG 4480 x 2880 px,
1,518,967 bytes; thumbnail 320 x 206 px, 38,201 bytes (`build_stamp.txt`, gated in `tests/poster/`).
