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
induced isoFLOP curves (Kaplan law on RedPajama, first epoch), as poster renders from the paper's own
code and the same released data. `src/tpp_analysis/visualizations/figure1_panels.py` with
`FIG1_STYLE=poster` draws each panel at its printed size, 41.733 x 30.41 cm (one column's text
width; the 3-D panel is drawn on a canvas 1.33 times that and saved as the printed-size window
around its ink, with the paper's 3-D view), with a transparent canvas, ticks at 23.9 pt, axis labels and legends at 29.9 pt, and the
legends inside the axes and at least 0.2 cm inside the saved figure (Fig. 1a in one column in the empty bottom-right corner, right of the D label, clear of the 3-D box
by at least 0.2 cm;
Fig. 1c upper right, inside its axes frame); each legend frame sits 0.30 +- 0.05 cm from the two edges
of its corner (spines, or the window edges for the 3-D panel), and Fig. 1c's colour bar 1.0 +- 0.2 cm
right of its axes (`isoflop_panel.py`: `POSTER_LEGEND_FRAME_CM`, `POSTER_CBAR_PAD`). The poster sets them at scale 1.00.
The paper style is unchanged; `tests/tpp_analysis/visualizations/test_characterize_figure1_panels.py`
proves both styles draw the same data. Regenerate with `TPP_DATA_ROOT=<released data root>
FIG1_STYLE=poster FIG1_TEXT_SCALE=0.45 uv run python src/tpp_analysis/visualizations/figure1_panels.py`
and copy the two panels from `full_coverage_results/figures/` into `figures/` (pinned by SHA-256 in
`tests/poster/test_poster.py`).

## Layout

Every element is placed with `textpos` on an exact grid (cm, origin top left):

| Item | Value |
|---|---|
| Margins | 3.00 on every side of the content |
| Columns | C1 3.000-46.333, C2 48.333-91.667, C3 93.667-137.000 (43.333 wide, 2.00 gutters), symmetric about x = 70 |
| Header | title block centred between the trim and the rule; QR + NeurIPS (left) and Syracuse AI Institute over Leibniz Lab (right) groups, 15.8 wide, 6.0 tall |
| Rules | header 13.00-13.20, footer 85.60-85.80, both 3-137 |
| Body | 15.20-83.60; cards 1.00 apart, 0.80 padding; the last card of each column stretches to 83.60 |
| Footer | two lines: URL and funding (left at 3), references (centred on 70), venue, arXiv and author notes (right at 137) |
| Type scale | 110 (hero numbers), 90 (title), 44 (block titles, authors, key numbers), 30 (body), 24 (caption, footer) pt |

C1 holds the theory (TL;DR to the TPP-diversity threshold), C2 the evidence (Fig. 1a over Fig. 1c),
C3 the results (headline numbers, Theorem 1, published designs, recipe, takeaways).

## Gates

`tests/poster/test_poster.py` holds the owner gates: page size, fonts (Latin Modern or Computer Modern
only, embedded, no Type 3), exactly the three logos, the QR code, title and authors against
`CITATION.cff`, no em dashes, at most 800 words above the footer rule, the pinned figure renders set
at scale 1.00 +- 1% with all their text on the type scale, no overlapping text, no ink within 1 cm of
an edge, no overfull `\vbox`, PNGs within the NeurIPS limits and rendered from the current PDF.
`tests/poster/test_grid.py` measures the layout on the PDF: card edges on the grid, the mirror about
x = 70, column bottoms, 1.00 card gaps, rules, the type scale, card slack under 3 cm, no empty 3 x 3 cm
area in a card, no orphan lines, the 0.80 text inset, header and footer margins, and one-line block
titles, the 0.70 cm title-to-body gap, equal column ink bottoms, the footer on the grid with one
first baseline, and the hero sub-grid; `tests/tpp_analysis/visualizations/` checks that the poster
legends sit inside their axes. The figure slots hold no empty 3 x 3 cm rectangle; body lines keep a 1.30 cm pitch (1.55 at a
paragraph or item break); no paragraph ends on a one-word line; tables and the kappa row end at both text
edges or are centred; block gaps inside a card are equal within 0.15 cm (except the hero card's two
spare gaps). The hand-tuned offsets these gates guard are collected at the top of the `.tex`.
No two glyphs of different lines overlap or come within 0.10 cm (tight glyph boxes from the
embedded fonts; the M7 threshold card, frozen in `tests/poster/fixtures/`, is the planted case),
and the two figures' "Loss" labels sit 0.5 +- 0.2 cm from their ticks.
Every gate has a planted failure.
