# poster

The NeurIPS 2026 poster, A0 landscape (118.9 x 84.1 cm, inside the conference limit of 140 x 90 cm),
built with the [gemini](https://github.com/anishathalye/gemini) beamerposter theme (vendored in
`theme/` with its MIT licence; one local change loads Raleway by file name). Text is Lato with
Raleway headings, mathematics is Latin Modern Math (unicode-math, x-height matched to Lato), and every font is embedded (no Type 3).

`make` converts the SVG logos in `../images/` to PDF, draws the QR code of the project page, and
builds with `latexmk -xelatex` into `build/`; only `tpp_neurips2026_poster.pdf` is kept next to the
source. `make png` renders the virtual poster (`build/tpp_neurips2026_poster_virtual.png`, 4073 x
2880 px) and the thumbnail (`build/tpp_neurips2026_poster_thumbnail.png`, 320 x 226 px) within the
NeurIPS limits. Needs `latexmk`, `xelatex`, poppler's `pdftoppm` and `uv` (for `cairosvg` and `segno`).

Two figures are drawn in TikZ in the source: the hero figure, the two designs in the $(N, D)$ plane with
their held-out runs (from the grids of the paper's Appendix F.1), and the threshold strip, $V_K / \tau_K$ of
the published designs (from the paper's Table 11). `figures/` holds the two data panels of the paper's
Figure 1 (b) and (c), Kaplan law on RedPajama, first epoch, redrawn in the poster style by the paper's own
plotting code: `make figures TPP_DATA_ROOT=<root>`, where `<root>` holds the released `extracted_losses/`,
`full_coverage_results/` and `seed_variance_results/` (`download_and_extract.py --results-only`). The
QR code of the project page sits in the header, under the NeurIPS logo. `tests/poster/` checks the page size, the fonts,
the three logos, the QR code, the title and authors against `CITATION.cff`, the absence of em dashes, at
most 750 words, LaTeX math fonts only (Lato, Raleway, CM/LM, in the poster and its figures), a header under 17% of the height, column bottoms within 1 cm and tick labels of at least
24 pt.
