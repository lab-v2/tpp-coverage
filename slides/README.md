# slides

The five-minute NeurIPS 2026 talk (Beamer, 16:9). `make` converts the SVG logos in `../images/` to
PDF, draws the QR code of the project page, and builds into `build/`; only
`tpp_neurips2026_slides.pdf` is kept next to the source. Needs `latexmk` and `uv` (for `cairosvg`
and `segno`). `figures/` holds the one paper figure the deck shows,
`01_convergence_Red_Pajama_Kaplan_First.pdf` (holdout $R^2$ against the share of TPP levels, Kaplan
law on RedPajama), with its text drawn as outlines. `make check` builds the deck and, given
`PDF_WIDOWS=<checker.py>`, runs that layout checker on it; the repository ships no checker.
