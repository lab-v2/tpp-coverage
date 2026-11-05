# Tables of the public-checkpoint appendix

[Reanalysis README](../README.md)

The appendix's five tables are written from committed outputs by these scripts, so no cell is
typed by hand. The bodies they write are committed in
[`../results/tables/`](../results/tables) and are the files the paper inputs.

| Script | Reads | Writes |
|---|---|---|
| `per_study.py` | the HuggingFace run table `runs_expanded.csv` (via `get_runs.py`) | `per_study.csv`: points, sizes, N and TPP ranges and windows per study, after the protocol's screen (984 final-loss points, 12,531 in all) |
| `make_table.py` | `per_study.csv` | `public_checkpoints_studies.tex`, the ten releases |
| `tables_tex.py` | `estimates_seeds512.csv`, `regime_cells.csv` | `public_checkpoints_results.tex` (the margin regression) and `public_checkpoints_regimes.tex` (the regime test) |
| `tables_a5_tex.py` | `amendment5_1b.csv` | `public_checkpoints_1b.tex` (Amendment 5's estimators) and `public_checkpoints_windows.tex` (its window constructions) |

The three table writers read the committed CSVs in `../results/` by default and write into
`tables/` in the work directory (`--out-dir` moves it). `per_study.py` reads and writes in the
work directory.

Run from the folder above, for example `uv run python tables/tables_tex.py`.
