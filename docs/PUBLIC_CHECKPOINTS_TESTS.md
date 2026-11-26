# Public-checkpoint tests and reruns

[README](../README.md) · [Public checkpoints](../camera_ready/public_checkpoints/README.md)

What the tests of the public-checkpoint reanalysis check, what a rerun reproduces, and why the
committed dose-response intervals do not repeat. Script and file names are relative to
`camera_ready/public_checkpoints/`.

## What the tests check

`tests/camera_ready/test_public_checkpoints.py` refits two studies from the downloaded tables and
requires the committed rows back. It also rebuilds every margin and checks the 5-seed estimates and
the headline numbers of the folder's README. A full rerun of the E corpus reproduces
`results_E.csv` exactly.

`tests/camera_ready/test_public_checkpoints_intervals.py` and `test_public_checkpoints_amendments.py`:

- check the values the appendix prints against the committed outputs of `intervals/` and
  `amendments/`;
- run each script on the committed seeds 0-7 slice against a direct computation;
- when `results_E_seeds512.csv` and `results_X_seeds512.csv.gz` (and, for Amendment 5, the
  `results_E_win<V>.csv` tables) are in the work directory, rerun each script and require its
  committed output back.

The repository ships only seeds 0-7 of the E enumeration
(`results/results_E_seeds512_seeds0-7.csv.gz`), so without the full tables the rerun tests skip
and name the missing file. A passing suite has then checked the printed values against the
committed outputs and each script on the seeds 0-7 slice; it has not rebuilt the 512-seed
outputs. Writing the tables with `enumerate_seeds.py` (see the folder README's Reproduce section)
runs those tests too.

`tests/camera_ready/test_public_checkpoints_tables.py` rebuilds the five table bodies from the
committed CSVs byte for byte, checks the printed study counts, and reruns `per_study.py` on the
downloaded tables.

## Building the tables from a rerun

The table writers read the committed CSVs in `results/` by default; pass `--per-study`,
`--estimates`, `--regime-cells` or `--amendment5` to build the tables from a rerun.

## The dose-response intervals

The interval columns of the committed `results/doseresponse_seeds512.csv` were drawn with
Python's per-process string hash as the seed, which changes from one process to the next, so no
rerun repeats them. `estimate.py` now seeds them with a CRC32 of the cell label, and a rerun
gives slightly different bounds. The table's medians and win rates do not depend on that seed,
and the appendix takes its regime intervals from `regime_cells.csv` (200,000 draws) instead.
