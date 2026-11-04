# Audit tables

Statistics over the fits' CSVs, which they read from `../results/` unless a path is given:
`analyze_split.py` (win rates per holdout split, seed-clustered bootstrap, windowed size
regression, from `split_rows.csv`), `ci_per_law.py` (per-law confidence intervals on the
between-stratum off-ray difference), `conditioned_delta.py` (the size comparison conditioned on
the design contrast) and `n_adjusted.py` (the size comparison adjusted for subset size), the last
three from `bbox_strata.csv`.
