# Audit tables

Statistics over the fits' CSVs, which the scripts read from `../results/` unless a path is given:

| Script | Statistic | Input |
|---|---|---|
| `analyze_split.py` | win rates per holdout split, seed-clustered bootstrap, windowed size regression | `split_rows.csv` |
| `ci_per_law.py` | per-law confidence intervals on the between-stratum off-ray difference | `bbox_strata.csv` |
| `conditioned_delta.py` | the size comparison conditioned on the design contrast | `bbox_strata.csv` |
| `n_adjusted.py` | the size comparison adjusted for subset size | `bbox_strata.csv` |
