# Running the analysis

[README](../README.md) · [Downloading](DOWNLOADING.md) · [Training](TRAINING.md) · [Figures](VISUALIZATIONS.md)

Everything here runs on CPU. Inputs are the loss tables in `extracted_losses/`
(see [DOWNLOADING](DOWNLOADING.md)); outputs are written at the repository root. Set
`--n-workers` to your core count.

## Main fits: full coverage and seed variance

Four laws x five corpora x three epoch modes (first, second, final epoch), each fitted with 30
optimizer seeds on the NC and on the CO design and scored on the shared holdouts.

```bash
uv run python src/tpp_analysis/run_analysis.py full_coverage --data-dir extracted_losses/ --n-workers "$(nproc)"
uv run python src/tpp_analysis/run_analysis.py seed_variance --data-dir extracted_losses/ --n-workers "$(nproc)"
# or both: run_analysis.py all ...
uv run python src/tpp_analysis/generate_appendix_tables.py full_coverage_results/
```

| Output | Paper item |
|---|---|
| `full_coverage_results/table_winrate.tex`, `winrate_per_combo.csv` | main win-rate table |
| `full_coverage_results/table_summary_full.tex`, `summary_table.csv` | R2/RMSE summary (main text uses the first two holdout splits) |
| `full_coverage_results/table_winrate_by_holdout_split_appendix.tex` | win rate by holdout split |
| `full_coverage_results/appendix_full_metrics.tex`, `appendix_fitted_params.tex` | per-dataset metrics and fitted parameters |
| `full_coverage_results/full_coverage_all_seeds.parquet` | every fit, one row per corpus x law x mode x design x seed |
| `seed_variance_results/` | coverage-fraction data behind the convergence panels |

Fits are cached per combination (`fullcov_*.parquet`, `seed_variance_*.parquet`); a cached
combination is re-rendered, not refitted. `--clear-cache` refits from scratch. The paper's
tables are hand-edited from these files (rows dropped, parameters rounded to three decimals),
so the generated `.tex` is the source of the numbers, not a drop-in copy.

## BF16 replication and BigTPP preliminary

```bash
uv run python src/tpp_analysis/download_and_extract.py --include-special --include-prelim   # if not done yet
uv run python src/tpp_analysis/run_special_analysis.py --data-dir extracted_losses/ --n-workers "$(nproc)"
uv run python src/tpp_analysis/generate_appendix_tables.py special_results/ --special
uv run python src/tpp_analysis/run_prelim_bigtpp.py --data-dir extracted_losses/ --n-workers "$(nproc)"
```

Outputs: `special_results/table_summary_bf16.tex`, `table_winrate_bf16.tex`,
`table_winrate_by_holdout_split_bf16.tex`, `appendix_*_bf16.tex`; and
`prelim_bigtpp_results/table_winrate_bigtpp_prelim.tex` (Chinchilla only, Wilson interval).

## Regime-A subset enumeration

For every corpus, law and epoch mode, this enumerates budget-matched CO and NC training subsets,
classifies each by the paper's V_K >= tau_K criterion and records which one extrapolates
better. It is the expensive step: about an hour per seed on 480 cores, 22 seeds.

```bash
bash src/tpp_analysis/dispatch_seeds.sh 0 21 subset_bbox_seeds "$(nproc)"   # run from the repository root
uv run python src/tpp_analysis/subset_analysis/aggregate_seed_runs.py \
    --seed-dir subset_bbox_seeds/ --out-dir subset_bbox_seeds/aggregated
uv run python src/tpp_analysis/subset_analysis/per_law_winrate_ci.py \
    --seed-dir subset_bbox_seeds/ --out-dir subset_bbox_seeds/aggregated --mode first
```

`dispatch_seeds.sh` calls `python`, so run it inside the environment (`source .venv/bin/activate`
or `uv run bash ...`). The per-seed CSVs of the paper are on HuggingFace
(`subset_bbox_seeds/`, 3.9 GB), so the two aggregation steps can run without the enumeration.
Outputs: `subset_bbox_seeds/aggregated/per_law_ci.csv` and `per_dataset_ci.csv` (the Regime-A
table) and the kappa_target sweep figures.

## Camera-ready analyses

- [`camera_ready/grid_classification/`](../camera_ready/grid_classification/README.md):
  the V_K >= tau_K classification of 24 published scaling-law designs.
- [`camera_ready/public_checkpoints/`](../camera_ready/public_checkpoints/README.md):
  the budget-matched CO-versus-NC test on ten public training-loss releases, with the
  scale-sensitivity analysis.
