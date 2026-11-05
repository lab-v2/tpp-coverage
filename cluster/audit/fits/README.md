# Audit fits

Each script fits the scaling laws on budget-matched CO and NC arms of the WIKI300M runs and writes
one CSV (first argument: fit seeds; second: restarts). `wiki300m_paths.py` puts the paper's code
on `sys.path` and reads the run directory from `WIKI300M_DIR`.

| Script | Writes | What |
|---|---|---|
| `interim_experiment.py` | `rows.csv` | seed-paired CO/NC holdout RMSE, all rows and per size stratum |
| `split_holdout.py` | `split_rows.csv` | the same fits scored on the pooled, on-ray and off-ray holdouts |
| `rerun_fixed.py` | `rows_fixed.csv` | extreme-preserving arm balancing and a convergence filter on training SSE |
| `droppo_fix.py` | `droppo_fixed.csv` | Droppo-Elibol with its N_c, D_c bounds scaled to the data |
| `droppo_published_check.py` | `droppo_published_check.csv` | whether that rescaling moves the paper's own Wikipedia fit (it does not) |
| `bbox_strata.py` | `bbox_strata.csv` | per-stratum comparisons with the paper's bounding-box NC selection, which holds the design contrast fixed |
| `cell_bootstrap.py` | (printed) | resamples trained cells within each size row and refits |
| `weighted_bootstrap.py` | (printed) | a random-weight (Dirichlet) bootstrap over trained cells, every cell present in every replicate |
