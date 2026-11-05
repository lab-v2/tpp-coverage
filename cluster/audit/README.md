# Audit re-analyses of the WIKI300M runs

These compare CO and NC designs on the stratum's runs, per model-size stratum, with the paper's
own fitters and design machinery (`src/tpp_analysis`). They were run on 147 runs, an interim set
taken before the queue finished, so every number in `results/` and in the tables computed from it
is interim, and none of them is in the paper. For final numbers, rerun the fits with
`WIKI300M_DIR` pointing at the metrics of the whole queue.

| Folder | What |
|---|---|
| `fits/` | the fitting runs; each reads the runs' training metrics from `$WIKI300M_DIR/shared/training_metrics/` and writes its CSV to `$WIKI300M_DIR` |
| `tables/` | win rates, confidence intervals and size contrasts computed from those CSVs; they read `results/` by default |
| `results/` | the CSVs the fits wrote |
