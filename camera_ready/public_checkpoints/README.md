# Reanalysis of public training-loss releases

[README](../../README.md)

Does the paper's criterion predict extrapolation error on scaling-law data we did not
generate, and does the effect hold at large model sizes? Ten published studies release
per-run losses with recoverable (N, D) and a two-dimensional core of model sizes and TPP
ratios: Farseer, the Chinchilla isoFLOP sweep, Gemstones (annealed and constant-LR), Gadre et
al., the OLMo ladder, DataDecide, MAD, GPT-3 curves and OPT trajectories. The first five give
984 fully trained runs (the E corpus); the last five add 11,547 mid-training checkpoints,
each a further (N, D) point (the X corpus: 12,531 points, the 984 included). Within a study and a window of adjacent model sizes, we carve a collinear and a
non-collinear training subset of equal size, classify both with V_K and tau_K, fit the law to
each, and score both on the same held-out runs off their TPP range. The protocol was
pre-registered before any loss was read: [`CRITERIA.md`](CRITERIA.md).

## Headline numbers

All from `results/` (512 seeds; 95% bootstrap intervals clustered by window):

| Quantity | Value | Row |
|---|---|---|
| Extra off-ray error per tenfold V_K shortfall, within a study | 40% [24, 52] | E, `cell FE (PRIMARY)` |
| Same, across studies | 12% [5, 18] | E, `no fixed effect (pooled)` |
| Same, DataDecide (25 corpora) | 19% [4, 35] | X, `suite = datadecide` |
| Same, models 100M-1B / 1B and above | 42% [20, 58] / 28% [-9, 41] | E, `stratum ...` |
| Change in the effect per tenfold N | -14% [-40, 18] | E, `b1 Delta x logN` |
| Median ratio of identifiability margins, (V_K/tau_K)_NC / (V_K/tau_K)_CO, below 1B / at 1B and above | 5.8 [5.4, 6.9] / 3.7 [2.1, 5.6] | `vk_ratio.csv`, `vk_ratio_seeds.csv` (pooled) |
| Ratio of those two medians, 1B and above over below 1B | 0.64 [0.2, 1.3] | `vk_ratio_seeds.csv` (pooled) |

The two ratio intervals pool 30 seeds' window-cluster draws (300,000); one stream of 10,000
draws (`vk_ratio_boot.csv`, `ratio_of_medians_boot.csv`) gives [2.1, 5.5] and [0.20, 1.32].

Win rates by regime are in `results/regime_cells.csv` (intervals at 200,000 draws) and
`results/doseresponse_seeds512.csv` (2,000): where the criterion puts CO in Regime A and NC in
Regime B, NC wins 59%, 61% and 76% of pairs at kappa_target = 1e4, 1e5 and 1e6; in the large
same-regime cells it wins 44-47%. The `band` rows of `regime_cells.csv` repeat these arms
inside each size band. Classified by its collinear design alone (the `coonly` rows), every
pair is in Regime A at kappa_target = 100, where NC wins 47% (E) and 50% (X); at 1e4, 1e5 and
1e6 it wins 58%, 61% and 76% (E) and 58%, 56% and 76% (X).

## Reproduce

All commands run from this folder inside the environment (`uv run python ...` from the
repository root works too). Outputs go to `work/` (set `TPP_PC_WORK` to move it).

```bash
cd camera_ready/public_checkpoints
uv run python get_runs.py                   # run tables from HuggingFace, or: fetch_data.py --expanded
for C in E X; do                            # 5-seed fits, ~1 min (E) on 400 cores
  uv run python analysis.py --runs runs_expanded.csv --corpus $C \
      --screening screening_$C.csv --out results_$C.csv --workers "$(nproc)"
  uv run python margins.py --runs runs_expanded.csv --corpus $C \
      --results results_$C.csv --out margins_$C.csv
done
uv run python conditional.py --results results_E.csv --margins margins_E.csv --prefix conditionalE
# 512-seed enumeration: ~16 min (E) and ~2 h (X) on 480 cores
uv run python enumerate_seeds.py --corpus E --runs runs_expanded.csv --out results_E_seeds512.csv
uv run python enumerate_seeds.py --corpus X --runs runs_expanded.csv --out results_X_seeds512.csv.gz
uv run python estimate.py                   # headline table; --skip-x without the X corpus
uv run python vk_ratio.py
uv run python intervals/vk_ratio_boot.py --workers "$(nproc)"      # intervals of the median ratio, ~15 s
uv run python intervals/ratio_of_medians_boot.py                    # ratio of the two medians, ~35 s
uv run python intervals/regime_by_band_ci.py --workers "$(nproc)"  # regime arms per size band, ~25 s
uv run python intervals/co_only_regime.py --workers "$(nproc)"     # regimes of the CO design alone, ~40 s
uv run python intervals/vk_ratio_seeds.py --workers "$(nproc)"     # the ratio intervals over 30 seeds, ~1 min
uv run python intervals/regime_cells.py --workers "$(nproc)"       # every regime cell at 200,000 draws, ~35 min
# alternative window constructions for Amendment 5, 128 seeds each
for V in V1 V2 V3 V4 V7; do
  uv run python enumerate_seeds.py --corpus E --runs runs_expanded.csv --construction $V \
      --seeds 128 --out results_E_win$V.csv
done
uv run python amendments/three_study_pooled.py                     # the estimate before Amendment 4
uv run python amendments/amendment5_1b.py                          # Amendment 5 at 1B and above, ~20 min
uv run python amendments/interaction_check.py                      # the size interaction at given N
uv run python tables/per_study.py                                  # the study counts
uv run python tables/make_table.py                                 # the five table bodies, into work/tables/
uv run python tables/tables_tex.py
uv run python tables/tables_a5_tex.py
```

The times are on 480 cores. The table writers read the committed CSVs in `results/` by
default; pass `--per-study`, `--estimates`, `--regime-cells` or `--amendment5` to build the
tables from a rerun.

`get_runs.py` downloads the two normalised tables published with the paper
([`leibnitz-lab/public_checkpoint_reanalysis`](https://huggingface.co/datasets/leibnitz-lab/public_checkpoint_reanalysis));
`fetch_data.py` rebuilds them from the upstream releases (`--expanded` for the ten-study table).

| File | Role |
|---|---|
| `analysis.py` | screening, size windows, holdout, CO/NC carving and fitting (laws, fitter, carving and V_K come from `src/tpp_analysis/`) |
| `margins.py` | identifiability margins m = log10(V_K / tau_K) of every design, checked against the fits |
| `conditional.py` | the pre-registered regressions and dose-response tables at 5 seeds |
| `enumerate_seeds.py` | the same fits over 512 carving seeds and alternative window constructions |
| `cluster_bootstrap.py` | fast window-clustered bootstrap, asserted equal to `conditional.py` |
| `estimate.py` | every estimate at 5 and at 512 seeds |
| `vk_ratio.py` | the median ratio of identifiability margins by size band |
| `intervals/vk_ratio_boot.py` | its 95% intervals (E2 and two comparison schemes) and the median per window at 1B and above |
| `intervals/ratio_of_medians_boot.py` | the ratio of the two band medians, 1B and above over below 1B, with its interval |
| `intervals/regime_by_band_ci.py` | the regime test's confirmatory and same-regime arms inside each size band (A3.4) |
| `intervals/co_only_regime.py` | the regime test with each pair classified by its collinear design alone |
| `intervals/vk_ratio_seeds.py` | the margin-ratio intervals over 30 seeds, and the pooled interval the appendix prints |
| `intervals/median_ci.py` | the window-cluster interval of a median at 200,000 draws (a module) |
| `intervals/regime_cells.py` | every regime cell the appendix prints, at 200,000 draws |
| `amendments/three_study_pooled.py` | the primary and the pooled slope on the three studies screened first, at 5 and 512 seeds |
| `amendments/amendment5_1b.py` | every specification Amendment 5 fixes for windows centred at 1B and above |
| `amendments/interaction_check.py` | the size interaction's centre, and the effect there and one decade above |
| `tables/per_study.py` | points, sizes and ranges per study after the screen |
| `tables/make_table.py`, `tables/tables_tex.py`, `tables/tables_a5_tex.py` | the bodies of the appendix's five tables |
| `results/` | committed reference outputs (5-seed fits and margins of the E corpus, seeds 0-7 of the 512-seed enumeration, screening logs, estimates, the outputs of the scripts above, and the table bodies in `results/tables/`) |

Each folder has a README with what the appendix prints from each output:
[`intervals/`](intervals/README.md), [`amendments/`](amendments/README.md),
[`tables/`](tables/README.md).

## Appendix item to script

| Appendix item | Script | Output |
|---|---|---|
| Studies paragraph and study table: 984 and 12,531 points, sizes, N and TPP ranges | `tables/per_study.py`, `tables/make_table.py` | `per_study.csv`, `tables/public_checkpoints_studies.tex` |
| Protocol history: 753 and 231 runs | `tables/per_study.py` | `per_study.csv` |
| Protocol history: the three-study estimates | `amendments/three_study_pooled.py` | `three_study_pooled.csv` |
| Sources and licenses: 14,065, 2,215 and 1,462 rows | the HuggingFace tables (`get_runs.py`, built by `fetch_data.py`) | `runs_expanded.csv`, `runs_normalised.csv` |
| Test paragraph: windows, holdout, designs, margins, bootstrap depths | `analysis.py`, `enumerate_seeds.py`, `estimate.py` | [`CRITERIA.md`](CRITERIA.md) |
| Results paragraph and results table, per-study slopes, refits without one study | `estimate.py`, `tables/tables_tex.py` | `estimates_seeds512.csv`, `tables/public_checkpoints_results.tex` |
| Regimes paragraph and regime table | `intervals/regime_cells.py`, `tables/tables_tex.py` | `regime_cells.csv`, `tables/public_checkpoints_regimes.tex` |
| Scale paragraph: bands, size interaction | `estimate.py`, `amendments/interaction_check.py` | `estimates_seeds512.csv`, `interaction_check.csv` |
| Scale paragraph and the two Amendment-5 tables | `amendments/amendment5_1b.py`, `tables/tables_a5_tex.py` | `amendment5_1b.csv`, `tables/public_checkpoints_1b.tex`, `tables/public_checkpoints_windows.tex` |
| The margin ratio and its intervals | `vk_ratio.py`, `intervals/vk_ratio_seeds.py`, `intervals/vk_ratio_boot.py` | `vk_ratio.csv`, `vk_ratio_seeds.csv`, `vk_ratio_boot_ge1B_windows.csv` |

`tests/camera_ready/test_public_checkpoints.py` refits two studies from the downloaded tables and requires
the committed rows back, rebuilds every margin, and checks the 5-seed estimates and the
headline numbers above. A full rerun of the E corpus reproduces `results_E.csv` exactly.
`tests/camera_ready/test_public_checkpoints_intervals.py` and `test_public_checkpoints_amendments.py`
check the values the appendix prints against the committed outputs of `intervals/` and
`amendments/`, run each script on the committed seeds 0-7 slice against a direct computation,
and, when `results_E_seeds512.csv` and `results_X_seeds512.csv.gz` (and, for Amendment 5, the
`results_E_win<V>.csv` tables) are in the work directory, rerun each script and require its
committed output back. The repository ships only seeds 0-7 of the E enumeration
(`results/results_E_seeds512_seeds0-7.csv.gz`), so without the full tables the rerun tests skip
and name the missing file. A passing suite has then checked the printed values against the
committed outputs and each script on the seeds 0-7 slice; it has not rebuilt the 512-seed
outputs. Writing the tables with `enumerate_seeds.py` (see Reproduce) runs those tests too.
`tests/camera_ready/test_public_checkpoints_tables.py` rebuilds the five table bodies from the
committed CSVs byte for byte, checks the printed study counts, and reruns `per_study.py` on the
downloaded tables.

The interval columns of the committed `results/doseresponse_seeds512.csv` were drawn with
Python's per-process string hash as the seed, which changes from one process to the next, so no
rerun repeats them. `estimate.py` now seeds them with a CRC32 of the cell label, and a rerun
gives slightly different bounds. The table's medians and win rates do not depend on that seed,
and the appendix takes its regime intervals from `regime_cells.csv` (200,000 draws) instead.
