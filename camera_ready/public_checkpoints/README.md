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
| Median ratio of identifiability margins, (V_K/tau_K)_NC / (V_K/tau_K)_CO, below 1B / at 1B and above | 5.8 [5.4, 6.9] / 3.7 [2.1, 5.5] | `vk_ratio.csv`, `vk_ratio_boot.csv` (E2) |
| Ratio of those two medians, 1B and above over below 1B | 0.64 [0.20, 1.32] | `ratio_of_medians_boot.csv` |

Win rates by regime are in `results/doseresponse_seeds512.csv`: where the criterion puts CO in
Regime A and NC in Regime B, NC wins 59%, 61% and 76% of pairs at kappa_target = 1e4, 1e5
and 1e6; in the large same-regime cells it wins 44-47%. `regime_by_band_ci.csv` repeats these
arms inside each size band. Classified by its collinear design alone (`co_only_regime.csv`),
every pair is in Regime A at kappa_target = 100, where NC wins 47% (E) and 50% (X); at 1e4,
1e5 and 1e6 it wins 58%, 61% and 76% (E) and 58%, 56% and 76% (X).

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
```

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
| `results/` | committed reference outputs (5-seed fits and margins of the E corpus, seeds 0-7 of the 512-seed enumeration, screening logs, estimates, and the outputs of the four interval scripts) |

`tests/test_public_checkpoints.py` refits two studies from the downloaded tables and requires
the committed rows back, rebuilds every margin, and checks the 5-seed estimates and the
headline numbers above. A full rerun of the E corpus reproduces `results_E.csv` exactly.
`tests/test_public_checkpoints_intervals.py` checks the values the appendix prints against the
committed interval tables, runs each interval script on the committed seeds 0-7 slice against
a direct computation, and, when `results_E_seeds512.csv` and `results_X_seeds512.csv.gz` are in
the work directory, reruns each script and requires its committed output back. The repository
does not ship those two tables, so without them the rerun tests skip and name the file.

Note: `estimate.py` now seeds the median intervals of the dose-response table with a CRC32 of
the cell label. The committed table was computed with Python's per-process string hash, so
its interval columns (not its medians or win rates) differ slightly from a rerun.
