# Intervals of the public-checkpoint appendix

[Reanalysis README](../README.md)

The scripts behind the intervals the appendix prints from the 512-seed enumeration. They
import the reanalysis modules one folder up (`conditional.py`, `estimate.py`, `vk_ratio.py`)
and read `results_E_seeds512.csv` and `results_X_seeds512.csv.gz` from the work directory.
Their committed outputs are in [`../results/`](../results).

| Script | Output | What the appendix prints from it |
|---|---|---|
| `vk_ratio_boot.py` | `vk_ratio_boot.csv`, `vk_ratio_boot_ge1B_windows.csv` | the median margin ratio per band, 5.8 below 1B and 3.7 at 1B and above; below 1 in 3 of 13 windows. Its one stream of 10,000 draws gives [5.4, 6.9] and [2.1, 5.5] |
| `ratio_of_medians_boot.py` | `ratio_of_medians_boot.csv` | the ratio of the two medians, 0.64; one stream gives [0.20, 1.32] |
| `vk_ratio_seeds.py` | `vk_ratio_seeds.csv` | the printed intervals of those three statistics, from 30 seeds' draws pooled (300,000): [5.4, 6.9], [2.1, 5.6] and [0.2, 1.3] |
| `median_ci.py` | (module) | the 200,000-draw window-cluster interval of a median: the loop of `estimate._median_job`, with more draws |
| `regime_cells.py` | `regime_cells.csv` | every regime cell the appendix prints, at 200,000 draws: the regime table; each pair classified by its collinear subset alone (NC wins 47% / 50% at kappa_target 100); the same rule with each law's own exponent (54%, 62%, 67% / 57%, 54%, 67%); and the regime test inside each size band |
| `regime_by_band_ci.py` | `regime_by_band_ci.csv` | the band cells at the protocol's 2,000 draws |
| `co_only_regime.py` | `co_only_regime.csv` | the collinear-only cells at the protocol's 2,000 draws |

The two 2,000-draw tables hold the same pairs, win rates and medians as `regime_cells.csv`;
the appendix prints the 200,000-draw intervals. One verdict differs: in windows centred at 1B
and above, the five releases' same-regime cell at kappa_target 100 excludes zero at 2,000
draws and only just contains it at 200,000. `regime_cells.py` repeats every interval with 10
seeds, and no verdict on zero moves across them.

Run from the folder above, for example `uv run python intervals/vk_ratio_boot.py`.
