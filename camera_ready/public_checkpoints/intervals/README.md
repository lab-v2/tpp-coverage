# Intervals of the public-checkpoint appendix

[Reanalysis README](../README.md)

The four scripts behind the intervals the appendix prints from the 512-seed enumeration. They
import the reanalysis modules one folder up (`conditional.py`, `estimate.py`, `vk_ratio.py`)
and read `results_E_seeds512.csv` and `results_X_seeds512.csv.gz` from the work directory.
Their committed outputs are in [`../results/`](../results).

| Script | Output | What the appendix prints from it |
|---|---|---|
| `vk_ratio_boot.py` | `vk_ratio_boot.csv`, `vk_ratio_boot_ge1B_windows.csv` | median margin ratio 5.8 [5.4, 6.9] below 1B and 3.7 [2.1, 5.5] at 1B and above; below 1 in 3 of 13 windows |
| `ratio_of_medians_boot.py` | `ratio_of_medians_boot.csv` | ratio of the two medians, 0.64 [0.20, 1.32] |
| `regime_by_band_ci.py` | `regime_by_band_ci.csv` | the regime test's arms inside each size band |
| `co_only_regime.py` | `co_only_regime.csv` | NC win rates with the CO design alone classified: 47% / 50% at kappa_target 100 |

Run from the folder above, for example `uv run python intervals/vk_ratio_boot.py`.
