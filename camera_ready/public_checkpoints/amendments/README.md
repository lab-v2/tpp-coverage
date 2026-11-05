# Amendment and scale estimates of the public-checkpoint appendix

[Reanalysis README](../README.md)

The scripts behind the appendix's *Protocol history* and *Scale* paragraphs and its
Amendment-5 tables. They import the reanalysis modules one folder up (`estimate.py`,
`cluster_bootstrap.py`, `conditional.py`) and read the 512-seed tables from the work
directory. Their committed outputs are in [`../results/`](../results).

| Script | Output | What the appendix prints from it |
|---|---|---|
| `three_study_pooled.py` | `three_study_pooled.csv` | the estimate before Amendment 4, on Farseer, Chinchilla and Gadre et al. (753 runs): pooled 8% [-1%, 16%] at 5 seeds; at 512 seeds pooled 6% [-4%, 15%] and within a cell 33% [17%, 43%] |
| `amendment5_1b.py` | `amendment5_1b.csv` | every specification Amendment 5 fixes for windows centred at 1B and above: the estimator table, the window-construction table, and the Scale paragraph's second subset (36% [2%, 54%]), 5-size windows under the moving-block interval (26% [2%, 47%]), smooth slope (29% [5%, 50%] at 1.8B, one-sided floor 9%) and attenuation bounds (112%, 77%) |
| `interaction_check.py` | `interaction_check.csv` | the size interaction's centre (630M on the five, 152M on the ten) and the effect there (38%) and one decade above (19% [-19%, 62%] at 6.3B) |

`amendment5_1b.py` also reads the 128-seed fits of the alternative window constructions,
`results_E_win{V1,V2,V3,V4,V7}.csv`, which `enumerate_seeds.py --construction V --seeds 128`
writes (`--windows-dir none` skips them). It takes about 20 minutes; the other two take
seconds.

Run from the folder above, for example `uv run python amendments/three_study_pooled.py`.
