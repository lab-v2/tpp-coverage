# WIKI300M stratum

`generate_tasks.py` writes the stratum's first 100 cells into a queue (`tasks/` and
`manifest.tsv`, longest first). The cells are 9 model sizes (6 training, 3 holdout-only) by 8
columns, in the paper's L-shaped occupancy. The columns are 5 training and 3 test TPP ratios for
CO, and 5 and 3 token budgets for NC. CO training (30 cells) is budget-matched to NC training
(30 cells), and one pooled holdout holds 40 cells. The generator asserts the design's counts, that
the CO training ratios are ill-conditioned (regime A), and the maximum token budget.

`tasks/` is the queue that was trained, 468 cells: the first 100 as generated
(`tasks/000` to `tasks/099`, byte-identical to a fresh run of the generator), then 368 added
later. The added cells densify the training arms to 66 cells each, add the 339.6M and 380.6M
holdout rows, and replicate every training cell with training seeds 43 and 44.

`arms_ready.py` exits 0 once both seed-42 training arms hold all 66 distinct cells, counting
distinct (size, tokens) pairs, not files.
