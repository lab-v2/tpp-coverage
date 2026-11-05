# Known behaviours

[README](../README.md) · [Training](TRAINING.md) · [Downloading](DOWNLOADING.md) · [Analysis](ANALYSIS.md)

Each behaviour below changes a reproduced or reused number when it is missed.

## Padding in the loss

Each article is truncated or padded to 256 tokens, and the released runs averaged the
cross-entropy over every position, the padded ones included. A model learns early to predict
the padding token after an article ends, so each reported loss is close to (1 - p) times the
loss on real tokens, where p is the corpus's share of padded positions. The released
checkpoints have not been re-evaluated with padding masked.

- **Changes:** absolute losses, which sit below the real-token loss and differ between corpora
  by their padding share, so they are not comparable across corpora; and each corpus's fitted
  E, A and B, which shrink by a common factor.
- **Leaves alone:** the fitted exponents, nearly; the win rates and the conditioning
  comparisons, because the CO and NC designs are always fitted to the same corpus, so the
  factor cancels; and the theorems.
- **What to do:** the trainer now masks padding out of the loss by default. Pass
  `--score-padding` to reproduce, check or extend a released run, and never fit masked and
  scored runs in one table.

## BF16 precision

`--use-bf16`, which `generate_jobs.py --precision bf16` adds, calls `model.bfloat16()`: every
parameter and buffer becomes BF16, so the gradients and the AdamW states are BF16 as well, with
no FP32 master copy and no autocast. Only the logits are cast to FP32 for the loss. This is the
paper's pure-BF16 replication.

- **Changes:** the BF16 replication only. A run under `torch.autocast`, which keeps FP32
  weights, trains a different model and does not reproduce it.
- **Leaves alone:** the five-corpus grid and the BigTPP runs, which train in FP32, and the
  analysis.
- **What to do:** reproduce the replication with `--use-bf16`, and keep runs of any other
  precision scheme out of its table (a run whose name contains `BF16` joins it, see
  [Run names](#run-names)).

## GPU count and the learning-rate schedule

Every run trains on 4 GPUs with 96 sequences per GPU, and `DistributedSampler` splits each
epoch across the GPUs. The GPU count therefore sets the global batch (384 sequences) and the
number of steps per epoch, and the 3% warmup and the cosine decay are laid over that step
count.

- **Changes:** a run on any other GPU count: its global batch, step count and schedule differ,
  so it trains a different model and reproduces neither the released checkpoints nor their
  losses.
- **Leaves alone:** the analysis, which runs on CPU.
- **What to do:** train every run on 4 GPUs. On GPUs with less memory, `--grad-accum-steps K`
  splits each GPU's 96 sequences into K micro-batches and keeps the optimizer step, the
  schedule and the step count. Its gradient is the full batch's up to rounding when every
  position is scored, as with `--score-padding`; with the padding mask it weights examples
  rather than tokens, so it approximates the full batch's.

## Training seed

Every run of the paper (the five-corpus grid, the BF16 replication and the BigTPP
preliminary) is trained once, with seed 42 for the initialization and the data order. The
analysis seeds are restarts of the fitter and draws of subsets on these fixed losses.

- **Changes:** the reading of the intervals: the 30-seed intervals of the summary and win-rate
  tables measure curve-fitting variance, not training variance.
- **Leaves alone:** reproduction, since a retrained run with seed 42 and `--score-padding` on
  the same setup repeats the released one.
- **What to do:** for training variance, retrain with `--seed S`. The
  [cluster stratum](../cluster/README.md), which is outside the paper, repeats each of its
  training cells with seeds 43 and 44.

## Released runs

The released loss tables, which the paper's main, BF16 and BigTPP fits used, hold the full
design for Wikipedia, Cosmopedia, RedPajama and peS2o (324 runs each) and for BigTPP (156
runs). Two are incomplete:

- C4 has 99% of its runs (321 of 324): the NC holdout runs at D = 375M of the 54.57M, 60.88M
  and 64.04M targets are missing.
- The BF16 replication has 93% of its runs (302 of 324): it lacks 14 CO holdout runs of the
  four holdout-only sizes at k = 6.2 to 7, the CO training runs of the 19M target at k = 1.9,
  2, 3 and 4, and the NC training runs of the 10M and 19M targets at D = 150M and 225M.

The paper's C4 and BF16 numbers come from these subsets.

- **Changes:** a retrain from `generate_jobs.py`, which writes the whole design: its C4 and
  BF16 tables hold runs the paper's fits never saw, so their fits and holdout scores differ
  from the paper's.
- **Leaves alone:** the pairing, since within a corpus the CO and NC fits are scored on the
  same holdout runs.
- **What to do:** drop these runs from a retrained table before fitting, or fit the released
  tables. `tests/tpp_analysis/test_release_match.py` compares the generated names with the
  released ones.

## Model sizes

`--model-size` is a target, not a parameter count. The paper's 14 targets map to fixed shapes
([TRAINING](TRAINING.md#3-training-one-run)), whose realized counts are up to 8.8% above the
target (the 70.36M target trains 76.52M parameters). Any other target goes through a search
that matches on `legacy_param_count`, an older count formula kept so that the released shapes
are found again, rather than on the true count.

- **Changes:** the size a new target trains: `--model-size 150000000`, for example, trains
  169,809,408 parameters, 13% more than asked for.
- **Leaves alone:** the analysis, which reads the realized count from each run's name
  (`m<N>`) and loss table.
- **What to do:** read the realized N from the run name before using a run. For a new size,
  pass the legacy count of the intended shape, as `cluster/wiki300m/generate_tasks.py` does
  and checks, or give the shape itself with `--shape`.

## Run names

`get_results.py` takes everything about a run from its file name: a `BIGTPP` or `BF16` marker
sends it to the BigTPP or the BF16 table, the `_<dataset>_m<N>_` tag the trainer appends gives
its corpus and realized N, and the `COLINEAR_HOLDOUT`, `HOLDOUT`, `COLINEAR` or `NxD_GRID`
token gives its role.

- **Changes:** a run whose name has none of the role tokens is counted as NC training, and a
  BF16 or BigTPP run without its marker joins the main Wikipedia table, both without a warning.
- **Leaves alone:** runs named by `generate_jobs.py`, which uses the released prefixes. A run
  with a marker is extracted only with `--include-special` (BF16) or `--include-prelim`
  (BigTPP).
- **What to do:** keep the generator's `--output-prefix` names, or rename runs to them before
  extraction.

## Cached fits

`run_analysis.py` stores the fits of each combination in its output folder
(`fullcov_*.parquet`, `seed_variance_*.parquet`) and re-renders a stored combination instead of
refitting it. A stored file is found by its name alone, not by the loss tables it was fitted
on, and `download_and_extract.py --results-only` downloads the released ones.

- **Changes:** a rerun after the loss tables change (runs added or removed), which without
  `--clear-cache` prints the old fits. The quick start re-renders the released fits.
- **Leaves alone:** a rerun on unchanged tables, which gives the same output either way.
- **What to do:** pass `--clear-cache` whenever the inputs change. A full refit reproduces the
  released results bit for bit (about 20 minutes on 50 cores).

## Scale pairs left at a start draw

`_fit_generic` runs L-BFGS-B on the parameters in native units, where the bounds of a scale
coefficient span eleven or twelve orders of magnitude (Kaplan's and Droppo-Elibol's N_c and D_c
in [1e3, 1e14], Chinchilla's A and B in [1e-2, 1e10]). In 183 of the 900 Kaplan fits and 510 of
the 900 Droppo-Elibol fits of `full_coverage_all_seeds.parquet`, and in 1 Kaplan and 14
Droppo-Elibol fits of the 180 BF16 fits of each law, the optimizer left the scale pair within a
relative 1e-6 of one of its 100 start draws and moved only the other parameters. Chinchilla's A
and B move: only 3 of its 900 fits match a start draw exactly.

- **Changes:** any reuse of a fitted Kaplan or Droppo-Elibol scale pair as an estimate: in these
  fits it is the best of the start draws, not an optimum of the loss.
- **Leaves alone:** the holdout scores the paper compares: these fits' mean holdout R^2 is within
  0.07 of the other fits' (higher for Droppo-Elibol, lower for Kaplan: 0.69 against 0.75 on CO).
- **What to do:** to find these fits, replay each fit's start draws: `run_analysis.py` seeds fit
  s with `base_seed + s` (base 0), and `full_coverage_ci.py`, which wrote the BF16 fits, with
  `137 s`. To optimize the scale pair itself, fit it in log units, as `fit_skaling` does.

## Driver and hardware drift

The released runs used PyTorch 2.5 with CUDA 12.4 on GH200 GPUs. Training is bitwise
reproducible within a fixed CUDA driver version, which the paper checked by repeated runs on
NVIDIA driver 590.48; after a driver update, the training loss drifted by O(1e-3) over three
epochs.

- **Changes:** the check of a retrained run against the released one.
  `verify_reproducibility.py` compares losses at an absolute tolerance of 1e-5 and weights at
  1e-6 by default, which a faithful run on another driver or GPU can fail.
- **Leaves alone:** the paper's conclusions: a uniform shift of the losses moves the fitted
  irreducible loss of the CO and NC fits alike, and the drift is about two orders of magnitude
  below the NC-versus-CO holdout RMSE gap.
- **What to do:** on hardware other than the reference, compare with `--atol 1e-2`, and train
  with `--score-padding` ([Padding in the loss](#padding-in-the-loss)).
