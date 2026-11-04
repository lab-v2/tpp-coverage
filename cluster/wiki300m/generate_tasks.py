#!/usr/bin/env python3
"""
WIKI300M stratum - task generator (SEPARATE config; the published EXPERIMENTAL
DESIGN block in generate_grid_jobs.py is untouched).

Design (approved): 8 cols (5 train k + 3 test k) x 9 rows (6 train N + 3
holdout-only N), same L-shaped occupancy as the published appendix figure.
CO train 30 == NC train 30 (budget-matched); ONE pooled holdout of 40.
Wikipedia, 1 epoch, LR 3e-4, bs 96/GPU x 4 GPUs, seq 256, wd 0.1, llama.

Split protocol copied from the published generators:
  train cells   -> --skip-val          (train split)
  holdout cells -> --use-train-as-val  (val split)   [generate_holdout_grid.py]

Emits: <out>/tasks/NNN_<name>.task (KEY=VALUE files), ordered longest-first (LPT),
plus <out>/manifest.tsv, where <out> is the first argument, else $QUEUE_DIR. The queue
scripts in cluster/queue/ consume them.

    python cluster/wiki300m/generate_tasks.py /path/to/run

This writes the first 100 cells, tasks 000-099 of the committed queue in tasks/. Tasks
100-467 (densified arms and the seed-43/44 replicates) were added to that queue later and are
committed as they were trained.
"""
import os
import sys
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "src", "tpp_analysis"))
sys.path.insert(0, os.path.join(REPO, "src"))
from subset_analysis.subset_analysis_bbox_seeded import (
    compute_VK, compute_tauK, classify_regime)
from training_models.core.utils import generate_model_configs

OUT = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("QUEUE_DIR", "")
if not OUT:
    sys.exit("usage: generate_tasks.py <run dir>   (or set QUEUE_DIR)")
THRU = 131e12
NCOL, NROW, NTC, NTR = 8, 9, 5, 6
K_TRAIN = [1, 1.5, 2, 2.7, 3.3]
K_TEST = [4.0, 4.2, 4.6]
TRAIN_N = [22_796_832, 40_277_184, 68_127_488, 105_333_888, 153_028_096, 203_372_032]
HOLD_N = [229_536_000, 255_706_112, 305_045_760]
ROWS = TRAIN_N + HOLD_N
PUBLISHED_N = {22_796_832, 40_277_184, 68_127_488}

# --model-size argument per row: published sizes hit the hot path directly;
# new sizes must be requested by legacy_param_count (the search matches on it).
TRAINER_KW = dict(vocab_size=100277, max_seq_len=256,
                  min_params=100, max_params=1_000_000_000)
_pool = generate_model_configs(2000, **TRAINER_KW)
_by_total = {c.total_params: c for c in _pool}
ARG = {}
for N in ROWS:
    ARG[N] = N if N in PUBLISHED_N else _by_total[N].legacy_param_count
    got = generate_model_configs(1, target_params=[ARG[N]], **TRAINER_KW)[0]
    assert got.total_params == N, f"round-trip {ARG[N]} -> {got.total_params} != {N}"

lo = 0.9 * min(K_TRAIN) * min(TRAIN_N)
hi = 1.15 * max(K_TRAIN) * max(TRAIN_N)
D_TR = [int(round(d, -5)) for d in np.exp(np.linspace(np.log(lo), np.log(hi), NTC))]
D_TE = [int(round(d, -5)) for d in
        np.exp(np.linspace(np.log(1.10 * hi), np.log(1.46 * hi), NCOL - NTC))]
NC_COLS = D_TR + D_TE
CO_COLS = K_TRAIN + K_TEST

def occ(x, y):
    if x == NTC or y == NTR or (x > NTC and y > NTR):
        return "holdout"
    if x < NTC and y < NTR:
        return "train"
    return "unused"

def fmt(v):
    return f"{v/1e6:.2f}M"

tasks = []
for y, N in enumerate(ROWS):
    for x in range(NCOL):
        role = occ(x, y)
        if role == "unused":
            continue
        k = CO_COLS[x]
        D_co = int(round(k * N))                    # k is exact on the TRUE N
        tag = "COLINEAR" if role == "train" else "COLINEAR_HOLDOUT"
        tasks.append(dict(
            prefix=f"WIKI300M_{tag}_N{fmt(N)}_TPP{k}_D{fmt(D_co)}",
            arg=ARG[N], tokens=D_co, role=role, N=N,
            nh=6.0 * N * D_co / THRU / 3600.0))
        D_nc = NC_COLS[x]
        tag = "NxD_GRID" if role == "train" else "HOLDOUT"
        tasks.append(dict(
            prefix=f"WIKI300M_{tag}_N{fmt(N)}_D{fmt(D_nc)}",
            arg=ARG[N], tokens=D_nc, role=role, N=N,
            nh=6.0 * N * D_nc / THRU / 3600.0))

# ---- verification --------------------------------------------------------
co_tr = [t for t in tasks if "_COLINEAR_N" in t["prefix"]]
nc_tr = [t for t in tasks if "_NxD_GRID_" in t["prefix"]]
co_ho = [t for t in tasks if "_COLINEAR_HOLDOUT_" in t["prefix"]]
nc_ho = [t for t in tasks if "_HOLDOUT_" in t["prefix"] and "COLINEAR" not in t["prefix"]]
assert len(co_tr) == len(nc_tr) == 30, (len(co_tr), len(nc_tr))
assert len(co_ho) == len(nc_ho) == 20, (len(co_ho), len(nc_ho))
assert len(tasks) == 100
assert len({t["prefix"] for t in tasks}) == 100, "duplicate prefixes"
V = compute_VK(K_TRAIN, 0.28)
assert classify_regime(V, compute_tauK(K_TRAIN, 0.28, 100.0)) == "A"
d_max = max(t["tokens"] for t in tasks)
assert d_max <= 1.80e9, d_max
longest = max(t["nh"] for t in tasks)
assert longest < 12.0

# ---- emit ----------------------------------------------------------------
tdir = os.path.join(OUT, "tasks")
os.makedirs(tdir, exist_ok=True)
existing = [f for f in os.listdir(tdir) if f.endswith(".task")]
if existing:
    sys.exit(f"REFUSING to overwrite: {len(existing)} task files already in {tdir}")

tasks.sort(key=lambda t: -t["nh"])          # LPT: longest first
with open(os.path.join(OUT, "manifest.tsv"), "w") as mf:
    mf.write("rank\tprefix\tmodel_size_arg\ttokens\trole\test_node_h\n")
    for i, t in enumerate(tasks):
        name = f"{i:03d}_{t['prefix']}"
        with open(os.path.join(tdir, name + ".task"), "w") as f:
            f.write(f"PREFIX={t['prefix']}\n")
            f.write(f"MODEL_SIZE={t['arg']}\n")
            f.write(f"TOKENS={t['tokens']}\n")
            f.write(f"SPLIT_FLAG={'--skip-val' if t['role']=='train' else '--use-train-as-val'}\n")
            f.write(f"EST_NH={t['nh']:.3f}\n")
        mf.write(f"{i:03d}\t{t['prefix']}\t{t['arg']}\t{t['tokens']}\t{t['role']}\t{t['nh']:.3f}\n")

tot = sum(t["nh"] for t in tasks)
print(f"wrote {len(tasks)} tasks -> {tdir}")
print(f"total {tot:.1f} node-h | longest {longest:.2f} h | 4 slots ~{tot/4:.1f} h")
print(f"V_K={V:.4e} (published 1.0222e-02) regime A | D_max={d_max/1e9:.2f}B")
