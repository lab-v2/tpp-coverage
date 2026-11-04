#!/usr/bin/env python3
"""95% intervals for the median ratio of identifiability margins, per size stratum.

The statistic is vk_ratio.py's: for every design pair of the 512-seed E-corpus enumeration,
10 ** (m_NC - m_CO) = (V_K / tau_K)_NC / (V_K / tau_K)_CO at the primary beta_eff = 0.28,
and its median per coarse size stratum ("<1B" pools the two strata below 1B). Pairs whose CO
design is degenerate (V_K = 0) have no finite ratio and are left out, as in vk_ratio.py.

Each interval takes 10,000 bootstrap draws from its own `conditional.rng_for` stream:

    E2      (cell, window) clusters resampled within cell (`conditional._cluster_draws`,
            CRITERIA E2): the scheme of every interval the paper prints
    pooled  the (cell, window) clusters of the stratum resampled together, ignoring cell
    pairs   design pairs resampled iid, the convention of `conditional.boot_stat`

Reads ``results_E_seeds512.csv`` (written by enumerate_seeds.py) and writes
``vk_ratio_boot.csv``, plus the median per window of the >=1B stratum in
``vk_ratio_boot_ge1B_windows.csv``. Relative paths are taken in the work directory.

    uv run python intervals/vk_ratio_boot.py --workers "$(nproc)"
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the reanalysis modules
import conditional as C
from paths import WORK
from vk_ratio import TOP_STRATUM, margin_gaps, with_below_top

N_BOOT = 10_000
SCHEMES = ("E2", "pooled", "pairs")
STRATA = ("<100M", "100M-1B", "<1B", ">=1B")
COLUMNS = ["cell", "window", "N_gm", "N_min", "N_max",
           f"CO_m_beta{C.BETA_PRIMARY}", f"NC_m_beta{C.BETA_PRIMARY}"]


def median_draws(d: pd.DataFrame, scheme: str, rng: np.random.Generator,
                 n_boot: int) -> np.ndarray:
    """Bootstrap draws of the median of ``d["delta"]`` under one resampling scheme."""
    if scheme == "E2":
        return np.array([np.median(sub["delta"].to_numpy())
                         for sub, _ in C._cluster_draws(d, rng, n_boot)])
    if scheme == "pooled":
        groups = [g.to_numpy() for _, g in d.groupby(C.CLUSTER)["delta"]]
        picks = (rng.integers(0, len(groups), len(groups)) for _ in range(n_boot))
        return np.array([np.median(np.concatenate([groups[i] for i in idx])) for idx in picks])
    if scheme == "pairs":
        x = d["delta"].to_numpy()
        return np.array([np.median(x[rng.integers(0, len(x), len(x))]) for _ in range(n_boot)])
    raise ValueError(f"unknown resampling scheme {scheme!r}")


def stratum_interval(task: tuple[str, str, pd.DataFrame, int]) -> dict:
    """One output row: the median ratio of a stratum and its 95% interval under a scheme."""
    stratum, scheme, d, n_boot = task
    rng = C.rng_for("vk_ratio_boot", stratum, scheme)
    lo, hi = np.percentile(median_draws(d, scheme, rng, n_boot), [2.5, 97.5])
    return {"stratum": stratum, "scheme": scheme, "n_boot": n_boot, "n_pairs": len(d),
            "n_windows": d.groupby(C.CLUSTER).ngroups, "n_cells": d["cell"].nunique(),
            "median_vk_ratio": 10 ** float(np.median(d["delta"])),
            "lo95": 10 ** lo, "hi95": 10 ** hi}


def window_medians(pairs: pd.DataFrame, stratum: str = TOP_STRATUM) -> pd.DataFrame:
    """The median ratio of every window of one stratum, with the window's size range."""
    return (pairs[pairs["stratum"] == stratum]
            .groupby(C.CLUSTER)
            .agg(N_min=("N_min", "first"), N_max=("N_max", "first"), N_gm=("N_gm", "first"),
                 n_pairs=("delta", "size"),
                 median_vk_ratio=("delta", lambda s: 10 ** s.median())))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--results", default="results_E_seeds512.csv")
    ap.add_argument("--out", default="vk_ratio_boot.csv")
    ap.add_argument("--windows-out", default="vk_ratio_boot_ge1B_windows.csv")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--workers", type=int, default=min(len(STRATA) * len(SCHEMES),
                                                       os.cpu_count() or 1))
    args = ap.parse_args(argv)

    pairs = with_below_top(margin_gaps(pd.read_csv(WORK / args.results, usecols=COLUMNS)))
    tasks = [(s, k, pairs.loc[pairs["stratum"] == s, ["cell", "window", "delta"]]
              .reset_index(drop=True), args.n_boot)
             for s in STRATA for k in SCHEMES]
    with mp.get_context("fork").Pool(min(args.workers, len(tasks))) as pool:  # one per task
        out = pd.DataFrame(pool.map(stratum_interval, tasks, chunksize=1))
    print(out.to_string(index=False))
    out.to_csv(WORK / args.out, index=False)

    windows = window_medians(pairs)
    print(f"\n{TOP_STRATUM} windows:\n{windows.to_string()}")
    windows.to_csv(WORK / args.windows_out)


if __name__ == "__main__":
    main()
