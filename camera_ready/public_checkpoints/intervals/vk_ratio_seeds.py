#!/usr/bin/env python3
"""Seed robustness of the median margin-ratio intervals, and the pooled interval printed.

The three statistics of `vk_ratio_boot.py` (E2 scheme) and `ratio_of_medians_boot.py`: the
median of (V_K / tau_K)_NC / (V_K / tau_K)_CO in windows centred below 1B and at 1B and
above, and the ratio of the two medians. Each takes 10,000 window-cluster draws
(`conditional._cluster_draws`, CRITERIA E2) from each of `--seeds` independent streams,
seed s from `conditional.rng_for("vk_ratio_seeds", statistic, s)`, and once more from the
released stream of the two scripts. The interval the appendix prints is the one of the seeds'
draws pooled (30 seeds x 10,000 = 300,000 draws), where the ends have converged; the rows of
single streams show how far each end moves with the seed.

Reads ``results_E_seeds512.csv`` (written by enumerate_seeds.py) and writes
``vk_ratio_seeds.csv``: one row per statistic and stream (``released``, ``seed``) and one
``pooled`` row per statistic. Relative paths are taken in the work directory.

    uv run python intervals/vk_ratio_seeds.py --workers "$(nproc)"
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the reanalysis modules
import conditional as C
from paths import WORK
from ratio_of_medians_boot import ratio_of_medians
from vk_ratio import BELOW_TOP, TOP_STRATUM, margin_gaps, with_below_top
from vk_ratio_boot import COLUMNS, N_BOOT

RATIO = "ratio"  # >=1B median over <1B median; also the name of its seed streams
STATISTICS = (BELOW_TOP, TOP_STRATUM, RATIO)
_FRAMES: dict[str, pd.DataFrame] = {}  # the pairs each statistic resamples, set before forking


def statistic(name: str, d: pd.DataFrame) -> float:
    """The band's median ratio, or the ratio of the two band medians."""
    return ratio_of_medians(d) if name == RATIO else 10 ** float(np.median(d["delta"]))


def released_rng(name: str) -> np.random.Generator:
    """The stream `vk_ratio_boot.py` or `ratio_of_medians_boot.py` draws this statistic from."""
    if name == RATIO:
        return C.rng_for("vk_ratio_boot", "ratio_of_medians", "E2")
    return C.rng_for("vk_ratio_boot", name, "E2")


def draws(task: tuple[str, int, int]) -> tuple[str, int, np.ndarray]:
    """`n_boot` window-cluster draws of one statistic; seed -1 is the released stream."""
    name, seed, n_boot = task
    rng = released_rng(name) if seed < 0 else C.rng_for("vk_ratio_seeds", name, seed)
    d = _FRAMES[name]
    return name, seed, np.array([statistic(name, sub) for sub, _ in C._cluster_draws(d, rng, n_boot)])


def row(name: str, stream: str, seed: int, d: np.ndarray) -> dict:
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"statistic": name, "stream": stream, "seed": seed, "n_boot": len(d),
            "point": statistic(name, _FRAMES[name]), "lo95": lo, "hi95": hi,
            "share_draws_at_least_1": float(np.mean(d >= 1))}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--results", default="results_E_seeds512.csv")
    ap.add_argument("--out", default="vk_ratio_seeds.csv")
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    args = ap.parse_args(argv)

    gaps = margin_gaps(pd.read_csv(WORK / args.results, usecols=COLUMNS))
    bands = with_below_top(gaps)
    for name in (BELOW_TOP, TOP_STRATUM):
        _FRAMES[name] = bands[bands["stratum"] == name].reset_index(drop=True)
    _FRAMES[RATIO] = gaps.reset_index(drop=True)

    tasks = [(name, s, args.n_boot) for name in STATISTICS for s in range(-1, args.seeds)]
    fork = mp.get_context("fork")  # the workers read _FRAMES
    with ProcessPoolExecutor(min(args.workers, len(tasks)), mp_context=fork) as ex:  # one per stream
        results = list(ex.map(draws, tasks))
    rows = [row(name, "released" if s < 0 else "seed", s, d) for name, s, d in results]
    for name in STATISTICS:
        pooled = np.concatenate([d for n, s, d in results if n == name and s >= 0])
        rows.append(row(name, "pooled", -1, pooled))
    out = pd.DataFrame(rows)
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(out[out["stream"] != "seed"].to_string(index=False))
        print(out[out["stream"] == "seed"].groupby("statistic")[["lo95", "hi95"]]
              .agg(["min", "max"]).to_string())
    out.to_csv(WORK / args.out, index=False)


if __name__ == "__main__":
    main()
