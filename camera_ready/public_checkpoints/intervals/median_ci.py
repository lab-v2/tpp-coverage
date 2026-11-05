"""Window-cluster 95% interval of a median gap at 200,000 resamples, with fixed seeds.

The resampling is `estimate.cluster_medians`, the window-cluster loop of the released
`estimate._median_job`: within each cell, its (cell, window) clusters are drawn with
replacement, and the statistic is the median of the pair gaps of the drawn windows. Only the
depth and the seed differ. `_median_job` takes 2,000 draws from `default_rng(crc32(key))`
after its pair-level draws, where `key` is the label the job carries (for example
"E|512|NC|100000.0|CO=B,NC=B" from `estimate.dose_jobs`). Here the 200,000 draws come in 100
chunks of 2,000, chunk c of replicate r from `default_rng(crc32(f"{key}|r{r}|c{c}"))`.
Replicate 0 is the reported interval; replicates 1, 2, ... exist only to show that the
verdict on zero does not depend on the seed. `check_released` proves on given jobs that this
loop, fed the released generator state, returns the released interval exactly.
"""

from __future__ import annotations

import multiprocessing as mp
import sys
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the reanalysis modules
import estimate as EST

N_BOOT = 200_000
CHUNK = 2_000
_JOBS: list = []  # (tag, vals, groups, key), set before the pool forks


def _chunk(task: tuple[int, int, int]) -> tuple[int, int, np.ndarray]:
    j, rep, c = task
    _, vals, groups, key = _JOBS[j]
    rng = np.random.default_rng(zlib.crc32(f"{key}|r{rep}|c{c}".encode()))
    return j, c, EST.cluster_medians(vals, groups, rng, CHUNK)


def intervals(jobs: list, replicates: range, workers: int) -> list[dict]:
    """One row per (job, replicate): the job's statistics and its 200,000-draw interval.

    A job is `estimate.median_task`'s tuple (tag, gaps, window groups, seed key).
    """
    global _JOBS
    _JOBS = jobs
    usable = [j for j, (_, v, g, _) in enumerate(jobs) if len(v) and sum(len(x) for x in g) >= 2]
    usable.sort(key=lambda j: -len(jobs[j][1]))  # largest first, for load balance
    rows = []
    with mp.get_context("fork").Pool(workers) as pool:  # independent bootstrap chunks
        for rep in replicates:
            draws = {j: np.empty(N_BOOT) for j in usable}
            tasks = [(j, rep, c) for j in usable for c in range(N_BOOT // CHUNK)]
            for j, c, d in pool.imap_unordered(_chunk, tasks, chunksize=1):
                draws[j][c * CHUNK:(c + 1) * CHUNK] = d
            for j, (tag, vals, groups, key) in enumerate(jobs):
                lo = hi = np.nan
                if j in draws:
                    lo, hi = np.percentile(draws[j], [2.5, 97.5])
                rows.append(dict(tag=tag, key=key, replicate=rep, n_pairs=len(vals),
                                 n_windows=sum(len(x) for x in groups),
                                 win=float((vals > 0).mean()) if len(vals) else np.nan,
                                 median=float(np.median(vals)) if len(vals) else np.nan,
                                 lo=float(lo), hi=float(hi),
                                 excludes0=bool(lo > 0 or hi < 0)))
    return rows


def check_released(jobs: list) -> None:
    """Assert that `cluster_medians`, after the released pair draws, gives `_median_job`'s interval."""
    for job in jobs:
        _, vals, groups, key = job
        ref = EST._median_job(job)[1]
        rng = np.random.default_rng(zlib.crc32(key.encode()))
        EST.pair_medians(vals, rng, EST.MED_BOOT)  # the pair-level draws the released code takes first
        lo, hi = np.percentile(EST.cluster_medians(vals, groups, rng, EST.MED_BOOT), [2.5, 97.5])
        assert np.isclose(lo, ref["lo_cl"]) and np.isclose(hi, ref["hi_cl"]), (key, lo, hi, ref)
        print(f"  released-path check {key}: [{lo:+.5f}, {hi:+.5f}] == _median_job")
