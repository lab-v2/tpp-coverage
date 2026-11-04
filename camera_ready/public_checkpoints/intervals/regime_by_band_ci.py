#!/usr/bin/env python3
"""The regime test inside each coarse size stratum (CRITERIA A3.4), with window intervals.

A3.4 asks for the A3.3 confirmatory and falsification arms inside each coarse stratum. A pair
is classified by both of its designs (`conditional.regimes`: Regime B iff
m + log10(kappa_target / 100) >= 0, a degenerate CO design in Regime A) at every
kappa_target of the protocol grid, and its off-ray gap g > 0 favours the NC design. For the
confirmatory cell (CO=A,NC=B) and the two same-regime cells (CO=A,NC=A and CO=B,NC=B), in each
stratum and corpus, this reports the NC win rate and the median gap with both 95% intervals of
`estimate._median_job`, the code behind doseresponse_seeds512.csv: `lo`/`hi` resample pairs,
`lo_cl`/`hi_cl` resample (cell, window) clusters within cell and are the ones the paper prints.
`excludes_zero` refers to the window interval.

Reads ``results_E_seeds512.csv`` and ``results_X_seeds512.csv.gz`` (enumerate_seeds.py;
``--skip-x`` without the X corpus) and writes ``regime_by_band_ci.csv``. Relative paths are
taken in the work directory.

    uv run python intervals/regime_by_band_ci.py --workers "$(nproc)"
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the reanalysis modules
import conditional as C
import estimate as EST
from paths import WORK

# The confirmatory arm of A3.3 and its two falsification arms; the reversed cell is not tested.
REGIME_CELLS = ("CO=A,NC=B", "CO=A,NC=A", "CO=B,NC=B")
GAP = "gap_offray"
JobBuilder = Callable[[pd.DataFrame, str], list[tuple]]


def load_corpora(results: str, results_x: str | None) -> dict[str, pd.DataFrame]:
    """The prepared 512-seed frames (`estimate.prepare`), keyed by corpus; X only if named."""
    paths: dict[str, Path] = {"E": WORK / results}
    if results_x is not None:
        paths["X"] = WORK / results_x
    return {corpus: EST.prepare(path) for corpus, path in paths.items()}


def median_table(jobs: list[tuple], keys: Sequence[str], workers: int) -> pd.DataFrame:
    """Run `estimate._median_job` over `jobs` in parallel, one row per job.

    The job's tag fills the columns `keys`; `n_cells` is the number of cells its windows
    come from.
    """
    with mp.get_context("fork").Pool(min(workers, len(jobs))) as pool:  # one job per row
        results = pool.map(EST._median_job, jobs, chunksize=1)
    rows = [dict(zip(keys, tag), n_cells=len(groups), **stats)
            for (_, _, groups, _), (tag, stats) in zip(jobs, results)]
    out = pd.DataFrame(rows)
    out["kappa"] = out["kappa"].astype(int)
    out["excludes_zero"] = (out["lo_cl"] > 0) | (out["hi_cl"] < 0)
    return out


def band_jobs(df: pd.DataFrame, corpus: str) -> list[tuple]:
    """One job per (kappa_target, regime cell, stratum) that holds any pair."""
    jobs = []
    for kappa in C.KAPPA_GRID:
        cells = C.regimes(df, kappa, "NC")
        for cell in REGIME_CELLS:
            for stratum in EST.STRATA:
                sub = df[(cells == cell) & (df["stratum"] == stratum)]
                if len(sub):
                    jobs.append(EST.median_task((corpus, kappa, cell, stratum), sub, GAP,
                                                f"r2|{corpus}|{kappa}|{cell}|{stratum}"))
    return jobs


def regime_cli(argv: list[str] | None, doc: str, out: str, jobs_for: JobBuilder,
            keys: Sequence[str]) -> None:
    """The command line both regime scripts share: read, build the jobs, run, write `out`."""
    ap = argparse.ArgumentParser(description=doc.split("\n\n")[0])
    ap.add_argument("--results", default="results_E_seeds512.csv")
    ap.add_argument("--results-x", default="results_X_seeds512.csv.gz")
    ap.add_argument("--skip-x", action="store_true", help="run on the E corpus alone")
    ap.add_argument("--out", default=out)
    ap.add_argument("--workers", type=int, default=min(64, os.cpu_count() or 1))
    args = ap.parse_args(argv)

    corpora = load_corpora(args.results, None if args.skip_x else args.results_x)
    jobs = [job for corpus, df in corpora.items() for job in jobs_for(df, corpus)]
    table = median_table(jobs, keys, args.workers)
    with pd.option_context("display.width", 200):
        print(table.drop(columns=["lo", "hi", "signp"]).to_string(index=False))
    table.to_csv(WORK / args.out, index=False)


def main(argv: list[str] | None = None) -> None:
    regime_cli(argv, __doc__, "regime_by_band_ci.csv", band_jobs,
            ["corpus", "kappa", "regime_cell", "stratum"])


if __name__ == "__main__":
    main()
