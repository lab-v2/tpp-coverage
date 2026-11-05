#!/usr/bin/env python3
"""Every regime cell whose interval the appendix prints or reads, at 200,000 resamples.

Four families of cells, all on the 512-seed fits, NC arm, off-ray gap, beta_eff = 0.28
unless noted:

- ``joint``: the protocol's regime test (`estimate.dose_jobs`, unchanged), the appendix's
  regime table and its text;
- ``coonly``: each pair classified by its collinear subset alone (`co_only_regime.co_jobs`);
- ``band``: the joint cells inside each size band (`regime_by_band_ci.band_jobs`, CRITERIA
  A3.4);
- ``table7``: the collinear-subset rule with each law's own exponent, as the paper's Regime-A
  table classifies: Chinchilla pairs at 0.28, Kaplan pairs at 0.10 (the grid value nearest
  alpha_D = 0.095).

The jobs and their seed keys are those of the three released scripts; only the interval
differs: `median_ci.intervals`, the released window-cluster loop at 200,000 draws.
Replicate 0 is reported; the others only test whether each interval's verdict on zero moves.
Before any interval is drawn, `median_ci.check_released` proves on three small cells that
the loop, fed the released generator state, gives back the released 2,000-draw interval.

Reads ``results_E_seeds512.csv`` and ``results_X_seeds512.csv.gz`` (enumerate_seeds.py;
``--skip-x`` without the X corpus) and writes ``regime_cells.csv`` (replicate 0, plus the
range of the interval ends over the replicates and the share that excludes zero) and, with
``--replicates-out``, one row per cell and replicate. Relative paths are taken in the work
directory.

    uv run python intervals/regime_cells.py --replicates 10 --workers "$(nproc)"
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the reanalysis modules
import conditional as C
import estimate as EST
import median_ci as M
from co_only_regime import co_jobs
from paths import WORK
from regime_by_band_ci import GAP, band_jobs, load_corpora

KEYS = ["family", "corpus", "kappa", "cell", "band"]


def table7_jobs(df: pd.DataFrame, corpus: str) -> list[tuple]:
    """The Regime-A pairs of the collinear-subset rule at each law's own exponent."""
    m = np.where(df["law"].to_numpy() == "Kaplan", df["CO_m_beta0.1"], df["CO_m_beta0.28"])
    m = np.where(np.isfinite(m.astype(float)), m.astype(float), -np.inf)
    jobs = []
    for kappa in C.KAPPA_GRID:
        sub = df[m + np.log10(kappa / 100.0) < 0]
        jobs.append(EST.median_task(("table7", corpus, kappa, "CO=A", "all"), sub, GAP,
                                    f"table7|{corpus}|{kappa}|A"))
    return jobs


def jobs_for(df: pd.DataFrame, corpus: str) -> list[tuple]:
    """Every job of the four families for one corpus, tagged with `KEYS`."""
    out = [(("joint", corpus, kappa, cell, "all"), *rest)
           for (_, arm, kappa, cell), *rest in EST.dose_jobs(df, f"{corpus}|512") if arm == "NC"]
    out += [(("coonly", corpus, kappa, f"CO={regime}", "all"), *rest)
            for (_, kappa, regime), *rest in co_jobs(df, corpus)]
    out += table7_jobs(df, corpus)
    out += [(("band", corpus, kappa, cell, stratum), *rest)
            for (_, kappa, cell, stratum), *rest in band_jobs(df, corpus)]
    return out


def summarise(rows: pd.DataFrame) -> pd.DataFrame:
    """Replicate 0 of every cell, with the range of its ends and verdicts over the replicates."""
    rep0 = rows[rows["replicate"] == 0].set_index(KEYS)
    g = rows.groupby(KEYS)
    out = rep0.assign(lo_min=g["lo"].min(), lo_max=g["lo"].max(), hi_min=g["hi"].min(),
                      hi_max=g["hi"].max(), share_excl0=g["excludes0"].mean(),
                      replicates=g.size())
    return out.reset_index().drop(columns=["replicate"])


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--results", default="results_E_seeds512.csv")
    ap.add_argument("--results-x", default="results_X_seeds512.csv.gz")
    ap.add_argument("--skip-x", action="store_true", help="run on the E corpus alone")
    ap.add_argument("--replicates", type=int, default=10)
    ap.add_argument("--out", default="regime_cells.csv")
    ap.add_argument("--replicates-out", default=None)
    ap.add_argument("--workers", type=int, default=min(64, os.cpu_count() or 1))
    args = ap.parse_args(argv)

    corpora = load_corpora(args.results, None if args.skip_x else args.results_x)
    jobs = [job for corpus, df in corpora.items() for job in jobs_for(df, corpus)]
    small = [j for j in jobs if j[0][0] == "joint" and 0 < len(j[1]) < 20_000
             and sum(len(g) for g in j[2]) >= 2][:3]
    M.check_released(small)

    rows = pd.DataFrame(M.intervals(jobs, range(args.replicates), args.workers))
    tags = pd.DataFrame(rows["tag"].tolist(), columns=KEYS)
    rows = pd.concat([tags, rows.drop(columns="tag")], axis=1)
    if args.replicates_out:
        rows.to_csv(WORK / args.replicates_out, index=False)
    summ = summarise(rows)
    summ.to_csv(WORK / args.out, index=False)
    show = summ[summ["n_pairs"] > 0][KEYS + ["n_pairs", "n_windows", "win", "median", "lo", "hi",
                                             "share_excl0"]]
    with pd.option_context("display.width", 250):
        print(show.to_string(index=False, float_format=lambda v: f"{v:+.5f}"))
    flips = summ[(summ["share_excl0"] > 0) & (summ["share_excl0"] < 1)]
    print(f"\ncells whose verdict on zero moves across {args.replicates} replicates: {len(flips)}")


if __name__ == "__main__":
    main()
