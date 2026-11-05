#!/usr/bin/env python3
"""The size interaction of the Scale paragraph, read at its centre and one decade above.

The regression is the one behind the ``b1 Delta x logN`` row of estimates_seeds512.csv: the
off-ray gap on Delta, Delta x (log10 N_gm - centre) and the centred log10 N_gm, with a cell
fixed effect, the centre being the pairs' mean log10 N_gm (`amendment5_1b.smooth_engine`).
This reads the effect b0 at the centre and b0 + b1 one decade above it, each as a linear
combination with its own window-cluster interval (200,000 draws, seed 20260802, the seed of
`estimate.est`).

Reads ``results_E_seeds512.csv`` and ``results_X_seeds512.csv.gz`` (enumerate_seeds.py;
``--skip-x`` without the X corpus) and writes ``interaction_check.csv``. Relative paths are
taken in the work directory.

    uv run python amendments/interaction_check.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the reanalysis modules
import estimate as EST
from amendment5_1b import BIG, centred, smooth_engine
from paths import WORK

READINGS = (("b0, at the centre", 0.0), ("b0 + b1, one decade above the centre", 1.0))


def readings(corpus: str, df: pd.DataFrame) -> list[dict]:
    d, ctr = centred(df, "Delta_NC")
    eng = smooth_engine(d, "gap_offray", "Delta_NC")
    rows = []
    for spec, t in READINGS:
        e = eng.estimate_combo(np.random.default_rng(20260802), BIG, np.array([1.0, t, 0.0]))
        rows.append(dict(corpus=corpus, spec=spec, N_centre=10 ** ctr, N_at=10 ** (ctr + t),
                         beta=e.value, lo=e.lo, hi=e.hi, pct=EST.pct(e.value),
                         pct_lo=EST.pct(e.lo), pct_hi=EST.pct(e.hi), n_pairs=e.n_obs,
                         n_windows=e.n_clusters))
    return rows


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--results", default="results_E_seeds512.csv")
    ap.add_argument("--results-x", default="results_X_seeds512.csv.gz")
    ap.add_argument("--skip-x", action="store_true", help="run on the E corpus alone")
    ap.add_argument("--out", default="interaction_check.csv")
    args = ap.parse_args(argv)

    corpora = {"E": args.results} if args.skip_x else {"E": args.results, "X": args.results_x}
    out = pd.DataFrame([r for corpus, path in corpora.items()
                        for r in readings(corpus, EST.prepare(WORK / path))])
    with pd.option_context("display.width", 200):
        print(out.to_string(index=False))
    out.to_csv(WORK / args.out, index=False)


if __name__ == "__main__":
    main()
