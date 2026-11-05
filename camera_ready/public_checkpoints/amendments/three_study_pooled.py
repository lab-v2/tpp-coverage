#!/usr/bin/env python3
"""The primary and the pooled slope on the corpus before Amendment 4, at 5 and 512 seeds.

Amendment 4 followed a pooled estimate whose interval contained zero on the three studies
screened first (Farseer, Chinchilla isoFLOP and Gadre et al., 753 runs). This recomputes that
estimate and the within-cell one with the released estimator (`estimate.est`, 200,000
window-cluster draws) on seeds 0-4, the depth of the original run, and on all 512 seeds (post
hoc). The winsorisation floor of a degenerate collinear margin is taken over each analysis
set, as A3.1 prescribes (`estimate.prepare_frame`).

Reads ``results_E_seeds512.csv`` (enumerate_seeds.py) and writes ``three_study_pooled.csv``.
Relative paths are taken in the work directory.

    uv run python amendments/three_study_pooled.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the reanalysis modules
import conditional as C
import estimate as EST
from paths import WORK

STUDIES = ("farseer", "chinchilla_isoflop", "gadre_overtrain")
SPECS = (("cell FE (primary)", "cell"), ("pooled, no fixed effect", "_one"))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--results", default="results_E_seeds512.csv")
    ap.add_argument("--out", default="three_study_pooled.csv")
    args = ap.parse_args(argv)

    raw = pd.read_csv(WORK / args.results)
    raw = raw[C.suite_of(raw["cell"]).isin(STUDIES)]
    rows = []
    for seeds in (5, 512):
        df = EST.prepare_frame(raw[raw["seed"] < seeds].copy())
        for spec, fe in SPECS:
            e = EST.est(df, "gap_offray", "Delta_NC", fe=fe)
            rows.append(dict(seeds=seeds, spec=spec, beta=e.value, lo=e.lo, hi=e.hi,
                             pct=EST.pct(e.value), pct_lo=EST.pct(e.lo), pct_hi=EST.pct(e.hi),
                             n_pairs=e.n_obs, n_windows=e.n_clusters,
                             n_cells=df["cell"].nunique()))
    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    out.to_csv(WORK / args.out, index=False)


if __name__ == "__main__":
    main()
