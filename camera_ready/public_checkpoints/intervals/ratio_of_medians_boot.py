#!/usr/bin/env python3
"""95% interval for the ratio of the two stratum medians of vk_ratio.py, >=1B over <1B.

With Delta = m_NC - m_CO per design pair (vk_ratio.py, beta_eff = 0.28, degenerate CO
designs left out), the statistic is 10 ** (median Delta at >=1B - median Delta below 1B):
how much of the NC design's identifiability advantage the windows centred at 1B and above
keep. Below 1 means less than below 1B. The interval is the paper's convention, (cell, window)
clusters resampled within cell (`conditional._cluster_draws`, CRITERIA E2), with 10,000 draws
from the stream `conditional.rng_for("vk_ratio_boot", "ratio_of_medians", "E2")`.

Reads ``results_E_seeds512.csv`` (written by enumerate_seeds.py) and writes the one-row
``ratio_of_medians_boot.csv``. Relative paths are taken in the work directory.

    uv run python intervals/ratio_of_medians_boot.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the reanalysis modules
import conditional as C
from paths import WORK
from vk_ratio import TOP_STRATUM, margin_gaps
from vk_ratio_boot import COLUMNS, N_BOOT


def ratio_of_medians(d: pd.DataFrame) -> float:
    """10 ** (median delta of the >=1B pairs - median delta of the pairs below 1B)."""
    top = d["stratum"].to_numpy() == TOP_STRATUM
    delta = d["delta"].to_numpy()
    return 10 ** (np.median(delta[top]) - np.median(delta[~top]))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--results", default="results_E_seeds512.csv")
    ap.add_argument("--out", default="ratio_of_medians_boot.csv")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    args = ap.parse_args(argv)

    pairs = margin_gaps(pd.read_csv(WORK / args.results, usecols=COLUMNS)).reset_index(drop=True)
    rng = C.rng_for("vk_ratio_boot", "ratio_of_medians", "E2")
    draws = np.array([ratio_of_medians(sub) for sub, _ in C._cluster_draws(pairs, rng, args.n_boot)])
    lo, hi = np.percentile(draws, [2.5, 97.5])
    out = pd.DataFrame([{
        "statistic": f"median ratio {TOP_STRATUM} / below 1B", "scheme": "E2",
        "n_boot": args.n_boot, "n_pairs": len(pairs),
        "n_windows": pairs.groupby(C.CLUSTER).ngroups,
        "ratio_of_medians": ratio_of_medians(pairs), "lo95": lo, "hi95": hi,
        "share_draws_at_least_1": float(np.mean(draws >= 1)),
    }])
    print(out.to_string(index=False))
    out.to_csv(WORK / args.out, index=False)


if __name__ == "__main__":
    main()
