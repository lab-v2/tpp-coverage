#!/usr/bin/env python3
"""How much better identified is the NC design than its budget-matched CO design, by size?

For every design pair of the 512-seed E-corpus enumeration this takes the ratio of the two
designs' identifiability margins, 10 ** (m_NC - m_CO) with m = log10(V_K / tau_K) at the
primary beta_eff = 0.28 and kappa_target = 100, and reports its median per coarse size
stratum. Pairs whose CO design is degenerate (V_K = 0) have no finite ratio and are left out.

Reads ``results_E_seeds512.csv`` from the work directory (written by enumerate_seeds.py);
writes ``vk_ratio.csv`` there.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import conditional as C
from paths import WORK


def vk_ratio_by_stratum(df: pd.DataFrame, beta: float = C.BETA_PRIMARY) -> pd.DataFrame:
    delta = (df[f"NC_m_beta{beta}"] - df[f"CO_m_beta{beta}"]).replace([np.inf, -np.inf], np.nan)
    d = df.assign(delta=delta, stratum=df["N_gm"].map(C.coarse_stratum)).dropna(subset=["delta"])
    d = pd.concat([d, d[d["stratum"] != ">=1B"].assign(stratum="<1B")])
    rows = []
    for stratum, g in d.groupby("stratum"):
        rows.append({"stratum": stratum, "n_pairs": len(g),
                     "n_windows": g.groupby(["cell", "window"]).ngroups,
                     "N_max": float(g["N_max"].max()),
                     "median_vk_ratio": float(10 ** g["delta"].median())})
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--results", default="results_E_seeds512.csv")
    ap.add_argument("--out", default="vk_ratio.csv")
    args = ap.parse_args(argv)
    cols = ["cell", "window", "N_gm", "N_max", f"CO_m_beta{C.BETA_PRIMARY}",
            f"NC_m_beta{C.BETA_PRIMARY}"]
    out = vk_ratio_by_stratum(pd.read_csv(WORK / args.results, usecols=cols))
    print(out.to_string(index=False))
    out.to_csv(WORK / args.out, index=False)


if __name__ == "__main__":
    main()
