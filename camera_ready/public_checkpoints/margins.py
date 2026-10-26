#!/usr/bin/env python3
"""Identifiability margins m_X = log10(V_K / tau_K) for every design in results.csv.

CRITERIA Amendment 3, section A3.5. The designs are NOT re-carved by new logic: this
reconstructs them through the same `analysis.enumerate_pairs` code path, with the same
seeds, and refuses to write anything unless the reconstruction reproduces the stored
V_K, tau_K, K and k-range of every design in ``results.csv``. Only the C2 beta grid is
new; the kappa grid needs no recomputation because tau_K is exactly inversely
proportional to kappa_target, so m_X(kappa) = m_X(100) + log10(kappa / 100).

Writes ``margins.csv``: one row per (cell, window, ray_bin, n_rays, seed) design pair.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import analysis as A

from paths import ARTIFACTS, WORK
ARMS = ("CO", "NC", "NCM")
KEY = ["cell", "window", "ray_bin", "n_rays", "seed"]


def _margin(ratios: list[float], beta: float, kappa: float) -> float:
    """log10(V_K / tau_K) from the repo's own primitives; -inf when V_K = 0."""
    vk = float(A.repo.compute_VK(ratios, beta))
    tk = float(A.repo.compute_tauK(ratios, beta, kappa))
    return float(np.log10(vk / tk)) if vk > 0 else float("-inf")


def rebuild(runs_file: str = "runs_normalised.csv", corpus: str = "all") -> pd.DataFrame:
    runs = pd.read_csv(ARTIFACTS / runs_file)
    if corpus != "all" and "corpus" in runs.columns:
        runs = runs[runs["corpus"].isin({"E": {"E"}, "X": {"E", "X"}}[corpus])]
    runs = runs.reset_index(drop=True)
    rows: list[dict] = []
    for cell, grp in runs.groupby("cell"):
        if not A.screen_cell(grp, cell).included:
            continue
        for window in A.build_windows(cell, grp["N"].unique()):
            split = A.Split.from_window(grp[grp["N"].isin(window.sizes)])
            if split.core.empty or len(split.holdout) < 3:
                continue
            for pair in A.enumerate_pairs(window, split):
                row = {
                    "cell": cell, "window": window.index, "ray_bin": pair.ray,
                    "n_rays": pair.n_rays, "seed": pair.seed,
                }
                for arm, idx in (("CO", pair.co_idx), ("NC", pair.nc_idx), ("NCM", pair.ncm_idx)):
                    ks = runs.loc[idx, "k"]
                    rats = A.merged_ratios(ks.tolist())
                    row[f"{arm}_K_chk"] = len(rats)
                    row[f"{arm}_V_K_chk"] = float(A.repo.compute_VK(rats, A.BETA_EFF))
                    row[f"{arm}_tau_K_chk"] = float(A.repo.compute_tauK(rats, A.BETA_EFF, A.KAPPA_TARGET))
                    row[f"{arm}_k_min_chk"] = float(ks.min())
                    row[f"{arm}_k_max_chk"] = float(ks.max())
                    for beta in A.BETA_GRID:
                        row[f"{arm}_m_beta{beta}"] = _margin(rats, beta, A.KAPPA_TARGET)
                rows.append(row)
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Identifiability margins for every design in a results table.")
    ap.add_argument("--runs", default="runs_normalised.csv")
    ap.add_argument("--corpus", choices=["E", "X", "all"], default="all")
    ap.add_argument("--results", default="results.csv")
    ap.add_argument("--out", default="margins.csv")
    args = ap.parse_args(argv)

    built = rebuild(args.runs, args.corpus)
    stored = pd.read_csv(WORK / args.results).drop_duplicates(subset=KEY)
    merged = stored.merge(built, on=KEY, how="inner", validate="one_to_one")
    assert len(merged) == len(stored) == len(built), (
        f"reconstruction mismatch: stored={len(stored)} built={len(built)} merged={len(merged)}"
    )

    for arm in ARMS:
        for col in ("K", "V_K", "tau_K", "k_min", "k_max"):
            a = merged[f"{arm}_{col}"].to_numpy(float)
            b = merged[f"{arm}_{col}_chk"].to_numpy(float)
            assert np.allclose(a, b, rtol=1e-9, atol=0), f"{arm}_{col} does not reproduce"
    print(f"reconstruction verified against results.csv on {len(merged)} design pairs "
          f"({len(ARMS) * 5} invariants per pair)")

    out = merged[KEY + [f"{a}_m_beta{b}" for a in ARMS for b in A.BETA_GRID]]
    out.to_csv(WORK / args.out, index=False)
    print(f"wrote {args.out} rows={len(out)}")


if __name__ == "__main__":
    main()
