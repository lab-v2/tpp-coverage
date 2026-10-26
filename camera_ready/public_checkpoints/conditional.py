#!/usr/bin/env python3
"""The conditional re-analysis of CRITERIA Amendment 3 (A3.1 - A3.4).

Contrasts designs by the theory's own conditioning quantity - the identifiability
margin m = log10(V_K / tau_K) of Proposition 2 / Eq. 5 - instead of by the nominal
design label. Reads ``results.csv`` (unchanged, from the Amendment 1-2 execution) and
``margins.csv`` (verified reconstruction of the same designs).

Writes ``conditional_primary.csv``, ``conditional_doseresponse.csv``,
``conditional_scale.csv`` (prefix set by --prefix) to the work directory and prints them.
"""

from __future__ import annotations

import argparse
import zlib
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from paths import WORK
N_BOOT = 2000
RNG_SEED = 20260801
BETA_PRIMARY = 0.28
BETA_GRID = (0.10, 0.20, 0.28, 0.35, 0.50)
KAPPA_GRID = (1e2, 1e4, 1e5, 1e6)
KAPPA_PRIMARY = 1e2
CLUSTER = ["cell", "window"]
COARSE_EDGES = [(0.0, 1e8, "<100M"), (1e8, 1e9, "100M-1B"), (1e9, np.inf, ">=1B")]


def rng_for(*parts: object) -> np.random.Generator:
    """A generator seeded from the identity of the statistic being computed.

    Every interval in this file draws from its own stream, so each reported number
    is reproducible on its own and does not move when unrelated code upstream of it
    consumes a different quantity of randomness.
    """
    key = "|".join(str(p) for p in parts).encode()
    return np.random.default_rng(RNG_SEED + zlib.crc32(key))


def coarse_stratum(n_gm: float) -> str:
    for lo, hi, name in COARSE_EDGES:
        if lo <= n_gm < hi:
            return name
    return ">=1B"


# ════════════════════════════════════════════════════════════════════════════
# Estimators
# ════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class Estimate:
    """A point estimate with a percentile bootstrap interval."""

    value: float
    lo: float
    hi: float
    n_obs: int
    n_clusters: int

    def __str__(self) -> str:
        return f"{self.value:+.4f} [{self.lo:+.4f}, {self.hi:+.4f}]"

    @property
    def excludes_zero(self) -> bool:
        return np.isfinite(self.lo) and np.isfinite(self.hi) and (self.lo > 0 or self.hi < 0)


def suite_of(cell: pd.Series) -> pd.Series:
    """CRITERIA A4.4: the suite a cell belongs to (cells are `suite/corpus`)."""
    return cell.astype(str).str.split("/").str[0]


def _design_matrix(df: pd.DataFrame, x_cols: Sequence[str], fe: str | None) -> np.ndarray:
    cols = [df[c].to_numpy(float) for c in x_cols]
    if fe is None:
        cols.append(np.ones(len(df)))
    else:
        groups = df[fe] if fe != "cellwindow" else df["cell"] + "#" + df["window"].astype(str)
        cols.extend((groups == g).to_numpy(float) for g in sorted(groups.unique()))
    return np.column_stack(cols)


def _ols(df: pd.DataFrame, y: str, x_cols: Sequence[str], fe: str | None,
         weight: str | None = None) -> np.ndarray:
    x = _design_matrix(df, x_cols, fe)
    y_v = df[y].to_numpy(float)
    if weight is not None:
        w = np.sqrt(df[weight].to_numpy(float))
        x, y_v = x * w[:, None], y_v * w
    return np.linalg.lstsq(x, y_v, rcond=None)[0]


def _cluster_draws(df: pd.DataFrame, rng: np.random.Generator, n_boot: int):
    """Resample (cell, window) clusters within cell, exactly as CRITERIA E2 does."""
    clusters = df.groupby(CLUSTER).indices
    by_cell: dict[str, list] = {}
    for key in clusters:
        by_cell.setdefault(key[0], []).append(key)
    for _ in range(n_boot):
        picked = []
        for cell_keys in by_cell.values():
            idx = rng.integers(0, len(cell_keys), size=len(cell_keys))
            picked.extend(cell_keys[i] for i in idx)
        yield df.iloc[np.concatenate([clusters[k] for k in picked])], len(clusters)


def regress(df: pd.DataFrame, y: str, x_cols: Sequence[str], fe: str | None,
            rng: np.random.Generator, n_boot: int = N_BOOT,
            suite_weighted: bool = False) -> dict[str, Estimate]:
    """OLS of y on x_cols with fixed effect `fe`; CIs from the E2 cluster bootstrap.

    `suite_weighted` is the A4.4 companion: observations are weighted so that every SUITE
    contributes equally regardless of how many cells or pairs it supplies.
    """
    df = df[np.isfinite(df[y]) & np.all([np.isfinite(df[c]) for c in x_cols], axis=0)]
    n_cl = df.groupby(CLUSTER).ngroups
    if len(df) < len(x_cols) + 2:
        return {c: Estimate(np.nan, np.nan, np.nan, len(df), n_cl) for c in x_cols}
    if suite_weighted:
        df = df.copy()
        share = suite_of(df["cell"]).map(suite_of(df["cell"]).value_counts())
        df["_w"] = 1.0 / share.to_numpy(float)
    w = "_w" if suite_weighted else None
    point = _ols(df, y, x_cols, fe, w)[: len(x_cols)]
    draws = []
    n_cells = df["cell"].nunique()
    if n_cl < 2:
        # A subgroup resting on one window has no between-cluster variation to resample.
        # Fall back to a pair-level bootstrap and flag it by n_clusters = 1: such an
        # interval understates the true uncertainty and is labelled not established.
        source = ((df.iloc[rng.integers(0, len(df), size=len(df))], 1) for _ in range(n_boot))
    else:
        source = _cluster_draws(df, rng, n_boot)
    for sub, _ in source:
        if sub["cell"].nunique() < n_cells:
            continue
        try:
            draws.append(_ols(sub, y, x_cols, fe, w)[: len(x_cols)])
        except np.linalg.LinAlgError:
            continue
    if not draws:
        return {c: Estimate(float(point[i]), np.nan, np.nan, len(df), n_cl)
                for i, c in enumerate(x_cols)}
    arr = np.asarray(draws)
    lo, hi = np.percentile(arr, [2.5, 97.5], axis=0)
    return {c: Estimate(float(point[i]), float(lo[i]), float(hi[i]), len(df), n_cl)
            for i, c in enumerate(x_cols)}


def boot_stat(x: np.ndarray, rng: np.random.Generator, stat=np.median) -> Estimate:
    """Percentile bootstrap over design pairs (CRITERIA E1)."""
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return Estimate(np.nan, np.nan, np.nan, 0, 0)
    draws = stat(x[rng.integers(0, len(x), size=(N_BOOT, len(x)))], axis=1)
    return Estimate(float(stat(x)), *np.percentile(draws, [2.5, 97.5]), len(x), 0)


def boot_stat_clustered(df: pd.DataFrame, col: str, rng: np.random.Generator, stat=np.median) -> Estimate:
    """Same statistic with the window-cluster bootstrap, for subgroups that may rest
    on very few windows. Deliberately wider than boot_stat; both are reported."""
    df = df[np.isfinite(df[col])]
    if df.empty:
        return Estimate(np.nan, np.nan, np.nan, 0, 0)
    point = float(stat(df[col].to_numpy(float)))
    draws = [float(stat(sub[col].to_numpy(float))) for sub, _ in _cluster_draws(df, rng, N_BOOT)]
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return Estimate(point, float(lo), float(hi), len(df), df.groupby(CLUSTER).ngroups)


# ════════════════════════════════════════════════════════════════════════════
# Data
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class Arm:
    """One budget-matched comparison arm: CO against `tag`."""

    tag: str
    gap_col: str
    label: str


ARMS = (
    Arm("NC", "gap_offray", "NC  (pre-registered bbox, PRIMARY)"),
    Arm("NCM", "gap_offray_ncm", "NCM (N-matched, Amendment 2)"),
)


def load(n_rays: int = 1, results: str = "results.csv", margins: str = "margins.csv") -> pd.DataFrame:
    res = pd.read_csv(WORK / results)
    mar = pd.read_csv(WORK / margins)
    df = res.merge(mar, on=["cell", "window", "ray_bin", "n_rays", "seed"], how="left", validate="many_to_one")
    assert df[f"CO_m_beta{BETA_PRIMARY}"].notna().all(), "margins did not join onto every row"
    df = df[df["n_rays"] == n_rays].copy()
    df["stratum"] = df["N_gm"].map(coarse_stratum)
    df["logN"] = np.log10(df["N_gm"])
    return df


def add_margins(df: pd.DataFrame, beta: float) -> pd.DataFrame:
    """Attach m_X at `beta` (kappa = 100) and the Delta regressors of A3.1.

    Non-finite m_CO (V_K = 0, a length-1 merged ratio list) is winsorised to the
    minimum finite m_CO over the analysis set, per A3.1. Complete-case columns are
    carried alongside so the disclosed alternative can be run unchanged.
    """
    df = df.copy()
    for arm in ("CO", "NC", "NCM"):
        df[f"m_{arm}"] = df[f"{arm}_m_beta{beta}"].replace([np.inf, -np.inf], np.nan)
    floor = df["m_CO"].min()
    df["m_CO_degenerate"] = df["m_CO"].isna()
    df["m_CO_cc"] = df["m_CO"]
    df["m_CO"] = df["m_CO"].fillna(floor)
    for arm in ("NC", "NCM"):
        df[f"Delta_{arm}"] = df[f"m_{arm}"] - df["m_CO"]
        df[f"Delta_{arm}_cc"] = df[f"m_{arm}"] - df["m_CO_cc"]
    return df


def regimes(df: pd.DataFrame, kappa: float, tag: str) -> pd.Series:
    """A3 fact (i): m(kappa) = m(100) + log10(kappa/100); Regime B iff m(kappa) >= 0."""
    shift = np.log10(kappa / 100.0)
    co = np.where(df["m_CO_cc"].fillna(-np.inf) + shift >= 0, "B", "A")
    nc = np.where(df[f"m_{tag}"] + shift >= 0, "B", "A")
    return pd.Series([f"CO={a},NC={b}" for a, b in zip(co, nc)], index=df.index)


# ════════════════════════════════════════════════════════════════════════════
# A3.1  primary
# ════════════════════════════════════════════════════════════════════════════

def primary(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []

    def record(kind: str, spec: str, arm: str, reg: str, est: Estimate, note: str = "") -> None:
        rows.append({
            "kind": kind, "spec": spec, "arm": arm, "regressor": reg,
            "slope": est.value, "ci_lo": est.lo, "ci_hi": est.hi,
            "n_pairs": est.n_obs, "n_windows": est.n_clusters,
            "excludes_zero": est.excludes_zero, "note": note,
        })

    print("=" * 104)
    print("A3.1  PRIMARY - continuous: off-ray gap regressed on the identifiability margin gap")
    print("      gap = log10(RMSE_offray(CO)/RMSE_offray(NC));  Delta = m_NC - m_CO, in decades")
    print("      theory predicts slope > 0.  Delta is exactly kappa-invariant (Amendment 3, fact ii)")
    print("=" * 104)
    for arm in ARMS:
        est = regress(df, arm.gap_col, [f"Delta_{arm.tag}"], "cell",
                      rng_for("primary", arm.tag))[f"Delta_{arm.tag}"]
        kind = "PRIMARY" if arm.tag == "NC" else "pre-registered companion"
        record(kind, "cell FE", arm.tag, "Delta", est)
        print(f"  {arm.label:38s} beta_delta = {est}   pairs={est.n_obs} windows={est.n_clusters}"
              f"{'   CI EXCLUDES ZERO' if est.excludes_zero else '   CI contains zero'}")

    print("\n  A4.4 suite-weighted pooled slope (each SUITE weighted equally; disclosed companion)")
    for arm in ARMS:
        for spec, fe in (("no fixed effect, suite-weighted", None),
                         ("cell FE, suite-weighted", "cell")):
            est = regress(df, arm.gap_col, [f"Delta_{arm.tag}"], fe,
                          rng_for("suitew", arm.tag, spec), suite_weighted=True)[f"Delta_{arm.tag}"]
            record("disclosed companion (A4.4)", spec, arm.tag, "Delta", est)
            print(f"    {arm.tag:4s} {spec:32s} {est}  pairs={est.n_obs}"
                  f"{'  excludes zero' if est.excludes_zero else ''}")

    print("\n  pre-registered companions to the primary (A3.1)")
    for arm in ARMS:
        for spec, fe, col in (
            ("no fixed effect", None, f"Delta_{arm.tag}"),
            ("(cell,window) FE", "cellwindow", f"Delta_{arm.tag}"),
            ("cell FE, complete case", "cell", f"Delta_{arm.tag}_cc"),
        ):
            est = regress(df, arm.gap_col, [col], fe, rng_for("companion", arm.tag, spec))[col]
            record("pre-registered companion", spec, arm.tag, "Delta", est)
            print(f"    {arm.tag:4s} {spec:24s} {est}  pairs={est.n_obs}")
        est = regress(df, arm.gap_col, ["m_CO"], "cell", rng_for("mco", arm.tag))["m_CO"]
        record("pre-registered companion", "cell FE, regressor = m_CO (predicted < 0)",
               arm.tag, "m_CO", est)
        print(f"    {arm.tag:4s} {'secondary regressor m_CO':24s} {est}  (predicted negative)")

    print("\n  per law and per cell (pre-registered companion)")
    for arm in ARMS:
        for law, sub in df.groupby("law"):
            est = regress(sub, arm.gap_col, [f"Delta_{arm.tag}"], "cell",
                          rng_for("law", arm.tag, law))[f"Delta_{arm.tag}"]
            record("pre-registered companion", f"law = {law}", arm.tag, "Delta", est)
            print(f"    {arm.tag:4s} law={law:12s} {est}  pairs={est.n_obs}")
        for suite, sub in df.groupby(suite_of(df["cell"])):
            fe = "cell" if sub["cell"].nunique() > 1 else None
            est = regress(sub, arm.gap_col, [f"Delta_{arm.tag}"], fe,
                          rng_for("suite", arm.tag, suite))[f"Delta_{arm.tag}"]
            record("A4.5 per suite", f"suite = {suite}", arm.tag, "Delta", est,
                   f"{sub['cell'].nunique()} cells")
            if arm.tag == "NC":
                print(f"    {arm.tag:4s} SUITE {suite:28s} {est}  pairs={est.n_obs} "
                      f"windows={est.n_clusters} cells={sub['cell'].nunique()}"
                      f"{'  excludes zero' if est.excludes_zero else ''}")
        for cell, sub in df.groupby("cell"):
            est = regress(sub, arm.gap_col, [f"Delta_{arm.tag}"], None,
                          rng_for("cell", arm.tag, cell))[f"Delta_{arm.tag}"]
            record("pre-registered companion", f"cell = {cell}", arm.tag, "Delta", est,
                   "single window" if est.n_clusters < 2 else "")
            if arm.tag == "NC":
                print(f"    {arm.tag:4s} {cell:44s} {est}  pairs={est.n_obs} "
                      f"windows={est.n_clusters}")

    print("\n  C2 beta grid applied to the primary (pre-registered companion)")
    base = df
    for beta in BETA_GRID:
        d = add_margins(base, beta)
        for arm in ARMS:
            est = regress(d, arm.gap_col, [f"Delta_{arm.tag}"], "cell",
                          rng_for("beta", arm.tag, beta))[f"Delta_{arm.tag}"]
            record("pre-registered companion", f"beta_eff = {beta}", arm.tag, "Delta", est)
            if arm.tag == "NC":
                print(f"    beta_eff={beta:<5} {est}  pairs={est.n_obs}"
                      f"{'  excludes zero' if est.excludes_zero else ''}")
    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════════════════════
# A3.2 / A3.3  dose-response and the regime-conditional tests
# ════════════════════════════════════════════════════════════════════════════

CELL_ORDER = ["CO=A,NC=B", "CO=A,NC=A", "CO=B,NC=B", "CO=B,NC=A"]
CELL_ROLE = {
    "CO=A,NC=B": "separated (Theorem 1 predicts gap > 0)",
    "CO=A,NC=A": "same regime (theory predicts no ordering) - FALSIFICATION",
    "CO=B,NC=B": "same regime (theory predicts no ordering) - FALSIFICATION",
    "CO=B,NC=A": "reversed (Theorem 1 predicts gap < 0)",
}


def dose_response(df: pd.DataFrame, stratum: str = "all") -> pd.DataFrame:
    rows: list[dict] = []
    for arm in ARMS:
        for kappa in KAPPA_GRID:
            cells = regimes(df, kappa, arm.tag)
            for name in CELL_ORDER:
                sub = df[cells == name]
                rng = rng_for("dose", stratum, arm.tag, kappa, name)
                g = sub[arm.gap_col].to_numpy(float)
                g = g[np.isfinite(g)]
                med = boot_stat(sub[arm.gap_col].to_numpy(float), rng)
                medc = boot_stat_clustered(sub, arm.gap_col, rng)
                wins = int((g > 0).sum())
                win = boot_stat(np.asarray(g > 0, float), rng, stat=np.mean)
                p = binomtest(wins, len(g)).pvalue if len(g) else np.nan
                rows.append({
                    "stratum": stratum, "arm": arm.tag, "kappa_target": kappa,
                    "regime_cell": name, "role": CELL_ROLE[name],
                    "n_pairs": len(g), "n_windows": medc.n_clusters,
                    "gap_median": med.value, "ci_lo": med.lo, "ci_hi": med.hi,
                    "ci_lo_cluster": medc.lo, "ci_hi_cluster": medc.hi,
                    "nc_win_rate": win.value, "win_lo": win.lo, "win_hi": win.hi,
                    "sign_test_p": p,
                    "excludes_zero_pairboot": med.excludes_zero,
                    "excludes_zero_clusterboot": medc.excludes_zero,
                })
    return pd.DataFrame(rows)


def print_dose(dr: pd.DataFrame, title: str) -> None:
    print("\n" + "=" * 118)
    print(title)
    print("=" * 118)
    print(f"{'arm':5s} {'kappa':>7s} {'regime cell':11s} {'pairs':>6s} {'wnds':>6s} "
          f"{'gap':>8s} {'95% CI (pairs)':>19s} {'95% CI (windows)':>19s} {'NCwin%':>7s} {'signp':>8s} role")
    for r in dr.itertuples():
        if r.n_pairs == 0:
            print(f"{r.arm:5s} {r.kappa_target:7.0e} {r.regime_cell:11s} {0:6d}      -        -"
                  f"{'':>21s}{'':>20s}       -        -  {r.role}")
            continue
        print(f"{r.arm:5s} {r.kappa_target:7.0e} {r.regime_cell:11s} {r.n_pairs:6d} {r.n_windows:6d} "
              f"{r.gap_median:8.3f} [{r.ci_lo:8.3f},{r.ci_hi:8.3f}] [{r.ci_lo_cluster:8.3f},{r.ci_hi_cluster:8.3f}] "
              f"{100 * r.nc_win_rate:6.1f}% {r.sign_test_p:8.1e}  {r.role}")


# ════════════════════════════════════════════════════════════════════════════
# A3.4  scale
# ════════════════════════════════════════════════════════════════════════════

def scale(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    df = df.copy()
    df["logN_c"] = df["logN"] - df["logN"].mean()

    print("\n" + "=" * 104)
    print("A3.4  SCALE - does the conditional effect attenuate with model size?")
    print("      gap ~ b0*Delta + b1*(Delta x log10 N_gm, centred) + b2*log10 N_gm + cell FE")
    print("=" * 104)
    for arm in ARMS:
        d = f"Delta_{arm.tag}"
        df["_ix"] = df[d] * df["logN_c"]
        est = regress(df, arm.gap_col, [d, "_ix", "logN_c"], "cell",
                      rng_for("scale", arm.tag))
        for name, key in (("b0 Delta", d), ("b1 Delta x logN", "_ix"), ("b2 logN", "logN_c")):
            e = est[key]
            rows.append({"kind": "interaction", "arm": arm.tag, "stratum": "all", "term": name,
                         "value": e.value, "ci_lo": e.lo, "ci_hi": e.hi,
                         "n_pairs": e.n_obs, "n_windows": e.n_clusters,
                         "excludes_zero": e.excludes_zero})
            print(f"  {arm.tag:4s} {name:18s} {e}"
                  f"{'   CI EXCLUDES ZERO' if e.excludes_zero else ''}")
        # CRITERIA E3: translate the interval on b1 into the per-decade change in the
        # conditional effect that the data cannot rule out, as a percentage of b0.
        b0, b1 = est[d].value, est["_ix"]
        print(f"  {arm.tag:4s} {'E3 translation':18s} per decade of N the data cannot rule out "
              f"anything from {100 * b1.lo / b0:+.0f}% to {100 * b1.hi / b0:+.0f}% of the "
              f"conditional effect (b0 = {b0:+.4f})")
        rows.append({"kind": "E3", "arm": arm.tag, "stratum": "all",
                     "term": "b1 as pct of b0 per decade", "value": 100 * b1.value / b0,
                     "ci_lo": 100 * b1.lo / b0, "ci_hi": 100 * b1.hi / b0,
                     "n_pairs": b1.n_obs, "n_windows": b1.n_clusters, "excludes_zero": b1.excludes_zero})

    print("\n  beta_delta estimated separately inside each coarse stratum (W2)")
    for arm in ARMS:
        for stratum in ["<100M", "100M-1B", ">=1B"]:
            sub = df[df["stratum"] == stratum]
            if sub.empty:
                continue
            e = regress(sub, arm.gap_col, [f"Delta_{arm.tag}"], "cell",
                        rng_for("stratum", arm.tag, stratum))[f"Delta_{arm.tag}"]
            rows.append({"kind": "stratum", "arm": arm.tag, "stratum": stratum, "term": "beta_delta",
                         "value": e.value, "ci_lo": e.lo, "ci_hi": e.hi,
                         "n_pairs": e.n_obs, "n_windows": e.n_clusters,
                         "excludes_zero": e.excludes_zero})
            print(f"    {arm.tag:4s} {stratum:9s} beta_delta = {e}  pairs={e.n_obs} windows={e.n_clusters}")
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="5-seed conditional analysis (CRITERIA Amendment 3): primary, dose-response, scale.")
    ap.add_argument("--results", default="results.csv")
    ap.add_argument("--margins", default="margins.csv")
    ap.add_argument("--prefix", default="conditional")
    args = ap.parse_args(argv)
    pre = args.prefix

    raw = load(n_rays=1, results=args.results, margins=args.margins)
    df = add_margins(raw, BETA_PRIMARY)

    print(f"single-ray design pairs = {len(df)}, windows = {df.groupby(CLUSTER).ngroups}, "
          f"cells = {df['cell'].nunique()}, suites = {suite_of(df['cell']).nunique()}")
    print(f"degenerate CO designs (V_K = 0, winsorised per A3.1) = {int(df['m_CO_degenerate'].sum())}")
    print(f"m_CO   median {df['m_CO'].median():+.3f}   m_NC median {df['m_NC'].median():+.3f}   "
          f"Delta_NC median {df['Delta_NC'].median():+.3f}  "
          f"[{df['Delta_NC'].min():+.3f}, {df['Delta_NC'].max():+.3f}]\n")

    primary(df).to_csv(WORK / f"{pre}_primary.csv", index=False)

    dr_all = dose_response(df)
    print_dose(dr_all, "A3.2 / A3.3  DOSE-RESPONSE over the pre-registered kappa grid, all pairs")
    parts = [dr_all]
    for stratum in ["<100M", "100M-1B", ">=1B"]:
        sub = df[df["stratum"] == stratum]
        if sub.empty:
            continue
        d = dose_response(sub, stratum=stratum)
        parts.append(d)
        print_dose(d[d["arm"] == "NC"], f"A3.4  same table, NC arm, stratum {stratum}")
    pd.concat(parts).to_csv(WORK / f"{pre}_doseresponse.csv", index=False)

    scale(df).to_csv(WORK / f"{pre}_scale.csv", index=False)

    print("\n" + "=" * 104)
    print("SECONDARY: two-ray CO (n_rays = 2), primary regression only")
    print("=" * 104)
    two = add_margins(load(n_rays=2, results=args.results, margins=args.margins), BETA_PRIMARY)
    for arm in ARMS:
        e = regress(two, arm.gap_col, [f"Delta_{arm.tag}"], "cell",
                    rng_for("tworay", arm.tag))[f"Delta_{arm.tag}"]
        print(f"  {arm.tag:4s} beta_delta = {e}  pairs={e.n_obs}")


if __name__ == "__main__":
    main()
