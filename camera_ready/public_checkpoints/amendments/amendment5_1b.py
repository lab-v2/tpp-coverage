#!/usr/bin/env python3
"""Every specification CRITERIA Amendment 5 fixes for windows centred at 1B and above.

A5.5 requires each of them to be reported. They are computed with the released estimator
(`estimate.prepare`, `cluster_bootstrap.ClusterOLS`), with these seeds and resample counts:

- pre-registered: W1 windows, cell fixed effect, window-cluster bootstrap, 200,000 draws,
  seed 20260802;
- A5.3: moving-block bootstrap over each cell's window sequence, L = ceil(width / stride),
  on W1 at 512 seeds, and on the six constructions at 128 seeds (their fit tables
  ``results_E_win<V>.csv`` exist for the five fully trained releases only);
- A5.4(a): pairs resampled within cell, 20,000 draws (anti-conservative, as A5.4 says);
- A5.4(b): (cell, window, ray) clusters;
- companion: a fixed effect per window, seed 99;
- A5.4(c): a slope smooth in log N, linear and quadratic, read at the band's pair-weighted
  centroid and at log10 N = 9.0, 9.5 and 10.0, seed 4242, and its interaction term b1;
- the second non-collinear subset (Amendment 2), with the window cluster and smooth;
- A5.4(d): one-sided 95% lower bounds, and the attenuation 1 - beta(>=1B) / beta(<1B),
  unpooled (independent window-cluster draws per band) and smooth (one draw of the curve).

Percent is 100 (10 ** beta - 1), the paper's unit. ``N_at`` is the model size a smooth slope
is read at (for b1, the centre of log N).

Reads ``results_E_seeds512.csv``, ``results_X_seeds512.csv.gz`` and
``results_E_win{V1,V2,V3,V4,V7}.csv`` (enumerate_seeds.py, ``--construction V --seeds 128``)
and writes ``amendment5_1b.csv``. Relative paths are taken in the work directory.

    uv run python amendments/amendment5_1b.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the reanalysis modules
import estimate as EST
from cluster_bootstrap import ClusterOLS, Estimate
from paths import WORK

BIG, PAIRBOOT = 200_000, 20_000
TOP = ">=1B"
CONSTRUCTIONS = {name: EST.CONSTRUCTIONS[name] for name in ("W1", "V1", "V2", "V3", "V4", "V7")}
ARMS = (("NC", "gap_offray"), ("NCM", "gap_offray_ncm"))
SMOOTH = {  # regressors after Delta, and the combination that reads the slope at centred t
    "linear": (["_ix", "logN_c"], lambda t: [1.0, t, 0.0]),
    "quadratic": (["_ix", "_ix2", "logN_c", "_l2"], lambda t: [1.0, t, t * t, 0.0, 0.0]),
}
pct = EST.pct


def centred(d: pd.DataFrame, x: str) -> tuple[pd.DataFrame, float]:
    """`d` with log10 N_gm centred at its pair mean and the size terms of regressor `x`."""
    ctr = float(d["logN"].mean())
    lc = d["logN"] - ctr
    return d.assign(logN_c=lc, _ix=d[x] * lc, _ix2=d[x] * lc ** 2, _l2=lc ** 2), ctr


def smooth_engine(d: pd.DataFrame, gap: str, x: str, form: str = "linear") -> ClusterOLS:
    """The A5.4(c) regression of `gap` on `x` with a slope smooth in centred log N."""
    return ClusterOLS(d, gap, [x, *SMOOTH[form][0]], order_col="window")


def slope(df: pd.DataFrame, gap: str, delta: str, cluster=("cell", "window"), n_boot: int = BIG,
          block: int = 1, seed: int = 20260802, keep: bool = False) -> Estimate:
    eng = ClusterOLS(df, gap, [delta], cluster_cols=cluster, order_col="window")
    return eng.estimate(np.random.default_rng(seed), n_boot, block=block, keep_draws=keep)


class Report:
    """The output rows, printed as they are recorded."""

    def __init__(self) -> None:
        self.rows: list[dict] = []

    def add(self, corpus: str, spec: str, arm: str, e: Estimate, n_at: float = np.nan) -> None:
        self.rows.append(dict(corpus=corpus, spec=spec, arm=arm, beta=e.value, lo=e.lo, hi=e.hi,
                              pct=pct(e.value), pct_lo=pct(e.lo), pct_hi=pct(e.hi),
                              pct_one_sided_lo=pct(e.lo_one_sided), n_pairs=e.n_obs,
                              n_windows=e.n_clusters, excludes0=e.excludes_zero, N_at=n_at,
                              note=""))
        print(f"  {corpus} {arm:3s} {spec:58s} {pct(e.value):+6.1f}% [{pct(e.lo):+6.1f}%, "
              f"{pct(e.hi):+6.1f}%]  one-sided lo {pct(e.lo_one_sided):+6.1f}%  "
              f"windows {e.n_clusters}")

    def attenuation(self, corpus: str, name: str, a1: np.ndarray, a0: np.ndarray) -> None:
        frac, diff = 1.0 - a1 / a0, a0 - a1
        self.rows.append(dict(
            corpus=corpus, spec=f"A5.4d attenuation fraction, {name}", arm="NC",
            beta=float(np.median(frac)), lo=float(np.percentile(frac, 2.5)),
            hi=float(np.percentile(frac, 97.5)), pct_one_sided_lo=float(np.percentile(frac, 95)),
            note=f"95th percentile = largest attenuation excluded at 5%; difference "
                 f"{np.mean(a0) - np.mean(a1):+.4f} dex [{np.percentile(diff, 2.5):+.4f}, "
                 f"{np.percentile(diff, 97.5):+.4f}]"))
        print(f"  {corpus} A5.4d {name}: attenuation fraction median {100 * np.median(frac):+.0f}%,"
              f" 95th percentile {100 * np.percentile(frac, 95):+.0f}%")


def run(rep: Report, corpus: str, big: pd.DataFrame, win_dir: Path | None) -> None:
    hi = big[big["stratum"] == TOP]
    for arm, gap in ARMS:
        rep.add(corpus, "pre-registered (W1, window cluster)", arm, slope(hi, gap, f"Delta_{arm}"))
    rep.add(corpus, "A5.3 W1, moving block L=4", "NC", slope(hi, "gap_offray", "Delta_NC", block=4))
    if win_dir is not None:
        for name, (w, s) in CONSTRUCTIONS.items():
            d = big[big["seed"] < 128] if name == "W1" else \
                EST.prepare(win_dir / f"results_E_win{name}.csv")
            d = d[d["stratum"] == TOP]
            L = int(np.ceil(w / s))
            rep.add(corpus, f"A5.3 {name} ({w},{s}), 128 seeds, window cluster", "NC",
                    slope(d, "gap_offray", "Delta_NC"))
            rep.add(corpus, f"A5.3 {name} ({w},{s}), 128 seeds, moving block L={L}", "NC",
                    slope(d, "gap_offray", "Delta_NC", block=L))
    for arm, gap in ARMS:
        rep.add(corpus, "A5.4a pair cluster (anti-conservative)", arm,
                slope(hi, gap, f"Delta_{arm}", cluster=("cell", "row_id"), n_boot=PAIRBOOT))
        rep.add(corpus, "A5.4b (cell, window, ray) cluster", arm,
                slope(hi, gap, f"Delta_{arm}", cluster=("cell", "ray_id")))
    e_cw = ClusterOLS(hi, "gap_offray", ["Delta_NC"], order_col="window", fe_col="cell_win",
                      group_col="cell").estimate(np.random.default_rng(99), BIG)
    rep.add(corpus, "companion: fixed effect per window", "NC", e_cw)

    smooth: dict[str, np.ndarray] = {}
    for arm, gap in ARMS:
        d, ctr = centred(big, f"Delta_{arm}")
        l1 = d.loc[d["stratum"] == TOP, "logN"].mean() - ctr
        l0 = d.loc[d["stratum"] != TOP, "logN"].mean() - ctr
        for form, (_, combo) in SMOOTH.items():
            eng = smooth_engine(d, gap, f"Delta_{arm}", form)
            rep.add(corpus, f"A5.4c smooth {form}, at the >=1B centroid", arm,
                    eng.estimate_combo(np.random.default_rng(4242), BIG, np.array(combo(l1))),
                    10 ** (l1 + ctr))
            rep.add(corpus, f"A5.4c smooth {form}, interaction b1", arm,
                    eng.estimate(np.random.default_rng(4242), BIG, index=1), 10 ** ctr)
            for lg in (9.0, 9.5, 10.0):
                rep.add(corpus, f"A5.4c smooth {form}, at log10 N = {lg}", arm,
                        eng.estimate_combo(np.random.default_rng(4242), BIG,
                                           np.array(combo(lg - ctr))), 10 ** lg)
            if arm == "NC" and form == "linear":
                b = eng.boot(np.random.default_rng(4242), BIG)
                smooth = dict(s1=b @ np.array(combo(l1)), s0=b @ np.array(combo(l0)))
    d1 = slope(hi, "gap_offray", "Delta_NC", keep=True).draws
    d0 = slope(big[big["stratum"] != TOP], "gap_offray", "Delta_NC", keep=True).draws
    rep.attenuation(corpus, "unpooled", d1, d0)
    rep.attenuation(corpus, "smooth", smooth["s1"], smooth["s0"])


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--results", default="results_E_seeds512.csv")
    ap.add_argument("--results-x", default="results_X_seeds512.csv.gz")
    ap.add_argument("--skip-x", action="store_true", help="run on the E corpus alone")
    ap.add_argument("--windows-dir", default=".",
                    help="folder of results_E_win<V>.csv; 'none' skips the A5.3 constructions")
    ap.add_argument("--out", default="amendment5_1b.csv")
    args = ap.parse_args(argv)

    rep = Report()
    corpora = {"E": args.results} if args.skip_x else {"E": args.results, "X": args.results_x}
    for corpus, path in corpora.items():
        print(f"== {corpus}")
        win_dir = None if corpus != "E" or args.windows_dir == "none" else WORK / args.windows_dir
        run(rep, corpus, EST.prepare(WORK / path), win_dir)
    pd.DataFrame(rep.rows).to_csv(WORK / args.out, index=False)


if __name__ == "__main__":
    main()
