#!/usr/bin/env python3
"""Exact-but-fast cluster bootstrap for the CRITERIA Amendment 5 estimators.

The estimator is the one already pre-registered in E2 / A3.1: OLS of the off-ray gap on
the identifiability-margin regressors with a per-cell fixed effect, and a percentile
bootstrap that resamples clusters WITHIN cell. This module changes only the arithmetic,
not the estimator.

By Frisch-Waugh, the coefficients on the regressors of interest are unchanged if the cell
dummies are replaced by within-cell centring, and the centred normal equations depend on
the resample only through per-cluster sums. So each cluster's sufficient statistics

    n,  sum x,  sum y,  sum x x^T,  sum x y

are accumulated once, and a bootstrap draw is a weighted sum of them followed by one
p x p solve. That makes 200000 resamples cost about the same as 2000 did, which is the
point of A5.2. Agreement with `conditional.regress` is asserted by `estimate.py`
before any Amendment-5 number is reported.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Estimate:
    value: float
    lo: float
    hi: float
    n_obs: int
    n_clusters: int
    lo_one_sided: float = np.nan
    draws: np.ndarray | None = None

    def __str__(self) -> str:
        return f"{self.value:+.4f} [{self.lo:+.4f}, {self.hi:+.4f}]"

    @property
    def excludes_zero(self) -> bool:
        return np.isfinite(self.lo) and np.isfinite(self.hi) and (self.lo > 0 or self.hi < 0)


class ClusterOLS:
    """Within-cell OLS on `x_cols` with a cluster bootstrap over `cluster_cols`.

    `weights` is the A4.4 suite weighting when it is used; None otherwise.
    """

    def __init__(self, df: pd.DataFrame, y: str, x_cols: Sequence[str],
                 cluster_cols: Sequence[str] = ("cell", "window"),
                 weights: np.ndarray | None = None, order_col: str | None = None,
                 fe_col: str = "cell", group_col: str = "cell"):
        keep = np.isfinite(df[y].to_numpy(float))
        for c in x_cols:
            keep &= np.isfinite(df[c].to_numpy(float))
        df = df.loc[keep]
        self.n_obs = len(df)
        self.p = p = len(x_cols)

        x = np.column_stack([df[c].to_numpy(float) for c in x_cols])
        yv = df[y].to_numpy(float)
        w = np.ones(len(df)) if weights is None else np.asarray(weights, float)[keep]

        cell_codes, self.cells = pd.factorize(df[fe_col], sort=True)
        group_codes, self.groups = pd.factorize(df[group_col], sort=True)
        key = pd.MultiIndex.from_frame(df[list(cluster_cols)])
        unit_codes, units = pd.factorize(key, sort=True)
        self.n_clusters = len(units)
        self.units = units

        # per-unit sufficient statistics, packed into one row each
        m = 1 + p + 1 + p * p + p
        z = np.empty((len(df), m))
        z[:, 0] = w
        z[:, 1:1 + p] = x * w[:, None]
        z[:, 1 + p] = yv * w
        z[:, 2 + p:2 + p + p * p] = (x[:, :, None] * x[:, None, :]).reshape(len(df), p * p) * w[:, None]
        z[:, 2 + p + p * p:] = x * (yv * w)[:, None]

        stats = np.zeros((self.n_clusters, m))
        np.add.at(stats, unit_codes, z)
        self.stats = stats
        self.unit_cell = np.zeros(self.n_clusters, int)
        self.unit_cell[unit_codes] = cell_codes
        self.unit_group = np.zeros(self.n_clusters, int)
        self.unit_group[unit_codes] = group_codes
        self.n_cells = len(self.cells)
        self.m = m

        # unit ordering inside a cell, for the moving-block bootstrap
        if order_col is not None:
            vals = df[order_col].to_numpy(float)
            o = np.zeros(self.n_clusters)
            o[unit_codes] = vals
            self.unit_order = o
        else:
            self.unit_order = np.arange(self.n_clusters, dtype=float)

        # `by_cell` collects the clusters that share a fixed effect (used to aggregate);
        # `by_group` collects the clusters resampled together (E2 resamples within cell).
        self.by_cell = [np.flatnonzero(self.unit_cell == c) for c in range(self.n_cells)]
        self.by_group = []
        for g in range(len(self.groups)):
            idx = np.flatnonzero(self.unit_group == g)
            self.by_group.append(idx[np.argsort(self.unit_order[idx], kind="stable")])

    # ── the estimator itself ────────────────────────────────────────────────
    def _solve(self, agg: np.ndarray) -> np.ndarray:
        """agg: (B, n_cells, m) -> (B, p) within-cell OLS coefficients."""
        p = self.p
        n = agg[..., 0]
        sx = agg[..., 1:1 + p]
        sy = agg[..., 1 + p]
        sxx = agg[..., 2 + p:2 + p + p * p].reshape(*agg.shape[:-1], p, p)
        sxy = agg[..., 2 + p + p * p:]
        safe = np.where(n > 0, n, 1.0)[..., None]
        a = sxx - (sx[..., :, None] * sx[..., None, :]) / safe[..., None]
        b = sxy - sx * (sy / np.where(n > 0, n, 1.0))[..., None]
        a = np.where((n > 0)[..., None, None], a, 0.0).sum(axis=-3)
        b = np.where((n > 0)[..., None], b, 0.0).sum(axis=-2)
        return np.einsum("...ij,...j->...i", np.linalg.pinv(a), b)

    def point(self) -> np.ndarray:
        agg = np.zeros((1, self.n_cells, self.m))
        np.add.at(agg[0], self.unit_cell, self.stats)
        return self._solve(agg)[0]

    # ── resampling schemes ──────────────────────────────────────────────────
    def _counts(self, rng: np.random.Generator, b: int, block: int) -> np.ndarray:
        """(B, n_clusters) resample multiplicities, drawn WITHIN cell."""
        counts = np.zeros((b, self.n_clusters))
        for idx in self.by_group:
            k = len(idx)
            if k == 0:
                continue
            if block <= 1 or k <= block:
                draw = rng.multinomial(k, np.full(k, 1.0 / k), size=b).astype(float)
            else:
                nb = int(np.ceil(k / block))
                starts = rng.integers(0, k, size=(b, nb))
                sel = (starts[:, :, None] + np.arange(block)[None, None, :]) % k
                sel = sel.reshape(b, -1)[:, :k]
                draw = np.zeros((b, k))
                np.add.at(draw, (np.arange(b)[:, None], sel), 1.0)
            counts[:, idx] = draw
        return counts

    def boot(self, rng: np.random.Generator, n_boot: int, block: int = 1,
             chunk: int | None = None) -> np.ndarray:
        if chunk is None:
            chunk = int(max(200, min(20000, 2e7 / max(1, self.n_clusters))))
        out = np.empty((n_boot, self.p))
        done = 0
        while done < n_boot:
            b = min(chunk, n_boot - done)
            counts = self._counts(rng, b, block)
            agg = np.zeros((b, self.n_cells, self.m))
            for c in range(self.n_cells):
                idx = self.by_cell[c]
                if len(idx):
                    agg[:, c, :] = counts[:, idx] @ self.stats[idx]
            out[done:done + b] = self._solve(agg)
            done += b
        return out

    def estimate(self, rng: np.random.Generator, n_boot: int, block: int = 1,
                 index: int = 0, keep_draws: bool = False) -> Estimate:
        pt = self.point()[index]
        d = self.boot(rng, n_boot, block)[:, index]
        lo, hi = np.percentile(d, [2.5, 97.5])
        one = float(np.percentile(d, 5.0))
        return Estimate(float(pt), float(lo), float(hi), self.n_obs, self.n_clusters,
                        one, d if keep_draws else None)

    def estimate_combo(self, rng: np.random.Generator, n_boot: int, combo: np.ndarray,
                       block: int = 1, keep_draws: bool = False) -> Estimate:
        """A linear combination c^T beta - used for the A5.4(c) slope at a given log N."""
        pt = float(self.point() @ combo)
        d = self.boot(rng, n_boot, block) @ combo
        lo, hi = np.percentile(d, [2.5, 97.5])
        return Estimate(pt, float(lo), float(hi), self.n_obs, self.n_clusters,
                        float(np.percentile(d, 5.0)), d if keep_draws else None)
