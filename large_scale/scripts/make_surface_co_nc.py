#!/usr/bin/env python
"""Dual Chinchilla surface (CO vs NC), the representative pair from the CO/NC
sweep, drawn with the companion paper's surface code (surface_helpers +
panel_style).

Among valid pairs (matched training-cell count, V_K_NC > V_K_CO) this is the one
with the largest R2 margin psi = R2_NC - R2_CO on the 354M frontier, i.e. the
clearest case of a non-collinear design beating a collinear one:
  CO = {136, 200}  (collinear, tiny V_K)   -> frontier R2 ~ -1.95
  NC = {20, 43}    (spread, larger V_K)     -> frontier R2 ~ +0.79
Both fit Chinchilla on the training cells (N <= 85M, 6 each) at n_restarts=20
(matching the sweep) and are scored by extrapolation R2 on the 354M frontier
(holdout, drawn as points + residual sticks). V_K = Var(v^-beta), beta = 0.971.

Output: figures/surface_co_vs_nc.png . Uses the paper's code in this repository (TPP_CODE_REPO selects another checkout).
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.random.seed(42)
REPO = Path(__file__).resolve().parents[1]
# The paper's code is this repository's src/tpp_analysis; TPP_CODE_REPO selects another checkout.
_code = os.environ.get("TPP_CODE_REPO") or str(REPO.parent)
_pkg = Path(_code).expanduser() / "src" / "tpp_analysis"
if not (_pkg / "visualizations" / "surface_helpers.py").is_file():
    sys.exit(f"paper visualizations not found under {_pkg}")
sys.path.insert(0, str(_pkg))
sys.path.insert(0, str(_pkg / "visualizations"))

from scaling_laws.laws import chinchilla_loss, fit_chinchilla          # paper's code
from surface_helpers import _build_adaptive_rgba, _draw_boundary_edges_occlusion
from panel_style import SURF_W, SURF_H, AXIS_LABEL_FS, LEGEND_FS, LEGEND_ALPHA, style_ticks_3d

CO_COLOR, NC_COLOR = "steelblue", "indianred"
SIDE_ELEV, SIDE_AZIM = 16, -50
TARGETS = np.array([20, 29, 43, 63, 93, 136, 200], float)
BETA_EFF = 0.971
FULL = [29, 43, 63, 93]   # (unused; kept for reference)
CO_TPPS = [136, 200]      # representative pair (max R2 margin) from scripts/co_nc_sweep.py
NC_TPPS = [20, 43]
HOLDOUT_N = 354196480     # 354M frontier = extrapolation holdout
N_RESTARTS = 20           # match the sweep (co_nc_sweep.py)

# ---- helpers (kappa_AB, value-based V_K, r2 from analyze.py conventions) ----
def chin_jac(N, D, p):
    E, A, al, B, be = p
    return np.column_stack([
        np.ones_like(N), np.power(N, -al), -A * np.power(N, -al) * np.log(N),
        np.power(D, -be), -B * np.power(D, -be) * np.log(D),
    ])


def kappa_AB(N, D, p, iA=1, iB=3):
    J = chin_jac(N, D, p)[:, [iA, iB]]
    w = np.linalg.eigvalsh(J.T @ J)
    return float(w[-1] / max(w[0], 1e-300))


def V_K_values(S):
    u = np.asarray(S, float) ** (-BETA_EFF)
    return float(u.var())


def r2(actual, pred):
    actual, pred = np.asarray(actual, float), np.asarray(pred, float)
    ss = np.sum((actual - pred) ** 2)
    st = np.sum((actual - actual.mean()) ** 2)
    return float(1 - ss / st) if st > 0 else float("nan")


# ---- first pair from the sweep: CO/NC fixed, fit on training, holdout = frontier ----
env = pd.read_csv(REPO / "results" / "envelope.csv")[["N", "Dt", "L", "TPPd"]].copy()
env["tgt"] = [int(TARGETS[np.argmin(np.abs(np.log(TARGETS) - np.log(t)))]) for t in env["TPPd"]]
CO, NC = CO_TPPS, NC_TPPS
train = env[env["N"] < HOLDOUT_N]
co = train[train["tgt"].isin(CO)]
nc = train[train["tgt"].isin(NC)]
holdout = env[env["N"] == HOLDOUT_N]


def fit(frame):
    N, D, L = frame["N"].values, frame["Dt"].values, frame["L"].values
    return np.asarray(fit_chinchilla(N, D, L, n_restarts=N_RESTARTS)[0], float), N, D, L


co_p, coN, coD, coL = fit(co)
nc_p, ncN, ncD, ncL = fit(nc)
hN, hD, hL = holdout["N"].values, holdout["Dt"].values, holdout["L"].values
r2h_co = r2(hL, chinchilla_loss(hN, hD, *co_p))
r2h_nc = r2(hL, chinchilla_loss(hN, hD, *nc_p))
for name, S, p, fN, fD in [("CO", CO, co_p, coN, coD), ("NC", NC, nc_p, ncN, ncD)]:
    print(f"{name} TPPs={sorted(S)} cells={len(fN)} V_K={V_K_values(S):.4g} "
          f"kappa_AB={kappa_AB(fN, fD, p):.4g} R2_hold={r2(hL, chinchilla_loss(hN, hD, *p)):.4f}")

# ---- dual surface (log10 coords) + holdout scatter + residual sticks ----
allN, allD, allL = env["N"].values, env["Dt"].values, env["L"].values
N1 = np.logspace(np.log10(allN.min()), np.log10(allN.max()), 160)
D1 = np.logspace(np.log10(allD.min()), np.log10(allD.max()), 160)
Nm, Dm = np.meshgrid(N1, D1)
Lco = chinchilla_loss(Nm, Dm, *co_p)
Lnc = chinchilla_loss(Nm, Dm, *nc_p)
Llo, Lhi = np.nanpercentile(allL, 1) * 0.7, np.nanpercentile(allL, 99) * 1.3
Lco_c, Lnc_c = np.clip(Lco, Llo, Lhi), np.clip(Lnc, Llo, Lhi)
Xm, Ym = np.log10(Nm), np.log10(Dm)
zlo = min(Llo, Lco_c.min(), Lnc_c.min()) * 0.98
zhi = max(Lhi, Lco_c.max(), Lnc_c.max()) * 1.02
xlim, ylim, zlim = (Xm.min(), Xm.max()), (Ym.min(), Ym.max()), (zlo, zhi)

fig = plt.figure(figsize=(SURF_W, SURF_H))
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(Xm, Ym, Lco_c, facecolors=_build_adaptive_rgba(CO_COLOR, Lco_c, Lnc_c),
                edgecolor="none", rstride=2, cstride=2, shade=False, zorder=1)
ax.plot_surface(Xm, Ym, Lnc_c, facecolors=_build_adaptive_rgba(NC_COLOR, Lnc_c, Lco_c),
                edgecolor="none", rstride=2, cstride=2, shade=False, zorder=2)
_draw_boundary_edges_occlusion(ax, Xm, Ym, Lco_c, Lnc_c, CO_COLOR, SIDE_ELEV, SIDE_AZIM, xlim, ylim, zlim)
_draw_boundary_edges_occlusion(ax, Xm, Ym, Lnc_c, Lco_c, NC_COLOR, SIDE_ELEV, SIDE_AZIM, xlim, ylim, zlim)
ax.plot([], [], [], color=CO_COLOR, lw=6, alpha=0.85, label=f"CO {sorted(CO)} min V_K (R2h={r2h_co:.3f})")
ax.plot([], [], [], color=NC_COLOR, lw=6, alpha=0.85, label=f"NC {sorted(NC)} max V_K (R2h={r2h_nc:.3f})")
ax.scatter(np.log10(hN), np.log10(hD), hL, c="black", s=35, alpha=0.9,
           edgecolors="white", linewidths=0.5, label="Holdout", zorder=5)
pco, pnc = chinchilla_loss(hN, hD, *co_p), chinchilla_loss(hN, hD, *nc_p)
for i in range(len(hN)):
    x, y = np.log10(hN[i]), np.log10(hD[i])
    ax.plot([x, x], [y, y], [hL[i], pco[i]], color="green", lw=1.5, alpha=0.85, zorder=4,
            label="CO residual" if i == 0 else None)
    ax.plot([x, x], [y, y], [hL[i], pnc[i]], color="black", lw=1.5, alpha=0.85, zorder=4,
            label="NC residual" if i == 0 else None)
ax.set_xlabel("log10 N", fontsize=AXIS_LABEL_FS, labelpad=18)
ax.set_ylabel("log10 D", fontsize=AXIS_LABEL_FS, labelpad=18)
ax.set_zlabel("")
ax.text2D(1.08, 0.5, "Loss", transform=ax.transAxes, rotation=90, va="center", ha="left", fontsize=AXIS_LABEL_FS)
ax.invert_yaxis()
ax.set_zlim(zlo, zhi)
ax.view_init(elev=SIDE_ELEV, azim=SIDE_AZIM)
ax.legend(fontsize=int(LEGEND_FS * 0.6), loc="upper left", framealpha=LEGEND_ALPHA)
style_ticks_3d(ax)
plt.tight_layout()
out = REPO / "figures" / "surface_co_vs_nc.png"
out.parent.mkdir(exist_ok=True)
fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.4)
print(f"wrote {out}")
