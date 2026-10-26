"""
Visualization utilities for scaling-law analysis.

All plots are rendered as **individual figures** (one per plt.figure / plt.show)
so each image can be right-click-saved independently.

Plot families
─────────────
1. Data exploration   - 3-D scatter, overlay
2. Distributions      - N, D, Loss histograms per dataset × design × split
3. Scaling-law surface - 3-D "blanket" mesh of L(N,D) from each fitted law,
                         with actual data points scattered on top
4. IsoFLOP curves     - 2-D slices at fixed compute budgets
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from scaling_laws.parsing import parse_epoch_results

matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["font.family"] = "STIXGeneral"


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

_COLOR_MAP = {
    "nxd_grid":         "green",
    "holdout":          "orange",
    "colinear":         "blue",
    "colinear_holdout": "red",
}

_LABEL_MAP = {
    "nxd_grid":         "NC Grid",
    "holdout":          "NC Holdout",
    "colinear":         "CO Grid",
    "colinear_holdout": "CO Holdout",
}

_CANONICAL_TYPES = {"nxd_grid", "holdout", "colinear", "colinear_holdout"}


def _normalize_run_type(rt: str) -> str:
    for canonical in sorted(_CANONICAL_TYPES, key=len, reverse=True):
        if rt == canonical or rt.endswith("_" + canonical):
            return canonical
    return rt


def _runs_to_df(filepath: str) -> pd.DataFrame:
    runs = parse_epoch_results(filepath)
    records = []
    for r in runs:
        records.append({
            "N":          r["N"],
            "D":          r["D"],
            "n_epochs":   r["n_epochs"],
            "run_type":   _normalize_run_type(r["run_type"]),
            "Loss":       r["epoch_losses"][-1],
            "Loss_First": r["epoch_losses"][0],
        })
    return pd.DataFrame(records)


def _savefig_and_show(fig, save_path=None, dpi=150):
    """Optionally save, then show (and close) a figure."""
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.show()
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# 1. Data exploration - 3-D scatter (one figure each)
# ═══════════════════════════════════════════════════════════════════════════

def plot_3d_overview(filepath: str, title: str = "", save_dir: str | None = None):
    """
    Two **separate** 3-D scatter figures:
      • Non-Colinear (train + holdout)
      • Colinear      (train + holdout)
    """
    df = _runs_to_df(filepath)

    groups = [
        ("Non-Colinear", [("nxd_grid", "blue"), ("holdout", "red")]),
        ("Colinear",     [("colinear", "blue"), ("colinear_holdout", "red")]),
    ]

    for group_name, run_types in groups:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        for rt, color in run_types:
            d = df[df["run_type"] == rt]
            if not d.empty:
                ax.scatter(d["N"], d["D"], d["Loss"], c=color, s=40,
                           alpha=0.8, edgecolors="w",
                           label=_LABEL_MAP.get(rt, rt))
        ax.invert_yaxis()
        ax.set_xlabel("N (params)")
        ax.set_ylabel("D (tokens)")
        ax.set_zlabel("Loss")
        ax.set_title(f"{title or filepath} - {group_name}", fontweight="bold")
        ax.legend(fontsize=9)
        plt.tight_layout()
        sp = f"{save_dir}/{group_name.lower().replace('-','_')}_3d.png" if save_dir else None
        _savefig_and_show(fig, sp)


def plot_overlay(filepath: str, title: str = "", save_path: str | None = None):
    """Single 3-D scatter with all four run types colour-coded."""
    df = _runs_to_df(filepath)
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    for rt, color in _COLOR_MAP.items():
        d = df[df["run_type"] == rt]
        if not d.empty:
            ax.scatter(d["N"], d["D"], d["Loss"], c=color, s=40,
                       alpha=0.7, edgecolors="w",
                       label=_LABEL_MAP.get(rt, rt))
    ax.invert_yaxis()
    ax.set_xlabel("N (params)")
    ax.set_ylabel("D (tokens)")
    ax.set_zlabel("Loss")
    ax.set_title(title or "Colinear vs Non-Colinear Overlay", fontweight="bold")
    ax.legend(loc="upper left")
    plt.tight_layout()
    _savefig_and_show(fig, save_path)


# ═══════════════════════════════════════════════════════════════════════════
# 2. Distribution plots
# ═══════════════════════════════════════════════════════════════════════════

def plot_distributions(
    prepared_data,
    save_dir: str | None = None,
):
    """
    For each dataset × {Colinear, Non-Colinear} × {Train, Holdout},
    produce **one figure** with four distribution sub-panels:
      • N (params)  histogram
      • D (tokens)  histogram
      • Loss        histogram
      • N vs D      scatter coloured by loss

    Parameters
    ----------
    prepared_data : PreparedData or dict {name: PreparedData}
    save_dir      : if set, save PNGs here
    """
    from scaling_laws.data import PreparedData

    if isinstance(prepared_data, PreparedData):
        pd_map = {prepared_data.dataset_name: prepared_data}
    else:
        pd_map = prepared_data

    for ds_name, pdata in pd_map.items():
        slices = {
            "Colinear - Train":         pdata.colinear_train,
            "Colinear - Holdout":       pdata.colinear_holdout,
            "Non-Colinear - Train":     pdata.noncolinear_train,
            "Non-Colinear - Holdout":   pdata.noncolinear_holdout,
        }

        for slice_label, run_list in slices.items():
            if not run_list:
                continue

            Ns    = np.array([d["N"] for d in run_list])
            Ds    = np.array([d["D"] for d in run_list])
            Ls    = np.array([d["final_loss"] for d in run_list])

            fig, axes = plt.subplots(2, 2, figsize=(13, 10))
            fig.suptitle(f"{ds_name} - {slice_label}  ({len(run_list)} runs)",
                         fontsize=14, fontweight="bold")

            # (0,0) N histogram
            ax = axes[0, 0]
            unique_N = np.unique(Ns)
            if len(unique_N) <= 20:
                ax.bar(range(len(unique_N)),
                       [np.sum(Ns == n) for n in unique_N],
                       tick_label=[f"{int(n):,}" for n in unique_N],
                       color="steelblue", edgecolor="white")
                ax.set_xlabel("N (params)")
                ax.tick_params(axis="x", rotation=45, labelsize=8)
            else:
                ax.hist(np.log10(Ns), bins=30, color="steelblue", edgecolor="white")
                ax.set_xlabel("log₁₀(N)")
            ax.set_ylabel("Count")
            ax.set_title("Model Size Distribution")

            # (0,1) D histogram
            ax = axes[0, 1]
            unique_D = np.unique(Ds)
            if len(unique_D) <= 20:
                ax.bar(range(len(unique_D)),
                       [np.sum(Ds == d) for d in unique_D],
                       tick_label=[f"{int(d):,}" for d in unique_D],
                       color="darkorange", edgecolor="white")
                ax.set_xlabel("D (unique tokens)")
                ax.tick_params(axis="x", rotation=45, labelsize=8)
            else:
                ax.hist(np.log10(Ds), bins=30, color="darkorange", edgecolor="white")
                ax.set_xlabel("log₁₀(D)")
            ax.set_ylabel("Count")
            ax.set_title("Dataset Size Distribution")

            # (1,0) Loss histogram
            ax = axes[1, 0]
            ax.hist(Ls, bins=30, color="seagreen", edgecolor="white")
            ax.set_xlabel("Final Loss")
            ax.set_ylabel("Count")
            ax.set_title("Loss Distribution")
            ax.axvline(np.median(Ls), color="black", ls="--", lw=1.2,
                       label=f"median={np.median(Ls):.3f}")
            ax.legend(fontsize=8)

            # (1,1) N vs D scatter, coloured by loss
            ax = axes[1, 1]
            sc = ax.scatter(Ns, Ds, c=Ls, cmap="viridis_r", s=50,
                            edgecolors="white", linewidths=0.5, alpha=0.85)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("N (params)")
            ax.set_ylabel("D (unique tokens)")
            ax.set_title("N vs D (colour = Loss)")
            fig.colorbar(sc, ax=ax, label="Loss", shrink=0.8)

            plt.tight_layout()
            tag = slice_label.lower().replace(" ", "_").replace("\u2014", "").replace("__", "_")
            sp = f"{save_dir}/{ds_name}_{tag}_dist.png" if save_dir else None
            _savefig_and_show(fig, sp)


# ═══════════════════════════════════════════════════════════════════════════
# 3. Scaling-law surface ("blanket") plots
# ═══════════════════════════════════════════════════════════════════════════

def _build_surface_grid(N_pts, D_pts, loss_fn, params, log_space=True):
    """
    Evaluate a standard (N, D, *params) loss function on a 2-D mesh.

    Returns (N_mesh, D_mesh, L_mesh) - all 2-D arrays for plot_surface.
    """
    if log_space:
        N_1d = np.logspace(np.log10(N_pts.min()), np.log10(N_pts.max()), 60)
        D_1d = np.logspace(np.log10(D_pts.min()), np.log10(D_pts.max()), 60)
    else:
        N_1d = np.linspace(N_pts.min(), N_pts.max(), 60)
        D_1d = np.linspace(D_pts.min(), D_pts.max(), 60)

    N_mesh, D_mesh = np.meshgrid(N_1d, D_1d)
    L_mesh = loss_fn(N_mesh, D_mesh, *params)
    return N_mesh, D_mesh, L_mesh


_HOLDOUT_STYLES = [
    {"color": "red",       "marker": "^", "label_prefix": ""},
    {"color": "darkorange","marker": "s", "label_prefix": ""},
    {"color": "purple",    "marker": "D", "label_prefix": ""},
]


def plot_scaling_law_surface(
    result_dict: Dict,
    train_data: List[Dict],
    holdout_data: List[Dict] | Dict[str, List[Dict]] | None = None,
    mode: str = "final",
    elev: float = 25,
    azim: float = -60,
    save_path: str | None = None,
):
    """
    Plot a single scaling-law's predicted surface as a 3-D mesh ("blanket")
    with actual train + holdout points scattered on top.

    Parameters
    ----------
    result_dict  : one result dict from the pipeline (must contain 'law',
                   'params', 'design', 'mode')
    train_data   : list of run dicts used for fitting
    holdout_data : holdout scatter overlay - accepts either:
                     • a plain list of run dicts  (labelled "Holdout")
                     • a dict {label: list_of_run_dicts} for multiple sets,
                       e.g. {"Colinear Holdout": [...], "Unified Holdout": [...]}
    mode         : 'final', 'first', or 'all' - how D is computed
    elev, azim   : camera angles
    """
    from scaling_laws.data import prepare_arrays
    from scaling_laws.laws import LAWS

    law_name = result_dict["law"]
    params   = result_dict["params"]
    design   = result_dict["design"]

    # Find the matching law object
    law_obj = None
    for l in LAWS:
        if l.name == law_name:
            law_obj = l
            break

    if law_obj is None:
        print(f"Unknown law: {law_name}")
        return
    if law_obj.is_epoch_law:
        plot_epoch_law_surface(result_dict, train_data, holdout_data,
                               elev=elev, azim=azim, save_path=save_path)
        return

    # Normalise holdout_data to a dict
    if holdout_data is None:
        holdouts: Dict[str, List[Dict]] = {}
    elif isinstance(holdout_data, list):
        holdouts = {"Holdout": holdout_data}
    else:
        holdouts = holdout_data

    # Prepare actual data points
    N_tr, D_tr, L_tr = prepare_arrays(train_data, mode)
    if len(L_tr) == 0:
        print(f"No training data for {law_name} / {design}")
        return

    # Collect all N/D/L values (train + every holdout) for grid extent & clipping
    all_N, all_D, all_L = N_tr.copy(), D_tr.copy(), L_tr.copy()
    prepped_holdouts: Dict[str, tuple] = {}
    for h_label, h_list in holdouts.items():
        if not h_list:
            continue
        N_h, D_h, L_h = prepare_arrays(h_list, mode)
        if len(L_h) > 0:
            prepped_holdouts[h_label] = (N_h, D_h, L_h)
            all_N = np.concatenate([all_N, N_h])
            all_D = np.concatenate([all_D, D_h])
            all_L = np.concatenate([all_L, L_h])

    # Build the surface mesh spanning train + all holdouts
    N_mesh, D_mesh, L_mesh = _build_surface_grid(
        all_N, all_D, law_obj.loss_fn, params,
    )

    # Clip extreme predictions for visual sanity
    L_clip_hi = np.nanpercentile(all_L, 99) * 1.5
    L_clip_lo = np.nanpercentile(all_L, 1) * 0.5
    L_mesh_clipped = np.clip(L_mesh, L_clip_lo, L_clip_hi)

    # ── Plot ──
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    # Surface blanket
    ax.plot_surface(
        N_mesh, D_mesh, L_mesh_clipped,
        cmap="coolwarm", alpha=0.45, edgecolor="none",
        rstride=2, cstride=2,
    )

    # Train scatter
    ax.scatter(
        N_tr, D_tr, L_tr,
        c="black", s=30, alpha=0.85, edgecolors="white", linewidths=0.5,
        label="Train",
        zorder=5,
    )

    # Holdout scatter(s)
    for idx, (h_label, (N_h, D_h, L_h)) in enumerate(prepped_holdouts.items()):
        sty = _HOLDOUT_STYLES[idx % len(_HOLDOUT_STYLES)]
        ax.scatter(
            N_h, D_h, L_h,
            c=sty["color"], s=30, alpha=0.85, edgecolors="white", linewidths=0.5,
            marker=sty["marker"], label=h_label,
            zorder=5,
        )

    ax.set_xlabel("N (params)")
    ax.set_ylabel("D (tokens)")
    ax.set_zlabel("Loss")
    ax.invert_yaxis()
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(
        f"{law_name}  |  {design}  |  {result_dict.get('mode', mode)}\n"
        f"R\u00b2 train = {result_dict.get('r2_train', float('nan')):.4f}",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=9, loc="upper left")
    plt.tight_layout()
    _savefig_and_show(fig, save_path)


def plot_epoch_law_surface(
    result_dict: Dict,
    train_data: List[Dict],
    holdout_data: List[Dict] | Dict[str, List[Dict]] | None = None,
    fixed_epoch: int = 1,
    elev: float = 25,
    azim: float = -60,
    save_path: str | None = None,
):
    """
    3-D surface for the Parametric-Epoch law at a **fixed epoch** number.

    holdout_data accepts a plain list or a dict {label: list} for
    multiple holdout sets (same as plot_scaling_law_surface).
    """
    from scaling_laws.data import prepare_epoch_expanded
    from scaling_laws.laws import parametric_epoch_loss

    params = result_dict["params"]
    design = result_dict["design"]
    R_fixed = fixed_epoch - 1

    # Normalise holdout_data to a dict
    if holdout_data is None:
        holdouts: Dict[str, List[Dict]] = {}
    elif isinstance(holdout_data, list):
        holdouts = {"Holdout": holdout_data}
    else:
        holdouts = holdout_data

    N_tr, U_D_tr, R_tr, ep_tr, L_tr = prepare_epoch_expanded(train_data)
    if len(L_tr) == 0:
        return

    mask_tr = (ep_tr == fixed_epoch)

    # Collect all N/U_D/L for grid extent & clipping
    all_N, all_D, all_L = N_tr.copy(), U_D_tr.copy(), L_tr.copy()
    prepped_holdouts: Dict[str, tuple] = {}
    for h_label, h_list in holdouts.items():
        if not h_list:
            continue
        N_h, U_D_h, R_h, ep_h, L_h = prepare_epoch_expanded(h_list)
        mask_h = (ep_h == fixed_epoch)
        if mask_h.any():
            prepped_holdouts[h_label] = (N_h[mask_h], U_D_h[mask_h], L_h[mask_h])
            all_N = np.concatenate([all_N, N_h])
            all_D = np.concatenate([all_D, U_D_h])
            all_L = np.concatenate([all_L, L_h])

    N_1d = np.logspace(np.log10(all_N.min()), np.log10(all_N.max()), 60)
    D_1d = np.logspace(np.log10(all_D.min()), np.log10(all_D.max()), 60)
    N_mesh, D_mesh = np.meshgrid(N_1d, D_1d)
    R_mesh = np.full_like(N_mesh, R_fixed, dtype=np.float64)

    L_mesh = parametric_epoch_loss(
        N_mesh, D_mesh, R_mesh,
        params["A"], params["alpha"], params["B"], params["beta"],
        params["E"], params["R_star_D"], params["R_star_N"],
    )

    L_clip_hi = np.nanpercentile(all_L, 99) * 1.5
    L_clip_lo = np.nanpercentile(all_L, 1) * 0.5
    L_mesh_clipped = np.clip(L_mesh, L_clip_lo, L_clip_hi)

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(
        N_mesh, D_mesh, L_mesh_clipped,
        cmap="coolwarm", alpha=0.45, edgecolor="none",
        rstride=2, cstride=2,
    )

    if mask_tr.any():
        ax.scatter(
            N_tr[mask_tr], U_D_tr[mask_tr], L_tr[mask_tr],
            c="black", s=30, alpha=0.85, edgecolors="white", linewidths=0.5,
            label=f"Train (epoch {fixed_epoch})", zorder=5,
        )

    for idx, (h_label, (N_h, D_h, L_h)) in enumerate(prepped_holdouts.items()):
        sty = _HOLDOUT_STYLES[idx % len(_HOLDOUT_STYLES)]
        ax.scatter(
            N_h, D_h, L_h,
            c=sty["color"], s=30, alpha=0.85, edgecolors="white", linewidths=0.5,
            marker=sty["marker"], label=f"{h_label} (ep {fixed_epoch})", zorder=5,
        )

    ax.set_xlabel("N (params)")
    ax.set_ylabel("U_D (unique tokens)")
    ax.set_zlabel("Loss")
    ax.invert_yaxis()
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(
        f"Parametric-Epoch  |  {design}  |  Epoch {fixed_epoch}\n"
        f"R\u00b2 train = {result_dict.get('r2_train', float('nan')):.4f}",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=9, loc="upper left")
    plt.tight_layout()
    _savefig_and_show(fig, save_path)


def plot_all_surfaces(
    all_results: List[Dict],
    prepared_data,
    mode: str | None = None,
    dataset_filter: str | None = None,
    elev: float = 25,
    azim: float = -60,
    save_dir: str | None = None,
    epoch_law_epochs: List[int] | None = None,
):
    """
    Iterate over every fitted result and produce one surface plot each.

    Parameters
    ----------
    all_results     : list of result dicts from the pipeline
    prepared_data   : PreparedData or dict {name: PreparedData}
    mode            : filter to a specific epoch mode (None = plot all)
    dataset_filter  : filter to a specific dataset name
    elev, azim      : camera angles for 3-D view
    save_dir        : directory to save PNGs (None = just show)
    epoch_law_epochs: which epoch numbers to render for the Parametric-Epoch
                      law surface (default [1])
    """
    from scaling_laws.data import PreparedData

    if isinstance(prepared_data, PreparedData):
        pd_map = {prepared_data.dataset_name: prepared_data}
    else:
        pd_map = prepared_data

    if epoch_law_epochs is None:
        epoch_law_epochs = [1]

    for r in all_results:
        ds = r.get("dataset", "")
        if dataset_filter and ds != dataset_filter:
            continue
        if mode and r["mode"] != mode:
            continue
        if ds not in pd_map:
            continue

        pdata = pd_map[ds]
        design = r["design"]

        # Pick train slice + build holdout dict with design-specific AND unified
        holdouts: Dict[str, List[Dict]] = {}
        if design == "Colinear":
            train_data = pdata.colinear_train
            if pdata.colinear_holdout:
                holdouts["CO Holdout"] = pdata.colinear_holdout
        elif design == "Non-Colinear":
            train_data = pdata.noncolinear_train
            if pdata.noncolinear_holdout:
                holdouts["NC Holdout"] = pdata.noncolinear_holdout
        else:
            train_data = pdata.colinear_train + pdata.noncolinear_train

        # Always include unified holdout
        if pdata.unified_holdout:
            holdouts["Unified Holdout"] = pdata.unified_holdout

        law_name = r["law"]

        if law_name == "Parametric-Epoch":
            for ep in epoch_law_epochs:
                tag = f"{ds}_{design}_{law_name}_ep{ep}".replace(" ", "_").replace("-", "_")
                sp = f"{save_dir}/{tag}.png" if save_dir else None
                plot_epoch_law_surface(
                    r, train_data, holdouts,
                    fixed_epoch=ep, elev=elev, azim=azim, save_path=sp,
                )
        else:
            tag = f"{ds}_{design}_{law_name}_{r['mode']}".replace(" ", "_").replace("-", "_")
            sp = f"{save_dir}/{tag}.png" if save_dir else None
            plot_scaling_law_surface(
                r, train_data, holdouts,
                mode=r["mode"].lower(),
                elev=elev, azim=azim, save_path=sp,
            )


# ═══════════════════════════════════════════════════════════════════════════
# 4. IsoFLOP curves - individual figures
# ═══════════════════════════════════════════════════════════════════════════

def _chinchilla_isoflop(N_range, C, E, A, alpha, B, beta):
    D = C / (6.0 * N_range)
    return E + A / np.power(N_range, alpha) + B / np.power(D, beta)


def _chinchilla_optimal_N(C, A, alpha, B, beta):
    return np.power(
        alpha * A / (beta * B * np.power(6.0, beta)),
        1.0 / (alpha + beta),
    ) * np.power(C, beta / (alpha + beta))


def plot_isoflop_single(
    all_results,
    prepared_data,
    holdout_type="unified",
    dataset_filter=None,
    mode_filter=None,
    flop_budgets=None,
    N_range=None,
    y_lim=None,
    show_optimal=True,
    save_dir=None,
):
    """
    One **separate figure** per (dataset, mode) for a given holdout type.
    """
    from scaling_laws.data import PreparedData

    if N_range is None:
        N_range = np.logspace(4, 11, 500)
    if flop_budgets is None:
        flop_budgets = np.logspace(15, 21, 5)

    if isinstance(prepared_data, PreparedData):
        pd_map = {prepared_data.dataset_name: prepared_data}
    else:
        pd_map = prepared_data

    holdout_labels = {
        "unified":      "Unified (Holdout)",
        "colinear":     "Colinear (Holdout)",
        "noncolinear":  "Non-Colinear (Holdout)",
    }

    chinch = [r for r in all_results if r["law"] == "Chinchilla"]
    if dataset_filter:
        chinch = [r for r in chinch if r["dataset"] == dataset_filter]
    if mode_filter:
        chinch = [r for r in chinch if r["mode"] == mode_filter]
    if not chinch:
        print(f"No Chinchilla fits found for dataset={dataset_filter}, mode={mode_filter}.")
        return

    groups = {}
    for r in chinch:
        groups.setdefault((r["dataset"], r["mode"]), []).append(r)

    for (ds, mode_label), fits in sorted(groups.items()):
        fig, ax = plt.subplots(figsize=(10, 7))

        h_data = None
        if ds in pd_map:
            pdata = pd_map[ds]
            if holdout_type == "unified":
                h_data = pdata.unified_holdout
            elif holdout_type == "colinear":
                h_data = pdata.colinear_holdout
            elif holdout_type == "noncolinear":
                h_data = pdata.noncolinear_holdout

        _draw_isoflop(
            ax, fits, h_data, holdout_labels.get(holdout_type, holdout_type),
            flop_budgets, N_range, y_lim, show_optimal=show_optimal,
        )
        ax.set_title(
            f"{ds} \u2014 {mode_label}  |  {holdout_labels.get(holdout_type, '')}",
            fontsize=13, fontweight="bold",
        )
        plt.tight_layout()
        tag = f"{ds}_{mode_label}_{holdout_type}_isoflop".replace(" ", "_")
        sp = f"{save_dir}/{tag}.png" if save_dir else None
        _savefig_and_show(fig, sp)


def plot_isoflops(
    all_results,
    prepared_data=None,
    flop_budgets=None,
    N_range=None,
    y_lim=None,
    show_optimal=True,
    save_dir=None,
):
    """
    One figure per (dataset, mode, holdout_type) - fully separated.
    """
    from scaling_laws.data import PreparedData

    if N_range is None:
        N_range = np.logspace(4, 11, 500)
    if flop_budgets is None:
        flop_budgets = np.logspace(15, 21, 5)

    if isinstance(prepared_data, PreparedData):
        pd_map = {prepared_data.dataset_name: prepared_data}
    elif prepared_data is None:
        pd_map = {}
    else:
        pd_map = prepared_data

    holdout_types = [
        ("unified",      "Unified (Holdout)"),
        ("colinear",     "Colinear (Holdout)"),
        ("noncolinear",  "Non-Colinear (Holdout)"),
    ]

    chinch = [r for r in all_results if r["law"] == "Chinchilla"]
    if not chinch:
        print("No Chinchilla fits found.")
        return

    groups = {}
    for r in chinch:
        groups.setdefault((r["dataset"], r["mode"]), []).append(r)

    for (ds, mode_label), fits in sorted(groups.items()):
        for h_type, h_label in holdout_types:
            fig, ax = plt.subplots(figsize=(10, 7))

            h_data = None
            if ds in pd_map:
                pdata = pd_map[ds]
                if h_type == "unified":
                    h_data = pdata.unified_holdout
                elif h_type == "colinear":
                    h_data = pdata.colinear_holdout
                elif h_type == "noncolinear":
                    h_data = pdata.noncolinear_holdout

            _draw_isoflop(
                ax, fits, h_data, h_label,
                flop_budgets, N_range, y_lim, show_optimal=show_optimal,
            )
            ax.set_title(
                f"{ds} \u2014 {mode_label}  |  {h_label}",
                fontsize=13, fontweight="bold",
            )
            plt.tight_layout()
            tag = f"{ds}_{mode_label}_{h_type}_isoflop".replace(" ", "_")
            sp = f"{save_dir}/{tag}.png" if save_dir else None
            _savefig_and_show(fig, sp)


# ═══════════════════════════════════════════════════════════════════════════
# 5. Dual-blanket comparison (CO vs NC surfaces + holdout + gap bars)
# ═══════════════════════════════════════════════════════════════════════════

_HOLDOUT_SCATTER_STYLES = {
    # label substring → style
    "unified":      {"color": "black",      "marker": "o", "s": 40},
    "non-colinear": {"color": "darkorange", "marker": "^", "s": 40},
    "nc":           {"color": "darkorange", "marker": "^", "s": 40},
    "colinear":     {"color": "purple",     "marker": "s", "s": 35},
    "co":           {"color": "purple",     "marker": "s", "s": 35},
}


def _holdout_style(label: str) -> dict:
    """Pick scatter style based on holdout label (case-insensitive substring match)."""
    low = label.lower()
    # Check longest keys first to avoid "colinear" matching "non-colinear"
    for key in sorted(_HOLDOUT_SCATTER_STYLES, key=len, reverse=True):
        if key in low:
            return _HOLDOUT_SCATTER_STYLES[key]
    return {"color": "gray", "marker": "x", "s": 30}


def plot_dual_blanket(
    co_result: Dict,
    nc_result: Dict,
    holdouts: Dict[str, List[Dict]],
    mode: str = "final",
    dataset_name: str = "",
    # ── 3-D perspective view ──
    elev: float = 25,
    azim: float = -60,
    # ── Side view ──
    side_elev: float = 0,
    side_azim: float = 0,
    # ── Gap bars ──
    show_gap_bars: bool = True,
    gap_bar_holdout: str | None = None,
    gap_bar_every: int = 1,
    # ── Appearance ──
    co_cmap: str = "Blues",
    nc_cmap: str = "Reds",
    co_alpha: float = 0.35,
    nc_alpha: float = 0.35,
    title: str | None = None,
    figsize: tuple = (13, 9),
    # ── Output ──
    save_dir: str | None = None,
):
    """
    Overlay CO and NC fitted surfaces on the same 3-D plot with holdout
    scatter (no training data), then produce a **side view** showing the
    gap between surfaces.

    Produces two separate figures (right-click saveable):
      1. Perspective view - both blankets + holdout scatter
      2. Side view - both blankets + vertical gap bars at holdout points

    Parameters
    ----------
    co_result, nc_result : result dicts from the pipeline for the *same* law
                           and mode, but different designs
    holdouts      : dict {label: run_list}, e.g.
                    {"Unified": unified_holdout,
                     "NC Holdout": nc_holdout,
                     "CO Holdout": co_holdout}
    mode          : 'final', 'first', 'all', or 'second'
    dataset_name  : name of the dataset (shown in title and filenames)
    elev, azim    : camera for the perspective figure
    side_elev, side_azim : camera for the side figure (0, 0 = looking along D)
    show_gap_bars : draw vertical bars between surfaces at holdout points
    gap_bar_holdout : which holdout key to draw bars for (None = first key)
    gap_bar_every : draw a bar every N-th holdout point (1 = all)
    co_cmap, nc_cmap : colormaps for the two surfaces
    co_alpha, nc_alpha : surface transparency
    title         : override the auto-generated title
    figsize       : figure size
    save_dir      : directory to save PNGs
    """
    from scaling_laws.data import prepare_arrays
    from scaling_laws.laws import LAWS

    law_name = co_result["law"]
    assert co_result["law"] == nc_result["law"], \
        f"Laws must match: {co_result['law']} vs {nc_result['law']}"

    # Find law object
    law_obj = None
    for l in LAWS:
        if l.name == law_name:
            law_obj = l
            break
    if law_obj is None or law_obj.is_epoch_law:
        print(f"plot_dual_blanket: unsupported law '{law_name}' (use standard N,D laws)")
        return

    co_params = co_result["params"]
    nc_params = nc_result["params"]

    # ── Collect all points for mesh extent ──
    all_N, all_D = np.array([]), np.array([])
    prepped: Dict[str, tuple] = {}
    for h_label, h_list in holdouts.items():
        if not h_list:
            continue
        N_h, D_h, L_h = prepare_arrays(h_list, mode)
        if len(L_h) > 0:
            prepped[h_label] = (N_h, D_h, L_h)
            all_N = np.concatenate([all_N, N_h])
            all_D = np.concatenate([all_D, D_h])

    if len(all_N) == 0:
        print("No holdout data to plot.")
        return

    # ── Build mesh spanning all holdout points ──
    N_1d = np.logspace(np.log10(all_N.min()), np.log10(all_N.max()), 60)
    D_1d = np.logspace(np.log10(all_D.min()), np.log10(all_D.max()), 60)
    N_mesh, D_mesh = np.meshgrid(N_1d, D_1d)

    L_co = law_obj.loss_fn(N_mesh, D_mesh, *co_params)
    L_nc = law_obj.loss_fn(N_mesh, D_mesh, *nc_params)

    # Clipping based on actual holdout loss range
    all_L = np.concatenate([v[2] for v in prepped.values()])
    L_lo = np.nanpercentile(all_L, 1) * 0.5
    L_hi = np.nanpercentile(all_L, 99) * 1.5
    L_co_clip = np.clip(L_co, L_lo, L_hi)
    L_nc_clip = np.clip(L_nc, L_lo, L_hi)

    ds_tag = dataset_name or co_result.get("dataset", "")
    mode_tag = co_result.get("mode", mode)
    auto_title = title or (
        f"{ds_tag}  |  {law_name}  |  {mode_tag}  -  "
        f"CO vs NC Surfaces"
    )
    file_prefix = f"{ds_tag}_{law_name}_{mode_tag}".replace(" ", "_").replace("-", "_")

    # ── Which holdout to draw gap bars on ──
    if gap_bar_holdout is None:
        gap_bar_holdout = next(iter(prepped))
    bar_data = prepped.get(gap_bar_holdout)

    # ==================================================================
    # FIGURE 1 - Perspective view
    # ==================================================================
    fig1 = plt.figure(figsize=figsize)
    ax1 = fig1.add_subplot(111, projection="3d")

    ax1.plot_surface(N_mesh, D_mesh, L_co_clip,
                     cmap=co_cmap, alpha=co_alpha, edgecolor="none",
                     rstride=2, cstride=2)
    ax1.plot_surface(N_mesh, D_mesh, L_nc_clip,
                     cmap=nc_cmap, alpha=nc_alpha, edgecolor="none",
                     rstride=2, cstride=2)

    # Invisible proxies for legend (plot_surface doesn't support label)
    ax1.plot([], [], [], color="steelblue", lw=6, alpha=0.6, label="Colinear fit")
    ax1.plot([], [], [], color="indianred", lw=6, alpha=0.6, label="Non-Colinear fit")

    for h_label, (N_h, D_h, L_h) in prepped.items():
        sty = _holdout_style(h_label)
        ax1.scatter(N_h, D_h, L_h,
                    c=sty["color"], s=sty["s"], alpha=0.85,
                    edgecolors="white", linewidths=0.5,
                    marker=sty["marker"], label=h_label, zorder=5)

    ax1.set_xlabel("N (params)")
    ax1.set_ylabel("D (tokens)")
    ax1.set_zlabel("Loss")
    ax1.invert_yaxis()
    ax1.view_init(elev=elev, azim=azim)
    ax1.set_title(auto_title, fontsize=12, fontweight="bold")
    ax1.legend(fontsize=9, loc="upper left")
    plt.tight_layout()
    sp1 = f"{save_dir}/{file_prefix}_dual_perspective.png" if save_dir else None
    _savefig_and_show(fig1, sp1)

    # ==================================================================
    # FIGURE 2 - Side view with gap bars
    # ==================================================================
    fig2 = plt.figure(figsize=figsize)
    ax2 = fig2.add_subplot(111, projection="3d")

    ax2.plot_surface(N_mesh, D_mesh, L_co_clip,
                     cmap=co_cmap, alpha=co_alpha, edgecolor="none",
                     rstride=2, cstride=2)
    ax2.plot_surface(N_mesh, D_mesh, L_nc_clip,
                     cmap=nc_cmap, alpha=nc_alpha, edgecolor="none",
                     rstride=2, cstride=2)

    ax2.plot([], [], [], color="steelblue", lw=6, alpha=0.6, label="Colinear fit")
    ax2.plot([], [], [], color="indianred", lw=6, alpha=0.6, label="Non-Colinear fit")

    for h_label, (N_h, D_h, L_h) in prepped.items():
        sty = _holdout_style(h_label)
        ax2.scatter(N_h, D_h, L_h,
                    c=sty["color"], s=sty["s"], alpha=0.85,
                    edgecolors="white", linewidths=0.5,
                    marker=sty["marker"], label=h_label, zorder=5)

    # ── Gap bars ──
    if show_gap_bars and bar_data is not None:
        N_b, D_b, L_b = bar_data
        L_co_at = law_obj.loss_fn(N_b, D_b, *co_params)
        L_nc_at = law_obj.loss_fn(N_b, D_b, *nc_params)

        for i in range(0, len(N_b), gap_bar_every):
            lo = min(L_co_at[i], L_nc_at[i])
            hi = max(L_co_at[i], L_nc_at[i])
            ax2.plot(
                [N_b[i], N_b[i]], [D_b[i], D_b[i]], [lo, hi],
                color="black", lw=1.2, alpha=0.6, zorder=4,
            )
            # Small tick marks at the ends
            tick_len_n = (all_N.max() - all_N.min()) * 0.008
            ax2.plot(
                [N_b[i] - tick_len_n, N_b[i] + tick_len_n],
                [D_b[i], D_b[i]], [lo, lo],
                color="black", lw=1.0, alpha=0.6, zorder=4,
            )
            ax2.plot(
                [N_b[i] - tick_len_n, N_b[i] + tick_len_n],
                [D_b[i], D_b[i]], [hi, hi],
                color="black", lw=1.0, alpha=0.6, zorder=4,
            )

        # Legend proxy for gap bars
        ax2.plot([], [], [], color="black", lw=1.5, label=f"Gap ({gap_bar_holdout})")

    ax2.set_xlabel("N (params)")
    ax2.set_ylabel("D (tokens)")
    ax2.set_zlabel("Loss")
    ax2.invert_yaxis()
    ax2.view_init(elev=side_elev, azim=side_azim)
    ax2.set_title(f"{auto_title}  -  Side View", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=9, loc="upper left")
    plt.tight_layout()
    sp2 = f"{save_dir}/{file_prefix}_dual_side.png" if save_dir else None
    _savefig_and_show(fig2, sp2)


def plot_dual_blanket_from_results(
    all_results: List[Dict],
    prepared_data,
    law_name: str = "Chinchilla",
    mode_filter: str = "Final",
    dataset_filter: str | None = None,
    holdout_keys: List[str] | None = None,
    **kwargs,
):
    """
    Convenience wrapper: find the CO and NC results for a given law + mode
    and call plot_dual_blanket with the right holdout data.

    Parameters
    ----------
    all_results    : full pipeline results list
    prepared_data  : PreparedData or dict {name: PreparedData}
    law_name       : which law to compare ("Kaplan", "Chinchilla", "Droppo-Elibol")
    mode_filter    : which epoch mode ("Final", "First", "All", "Second")
    dataset_filter : which dataset (None = first available)
    holdout_keys   : which holdouts to include, from
                     ["Unified", "CO Holdout", "NC Holdout"]
                     (default: all three)
    **kwargs       : passed to plot_dual_blanket (elev, azim, side_elev, etc.)
    """
    from scaling_laws.data import PreparedData

    if isinstance(prepared_data, PreparedData):
        pd_map = {prepared_data.dataset_name: prepared_data}
    else:
        pd_map = prepared_data

    # Find matching CO and NC results
    matches = [
        r for r in all_results
        if r["law"] == law_name and r["mode"] == mode_filter
    ]
    if dataset_filter:
        matches = [r for r in matches if r["dataset"] == dataset_filter]

    co = [r for r in matches if r["design"] == "Colinear"]
    nc = [r for r in matches if r["design"] == "Non-Colinear"]

    if not co or not nc:
        print(f"Could not find CO and NC results for {law_name} / {mode_filter}"
              f" / {dataset_filter}")
        return

    co_result = co[0]
    nc_result = nc[0]
    ds = co_result["dataset"]

    if ds not in pd_map:
        print(f"No PreparedData for dataset '{ds}'")
        return

    pdata = pd_map[ds]

    if holdout_keys is None:
        holdout_keys = ["Unified", "CO Holdout", "NC Holdout"]

    holdouts = {}
    for key in holdout_keys:
        low = key.lower()
        if "unified" in low and pdata.unified_holdout:
            holdouts[key] = pdata.unified_holdout
        elif ("co" in low or "colinear" in low) and "non" not in low and pdata.colinear_holdout:
            holdouts[key] = pdata.colinear_holdout
        elif ("nc" in low or "non" in low) and pdata.noncolinear_holdout:
            holdouts[key] = pdata.noncolinear_holdout

    plot_dual_blanket(
        co_result, nc_result, holdouts,
        mode=mode_filter.lower(),
        dataset_name=ds,
        **kwargs,
    )


# ═══════════════════════════════════════════════════════════════════════════
# 6. IsoFLOP curves - individual figures
# ═══════════════════════════════════════════════════════════════════════════

def _draw_isoflop(ax, fits, holdout_data, holdout_label,
                   flop_budgets, N_range, y_lim, show_optimal=True):
    """Core isoFLOP drawing on a single Axes - shared by public functions."""
    design_cfg = {
        "Non-Colinear": {"ls": "-",  "opt_color": "red"},
        "Colinear":     {"ls": "--", "opt_color": "blue"},
    }
    flop_cmap = plt.cm.viridis

    if holdout_data:
        raw_N = np.array([r["N"] for r in holdout_data])
        if len(raw_N):
            n_lo = float(min(N_range.min(), raw_N.min()))
            n_hi = float(max(N_range.max(), raw_N.max()))
            N_range = np.logspace(np.log10(n_lo), np.log10(n_hi), 500)

    for r in fits:
        design = r["design"]
        E, A, alpha, B, beta = r["params"]
        cfg = design_cfg.get(design, {"ls": "-.", "opt_color": "gray"})

        for ci, C in enumerate(flop_budgets):
            color = flop_cmap(ci / max(1, len(flop_budgets) - 1))
            L = _chinchilla_isoflop(N_range, C, E, A, alpha, B, beta)
            valid = (C / (6.0 * N_range)) > 1
            label = f"{design}" if ci == 0 else None
            ax.plot(N_range[valid], L[valid], color=color,
                    ls=cfg["ls"], lw=2.0, alpha=0.7, label=label)

        if show_optimal:
            C_ridge = np.logspace(
                np.log10(flop_budgets[0]), np.log10(flop_budgets[-1]), 300,
            )
            N_stars = _chinchilla_optimal_N(C_ridge, A, alpha, B, beta)
            D_stars = C_ridge / (6.0 * N_stars)
            L_stars = E + A / np.power(N_stars, alpha) + B / np.power(D_stars, beta)
            ax.plot(N_stars, L_stars, color=cfg["opt_color"],
                    ls=cfg["ls"], lw=3, alpha=0.9,
                    label=f"{design} optimal $N^*$")

    if holdout_data:
        from scaling_laws.data import prepare_arrays
        N_h, D_h, L_h = prepare_arrays(holdout_data, "final")
        if len(L_h) > 0:
            ax.scatter(N_h, L_h, c="black", s=25, alpha=0.5, zorder=5,
                       edgecolors="white", linewidths=0.5,
                       label=f"{holdout_label} actual")

            resid_cfg = {
                "Colinear":     {"color": "green", "label": "CO residual", "offset": -1},
                "Non-Colinear": {"color": "black", "label": "NC residual", "offset": +1},
            }
            log_span = (np.log10(N_h.max() / N_h.min())
                        if N_h.max() > N_h.min() else 1.0)
            nudge_pos = 10 ** (np.log10(N_h) + 0.008 * log_span)
            nudge_neg = 10 ** (np.log10(N_h) - 0.008 * log_span)
            tick_hw = 10 ** (np.log10(N_h) + 0.006 * log_span) - N_h

            for r in fits:
                design = r["design"]
                E, A, alpha, B, beta = r["params"]
                cfg = resid_cfg.get(design)
                if cfg is None:
                    continue
                L_pred = E + A / np.power(N_h, alpha) + B / np.power(D_h, beta)
                Nx = nudge_neg if cfg["offset"] < 0 else nudge_pos
                drawn = False
                for j in range(len(N_h)):
                    nj = Nx[j]
                    ax.plot([nj, nj], [L_h[j], L_pred[j]],
                            color=cfg["color"], lw=1.8, alpha=0.85, zorder=4,
                            label=cfg["label"] if not drawn else None)
                    ax.plot([nj - tick_hw[j], nj + tick_hw[j]],
                            [L_pred[j], L_pred[j]],
                            color=cfg["color"], lw=1.0, alpha=0.85, zorder=4)
                    drawn = True

    ax.set_xscale("log")
    ax.set_xlabel("$N$ (parameters)", fontsize=11)
    ax.set_ylabel("Loss $L$", fontsize=11)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3, which="both")

    if y_lim is not None:
        ax.set_ylim(y_lim)

    for ci, C in enumerate(flop_budgets):
        color = flop_cmap(ci / max(1, len(flop_budgets) - 1))
        exp = np.log10(C)
        txt = f"$C=10^{{{exp:.1f}}}$" if exp != int(exp) else f"$C=10^{{{int(exp)}}}$"
        ax.annotate(txt, xy=(0.02, 0.95 - ci * 0.055),
                    xycoords="axes fraction", fontsize=7, color=color, fontweight="bold")