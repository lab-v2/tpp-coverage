"""Shared isoFLOP-panel renderer used by ``figure1_panels.py``,
``visuals.ipynb`` and ``special_viz_notebook_bf16_v2.ipynb``.

The single source of truth for the function is here so changes to the
isoFLOP plot (curve set, colors, residual style, etc.) propagate to
every caller without hand-editing each notebook.
"""
from collections.abc import Callable

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.optimize import minimize_scalar

from scaling_laws.data import prepare_arrays
from scaling_laws.laws import (
    chinchilla_loss,
    kaplan_loss,
    droppo_elibol_loss,
)


LAW_LOSS_FNS: dict[str, Callable] = {
    'Chinchilla':    chinchilla_loss,
    'Kaplan':        kaplan_loss,
    'Droppo-Elibol': droppo_elibol_loss,
}

ISOFLOP_COLORS: list[str] = [
    '#000000', '#E60000', '#FF8C00', '#0055FF', '#FFD700',
]

LEGEND_ALPHA: float = 0.9


def compute_flop_budgets(
    pdata, holdout_type: str = 'unified',
    n_curves: int | None = None,
) -> np.ndarray:
    """Top 25% of unique C values in the holdout, evenly sampled to
    exactly ``n_curves`` values (default: ``len(ISOFLOP_COLORS)``).
    Used by callers that want a thinned curve set; the panel renderer
    itself overrides this with the per-holdout induced budgets when
    holdout data is supplied."""
    if n_curves is None:
        n_curves = len(ISOFLOP_COLORS)

    if holdout_type == 'unified':
        h_list = pdata.unified_holdout
    elif holdout_type == 'colinear':
        h_list = pdata.colinear_holdout
    else:
        h_list = pdata.noncolinear_holdout

    C_all = np.array([6.0 * r['N'] * r['D'] for r in h_list])
    C_unique = np.unique(C_all)
    if len(C_unique) == 0:
        return np.array([])

    C_top25 = C_unique[3 * len(C_unique) // 4:]
    if len(C_top25) <= n_curves:
        sampled = C_top25
    else:
        idxs = np.linspace(0, len(C_top25) - 1, n_curves, dtype=int)
        sampled = C_top25[idxs]

    print(
        f"\n[DEBUG] Selected {len(sampled)} FLOP Budgets "
        f"for '{holdout_type}':"
    )
    for C_val in sampled:
        for r in h_list:
            if np.isclose(6.0 * r['N'] * r['D'], C_val, rtol=1e-5):
                print(
                    f"  C = {C_val:.3e}  <--  "
                    f"(N = {r['N']:.3e}, D = {r['D']:.3e})"
                )
                break

    return sampled


def generic_isoflop(
    N_range: np.ndarray, C: float, law_name: str,
    params: tuple | list,
) -> np.ndarray:
    D = C / (6.0 * N_range)
    return LAW_LOSS_FNS[law_name](N_range, D, *params)


def generic_optimal_N(
    C_range: np.ndarray | float, law_name: str,
    params: tuple | list,
    N_lo: float = 1e4, N_hi: float = 1e12,
) -> np.ndarray:
    loss_fn = LAW_LOSS_FNS[law_name]

    def _opt_one(C_val: float) -> float:
        def _obj(log_n: float) -> float:
            N = 10 ** log_n
            D = C_val / (6.0 * N)
            if D < 1:
                return 1e10
            try:
                v = float(loss_fn(np.array([N]), np.array([D]), *params))
                return v if np.isfinite(v) else 1e10
            except Exception:
                return 1e10

        res = minimize_scalar(
            _obj, bounds=(np.log10(N_lo), np.log10(N_hi)),
            method='bounded',
        )
        return 10 ** res.x

    return np.array([_opt_one(c) for c in np.atleast_1d(C_range)])


def draw_isoflop_panel(
    ax,
    fits: list[dict],
    law_name: str,
    holdout_data: list[dict] | None,
    holdout_label: str,
    flop_budgets: np.ndarray,
    N_range: np.ndarray,
    y_lim: tuple[float, float] | None,
    x_lim: tuple[float | None, float | None] | None = None,
    show_optimal: bool = False,
    fontsize: float = 18,
    legend_fontsize: float = 18,
    legend_loc: str = 'lower left',
    line_width: float = 2.0,
    holdout_mode: str = 'final',
    style_legends_fn: Callable | None = None,
    debug: bool = False,
):
    """
    Draw isoFLOP curves.  When ``holdout_data`` is supplied the
    n_H induced budgets C_i = 6 N_i D_i (one per kept holdout point,
    after the highest-D-per-unique-N filter) are used as the curve
    set, and each residual lands exactly on its own induced curve.

    ``style_legends_fn`` is an optional callback ``(ax) -> None`` used
    by callers to harmonize legend frames across panels.
    """
    _DESIGN_ABBREV = {'Non-Colinear': 'NC', 'Colinear': 'CO'}
    design_cfg = {
        'Non-Colinear': {'ls': '-',  'opt_color': 'red'},
        'Colinear':     {'ls': '--', 'opt_color': 'blue'},
    }

    if holdout_data:
        raw_N_all = np.array([r['N'] for r in holdout_data])
        if len(raw_N_all):
            n_lo = float(min(N_range.min(), raw_N_all.min()))
            n_hi = float(max(N_range.max(), raw_N_all.max()))
            N_range = np.logspace(np.log10(n_lo), np.log10(n_hi), 500)

    N_h = D_h = L_h = raw_C = None
    if holdout_data:
        _N_full, _D_full, _L_full = prepare_arrays(
            holdout_data, holdout_mode,
        )
        _C_full = 6.0 * _N_full * _D_full
        if len(_N_full) > 0:
            _order = np.lexsort((-_D_full, _N_full))
            _N_sorted = _N_full[_order]
            _keep = np.concatenate(
                ([True], _N_sorted[1:] != _N_sorted[:-1]),
            )
            _idx = _order[_keep]
            N_h = _N_full[_idx]
            D_h = _D_full[_idx]
            L_h = _L_full[_idx]
            raw_C = _C_full[_idx]
            _csort = np.argsort(raw_C)
            N_h, D_h, L_h, raw_C = (
                N_h[_csort], D_h[_csort], L_h[_csort], raw_C[_csort],
            )
            flop_budgets = raw_C

    n_budgets = len(flop_budgets)
    if n_budgets <= len(ISOFLOP_COLORS):
        colors = ISOFLOP_COLORS[:n_budgets]
    else:
        colors = plt.cm.plasma(np.linspace(0.05, 0.95, n_budgets))

    for r in fits:
        design = r['design']
        params = r['params']
        cfg = design_cfg.get(
            design, {'ls': '-.', 'opt_color': 'gray'},
        )

        for ci, C in enumerate(flop_budgets):
            L = generic_isoflop(N_range, C, law_name, params)
            valid = (C / (6.0 * N_range)) > 1
            label = (
                _DESIGN_ABBREV.get(design, design) if ci == 0 else None
            )
            ax.plot(
                N_range[valid], L[valid], color=colors[ci],
                ls=cfg['ls'], lw=line_width, alpha=0.7, label=label,
                zorder=3,
            )

        if show_optimal:
            C_ridge = np.logspace(
                np.log10(flop_budgets[0]),
                np.log10(flop_budgets[-1]), 300,
            )
            N_stars = generic_optimal_N(C_ridge, law_name, params)
            D_stars = C_ridge / (6.0 * N_stars)
            L_stars = LAW_LOSS_FNS[law_name](
                N_stars, D_stars, *params,
            )
            ax.plot(
                N_stars, L_stars, color=cfg['opt_color'],
                ls=cfg['ls'], lw=3, alpha=0.9,
                label=(
                    f"{_DESIGN_ABBREV.get(design, design)} "
                    f"optimal $N^*$"
                ),
            )

    if N_h is not None and len(N_h) > 0:
        for ci in range(n_budgets):
            ax.scatter(
                [N_h[ci]], [L_h[ci]],
                color=colors[ci], s=60, alpha=1.0,
                zorder=5, edgecolors='black', linewidths=1.0,
                label='Holdout' if ci == 0 else None,
            )

    _RESID_CFG = {
        'Colinear':     {'color': 'green', 'label': 'CO residual'},
        'Non-Colinear': {'color': 'black', 'label': 'NC residual'},
    }
    if N_h is not None and len(N_h) > 0:
        for r in fits:
            design = r['design']
            cfg = _RESID_CFG.get(design)
            lfn = LAW_LOSS_FNS.get(law_name)
            if cfg is None or lfn is None:
                continue
            L_pred = lfn(N_h, D_h, *r['params'])
            drawn = False
            for j in range(len(N_h)):
                ax.plot(
                    [N_h[j], N_h[j]], [L_h[j], L_pred[j]],
                    color=cfg['color'], lw=1.8, alpha=0.85, zorder=4,
                    label=cfg['label'] if not drawn else None,
                )
                drawn = True

    ax.set_xscale('log')
    ax.set_xlabel('N', fontsize=fontsize)
    ax.set_ylabel('Loss', fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    ax.tick_params(axis='both', which='minor', labelsize=fontsize - 2)
    ax.legend(
        fontsize=legend_fontsize, loc=legend_loc,
        framealpha=LEGEND_ALPHA,
    )
    ax.grid(True, alpha=0.3, which='both')

    if y_lim is not None:
        ax.set_ylim(y_lim)
    elif holdout_data:
        _Nh2, _Dh2, _Lh2 = prepare_arrays(holdout_data, holdout_mode)
        if len(_Lh2) > 0:
            ax.set_ylim(
                float(_Lh2.min()) - 1.0, float(_Lh2.max()) + 1.6,
            )

    if x_lim is not None:
        ax.set_xlim(*x_lim)

    if n_budgets <= 6:
        budget_fontsize = 19.5
    else:
        budget_fontsize = max(8, 19.5 - 0.7 * (n_budgets - 6))

    budget_proxies = []
    budget_labels = []

    if holdout_data:
        raw_N_all = np.array([r['N'] for r in holdout_data])
        raw_D_all = np.array([r['D'] for r in holdout_data])
        raw_C_all = 6.0 * raw_N_all * raw_D_all
    else:
        raw_C_all, raw_N_all, raw_D_all = (
            np.array([]), np.array([]), np.array([]),
        )

    def fmt_sci(val: float) -> str:
        b, e = f"{val:.1e}".split('e')
        b_str = f"{float(b):.1f}".replace(".0", "")
        if b_str == "1":
            return f"10^{{{int(e)}}}"
        return f"{b_str}\\cdot 10^{{{int(e)}}}"

    for ci, C in enumerate(flop_budgets):
        exp = np.log10(C)
        if exp != int(exp):
            c_str = f"10^{{{exp:.2f}}}"
        else:
            c_str = f"10^{{{int(exp)}}}"
        if len(raw_C_all) > 0:
            idx = np.argmin(np.abs(raw_C_all - C))
            best_N = raw_N_all[idx]
            best_D = raw_D_all[idx]
            txt = (
                f"$C={c_str}$  "
                f"($N={fmt_sci(best_N)}, D={fmt_sci(best_D)}$)"
            )
        else:
            txt = f"$C={c_str}$"
        budget_proxies.append(
            Line2D([0], [0], color=colors[ci], lw=line_width),
        )
        budget_labels.append(txt)

    main_leg = ax.get_legend()

    ax.legend(
        budget_proxies, budget_labels, loc='upper right',
        fontsize=budget_fontsize, framealpha=LEGEND_ALPHA,
    )

    if main_leg:
        ax.add_artist(main_leg)

    if style_legends_fn is not None:
        style_legends_fn(ax)
