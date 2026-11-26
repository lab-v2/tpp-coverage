"""Characterization of the panel helpers Figure 1 does not reach with its default settings.

`isoflop_panel` with the compute-optimal ridge and every holdout split, and the `surface_helpers`
geometry (camera depth, the CO/NC surface intersection), on the released Wikipedia loss table and
the released full-coverage fits (seed 0). Expected values are the base commit's outputs, frozen
in `frozen/` (see `tests/characterize.py`).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from characterize import FigureCapture, check_frozen, released, released_fits
from conftest import ANALYSIS

pytestmark = pytest.mark.network
FROZEN = Path(__file__).parent / "frozen"
sys.path.insert(0, str(ANALYSIS / "visualizations"))


@pytest.fixture(scope="module")
def inputs():
    from scaling_laws.data import load_and_prepare

    root = released(
        "extracted_losses/wikipedia.csv", "full_coverage_results/full_coverage_all_seeds.parquet"
    )
    wiki = load_and_prepare(
        str(root / "extracted_losses/wikipedia.csv"), "Wikipedia", verbose=False
    )
    table = pd.read_parquet(root / "full_coverage_results/full_coverage_all_seeds.parquet")
    fits = [f for f in released_fits(table) if f["dataset"] == "Wikipedia"]
    return wiki, fits


@pytest.mark.parametrize("law", ["Chinchilla", "Kaplan", "Droppo-Elibol"])
def test_isoflop_panel(inputs, law):
    import isoflop_panel as ip

    wiki, fits = inputs
    fits = [f for f in fits if f["law"] == law and f["mode"] == "Final"]
    holdouts = {
        "unified": wiki.unified_holdout,
        "colinear": wiki.colinear_holdout,
        "noncolinear": wiki.noncolinear_holdout,
    }
    budgets = {h: ip.compute_flop_budgets(wiki, h) for h in holdouts}
    n_range = np.logspace(6, 9, 120)
    with FigureCapture() as cap:
        for h, data in holdouts.items():
            fig, ax = plt.subplots()
            ip.draw_isoflop_panel(
                ax,
                fits,
                law,
                data,
                h,
                budgets[h],
                n_range,
                None,
                show_optimal=True,
                holdout_mode="final",
            )
        fig, ax = plt.subplots()
        ip.draw_isoflop_panel(
            ax, fits, law, None, "none", budgets["unified"], n_range, (2.0, 6.0), x_lim=(1e6, None)
        )
    optimal = ip.generic_optimal_N(budgets["unified"], law, fits[0]["params"])
    check_frozen(
        FROZEN / f"isoflop_panel_{law}.json",
        {
            "budgets": budgets,
            "budgets_3": ip.compute_flop_budgets(wiki, "colinear", n_curves=3),
            "optimal": optimal,
            "figures": cap.figures,
        },
    )


def test_surface_helpers(inputs):
    import surface_helpers as sh
    from scaling_laws.laws import LAWS

    _, fits = inputs
    law = next(x for x in LAWS if x.name == "Chinchilla")
    co, nc = (
        next(f for f in fits if f["law"] == law.name and f["mode"] == "Final" and f["design"] == d)
        for d in ("Colinear", "Non-Colinear")
    )
    N, D = np.meshgrid(np.logspace(6.7, 7.9, 40), np.logspace(7, 9.5, 40))
    L_co, L_nc = law.loss_fn(N, D, *co["params"]), law.loss_fn(N, D, *nc["params"])
    lims = [(float(a.min()), float(a.max())) for a in (N, D, L_co)]
    labels = ["Unified Holdout", "CO Holdout", "NC Holdout", "Non-Colinear", "train"]
    check_frozen(
        FROZEN / "surface_helpers.json",
        {
            "styles": [sh._holdout_style(x) for x in labels],
            "r2_keys": [sh._r2_key_for_holdout(x) for x in labels],
            "rgba": sh._build_adaptive_rgba("steelblue", L_co, L_nc),
            "depth": sh._camera_depth(N, D, L_co, 15, -35, *lims),
            "intersection": sh._compute_intersection_3d(
                N, D, L_co, L_nc, law, co["params"], 2.0, 8.0
            ),
        },
    )
