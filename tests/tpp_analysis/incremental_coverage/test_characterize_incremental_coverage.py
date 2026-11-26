"""Characterization of `incremental_coverage`: the step builders, the incremental fits and the plots.

Inputs are the released Wikipedia loss table. The plots draw from `frozen/incremental_wikipedia.parquet`,
the base commit's `run_full_incremental` output for every law (1 restart), written when the frozen
files are (TPP_FREEZE=1); the other expected values are frozen in `frozen/*.json`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from characterize import FREEZE, FigureCapture, check_frozen, released

pytestmark = pytest.mark.network
FROZEN = Path(__file__).parent / "frozen"
INCREMENTAL = FROZEN / "incremental_wikipedia.parquet"


@pytest.fixture(scope="module")
def wiki():
    from scaling_laws.data import load_and_prepare

    root = released("extracted_losses/wikipedia.csv")
    return load_and_prepare(
        str(root / "extracted_losses/wikipedia.csv"), "Wikipedia", verbose=False
    )


def _holdouts(data) -> dict:
    return {
        "unified": data.unified_holdout,
        "colinear": data.colinear_holdout,
        "noncolinear": data.noncolinear_holdout,
    }


def _step_view(steps: list[dict]) -> list[dict]:
    return [
        {k: s[k] for k in ("step", "n_levels", "levels", "n_runs", "coverage_frac")}
        | {"N": [d["N"] for d in s["data"]], "D": [d["D"] for d in s["data"]]}
        for s in steps
    ]


def test_step_builders(wiki):
    from incremental_coverage import coverage as cov

    check_frozen(
        FROZEN / "steps.json",
        {
            "bins": {m: cov.effective_tpp_bins(m) for m in ("first", "second", "final", "other")},
            "binned": {m: cov.assign_tpp_bins(wiki.colinear_train, m) for m in cov.EPOCH_MODES},
            "colinear": {
                m: _step_view(cov.build_colinear_steps(wiki.colinear_train, m))
                for m in cov.EPOCH_MODES
            },
            "noncolinear": _step_view(cov.build_noncolinear_steps(wiki.noncolinear_train)),
            "noncolinear_odd": _step_view(
                cov.build_noncolinear_steps(wiki.noncolinear_train[:-10])
            ),
        },
    )


@pytest.mark.slow
def test_run_full_incremental(wiki, monkeypatch):
    """The whole CO and NC sweep for the Kaplan law (the fastest; 1 restart), deltas included."""
    from incremental_coverage import coverage as cov

    monkeypatch.setattr(cov, "LAWS", [law for law in cov.LAWS if law.name == "Kaplan"])
    df = cov.run_full_incremental(wiki, n_restarts=1)
    check_frozen(FROZEN / "run_full_incremental_kaplan.json", df)


def test_epoch_law_at_one_step(wiki):
    from incremental_coverage import coverage as cov
    from scaling_laws.laws import LAWS

    epoch = [law for law in LAWS if law.is_epoch_law]
    steps = cov.build_colinear_steps(wiki.colinear_train, mode="first")[2:4]
    df = cov.run_incremental_analysis(
        steps, _holdouts(wiki), "Colinear", modes=["first"], laws_to_fit=epoch, n_restarts=1
    )
    check_frozen(FROZEN / "epoch_law_steps.json", cov.compute_deltas(df))


@pytest.fixture(scope="module")
def incremental(wiki) -> pd.DataFrame:
    """The base `run_full_incremental` output for every law: computed and stored when freezing."""
    if FREEZE:
        from incremental_coverage import coverage as cov

        df = cov.run_full_incremental(wiki, n_restarts=1, verbose=False)
        stored = df.assign(params=df["params"].map(lambda p: json.dumps(_plain(p))))
        INCREMENTAL.parent.mkdir(parents=True, exist_ok=True)
        stored.to_parquet(INCREMENTAL, index=False)
    df = pd.read_parquet(INCREMENTAL)
    return df.assign(params=df["params"].map(json.loads))


def _plain(p):
    if isinstance(p, dict):
        return {k: float(v) for k, v in p.items()}
    return None if p is None else [float(v) for v in p]


def test_plots(incremental):
    from incremental_coverage import viz

    df = incremental
    with FigureCapture() as cap:
        viz.plot_r2_convergence(df, suptitle="Wikipedia")
        viz.plot_r2_convergence(
            df, holdout_col="r2_noncolinear", mode_filter="First", min_coverage=0.3
        )
        viz.plot_delta_convergence(df, mode_filter="Final")
        viz.plot_co_vs_nc_overlay(df, holdout_col="r2_colinear", y_min=None, y_max=None)
        for law in ("Chinchilla", "Parametric-Epoch"):
            viz.plot_parameter_trajectories(df, law)
        viz.plot_parameter_trajectories(
            df, "Kaplan", param_names=["N_c", "D_c", "a_N", "a_D"], mode_filter="Second"
        )
        for design in ("Colinear", "Non-Colinear"):
            viz.plot_r2_heatmap(df, design)
        viz.plot_r2_heatmap(df, "Colinear", holdout_col="r2_train", mode_filter="First")
        viz.plot_summary_dashboard(df, suptitle="Wikipedia", min_coverage=0.2)
        viz.plot_r2_rate_of_change(df, mode_filter="Final")
        viz.plot_mode_comparison(df, "Droppo-Elibol")
        viz.plot_single_comparison(df, "Chinchilla", "Final")
        viz.plot_single_comparison(
            df, "Kaplan", "Second", holdout_col="r2_colinear", min_coverage=0.5
        )
    check_frozen(FROZEN / "plots.json", cap.figures)
