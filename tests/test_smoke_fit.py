"""End-to-end fit on synthetic data in the released CSV format: parse, split, fit, recover.

Losses are generated from known Chinchilla and Kaplan parameters, written in the exact format
of the released ``extracted_losses/*.csv`` files, loaded through the paper's loader, and
fitted with the paper's fitters. The fitted parameters must match the planted ones, and a
single-ray (collinear) design must extrapolate off-ray far worse than a design of the same
size that spreads its tokens-per-parameter ratios - the paper's central claim in miniature.
"""

from __future__ import annotations

import numpy as np
import pytest

from scaling_laws.data import load_and_prepare, prepare_arrays
from scaling_laws.laws import chinchilla_loss, fit_chinchilla, fit_kaplan, kaplan_loss

CHINCHILLA = {"E": 1.69, "A": 406.4, "alpha": 0.34, "B": 410.7, "beta": 0.28}
KAPLAN = {"N_c": 8.8e13, "D_c": 5.4e13, "alpha_N": 0.076, "alpha_D": 0.095}

N_GRID = np.geomspace(1e6, 1e9, 6)
D_GRID = np.geomspace(1e8, 1e11, 6)
EPOCHS = 3


def _write_released_csv(path, loss_fn, params) -> None:
    """Blocks in the released format: header lines, then one row per epoch."""
    blocks = []
    for n in N_GRID:
        for d in D_GRID:
            blocks.append((int(n), int(d), "nxd_grid", False, False))
    for n in N_GRID:
        for k in (20.0, 40.0):
            blocks.append((int(n), int(k * n), "colinear", False, True))
    blocks.append((int(2e9), int(2e11), "holdout", True, False))
    blocks.append((int(2e9), int(40 * 2e9), "colinear_holdout", True, True))
    with open(path, "w") as f:
        for n, d, run_type, is_holdout, is_colinear in blocks:
            f.write(f"N={n}, D={d}, WD=0.1 [{run_type}]\n")
            f.write(f"run_type={run_type},is_holdout={is_holdout},is_colinear={is_colinear}\n")
            f.write("Epoch,Train Loss,Val Loss,Cumulative Time (s)\n")
            for ep in range(1, EPOCHS + 1):
                loss = float(loss_fn(n, d * ep, **params))
                f.write(f"{ep},{loss:.10f},,{ep * 10.0}\n")
            f.write("\n")


@pytest.fixture(scope="module")
def chinchilla_csv(tmp_path_factory):
    path = tmp_path_factory.mktemp("smoke") / "synthetic.csv"
    _write_released_csv(path, chinchilla_loss, CHINCHILLA)
    return path


def test_loader_splits_designs(chinchilla_csv):
    data = load_and_prepare(filepath=str(chinchilla_csv), dataset_name="synthetic", verbose=False)
    assert len(data.noncolinear_train) == len(N_GRID) * len(D_GRID)
    assert len(data.colinear_train) == 2 * len(N_GRID)
    N, D, L = prepare_arrays(data.noncolinear_train, mode="final")
    # 'final' mode: D is tokens seen over all epochs, L is the last-epoch loss
    np.testing.assert_allclose(L, chinchilla_loss(N, D, **CHINCHILLA), rtol=1e-8)


def test_chinchilla_parameters_recovered(chinchilla_csv):
    data = load_and_prepare(filepath=str(chinchilla_csv), dataset_name="synthetic", verbose=False)
    N, D, L = prepare_arrays(data.noncolinear_train, mode="final")
    params, sse, _ = fit_chinchilla(N, D, L, n_restarts=30)
    assert params is not None and sse < 1e-10
    np.testing.assert_allclose(params, list(CHINCHILLA.values()), rtol=1e-3)


def test_kaplan_parameters_recovered(tmp_path):
    path = tmp_path / "kaplan.csv"
    _write_released_csv(path, kaplan_loss, KAPLAN)
    data = load_and_prepare(filepath=str(path), dataset_name="synthetic", verbose=False)
    N, D, L = prepare_arrays(data.noncolinear_train, mode="final")
    params, sse, _ = fit_kaplan(N, D, L, n_restarts=30)
    assert params is not None and sse < 1e-12
    np.testing.assert_allclose(params, list(KAPLAN.values()), rtol=1e-3)


def test_collinear_design_does_not_extrapolate_off_ray():
    """Nine runs on one TPP ray vs nine runs spread over TPP 2-200, same sizes, noiseless."""
    n = np.geomspace(1e6, 1e9, 9)
    rng = np.random.default_rng(0)
    designs = {"CO": (n, 20 * n), "NC": (n, n * rng.permutation(np.geomspace(2, 200, 9)))}
    n_off, k_off = np.meshgrid(np.geomspace(1e6, 1e9, 5), [1.0, 1000.0])
    n_off = n_off.ravel()
    d_off = k_off.ravel() * n_off
    truth = chinchilla_loss(n_off, d_off, **CHINCHILLA)
    rmse = {}
    for name, (N, D) in designs.items():
        params, _, _ = fit_chinchilla(N, D, chinchilla_loss(N, D, **CHINCHILLA), n_restarts=30)
        rmse[name] = float(np.sqrt(np.mean((chinchilla_loss(n_off, d_off, *params) - truth) ** 2)))
    assert rmse["NC"] < 1e-4, rmse
    assert rmse["CO"] > 0.1, rmse
