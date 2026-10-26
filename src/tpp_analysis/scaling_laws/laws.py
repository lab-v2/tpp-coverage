"""
Scaling law definitions, fitting, and evaluation.

Three laws:
  1. Kaplan (2020)        - L = [(Nc/N)^(αN/αD) + Dc/D]^αD
  2. Chinchilla (2022)    - L = E + A/N^α + B/D^β
  3. Droppo-Elibol (2021) - L = [L∞^(1/α) + (Nc/N)^(αN/α) + (Dc/D)^(αD/α)]^α
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize, differential_evolution
from sklearn.metrics import r2_score, mean_squared_error
import warnings

warnings.filterwarnings("ignore")


# ════════════════════════════════════════════════════════════════════════════
# Loss functions
# ════════════════════════════════════════════════════════════════════════════

def kaplan_loss(N, D, N_c, D_c, alpha_N, alpha_D):
    """Kaplan et al. (2020) - no irreducible loss term."""
    ratio = alpha_N / alpha_D
    return np.power(np.power(N_c / N, ratio) + D_c / D, alpha_D)


def chinchilla_loss(N, D, E, A, alpha, B, beta):
    """Chinchilla (2022) - with irreducible loss E."""
    return E + A / np.power(N, alpha) + B / np.power(D, beta)


def droppo_elibol_loss(N, D, L_inf, N_c, D_c, alpha_N, alpha_D, alpha):
    """Droppo & Elibol (2021) - power-law with irreducible loss."""
    inner = (
        np.power(L_inf, 1.0 / alpha)
        + np.power(N_c / N, alpha_N / alpha)
        + np.power(D_c / D, alpha_D / alpha)
    )
    return np.power(inner, alpha)


# ════════════════════════════════════════════════════════════════════════════
# Generic multi-start fitter
# ════════════════════════════════════════════════════════════════════════════

def _fit_generic(
    loss_fn: Callable,
    N: np.ndarray,
    D: np.ndarray,
    L: np.ndarray,
    bounds: list,
    sampler: Callable,
    n_restarts: int = 300,
    seed: int = 42,
) -> Tuple[Optional[np.ndarray], float, bool]:
    """
    Multi-start L-BFGS-B + differential evolution fallback.

    Parameters
    ----------
    loss_fn   : vectorised function  (N, D, *params) -> L_pred
    bounds    : list of (lo, hi) for each parameter
    sampler   : callable(rng) -> list of initial guesses
    """
    def objective(params):
        try:
            pred = loss_fn(N, D, *params)
            if np.any(np.isnan(pred)) or np.any(np.isinf(pred)):
                return 1e10
            return np.sum((pred - L) ** 2)
        except Exception:
            return 1e10

    rng = np.random.RandomState(seed)
    best_result, best_loss = None, np.inf

    for _ in range(n_restarts):
        x0 = sampler(rng, L)
        try:
            result = minimize(
                objective, x0, method="L-BFGS-B", bounds=bounds,
                options={"maxiter": 2000},
            )
            if result.fun < best_loss:
                best_loss = result.fun
                best_result = result
        except Exception:
            continue

    # Differential evolution as a global fallback
    try:
        de = differential_evolution(
            objective, bounds, seed=seed, maxiter=1000, polish=True,
        )
        if de.fun < best_loss:
            best_loss = de.fun
            best_result = de
    except Exception:
        pass

    if best_result is None:
        return None, np.inf, False
    return best_result.x, best_result.fun, best_result.success


# ════════════════════════════════════════════════════════════════════════════
# Specific fitters
# ════════════════════════════════════════════════════════════════════════════

def fit_kaplan(N, D, L, n_restarts=300):
    """Fit Kaplan law. Returns (params, sse, success)."""
    bounds = [(1e3, 1e14), (1e3, 1e14), (0.01, 2.0), (0.01, 2.0)]

    def sampler(rng, L):
        return [
            10 ** rng.uniform(5, 12),
            10 ** rng.uniform(5, 12),
            rng.uniform(0.05, 1.5),
            rng.uniform(0.05, 1.5),
        ]

    return _fit_generic(kaplan_loss, N, D, L, bounds, sampler, n_restarts)


def fit_chinchilla(N, D, L, n_restarts=300):
    """Fit Chinchilla law. Returns (params, sse, success)."""
    bounds = [(0.0, 10.0), (1e-2, 1e10), (0.01, 2.0), (1e-2, 1e10), (0.01, 2.0)]

    def sampler(rng, L):
        return [
            rng.uniform(0.0, min(L) * 0.9),
            10 ** rng.uniform(0, 8),
            rng.uniform(0.1, 1.5),
            10 ** rng.uniform(0, 8),
            rng.uniform(0.1, 1.5),
        ]

    return _fit_generic(chinchilla_loss, N, D, L, bounds, sampler, n_restarts)


def fit_droppo_elibol(N, D, L, n_restarts=300):
    """Fit Droppo-Elibol law. Returns (params, sse, success)."""
    bounds = [
        (1e-6, min(L) * 0.99),
        (1e3, 1e14), (1e3, 1e14),
        (0.01, 2.0), (0.01, 2.0), (0.01, 2.0),
    ]

    def sampler(rng, L):
        return [
            rng.uniform(0.01, min(L) * 0.8),
            10 ** rng.uniform(5, 12),
            10 ** rng.uniform(5, 12),
            rng.uniform(0.05, 1.5),
            rng.uniform(0.05, 1.5),
            rng.uniform(0.05, 1.5),
        ]

    return _fit_generic(droppo_elibol_loss, N, D, L, bounds, sampler, n_restarts)


# ════════════════════════════════════════════════════════════════════════════
# Evaluation
# ════════════════════════════════════════════════════════════════════════════

def evaluate(
    loss_fn: Callable,
    params,
    N_train, D_train, L_train,
    N_hold, D_hold, L_hold,
) -> Dict[str, float]:
    """Compute R² and RMSE on train and holdout sets."""
    pred_train = loss_fn(N_train, D_train, *params)
    r2_tr = r2_score(L_train, pred_train)
    rmse_tr = float(np.sqrt(mean_squared_error(L_train, pred_train)))

    if len(L_hold) > 0:
        pred_hold = loss_fn(N_hold, D_hold, *params)
        r2_ho = r2_score(L_hold, pred_hold)
        rmse_ho = float(np.sqrt(mean_squared_error(L_hold, pred_hold)))
    else:
        r2_ho = np.nan
        rmse_ho = np.nan

    return {
        "r2_train": r2_tr,
        "r2_holdout": r2_ho,
        "rmse_train": rmse_tr,
        "rmse_holdout": rmse_ho,
    }


# ════════════════════════════════════════════════════════════════════════════
# Parametric Epoch law  (Section 3.1 - with D' and N' half-lives)
# L = A/N'^α + B/D'^β + E
# where D' = U_D + U_D·R*_D·(1 − e^(−R/R*_D))
#       N' = U_N + U_N·R*_N·(1 − e^(−R/R*_N))
# 7 params: A, α, B, β, E, R*_D, R*_N
# ════════════════════════════════════════════════════════════════════════════

def effective_data(U_D, R, R_star_D):
    """D' = U_D + U_D · R*_D · (1 − exp(−R / R*_D)).  R=0 → D'=U_D."""
    return U_D + U_D * R_star_D * (1.0 - np.exp(-R / R_star_D))


def effective_params(U_N, R, R_star_N):
    """N' = U_N + U_N · R*_N · (1 − exp(−R / R*_N)).  R=0 → N'=U_N."""
    return U_N + U_N * R_star_N * (1.0 - np.exp(-R / R_star_N))


def parametric_epoch_loss(N, U_D, R, A, alpha, B, beta, E, R_star_D, R_star_N):
    """
    Full parametric scaling law with epoch half-lives.
    L = A / N'^α  +  B / D'^β  +  E
    """
    D_prime = effective_data(U_D, R, R_star_D)
    N_prime = effective_params(N, R, R_star_N)
    return A / np.power(N_prime, alpha) + B / np.power(D_prime, beta) + E


def _parametric_predict(params_dict, N, U_D, R):
    """Convenience: predict from a params dict."""
    return parametric_epoch_loss(
        N, U_D, R,
        params_dict["A"], params_dict["alpha"],
        params_dict["B"], params_dict["beta"],
        params_dict["E"],
        params_dict["R_star_D"], params_dict["R_star_N"],
    )


def fit_parametric_epoch(N, U_D, R, L, n_restarts=300, seed=42):
    """
    Fit the 7-param epoch law via Huber-loss minimisation.
    Returns (params_dict, loss, success).
    """
    bounds = [
        (1e-2, 1e12),   # A
        (0.01, 2.0),    # alpha
        (1e-2, 1e12),   # B
        (0.01, 2.0),    # beta
        (0.0, 10.0),    # E
        (0.1, 50.0),    # R*_D
        (0.1, 50.0),    # R*_N
    ]
    delta = 0.5  # Huber threshold

    def objective(params):
        A, alpha, B, beta, E_val, R_star_D, R_star_N = params
        try:
            pred = parametric_epoch_loss(N, U_D, R, A, alpha, B, beta, E_val, R_star_D, R_star_N)
            if np.any(np.isnan(pred)) or np.any(np.isinf(pred)) or np.any(pred < 0):
                return 1e10
            res = pred - L
            a = np.abs(res)
            return np.sum(np.where(a <= delta, 0.5 * res**2, delta * (a - 0.5 * delta)))
        except Exception:
            return 1e10

    rng = np.random.RandomState(seed)
    best_result, best_loss = None, np.inf

    for _ in range(n_restarts):
        x0 = [
            10 ** rng.uniform(0, 8),
            rng.uniform(0.05, 1.5),
            10 ** rng.uniform(0, 8),
            rng.uniform(0.05, 1.5),
            rng.uniform(0.0, min(L) * 0.9),
            rng.uniform(0.5, 20.0),
            rng.uniform(0.5, 20.0),
        ]
        try:
            result = minimize(
                objective, x0, method="L-BFGS-B", bounds=bounds,
                options={"maxiter": 3000, "ftol": 1e-15},
            )
            if result.fun < best_loss:
                best_loss, best_result = result.fun, result
        except Exception:
            continue

    # Differential evolution fallback
    try:
        de = differential_evolution(
            objective, bounds, seed=seed, maxiter=2000, polish=True,
            tol=1e-12, mutation=(0.5, 1.5), recombination=0.9,
        )
        if de.fun < best_loss:
            best_loss, best_result = de.fun, de
    except Exception:
        pass

    if best_result is None:
        return None, np.inf, False

    A, alpha, B, beta, E_val, R_star_D, R_star_N = best_result.x
    params_dict = {
        "A": A, "alpha": alpha, "B": B, "beta": beta,
        "E": E_val, "R_star_D": R_star_D, "R_star_N": R_star_N,
    }
    return params_dict, best_loss, best_result.success


def evaluate_parametric_epoch(params_dict, N_tr, U_D_tr, R_tr, L_tr,
                               N_ho, U_D_ho, R_ho, L_ho):
    """Compute R² and RMSE for the epoch law on train and holdout."""
    pred_tr = _parametric_predict(params_dict, N_tr, U_D_tr, R_tr)
    r2_tr = r2_score(L_tr, pred_tr)
    rmse_tr = float(np.sqrt(mean_squared_error(L_tr, pred_tr)))

    if len(L_ho) > 0:
        pred_ho = _parametric_predict(params_dict, N_ho, U_D_ho, R_ho)
        r2_ho = r2_score(L_ho, pred_ho)
        rmse_ho = float(np.sqrt(mean_squared_error(L_ho, pred_ho)))
    else:
        r2_ho, rmse_ho = np.nan, np.nan

    return {
        "r2_train": r2_tr, "r2_holdout": r2_ho,
        "rmse_train": rmse_tr, "rmse_holdout": rmse_ho,
    }


# ════════════════════════════════════════════════════════════════════════════
# Registry - makes looping over all three laws easy
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class ScalingLaw:
    name: str
    loss_fn: Callable
    fit_fn: Callable
    param_names: List[str]
    is_epoch_law: bool = False   # True = uses (N, U_D, R) interface

LAWS: List[ScalingLaw] = [
    ScalingLaw("Kaplan",        kaplan_loss,        fit_kaplan,        ["N_c", "D_c", "α_N", "α_D"]),
    ScalingLaw("Chinchilla",    chinchilla_loss,    fit_chinchilla,    ["E", "A", "α", "B", "β"]),
    ScalingLaw("Droppo-Elibol", droppo_elibol_loss, fit_droppo_elibol, ["L_∞", "N_c", "D_c", "α_N", "α_D", "α"]),
    ScalingLaw("Parametric-Epoch", parametric_epoch_loss, fit_parametric_epoch,
               ["A", "α", "B", "β", "E", "R*_D", "R*_N"], is_epoch_law=True),
]
