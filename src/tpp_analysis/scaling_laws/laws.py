"""
Scaling law definitions, fitting, and evaluation.

The paper's three (N, D) laws, plus the repeated-data epoch law:
  1. Kaplan (2020)        - L = [(Nc/N)^(αN/αD) + Dc/D]^αD
  2. Chinchilla (2022)    - L = E + A/N^α + B/D^β
  3. Droppo-Elibol (2021) - L = [L∞^(1/α) + (Nc/N)^(αN/α) + (Dc/D)^(αD/α)]^α

and one law that is not part of the paper's analysis, available for follow-up work:
  Skaling (2026)          - L = (A/N^α + B/D^β)^k + E     [arXiv:2608.07222]
It is in ``EXTRA_LAWS``, not ``LAWS``, so every table and figure of the paper is unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.optimize import minimize, differential_evolution
from sklearn.metrics import r2_score, mean_squared_error
import warnings

warnings.filterwarnings("ignore")


# ════════════════════════════════════════════════════════════════════════════
# Loss functions
# ════════════════════════════════════════════════════════════════════════════


def kaplan_loss(
    N: np.ndarray | float,
    D: np.ndarray | float,
    N_c: float,
    D_c: float,
    alpha_N: float,
    alpha_D: float,
) -> np.ndarray | float:
    """Kaplan et al. (2020) - no irreducible loss term."""
    ratio = alpha_N / alpha_D
    return np.power(np.power(N_c / N, ratio) + D_c / D, alpha_D)


def chinchilla_loss(
    N: np.ndarray | float,
    D: np.ndarray | float,
    E: float,
    A: float,
    alpha: float,
    B: float,
    beta: float,
) -> np.ndarray | float:
    """Chinchilla (2022) - with irreducible loss E."""
    return E + A / np.power(N, alpha) + B / np.power(D, beta)


def droppo_elibol_loss(
    N: np.ndarray | float,
    D: np.ndarray | float,
    L_inf: float,
    N_c: float,
    D_c: float,
    alpha_N: float,
    alpha_D: float,
    alpha: float,
) -> np.ndarray | float:
    """Droppo & Elibol (2021) - power-law with irreducible loss."""
    inner = (
        np.power(L_inf, 1.0 / alpha)
        + np.power(N_c / N, alpha_N / alpha)
        + np.power(D_c / D, alpha_D / alpha)
    )
    return np.power(inner, alpha)


def skaling_loss(
    N: np.ndarray | float,
    D: np.ndarray | float,
    A: float,
    alpha: float,
    B: float,
    beta: float,
    k: float,
    E: float,
) -> np.ndarray | float:
    """Skaling (2026), arXiv:2608.07222 Eq. 3: L = (A/N^alpha + B/D^beta)^k + E.

    Chinchilla's two additive terms, their sum raised to a free exponent k. At k == 1 it equals
    `chinchilla_loss` to floating-point equality, so any gain over Chinchilla is due to k alone.
    k != 1 gives d2L/dNdD = k(k-1) u^(k-2) alpha A N^(-alpha-1) beta B D^(-beta-1), u = A/N^alpha
    + B/D^beta, zero for additive laws; L still decreases in N and D for every k > 0.
    """
    inner = A / np.power(N, alpha) + B / np.power(D, beta)
    return np.power(inner, k) + E


# ════════════════════════════════════════════════════════════════════════════
# Generic multi-start fitter
# ════════════════════════════════════════════════════════════════════════════


def _fit_generic(
    loss_fn: Callable,
    N: np.ndarray,
    D: np.ndarray,
    L: np.ndarray,
    bounds: list[tuple[float, float]],
    sampler: Callable,
    n_restarts: int = 300,
    seed: int = 42,
) -> tuple[np.ndarray | None, float, bool]:
    """Multi-start L-BFGS-B with a differential-evolution fallback.

    Args:
        loss_fn: vectorised function (N, D, *params) -> L_pred.
        bounds: (lo, hi) for each parameter.
        sampler: callable(rng, L) -> list of initial guesses.
    """

    def objective(params: np.ndarray) -> float:
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
                objective,
                x0,
                method="L-BFGS-B",
                bounds=bounds,
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
            objective,
            bounds,
            seed=seed,
            maxiter=1000,
            polish=True,
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


def fit_kaplan(
    N: np.ndarray, D: np.ndarray, L: np.ndarray, n_restarts: int = 300
) -> tuple[np.ndarray | None, float, bool]:
    """Fit Kaplan law. Returns (params, sse, success)."""
    bounds = [(1e3, 1e14), (1e3, 1e14), (0.01, 2.0), (0.01, 2.0)]

    def sampler(rng: np.random.RandomState, L: np.ndarray) -> list[float]:
        return [
            10 ** rng.uniform(5, 12),
            10 ** rng.uniform(5, 12),
            rng.uniform(0.05, 1.5),
            rng.uniform(0.05, 1.5),
        ]

    return _fit_generic(kaplan_loss, N, D, L, bounds, sampler, n_restarts)


def fit_chinchilla(
    N: np.ndarray, D: np.ndarray, L: np.ndarray, n_restarts: int = 300
) -> tuple[np.ndarray | None, float, bool]:
    """Fit Chinchilla law. Returns (params, sse, success)."""
    bounds = [(0.0, 10.0), (1e-2, 1e10), (0.01, 2.0), (1e-2, 1e10), (0.01, 2.0)]

    def sampler(rng: np.random.RandomState, L: np.ndarray) -> list[float]:
        return [
            rng.uniform(0.0, min(L) * 0.9),
            10 ** rng.uniform(0, 8),
            rng.uniform(0.1, 1.5),
            10 ** rng.uniform(0, 8),
            rng.uniform(0.1, 1.5),
        ]

    return _fit_generic(chinchilla_loss, N, D, L, bounds, sampler, n_restarts)


def fit_droppo_elibol(
    N: np.ndarray, D: np.ndarray, L: np.ndarray, n_restarts: int = 300
) -> tuple[np.ndarray | None, float, bool]:
    """Fit Droppo-Elibol law. Returns (params, sse, success)."""
    bounds = [
        (1e-6, min(L) * 0.99),
        (1e3, 1e14),
        (1e3, 1e14),
        (0.01, 2.0),
        (0.01, 2.0),
        (0.01, 2.0),
    ]

    def sampler(rng: np.random.RandomState, L: np.ndarray) -> list[float]:
        return [
            rng.uniform(0.01, min(L) * 0.8),
            10 ** rng.uniform(5, 12),
            10 ** rng.uniform(5, 12),
            rng.uniform(0.05, 1.5),
            rng.uniform(0.05, 1.5),
            rng.uniform(0.05, 1.5),
        ]

    return _fit_generic(droppo_elibol_loss, N, D, L, bounds, sampler, n_restarts)


def _skaling_loss_logcoef(
    N: np.ndarray | float,
    D: np.ndarray | float,
    log_A: float,
    alpha: float,
    log_B: float,
    beta: float,
    k: float,
    E: float,
) -> np.ndarray | float:
    """`skaling_loss` reparameterised with log A and log B, for the optimiser only.

    A and B span ten orders of magnitude while alpha, beta, k sit in [0.01, 2], too far apart for
    L-BFGS-B's finite-difference gradient: from a start 30% off a planted optimum it stalls at
    SSE 1.2 in natural coefficients and reaches 1e-25 in log ones on the same grid. Only the
    optimiser's coordinates change; the surface and `fit_skaling`'s returns stay in natural units.
    """
    return skaling_loss(N, D, np.exp(log_A), alpha, np.exp(log_B), beta, k, E)


def fit_skaling(
    N: np.ndarray, D: np.ndarray, L: np.ndarray, n_restarts: int = 300, seed: int = 42
) -> tuple[np.ndarray | None, float, bool]:
    """Fit the Skaling law with `_fit_generic`'s multi-start + DE on raw SSE, A and B as logs.

    Sharing the other laws' optimiser keeps a comparison unconfounded; the Skaling paper also fits
    logs. Bounds are its Table 4 except k <= 4.0 (at 2.0, 90% of 900 fits pinned k, understating
    the N-D coupling) and A, B, E use `fit_chinchilla`'s ranges, as Table 4 normalises N and D.

    Args:
        seed: exposed (the other fitters hard-code `_fit_generic`'s default) so the seed-variance
            harnesses can delegate here instead of duplicating the log reparameterisation.

    Returns:
        (params, sse, success), params in the natural (A, alpha, B, beta, k, E) order of Eq. 3.
    """
    bounds = [
        (np.log(1e-2), np.log(1e10)),  # log A
        (0.01, 2.0),  # alpha
        (np.log(1e-2), np.log(1e10)),  # log B
        (0.01, 2.0),  # beta
        (0.01, 4.0),  # k  widened from Table 4's 2.0, see above
        (0.0, 10.0),  # E
    ]

    def sampler(rng: np.random.RandomState, L: np.ndarray) -> list[float]:
        return [
            np.log(10 ** rng.uniform(0, 8)),
            rng.uniform(0.1, 1.5),
            np.log(10 ** rng.uniform(0, 8)),
            rng.uniform(0.1, 1.5),
            rng.uniform(0.5, 3.0),  # k starts must reach above the old ceiling
            rng.uniform(0.0, min(L) * 0.9),
        ]

    params, sse, success = _fit_generic(
        _skaling_loss_logcoef,
        N,
        D,
        L,
        bounds,
        sampler,
        n_restarts,
        seed=seed,
    )
    if params is None:
        return None, sse, success

    log_A, alpha, log_B, beta, k, E = params
    return (
        np.array([np.exp(log_A), alpha, np.exp(log_B), beta, k, E]),
        sse,
        success,
    )


# ════════════════════════════════════════════════════════════════════════════
# Evaluation
# ════════════════════════════════════════════════════════════════════════════


def evaluate(
    loss_fn: Callable,
    params: np.ndarray,
    N_train: np.ndarray,
    D_train: np.ndarray,
    L_train: np.ndarray,
    N_hold: np.ndarray,
    D_hold: np.ndarray,
    L_hold: np.ndarray,
) -> dict[str, float]:
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


def effective_data(
    U_D: np.ndarray | float, R: np.ndarray | float, R_star_D: float
) -> np.ndarray | float:
    """D' = U_D + U_D · R*_D · (1 − exp(−R / R*_D)).  R=0 → D'=U_D."""
    return U_D + U_D * R_star_D * (1.0 - np.exp(-R / R_star_D))


def effective_params(
    U_N: np.ndarray | float, R: np.ndarray | float, R_star_N: float
) -> np.ndarray | float:
    """N' = U_N + U_N · R*_N · (1 − exp(−R / R*_N)).  R=0 → N'=U_N."""
    return U_N + U_N * R_star_N * (1.0 - np.exp(-R / R_star_N))


def parametric_epoch_loss(
    N: np.ndarray | float,
    U_D: np.ndarray | float,
    R: np.ndarray | float,
    A: float,
    alpha: float,
    B: float,
    beta: float,
    E: float,
    R_star_D: float,
    R_star_N: float,
) -> np.ndarray | float:
    """
    Full parametric scaling law with epoch half-lives.
    L = A / N'^α  +  B / D'^β  +  E
    """
    D_prime = effective_data(U_D, R, R_star_D)
    N_prime = effective_params(N, R, R_star_N)
    return A / np.power(N_prime, alpha) + B / np.power(D_prime, beta) + E


def _parametric_predict(
    params_dict: dict[str, float],
    N: np.ndarray | float,
    U_D: np.ndarray | float,
    R: np.ndarray | float,
) -> np.ndarray | float:
    """Convenience: predict from a params dict."""
    return parametric_epoch_loss(
        N,
        U_D,
        R,
        params_dict["A"],
        params_dict["alpha"],
        params_dict["B"],
        params_dict["beta"],
        params_dict["E"],
        params_dict["R_star_D"],
        params_dict["R_star_N"],
    )


def fit_parametric_epoch(
    N: np.ndarray,
    U_D: np.ndarray,
    R: np.ndarray,
    L: np.ndarray,
    n_restarts: int = 300,
    seed: int = 42,
) -> tuple[dict[str, float] | None, float, bool]:
    """
    Fit the 7-param epoch law via Huber-loss minimisation.
    Returns (params_dict, loss, success).
    """
    bounds = [
        (1e-2, 1e12),  # A
        (0.01, 2.0),  # alpha
        (1e-2, 1e12),  # B
        (0.01, 2.0),  # beta
        (0.0, 10.0),  # E
        (0.1, 50.0),  # R*_D
        (0.1, 50.0),  # R*_N
    ]
    delta = 0.5  # Huber threshold

    def objective(params: np.ndarray) -> float:
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
                objective,
                x0,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 3000, "ftol": 1e-15},
            )
            if result.fun < best_loss:
                best_loss, best_result = result.fun, result
        except Exception:
            continue

    # Differential evolution fallback
    try:
        de = differential_evolution(
            objective,
            bounds,
            seed=seed,
            maxiter=2000,
            polish=True,
            tol=1e-12,
            mutation=(0.5, 1.5),
            recombination=0.9,
        )
        if de.fun < best_loss:
            best_loss, best_result = de.fun, de
    except Exception:
        pass

    if best_result is None:
        return None, np.inf, False

    A, alpha, B, beta, E_val, R_star_D, R_star_N = best_result.x
    params_dict = {
        "A": A,
        "alpha": alpha,
        "B": B,
        "beta": beta,
        "E": E_val,
        "R_star_D": R_star_D,
        "R_star_N": R_star_N,
    }
    return params_dict, best_loss, best_result.success


def evaluate_parametric_epoch(
    params_dict: dict[str, float],
    N_tr: np.ndarray,
    U_D_tr: np.ndarray,
    R_tr: np.ndarray,
    L_tr: np.ndarray,
    N_ho: np.ndarray,
    U_D_ho: np.ndarray,
    R_ho: np.ndarray,
    L_ho: np.ndarray,
) -> dict[str, float]:
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
        "r2_train": r2_tr,
        "r2_holdout": r2_ho,
        "rmse_train": rmse_tr,
        "rmse_holdout": rmse_ho,
    }


# ════════════════════════════════════════════════════════════════════════════
# Registry - makes looping over the paper's laws easy
# ════════════════════════════════════════════════════════════════════════════


@dataclass
class ScalingLaw:
    name: str
    loss_fn: Callable
    fit_fn: Callable
    param_names: list[str]
    is_epoch_law: bool = False  # True = uses (N, U_D, R) interface


LAWS: list[ScalingLaw] = [
    ScalingLaw("Kaplan", kaplan_loss, fit_kaplan, ["N_c", "D_c", "α_N", "α_D"]),
    ScalingLaw("Chinchilla", chinchilla_loss, fit_chinchilla, ["E", "A", "α", "B", "β"]),
    ScalingLaw(
        "Droppo-Elibol",
        droppo_elibol_loss,
        fit_droppo_elibol,
        ["L_∞", "N_c", "D_c", "α_N", "α_D", "α"],
    ),
    ScalingLaw(
        "Parametric-Epoch",
        parametric_epoch_loss,
        fit_parametric_epoch,
        ["A", "α", "B", "β", "E", "R*_D", "R*_N"],
        is_epoch_law=True,
    ),
]

# Laws that are not part of the paper's analysis. Kept out of LAWS so that the paper's
# tables, figures and the large-scale parity check iterate over exactly the paper's laws.
SKALING = ScalingLaw("Skaling", skaling_loss, fit_skaling, ["A", "α", "B", "β", "k", "E"])
EXTRA_LAWS: list[ScalingLaw] = [SKALING]
