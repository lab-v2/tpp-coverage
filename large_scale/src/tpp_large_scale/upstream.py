"""Single import choke point for the companion paper's own analysis code.

Everything this project borrows from the paper's code package (`tpp_analysis`)
is re-exported here and nowhere else, so the question "what did we reuse and
what did we write ourselves?" is answered by reading one module.

Why a path bootstrap instead of a package import
-------------------------------------------------
The paper's analysis code is ``src/tpp_analysis`` in this repository, and its modules
import each other by bare top-level name (``from scaling_laws.laws import ...``). The
directory that must go on ``sys.path`` is therefore ``<repo>/src/tpp_analysis`` itself,
not ``<repo>/src``.

Set ``TPP_CODE_REPO`` to another checkout of this repository to run the large-scale
analysis against that copy of the paper's code instead.

Deliberately NOT imported
-------------------------
``full_coverage_ci``
    Seed-variance confidence intervals, which this analysis does not use.
``incremental_coverage.coverage.snap_to_nearest``
    Snaps on **linear** distance (``min(bins, key=lambda b: abs(b - value))``).
    Our TPP bands are quarter-decade spaced, so the correct nearest-bin rule is
    **log** distance; the two disagree for values near a band midpoint (TPP 90
    snaps to 67 linearly but to 120 in log space). See
    :class:`tpp_large_scale.model.grid.LogBandAssigner`.

Both are recorded in ``docs/upstream_notes.md``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

CODE_REPO_ENV = "TPP_CODE_REPO"
PACKAGE_RELPATH = Path("src") / "tpp_analysis"
_SENTINEL_RELPATH = Path("scaling_laws") / "laws.py"


class UpstreamNotFound(RuntimeError):
    """The paper's code checkout could not be located or is not usable."""


def _candidate_roots() -> list[Path]:
    """Return the repository whose ``src/tpp_analysis`` is the paper's code."""
    override = os.environ.get(CODE_REPO_ENV)
    if override:
        return [Path(override).expanduser().resolve()]
    # <repo>/large_scale/src/tpp_large_scale/upstream.py -> parents[3] is the repository root.
    return [Path(__file__).resolve().parents[3]]


def locate_upstream_package() -> Path:
    """Return the directory to place on ``sys.path``: ``<code repo>/src/tpp_analysis``.

    Raises:
        UpstreamNotFound: with an actionable message if no usable checkout exists.
    """
    roots = _candidate_roots()
    if not roots:
        raise UpstreamNotFound(
            f"Could not find the paper's code repository; set {CODE_REPO_ENV} to a checkout."
        )
    for root in roots:
        package_dir = root / PACKAGE_RELPATH
        if (package_dir / _SENTINEL_RELPATH).is_file():
            return package_dir
    tried = ", ".join(str(r) for r in roots)
    raise UpstreamNotFound(
        f"Found candidate checkout(s) [{tried}] but none contained "
        f"{PACKAGE_RELPATH / _SENTINEL_RELPATH}. Set {CODE_REPO_ENV} to a checkout of this "
        "repository."
    )


def _bootstrap() -> Path:
    package_dir = locate_upstream_package()
    entry = str(package_dir)
    if entry not in sys.path:
        sys.path.insert(0, entry)
    return package_dir


UPSTREAM_PACKAGE_DIR = _bootstrap()


def package_dir_label(package_dir: Path = UPSTREAM_PACKAGE_DIR) -> str:
    """The package directory as recorded in results: relative to this repository when it is
    this repository's own copy, so a committed results file names no local path."""
    repo_root = Path(__file__).resolve().parents[3]
    try:
        return package_dir.resolve().relative_to(repo_root).as_posix()
    except ValueError:
        return str(package_dir)


# ruff: noqa: E402  -- these imports require the sys.path bootstrap above.
from scaling_laws.data import PreparedData
from scaling_laws.data import _organise as organise_runs
from scaling_laws.laws import (
    LAWS,
    ScalingLaw,
    _fit_generic,
    chinchilla_loss,
    droppo_elibol_loss,
    evaluate,
    fit_chinchilla,
    fit_droppo_elibol,
    fit_kaplan,
    kaplan_loss,
)
from subset_analysis.subset_analysis_bbox_seeded import (
    BETA_EFF_INDEX,
    FIT_SEED_BASE,
    KAPPA_TARGETS,
    MIN_POINTS_FOR_FIT,
    N_RESTARTS,
    classify_regime,
    compute_tauK,
    compute_VK,
)

#: Multi-start L-BFGS-B + differential-evolution fitter used by the paper.
#: Re-exported under a public name because it is upstream's only seed-parameterised
#: entry point; the leading underscore is upstream's, not a signal of privacy here.
fit_generic = _fit_generic

__all__ = [
    "BETA_EFF_INDEX",
    "CODE_REPO_ENV",
    "FIT_SEED_BASE",
    "KAPPA_TARGETS",
    "LAWS",
    "MIN_POINTS_FOR_FIT",
    "N_RESTARTS",
    "PreparedData",
    "UPSTREAM_PACKAGE_DIR",
    "package_dir_label",
    "ScalingLaw",
    "organise_runs",
    "UpstreamNotFound",
    "chinchilla_loss",
    "classify_regime",
    "compute_VK",
    "compute_tauK",
    "droppo_elibol_loss",
    "evaluate",
    "fit_chinchilla",
    "fit_droppo_elibol",
    "fit_generic",
    "fit_kaplan",
    "kaplan_loss",
    "locate_upstream_package",
]
