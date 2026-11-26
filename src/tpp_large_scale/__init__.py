"""Large-scale validation of TPP-coverage scaling-law conditioning.

Reruns the companion paper's estimation-conditioning pipeline on an independent,
larger training grid, reusing the paper's own law definitions, fitters and
diversity criterion via :mod:`tpp_large_scale.upstream`.

Determinism: the BLAS thread caps below are set before anything imports numpy, and that
order is load-bearing. A multi-threaded BLAS sums partial reductions in completion order,
so last bits differ between runs; a near-singular Gauss-Newton system amplifies them into
different condition numbers or even a different local optimum. One BLAS thread per
process makes every fit bit-reproducible, with parallelism across processes instead, as
the paper's own ``subset_analysis`` module does.
"""

from __future__ import annotations

import os

#: Environment variables that cap BLAS/OpenMP threading, applied at import time.
_THREAD_CAP_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)

for _var in _THREAD_CAP_VARS:
    os.environ.setdefault(_var, "1")

__version__ = "1.0.0"
