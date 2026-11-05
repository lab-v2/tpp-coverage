"""Large-scale validation of TPP-coverage scaling-law conditioning.

Reruns the companion paper's estimation-conditioning pipeline on an independent,
larger training grid, reusing the paper's own law definitions, fitters and
diversity criterion via :mod:`tpp_large_scale.upstream`.

Determinism note
----------------
The BLAS thread caps below are set before anything imports numpy, and that
ordering is load-bearing. A multi-threaded BLAS splits reductions across threads
and sums the partial results in completion order, so the same dot product can
differ in its last bits between runs. On a near-singular Gauss-Newton system
those last bits are amplified into visibly different condition numbers, and
fitted exponents on a degenerate design can land in a different local optimum
entirely. Pinning the BLAS to one thread per process makes every fit
bit-reproducible; parallelism is recovered across processes instead, where each
worker is an independent deterministic unit. This mirrors what the paper's own
``subset_analysis`` module does.
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
