"""Deterministic parallel execution of independent fitting trials.

The determinism contract
------------------------
Parallelism here can never change a result, because randomness and computation
are separated in time:

1. **Plan sequentially.** A single-threaded pass draws every random choice from
   the driver RNG in a fixed order and freezes it into a list of work items.
   Each item carries its own explicit seed. This is the only place the driver
   RNG is touched, so the draw sequence is identical no matter how many workers
   run later.
2. **Execute in parallel.** Each work item is mapped through a pure function of
   that item alone. Workers share no mutable state and never consult a global
   RNG, so the result for item *i* depends only on item *i*.
3. **Reduce in plan order.** Results come back indexed and are sorted back into
   plan order before aggregation, so no order-sensitive statistic (median, win
   count, bootstrap) can see a different sequence.

Each worker also runs with the BLAS pinned to one thread (see the package
docstring), so a worker's arithmetic is bit-identical to the serial path.

The practical consequence: ``workers=1`` and ``workers=256`` produce identical
output, and the test suite asserts exactly that.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor
from typing import Optional, TypeVar

Work = TypeVar("Work")
Outcome = TypeVar("Outcome")

#: Leave a couple of cores for the parent process and the OS.
_RESERVED_CORES = 2

#: Below this many items, process startup costs more than it saves.
MIN_ITEMS_FOR_POOL = 8


def default_worker_count() -> int:
    """Worker count to use when the caller does not specify one.

    Uses the physical CPU allowance visible to this process (respecting cgroup
    /affinity limits) minus a small reserve.
    """
    try:
        available = len(os.sched_getaffinity(0))
    except AttributeError:  # pragma: no cover - non-Linux
        available = os.cpu_count() or 1
    return max(1, available - _RESERVED_CORES)


def resolve_worker_count(requested: Optional[int], item_count: int) -> int:
    """Clamp a requested worker count to something sensible for this workload."""
    if requested is not None and requested < 1:
        raise ValueError(f"workers must be >= 1, got {requested}")
    workers = default_worker_count() if requested is None else requested
    if item_count < MIN_ITEMS_FOR_POOL:
        return 1
    return max(1, min(workers, item_count))


def _init_worker() -> None:
    """Re-assert the BLAS thread cap inside a freshly spawned worker."""
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = "1"


class DeterministicExecutor:
    """Maps a pure function over pre-planned work items, preserving order.

    Args:
        workers: Requested worker processes. ``None`` picks a default from the
            machine size; ``1`` forces the serial path.

    The mapped function must be a module-level callable (so it can be pickled)
    and must be a pure function of its argument.
    """

    def __init__(self, workers: Optional[int] = None) -> None:
        self._requested = workers

    def map(
        self,
        function: Callable[[Work], Outcome],
        items: Sequence[Work],
    ) -> list[Outcome]:
        """Apply ``function`` to every item, returning results in item order."""
        if not items:
            return []
        workers = resolve_worker_count(self._requested, len(items))
        if workers == 1:
            return [function(item) for item in items]
        chunk = max(1, len(items) // (workers * 4))
        with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as pool:
            # Executor.map preserves input order regardless of completion order.
            return list(pool.map(function, items, chunksize=chunk))

    def describe(self, item_count: int) -> str:
        workers = resolve_worker_count(self._requested, item_count)
        return f"{item_count} items on {workers} worker(s)"


def drop_failures(outcomes: Iterable[Optional[Outcome]]) -> list[Outcome]:
    """Filter out trials that could not be evaluated, preserving order."""
    return [o for o in outcomes if o is not None]
