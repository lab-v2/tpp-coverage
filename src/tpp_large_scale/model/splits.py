"""Train/holdout split, bucketed by the paper's own ``organise_runs``.

The paper defines the split as four buckets keyed on two per-run flags,
``is_colinear`` and ``is_holdout`` (``scaling_laws.data._organise``, re-exported
as :func:`tpp_large_scale.upstream.organise_runs`). We reuse that bucketer
rather than re-implementing it: each envelope cell is turned into a run dict in
the paper's schema, tagged with the two flags per this grid's geometry, and
handed to the package.

Geometry (paper Appendix D.2): the holdout is the extrapolation frontier, held
out *beyond* the training range, not an interior subset. On this grid that is
the largest model size (354M non-embedding); the collinear design is the single
clean TPP ray.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..config import CLEAN_RAY_BAND
from .grid import TrainingGrid
from ..upstream import organise_runs

#: Bucket names the package's ``organise_runs`` emits.
COLINEAR_TRAIN = "colinear_train"
COLINEAR_HOLDOUT = "colinear_holdout"
NONCOLINEAR_TRAIN = "noncolinear_train"
NONCOLINEAR_HOLDOUT = "noncolinear_holdout"


def _to_run(row: pd.Series, *, is_colinear: bool, is_holdout: bool) -> dict:
    """One envelope cell as a run dict in the paper's schema."""
    return {
        "N": float(row["N"]),
        "D": float(row["Dt"]),
        "n_epochs": 1,
        "epoch_losses": [float(row["L"])],
        "is_colinear": is_colinear,
        "is_holdout": is_holdout,
    }


@dataclass(frozen=True)
class FrontierSplit:
    """The envelope bucketed into the paper's four train/holdout categories.

    Attributes hold the raw envelope rows per bucket; ``buckets`` is the
    package's own dict-of-run-dicts, kept so the split provenance is visible.
    """

    colinear_train: pd.DataFrame
    colinear_holdout: pd.DataFrame
    noncolinear_train: pd.DataFrame
    noncolinear_holdout: pd.DataFrame
    buckets: dict[str, list[dict]]

    @property
    def train(self) -> pd.DataFrame:
        return pd.concat([self.noncolinear_train, self.colinear_train])

    @property
    def holdout(self) -> pd.DataFrame:
        return pd.concat([self.noncolinear_holdout, self.colinear_holdout])

    def counts(self) -> dict[str, int]:
        return {name: len(runs) for name, runs in self.buckets.items()}


def frontier_split(grid: TrainingGrid, ray_band: float = CLEAN_RAY_BAND) -> FrontierSplit:
    """Split the envelope into the paper's four buckets via ``organise_runs``.

    ``is_holdout`` marks the largest-model extrapolation frontier; ``is_colinear``
    marks the clean single-TPP ray. Bucketing itself is the package's.
    """
    envelope = grid.envelope
    frontier_n = envelope["N"].max()
    is_holdout = (envelope["N"] == frontier_n).to_numpy()
    is_colinear = (envelope["band"] == ray_band).to_numpy()

    runs = [
        _to_run(row, is_colinear=bool(c), is_holdout=bool(h))
        for (_, row), c, h in zip(envelope.iterrows(), is_colinear, is_holdout)
    ]
    buckets = organise_runs(runs)

    def rows(want_colinear: bool, want_holdout: bool) -> pd.DataFrame:
        mask = (is_colinear == want_colinear) & (is_holdout == want_holdout)
        return envelope[mask]

    return FrontierSplit(
        colinear_train=rows(True, False),
        colinear_holdout=rows(True, True),
        noncolinear_train=rows(False, False),
        noncolinear_holdout=rows(False, True),
        buckets=buckets,
    )
