#!/usr/bin/env python3
"""The regime test with each pair classified by its collinear design alone.

Theorem 1 and the paper's Regime-A table classify a comparison by its CO design: Regime A
iff m_CO + log10(kappa_target / 100) < 0, a degenerate CO design (V_K = 0) included. The
protocol's own test (doseresponse_seeds512.csv, regime_by_band_ci.py) classifies both designs;
this script applies the CO-only rule, through the same `conditional.regimes`, at every
kappa_target of the protocol grid and for both corpora. It reports the NC win rate and the
median off-ray gap (g > 0 favours NC) with both 95% intervals of `estimate._median_job`;
`lo_cl`/`hi_cl`, over (cell, window) clusters within cell, are the ones the paper prints, and
`excludes_zero` refers to them.

Reads ``results_E_seeds512.csv`` and ``results_X_seeds512.csv.gz`` (enumerate_seeds.py;
``--skip-x`` without the X corpus) and writes ``co_only_regime.csv``. Relative paths are taken
in the work directory.

    uv run python intervals/co_only_regime.py --workers "$(nproc)"
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the reanalysis modules
import conditional as C
import estimate as EST
from regime_by_band_ci import GAP, regime_cli


def co_jobs(df: pd.DataFrame, corpus: str) -> list[tuple]:
    """One job per (kappa_target, regime of the CO design) that holds any pair."""
    jobs = []
    for kappa in C.KAPPA_GRID:
        cells = C.regimes(df, kappa, "NC")
        for regime in ("A", "B"):
            sub = df[cells.str.startswith(f"CO={regime},")]
            if len(sub):
                jobs.append(EST.median_task((corpus, kappa, regime), sub, GAP,
                                            f"coonly|{corpus}|{kappa}|{regime}"))
    return jobs


def main(argv: list[str] | None = None) -> None:
    regime_cli(argv, __doc__, "co_only_regime.csv", co_jobs, ["corpus", "kappa", "co_regime"])


if __name__ == "__main__":
    main()
