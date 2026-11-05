"""Paths shared by the WIKI300M audit fits: the paper's code and the run directory.

Importing this module puts this repository's ``src/tpp_analysis`` and ``src`` on ``sys.path``
(``TPP_CODE_REPO`` selects another checkout) and reads the run directory from
``WIKI300M_DIR``: the directory that holds ``shared/training_metrics/`` of the WIKI300M runs.
Every fit writes its CSV there.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(os.environ.get("TPP_CODE_REPO") or Path(__file__).resolve().parents[3])
for _p in (REPO / "src" / "tpp_analysis", REPO / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

RUN_DIR = os.environ.get("WIKI300M_DIR", "")
if not RUN_DIR and __name__ != "__main__":
    sys.exit("set WIKI300M_DIR to the WIKI300M run directory (it holds shared/training_metrics/)")
