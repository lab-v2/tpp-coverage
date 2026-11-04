#!/usr/bin/env python3
"""Run the large-scale analysis (the tpp_large_scale command line) from a checkout.

    uv run python large_scale/scripts/run_analysis.py --help
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tpp_large_scale.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
