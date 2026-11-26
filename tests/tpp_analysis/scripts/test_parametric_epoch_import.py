"""A missing Parametric-Epoch law fails the run instead of being skipped.

`_run_full_coverage_combo` in run_analysis.py and run_special_analysis.py used to catch the
ImportError of `scaling_laws.laws.parametric_epoch_loss` / `fit_parametric_epoch`, print a
[SKIP] line and return no rows, so a broken install silently dropped a law from the tables.
With the law removed from the module, the worker must now raise.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

import pytest

from characterize import released

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.network


@pytest.mark.parametrize("module", ["run_analysis", "run_special_analysis"])
def test_missing_parametric_epoch_law_raises(
    module: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import scaling_laws.laws as laws

    csv = released("extracted_losses/wikipedia.csv") / "extracted_losses" / "wikipedia.csv"
    monkeypatch.delattr(laws, "parametric_epoch_loss")
    worker = importlib.import_module(module)._run_full_coverage_combo
    args = ("Wikipedia", str(csv), "Parametric-Epoch", "Epoch-Expanded", 1, 1, 0, str(tmp_path))
    with pytest.raises(ImportError, match="parametric_epoch_loss"):
        worker(args)
