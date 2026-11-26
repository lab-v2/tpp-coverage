"""Characterization of `get_results/get_results.py`: loss tables extracted from released metrics.

Inputs are the released training metrics (`.npz`) of every run with N = 10,070,160, for all
seven datasets. Both modes run: `from_hf` on the released directory layout and `from_local` on
a flat directory of the same files. Expected values are the base commit's CSVs, frozen in
`frozen/` (see `tests/characterize.py`); each extracted run block must also appear verbatim in
the released loss table of its dataset.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from characterize import check_frozen, released
from conftest import ANALYSIS, run

pytestmark = pytest.mark.network
FROZEN = Path(__file__).parent / "frozen"
SCRIPT = ANALYSIS / "get_results/get_results.py"
TIERS = ["--include-special", "--include-prelim"]


@pytest.fixture(scope="module")
def metrics() -> Path:
    return released("training_metrics/*/*/N_10070160/*_metrics.npz", "extracted_losses/*.csv")


def _tables(out: Path) -> dict[str, str]:
    return {p.name: p.read_text() for p in sorted(out.glob("*.csv"))}


def _blocks(text: str) -> list[str]:
    return [b.strip() for b in text.split("\n\n") if b.strip()]


def test_from_hf(metrics: Path, tmp_path: Path) -> None:
    run(["python", SCRIPT, "from_hf", metrics / "training_metrics", "-o", tmp_path, *TIERS])
    tables = _tables(tmp_path)
    assert len(tables) == 7
    for name, text in tables.items():
        whole = set(_blocks((metrics / "extracted_losses" / name).read_text()))
        assert set(_blocks(text)) <= whole, name
    check_frozen(FROZEN / "from_hf.json", tables)


@pytest.mark.parametrize(
    "flags", [[], TIERS, [*TIERS, "--no-final-loss"]], ids=["main", "tiers", "epoch-avg"]
)
def test_from_local(metrics: Path, tmp_path: Path, flags: list[str]) -> None:
    flat = tmp_path / "flat"
    flat.mkdir()
    for f in (metrics / "training_metrics").rglob("*_metrics.npz"):
        (flat / f.name).symlink_to(f)
    out = tmp_path / "out"
    run(["python", SCRIPT, "from_local", flat, "-o", out, *flags])
    check_frozen(
        FROZEN / f"from_local_{'_'.join(f.strip('-') for f in flags) or 'main'}.json", _tables(out)
    )
