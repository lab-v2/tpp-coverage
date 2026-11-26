"""Characterization of `subset_analysis`: the seeded bbox subset fits and both seed aggregators.

`subset_analysis_bbox_seeded.py` runs on the released C4 loss table (subsets of up to 2 CO TPP
levels, 1 restart). The aggregators read the released per-seed subset results of seeds 0 and 1
for C4 and Wikipedia. Expected values are the base commit's outputs, frozen in `frozen/` (see
`tests/characterize.py`); figures are compared by their data, and saving them is recorded, not
rendered.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

from characterize import FigureCapture, check_frozen, released
from conftest import ANALYSIS, pool_width, run

pytestmark = pytest.mark.network
FROZEN = Path(__file__).parent / "frozen"
SUBSET = ANALYSIS / "subset_analysis"
SEEDS, DATASETS = (0, 1), ("C4", "Wikipedia")


@pytest.fixture(scope="module")
def seed_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A seed directory (seed_0/, seed_1/) linking the released per-seed subset results."""
    patterns = [f"subset_bbox_seeds/seed_{s}/subset_bbox_{d}.csv" for s in SEEDS for d in DATASETS]
    root = released(*patterns) / "subset_bbox_seeds"
    out = tmp_path_factory.mktemp("subset_bbox_seeds")
    for s in SEEDS:
        (out / f"seed_{s}").mkdir()
        for d in DATASETS:
            (out / f"seed_{s}" / f"subset_bbox_{d}.csv").symlink_to(
                root / f"seed_{s}/subset_bbox_{d}.csv"
            )
    return out


def _main(module, argv: list[str], monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", [module.__file__, *map(str, argv)])
    module.main()


def _csvs(out: Path) -> dict[str, pd.DataFrame]:
    return {p.name: pd.read_csv(p) for p in sorted(out.glob("*.csv"))}


def test_aggregate_seed_runs(
    seed_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from matplotlib.figure import Figure

    from subset_analysis import aggregate_seed_runs as agg

    saved: list[str] = []
    monkeypatch.setattr(Figure, "savefig", lambda self, path, **kw: saved.append(Path(path).name))
    with FigureCapture() as cap:
        _main(agg, ["--seed-dir", seed_dir, "--out-dir", tmp_path, "--n-points", 12], monkeypatch)
    check_frozen(
        FROZEN / "aggregate_seed_runs.json",
        {"csv": _csvs(tmp_path), "saved": saved, "figures": cap.figures},
    )


def test_aggregate_helpers() -> None:
    from subset_analysis import aggregate_seed_runs as agg

    check_frozen(
        FROZEN / "aggregate_helpers.json",
        {
            "tpps": [
                agg.parse_tpp_list(s)
                for s in ("[1.5, 3.0]", "[np.float64(2.0), 4e1]", "", "[]", None)
            ],
            "ci_one": agg.bootstrap_ci([42.0]),
            "ci_empty": agg.bootstrap_ci([float("nan")]),
            "ci": agg.bootstrap_ci([10.0, 20.0, 35.0, 50.0]),
        },
    )


def test_aggregate_released_seeds() -> None:
    """The bootstrap aggregation over all released seeds (the per-seed sweep by law)."""
    from subset_analysis import aggregate_seed_runs as agg

    root = released("subset_bbox_seeds/aggregated/per_seed_sweep_by_law.csv")
    per_seed = pd.read_csv(root / "subset_bbox_seeds/aggregated/per_seed_sweep_by_law.csv")
    kts = per_seed["kappa_target"].unique()[::10]
    per_seed = per_seed[per_seed["kappa_target"].isin(kts)]
    check_frozen(
        FROZEN / "aggregate_released_seeds.json",
        {
            "all": agg.aggregate(per_seed, ["law", "mode"], kts),
            "masked": agg.aggregate(per_seed, ["law", "mode"], kts, min_cell_n=agg.MIN_CELL_N),
        },
    )


def test_per_law_winrate_ci(
    seed_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from subset_analysis import per_law_winrate_ci as pl

    for mode in ("first", "final"):
        out = tmp_path / mode
        _main(
            pl,
            ["--seed-dir", seed_dir, "--out-dir", out, "--mode", mode, "--n-workers", 2],
            monkeypatch,
        )
        check_frozen(FROZEN / f"per_law_winrate_ci_{mode}.json", _csvs(out))


def test_subset_analysis_bbox_seeded(tmp_path: Path) -> None:
    losses = released("extracted_losses/c4.csv") / "extracted_losses"
    args = [
        "--data-dir",
        losses,
        "--out-dir",
        tmp_path,
        "--datasets",
        "C4",
        "--laws",
        "Kaplan",
        "Chinchilla",
    ]
    args += ["--modes", "first", "final", "--n-restarts", "1", "--max-subset-size", "2"]
    script = SUBSET / "subset_analysis_bbox_seeded.py"
    for seed in (0, 3):
        run(["python", script, "--seed", str(seed), *args, "--n-workers", str(pool_width(64))])
    # A rerun finds every combo cached and leaves the results as they are.
    before = (tmp_path / "seed_0/subset_bbox_C4.csv").read_bytes()
    assert "All cached!" in run(["python", script, "--seed", "0", *args, "--n-workers", "2"]).stdout
    assert (tmp_path / "seed_0/subset_bbox_C4.csv").read_bytes() == before
    keys = ["law", "mode", "subset_mask", "design"]
    rows = {
        seed: pd.read_csv(tmp_path / f"seed_{seed}/subset_bbox_C4.csv")
        .sort_values(keys)
        .reset_index(drop=True)
        for seed in (0, 3)
    }
    check_frozen(FROZEN / "subset_analysis_bbox_seeded.json", rows)
