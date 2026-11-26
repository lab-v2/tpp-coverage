"""Characterization of `utility/`: the checkpoint architecture audit and the n_layers histogram.

The audit reads two released checkpoints (the smallest model, epochs 2 and 3, about 61 MB each)
laid out as on HuggingFace; the histogram plots the paper's 14 architectures and the audit's
checkpoints. Expected values are the base commit's outputs, frozen in `frozen/` (see
`tests/characterize.py`); saving the figure is recorded, not rendered.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

from characterize import FigureCapture, check_frozen, released
from conftest import ANALYSIS, run

pytestmark = [pytest.mark.network, pytest.mark.slow]
FROZEN = Path(__file__).parent / "frozen"
RUN = "wikipedia/non_colinear/N_5035656/WIKI_NxD_GRID_N5M_D10M_wikipedia_m5035656_token10000000"


@pytest.fixture(scope="module")
def audit(tmp_path_factory) -> Path:
    """The audit CSV of the two checkpoints, written by the base script."""
    src = released(f"{RUN}_*_epoch[23]_*.pt")
    root = tmp_path_factory.mktemp("checkpoints")
    for f in src.glob(f"{RUN}_*.pt"):
        dest = root / f.relative_to(src)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.symlink_to(f)
    out = root.parent / "audit.csv"
    run(["python", ANALYSIS / "utility/audit_architectures.py", root, "--out", out])
    only3 = root.parent / "audit_e3.csv"
    run(
        [
            "python",
            ANALYSIS / "utility/audit_architectures.py",
            root,
            "--out",
            only3,
            "--only-epoch",
            "3",
        ]
    )
    return out


def _rows(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return sorted(
            ({k: v for k, v in r.items() if k != "path"} for r in csv.DictReader(f)), key=str
        )


def test_audit(audit):
    check_frozen(
        FROZEN / "audit_architectures.json",
        {"all": _rows(audit), "epoch3": _rows(audit.parent / "audit_e3.csv")},
    )


def test_nlayers_histogram(audit, tmp_path, monkeypatch):
    """The unique-architecture histogram. Base behaviour pinned as found: `counts_runs` reads an
    `actual_n_layers` column, but the audit writes `n_layers`, so `--mode runs` on the audit's
    own CSV counts nothing and `plot` fails on the empty counts."""
    from matplotlib.figure import Figure

    sys.path.insert(0, str(ANALYSIS / "utility"))
    import plot_nlayers_distribution as pn

    saved: list[str] = []
    monkeypatch.setattr(Figure, "savefig", lambda self, path, **kw: saved.append(Path(path).name))
    with FigureCapture() as cap:
        monkeypatch.setattr(sys, "argv", ["plot", "--mode", "unique", "--out", str(tmp_path / "u")])
        pn.main()
    assert pn.counts_runs(audit) == {}
    monkeypatch.setattr(sys, "argv", ["plot", "--mode", "runs", "--audit-csv", str(audit)])
    with pytest.raises(ValueError, match="empty"):
        pn.main()
    check_frozen(
        FROZEN / "nlayers_histogram.json",
        {"counts": dict(pn.counts_unique()), "saved": saved, "figures": cap.figures},
    )
