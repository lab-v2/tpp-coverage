"""Characterization helpers: freeze what the code does today, and fail when it changes.

The tests under `tests/tpp_analysis/<component>/test_characterize_*.py` run the analysis code on
the released inputs (the loss tables, per-seed results and metrics on HuggingFace, pinned to one
revision) and compare what it returns, writes and draws with values frozen from the base commit
(`frozen/*.json` beside each test). They pin behaviour before refactors, not correctness: the
paper gate and the other tests check the numbers against the paper.

Refreezing is deliberate: `TPP_FREEZE=1 uv run pytest <test>` rewrites the frozen files from the
code as it now stands, so run it only on the base commit, never to make a failing test pass.
Figures are compared by their data (line and scatter coordinates, collection vertices and
colours, zorder and alpha of every line and collection, bar geometry, labels, limits, legend
entries), never by image bytes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import json
import math
import os
from pathlib import Path
from collections.abc import Iterable

import numpy as np
import pandas as pd
import pytest

if TYPE_CHECKING:
    from matplotlib.collections import Collection
    from matplotlib.figure import Figure

# The released dataset at the revision every frozen value was taken from.
HF_REVISION = "1b4cdafcbfa4cbf58d96876a06677724a6f0dac4"
FREEZE = os.environ.get("TPP_FREEZE") == "1"
RTOL = 1e-6  # optimiser fits agree to about 1e-9 on one host; 1e-6 leaves room across hosts
ATOL = 1e-9
FULL_ARRAY_MAX = 64  # arrays up to this many elements are frozen whole, larger ones summarised


def released(*patterns: str) -> Path:
    """Local snapshot root holding the released files matching `patterns` (cached by the hub)."""
    from huggingface_hub import snapshot_download

    from hf_dataset import REPO_ID

    root = snapshot_download(
        REPO_ID, repo_type="dataset", revision=HF_REVISION, allow_patterns=list(patterns)
    )
    return Path(root)


def summarize(values: object) -> list[float] | list[str] | dict[str, object]:
    """A float array as its values when small; else its size, NaN count, sums and extremes.

    `wsum` weights element i by cos(i), so a reordering or a change in any one element moves it.
    """
    try:
        arr = np.asarray(values, dtype=float)
    except (TypeError, ValueError):
        return [str(v) for v in np.ravel(np.asarray(values, dtype=object))]
    flat = arr.ravel()
    if flat.size <= FULL_ARRAY_MAX:
        return flat.tolist()
    ok = np.isfinite(flat)
    finite = flat[ok]
    return {
        "shape": list(arr.shape),
        "n_nan": int((~ok).sum()),
        "sum": float(finite.sum()),
        "sq_sum": float((finite**2).sum()),
        "wsum": float((finite * np.cos(np.flatnonzero(ok))).sum()),
        "min": float(finite.min()) if finite.size else None,
        "max": float(finite.max()) if finite.size else None,
    }


def observe(obj: object) -> object:
    """A JSON-friendly view of `obj`: frames by column, arrays summarised, numbers as floats."""
    if isinstance(obj, pd.DataFrame):
        return {
            "columns": [str(c) for c in obj.columns],
            "n_rows": len(obj),
            "data": {str(c): observe_column(obj[c]) for c in obj.columns},
        }
    if isinstance(obj, pd.Series):
        return observe_column(obj)
    if isinstance(obj, dict):
        return {str(k): observe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        if obj and all(
            isinstance(v, (int, float, np.number)) and not isinstance(v, bool) for v in obj
        ):
            return summarize(obj)
        return [observe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return summarize(obj) if obj.dtype != object else [observe(v) for v in obj]
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, float, np.number)):
        return float(obj)
    if obj is None or isinstance(obj, str):
        return obj
    return str(obj)


def observe_column(col: pd.Series) -> object:
    """A numeric column summarised as an array; any other column as its values."""
    if pd.api.types.is_bool_dtype(col):
        return summarize(col.astype(float).to_numpy())
    if pd.api.types.is_numeric_dtype(col):
        return summarize(col.to_numpy(dtype=float))
    return [observe(v) for v in col.tolist()]


def assert_matches(observed: object, frozen: object, where: str = "") -> None:
    """Equal structure and strings; numbers equal to RTOL (NaN equals NaN)."""
    if isinstance(frozen, dict):
        assert isinstance(observed, dict), f"{where}: expected a mapping, got {observed!r:.200}"
        assert sorted(observed) == sorted(frozen), (
            f"{where}: keys {sorted(observed)} != {sorted(frozen)}"
        )
        for k in frozen:
            assert_matches(observed[k], frozen[k], f"{where}.{k}")
    elif isinstance(frozen, list):
        assert isinstance(observed, list), f"{where}: expected a list, got {observed!r:.200}"
        assert len(observed) == len(frozen), f"{where}: length {len(observed)} != {len(frozen)}"
        for i, (o, f) in enumerate(zip(observed, frozen)):
            assert_matches(o, f, f"{where}[{i}]")
    elif isinstance(frozen, float) and not isinstance(observed, bool):
        assert isinstance(observed, (int, float)), f"{where}: expected a number, got {observed!r}"
        if math.isnan(frozen):
            assert math.isnan(observed), f"{where}: {observed} != nan"
        else:
            assert math.isclose(observed, frozen, rel_tol=RTOL, abs_tol=ATOL), (
                f"{where}: {observed!r} != frozen {frozen!r}"
            )
    elif isinstance(frozen, str) and isinstance(observed, str) and observed != frozen:
        i = next((k for k, (a, b) in enumerate(zip(observed, frozen)) if a != b), None)
        i = min(len(observed), len(frozen)) if i is None else i
        pytest.fail(
            f"{where}: text differs at character {i}:\n"
            f"  now:    {observed[max(0, i - 80) : i + 80]!r}\n"
            f"  frozen: {frozen[max(0, i - 80) : i + 80]!r}"
        )
    else:
        assert observed == frozen, f"{where}: {observed!r:.300} != frozen {frozen!r:.300}"


def _rounded(view: object) -> object:
    """Floats to 12 significant digits (far inside RTOL), to keep the frozen files small."""
    if isinstance(view, dict):
        return {k: _rounded(v) for k, v in view.items()}
    if isinstance(view, list):
        return [_rounded(v) for v in view]
    if isinstance(view, float) and math.isfinite(view):
        return float(f"{view:.12g}")
    return view


def check_frozen(path: Path, observed: object) -> None:
    """Compare `observed` with the frozen file at `path`; with TPP_FREEZE=1, write it instead."""
    view = json.loads(json.dumps(observe(observed)))
    if FREEZE:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_rounded(view), sort_keys=True, separators=(",", ":")) + "\n")
        return
    if not path.exists():
        pytest.fail(f"{path} is missing: freeze it on the base commit with TPP_FREEZE=1")
    assert_matches(view, json.loads(path.read_text()), path.stem)


# ── Figure data ──────────────────────────────────────────────────────────


def _color(c: object) -> object:
    from matplotlib.colors import to_hex

    try:
        return to_hex(c, keep_alpha=True)
    except (TypeError, ValueError):
        return summarize(c)


def _lines(lines: list[object]) -> dict[str, object]:
    """All lines of an axes: per line its size, colour, style, zorder, alpha and label; data pooled."""
    data = [ln.get_data_3d() if hasattr(ln, "get_data_3d") else ln.get_data() for ln in lines]
    out: dict[str, object] = {
        "points": [len(d[0]) for d in data],
        "colors": [_color(ln.get_color()) for ln in lines],
        "styles": [f"{ln.get_linestyle()}|{ln.get_marker()}|{ln.get_linewidth()}" for ln in lines],
        "zorders": [ln.get_zorder() for ln in lines],
        "alphas": [ln.get_alpha() for ln in lines],
        "labels": [ln.get_label() for ln in lines if not ln.get_label().startswith("_")],
    }
    for axis in range(max((len(d) for d in data), default=0)):
        out["xyz"[axis]] = summarize(_pooled([d[axis] for d in data]))
    return out


def _pooled(arrays: Iterable[object]) -> np.ndarray:
    try:
        return np.concatenate([np.asarray(a, dtype=float).ravel() for a in arrays])
    except (TypeError, ValueError):
        return np.asarray([str(v) for a in arrays for v in np.ravel(a)], dtype=object)


def _collection(coll: Collection) -> dict[str, object]:
    out: dict[str, object] = {"kind": type(coll).__name__}
    if hasattr(coll, "_offsets3d"):
        out["xyz"] = [summarize(np.asarray(v, dtype=float)) for v in coll._offsets3d]
    elif getattr(coll, "_vec", None) is not None:
        out["vec"] = summarize(coll._vec)
    elif getattr(coll, "_segments3d", None) is not None:
        out["segments"] = summarize(_pooled(coll._segments3d))
    else:
        out["offsets"] = summarize(coll.get_offsets())
        paths = coll.get_paths()
        if paths:
            out["vertices"] = summarize(_pooled([p.vertices for p in paths]))
    arr = coll.get_array()
    if arr is not None:
        out["array"] = summarize(np.ma.filled(np.ma.asarray(arr, dtype=float), np.nan))
    out["facecolor"] = summarize(coll.get_facecolor())
    out["zorder"] = coll.get_zorder()
    out["alpha"] = coll.get_alpha()
    out["label"] = coll.get_label() if not coll.get_label().startswith("_") else ""
    return out


def _patches(patches: list[object]) -> dict[str, object]:
    """All patches of an axes: kinds and colours per patch, bar and polygon geometry pooled."""
    from matplotlib.patches import Polygon, Rectangle

    geoms = []
    for p in patches:
        if isinstance(p, Rectangle):
            geoms.append([p.get_x(), p.get_y(), p.get_width(), p.get_height()])
        elif isinstance(p, Polygon):
            geoms.append(p.get_xy())
    return {
        "kinds": [type(p).__name__ for p in patches],
        "facecolors": [_color(p.get_facecolor()) for p in patches],
        "geom": summarize(_pooled(geoms)) if geoms else [],
    }


def figure_data(fig: Figure) -> dict[str, object]:
    """Everything a figure shows, as data: per axes its artists, labels, limits and legend."""
    axes = []
    for ax in fig.axes:
        entry = {
            "title": ax.get_title(),
            "xlabel": ax.get_xlabel(),
            "ylabel": ax.get_ylabel(),
            "xlim": summarize(ax.get_xlim()),
            "ylim": summarize(ax.get_ylim()),
            "scale": f"{ax.get_xscale()}|{ax.get_yscale()}",
            "lines": _lines(list(ax.lines)),
            "collections": [_collection(c) for c in ax.collections],
            "patches": _patches(list(ax.patches)),
            "texts": [t.get_text() for t in ax.texts],
            "images": [summarize(np.ma.filled(im.get_array(), np.nan)) for im in ax.images],
        }
        if hasattr(ax, "get_zlim"):
            entry["zlabel"] = ax.get_zlabel()
            entry["zlim"] = summarize(ax.get_zlim())
        legend = ax.get_legend()
        entry["legend"] = [t.get_text() for t in legend.get_texts()] if legend else []
        axes.append(entry)
    suptitle = fig._suptitle.get_text() if fig._suptitle is not None else ""
    return {"suptitle": suptitle, "axes": axes}


class FigureCapture:
    """Record the data of every figure drawn inside the block, as it is shown or closed.

    `plt.show` and `plt.close` are wrapped so a figure is read before the code under test closes
    it; figures still open at the end are read then, and all are closed.
    """

    def __enter__(self) -> FigureCapture:
        import matplotlib.pyplot as plt

        plt._get_backend_mod()  # resolve the backend first: resolving it rebinds plt.show
        # A figure an earlier test left open is not this block's: on a shared xdist worker it was
        # read as an extra panel ("length 4 != 3" in test_figure1_panels, 2026-11-26).
        plt.close("all")
        self.plt, self.figures = plt, []
        self._show, self._close = plt.show, plt.close
        plt.show, plt.close = self._on_show, self._on_close
        return self

    def _read(self, fig: Figure) -> None:
        if not getattr(fig, "_characterized", False):
            fig._characterized = True
            self.figures.append(figure_data(fig))

    def _open(self) -> list[object]:
        return [self.plt.figure(n) for n in self.plt.get_fignums()]

    def _on_show(self, *args: object, **kwargs: object) -> None:
        for fig in self._open():
            self._read(fig)

    def _on_close(self, fig: object = None) -> None:
        from matplotlib.figure import Figure

        if isinstance(fig, Figure):
            self._read(fig)
        elif fig == "all" or (fig is None and self.plt.get_fignums()):
            for f in self._open() if fig == "all" else [self.plt.gcf()]:
                self._read(f)
        self._close(fig) if fig is not None else self._close()

    def __exit__(self, *exc: object) -> None:
        for fig in self._open():
            self._read(fig)
        self.plt.show, self.plt.close = self._show, self._close
        self._close("all")


# ── Released fits in the pipeline's result format ─────────────────────────

# Parameter columns of the released full-coverage table, in each law's loss_fn order.
PARAM_COLUMNS = {
    "Kaplan": ["param_0", "param_1", "param_2", "param_3"],
    "Chinchilla": ["param_0", "param_1", "param_2", "param_3", "param_4"],
    "Droppo-Elibol": [f"param_{i}" for i in range(6)],
}
EPOCH_PARAMS = ["A", "alpha", "B", "beta", "E", "R_star_D", "R_star_N"]


def released_fits(table: pd.DataFrame, seed: int = 0) -> list[dict[str, object]]:
    """One result dict per (dataset, law, mode, design) of the released per-seed table.

    The dicts have the keys `run_dataset_fits` returns (design "Colinear" / "Non-Colinear", mode
    "Final" / "First" / "Second" / "Epoch-Expanded", params as loss_fn takes them), taken from the
    given seed's row, so tables and figures can be drawn from released fits without refitting.
    """
    designs = {"CO": "Colinear", "NC": "Non-Colinear"}
    out = []
    rows = table[table["seed"] == seed].sort_values(["dataset", "law", "mode", "design"])
    for row in rows.to_dict("records"):
        law, mode = row["law"], row["mode"]
        if law == "Parametric-Epoch":
            params: dict[str, float] | np.ndarray = {
                p: float(row[f"param_{p}"]) for p in EPOCH_PARAMS
            }
        else:
            params = np.array([float(row[c]) for c in PARAM_COLUMNS[law]])
        entry = {
            "dataset": row["dataset"],
            "law": law,
            "design": designs[row["design"]],
            "mode": mode if mode == "Epoch-Expanded" else mode.capitalize(),
            "params": params,
            "sse": float("nan"),
        }
        for split in ("train", "unified", "colinear", "noncolinear"):
            entry[f"r2_{split}"] = float(row[f"r2_{split}"])
            entry[f"rmse_{split}"] = float(row[f"rmse_{split}"])
        out.append(entry)
    return out
