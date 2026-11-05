#!/usr/bin/env python3
"""Per-study points, distinct sizes, size range and TPP range of the cells the screen admits.

Reads the expanded run table (HuggingFace leibnitz-lab/public_checkpoint_reanalysis, via
get_runs.py), applies the protocol's screen (`analysis.screen_cell`, CRITERIA P3) per cell
with the corpus filter of analysis.py (E: rows with corpus E; X: rows with corpus E or X),
and counts per study: the points used, the distinct model sizes, N and D/N ranges, the W1
windows (`analysis.build_windows`) and the rows that enter at least one window. A point is a
fully trained run in the five final-loss releases and a thinned checkpoint in the five
trajectory releases.

Writes ``per_study.csv``, the input of make_table.py. Relative paths are taken in the work
directory.

    uv run python tables/per_study.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the reanalysis modules
from analysis import build_windows, screen_cell
from paths import WORK

STUDY = {  # cell prefix -> the study's name in per_study.csv, in the table's order
    "farseer": "Farseer", "chinchilla_isoflop": "Chinchilla isoFLOP",
    "gemstones_cooldown": "Gemstones annealed", "gadre_overtrain": "Gadre et al.",
    "olmo_ladder": "OLMo ladder", "datadecide": "DataDecide", "mad": "MAD",
    "gemstones_hot": "Gemstones constant LR", "gpt3": "GPT-3 curves",
    "opt_trajectories": "OPT trajectories",
}
CORPORA = {"E": {"E"}, "X": {"E", "X"}}


def admitted(runs: pd.DataFrame, corpus: str) -> list[str]:
    """The cells of `corpus` that pass the screen."""
    sub = runs[runs["corpus"].isin(CORPORA[corpus])]
    return [c for c, g in sub.groupby("cell") if screen_cell(g, c).included]


def study_row(prefix: str, g: pd.DataFrame) -> dict:
    tpp = g["D"] / g["N"]
    per_cell = g.groupby("cell")["N"].nunique()
    in_win: set = set()
    n_windows = 0
    for c, gc in g.groupby("cell"):
        ws = build_windows(c, gc["N"].unique())
        n_windows += len(ws)
        in_win |= {(c, n) for w in ws for n in w.sizes}
    gw = g[[(c, n) in in_win for c, n in zip(g["cell"], g["N"])]]
    return {"study": STUDY[prefix], "cell_prefix": prefix,
            "corpus": "/".join(sorted(g["corpus"].unique())),
            "n_cells": g["cell"].nunique(), "runs_used": len(g),
            "distinct_N_union": g["N"].nunique(),
            "distinct_N_sum_over_cells": int(per_cell.sum()),
            "distinct_N_max_cell": int(per_cell.max()),
            "N_min": g["N"].min(), "N_max": g["N"].max(),
            "TPP_min": tpp.min(), "TPP_max": tpp.max(),
            "n_windows_W1": n_windows, "rows_in_some_window": len(gw),
            "distinct_N_in_some_window": gw["N"].nunique(),
            "N_max_in_some_window": gw["N"].max()}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--runs", default="artifacts/runs_expanded.csv")
    ap.add_argument("--out", default="per_study.csv")
    args = ap.parse_args(argv)

    runs = pd.read_csv(WORK / args.runs)
    for corpus in CORPORA:
        print(f"corpus {corpus}: {len(admitted(runs, corpus))} cells admitted")
    xs = runs[runs["corpus"].isin(CORPORA["X"]) & runs["cell"].isin(admitted(runs, "X"))]
    prefix = xs["cell"].str.split("/").str[0]
    out = pd.DataFrame([study_row(p, xs[prefix == p]) for p in STUDY])
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(out.to_string(index=False))
    e = out.loc[out["corpus"] == "E", "runs_used"].sum()
    print(f"final-loss points {e}, all points {out['runs_used'].sum()}")
    out.to_csv(WORK / args.out, index=False)


if __name__ == "__main__":
    main()
