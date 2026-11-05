"""The tables of the public-checkpoint appendix and the study counts behind them.

1. Each table writer of camera_ready/public_checkpoints/tables/, run on the committed CSVs,
   writes back the committed table body in results/tables/ byte for byte. Those bodies are
   the paper's; the rows checked here are literals copied from the appendix.
2. The study counts the appendix prints are in the committed per_study.csv and screening
   logs.
3. From the HuggingFace run tables (network), per_study.py writes back per_study.csv, and the
   tables hold the rows the appendix's "Sources and licenses" paragraph counts.
"""

from __future__ import annotations

import pandas as pd
import pytest

from conftest import A2, assert_same_rows, run

RESULTS = A2 / "results"
TABLES = RESULTS / "tables"
SCRIPTS = A2 / "tables"
WRITERS = {
    "make_table.py": ["public_checkpoints_studies.tex"],
    "tables_tex.py": ["public_checkpoints_results.tex", "public_checkpoints_regimes.tex"],
    "tables_a5_tex.py": ["public_checkpoints_1b.tex", "public_checkpoints_windows.tex"],
}


# ── 1. the table bodies ────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("script", sorted(WRITERS))
def test_table_writers_reproduce_committed_bodies(tmp_path, script):
    run(["python", SCRIPTS / script, "--out-dir", tmp_path], cwd=A2)
    for name in WRITERS[script]:
        assert (tmp_path / name).read_bytes() == (TABLES / name).read_bytes(), name
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(WRITERS[script])


@pytest.mark.parametrize("name,line", [
    ("public_checkpoints_results.tex",
     r"Within a cell (primary) & $40\%$ & $[24\%, 52\%]$ & $4\%$ & $[-10\%, 21\%]$ \\"),
    ("public_checkpoints_results.tex",
     r"Size interaction, per tenfold $N$ & $-14\%$ & $[-40\%, 18\%]$ & $-39\%$ & $[-57\%, -18\%]$ \\"),
    ("public_checkpoints_regimes.tex",
     r"$10^4$ & A & B & $59.0\%$ & $0.074$ $[0.003, 0.153]$ & $58.6\%$ & $0.076$ $[0.047, 0.103]$ \\"),
    ("public_checkpoints_regimes.tex",
     r"$10^2$ & A & B & \multicolumn{2}{c}{no pairs} & $53.1\%$ & $0.437$ $[0.437, 0.437]$ \\"),
    ("public_checkpoints_1b.tex",
     r"Fixed effect per window & $32\%$ & $[-0.3\%, 39\%]$ & $27\%$ & $[-6\%, 35\%]$ \\"),
    ("public_checkpoints_1b.tex",
     r"Attenuation bound, share of the effect below 1B & \multicolumn{2}{c}{$112\%$} & \multicolumn{2}{c}{-} \\"),
    ("public_checkpoints_windows.tex",
     r"8 & 8 & 4 & $2\%$ & $[2\%, 47\%]$ & same \\"),
    ("public_checkpoints_studies.tex",
     r"\raggedright MAD~\citep{poli2024mechanistic,choshen2024hitchhiker} & trajectory & 648 & 40 & 40.2M--3.08B & 0.00026--2816 \\"),
    ("public_checkpoints_studies.tex",
     r"\raggedright Chinchilla isoFLOP~\citep{hoffmann2022training,besiroglu2024chinchilla,choshen2024hitchhiker} & final & 245 & 43 & 60M--16.2B & 0.036--379 \\"),
])
def test_table_rows_as_printed(name, line):
    assert line in (TABLES / name).read_text().splitlines()


# ── 2. the study counts ───────────────────────────────────────────────────────────────────

def test_study_counts_as_printed():
    per = pd.read_csv(RESULTS / "per_study.csv")
    assert len(per) == 10
    final = per[per["corpus"] == "E"]
    assert len(final) == 5 and final["runs_used"].sum() == 984  # the primary corpus
    assert per["runs_used"].sum() == 12_531  # the wider corpus, the 984 included
    assert per.set_index("study").loc["Farseer", "runs_used"] == 404
    assert per.set_index("study").loc["Gemstones annealed", "n_cells"] == 1
    # Gadre et al.'s three cells each hold one window
    assert per.set_index("study").loc["Gadre et al.", ["n_cells", "n_windows_W1"]].tolist() == [3, 3]


def test_screening_as_printed():
    screen = pd.read_csv(RESULTS / "screening_E.csv")
    muen = screen[screen["cell"].str.startswith("datablations/")]
    assert not muen["included"].any() and muen["n_distinct_N"].max() == 3
    src = pd.read_csv(RESULTS / "screening_sources.csv")
    out = src[src["verdict"] == "OUT"]["rule"].value_counts()
    assert out.idxmax() == "P3" and out["P3"] > len(src[src["verdict"] == "OUT"]) / 2


# ── 3. from the HuggingFace run tables ─────────────────────────────────────────────────────

@pytest.mark.network
def test_per_study_reproduces(pc_work):
    path, env = pc_work
    run(["python", SCRIPTS / "per_study.py", "--out", "per_study_fresh.csv"], cwd=A2, env=env)
    assert_same_rows(pd.read_csv(path / "per_study_fresh.csv"),
                     pd.read_csv(RESULTS / "per_study.csv"), ["study"])


@pytest.mark.network
def test_released_tables_as_printed(pc_work):
    path, _ = pc_work
    expanded = pd.read_csv(path / "artifacts" / "runs_expanded.csv")
    normalised = pd.read_csv(path / "artifacts" / "runs_normalised.csv")
    assert len(expanded) == 14_065 and len(normalised) == 2_215
    for table in (expanded, normalised):
        assert table["cell"].str.startswith("datablations/").sum() == 1_462
    studies = set(normalised["cell"].str.split("/").str[0])
    assert studies == {"farseer", "chinchilla_isoflop", "gadre_overtrain", "datablations"}
    assert (normalised["cell"].str.startswith("farseer/")).sum() == 404
