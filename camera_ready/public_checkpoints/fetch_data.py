#!/usr/bin/env python3
"""Fetch the public loss artifacts named in CRITERIA.md and normalise them.

GET-only, no authentication. Everything is cached under ``<work>/artifacts/`` so the
analysis is reproducible without re-downloading.

Normalised schema (one row per released run/observation)::

    cell, run_id, N, D, k, loss

``cell`` is ``suite/corpus`` and is never pooled across suites (CRITERIA P4).
"""

from __future__ import annotations

import io
import re
import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import zstandard as zstd

from paths import ARTIFACTS, WORK

FARSEER_URL = "https://raw.githubusercontent.com/Farseer-Scaling-Law/Farseer/main/ipynb/data/1222_full.csv"
COLPRET_URL = "https://raw.githubusercontent.com/IBM/ColPret/main/test_cache/data.csv.zst"
GEMSTONE_RAW = "https://raw.githubusercontent.com/mcleish7/gemstone-scaling-laws/main"
LADDER_RAW = "https://raw.githubusercontent.com/allenai/OLMo-ladder/main/src/scripts/paper/data/ladder-runs"
DATADECIDE_URL = "https://raw.githubusercontent.com/allenai/DataDecide/main/perplexity_metrics_by_group.csv"

# CRITERIA Amendment 4, A4.1a: at most this many checkpoints are kept per run, spaced
# log-uniformly in D. Fixed before any new loss value entered a fit.
THIN_MAX_CKPT = 12

# CRITERIA Amendment 4, A4.3: the Ai2 suites release the same eleven validation sets; C4 is
# the one common to all of them, so it is the primary and the loss definition is held fixed.
AI2_PRIMARY_EVAL = "c4_en-validation"
AI2_SENSITIVITY_EVAL = "pile-validation"

# DataDecide token reconstruction, from the suite's own released constants
# (allenai/DataDecide, scaling_laws/utils/constants/constants_models.py):
# D = step * batch_size(size) * sequence_length, sequence_length = 2048.
DD_SEQ_LEN = 2048
DD_BATCH = {
    "4M": 32, "6M": 32, "8M": 32, "10M": 32, "14M": 32, "16M": 32, "20M": 64,
    "60M": 96, "90M": 160, "150M": 192, "300M": 320, "530M": 448, "750M": 576, "1B": 704,
}
DD_FULL_SCHEDULE = {
    "4M": 5725, "6M": 9182, "8M": 13039, "10M": 15117, "14M": 21953, "16M": 24432,
    "20M": 14584, "60M": 29042, "90M": 29901, "150M": 38157, "300M": 45787,
    "530M": 57786, "750M": 63589, "1B": 69369,
}
DD_PARAMS = {
    "4M": 3_744_832, "6M": 6_010_464, "8M": 8_538_240, "10M": 9_900_432, "12M": 12_066_600,
    "14M": 14_380_224, "16M": 16_004_560, "20M": 19_101_888, "60M": 57_078_144,
    "90M": 97_946_640, "150M": 151_898_880, "300M": 319_980_544, "530M": 530_074_944,
    "750M": 681_297_408, "1B": 1_176_832_000,
}

# OLMo ladder parameter counts, from the suite's own src/scaling/utils.py MODEL_PARAMS.
LADDER_PARAMS = {"190M": 190_354_176, "370M": 371_262_464, "760M": 758_220_288,
                 "1B": 1_279_395_840, "3B": 3_169_537_280}
LADDER_MULTIPLIERS = ("0.5xC", "1xC", "1xC-rerun", "2xC", "5xC", "10xC")

# Source-level screening verdicts (P1/P2'/P3/P4), logged for every candidate examined.
SOURCE_SCREEN: list[dict] = []

COLPRET_KEEP = [
    "original_paper", "model_name", "model_type", "arch", "domain", "data",
    "num_params", "tokens_seen", "epochs", "seed", "flops",
    "loss", "openLM_loss", "train_loss", "val_loss", "compute_opt", "scaled_set",
]

HTTP_TIMEOUT_S = 600
CHUNK = 1 << 20


@dataclass(frozen=True)
class Artifact:
    """A cached public download."""

    url: str
    path: Path

    def ensure(self) -> Path:
        if self.path.exists() and self.path.stat().st_size > 0:
            return self.path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".part")
        with urllib.request.urlopen(self.url, timeout=HTTP_TIMEOUT_S) as resp, tmp.open("wb") as out:
            shutil.copyfileobj(resp, out, CHUNK)
        tmp.rename(self.path)
        return self.path


def _colpret_slim() -> pd.DataFrame:
    """Stream-decompress the ColPret master table down to the columns we use."""
    slim = ARTIFACTS / "colpret_slim.csv"
    if not slim.exists():
        raw = Artifact(COLPRET_URL, ARTIFACTS / "colpret_data.csv.zst").ensure()
        dctx = zstd.ZstdDecompressor()
        with raw.open("rb") as fh, dctx.stream_reader(fh) as reader:
            text = io.TextIOWrapper(reader, encoding="utf-8", newline="")
            chunks = []
            for chunk in pd.read_csv(text, chunksize=200_000, low_memory=False):
                cols = [c for c in COLPRET_KEEP if c in chunk.columns]
                chunks.append(chunk[cols])
        pd.concat(chunks, ignore_index=True).to_csv(slim, index=False)
    return pd.read_csv(slim, low_memory=False)


def load_farseer() -> pd.DataFrame:
    """Li et al. (2025) Farseer: 404 released runs, final loss, crossed (N, D) grid."""
    path = Artifact(FARSEER_URL, ARTIFACTS / "farseer_1222_full.csv").ensure()
    raw = pd.read_csv(path)
    out = pd.DataFrame({
        "cell": "farseer/IntelliVal",
        "run_id": [f"farseer-{i}" for i in range(len(raw))],
        "N": raw["N"].astype(float),
        "D": raw["D"].astype(float),
        "loss": raw["loss"].astype(float),
        "loss_smooth": raw["smooth loss"].astype(float),
        "loss_val": raw["L"].astype(float),
    })
    return out[(out["N"] > 0) & (out["D"] > 0) & (out["loss"] > 0)]


def load_gadre_overtrain(colpret: pd.DataFrame) -> pd.DataFrame:
    """Gadre et al. (2024): 6 sizes x 8 token multipliers x 3 corpora, final losses.

    ColPret stores the whole training trajectory; CRITERIA P2 admits only the
    end-of-run point, so each (corpus, N, multiplier) run is reduced to its
    largest-D observation.
    """
    df = colpret[(colpret["original_paper"] == "overtrain") & colpret["openLM_loss"].notna()].copy()
    # Six ColPret rows (all three 6.9B runs and one 1B run per corpus) carry a null
    # `data` field. The corpus is unambiguous from the model_name prefix, so it is
    # repaired here rather than silently losing the suite's largest models. This is a
    # mechanical repair made before any fit; see README.md "Data-prep notes".
    prefix_to_corpus = {"c4_original": "C4", "rpj": "RedPajama", "rw_original": "RefinedWeb"}
    recovered = df["model_name"].astype(str).str.split("-").str[0].map(prefix_to_corpus)
    df["data"] = df["data"].where(df["data"].isin(prefix_to_corpus.values()), recovered)
    df = df[df["data"].isin(["C4", "RedPajama", "RefinedWeb"])]
    df["N"] = df["num_params"].astype(float)
    df["D"] = df["tokens_seen"].astype(float)
    df = df[(df["N"] > 0) & (df["D"] > 0) & (df["openLM_loss"] > 0)]
    df["run_key"] = df["data"] + "|" + df["N"].astype(str) + "|" + df["compute_opt"].astype(str)
    final = df.loc[df.groupby("run_key")["D"].idxmax()]
    return pd.DataFrame({
        "cell": "gadre_overtrain/" + final["data"],
        "run_id": final["run_key"].values,
        "N": final["N"].values,
        "D": final["D"].values,
        "loss": final["openLM_loss"].astype(float).values,
    })


def load_chinchilla(colpret: pd.DataFrame) -> pd.DataFrame:
    """Chinchilla Fig. 4 isoFLOP sweep, digitized by Besiroglu et al. (2024).

    Every row is an already-terminal (N, D, loss) observation of the sweep, i.e.
    the observation set Hoffmann et al. themselves fit - admitted by CRITERIA P2.
    """
    df = colpret[(colpret["original_paper"] == "chinchilla") & colpret["loss"].notna()].copy()
    df["N"] = df["num_params"].astype(float)
    df["D"] = df["tokens_seen"].astype(float)
    df = df[(df["N"] > 0) & (df["D"] > 0) & (df["loss"] > 0)]
    return pd.DataFrame({
        "cell": "chinchilla_isoflop/MassiveText",
        "run_id": [f"chin-{i}" for i in range(len(df))],
        "N": df["N"].values,
        "D": df["D"].values,
        "loss": df["loss"].astype(float).values,
    })


def load_datablations(colpret: pd.DataFrame) -> pd.DataFrame:
    """Muennighoff et al. (2023), single-epoch rows only (repeated data is a
    separate regime the paper treats in App. B.7)."""
    df = colpret[(colpret["original_paper"] == "datablations") & colpret["loss"].notna()].copy()
    df = df[df["epochs"].astype(float) == 1.0]
    df["N"] = df["num_params"].astype(float)
    df["D"] = df["tokens_seen"].astype(float)
    df = df[(df["N"] > 0) & (df["D"] > 0) & (df["loss"] > 0)]
    return pd.DataFrame({
        "cell": "datablations/" + df["data"].astype(str),
        "run_id": df["model_name"].astype(str).values,
        "N": df["N"].values,
        "D": df["D"].values,
        "loss": df["loss"].astype(float).values,
    })


# ════════════════════════════════════════════════════════════════════════════
# CRITERIA Amendment 4 - suites admitted by the reopened candidate list
# ════════════════════════════════════════════════════════════════════════════

def _note(source: str, verdict: str, rule: str, detail: str = "") -> None:
    SOURCE_SCREEN.append({"source": source, "verdict": verdict, "rule": rule, "detail": detail})


def thin_trajectory(df: pd.DataFrame, run_col: str, d_col: str = "D") -> pd.DataFrame:
    """CRITERIA A4.1a: keep <= THIN_MAX_CKPT log-uniformly spaced checkpoints per run.

    The run's largest D is always kept. Selection touches the D axis only and never
    looks at a loss, so it cannot be an outcome-dependent filter.
    """
    keep: list[pd.Index] = []
    for _, grp in df.groupby(run_col, sort=False):
        grp = grp.sort_values(d_col)
        d = grp[d_col].to_numpy(float)
        if len(grp) <= THIN_MAX_CKPT:
            keep.append(grp.index)
            continue
        targets = np.logspace(np.log10(d[0]), np.log10(d[-1]), THIN_MAX_CKPT)
        picked = sorted({int(np.abs(d - t).argmin()) for t in targets} | {len(d) - 1})
        keep.append(grp.index[picked])
    return df.loc[np.concatenate([k.to_numpy() for k in keep])]


def _gemstone(name: str) -> pd.DataFrame:
    path = Artifact(f"{GEMSTONE_RAW}/wandb_dfs/{name}.jsonl", ARTIFACTS / f"gemstones_{name}.jsonl").ensure()
    raw = pd.read_json(path, lines=True)
    return pd.DataFrame({
        "run_id": raw["run_name"].astype(str),
        "N": raw["params_active_precise"].astype(float),
        "D": raw["tokens"].astype(float),
        "loss": raw["final_loss"].astype(float),
    })


def load_gemstones_cooldown() -> pd.DataFrame:
    """Gemstones (McLeish et al. 2025), cooldown-END checkpoints.

    Each pretraining run is branched at ten token counts and each branch is annealed to
    completion, so every row is a TERMINAL observation of a fully decayed model. Admissible
    under the ORIGINAL P2; it belongs to the E-corpus and needs no amendment.
    """
    df = _gemstone("wandb_df_for_fitting_cool_end")
    _note("Gemstones cooldown-end (mcleish7/gemstone-scaling-laws)", "IN (E-corpus)",
          "P1-P4 pass", f"{len(df)} annealed branch ends, {df['N'].nunique()} distinct N")
    return df.assign(cell="gemstones_cooldown/pretrain-mix", corpus="E")


def load_gemstones_hot() -> pd.DataFrame:
    """Gemstones constant-learning-rate ('hot') checkpoints, thinned per A4.1a.

    The main Gemstones runs hold the learning rate constant, so P2's decaying-LR premise
    does not apply; they are still intermediate, so they enter the X-corpus.
    """
    df = _gemstone("wandb_df_for_fitting_hot_100b+_477")
    df = thin_trajectory(df, "run_id")
    _note("Gemstones hot / constant-LR checkpoints", "IN (X-corpus)",
          "P2' (A4.1); P1,P3,P4 pass", f"{len(df)} thinned points, {df['N'].nunique()} distinct N")
    return df.assign(cell="gemstones_hot/pretrain-mix", corpus="X")


def _ladder_run(fname: str, n_params: float) -> pd.DataFrame | None:
    """One OLMo-ladder run CSV -> its terminal (N, D, loss) observation."""
    try:
        path = Artifact(f"{LADDER_RAW}/{fname}", ARTIFACTS / f"ladder_{fname}").ensure()
    except Exception:
        return None
    raw = pd.read_csv(path, low_memory=False)
    loss_col = f"eval/{AI2_PRIMARY_EVAL}/CrossEntropyLoss"
    if loss_col not in raw.columns:
        return None
    raw = raw[raw[loss_col].notna()].copy()
    if raw.empty:
        return None
    if "throughput/total_tokens" in raw.columns and raw["throughput/total_tokens"].notna().any():
        raw["D"] = raw["throughput/total_tokens"].astype(float)
    elif "batch_size_in_tokens" in raw.columns:
        raw["D"] = raw["_step"].astype(float) * raw["batch_size_in_tokens"].astype(float)
    else:
        return None
    raw = raw[raw["D"] > 0]
    if raw.empty:
        return None
    final = raw.loc[[raw["D"].idxmax()]]
    return pd.DataFrame({
        "run_id": [fname.replace(".csv", "")],
        "N": [float(n_params)],
        "D": [float(final["D"].iloc[0])],
        "loss": [float(final[loss_col].iloc[0])],
    })


def load_olmo_ladder() -> pd.DataFrame:
    """Bhagia et al. OLMo model ladder: 5 sizes x 6 token multipliers, plus the 7B and 13B.

    The final checkpoint of each ladder run is the end of that run's linear decay, i.e. a
    terminal observation - admissible under the ORIGINAL P2, so this is an E-corpus cell.
    """
    parts = [
        _ladder_run(f"{size}-{mult}.csv", n)
        for size, n in LADDER_PARAMS.items() for mult in LADDER_MULTIPLIERS
    ]
    parts.append(_ladder_run("peteish7_eval_anneal.csv", 6_887_575_552))
    parts.append(_ladder_run("peteish13_eval_full.csv", 13_202_396_160))
    df = pd.concat([p for p in parts if p is not None], ignore_index=True)
    _note("OLMo ladder / Bhagia et al. (allenai/OLMo-ladder)", "IN (E-corpus)",
          "P1-P4 pass", f"{len(df)} terminal runs, {df['N'].nunique()} distinct N")
    return df.assign(cell="olmo_ladder/dolma17-peteish", corpus="E")


def _mad(colpret: pd.DataFrame) -> pd.DataFrame:
    df = colpret[(colpret["original_paper"] == "MAD arxiv.org/2403.17844")
                 & colpret["loss"].notna()].copy()
    df["family"] = df["model_name"].astype(str).str.split("_").str[0]
    return pd.DataFrame({
        # CRITERIA P4: one architecture family per cell, so MAD splits in two.
        "cell": "mad/pile-" + df["family"].str.replace(" ", ""),
        "run_id": df["model_name"].astype(str).values,
        "N": df["num_params"].astype(float).values,
        "D": df["tokens_seen"].astype(float).values,
        "loss": df["loss"].astype(float).values,
    })


def load_mad_endofrun(colpret: pd.DataFrame) -> pd.DataFrame:
    df = _mad(colpret)
    return df.loc[df.groupby("run_id")["D"].idxmax()].assign(corpus="E")


def load_mad_thinned(colpret: pd.DataFrame) -> pd.DataFrame:
    return thin_trajectory(_mad(colpret), "run_id").assign(corpus="X")


def load_gpt3(colpret: pd.DataFrame) -> pd.DataFrame:
    """GPT-3 digitized training curves. C5: validation loss is primary where both exist."""
    df = colpret[(colpret["original_paper"] == "gpt3 - arxiv.org-abs-2005.14165")
                 & colpret["val_loss"].notna()].copy()
    out = pd.DataFrame({
        "cell": "gpt3/gpt3-mix",
        "run_id": df["model_name"].astype(str).values,
        "N": df["num_params"].astype(float).values,
        "D": df["tokens_seen"].astype(float).values,
        "loss": df["val_loss"].astype(float).values,
    })
    out = thin_trajectory(out, "run_id")
    _note("GPT-3 digitized curves (ColPret)", "IN (X-corpus)", "P2' (A4.1); P1,P3,P4 pass",
          f"{len(out)} thinned points, {out['N'].nunique()} distinct N; digitized")
    return out.assign(corpus="X")


def load_opt_trajectories(colpret: pd.DataFrame) -> pd.DataFrame:
    """OPT trajectories (ColPret `training_trajectories`).

    The released column spans 9.8 to 67.5, which cannot be a per-token cross-entropy in nats
    (that is bounded by log(vocab)); it is a perplexity, and A4.3 enters it as log(ppl).
    """
    df = colpret[(colpret["original_paper"] == "training_trajectories")
                 & colpret["loss"].notna()].copy()
    out = pd.DataFrame({
        "cell": "opt_trajectories/roberta-pile-reddit",
        "run_id": df["model_name"].astype(str).values,
        "N": df["num_params"].astype(float).values,
        "D": df["tokens_seen"].astype(float).values,
        "loss": np.log(df["loss"].astype(float).values),
    })
    out = thin_trajectory(out, "run_id")
    _note("OPT training trajectories (ColPret)", "IN (X-corpus)",
          "P2' (A4.1); P1 via log(perplexity) per A4.3",
          f"{len(out)} thinned points, {out['N'].nunique()} distinct N")
    return out.assign(corpus="X")


_DD_GROUP = re.compile(r"^(?P<recipe>.+)-(?P<size>\d+[MB])-(?P<mult>\d+xC)(?:-(?P<seed>.+))?$")


def load_datadecide() -> pd.DataFrame:
    """DataDecide (Magnusson et al. 2025): 14 sizes x 25+ data recipes, per-step perplexity.

    D is reconstructed from the suite's own released constants,
    D = step * batch_size(size) * 2048, and the loss is log(perplexity) on the A4.3 primary
    validation set. Every recipe is a distinct training corpus, so P4 makes each one a cell.
    """
    path = Artifact(DATADECIDE_URL, ARTIFACTS / "datadecide_ppl_by_group.csv").ensure()
    raw = pd.read_csv(path)
    col = f"eval/{AI2_PRIMARY_EVAL}/Perplexity"
    meta = raw["group"].astype(str).str.extract(_DD_GROUP)
    df = raw.assign(**{c: meta[c] for c in meta.columns})
    df = df[df[col].notna() & df["size"].isin(DD_PARAMS) & df["recipe"].notna()].copy()
    # Integrity check on the reconstruction, not an outcome filter: the suite publishes each
    # size's full step count, and every DataDecide run is a 5xC run. 48 of the 1103 released
    # groups (all seed-14/15 at 150M/300M) carry step values up to 11x that schedule, which
    # would put a 150M model at 1097 tokens per parameter instead of 100. For those rows the
    # `step` field is not the quantity this reconstruction assumes, so D is NOT recoverable
    # and they fail P1. They are dropped and counted.
    over = df["step"].astype(float) > 1.02 * df["size"].map(DD_FULL_SCHEDULE).astype(float)
    n_over, n_groups_over = int(over.sum()), int(df.loc[over, "group"].nunique())
    df = df[~over]
    out = pd.DataFrame({
        "cell": "datadecide/" + df["recipe"].astype(str),
        "run_id": df["group"].astype(str).values,
        "N": df["size"].map(DD_PARAMS).astype(float).values,
        "D": (df["step"].astype(float) * df["size"].map(DD_BATCH).astype(float)
              * DD_SEQ_LEN).values,
        "loss": np.log(df[col].astype(float).values),
    })
    out = out[out["D"] > 0]
    out = thin_trajectory(out, "run_id")
    _note("DataDecide (allenai/DataDecide)", "IN (X-corpus)",
          "P2' (A4.1); P1 via log(perplexity); D reconstructed per A4.3",
          f"{len(out)} thinned points, {out['N'].nunique()} distinct N, "
          f"{out['cell'].nunique()} recipe cells; dropped {n_over} rows in "
          f"{n_groups_over} groups whose step exceeds the suite's own full schedule (P1: D "
          f"not recoverable for those rows)")
    return out.assign(corpus="X")


# Every remaining constituent of the ColPret aggregation, screened individually. The wide
# ColPret table carries loss columns beyond the four the slim cache keeps, so each of these
# was re-checked against the FULL column list before its verdict was recorded; the loss
# column that exists (if any) is named in the detail so the verdict can be audited.
COLPRET_REJECTS = (
    ("t5-pile", "T5-Pile (ColPret)", "P3",
     "`cross_ent_loss` IS a genuine LM loss (796 values), so P1 passes - but only 4 distinct N"),
    ("datablations", "Muennighoff et al. datablations, single-epoch slice", "P3",
     "`loss` released; 3 distinct N per corpus cell (2.8B/4.2B/8.7B), and 3 < 6"),
    ("pythia", "Pythia (ColPret + EleutherAI/pythia repo)", "P3",
     "the only LM-perplexity column is `wikitext word_perplexity`: 240 values, 5 distinct N, "
     "15 checkpoints each - 5 < 6. `lambada_openai ppl` spans 22 N but mixes bloom/opt/pythia "
     "(P4) and is a task perplexity. The EleutherAI/pythia repo publishes downstream eval CSVs "
     "only, no loss logs"),
    ("olmo", "OLMo / OLMo-2 (ColPret)", "P3",
     "`c4_en-validation/CrossEntropyLoss` and `pile-validation/CrossEntropyLoss` ARE released "
     "(529 / 737 values), so P1 passes - but only 2 distinct N (1B, 7B)"),
    ("blog-redpajama-7b", "RedPajama-INCITE", "P3",
     "`train_loss` released; 2 distinct N (2.4B, 6.7B)"),
    ("LM360:arxiv-2312.06550", "LLM360 Amber / K2", "P3",
     "`train - loss` (99999 values) and `k2 training - lm loss` (9985) ARE released, so P1 "
     "passes - but 2 distinct N (6.7B Amber, 65B K2), one per family"),
    ("llm360-private", "LLM360 CrystalCoder / Mamba-3B", "P3", "`loss` released; 1 distinct N"),
    ("DataMixing arxiv.org/abs/2405.14908", "DataMixing (Ye et al. 2024)", "P3",
     "`avg_ppl` released; 1 distinct N (280M) across 12 data mixtures"),
    ("Revisiting Neural Scaling Laws in Language and Vision",
     "Alabdulmohsin et al., Revisiting Neural Scaling Laws", "P3",
     "5 distinct N carry a released loss (the other 5 model names have num_params = 0)"),
)


def note_rejected(colpret: pd.DataFrame) -> None:
    """Log every remaining candidate examined, with the rule it failed."""
    for paper, label, rule, detail in COLPRET_REJECTS:
        g = colpret[colpret["original_paper"] == paper]
        _note(label, "OUT", rule, f"{detail} [{len(g)} ColPret rows]")
    _note("Cerebras-GPT", "OUT", "P3", "one budget per size, D = 20N: a single ray, K = 1")
    _note("Sardana et al. (2025)", "OUT", "P1",
          "run-level loss table not recovered at a public GET-only path (not-recovered, not absent)")
    _note("OPT / BLOOM / GPT-NeoX / Qwen / Llama technical reports", "OUT", "P1/P3",
          "final losses in report tables only, 1-3 sizes each, single ray")
    _note("HuggingFace dataset sweep (datasets API, GET-only)", "OUT", "P1",
          "smcleish/scaling-laws-cache holds fitted law parameters and predicted losses, not "
          "run-level observations; molmohsen/scaling-laws-neural-lm is a PDF; no other public "
          "HF dataset aggregating scaling-law LM loss curves was found")


def load_all(expanded: bool = False) -> pd.DataFrame:
    colpret = _colpret_slim()
    frames = [
        load_farseer().assign(corpus="E"),
        load_gadre_overtrain(colpret).assign(corpus="E"),
        load_chinchilla(colpret).assign(corpus="E"),
        load_datablations(colpret).assign(corpus="E"),
    ]
    if expanded:
        mad_end = load_mad_endofrun(colpret)
        frames += [
            load_gemstones_cooldown(),
            load_olmo_ladder(),
            mad_end,
            load_gemstones_hot(),
            load_gpt3(colpret),
            load_opt_trajectories(colpret),
            load_datadecide(),
        ]
        # A4.1b: a suite contributes its terminal slice OR its thinned trajectory, never
        # both. MAD's terminal slice is kept only if it clears P3; the decision is made by
        # the pre-registered screen, not by any outcome.
        import analysis  # local import: analysis.py owns the P3 screen
        keep_end = {
            cell for cell, grp in mad_end.groupby("cell")
            if analysis.screen_cell(grp.assign(k=grp["D"] / grp["N"]), cell).included
        }
        if not keep_end:
            frames = [f for f in frames if f is not mad_end]
            frames.append(load_mad_thinned(colpret))
            _note("MAD (arXiv:2403.17844) end-of-run slice", "OUT", "P3",
                  "the 54 end-of-run runs do not populate 3 k-bins at 4+ sizes; the thinned "
                  "trajectory slice is used instead (X-corpus)")
            _note("MAD (arXiv:2403.17844) thinned trajectories", "IN (X-corpus)",
                  "P2' (A4.1); P1,P3,P4 pass", "split by architecture family per P4")
        else:
            _note("MAD (arXiv:2403.17844) end-of-run slice", "IN (E-corpus)", "P1-P4 pass",
                  f"cells {sorted(keep_end)}")
        note_rejected(colpret)
    out = pd.concat(frames, ignore_index=True)
    out["k"] = out["D"] / out["N"]
    return out[(out["N"] > 0) & (out["D"] > 0) & np.isfinite(out["loss"])]


def main(argv: list[str] | None = None) -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Fetch the public loss artifacts and normalise them "
                                 "(writes artifacts/runs_normalised.csv or runs_expanded.csv "
                                 "under the work directory).")
    ap.add_argument("--expanded", action="store_true",
                    help="the ten-suite corpus with mid-training checkpoints (Amendment 4)")
    args = ap.parse_args(argv)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    expanded = args.expanded
    data = load_all(expanded=expanded)
    name = "runs_expanded.csv" if expanded else "runs_normalised.csv"
    data.to_csv(ARTIFACTS / name, index=False)
    if SOURCE_SCREEN:
        pd.DataFrame(SOURCE_SCREEN).to_csv(WORK / "screening_sources.csv", index=False)
    for cell, grp in data.groupby("cell"):
        corpus = grp['corpus'].iloc[0] if 'corpus' in grp else '-'
        print(f"{cell:44s} {corpus} pts={len(grp):5d} #N={grp['N'].nunique():3d} "
              f"N=[{grp['N'].min():.3g},{grp['N'].max():.4g}] "
              f"k=[{grp['k'].min():.3g},{grp['k'].max():.4g}]")
    print(f"\nwrote {name}: {len(data)} rows, {data['cell'].nunique()} cells")


if __name__ == "__main__":
    main()
