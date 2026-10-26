#!/usr/bin/env python3
"""Write SLURM scripts that train the paper's (N, D) designs, one model per run.

One script covers every design that was trained:

    --design main     the 324-run design trained on each of the five corpora
                      (and, with --precision bf16, the BF16 replication on Wikipedia)
    --design bigtpp   the 156-run single-epoch BigTPP design (Wikipedia)

Per corpus the main design is

    NC train   10 sizes x 12 token budgets                       = 120 runs
    CO train   10 sizes x 12 TPP ratios (D = k * N)              = 120 runs
    NC holdout L-shaped: every train size at D = 300M,
               the 54.57M size at all 17 budgets,
               the three largest sizes at the 5 holdout budgets  =  42 runs
    CO holdout the same L shape on TPP ratios                    =  42 runs

Holdout runs train on the validation split (``--use-train-as-val``) so the scaling-law
fits never see their data. Model sizes are targets: the trainer snaps each to the nearest
valid LlamaForCausalLM configuration, so realized parameter counts differ slightly
(paper Appendix, Table "model architectures").

Cluster settings come from flags or environment variables, never from this file:

    SLURM_PARTITION, SLURM_ACCOUNT, SLURM_EXCLUDE, SLURM_TIME
    WORK_DIR   repository checkout on the cluster        (default: this repository)
    DATA_DIR   tokenized corpora, one folder per dataset  (default: <WORK_DIR>/corpora)
    VENV       Python environment to activate             (default: <WORK_DIR>/.venv)

Example:
    python src/tpp_analysis/generate_jobs/generate_jobs.py --dataset c4 --partition gpu
    bash slurm_scripts/c4/submit_all.sh
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

# Peak learning rate per corpus, as trained (checked against the recorded learning-rate
# schedules in the released training metrics).
LEARNING_RATES = {
    "wikipedia": 3e-4,
    "cosmopedia": 1e-3,
    "redpajama": 3e-4,
    "pes2o": 1e-3,
    "c4": 1e-3,
}

# Run-name prefixes used for the released runs. ``get_results.py`` classifies a run from
# the COLINEAR / HOLDOUT / COLINEAR_HOLDOUT tokens in its name, so keep those tokens.
NAME_PREFIX = {
    "wikipedia": "WIKI",
    "cosmopedia": "cosmopedia",
    "redpajama": "RP",
    "pes2o": "PES2O",
    "c4": "C4",
}

TRAIN_SIZES = [
    5_000_000, 10_000_000, 14_000_000, 19_000_000, 22_000_000,
    28_000_000, 32_000_000, 35_000_000, 40_000_000, 45_000_000,
]
HOLDOUT_SIZES = [54_566_912, 60_884_992, 64_044_032, 70_362_112]


@dataclass(frozen=True)
class Design:
    """One trained design: NC token budgets, CO TPP ratios and their holdout extensions."""

    nc_tokens: tuple[int, ...]
    nc_holdout_tokens: tuple[int, ...]
    co_tpp: tuple[float, ...]
    co_holdout_tpp: tuple[float, ...]
    epochs: int
    # main design only: every train size also gets the first holdout budget / ratio
    edge_row: bool
    prefix: str  # "" or a tag such as "BIGTPP_"


DESIGNS = {
    "main": Design(
        nc_tokens=(10_000_000, 35_000_000, 60_000_000, 80_000_000, 100_000_000,
                   130_000_000, 150_000_000, 180_000_000, 200_000_000, 225_000_000,
                   250_000_000, 275_000_000),
        nc_holdout_tokens=(300_000_000, 325_000_000, 350_000_000, 375_000_000, 400_000_000),
        # 3.0 (not 3) reproduces the released run names, e.g. C4_COLINEAR_N10.00M_TPP3.0_D30M
        co_tpp=(1, 1.5, 1.9, 2, 2.5, 2.7, 3.0, 3.3, 3.5, 4, 4.5, 5),
        co_holdout_tpp=(6, 6.2, 6.5, 6.7, 7),
        epochs=3,
        edge_row=True,
        prefix="",
    ),
    "bigtpp": Design(
        nc_tokens=(100_000_000, 225_000_000, 400_000_000, 550_000_000, 700_000_000,
                   825_000_000),
        nc_holdout_tokens=(900_000_000, 1_100_000_000, 1_400_000_000),
        co_tpp=(10, 11, 12, 13, 14, 15),
        co_holdout_tpp=(16, 18, 20),
        epochs=1,
        edge_row=False,
        prefix="BIGTPP_",
    ),
}

BATCH_SIZE = 96   # per GPU; with 4 GPUs the global batch is 384 sequences
SEQ_LEN = 256
WEIGHT_DECAY = 0.1
ARCH = "llama"


@dataclass(frozen=True)
class Run:
    kind: str        # "nxd", "colinear", "holdout", "colinear_holdout"
    n: int
    d: int
    tpp: float | None


def _fmt_m(x: float, digits: int = 2) -> str:
    v = x / 1e6
    return f"{int(v)}M" if v == int(v) else f"{v:.{digits}f}M"


def _fmt_tpp(k: float) -> str:
    return str(k)


def expand(design: Design) -> list[Run]:
    """Every (N, D) run of a design, in a fixed order."""
    runs: list[Run] = []
    for d in design.nc_tokens:
        runs += [Run("nxd", n, d, None) for n in TRAIN_SIZES]
    for k in design.co_tpp:
        runs += [Run("colinear", n, int(n * k), k) for n in TRAIN_SIZES]

    pivot, *largest = HOLDOUT_SIZES
    nc_hold: list[tuple[int, int]] = []
    co_hold: list[tuple[int, float]] = []
    if design.edge_row:
        nc_hold += [(n, design.nc_holdout_tokens[0]) for n in TRAIN_SIZES]
        co_hold += [(n, design.co_holdout_tpp[0]) for n in TRAIN_SIZES]
    nc_hold += [(pivot, d) for d in design.nc_tokens + design.nc_holdout_tokens]
    co_hold += [(pivot, k) for k in design.co_tpp + design.co_holdout_tpp]
    nc_hold += [(n, d) for n in largest for d in design.nc_holdout_tokens]
    co_hold += [(n, k) for n in largest for k in design.co_holdout_tpp]
    runs += [Run("holdout", n, d, None) for n, d in nc_hold]
    runs += [Run("colinear_holdout", n, int(n * k), k) for n, k in co_hold]
    return runs


def run_name(dataset: str, design: Design, bf16: bool, run: Run) -> str:
    stem = ("BF16" if bf16 else "") + design.prefix + NAME_PREFIX[dataset]
    tag = {"nxd": "NxD_GRID", "colinear": "COLINEAR", "holdout": "HOLDOUT",
           "colinear_holdout": "COLINEAR_HOLDOUT"}[run.kind]
    if run.tpp is None:
        return f"{stem}_{tag}_N{_fmt_m(run.n)}_D{_fmt_m(run.d)}"
    return f"{stem}_{tag}_N{run.n / 1e6:.2f}M_TPP{_fmt_tpp(run.tpp)}_D{_fmt_m(run.d, 1)}"


HEADER = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --time={time}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node={gpus}
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
{optional}#SBATCH --output={log_dir}/{job_name}_%j.out
#SBATCH --error={log_dir}/{job_name}_%j.err
set -eo pipefail

export DATA_DIR={data_dir}
cd {work_dir}
source {venv}/bin/activate
"""

EXPERIMENT = """
echo "[{i}/{total}] {name}"
torchrun --nproc_per_node={gpus} --standalone \\
    src/training_models/training/dist_trainer_main.py \\
    --model-size {n} \\
    --tokens {d} \\
    --learning-rate {lr} \\
    --weight-decay {wd} \\
    --epochs {epochs} \\
    --batch-size {batch_size} \\
    --max-seq-len {seq_len} \\
    --dataset {dataset} \\
    --architecture {arch} \\
    {mode_flag}{bf16_flag}--output-prefix {name}
"""


def write_scripts(args: argparse.Namespace) -> list[Path]:
    design = DESIGNS[args.design]
    bf16 = args.precision == "bf16"
    runs = expand(design)
    work_dir = args.work_dir or str(REPO_ROOT)
    data_dir = args.data_dir or os.path.join(work_dir, "corpora")
    venv = args.venv or os.path.join(work_dir, ".venv")
    lr = LEARNING_RATES[args.dataset] if args.learning_rate is None else args.learning_rate

    optional = ""
    for flag, value in (("partition", args.partition), ("account", args.account),
                        ("exclude", args.exclude)):
        if value:
            optional += f"#SBATCH --{flag}={value}\n"

    out_dir = Path(args.out_dir) / f"{args.dataset}_{args.design}_{args.precision}"
    out_dir.mkdir(parents=True, exist_ok=True)
    scripts: list[Path] = []
    for b, lo in enumerate(range(0, len(runs), args.runs_per_job), 1):
        chunk = runs[lo:lo + args.runs_per_job]
        job_name = f"{args.dataset}_{args.design}_{args.precision}_{b:03d}"
        body = HEADER.format(job_name=job_name, time=args.time, gpus=args.gpus,
                             optional=optional, log_dir=args.log_dir, data_dir=data_dir,
                             work_dir=work_dir, venv=venv)
        for i, run in enumerate(chunk, 1):
            holdout = run.kind in ("holdout", "colinear_holdout")
            body += EXPERIMENT.format(
                i=i, total=len(chunk), name=run_name(args.dataset, design, bf16, run),
                gpus=args.gpus, n=run.n, d=run.d, lr=lr, wd=WEIGHT_DECAY,
                epochs=design.epochs, batch_size=BATCH_SIZE, seq_len=SEQ_LEN,
                dataset=args.dataset, arch=ARCH,
                mode_flag="--use-train-as-val \\\n    " if holdout else "--skip-val \\\n    ",
                bf16_flag="--use-bf16 \\\n    " if bf16 else "",
            )
        path = out_dir / f"{job_name}.sh"
        path.write_text(body)
        path.chmod(0o755)
        scripts.append(path)

    submit = out_dir / "submit_all.sh"
    submit.write_text("#!/bin/bash\nset -euo pipefail\n"
                      f"mkdir -p {args.log_dir}\n"
                      + "".join(f"sbatch {p}\n" for p in scripts))
    submit.chmod(0o755)
    print(f"{args.dataset} / {args.design} / {args.precision}: {len(runs)} runs "
          f"in {len(scripts)} jobs -> {out_dir}/ (submit with {submit})")
    return scripts


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    env = os.environ.get
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, choices=sorted(LEARNING_RATES))
    p.add_argument("--design", default="main", choices=sorted(DESIGNS))
    p.add_argument("--precision", default="fp32", choices=["fp32", "bf16"],
                   help="bf16 adds --use-bf16 (the paper's BF16 replication is Wikipedia, main design)")
    p.add_argument("--learning-rate", type=float, default=None,
                   help="override the per-corpus peak learning rate")
    p.add_argument("--list", action="store_true", help="print the runs and exit")
    p.add_argument("--runs-per-job", type=int, default=7)
    p.add_argument("--gpus", type=int, default=4,
                   help="GPUs per run (the paper used 4; the global batch is 96 x GPUs)")
    p.add_argument("--partition", default=env("SLURM_PARTITION", ""))
    p.add_argument("--account", default=env("SLURM_ACCOUNT", ""))
    p.add_argument("--exclude", default=env("SLURM_EXCLUDE", ""), help="nodes to avoid")
    p.add_argument("--time", default=env("SLURM_TIME", "08:00:00"))
    p.add_argument("--work-dir", default=env("WORK_DIR"))
    p.add_argument("--data-dir", default=env("DATA_DIR"))
    p.add_argument("--venv", default=env("VENV"))
    p.add_argument("--out-dir", default="slurm_scripts")
    p.add_argument("--log-dir", default="slurm_logs")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.list:
        design = DESIGNS[args.design]
        for run in expand(design):
            print(f"{run.kind:17s} N={run.n:>11,d} D={run.d:>14,d} "
                  f"{run_name(args.dataset, design, args.precision == 'bf16', run)}")
        return
    write_scripts(args)


if __name__ == "__main__":
    main()
