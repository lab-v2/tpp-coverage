"""
Parse the custom epoch-results CSV format used by the training pipeline.

Each run block looks like:
    N=3145728, D=8388608
    run_type=nxd_grid,is_holdout=False,is_colinear=False
    Epoch,Train Loss
    1,4.321
    2,3.987
    3,3.654
"""

from typing import List, Dict


def parse_epoch_results(filepath: str) -> List[Dict]:
    """
    Parse the custom CSV with multiple run blocks.

    Returns list of dicts:
        {N, D, run_type, is_holdout, is_colinear, epoch_losses, n_epochs}
    """
    with open(filepath, "r") as f:
        content = f.read()

    blocks = content.strip().split("\n\n")
    runs: List[Dict] = []

    for block in blocks:
        if not block.strip():
            continue

        lines = block.strip().split("\n")
        if len(lines) < 4:
            continue

        # --- Header: N=..., D=... ---
        header = lines[0]
        try:
            parts = header.split(",")
            N = int(parts[0].split("=")[1].strip())
            D = int(parts[1].split("=")[1].strip())
        except (IndexError, ValueError):
            continue

        # --- Metadata: run_type=...,is_holdout=...,is_colinear=... ---
        meta = lines[1]
        meta_dict = {}
        for p in meta.split(","):
            if "=" in p:
                k, v = p.split("=", 1)
                meta_dict[k.strip()] = v.strip()

        run_type = meta_dict.get("run_type", "unknown")
        is_holdout = meta_dict.get("is_holdout", "False") == "True"
        is_colinear = meta_dict.get("is_colinear", "False") == "True"

        # --- Epoch losses (skip header row) ---
        epoch_losses = []
        for line in lines[3:]:
            if not line.strip():
                continue
            parts = line.split(",")
            if len(parts) >= 2:
                try:
                    epoch_losses.append(float(parts[1]))
                except ValueError:
                    continue

        if epoch_losses:
            runs.append({
                "N": N,
                "D": D,
                "run_type": run_type,
                "is_holdout": is_holdout,
                "is_colinear": is_colinear,
                "epoch_losses": epoch_losses,
                "n_epochs": len(epoch_losses),
            })

    return runs
