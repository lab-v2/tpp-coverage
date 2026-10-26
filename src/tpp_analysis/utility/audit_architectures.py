#!/usr/bin/env python3
"""
Audit architectures across all training-run checkpoints.

For each .pt checkpoint under --root:
  1. Read model_config (the architecture that was actually built).
  2. Sum state_dict.numel() (independent verification of param count).
  3. Recompute the expected param count from (d_model, n_layers, d_ff,
     vocab_size) using the Llama+RoPE+SwiGLU formula and check it matches
     the actual state_dict. This is a per-checkpoint internal-consistency
     sanity check - the config and the weights should agree.

Outputs:
  --out  (default architecture_audit.csv)
        One row per checkpoint with the recorded architecture and the
        true parameter count from the state_dict.

Run inside tmux for backgrounding:
  tmux new -s audit
  python audit_architectures.py /path/to/checkpoints/
  # detach: Ctrl-b d
  # reattach: tmux attach -t audit

To restrict to final-epoch checkpoints (recommended - usually only the
final epoch is the artifact of record):
  python audit_architectures.py /path/to/checkpoints/ --only-epoch 3
"""

import sys
import csv
import argparse
from pathlib import Path

import torch
from tqdm import tqdm


# ============================================================
# Llama param-count formula (RoPE, SwiGLU, RMSNorm, no bias)
# ============================================================

def llama_param_count(d_model, n_layers, d_ff, vocab_size):
    """Parameter count for a Llama-style transformer with rotary position
    embeddings (no learned positional embeddings), SwiGLU FFN (3 weight
    matrices: gate_proj, up_proj, down_proj), and RMSNorm (1 weight, no bias).
    Matches state_dict.numel() exactly."""
    embed_params = vocab_size * d_model
    attn_params = 4 * d_model * d_model           # Q, K, V, O - no bias
    ff_params = d_model * d_ff * 3                # SwiGLU: gate, up, down
    ln_params = 2 * d_model                       # 2 RMSNorm per layer
    layer_params = attn_params + ff_params + ln_params
    transformer_params = n_layers * layer_params
    final_params = d_model + vocab_size * d_model  # final RMSNorm + LM head
    return embed_params + transformer_params + final_params


# ============================================================
# Checkpoint inspection
# ============================================================

def inspect_checkpoint(path: Path):
    """Read architecture from a .pt checkpoint without GPU."""
    try:
        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    except Exception as e:
        return {"error": f"load_failed: {type(e).__name__}: {str(e)[:200]}"}

    cfg = ckpt.get("model_config", {})
    if not cfg:
        return {"error": "no_model_config"}

    sd = ckpt.get("model_state_dict", {})
    if not sd:
        return {"error": "no_state_dict"}

    actual_params = sum(v.numel() for v in sd.values())

    return {
        "d_model": cfg.get("d_model"),
        "n_layers": cfg.get("n_layers"),
        "n_heads": cfg.get("n_heads"),
        "d_ff": cfg.get("d_ff"),
        "vocab_size": cfg.get("vocab_size"),
        "max_seq_len": cfg.get("max_seq_len"),
        "actual_total_params": actual_params,
        "epoch": ckpt.get("epoch"),
        "train_loss": ckpt.get("train_loss"),
        "seed": ckpt.get("seed"),
    }


# ============================================================
# Walk + audit
# ============================================================

def find_checkpoints(root: Path, only_epoch=None):
    """Yield all .pt files under root."""
    for p in root.rglob("*.pt"):
        if only_epoch is not None and f"_epoch{only_epoch}" not in p.name:
            continue
        yield p


def categorize_path(path: Path, root: Path):
    """Extract dataset / design / N folder structure from path.
    Expected layout: root / <dataset> / <design> / N_<count> / <file>.pt"""
    try:
        rel = path.relative_to(root)
        parts = rel.parts
        if len(parts) >= 4:
            return {
                "dataset": parts[0],
                "design": parts[1],
                "N_folder": parts[2],
                "filename": parts[-1],
            }
        return {"dataset": "?", "design": "?", "N_folder": "?",
                "filename": path.name}
    except ValueError:
        return {"dataset": "?", "design": "?", "N_folder": "?",
                "filename": path.name}


def audit_checkpoint(path: Path, root: Path):
    """Inspect one checkpoint and check internal consistency."""
    cat = categorize_path(path, root)
    info = inspect_checkpoint(path)

    base = {
        "path": str(path),
        "dataset": cat["dataset"],
        "design": cat["design"],
        "N_folder": cat["N_folder"],
        "filename": cat["filename"],
    }

    if "error" in info:
        return {**base, "error": info["error"], "internally_consistent": None}

    # Internal-consistency check: does the recorded architecture predict
    # the actual state_dict.numel()? If not, config and weights disagree.
    d_model = info["d_model"]
    n_layers = info["n_layers"]
    d_ff = info["d_ff"]
    vocab_size = info["vocab_size"]
    actual_params = info["actual_total_params"]

    if None in (d_model, n_layers, d_ff, vocab_size):
        return {**base, "error": "missing_config_field",
                "internally_consistent": None, **info}

    predicted_params = llama_param_count(d_model, n_layers, d_ff, vocab_size)
    consistent = (predicted_params == actual_params)

    return {
        **base,
        "error": "" if consistent else "config_state_dict_mismatch",
        "d_model": info["d_model"],
        "n_layers": info["n_layers"],
        "n_heads": info["n_heads"],
        "d_ff": info["d_ff"],
        "head_dim": (info["d_model"] // info["n_heads"]
                     if info["n_heads"] else None),
        "vocab_size": vocab_size,
        "max_seq_len": info["max_seq_len"],
        "actual_total_params": actual_params,
        "predicted_total_params": predicted_params,
        "internally_consistent": consistent,
        "epoch": info["epoch"],
        "train_loss": info["train_loss"],
        "seed": info["seed"],
    }


# ============================================================
# CSV output
# ============================================================

AUDIT_COLS = [
    "path", "dataset", "design", "N_folder", "filename",
    "internally_consistent", "error",
    "d_model", "n_layers", "n_heads", "d_ff", "head_dim",
    "actual_total_params", "predicted_total_params",
    "vocab_size", "max_seq_len",
    "epoch", "train_loss", "seed",
]


def write_audit(rows, out_path):
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=AUDIT_COLS, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Audit architectures across training-run checkpoints",
    )
    parser.add_argument("root", type=Path,
                        help="Root directory containing .pt checkpoints")
    parser.add_argument("--out", type=Path,
                        default=Path("architecture_audit.csv"),
                        help="Per-checkpoint audit CSV")
    parser.add_argument("--only-epoch", type=int, default=None,
                        help="Only audit checkpoints with _epoch{N} in filename "
                             "(e.g. --only-epoch 3 for final-epoch only)")
    args = parser.parse_args()

    if not args.root.exists():
        sys.exit(f"Root does not exist: {args.root}")

    print(f"Scanning: {args.root}")
    if args.only_epoch is not None:
        print(f"Filtering: only _epoch{args.only_epoch} files")
    paths = list(find_checkpoints(args.root, only_epoch=args.only_epoch))
    print(f"Found {len(paths)} checkpoint(s)")
    if not paths:
        sys.exit("Nothing to audit.")

    rows = []
    n_inconsistent = 0
    n_error = 0

    for path in tqdm(paths, desc="Auditing", unit="ckpt"):
        row = audit_checkpoint(path, args.root)
        rows.append(row)
        if row.get("error"):
            n_error += 1
        elif row.get("internally_consistent") is False:
            n_inconsistent += 1

    write_audit(rows, args.out)

    print()
    print("=" * 60)
    print(f"Audited:                {len(rows)}")
    print(f"Internally inconsistent: {n_inconsistent}")
    print(f"Errors:                  {n_error}")
    print("=" * 60)
    print(f"Per-checkpoint audit: {args.out}")

    if n_inconsistent > 0:
        print()
        print("Inconsistent checkpoints (config does not predict state_dict size):")
        shown = 0
        for r in rows:
            if r.get("internally_consistent") is False:
                print(f"  {r['dataset']}/{r['design']}/{r['N_folder']}/"
                      f"{r['filename']}")
                print(f"    config: d_model={r['d_model']}, n_layers={r['n_layers']}, "
                      f"n_heads={r['n_heads']}, d_ff={r['d_ff']}")
                print(f"    actual params:    {r['actual_total_params']:,}")
                print(f"    predicted params: {r['predicted_total_params']:,}")
                shown += 1
                if shown >= 20:
                    print(f"  ... and {n_inconsistent - 20} more (see CSV)")
                    break


if __name__ == "__main__":
    main()
