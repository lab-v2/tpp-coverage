#!/usr/bin/env bash
# Build figures/surface_co_vs_nc.png: dual Chinchilla surface (CO vs NC) via the
# paper's surface code (scripts/make_surface_co_nc.py).
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"

: "${TPP_CODE_REPO:=$(cd "$REPO/.." && pwd)}"   # the paper's code: this repository
export TPP_CODE_REPO
export MPLBACKEND=Agg

cd "$REPO"
uv run python scripts/make_surface_co_nc.py
