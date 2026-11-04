#!/usr/bin/env python3
"""Exit 0 only when both seed-42 training arms hold all 66 DISTINCT (N,D) cells.

Counting *files* overcounts: copies collected from other machines land in the same
directory, which is how the earlier trigger fired at 36/35 while reporting
108/66. Dedup on (m, token) parsed from the filename instead.
"""
import glob
import os
import re
import sys

# The metrics directory: the first argument, else $WIKI300M_DIR/shared/training_metrics.
METRICS = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.environ.get("WIKI300M_DIR", "."), "shared", "training_metrics")

co, nc = set(), set()
for f in glob.glob(os.path.join(METRICS, '*_metrics.npz')):
    b = os.path.basename(f)
    if 'SEED' in b or 'HOLDOUT' in b:
        continue
    m = re.search(r'_m(\d+)_token(\d+)_', b)
    if not m:
        continue
    key = (int(m.group(1)), int(m.group(2)))
    if 'COLINEAR_N' in b:
        co.add(key)
    elif 'NxD_GRID' in b:
        nc.add(key)
print(f"CO {len(co)}/66  NC {len(nc)}/66")
sys.exit(0 if (len(co) >= 66 and len(nc) >= 66) else 1)
