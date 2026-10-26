"""Where the public-checkpoint reanalysis reads its inputs and writes its outputs.

WORK holds everything regenerated (run tables, fits, estimates); set TPP_PC_WORK to move it.
RESULTS holds the committed reference outputs that the tests compare against.
"""

from __future__ import annotations

import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORK = Path(os.environ.get("TPP_PC_WORK", HERE / "work")).resolve()
ARTIFACTS = WORK / "artifacts"
RESULTS = HERE / "results"
REPO_SRC = HERE.parents[1] / "src" / "tpp_analysis"

# Normalised run tables published with the paper (anonymous read access).
HF_REPO_ID = "TPPIsCritical/public_checkpoint_reanalysis"
