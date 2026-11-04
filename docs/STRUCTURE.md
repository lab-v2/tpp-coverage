# Repository structure

## Structure

| Path | What it holds |
|---|---|
| `src/` | The code that trained the models and analysed them |
| `large_scale/` | The validation of the paper's analysis on an independent, larger grid (461 C4 runs, up to 354M non-embedding parameters): package, scripts, tests, results and report |
| `cluster/` | The SLURM task queue and re-analyses of the larger Wikipedia stratum (WIKI300M) trained on the paper's protocol |
| `camera_ready/` | The published-design classification and the public-checkpoint reanalysis added for the camera-ready version |
| `docs/` | Step-by-step guides: downloading, training, analysis, figures, n-gram KL |
| `tests/` | The pytest suite (`uv run pytest -n auto`) |
| `images/` | Logos shown in the README |
