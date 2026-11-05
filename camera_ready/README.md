# Camera-ready analyses

[README](../README.md)

The two analyses added for the camera-ready version of the paper. Each folder is self-contained:
a README with how to run it, a pre-registered protocol (`CRITERIA.md`), its scripts and its
committed outputs.

| Folder | What it holds |
|---|---|
| [`grid_classification/`](grid_classification/README.md) | The paper's design criterion applied to every published scaling-law design whose tokens-per-parameter ratios can be recovered (24 designs) |
| [`public_checkpoints/`](public_checkpoints/README.md) | The criterion tested on ten public training-loss releases: collinear versus non-collinear subsets, fitted and scored off their TPP range |

Their tests are in `tests/camera_ready/`: `uv run pytest tests/camera_ready -n auto`.
