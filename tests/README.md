# Tests

[README](../README.md)

The pytest suite: `uv run pytest -n auto` from the repository root. Each folder tests the
component of the same name; `conftest.py` holds the shared paths and helpers.

| Folder | Tests |
|---|---|
| `tpp_analysis/` | `src/tpp_analysis/`: law recovery on synthetic data, the Skaling law, the job generator against the released runs, the anonymous download |
| `training_models/` | `src/training_models/`: the trainer's loss masking, gradient accumulation, optimizers and schedules |
| `large_scale/` | `large_scale/` and `src/tpp_large_scale/`: correctness, determinism, the pipeline |
| `camera_ready/` | `camera_ready/`: both analyses rerun against their committed outputs |
| `cluster/` | `cluster/`: the task generator, the queue scripts and the audit |
| `ci/` | `.github/`: the workflows and the merge rule |
| `repository/` | The whole tree: README, links, entry points, vocabulary, privacy |

Tests marked `network` download public data from HuggingFace; `slow` ones run the full grid.
