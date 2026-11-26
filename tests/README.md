# Tests

[README](../README.md)

The pytest suite: `uv run pytest -n auto` from the repository root. Each folder tests the
component of the same name; `conftest.py` holds the shared paths and helpers.

| Folder | Tests |
|---|---|
| `tpp_analysis/` | `src/tpp_analysis/`: law recovery on synthetic data, the Skaling law, the job generator against the released runs, the anonymous download, and the paper's main analysis refit from scratch against the released results |
| `training_models/` | `src/training_models/`: the trainer's loss masking, gradient accumulation, optimizers and schedules |
| `large_scale/` | `large_scale/` and `src/tpp_large_scale/`: correctness, determinism, the pipeline |
| `camera_ready/` | `camera_ready/`: both analyses rerun against their committed outputs |
| `cluster/` | `cluster/`: the task generator, the queue scripts and the audit |
| `ci/` | `.github/`: the workflows and the merge rule |
| `website/` | `website/`: one section per talk slide, showing and citing only what the deck does |
| `repository/` | The whole tree: README, links, entry points, vocabulary, privacy |

The suite checks law recovery on synthetic data, every entry point's `--help`, that the
released tables are reproduced, and the camera-ready analyses. It also checks the README's
links, images, diagram and citation, and that no private host or path is in the tree.

CI runs the suite, the hooks of `.pre-commit-config.yaml`, `ruff check .` and the privacy
scan on every pull request and every push to `main`. A pull request may merge only when its author is a lab-v2 member, or when a
member approved its latest commit (the `pr-authorization` check).

Tests marked `network` download public data from HuggingFace; `slow` ones run the full grid.
The `paper` test refits the paper's main analysis on every thread, so the default run leaves it out
and it runs alone: `uv run pytest -n 0 -m paper` (CI runs both).

`uv run python tests/measure_coverage.py <out-dir>` measures line and branch coverage of the
suite and the paper gate together, following pool workers and subprocesses, and writes
`coverage-<sha>-with-gate.json` stamped with the commit (a clean tree only; about 8 minutes).
