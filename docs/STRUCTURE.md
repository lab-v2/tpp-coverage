# Repository structure

## Structure

| Path | What it holds |
|---|---|
| `src/` | The code that trained the models and analysed them |
| `camera_ready/` | The published-design classification and the public-checkpoint reanalysis added for the camera-ready version |
| `docs/` | Step-by-step guides: downloading, training, analysis, figures, n-gram KL |
| `tests/` | The pytest suite (`uv run pytest -n auto`); `tests/ci/` checks the CI workflows and the merge rule, `tests/public_checkpoints/` the public-checkpoint reanalysis |
| `.github/` | CI on the project's self-hosted runners: tests, `ruff check .`, the privacy scan, the `pr-authorization` merge rule and the runner guard |
| `images/` | Logos shown in the README |
