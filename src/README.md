# Packages

[README](../README.md)

The importable Python code. Scripts inside each package are run by path from the repository
root, in the one environment `uv sync` creates.

| Package | What it holds | Guide |
|---|---|---|
| [`tpp_analysis/`](tpp_analysis) | The paper's analysis: corpus downloaders, job generation, loss extraction, the four scaling laws and their fits, the tables and the figures | [ANALYSIS](../docs/ANALYSIS.md), [VISUALIZATIONS](../docs/VISUALIZATIONS.md) |
| [`training_models/`](training_models) | Llama-style models, the distributed trainer and its reproducibility check | [TRAINING](../docs/TRAINING.md) |
| [`tpp_large_scale/`](tpp_large_scale) | The analysis of the independent 461-run C4 grid; it imports the paper's laws from `tpp_analysis` | [large_scale](../large_scale/README.md) |

```bash
uv run python src/tpp_analysis/run_analysis.py all --data-dir extracted_losses/ --n-workers "$(nproc)"
uv run python src/training_models/training/dist_trainer_main.py --help
uv run python large_scale/scripts/run_analysis.py --help
```

Their tests are `tests/tpp_analysis/`, `tests/training_models/` and `tests/large_scale/`.
