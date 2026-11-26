# Tests

[README](../README.md)

The pytest suite: `uv run pytest -n auto` from the repository root. Each folder tests the
component of the same name; `conftest.py` holds the shared paths and helpers.

| Folder | Tests |
|---|---|
| `tpp_analysis/` | `src/tpp_analysis/`: law recovery on synthetic data, the Skaling law, the job generator against the released runs, the anonymous download, the paper's main analysis refit from scratch against the released results, and characterization tests of the analysis modules (subfolders mirror `src/tpp_analysis/`; `scripts/` holds its top-level scripts) |
| `training_models/` | `src/training_models/`: the trainer's loss masking, gradient accumulation, optimizers and schedules, and characterization tests (subfolders mirror `src/training_models/`; `trainer_run/` trains on a GPU) |
| `large_scale/` | `large_scale/` and `src/tpp_large_scale/`: correctness, determinism, the pipeline |
| `camera_ready/` | `camera_ready/`: both analyses rerun against their committed outputs |
| `cluster/` | `cluster/`: the task generator, the queue scripts and the audit |
| `ci/` | `.github/`: the workflows and the merge rule; `affected.py`, the selector of the tests a pull request runs |
| `website/` | `website/`: one section per talk slide, each bullet the deck's item word for word (markup, punctuation, case, number words and citation form aside) with its numbers, formulas, citations and names; references, authors, BibTeX, arXiv id, abstract and the Code and Data links as the deck, `CITATION.cff` and the README give them; the summary pinned in the test; the header logos above and below the title as on the title slide, the NeurIPS one a banner at least 300 px wide, Syracuse only through its AI Institute logo; every image present; each figure sized so its tick labels match the body text (measured from the PNG, `test_figure_scale.py`) |
| `repository/` | The whole tree: README, links, entry points, vocabulary, privacy |

The suite checks law recovery on synthetic data, every entry point's `--help`, that the
released tables are reproduced, the camera-ready analyses, and that the analysis modules still
return, write and draw what the base commit did. It also checks the README's
links, images, diagram (a PNG rendered from `images/pipeline/pipeline.mmd`, stamped with its hash) and citation, and that no private host, path, or path to a tool only the owner's machines hold, is in the tree.

CI runs the hooks of `.pre-commit-config.yaml`, `ruff check .` and the privacy scan on every
pull request and every push to `main`. A push to `main` runs the whole suite and the paper gate; a
pull request runs only the test files its change can affect, and the paper gate only when its
inputs changed (see "Which tests a pull request runs" below). A pull request may merge only when its author is a lab-v2 member, or when a
member approved its latest commit (the `pr-authorization` check).

The characterization tests (`tpp_analysis/**/test_characterize_*.py`, helpers in
`characterize.py`) run the analysis modules on the released inputs (pinned to one HuggingFace
revision) and compare their tables, CSVs, fits and figure data with the base commit's outputs,
frozen in `frozen/` beside each test. They pin behaviour for refactors; `TPP_FREEZE=1` refreezes
them, on the base commit only. Figures are compared by their data (including every line's and collection's zorder and alpha), never by image bytes.

The `training_models/` ones compare what the code returns and writes with the
base commit's outputs, frozen in `frozen/` beside each test (helpers in `characterize.py`;
`TPP_FREEZE=1` refreezes them, on the base commit only). `models/`, `core/`, `utils/`, `distributed/`
and `training/` run on the CPU: seeded models on the LICENSE's text, the pre-tokenized dataset and
loaders, the checkpoint and metrics files and the resume helpers. `trainer_run/` runs
`dist_trainer_main.py` itself, which needs CUDA and nccl: three two-epoch runs of a one-layer
model on the first 64 rows of the real Wikipedia corpus, compared by checkpoint tensors, loss
arrays and metadata (found by glob, never by file bytes or names, which carry time stamps).

Tests marked `network` download public data from HuggingFace; `slow` ones run the full grid.
Tests marked `gpu` are skipped where torch sees no CUDA device, as on the CI hosts. On a GPU host,
with the CUDA build of the locked torch (`uv pip install --reinstall --extra-index-url
https://download.pytorch.org/whl/cu124 torch==2.5.1` after `uv sync --locked`; the locked aarch64
wheel is CPU-only) and `DATA_DIR` holding `wikipedia/{train,val}_{tokens,lengths}.bin`, run them
alone (about 30 s on a GH200; their frozen values were taken on a GH200 with torch 2.5.1+cu124).
Use the environment's interpreter, `DATA_DIR=<corpora> .venv/bin/python -m pytest -n 0 -m gpu`, not `uv run`:
`uv run` first syncs the environment to the lock, which puts the CPU-only torch back.
The `paper` test refits the paper's main analysis on every thread, so the default run leaves it out
and it runs alone: `uv run pytest -n 0 -m paper`. CI runs it as a job of its own, `paper`,
beside the suite's `tests` job: on every push to `main`, and on a pull request that changes its
inputs. Its table step alone,
`run_analysis.build_paper_tables` on the released per-seed fits, is checked byte for byte against
the released tables in the default run (`tpp_analysis/scripts/test_characterize_run_analysis_tables.py`).

`uv run python tests/measure_coverage.py <out-dir>` measures line and branch coverage of the
suite and the paper gate together, following pool workers and subprocesses, and writes
`coverage-<sha>-with-gate.json` stamped with the commit (a clean tree only; about 8 minutes).

## Which tests a pull request runs

`tests/affected.py` holds the map and the selector. On a push to `main`, the `tests` job runs the
whole suite (`pytest -n auto`) and then `tests/affected.py build --part suite`: the whole suite
again, each test file in its own pytest process (all at once); the `paper` job runs the paper gate
as `tests/affected.py build --part paper`. Each process tree runs under `strace -f` (the CI hosts
have it). A test file's inputs are the
repository files it and every process it starts (pool workers, scripts, git) opened or executed
(cached bytecode counts as its source); a path it only probed for (a stat, a failed open) and a
directory it listed count only when a change creates or removes something there. Each part of the map, with the commit it was built on, is kept as an
artifact of the run (`affected-map`, `affected-map-paper`), and only when every test passed and every trace shows its test file
reading itself and `tests/conftest.py`.

On a pull request, both jobs run `tests/affected.py fetch`, which downloads and joins the two parts
from the newest successful push run on `main` at an ancestor of the base, and `tests/affected.py select` diffs its commit against the pull request's head. It runs
a test file when its inputs or its own file changed, when the change creates or removes a path it
probed for or an entry of a directory it listed, or when it reads the git directory (`git
ls-files`); the paper gate runs when its inputs changed or a changed test file holds a paper test.
It runs the whole suite instead when the map is missing or incomplete, its commit is not an
ancestor of the pull request's base, or a shared input changed: a `conftest.py`, `pyproject.toml`,
the pytest configuration, `uv.lock`, a workflow or `tests/affected.py` itself.
`tests/ci/test_affected.py` tests those choices.
