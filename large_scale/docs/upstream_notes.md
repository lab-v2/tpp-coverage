# Notes on reusing the companion paper's analysis code

This analysis imports the paper's own implementation rather than reimplementing it.
`src/tpp_large_scale/upstream.py` is the single choke point: every borrowed symbol is
re-exported there and nowhere else, so the "what did we reuse?" question is answered by
reading one file.

## What is reused

| Concern | Upstream symbol |
| --- | --- |
| Chinchilla / Kaplan / Droppo-Elibol functional forms | `scaling_laws.laws.{chinchilla,kaplan,droppo_elibol}_loss` |
| Multi-start L-BFGS-B + differential-evolution fitter | `scaling_laws.laws._fit_generic` |
| Per-law fitters (bounds and samplers baked in) | `scaling_laws.laws.fit_{chinchilla,kaplan,droppo_elibol}` |
| Train/holdout R2 and RMSE | `scaling_laws.laws.evaluate` |
| Law registry and dataclass | `scaling_laws.laws.LAWS`, `ScalingLaw` |
| TPP diversity statistic V_K | `subset_analysis.subset_analysis_bbox_seeded.compute_VK` |
| Diversity threshold tau_K | `subset_analysis.subset_analysis_bbox_seeded.compute_tauK` |
| Regime A/B decision | `subset_analysis.subset_analysis_bbox_seeded.classify_regime` |
| Restart count, seed base, kappa targets, beta index, min points | `subset_analysis.subset_analysis_bbox_seeded` constants |
| Train/holdout bucketing (`organise_runs`, `PreparedData`) | `scaling_laws.data._organise` / `PreparedData` |

`tests`-free assurance: `verify_upstream_parity()` in `model/fitting.py` fits all three laws twice,
once through the paper's `fit_*` entry points and once through this package, and asserts the
sums of squared errors are bit-identical. It runs at the start of every analysis and is also
available on its own via `uv run python large_scale/scripts/run_analysis.py --check-only`.

## What is NOT reused, and why

### The Gauss-Newton conditioning objects have no upstream implementation

A repo-wide search of the paper's code package finds no Jacobian construction, no condition
number, and no covariance/CI machinery. `kappa`, `kappa_{A,B}`, the column-scaled variant and
the CI-inflation ratio are defined analytically in the paper's tables only. Everything in
`model/conditioning.py` is therefore local by necessity, not by choice.

### `full_coverage_ci.py` is not needed

It fits the paper's seed-variance confidence intervals, which this analysis does not use. In the
code package as first released it did not import (`sys.path.insert(...)` above
`from __future__ import annotations`, and `os` used before its import); in this repository the
import order is fixed and it imports.

### `incremental_coverage.coverage.snap_to_nearest` snaps on the wrong metric

```python
return min(bins, key=lambda b: abs(b - value))
```

That is linear distance. The design-TPP bands here are quarter-decade spaced
(20, 37, 67, 120, 215), so the correct nearest-bin rule is log distance. The two disagree
near every band midpoint: TPP 90 snaps to 67 linearly but to 120 in log space. Using the
upstream function would silently reassign envelope cells and move every downstream ray,
holdout and robustness number. `model.grid.LogBandAssigner` implements the log rule instead.

### Bounds and samplers are transcribed, not imported

Upstream defines each law's bounds and initial-guess sampler as *function locals* inside
`fit_chinchilla` / `fit_kaplan` / `fit_droppo_elibol`, so they cannot be imported. They are
transcribed verbatim into `model/fitting.py`, and `verify_upstream_parity()` exists precisely to
catch any future drift in that transcription.

### The differential-evolution opt-out

Upstream `_fit_generic` always runs differential evolution after the multi-start phase and
exposes no way to skip it. The resampling loops here run thousands of fits, so
`LawFitter(global_search=False)` uses a DE-free multi-start that is otherwise identical to
upstream's. This was verified to reproduce the previous script's `use_de=False` path to
within 1e-12 in sum of squared errors.

## Environment

This module runs in the repository's single environment (Python 3.10, `uv sync`), beside the
paper's code. Only numpy, scipy, pandas, scikit-learn and matplotlib are imported; none of the
training stack is needed. Set `TPP_CODE_REPO` to another checkout of this repository to run
against that copy of the paper's code.
