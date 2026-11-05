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

`tests`-free assurance: `verify_upstream_parity()` in `src/tpp_large_scale/model/fitting.py` fits all three laws twice,
once through the paper's `fit_*` entry points and once through this package, and asserts the
sums of squared errors are bit-identical. It runs at the start of every analysis and is also
available on its own via `uv run python large_scale/scripts/run_analysis.py --check-only`.
It guards the one transcription in this package: upstream defines each law's bounds and
initial-guess sampler as function locals of `fit_chinchilla`, `fit_kaplan` and
`fit_droppo_elibol`, so `fitting.py` copies them verbatim.

## What is NOT reused, and why

### Conditioning

The paper's code has no Jacobian, condition number or covariance machinery (`kappa`,
`kappa_{A,B}`, the column-scaled variant and the CI-inflation ratio are analytic in the paper),
so `src/tpp_large_scale/model/conditioning.py` implements them here. `full_coverage_ci.py`,
which fits the paper's seed-variance confidence intervals, is not used.

### TPP band snapping

`incremental_coverage.coverage.snap_to_nearest` snaps a TPP to the nearest bin in linear
distance:

```python
return min(bins, key=lambda b: abs(b - value))
```

The design-TPP bands here are quarter-decade spaced (20, 37, 67, 120, 215), and on such bands
the two rules disagree near every band midpoint: TPP 90 snaps to 67 linearly but to 120 in log
space. Using the upstream function here would silently reassign cells of
`results/envelope.csv` and move every downstream ray, holdout and robustness number, so
`model.grid.LogBandAssigner` snaps in log distance instead. In the paper's own subset analysis
the linear rule is the definition behind the published numbers, so it stays there; anyone
reusing `snap_to_nearest` on log-spaced bins should snap in log distance as this package does.

### Fits without differential evolution

Upstream `_fit_generic` always follows its multi-start L-BFGS-B with a differential-evolution
search (the paper's "global fallback") and keeps the better fit, with no way to skip it. The
resampling experiments here run thousands of fits, so they use `LawFitter(global_search=False)`,
the same multi-start without the differential-evolution step: every CO-versus-NC comparison
trial (the holdout win rate and the robustness checks that resample it), every conditioning
trial, and the subset fits of the formal-statement validation. Their numbers in
`results/results.json` are therefore not exactly the paper fitter's, though a fit whose
multi-start already reaches the best optimum is identical. The full-grid fits, the ray fits, the
other robustness fits and the README's pair sweep keep the differential-evolution step.

## Environment

This module runs in the repository's single environment (Python 3.10, `uv sync`), beside the
paper's code. Only numpy, scipy, pandas, scikit-learn and matplotlib are imported; none of the
training stack is needed. Set `TPP_CODE_REPO` to another checkout of this repository to run
against that copy of the paper's code.
