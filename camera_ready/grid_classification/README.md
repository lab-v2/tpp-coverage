# Classifying published scaling-law designs

[README](../../README.md)

Applies the paper's design criterion (Proposition 2: a design is well conditioned, Regime B,
when V_K >= tau_K) to every published bivariate L(N, D) scaling-law design whose list of
tokens-per-parameter ratios can be recovered. Selection was fixed before any design was
classified; the protocol is in [`CRITERIA.md`](CRITERIA.md).

```bash
uv run python camera_ready/grid_classification/classify.py           # rewrites results.csv
uv run python camera_ready/grid_classification/classify.py --out /tmp/check.csv
```

Result, at each paper's own fitted data exponent and kappa_target = 100:

- 206 candidates screened, 32 fit a bivariate N-D law, 24 have a recoverable ratio list;
- 13 pass (Regime B) and 11 fail (Regime A), 3 of them single-ray designs (K = 1);
- at the old corpus-wide exponent 0.28 the split is 15 / 9; two rows flip, both pass to fail.

| File | Contents |
|---|---|
| `CRITERIA.md` | pre-registered protocol and its dated amendments |
| `candidates.csv` | all 206 candidates with the include/exclude decision and reason |
| `designs.py` | the recovered design of each included paper: ratios, fitting protocol, exponent, provenance |
| `artifacts/*_k.txt` | run-level ratio lists rebuilt from released artifacts (Chinchilla, Muennighoff, Farseer, Pythia) |
| `classify.py` | the classification; V_K, tau_K and the regime come from `src/tpp_analysis/subset_analysis/subset_analysis_bbox_seeded.py` |
| `results.csv` | per design: K, ratios, exponent, V_K, tau_K, verdicts under both exponents and the sensitivity grids |

`tests/test_grid_classification.py` reruns `classify.py` and requires `results.csv` unchanged.
