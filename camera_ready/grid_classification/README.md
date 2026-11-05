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
- 11 pass (Regime B) and 13 fail (Regime A), 3 of them single-ray designs (K = 1);
- at the old corpus-wide exponent 0.28 the split is 14 / 10; three rows flip, all pass to fail;
- Amendments 4 in `CRITERIA.md` read three designs from their papers' run tables instead of
  the ranges stated in the text: krajewski2024 and shukor2025mixtures now fail, and
  aghajanyan2023 still passes. Before it the split was 13 / 11;
- Amendments 5 read kaplan2020's 38 runs from its own vector figure (Fig. 9 left) instead of
  the whole study's size range: K = 34, margin 2.07, still a pass. No pass now rests on an
  endpoints-mode ratio list.

Prop. 2 covers collinear designs, which train every size at every ratio. Six of the 24 designs
are collinear, and all six fail. The other 18, all 11 passes among them, train different sizes
at different ratios, so for them the pre-registered test, one weight per distinct ratio, is a
heuristic. The proof's Gram block weights each run by N^(-2 alpha) instead, alpha the
model-side exponent. On the 17 designs whose runs can be rebuilt, those weights turn two passes
into fails, gadre2024 and rosenfeld2019, whose smallest models carry most of the weight. No
other verdict changes, so the split would be 9 pass and 15 fail if muennighoff2023 keeps its
pass. The camera-ready table marks those two passes. A verdict `classify.py` gives a
non-collinear design, a new one included, rests on the same heuristic (Amendments 5 in
`CRITERIA.md`).

| File | Contents |
|---|---|
| `CRITERIA.md` | pre-registered protocol and its dated amendments |
| `candidates.csv` | all 206 candidates with the include/exclude decision and reason |
| `designs.py` | the recovered design of each included paper: ratios, fitting protocol, exponent, provenance |
| `artifacts/*_k.txt` | run-level ratio lists rebuilt from released artifacts (Chinchilla, Muennighoff, Farseer, Pythia) |
| `classify.py` | the classification; V_K, tau_K and the regime come from `src/tpp_analysis/subset_analysis/subset_analysis_bbox_seeded.py` |
| `results.csv` | per design: K, ratios, exponent, V_K, tau_K, verdicts under both exponents and the sensitivity grids |

`tests/camera_ready/test_grid_classification.py` reruns `classify.py` and requires `results.csv` unchanged.
