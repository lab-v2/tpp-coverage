# Pre-registered criteria for the published-grid classification

Release note: this is the pre-registration as written, with only references to the
private review process removed for the public release. The protocol text is unchanged.

Written 2026-07-30, BEFORE any candidate paper was collected or screened. This file fixes
the inclusion rule, the formal criterion, and every free convention in advance, so that no
choice can be made after seeing a paper's verdict. Selection is blind to outcome: a paper
is never included or excluded because of how it classifies.

## Purpose

For the NeurIPS 2026 paper "Tokens-per-Parameter Coverage Is Critical for Robust LLM
Scaling Law Extrapolation": apply the paper's Proposition 2 design diagnostic, uniformly and a priori, to every published bivariate scaling-law experimental
design whose token-per-parameter (TPP) ratio list is recoverable.

## Formal criterion (fixed; from the paper)

Proposition 2 / Eq. 5 of the paper (source: src/sections/formal_analysis.tex, label
eqn:tpp_diversity; appendix algebra in proofs_tpp_design_recipe.tex, labels
eqn:tpp_diversity_app, eqn:div_condition, eqn:R_min - Eqs. 69-71 and Eq. 95 of the
compiled paper): for ordered training ratios k_1 <= ... <= k_K and law-specific data-size
exponent beta_eff, the design is well-conditioned at target condition number kappa_target
iff

    V_K >= tau_K,

where V_K is the variance of the set of k_l^(-beta_eff) values and

    tau_K = (K + sum_l k_l^(-2 beta_eff))^2 / (K^2 * kappa_target).

Verdict "Regime B" (well-identified) iff V_K >= tau_K, else "Regime A" (ill-conditioned).
K = 1 gives V_1 = 0 < tau_1, so single-ratio designs fail at every kappa_target.

The implementation is the paper's own code, functions compute_VK, compute_tauK,
classify_regime in src/tpp_analysis/subset_analysis/subset_analysis_bbox_seeded.py of this
repository (lines 159-169), imported by classify.py. The math is reused exactly; nothing is
reinvented.

## Fixed conventions (chosen now, before data collection)

1. beta_eff = 0.28 (originally primary; SUPERSEDED as the primary by convention 9 on
   2026-07-31, retained as the comparison convention and as the fallback for papers
   that report no exponent). This is the paper's own stated a-priori convention: "a
   literature estimate of beta_eff ... e.g. beta ~ 0.28 for Chinchilla" (formal_analysis.tex,
   after Prop. 2). Sensitivity grid recorded in results.csv: beta_eff in
   {0.10, 0.20, 0.28, 0.35, 0.50}. A verdict-flip anywhere on the grid is flagged.
2. kappa_target = 100 (primary). This is the paper's main-text convention (empirical.tex
   line 138; experimental_design.tex; the R_min lookup's headline column). Sensitivity grid
   recorded in results.csv: kappa_target in {1e2, 1e4, 1e5, 1e6} (the latter three are the
   code package's KAPPA_TARGETS). Larger kappa_target is more lenient (smaller tau_K); a
   design that fails even at 1e6 fails decisively. A verdict-flip across the grid is flagged.
3. The k-list is the list of DISTINCT TPP ratios k = D/N over the training runs that enter
   one pooled fit, one entry per distinct ratio, matching Prop. 2's statement ("ordered
   training ratios k_1 <= ... <= k_K") and the paper's own usage (K = 12 for its CO fan).
   N is the paper's own parameter-count convention (if the paper fits with non-embedding
   parameters, we use that N; recorded per paper). AMENDED 2026-11-03 (Amendments 5): Prop. 2
   is now stated for a collinear design (Definition 1: every size trained at every ratio),
   for which one weight per distinct ratio is what the proof gives. For any other design the
   proof's Gram block weights each run by its Jacobian row, so one weight per distinct ratio
   is a convention, not the proposition.
4. Near-duplicate merging: two recovered ratios are the same "distinct ratio" if they
   differ by <= 1 percent relative; merged to their geometric mean. (V_K and tau_K are
   continuous in k, so this affects K only, and only for adjacent near-identical rays.)
5. Designs that are not fans of rays (crossed N x D grids, IsoFLOP sweeps, pooled
   intermediate-checkpoint fits): the k-list is the set of distinct D/N values over the
   fitted observations, computed from the published design. When the full observation list
   is large, the k-list is derived from the published axes (e.g. all D_j / N_i pairs of a
   crossed grid); when only a range is recoverable (e.g. "checkpoints from d_min to d_max
   tokens for each of the sizes"), we evaluate the criterion on the ENDPOINT ratios plus any
   explicitly published interior ratios. The endpoints-only convention rests on a HEURISTIC,
   not a proved monotonicity: the appendix (proofs_tpp_design_recipe.tex) argues that adding
   interior rays at fixed endpoint spread generally lowers V_K while tau_K tightens at a
   comparable rate, so "the two effects largely cancel" and endpoint mass is near-optimal.
   On that heuristic, endpoints-mode is favorable-or-neutral to the design under test: it
   can only create false PASSES (a design passing on endpoints might fail with its interior
   ratios included), not false failures, so a fail under endpoints-mode is a genuine fail
   while an endpoints-mode pass carries this caveat. AMENDED 2026-11-03 (Amendments 5): the
   simulation behind the camera-ready appendix shows the claim above is too strong;
   endpoints mode usually over-credits a design, and the reverse is rare (56 of 200,000
   random crossed grids), but it happens. Endpoints-mode rows are therefore
   counted and disclosed separately among the passes. The k_list_source column records which
   recovery mode was used: exact / axes-derived / endpoints / reconstruction (the last for
   run lists recovered by third-party digitization of published figures, which may be
   partial - e.g. the Besiroglu et al. SVG extraction of Chinchilla's Fig. 4, ~245 of ~400
   runs). AMENDED 2026-11-03 (Amendments 4): a run table that gives each size its own token
   counts or token range is used as printed, never the cross of the size and token ranges
   stated in the text; endpoints mode then takes the end ratios of each size's own range.
6. Fitting protocol: pooled (one fit over all ratios; K = number of distinct ratios in the
   pool) vs per-slice (separate fit per ratio or per model size; K = ratios inside one
   slice - for a per-ratio slice, K = 1). The protocol_source column records: stated (paper
   says it), artifact (released code/data shows it), or inferred (unambiguous from the
   paper's figures/tables; the inference is written down). Papers where the protocol is
   genuinely undeterminable fail inclusion rule (c) and are excluded with that reason.

### Conventions added by amendment (NOT pre-registered)

These were added on 2026-07-31 in response to the second and third audits and are recorded
here for findability; the full rationale and the rows they touch are in "Amendments 3" and
"Amendments 2" below. They are flagged as post-hoc, and each was re-run uniformly on the
full corpus.

7. Multi-fit papers: when a paper reports more than one bivariate N-and-D fit, the row
   classified is the paper's OWN primary/headline bivariate fit - the one whose constants
   it reports as its result - and every alternative fit is named in the design's provenance
   with the reason it was not the classified one. If two fits are co-equal headline fits,
   both are classified as separate rows. If the alternative fit's observation set is not
   stated run-by-run, that is recorded too (it would fail rule (b) on its own).
8. Additive-family restriction: only laws in the family Proposition 2 is actually stated
   for are classifiable (see rule (a) below).
9. PER-PAPER beta_eff (primary since 2026-07-31; supersedes convention 1 as the primary).
   tau_K depends on beta_eff, and each paper fits its own law with its own data exponent,
   so evaluating the whole corpus at one global 0.28 was wrong. Each design is now
   evaluated at the exponent ITS OWN paper reports for the D term of the additive core it
   fits: beta for Chinchilla/repeated-data, alpha_D for Kaplan, gamma_D for Droppo-Elibol
   (the three cases formal_analysis.tex line 222 names when it defines beta_eff), and for
   supersets (sparsity, precision, vocabulary, MoE, mixture, compression) the exponent on
   the D term of the additive core, recorded under the symbol that paper prints. Sub-rules:
   (a) Source of record is the paper's own LaTeX e-print where available
       (arxiv.org/e-print), else the venue PDF; the file and the equation/table number are
       recorded per row in designs.py and in the beta_eff_source column of results.csv.
   (b) Multi-fit papers follow convention 7: the headline fit's exponent is the value used,
       and the spread across the paper's other fits is recorded in beta_eff_range.
   (c) A law whose data exponent is not a constant (Farseer's A(N)) is evaluated at the
       geometric-mean N of its fitted range, with the full range recorded.
   (d) A nested-power law (Kaplan-style, SoftQ) uses the exponent of its data-limited
       limit, which is the exponent of the equivalent additive core.
   (e) If the paper fits the law but never prints a numeric exponent, the row is marked
       beta_eff_reported = False and falls back to 0.28, flagged in results.csv. No value
       is ever guessed or re-fit by us.
   (f) kappa_target stays a pre-registered grid, NOT a per-paper quantity: no paper reports
       a target condition number, so inventing one per paper would be fabrication. The grid
       {1e2, 1e4, 1e5, 1e6} of convention 2 is unchanged and verdicts are reported per
       target.
   (g) The pre-registered beta_eff grid of convention 1 remains the sensitivity check, and
       is widened per row to include that row's own exponent so a design whose beta_eff
       falls outside 0.10-0.50 is still probed at the value that decides its verdict
       (columns beta_probe_grid / beta_probe_verdicts).
   (h) Circularity disclosure: for a design that FAILS, its published exponent generally
       came from the very fit the criterion calls ill-conditioned. The column
       beta_eff_from_classified_fit records whether the exponent was estimated on the
       design under test, and exponent_circularity marks the failing rows where it was.
       Those rows are defensible only if fails_at_every_probed_beta is True.

## Inclusion rule (all three required)

(a) The work fits a bivariate parametric scaling law with BOTH N (parameters) and D
    (training tokens, or the paper's token-equivalent data axis for language/speech
    modeling) as regressors, AND that law is in the family Proposition 2 governs: an
    ADDITIVE-family law whose data term contributes a separable power of D with an
    identifiable data-side exponent beta_eff - Chinchilla (beta), Kaplan (alpha_D),
    Droppo-Elibol (gamma_D), the three cases formal_analysis.tex names when it defines
    beta_eff - or a superset containing such a core (e.g. laws with extra regressors such
    as sparsity, precision, vocabulary, or epochs, provided the additive N-and-D core is
    present). AMENDED 2026-07-31 (Amendments 2, item 1): the pre-registered text said
    "additive or multiplicative power laws". That was wrong. A purely multiplicative law
    D^gD / N^gN is linear in (log N, log D) with no irreducible constant and no curvature
    in k, so its identifiability is not what V_K >= tau_K measures; applying the criterion
    to one is a category error, not a strict test. Multiplicative-only laws are now
    excluded under (a), and a paper that ALSO fits an additive core is classified on that
    core (or excluded under (b) if the core's run list is unrecoverable).
(b) The training-run design's TPP ratio list (k = D/N per run, or enough information to
    derive it under conventions 3-5 above) is recoverable from the paper, its appendix, or
    released artifacts (code, configs, model cards, released loss tables).
(c) The fitting protocol (pooled across ratios vs per-slice) is determinable or safely
    inferable under convention 6.

## Exclusion rule

Anything failing (a), (b), or (c), logged in candidates.csv with the specific reason:

- excl-a-univariate: law is univariate (L(N), L(D), or L(C) only), or bivariate but not in
  N and D (e.g. encoder/decoder sizes, model size and RL environment steps).
- excl-a-nonparametric: no parametric law is fit (surveys, position papers, theory-only,
  benchmark releases).
- excl-a-multiplicative (added 2026-07-31): the only N-and-D law fit is multiplicative with
  no additive core, so Prop. 2 does not govern it (amended rule (a)). Where such a paper
  also fits an additive core, the row is decided on that core instead, and the recorded
  reason is whichever rule the core fails - not this one.
- excl-b-ratios: N-and-D law is fit but the run-level ratio list is not recoverable from
  paper or artifacts.
- excl-c-protocol: ratios recoverable but pooled-vs-per-slice is undeterminable.
- excl-dup: duplicate of another candidate row (e.g. arXiv + venue versions).

NO other filters. In particular: no filter on venue, year, model scale, domain
(language/speech/vision-language accepted if the law is an N-and-D token-style law), or -
under any circumstances - on the paper's classification outcome. Every candidate swept is
logged, included or not.

## Candidate sweep (defined now; executed after this file is written)

Target: at least 100 candidate papers. Seed sources, fixed in advance:

1. The two surveys the paper cites: Li, Kudugunta, Zettlemoyer, "(Mis)Fitting: A Survey of
   Scaling Laws" (arXiv:2502.18969), whose study table spans 50+ scaling-law papers; and
   Choshen, Zhang, Andreas, "A Hitchhiker's Guide to Scaling Law Estimation"
   (arXiv:2410.11840), with released per-model (N, D) data.
2. The paper's own cited pipelines: Kaplan et al. 2020; Hoffmann et al. 2022; Dey et al.
   2023 (Cerebras-GPT); Bhagia et al. 2025 (OLMo ladder); Magnusson et al. 2025
   (DataDecide); Gadre et al. 2024; Sardana et al. 2025; Li et al. 2025 (Farseer); Droppo
   and Elibol 2021; Muennighoff et al. 2023; Besiroglu et al. 2024; Porian et al. 2025.
3. Citation chains from 1-2 and arXiv keyword search for joint L(N, D) fits (MoE, precision,
   vocabulary, distillation, transfer, NMT, data-constrained, hyperparameter-aware laws).

Every candidate goes into candidates.csv at first sight, BEFORE its design is analyzed,
with the include/exclude decision and reason appended after applying rules (a)-(c) only.

## Outputs

- candidates.csv: every candidate, include/exclude + reason + source of the decision.
- results.csv: per included paper: K, ratio list, k_list_source, protocol,
  protocol_source, the paper's own beta_eff (value, symbol, law family, source,
  reported/not, range across its other fits), V_K and tau_K and verdict and regime BOTH at
  that own beta_eff (primary) and at the old global 0.28 (comparison), a verdict_changed
  flag, per-kappa_target verdict columns under both exponents, the widened beta probe grid
  with its per-beta verdicts, and the circularity flags.
- README.md: reproduction instructions.

## Amendments 5 (2026-11-03, Kaplan's runs from its figure; scope of the weights convention)

A fifth audit, of the camera-ready appendix, found that Kaplan's row used the size range of
the whole study, and that Prop. 2, now stated for collinear designs only, does not give the
distinct-ratio convention for the other designs. What was done and what changed:

1. kaplan2020: from endpoints to reconstruction mode. Amendments 4 item 5 said the paper
   publishes no run table. Its L(N, D) sweep is plotted, as vector graphics, in Fig. 9 left
   (figures/DatasetModelSizevsPerformance.pdf of the arXiv e-print); main.tex line 474 says the
   fit is displayed there. Reading each marker against the 10^6 to 10^9 ticks gives 38 runs:
   sizes 3.93e5, 3.15e6, 2.52e7, 8.49e7, 3.02e8 and 7.08e8 non-embedding parameters on the
   legend's data sizes 21M to 22.0B tokens, 6 sizes x 8 data sizes, not fully crossed
   (designs.py, _KAPLAN_FIG9_RUNS). The 768 to 1.5B range of the old row is the whole study's.
   K goes from 4 to 34, ratios 0.0297 to 5.59e4, V_K/tau_K from 8.875 to 2.065, kappa from 11
   to 48. The verdict stays pass, at every probed exponent and at 0.28 (9.46). Endpoints-mode
   passes fall from 1 to 0; shukor2025mixtures, a failure, is the one endpoints-mode row left.
   Reconstruction mode, which convention 5 defines for third-party digitizations, now also
   covers this reading of a vector figure.
2. Scope of convention 3. Six of the 24 designs are collinear designs in the sense of
   Definition 1 (hu2024, bhagia2025, volkova2026optimizers and the three with K = 1); all six
   fail. The other 18, all 11 passes among them, train different sizes on different ratios, so
   for them Eq. 5 on distinct ratios is a convention, as for a superset law. The proof's own
   alternative weights each run by N^(-2 alpha), alpha the model-side exponent (Step 2 of the
   proof). Recomputed on the 17 run lists that can be rebuilt (muennighoff2023's sizes are not
   in its artifact), with each paper's model-side exponent where it prints one and beta_eff
   otherwise, two passes fail: gadre2024 (1.063 to 0.929) and rosenfeld2019 (12.77 to 0.199;
   0.028 with beta_eff as the weight exponent), whose smallest models carry most of the weight.
   No other row changes verdict; choshen2024 is closest (3.737 to 1.055). Under that weighting
   the split would be 9 pass / 15 fail, if muennighoff2023 keeps its pass. The primary
   convention stays one weight per distinct ratio, as pre-registered; rosenfeld2019 and
   gadre2024 are marked as passing only with equal weight per ratio in the camera-ready table.
3. The endpoints-mode wording of convention 5 is corrected (see the note there): the reverse
   error is rare, not impossible.
4. designs.py's sardana2023 comment called the larger sizes' ratio lists prefixes of the 150M
   list; 2.5B runs 10, 15, 20, 50, 100, 250 and 500. No verdict effect (12 distinct ratios
   either way).
5. Regression check: the rerun of classify.py changes only the kaplan2020 row of results.csv
   (K, k_list_source, k_distinct, provenance, V_K and tau_K at both exponents); its verdicts,
   the other 23 rows and the calibration row are unchanged. The split stays 11 pass / 13 fail,
   and 14 / 10 at 0.28.

## Amendments 4 (2026-11-03, run tables read as published)

A fourth audit, of the camera-ready appendix, found two rows whose k-lists crossed the size
and token ranges stated in a paper's text although the paper publishes a run table.
Rechecking every other endpoints-mode row against its paper found a third. What was done and
what changed:

1. Why. Convention 5 derives the k-list from the published design, uses end points only when
   no more is recoverable, and adds "any explicitly published interior ratios". Three rows did
   not follow it:
   - krajewski2024 crossed the size range stated in the text (129M to 3.7B) with every token
     count in the paper (16, 24, 33, 66 and 130B). The appendix table of the MoE runs
     (tab:moe, E = 64) gives each size its own token counts: the largest size ran only at
     33B, 130B only at 64x13M, and 24B appears only in the dense table, whose runs are not in
     the classified MoE fit. The cross paired 129M with 130B and 3.7B with 16B, which no run
     reaches.
   - aghajanyan2023 used only the end sizes 8M and 6.7B, although the paper lists all seven
     sizes and trains each on all three token counts.
   - shukor2025mixtures crossed its extreme sizes with its extreme token counts (412M and 1.4B
     with 4B and 46B). Its table tab:train_val_points gives each size its own token range
     (412M: 4-20B, 834M: 7-26B, 1.1B: 7-36B, 1.4B: 13-46B), so the corners 412M at 46B and
     1.4B at 4B, which set the old range 2.86-112, match no run. Adding the interior sizes
     834M and 1.1B to the crossed corners was considered (margin 1.87); the table shows that
     the cross itself was the error.
2. What changed in designs.py:
   - krajewski2024: from endpoints to exact mode, the 17 (size, D) pairs of tab:moe with
     N = d_model^2 (8E + 4) n_blocks, the paper's total-parameter count, which reproduces its
     figure files granularity_N_1082130432 and granularity_N_2113536000. K = 16 (two pairs
     share k = 61.0), ratios 7.57-488.
   - aghajanyan2023: from endpoints to axes-derived mode, the full 7 x 3 grid. K = 21, ratios
     0.746-12500 as before.
   - shukor2025mixtures: stays in endpoints mode, because the checkpoints inside each range
     are not enumerated, but on the two end ratios of each size's own range. K = 7 (32.7 and
     32.9 merge), ratios 6.36-48.5.
   Convention 5 gains the sentence marked AMENDED 2026-11-03.
3. Outcome at the primary convention (own exponent, kappa_target = 100):
   - krajewski2024: from pass to fail, V_K/tau_K from 1.263 to 0.587, kappa from 79 to 170.
     It fails at every exponent the paper reports (0.147 for the MoE fit, 0.127 dense, 0.140
     for the validation refit, 0.169 at E = 16) and passes only at 0.28 (1.074) and 0.35
     (1.097) of the probe grid. Its exponent comes from the fit under test, so it joins the
     failing rows with a circular, verdict-deciding exponent.
   - shukor2025mixtures: from pass to fail, from 2.024 to 0.605, kappa from 49 to 165. It
     prints no exponent (0.28 is used) and fails at every probed exponent (margins 0.13 to
     0.75 over 0.10-0.50).
   - aghajanyan2023: stays a pass, from 6.686 to 3.420, kappa from 15 to 29, at every probed
     exponent.
   The split moves from 13 pass / 11 fail to 11 pass / 13 fail, 3 of them single-ratio. At the
   old corpus-wide 0.28 it moves from 15 / 9 to 14 / 10 (Shukor fails there too; Krajewski
   passes there). Endpoints-mode passes fall from 4 to 1 (kaplan2020). Failing rows whose
   exponent came from the fit under test: 9 of 13, six of which fail at every probed exponent.
   Multi-ratio failures that pass at kappa_target 1e4: 9 of 10; tao2024 passes at 1e5.
4. None of the three new verdicts changes under any of these variants, recomputed with code
   independent of classify.py:
   - the token counts the printed 16B, 33B, 66B and 130B round (2^k steps of 2048 x 256
     tokens): Krajewski 0.614;
   - merging nothing, or merging only within 1% of a group's first member;
   - counting each run once (granularities for Krajewski, domain weights for Shukor): 0.545
     and 0.639;
   - rescaling every k by a factor from 1/4 to 4: Krajewski 0.46-0.70, Shukor 0.35-0.88,
     Aghajanyan 2.36-4.31.
5. kaplan2020, the one endpoints-mode pass left, publishes no run table. Its text states the
   size range (768 to 1.5B non-embedding parameters) and the data range (22M to 23B tokens)
   but not that the smallest size ran on the largest subset. It stays in endpoints mode; its
   pass holds with the smallest size raised from 768 up to 3e8 parameters (margin 8.88 to
   2.74). Superseded by Amendments 5, item 1: the run list is recoverable from a figure.
6. Outcome-blind statement. The rows were rechecked because their recovery departed from
   convention 5, not because of their verdicts. The check covered the four endpoints-mode
   rows, the mode convention 5 marks as able to over-credit a design, and all four were
   passes; the axes-derived rows rest on grids their papers state as crossed (quoted in
   designs.py) and were not recoded. Each recoded row takes its paper's table as printed. The
   direction is recorded so it is not read as outcome-driven: two passes became failures and
   none moved the other way, so the corpus looks worse for the papers, as after Amendments 3.
7. Departure from the earlier count. Before this amendment the classification reported
   13 pass / 11 fail, with krajewski2024 (K = 10, corners 4.32-1008), aghajanyan2023 (K = 6)
   and shukor2025mixtures (K = 4, corners 2.86-112) in Regime B. The camera-ready appendix
   reports 11 / 13 and states the departure.
8. candidates.csv is unchanged. The include decisions stand (the ratio lists are recoverable,
   rule (b)); the reason_detail text of krajewski2024, aghajanyan2023 and shukor2025mixtures
   still describes the earlier reading, as logged at screening. designs.py is the record of
   each recovered design.
9. Regression check: the rerun of classify.py changes only these three rows of results.csv;
   the other 21 rows and the calibration row are byte-identical to the previous run.

## Amendments 3 (2026-07-31, per-paper beta_eff)

A third audit found the corpus-wide beta_eff = 0.28 to be a real methodological error, not
a conservatism: tau_K depends on beta_eff, so classifying a Kaplan-form NMT fit, a
vocabulary law fit on characters, and a Chinchilla refit all at Chinchilla's exponent tests
none of them at their own law. Convention 9 above is the fix. What was done and what
changed:

1. Every one of the 24 classified designs plus the calibration row was re-sourced from the
   paper's own LaTeX where obtainable: 23 arXiv e-prints were downloaded and extracted
   (arxiv.org/e-print/<ID>) and read for the fit table or fitted equation. gordon2021 has
   no arXiv e-print and was read from the ACL Anthology PDF (2021.emnlp-main.478); that is
   the only PDF-sourced row and it is labelled as such in designs.py.
2. 19 of the 24 corpus rows report a numeric data exponent; 5 do not and fall back to 0.28
   flagged (sedova2026, magnusson2025, choshen2024, shukor2025mixtures,
   panferov2025compressed). Three of those five are K = 1 designs, where the verdict is
   V_1 = 0 < tau_1 at every beta_eff, so the fallback cannot affect them; the other two
   (choshen2024, shukor2025mixtures) pass at 0.28 and are disclosed as exponent-unrecovered
   passes. For choshen2024 the released ColPret artifact was also checked - it carries raw
   per-checkpoint eval data, not fitted coefficients.
3. The recovered exponents span 0.10 to 0.74, i.e. a factor of seven, which is the reason
   the single-value convention could not stand: kaplan2020 alpha_D = 0.103,
   xu2026softq alpha/(1+alpha) = 0.125, krajewski2024 beta = 0.147, frantar2023
   b_D = 0.203, aghajanyan2023 beta = 0.22, hu2024 beta = 0.23, bhagia2025 beta = 0.24,
   sardana2023 beta = 0.24, gadre2024 beta = 0.242, hoffmann2022 beta = 0.2849,
   li2025farseer A(N_gm) = 0.327, muennighoff2023 beta = 0.3527, besiroglu2024
   beta = 0.3658, gordon2021 alpha_D = 0.38, volkova2026optimizers beta = 0.38,
   tao2024 beta = 0.447, kumar2024 beta = 0.4965, sun2025fp beta = 0.5162,
   rosenfeld2019 alpha = 0.735.
4. Verdict effect, reported without tuning: the split moves from 15 pass / 9 fail to
   13 pass / 11 fail. Exactly two rows flip, both pass -> fail:
   - xu2026softq: SoftQ's data-limited exponent alpha/(1+alpha) = 0.125 is much flatter
     than 0.28, and its 5x4 (N, U) grid straddles k = 1, so a flatter exponent compresses
     the k^-beta spread (V_K 0.151 -> 0.024) faster than it relaxes tau_K
     (0.061 -> 0.045).
   - kumar2024: its tied alpha = beta = 0.4965 is much steeper than 0.28, and its grid
     sits entirely at k >> 1, where a steeper exponent shrinks every k^-beta toward zero
     and collapses V_K (0.0136 -> 0.0088) while tau_K falls only toward the floor 1/kappa
     (0.0124 -> 0.0106). It was already flagged knife-edge at 0.28.
   No row flips fail -> pass. The direction is stated so it is not read as outcome-driven:
   the correction made the corpus look WORSE for the papers, not better, and both flips are
   rows the previous run had already flagged as beta-sensitive.
5. gadre2024 remains a pass but its margin narrows: at its own beta = 0.242, V_K 0.01401 vs
   tau_K 0.01318, a ~6 percent margin (it was ~10 percent at 0.28). It is still flagged
   knife-edge.
6. Circularity, recorded because it bites: 8 of the 11 failures carry an exponent estimated
   on the design being called ill-conditioned. Six of those eight fail at EVERY probed
   beta_eff - hu2024, frantar2023, tao2024, bhagia2025, volkova2026optimizers, sun2025fp -
   so their verdicts do not depend on trusting the self-reported number. The remaining two
   are exactly the two flips (xu2026softq, kumar2024), whose verdicts DO depend on the
   exponent and which are reported as such rather than as settled. The three K = 1 failures
   are exponent-free by construction. volkova2026optimizers is the sharpest case: the
   paper itself calls its per-optimizer fits ill-conditioned with highly correlated
   parameters and reports A = 4966 +/- 4612, and the design fails the criterion at every
   exponent in its own reported 0.24-0.38 range and beyond.
7. kappa_target sensitivity under the per-paper exponents: of the 8 non-K=1 failures,
   7 clear the criterion at kappa_target = 1e4 and the eighth (tao2024) only at 1e5 - the
   same structure as before, with two more rows in the numerator. The three K = 1 designs
   fail at every kappa_target.
8. Regression check: recomputing the corpus at 0.28 with the new code reproduces the
   previous run's K, V_K, tau_K and verdict for every row exactly, so the split change is
   attributable to the exponents alone and not to a code change.

## Amendments 2 (2026-07-31, post second audit)

A second hostile audit was run on the amended corpus. Its findings and the changes made,
again recorded rather than silently merged; every change was re-run uniformly on the full
corpus, and the resulting counts are those of item 6 below.

1. Rule (a) was narrowed to the additive family Prop. 2 is actually stated for (see the
   rule text above, and new convention 8). Consequence: chen2025qat moves from include to
   exclude. It had been classified on the QAT ERROR law delta_p = k D^gD (log2 G)^gG/N^gN
   (their Eq. 5), which is multiplicative and outside the family; its Prop.-2-governed law
   is the Chinchilla core of Eq. 4, fit in Appendix C on 145M-2.8B models plus the
   externally trained OLMO-2-7B and OLMO-2-13B releases, whose per-run token budgets are
   never stated - so that fit is excl-b-ratios. Direction of the correction, stated so it
   is not read as outcome-driven: the removed row was a FAIL, and a best-effort
   reconstruction of the additive fit would have PASSED. The whole corpus was re-screened
   for other multiplicative-only rows; none were found. sun2025fp was re-checked at full
   text and its Eq. 15 does carry an additive Chinchilla core (n/N^a + d/D^b + eps plus
   precision terms), so it stays. li2025farseer's Eq. 15 is additive but its data exponent
   varies with N (~0.46 at 1e8 params to ~0.21 at 6.4e9); it stays, with that recorded in
   its provenance, and its verdict does not flip anywhere on the pre-registered beta_eff
   grid, which brackets that range.
2. choshen2024 was recomputed in EXACT mode instead of endpoints mode. The released ColPret
   artifact enumerates every Pythia checkpoint (steps 1000..143000 at 2,097,152 tokens per
   step, 8 sizes), so the full fitted observation set is recoverable and there was no need
   for the endpoints heuristic: K goes 16 -> 243, the verdict stays pass, and the row no
   longer counts against the endpoints-mode caveat.
3. shukor2025nmm moves from include (endpoints mode, pass) to exclude, excl-b-ratios. Its
   457 runs are a compute sweep, not a crossed N x D grid, so convention 5's endpoint
   clause - which requires checkpoints d_min..d_max FOR EACH size - has its precondition
   fail: the corner 3B params x 250M tokens need not correspond to any run, and that corner
   drove a V_K of 0.561, the second largest in the corpus. This removes a PASS.
4. Convention 7 (which fit inside a multi-fit paper gets classified) was added. It was the
   rule actually being applied, but it had not been written down, and three rows turn on it
   (magnusson2025, sedova2026, sardana2023 - two of which land on the failing fit). Each of
   those three now records in its provenance why the alternative fit was not classified. No
   row changed verdict; DataDecide's alternative is its headline L(C), which is univariate
   in compute and so not classifiable at all.
5. Seed source 1 of the candidate sweep names Choshen et al. as a source of "released
   per-model (N, D) data", but no candidates.csv row carries a choshen sweep-source tag. To
   be accurate: Choshen et al. was used as a candidate row in its own right (key
   choshen2024, source pipeline_cited) and as an ARTIFACT source (the ColPret release,
   artifacts/choshen_pythia_k.txt), not as a list of further candidate papers to sweep. The
   sweep denominator therefore comes from the (Mis)Fitting survey table, the paper's cited
   pipelines, and the three fixed arXiv pages.
6. Corrected counts after 1-3: 206 screened (unchanged), 32 bivariate N-and-D laws
   (unchanged), 24 classified, 15 pass, 9 fail of which 3 single-ray, 4 endpoints-mode
   passes. Of the 6 non-single-ray failures, 5 clear the criterion at kappa_target 1e4 and
   the sixth (tao2024) only at 1e5 - the earlier "all 7 at 1e4" statement was wrong and was
   corrected.

## Amendments (2026-07-30, post-audit)

The following changes were made AFTER initial verdicts were computed, in response to a
hostile audit of the first run; they are recorded here rather than silently merged into
the pre-registered text, and every change was re-run uniformly on the full corpus:

1. A third fixed arXiv sweep page was added covering the submission-date gap 2024-06 to
   2026-04 between the two original pages (export.arxiv.org API, title search
   "scaling law(s)", submittedDate [202406010000, 202605010000], sortBy=relevance,
   max_results=60; source tag sweep_arxiv_gap). All 60 hits screened under rules (a)-(c).
2. k_list_source gained the "reconstruction" mode (see convention 5) and hoffmann2022 was
   relabeled exact -> reconstruction.
3. Convention 5's endpoint rule was reworded from a claimed proof ("fails a fortiori") to
   the honest heuristic above, and endpoints-mode passes are now disclosed as a count.
4. Per-kappa_target verdict columns were added to results.csv.
5. Screening errors found by the audit were corrected: sardana2023 (v1-based exclusion;
   the cited ICML version refits on 47 own models) and magnusson2025 (Appendix C fits an
   L(N,D) baseline) were re-included; choshen2024 was re-included after the ColPret
   artifact recovery attempt succeeded; pearce2024 was re-screened under rule (a) with a
   Gadre-consistent reason; caballero2026's exclusion reason was verified at full text.
   No verdict of a previously included row changed; the corrections only added rows.
