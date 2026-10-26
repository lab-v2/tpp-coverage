# Pre-registered criteria for the public-checkpoint re-analysis

Release note. This is the pre-registration as written, with references to the private review
process removed and "the paper" read as "the paper". Nothing in the protocol changed.
Several scripts were renamed for the release: `tighten1b_fit.py` is `enumerate_seeds.py`,
`tighten1b_boot.py` is `cluster_bootstrap.py`, `phase4_estimate.py` is `estimate.py`, and its
outputs `phase4_seedcompare.csv` / `phase4_doseresponse.csv` are `estimates_seeds512.csv` /
`doseresponse_seeds512.csv`. Probe scripts, logs, the per-phase RESULTS notes and the
diagnostic scripts named in the amendments are part of the analysis record and are not shipped;
README.md lists what is.

Written 2026-08-01, BEFORE any loss value was read, any subset was carved, and any scaling
law was fitted. This file fixes the suite-inclusion rule, the CO/NC carving rule, the
budget-matching rule, the holdout definition, the size strata, the seeds, and the estimator
for the size interaction, so that no choice can be made after seeing an outcome.

## Purpose

Two analyses for the NeurIPS 2026 paper "Tokens-per-Parameter Coverage Is Critical for
Robust LLM Scaling Law Extrapolation":

1. a re-analysis of public checkpoints with released losses;
2. at least one empirical scale-sensitivity test: measure the NC-versus-CO off-ray gap, or the corresponding conditioning diagnostic,
   across model-size strata - preferably including 1B+ checkpoints - and report either a
   size interaction or confidence intervals that exclude material attenuation with
   increasing N.

## What was inspected before this file was written (blindness disclosure)

A scoping pass over public artifacts read ONLY design metadata: which suites publish a loss
column at all, the distinct parameter counts N, the token counts D, the derived ratios
k = D/N, and the run/checkpoint structure. No loss VALUE was read, no subset was carved, no
law was fitted, and no RMSE was computed. The inventory that resulted is in README.md; the
suite candidate list in section "Candidate suites" below is closed as of this file and no
suite may be added or dropped later on the basis of how it comes out.

## Formal objects (taken from the paper, not reinvented)

Everything numeric is computed with the paper's own released implementations:

- the four law forms and the multi-start fitter: `scaling_laws/laws.py`
  (`chinchilla_loss`, `kaplan_loss`, `droppo_elibol_loss`, `_fit_generic`);
- the budget-matched non-collinear subset selector: `select_nc_bbox` and
  `_priority_subsample` in `subset_analysis/subset_analysis_bbox_seeded.py`;
- the conditioning diagnostic: `compute_VK`, `compute_tauK`, `classify_regime`
  (same file, lines 161-171),

    V_K   = Var_l( k_l^(-beta_eff) ),
    tau_K = (K + sum_l k_l^(-2 beta_eff))^2 / (K^2 * kappa_target),
    Regime B (well-identified) iff V_K >= tau_K, else Regime A.

Both are imported from
`code/Tokens-per-Parameter_Coverage_Is_Critical_for_Robust_LLM_Scaling_Law_Extrapolation/
src/tpp_analysis/`. Nothing about the fitting or the subset logic is re-implemented here.

## Fixed conventions

C1. **Ratio bins.** b(k) = floor(log2(k)); bins are factor-2 wide and anchored at k = 1.
    A "ray" is one bin. This is the only binning used anywhere in this analysis. It replaces
    the repo's `assign_tpp_bins` snap list, which is calibrated to the paper's own
    k in [1, 5] range and cannot express the k in [0.03, 2570] range that public suites
    span; the selector `select_nc_bbox` that consumes the bins is the repo's, unmodified.

C1b. **Ratio list for V_K / tau_K.** The distinct k values of the design's own runs, merged
    at 1 percent relative difference to their geometric mean - convention 4 of the sibling
    published-grid classification (`../grid_classification/CRITERIA.md`), reused verbatim
    so the two analyses report the diagnostic on the same footing.

C2. **beta_eff = 0.28** primary (the paper's own a-priori Chinchilla convention), with the
    pre-registered sensitivity grid {0.10, 0.20, 0.28, 0.35, 0.50}.

C3. **kappa_target = 100** primary (the paper's main-text convention), with the
    pre-registered grid {1e2, 1e4, 1e5, 1e6}.

C4. **Laws.** Chinchilla (5 parameters, minimum 5 points) and Kaplan (4 parameters,
    minimum 4 points), both from `laws.py`. `N_RESTARTS = 80` multi-start L-BFGS-B plus the
    deterministic differential-evolution fallback, fit seed 42 + 100003 * seed, exactly the
    seeding scheme of `subset_analysis_bbox_seeded.py`. Droppo-Elibol (6 parameters) is
    excluded a priori: several windows would not clear its 6-point minimum, and a law that
    can be fitted in some cells but not others would make the strata non-comparable.

C5. **Loss column per suite** (fixed now, never chosen by result):
    Farseer -> `loss`; sensitivity variants `smooth loss` and `L`.
    Gadre over-training -> `openLM_loss` (the only one released).
    Chinchilla/Besiroglu -> `loss`.
    Any other admitted suite -> the single released LM loss column; if a suite releases both
    a training and a validation loss, validation is primary.

C6. **Seeds.** NC bounding-box selection is run at seeds 0, 1, 2, 3, 4. CO enumerates every
    admissible ray and is deterministic. The holdout split is deterministic (it is a
    definition, not a knob), matching the paper's decision not to vary `HOLDOUT_SEED_BASE`.

## Inclusion rule for a suite (all four required)

(P1) A public, non-authenticated artifact releases a per-run LM loss together with
     recoverable N (parameters) and D (training tokens).
(P2) The released loss is an end-of-run observation, or is the observation set the source
     paper itself fits. Mid-schedule intermediate checkpoints from a run whose learning rate
     is still decaying are NOT admitted: they do not lie on the converged loss surface, and
     their D-variation is temporally autocorrelated. (This is a conservative exclusion. The
     paper's Section 5 argues that checkpoint autocorrelation biases in the direction
     that UNDERSTATES the pathology, so admitting such suites could only help us; they are
     excluded anyway and listed in README.md with that direction stated.)
(P3) The design has a genuinely 2D core: at least 6 distinct N, and at least 3 distinct
     k-bins each populated at 4 or more distinct N. A single-ray ladder fails here by
     construction, and is recorded as failing rather than quietly dropped.
(P4) Language modeling; one architecture family and one training corpus per analysis cell
     (a "cell" = suite x corpus). Cells are analysed separately and never pooled, so that no
     CO-versus-NC contrast is ever confounded by a corpus or architecture change.

A suite failing any of (P1)-(P4) is excluded with the specific rule recorded. No filter on
venue, year, scale, or outcome.

## Candidate suites (closed list, fixed before any fit)

Farseer (Li et al. 2025); Gadre et al. (2024) over-training; Chinchilla Fig. 4 as digitized
by Besiroglu et al. (2024); Muennighoff et al. (2023) data-constrained; MAD
(arXiv:2403.17844); GPT-3 digitized curves; T5-Pile; Pythia; OLMo / OLMo-2; the Bhagia et
al. ladder; DataDecide; LLM360 Amber / CrystalCoder / K2; RedPajama-INCITE; Cerebras-GPT;
Sardana et al. (2025); the ColPret aggregation as a whole.

## Carving rule

Everything below is computed independently inside each (cell, window).

W1. **Windows (primary strata).** Sort a cell's distinct N ascending. If the cell has >= 12
    distinct N, use contiguous windows of W = 8 with stride 2; if 8 <= #N < 12, W = 5 with
    stride 1; if #N < 8, a single window over all of them. A window's scale statistic is
    N_gm, the geometric mean of its distinct N. Windows deliberately hold the WITHIN-window
    N-span roughly constant while sliding the absolute scale, which is what makes the
    across-strata comparison controlled: a coarse tertile split instead confounds "bigger
    models" with "narrower N-lever", and would make any apparent attenuation an artifact of
    the shrinking lever.

W2. **Coarse strata (secondary).** Windows are additionally grouped
    by N_gm into <100M, 100M-1B, and >=1B and reported that way too.

W3. **Off-ray holdout (primary, fixed before any design is carved).** Within a window, rank
    all points by k. The lowest 15% and the highest 15% by k form the off-ray holdout. The
    remaining middle 70% is the CORE. The holdout is identical for every design in that
    window - CO and NC are scored on exactly the same points - and no training design may
    ever contain a holdout point.

W4. **On-ray holdout (secondary).** Core points not in the training design whose k lies
    inside [min k_train, max k_train]. Reported so the result is two-sided: the paper's
    own claim is that CO is competitive on-ray and fails only off-ray, so an analysis that
    only ever showed CO losing would be evidence that the baseline is rigged.

W5. **CO design.** For each k-bin b, CO(b) = every core point in bin b. Admissible iff it
    has at least the law's minimum point count and at least 4 distinct N. Every admissible
    bin is enumerated; none is chosen. Secondary variant CO2(b) = bins b and b+1 pooled.

W6. **NC design.** Budget-matched to identical cardinality n = |CO(b)|, drawn from the same
    core by `select_nc_bbox` with `target_tpps` set to CO(b)'s ratios and the seeded bbox
    RNG of C6. The bbox grid is the (sorted distinct N) x (sorted distinct D) index grid of
    the core. If the selector returns fewer than n points the pair is dropped, for both
    designs, and the drop is logged.

W7. **What is compared.** For each (cell, window, bin, seed, law): fit CO and fit NC on
    their own n points, predict the identical off-ray holdout, and record

        gap = log10( RMSE_holdout(CO) / RMSE_holdout(NC) ).

    gap > 0 means the non-collinear design extrapolates better, which is the paper's
    prediction. gap is the unit of analysis throughout.

## Estimator for the scale question

E1. **Per-stratum summary.** Median gap per window and per coarse stratum, with a percentile
    bootstrap 95% CI over 2000 resamples of the window's design pairs.

E2. **Size interaction (the scale-sensitivity test).** OLS of `gap` on `log10(N_gm)` with
    a fixed effect (dummy) per cell, so the slope is identified WITHIN a suite and cannot be
    driven by a between-suite difference:

        gap_i = beta_scale * log10(N_gm_i) + sum_c delta_c * 1[cell_i = c] + e_i.

    beta_scale is the change in the log10 CO/NC off-ray RMSE ratio per decade of N. Its 95%
    CI comes from a cluster bootstrap over WINDOWS (2000 resamples, clusters = window,
    resampled within cell), because design pairs inside a window share points and are not
    independent.

E3. **Reporting rule, fixed now.** The standard is explicit: a nonsignificant
    p-value alone does not establish scale independence. We therefore report the CI on
    beta_scale and translate its most negative endpoint into the largest per-decade
    attenuation the data cannot rule out, in percent of the gap. If beta_scale is negative
    with a CI excluding zero, the finding is reported as attenuation with scale, in the
    abstract of RESULTS.md and not in a footnote.

E4. **Conditioning diagnostic (the alternative test).** V_K, tau_K and the
    regime are computed for every CO and NC design at the C2/C3 primary values and over both
    grids. Per stratum we report the Regime A share for CO and for NC and the median
    log10(V_K / tau_K). This quantity is a property of the ratio list alone, so if it does
    not move across strata that is itself the scale-invariance evidence, and it is reported
    whether or not the RMSE gap cooperates.

## Amendments (post-registration, recorded rather than silently merged)

**Amendment 1 (2026-08-01): the on-ray holdout of W4 was vacuous as written.**
W4 defined the on-ray holdout as core points outside the training design whose k lies inside
the design's own k-range. For a CO design that is empty by construction, because W5 puts
*every* core point of the ray into CO - so the secondary two-sided check could never have
produced a number. Replaced by a like-for-like interpolation arm that keeps every property
W4 was meant to have: withhold a seeded 20 percent of the CO ray, refit CO on the remaining
80 percent, refit NC on a seeded 80 percent of ITS OWN points (identical cardinality, so the
arm is still budget-matched), and score both on exactly the points withheld from the CO ray.
The primary off-ray analysis is untouched by this: it uses the full designs and the full,
design-independent tail holdout of W3 exactly as pre-registered. Only the secondary
`gap_onray` column changes meaning. The amendment was made before any corrected result was
read.

**Amendment 2 (2026-08-01): an N-matched NC arm, added because the pre-registered NC
construction does not transfer cleanly to sparse public grids.**
`select_nc_bbox` grows a bounding box outward in (sorted N) x (sorted D) index space. On the
paper's own densely crossed grid that box keeps every model size. On public grids, where
most suites give a handful of token budgets per size and no two runs share a D, the box often
buys D-coverage by dropping sizes: on Farseer the NC design carries a median of 6 distinct N
against CO's 8. A design that loses model sizes loses the very lever the N-side coefficient
is estimated from, so a CO-versus-NC difference measured that way confounds ratio coverage
with N coverage - exactly the confound this analysis exists to avoid.

The fix adds a THIRD design, NCM, built by the same repo selector from a pool restricted to
the model sizes CO already uses. CO and NCM then have the same sizes, the same cardinality,
the same core, and the same holdout, and differ only in ratio coverage. The pre-registered NC
arm is kept and reported unchanged as the primary; NCM is reported alongside it as the
disclosed post-hoc arm. Both are in `results.csv` and both are printed by `strata.py`; the
direction each one moves the conclusion is stated in RESULTS.md whichever way it comes out.

**Amendment 3 (2026-08-01): the contrast variable was wrong. The nominal design label is
replaced by the theory's own conditioning quantity, and the primary analysis becomes
continuous rather than binary. Written before any corrected outcome was computed.**

*What was wrong.* The analysis of Amendments 1-2 contrasted designs by their NOMINAL label:
everything carved as a single ray was called "collinear", everything carved by
`select_nc_bbox` was called "non-collinear", and the pooled off-ray gap between those two
labels was reported as the test of the paper's claim. That is not what the paper
claims. Theorem 1 is **conditional**: it bounds the off-ray extrapolation error of a design
that fails the Proposition 2 / Eq. 5 criterion (Regime A, V_K < tau_K) and it *certifies*
designs that pass it (Regime B). A design's nominal shape is not its premise; V_K versus
tau_K is. On the paper's own densely crossed grid the two coincide, because every
single-ray design there is deep in Regime A and every bbox design is in Regime B, so the
label is a faithful proxy. On public grids they come apart: a nominally fixed-ratio public
ladder has runs whose realised k differs by a few percent, so V_K > 0, and at a loose
kappa_target such a ladder can sit in Regime B. Contrasting labels therefore averages the
regime where the theory predicts an ordering together with the regime where the theory
predicts none, and the resulting pooled number tests no statement the paper makes.

*What is not the fix.* Reporting the one kappa_target at which the regime-separated subgroup
looks best. The prior run did in effect that, quoted the 1e6 cell, and called it post-hoc.
The conditioning variable is a published a-priori criterion, so conditioning on it is not
post-hoc; but *choosing* the target that maximises the effect would be threshold-shopping and
is prohibited below.

*Two facts about the criterion, derived before running anything, that shape the design.*
`compute_tauK` is exactly inversely proportional to `kappa_target`, so writing
m_X = log10(V_K[X] / tau_K[X]) for the identifiability margin of design arm X
(m_X < 0 iff Regime A):

  (i) m_X(kappa) = m_X(100) + log10(kappa / 100). The kappa grid shifts every margin by one
      common constant. A regression slope on a margin is therefore IDENTICAL at every
      kappa_target; only the intercept moves. kappa can bite only through the threshold, i.e.
      only in the regime-classified analyses.
  (ii) The between-arm contrast Delta = m_NC - m_CO is exactly kappa-invariant, because the
      kappa factor cancels in the ratio. The primary regressor below therefore cannot be
      tuned by a threshold at all.

### A3.1 PRIMARY analysis (continuous, replaces the binary label contrast)

Unit of analysis is unchanged: one budget-matched design pair per
(cell, window, ray, seed, law), single-ray CO (n_rays = 1), pre-registered NC arm, at the
C2 primary beta_eff = 0.28. Outcome is unchanged: gap = log10(RMSE_offray(CO) / RMSE_offray(NC)).

Regressor: **Delta = m_NC - m_CO**, the amount by which the budget-matched non-collinear arm
is better identified than the collinear arm, in decades. Estimator:

    gap_i = beta_delta * Delta_i + sum_c delta_c * 1[cell_i = c] + e_i

by OLS with a per-cell fixed effect, so beta_delta is identified WITHIN a suite. Its 95% CI
comes from the same cluster bootstrap as E2: 2000 resamples, clusters = (cell, window),
resampled within cell.

**The paper's prediction is beta_delta > 0.** A CI on beta_delta that contains zero is a
null result for the theory on public data and is reported as such, in the summary of
RESULTS.md and not in a footnote. This uses every pair, involves no threshold, and is
kappa-invariant by fact (ii).

Pre-registered companions to the primary, all reported whichever way they come out:
- the same regression with a (cell, window) fixed effect instead of a cell fixed effect;
- the same regression with no fixed effect;
- the same regression on the NCM arm of Amendment 2;
- the same regression run separately per law and per cell;
- the secondary regressor m_CO alone (depth of the collinear arm's own deficit), whose slope
  is predicted negative; by fact (i) that slope is the same number at every kappa_target;
- the C2 beta grid {0.10, 0.20, 0.28, 0.35, 0.50} applied to the primary regression.

**Degenerate CO designs.** 25 of the 1250 CO designs have a merged ratio list of length 1, so
V_K = 0 and m_CO = -infinity. These are the most collinear designs in the corpus and dropping
them would discard the extreme of the regressor. Primary rule: winsorise a non-finite m_CO to
the minimum FINITE m_CO over the analysis set. This is conservative in the theory's own
direction, since it shrinks the largest Delta values toward the bulk and biases beta_delta
toward zero if the theory is right. The complete-case version, dropping those pairs, is
reported alongside with its count.

### A3.2 DOSE-RESPONSE over the kappa grid (pre-registered secondary)

For EVERY target in the pre-registered C3 grid {1e2, 1e4, 1e5, 1e6}, and for each of the four
joint regime cells (CO=A & NC=B, the separated cell the theory speaks to; CO=A & NC=A;
CO=B & NC=B; CO=B & NC=A, the reversed cell), report: number of pairs, median gap with
percentile bootstrap 95% CI, and NC win rate. The FULL curve is reported, including targets
where the criterion separates nothing and cells where the gap is null or negative. No target
is selected, and no target may be nominated as "the" answer after the fact. The claim to be
evaluated is a monotone trend in the separated cell across the grid, not the value at any one
target.

### A3.3 REGIME-CONDITIONAL test, as Theorem 1 states it (pre-registered secondary)

(a) *Confirmatory arm.* Among pairs the criterion separates (CO in Regime A, NC in Regime B),
    test the predicted ordering gap > 0: median gap with bootstrap CI, win rate, and a sign
    test. Reported at every kappa in the grid, never at a selected one.

(b) *Falsification arm.* Among pairs where BOTH arms are in the SAME regime, the theory
    predicts no systematic ordering: both-A means both designs are under-identified and
    Theorem 1 ranks neither; both-B means both are certified. A median gap whose CI EXCLUDES
    zero in either same-regime cell is a failure of this test and is reported as a failure, in
    the summary, with its sign and magnitude. This is a test the theory can lose, and it is
    run and reported at every kappa in the grid.

### A3.4 SCALE, on the same conditional footing (pre-registered secondary)

The scale question is re-asked of the conditional effect, not of the nominal one:

    gap_i = b0 * Delta_i + b1 * (Delta_i * log10 N_gm_i) + b2 * log10 N_gm_i
            + sum_c delta_c * 1[cell_i = c] + e_i

b1 is the size interaction of interest: the change per decade of N in how strongly an
identifiability deficit is punished off-ray. Its 95% CI comes from the same window-cluster
bootstrap. Reported alongside: beta_delta estimated separately inside each coarse stratum of
W2, and the A3.3 confirmatory and falsification arms inside each coarse stratum. Per E3, a
non-significant b1 alone is not evidence of scale invariance; the CI is reported and its most
adverse endpoint is translated into the largest attenuation the data cannot exclude. Where a
stratum cell rests on one window or a handful of pairs, the pair and window count is printed
next to the estimate and the estimate is labelled as not established.

### A3.5 Scope of this amendment

Nothing upstream of the regressor changes. The suite inclusion rule (P1-P4), the cells, the
windows (W1-W2), the off-ray holdout (W3), the CO and NC constructions (W5-W6), the laws
(C4), the loss columns (C5), the seeds (C6) and both existing NC arms are untouched, and no
run, cell, suite, window, law, seed or target is added or removed. `results.csv` from the
Amendment 1-2 execution is reused unchanged; the margins at the C2 beta grid are recomputed
from the same designs, reconstructed deterministically by the same code path, and the
reconstruction is verified against the stored V_K, tau_K and k-range of every design before
any outcome is read. Every specification listed in A3.1-A3.4 is run and reported, including
the null ones.

**Amendment 4 (2026-08-01): the candidate list is reopened, and P2 is relaxed to admit
mid-training checkpoints as (N, D) observations. Written BEFORE any new suite was fitted,
carved, screened for outcome, or entered into any regression.**

*Why the list is reopened at all.* The Amendment 3 execution answered the question it was
built for, but it exposed a defect the pre-registration could not have anticipated: the
pooled, no-fixed-effect slope is +0.033 [-0.004, +0.064] against a within-suite slope of
+0.131 [+0.089, +0.163]. The effect is currently within-suite only. With five cells - three
of which are Gadre cells resting on one window each - there is no way to tell whether the
pooled null is genuine between-suite heterogeneity or an artifact of having three suites.
That is a question about the CORPUS, not about the specification, and the only way to answer
it is to add suites. This amendment therefore adds data and changes no estimator. The
specifications of A3.1-A3.4 are run unchanged on the enlarged corpus, and the existing
five-cell result is re-reported alongside so the two can be compared line for line.

*What is NOT permitted by this amendment.* No suite may be added, dropped, or re-screened on
the basis of the gap, the slope, or any fitted quantity. Every candidate examined is logged
in `screening.csv` with the rule it passed or failed, whether or not it is used. The primary
regression, the outcome, the regressor, the laws, the seeds, the windows, the holdout and
both NC arms are unchanged. No specification is added that was not already in A3.1-A3.4,
except the two disclosed below (A4.4, A4.5), both of which are defined here before any new
number exists.

### A4.1 P2 is replaced by P2'

P2 excluded mid-schedule checkpoints for two stated reasons: they do not lie on the converged
loss surface, and their D-variation is temporally autocorrelated. Both reasons are real; both
are handled rather than ignored.

**P2' (replaces P2).** An observation is admissible if the released number is a genuine
language-modelling loss - a cross-entropy in nats per token, or a perplexity convertible to
one by a logarithm - evaluated on a stated corpus at a stated token count D for a model of
stated parameter count N. A mid-training checkpoint qualifies. A downstream benchmark
accuracy, a bits-per-byte figure on a task, a cumulative or unnormalised counter, or any
quantity whose scale is inconsistent with a per-token cross-entropy does not, and is excluded
under P1 as before.

The justification, stated before any outcome:

(i) *A checkpoint is a real observation of the object the law describes.* The four law forms
    in `laws.py` map (N, D) to a loss. A checkpoint gives exactly that triple. Whether the
    schedule had finished is a statement about the RECIPE, not about whether the observation
    exists. Every scaling-law fit in the literature that uses a token-budget sweep at fixed
    architecture is fitting the same kind of point.

(ii) *The deformation is common to both arms.* A decaying learning rate leaves an
    intermediate checkpoint above the loss a run terminated and annealed at the same D would
    have reached. That offset deforms the loss surface. But CO and NC are carved from the SAME
    core, fitted on the same deformed surface, and scored on the SAME holdout drawn from it.
    The deformation changes the target function; it does not enter the CO-versus-NC contrast
    as a between-arm difference. It is a misspecification term, and the paper's own
    misspecification extension (`eqn:mse_ratio_misspec`) predicts that misspecification
    COMPRESSES the measured ratio toward 1 while preserving the ordering. Admitting these
    suites therefore makes the test harder for the theory, not easier.

(iii) *P2 itself said so.* The original rule records that "the paper's Section 5 argues
    that checkpoint autocorrelation biases in the direction that UNDERSTATES the pathology, so
    admitting such suites could only help us". A rule that excludes evidence on the ground
    that the evidence is conservative against the excluder is a rule worth relaxing, and it is
    relaxed in the direction that can only cost the paper.

(iv) *One admitted suite is not even mid-decay.* The Gemstones suite trains at a constant
    learning rate with separate cooldown branches. Its "hot" checkpoints are all drawn from a
    single constant-LR phase, so P2's decaying-LR premise does not apply to them at all, and
    its "cool_end" checkpoints are the ends of fully annealed cooldown branches, which are
    terminal observations admissible under the ORIGINAL P2 without any amendment.

**A4.1a Trajectory thinning (the autocorrelation control), fixed now.** Within each run,
retain at most **T = 12** checkpoints: the run's final checkpoint, plus the checkpoints
nearest to 11 further points spaced log-uniformly in D across the run's own D-range,
deduplicated. T = 12 is chosen from the carving rule, not from any outcome: a window carries
5 to 8 model sizes (W1) and needs at least 3 factor-2 k-bins populated at 4 or more sizes
(P3), and 12 log-spaced points across the 1.5 to 3 decades a trajectory spans yields 5 to 10
populated bins per size, which is the same bins-per-size density the already-admitted Farseer
cell has. It also caps each trajectory cell near the size of the existing cells, so no single
dense run can dominate a window. Thinning is applied to the D axis only and never looks at a
loss.

**A4.1b Cells already admitted under P2 are NOT reconstructed.** Farseer, the digitized
Chinchilla sweep and the three Gadre cells keep their pre-registered end-of-run construction
in every corpus below, even where ColPret also carries their full trajectories. This
amendment adds cells; it does not rebuild existing ones. Where a newly admitted suite has
both a terminal slice and a trajectory slice of the SAME runs, only one of them is used - the
terminal slice if it clears P3, otherwise the thinned trajectory - so no run is counted twice.

### A4.2 Two corpora, both reported in full

Every specification of A3.1-A3.4 is run twice and both results are printed:

- **E-corpus** - only cells whose observations are terminal, i.e. admissible under the
  ORIGINAL P2. This is the pre-registered rule, enlarged only by suites that always satisfied
  it and were missed in Phase 1. The E-corpus is the primary corpus.
- **X-corpus** - the E-corpus plus every cell that exists only because of P2'. Reported
  alongside, never instead.

If the two corpora disagree, the disagreement is the finding and is reported as such in the
summary of RESULTS.md, not in a footnote.

### A4.3 Loss column (extends C5, fixed before any new loss value is read)

- A suite releasing perplexity rather than cross-entropy enters as `loss = ln(perplexity)`.
- A suite releasing a PANEL of validation-set losses (the Ai2 suites release the same eleven)
  uses **`c4_en-validation`**, chosen because C4 is the one validation corpus common to every
  such suite here, so the loss definition is held constant across those cells. The
  `pile-validation` column is the pre-registered sensitivity variant.
- Gemstones: the `final_loss` column of the released wandb dataframes (the suite's own
  validation loss). Its FineWeb-Edu and DCLM validation losses are sensitivity variants, not
  additional cells, because they are the same runs measured on another corpus and entering
  them as cells would be pseudo-replication.
- MAD releases one architecture-family column implicitly through `model_name`; P4 requires one
  architecture family per cell, so MAD splits into a LLaMA cell and a Striped-Hyena cell. This
  is P4 applied, not P4 loosened.

### A4.4 Suite-weighted pooled slope (disclosed companion, defined before any result)

DataDecide contributes one cell per data recipe, so a single suite can supply more cells than
all others combined. The pooled no-fixed-effect slope of A3.1 is therefore reported twice:
once unweighted, exactly as pre-registered, and once with observations weighted so that each
SUITE contributes equally regardless of how many cells or pairs it has. Both are printed.
Neither is nominated as "the" answer; the unweighted one remains the pre-registered primary.

### A4.5 Per-suite slopes are reported for every cell

A3.1 already pre-registers a per-cell regression. With more cells this becomes the main
vehicle for seeing heterogeneity, so the full per-cell table is printed and written to
`conditional_primary.csv` for every cell in both corpora, including the cells whose slope is
negative and the cells that rest on a single window. No cell is summarised away.

**Integrity note on the DataDecide reconstruction (2026-08-01).** DataDecide releases a step
index, not a token count, so D is reconstructed as `step x batch(size) x 2048` from the
suite's own published constants. A validity check against the suite's own published full
step count caught 48 of the 1103 released groups - all seed-14 / seed-15 groups at 150M and
300M - whose step values run up to 11x that schedule, which would place a 150M model at 1097
tokens per parameter when every DataDecide run is a 5xC (about 100 tokens per parameter) run.
For those rows the `step` field is not the quantity the reconstruction assumes, so D is not
recoverable and they fail P1. They are dropped, the count is logged in `screening_sources.csv`,
and the X-corpus was refitted from scratch. The check is on (N, D) alone. It was applied
before any X-corpus outcome was read: the first X-corpus fit had completed but its
`conditional.py` stage had not, and no X-corpus estimate, gap, slope or table from the
pre-repair run was inspected. The E-corpus contains no DataDecide cell and is unaffected.

**Integrity note (2026-08-01).** The first execution of `analysis.py` carried a defect:
`Split.from_window` reset the dataframe index, so the row labels handed to the fitter no
longer addressed the rows they were meant to. Every design in that run was therefore an
arbitrary set of rows rather than the carved CO / NC subsets, and its output is discarded,
not reported. The defect was caught by checking that a single-ray CO design spans at most a
factor of 2 in k; it spanned a factor of 63. `analysis.py` now asserts that invariant plus
budget-matching on every run and refuses to write `results.csv` if either fails.

**Amendment 5 (2026-08-02): the >=1B stratum estimate is tightened without changing the
pre-registered specification. Written BEFORE any enlarged-seed fit was run and BEFORE any
number produced by this amendment was read.**

*Why.* The scale test asks for intervals that exclude material attenuation with increasing N.
The E-corpus within-stratum estimate of A3.4 is `beta_delta = +0.089 [-0.082, +0.153]` on 590
pairs from 14 windows: the interval contains zero and the band is the load-bearing weakness of
the whole analysis. The stated cause is that the >=1B region supports only 14 comparison windows
and that the window is the correct clustering unit. Before that explanation is accepted, four
avoidable sources of width are removed and three alternative estimators are pre-declared. This
amendment adds no run, no suite, no cell and no observation, and drops nothing.

*What does not change.* The A3.4 within-stratum estimate computed from `results_E.csv` and
`margins_E.csv` at the pre-registered 5 seeds, 2000 resamples and W1 windows remains the
primary and is re-reported unchanged next to everything below. Nothing here may be nominated
as "the" answer in its place.

### A5.0 Which parts of the design are enumerated and which are sampled

A design-metadata probe run before this amendment was written, reading (N, D) only and no
loss, establishes the following. CO is exhaustive: W5 enumerates every admissible ray bin and
chooses none. NC and NCM are not. `select_nc_bbox` is randomised in three places (the centre
choice on an even grid axis, the two `_priority_subsample` shuffles, and the `rng.random()`
ring tie-break), and over 512 seeds it reaches between 1 and 239 distinct NC carves per
(cell, window, ray), with new carves still appearing at seed 511 in 51 of the 60 >=1B
combinations. The pre-registered 5 seeds are therefore a small sample of the reachable carve
set, not an enumeration of it. The seed also drives the fitter (fit seed
`42 + 100003 * seed`), so it carries multi-start noise as well as carve noise.

### A5.1 Enumeration: seeds 0-511 in place of seeds 0-4

Every E-corpus window is re-fitted at seeds 0 through 511, a factor of 102. Nothing else
changes: same `enumerate_pairs`, same `select_nc_bbox`, same `compute_VK` / `compute_tauK` /
`classify_regime`, same laws, same 80 restarts, same holdout, same cells. Two economies, both
declared here: only the single-ray CO of the primary is re-fitted (the two-ray secondary of
W5 is not), and only the off-ray arm is re-fitted (the on-ray secondary of Amendment 1 is not
recomputed and stands as it is). The enlarged run must reproduce `results_E.csv` bit for bit
on seeds 0-4, and refuses to write if it does not.

Reported: a seed-count convergence curve at 5, 10, 20, 50, 100, 200 and 512 seeds, so it is
visible how much of the point estimate and of each interval endpoint was enumeration noise;
and a companion in which each DISTINCT NC carve contributes once regardless of how many seeds
reach it.

### A5.2 Bootstrap resamples: 2000 -> 200000

The pre-registered `N_BOOT = 2000` puts a Monte Carlo error of its own on a 2.5 percent
percentile. Every interval below is recomputed at 200000 resamples and both endpoints are
printed before and after, so Monte Carlo error is separated from sampling variance.

### A5.3 Window construction, and the overlap the pre-registered bootstrap ignores

W1 stays the primary. Five further constructions are run and all are reported:

    V1 = width 8, stride 1      V2 = width 8, stride 8      V3 = width 5, stride 5
    V4 = width 5, stride 1      V7 = width 4, stride 4

W1's own windows overlap: width 8 with stride 2 means consecutive windows share 6 of their 8
model sizes, and the E2 cluster bootstrap resamples them as if they were independent. That is
anti-conservative, and it is the pre-registration's defect, not a choice made now. Every
construction is therefore also given a second interval from a **moving-block bootstrap over
each cell's ordered window sequence**, block length `L = ceil(width / stride)`, which is the
number of mutually overlapping windows (windows L apart in the sequence share no run). For a
disjoint construction L = 1 and the moving-block bootstrap coincides with the pre-registered
one. Both intervals are printed for every construction; the wider one is not hidden.

### A5.4 Alternative estimators, each declared here before its outcome

(a) **Pair-level clustering.** The same conditional slope with the bootstrap resampling design
    pairs rather than windows. This is **anti-conservative**: pairs inside a window are fitted
    from the same core, scored on the same holdout points, and drawn from the same runs, so
    they are not independent draws. It is reported to quantify how much of the width is
    between-window heterogeneity, and it is labelled as not admissible as the headline.

(b) **(cell, window, ray) clustering.** Intermediate: CO training sets in different ray bins of
    one window are disjoint, but the holdout and the NC pool are shared. Anti-conservative
    relative to window clustering, conservative relative to (a). Reported and labelled.

(c) **Strength borrowed across bands (slope smooth in log N).** Instead of three independent
    within-stratum estimates, one model over all pairs in which the conditional slope varies
    smoothly with size,

        gap = (b0 + b1*(logN - c) [+ b2*(logN - c)^2]) * Delta + b3*logN + cell FE,

    fitted linear and quadratic in logN. The >=1B statement is then the fitted slope evaluated
    at the >=1B pair-weighted centroid of logN, with its interval taken from the same
    window-cluster bootstrap applied to that linear combination, and the same curve is also
    read off at log10 N = 9.0, 9.5 and 10.0 so the size dependence is visible rather than
    summarised. This is a modelling assumption (smoothness) traded for precision, and it is
    reported as such.

(d) **One-sided and equivalence-style bounds, answering the scale question directly.**
    A one-sided 95 percent lower bound on beta_delta at >=1B; the bootstrap distribution of the
    attenuation ratio `rho = beta(>=1B) / beta(<1B)` and of the difference
    `beta(<1B) - beta(>=1B)`; and, from those, the largest attenuation the data exclude at the
    5 percent level. Each of these is computed twice, once from the unpooled within-stratum
    estimator and once from the smooth-in-log-N estimator of (c), because the two make different
    assumptions and the reader is entitled to both. All are stated in the analysis' own units
    (percent extra off-ray error per tenfold shortfall in V_K relative to tau_K, `10^beta - 1`).

### A5.5 Reporting rule

Every specification in A5.1-A5.4 is run and reported, including the ones that come out null and
the ones that come out wider than the pre-registered interval. No pair, window, ray, seed, cell
or suite is dropped, added or re-screened after a result is seen. If after all of it the >=1B
interval still contains zero, that is the finding, it is stated plainly in the summary, and the
tightest defensible statement is reported as a bound rather than as an effect.

Code and outputs added by this amendment: `tighten1b_probe.py` (the A5.0 metadata probe) with
`tighten1B_carvespace.csv`; `tighten1b_fit.py` with `results_E_seeds512.csv` and its
window-variant siblings `results_E_win<V>.csv`; `tighten1b_boot.py` and
`tighten1b_estimate.py` with `tighten1B_estimators.csv`, `tighten1B_seedcurve.csv`,
`tighten1B_windows.csv` and `tighten1B_estimate.log`; and section 8 of `RESULTS_PHASE3.md`.
No existing output is overwritten.

**Amendment 6 (2026-08-02): the enumeration depth of Amendment 5 is applied to EVERY reported
quantity rather than to the >=1B band alone, and the under-powered <100M band is diagnosed and
bounded. Written BEFORE any Amendment-6 fit was run and BEFORE any number produced by this
amendment was read.**

*Why.* Amendment 5 established that `select_nc_bbox` is randomised and that the pre-registered
5 seeds sample a median of 5 of the 24 distinct non-collinear carves reachable per
(cell, window, ray). It then re-enumerated only the >=1B band. The result is an analysis whose
numbers rest on two different enumeration depths: the >=1B band at 512 seeds and everything
else at 5. That is not a defensible way to report a set of estimates that are compared with
one another - the >=1B band is quoted next to the <1B bands, the per-suite slopes and the
dose-response, and a difference between them could be a difference in enumeration depth. This
amendment removes that confound by re-running everything at one depth. It adds no suite, no
cell, no window, no ray and no observation, changes no estimator, and drops nothing.

The second defect is the <100M band. Section 4 of `RESULTS_PHASE3.md` reports it as
`+0.338 [+0.316, +0.376]` on 50 pairs from 2 windows and labels it not established, and
section 8.2 reports `+0.3978 [+0.3064, +0.3978]` at 512 seeds with a degenerate upper endpoint.
An interval that narrow resting on two clusters inside a single cell is not a measurement, and
quoting it next to the other two bands invites the reader to treat it as one.

### A6.0 What was inspected before this amendment was written (blindness disclosure)

`phase4_probe.py` reads (N, D) only - no loss value, no fit, no gap - and reports, per cell,
the distinct-N inventory below 100M, and, per candidate window construction, how many windows
and how many admissible (cell, window, ray) design families each coarse stratum carries. Its
output (`phase4_sizeinventory.csv`, `phase4_windowspace_E.csv`, `phase4_windowspace_X.csv`,
`phase4_probe.log`) is the only thing used to choose the constructions listed in A6.2. No
outcome was read.

### A6.1 Enumeration: seeds 0-511 for every reported quantity, both corpora

Every quantity the analysis reports is recomputed at seeds 0-511:

- the A3.1 primary continuous slope, cell-FE and pooled no-FE, plus the A3.1 and A4.4
  companions;
- the A4.5 per-suite and per-cell slopes, and the post-hoc leave-one-suite-out refits;
- the DataDecide-only within-suite slope;
- all three coarse strata of W2 (<100M, 100M-1B, >=1B) and the A3.4 interaction;
- the A3.2 dose-response over the whole pre-registered kappa grid {1e2, 1e4, 1e5, 1e6},
  every one of the four joint regime cells, including the separated cell;
- the A3.3b same-regime falsification arm at every kappa;
- the X-corpus versions of all of the above, and the post-hoc k >= 1 (MAD-excluded)
  restriction of `RESULTS_PHASE3.md` section 6.

Nothing else changes: same `enumerate_pairs`, same `select_nc_bbox`, same `compute_VK` /
`compute_tauK` / `classify_regime`, same laws, same 80 restarts, same holdout, same cells,
same C2/C3 grids. Four things are fixed or declared here:

**(a) The two economies of A5.1 are inherited and are NOT quietly widened.** Only the
single-ray CO of the primary and only the off-ray arm are re-enumerated. The two-ray CO
secondary of W5 and the on-ray secondary of Amendment 1 stay at the pre-registered 5 seeds,
and every table that reports them says so.

**(b) The X-corpus is fitted at 512 seeds for the first time**, with `tighten1b_fit.py`
unchanged except for (c). The E-corpus reuses `results_E_seeds512.csv` from Amendment 5,
which already covers every window of every E cell and not only the >=1B ones.

**(c) A W1 construction defect, fixed in the direction of the pre-registration.**
`tighten1b_fit.py` implemented W1 as the single pair (width 8, stride 2). CRITERIA W1 is not a
single pair: it prescribes (8, 2) only for a cell with >= 12 distinct N and (5, 1) for a cell
with 8 <= #N < 12. No E-corpus cell falls in that band, so nothing already computed changes;
one X-corpus cell (gpt3, 8 distinct N) does, and it would otherwise have been fitted under a
construction W1 does not prescribe. `tighten1b_fit.build_windows` now delegates W1 to
`analysis.build_windows`. This is a correction toward the pre-registered rule, not a new
choice, and it was made before any X-corpus 512-seed number existed.

**(d) Both bootstrap depths are shown so the seed axis is isolated.** Each quantity is
reported at 5 seeds and at 512 seeds. The 5-seed column is quoted from the existing
pre-registered run (B = 2000) and is ALSO recomputed on seeds 0-4 of the enlarged file with
the A5.2 engine at B = 200000, so the 5-versus-512 comparison differs in seeds alone. Any
estimate whose interval contains zero at one depth and excludes it at the other is called out
explicitly, in the summary and not in a footnote.

### A6.2 The <100M band: diagnose, widen the construction, or bound it honestly

The pre-registered W1 <100M estimate stays the primary and is re-printed unchanged next to
everything below. Nothing here may be nominated as "the" answer in its place.

**A6.2a Every construction, restricted to <100M, all reported.** The five A5.3 constructions
(V1 = 8/1, V2 = 8/8, V3 = 5/5, V4 = 5/1, V7 = 4/4) are run at <100M exactly as they were run
at >=1B, at 512 seeds, and W1 is re-printed alongside. One construction is added:

    V5 = width 4, stride 1

Width 4 is the NARROWEST admissible window in this design, not a tuned choice: CRITERIA W5
requires a CO design to carry at least `MIN_DISTINCT_N_IN_DESIGN = 4` distinct model sizes, so
a window of 3 sizes can never produce a single admissible design pair. V5 is therefore the
floor of the construction family, and it is declared here so that the family is bounded from
below rather than extended until something works. No construction beyond this list may be
added later.

*Compute economy, declared before the runs.* The variant constructions are fitted at 512
seeds on their **<100M windows only**, because that is the band A6.2 exists to bound and the
Amendment-5 execution already reports every construction at >=1B. Windows are carved and
scored independently inside each (cell, window), so restricting which windows are fitted
changes no <100M number; it only means the variant files carry no 100M-1B or >=1B rows and
the A5.3 table above is not superseded. The pre-registered W1 is fitted over the whole corpus
at 512 seeds in both corpora, as A6.1 requires.

**A6.2b Three intervals per construction, all printed.**

1. the pre-registered E2 window-cluster bootstrap (resample windows within cell);
2. the A5.3 overlap-aware moving-block bootstrap over each cell's ordered window sequence,
   block length L = ceil(width / stride);
3. a **cell-level** cluster bootstrap that resamples CELLS with replacement, each carrying
   all of its windows.

(3) is added because of a structural fact about this band, stated here before any outcome:
when most cells contribute a single window, the pre-registered within-cell resample has
nothing to vary, and its interval is narrow for a reason that has nothing to do with the
precision of the estimate. Where windows-per-cell is 1 in most cells, (3) is the honest
interval and (1) is the anti-conservative one; both are printed and the wider is not hidden.
The primary specification is not changed by this: (1) remains what W1 reports.

**A6.2c Borrowed strength, labelled as assumption-laden.** The A5.4(c) smooth-in-log-N
estimator is read off at the <100M pair-weighted centroid and at log10 N = 7.5 and 8.0, with
the same window-cluster bootstrap applied to that linear combination. It buys precision with a
smoothness assumption and is reported as such, exactly as it was at >=1B.

**A6.2d The X-corpus <100M band is the independent check.** It is reported at 512 seeds
alongside the E-corpus one. Where the two disagree, the disagreement is the finding.

**A6.2e Reporting rule, fixed now.** If after all of A6.2a-A6.2d no construction produces a
<100M band with more than one cell contributing more than one window, that is stated plainly
in the summary, the band is reported as **not established** with its window and cell count
printed next to it, and the honest statement is given as a bound - the widest of the three
intervals, or a sign-only statement - rather than as the two-cluster interval. Every
construction run is reported, including the ones that produce no <100M window at all and the
ones that come out null.

### A6.3 Scope

No pair, window, ray, seed, cell or suite is dropped, added or re-screened after a result is
seen. The primary specification of A3.1, the outcome, the regressor, the laws, the holdout,
both NC arms, the coarse strata edges of W2 and the kappa and beta grids are untouched. Every
specification listed in A6.1 and A6.2 is run and reported, including the null ones and the
ones that come out wider than what they replace.

Code and outputs added by this amendment: `phase4_probe.py` (the A6.0 metadata probe) with
`phase4_sizeinventory.csv`, `phase4_windowspace_E.csv`, `phase4_windowspace_X.csv` and
`phase4_probe.log`; the V5 entry and the W1 delegation in `tighten1b_fit.py`;
`results_X_seeds512.csv.gz` and the <100M variant files `results_E_win<V>_small512.csv.gz`;
`phase4_chain.sh` with `phase4_chain.log`, `phase4_<V>.log`, `phase4_X_W1.log` and
`phase4_load.log`; `phase4_estimate.py` with `phase4_seedcompare.csv`,
`phase4_doseresponse.csv`, `phase4_small.csv` and `phase4_estimate.log`; and
`RESULTS_PHASE4.md`. No existing output is overwritten.

## Outputs

- `results.csv` - one row per (cell, window, bin, design-pair, seed, law) with n, the k
  list, V_K/tau_K/regime for both designs, RMSE on and off ray for both designs, and gap.
- `strata.csv` - per-window and per-stratum medians with bootstrap CIs.
- `RESULTS.md` - the finding, including the interaction estimate and its CI, reported
  whichever way it comes out.
- `README.md` - the Phase-1 suite inventory and reproduction instructions.

Amendment 4 adds, for each corpus C in {E, X}: `runs_expanded.csv` (the normalised expanded
run table, with a `corpus` column), `screening_sources.csv` (every candidate SOURCE examined,
in or out, with the rule), `screening_C.csv` (every cell, with the P3 verdict),
`results_C.csv`, `margins_C.csv`, `conditionalC_primary.csv`,
`conditionalC_doseresponse.csv`, `conditionalC_scale.csv`, and `heterogeneity_C.csv` (the
post-hoc leave-one-suite-out diagnostic, labelled as post-hoc where it is reported).
