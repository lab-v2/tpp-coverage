"""Proposition-2 classification of published LLM scaling-law experimental designs.

Applies the paper's a-priori design diagnostic (V_K >= tau_K, Prop. 2 / Eq. 5 of
"Tokens-per-Parameter Coverage Is Critical for Robust LLM Scaling Law Extrapolation")
to every included published design, under the conventions pre-registered in CRITERIA.md.

Primary convention since 2026-07-31 (CRITERIA.md convention 9): each design is
evaluated at ITS OWN paper's fitted data-side exponent (designs.BetaEff), because
tau_K depends on beta_eff and every paper fits its own law with its own exponent.
The former corpus-wide beta_eff = 0.28 is retained as a comparison column, and the
pre-registered beta_eff grid remains the sensitivity check.

The diagnostic functions (compute_VK, compute_tauK, classify_regime) are imported from
the paper's own code, src/tpp_analysis/subset_analysis/subset_analysis_bbox_seeded.py.

Usage (from the repository root):
    uv run python camera_ready/grid_classification/classify.py            # rewrites results.csv
    uv run python camera_ready/grid_classification/classify.py --out x.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from designs import DESIGNS, LITERATURE_BETA_EFF, Design

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src" / "tpp_analysis" / "subset_analysis"))

from subset_analysis_bbox_seeded import (  # noqa: E402
    classify_regime,
    compute_tauK,
    compute_VK,
)

CANDIDATES_CSV = HERE / "candidates.csv"
RESULTS_CSV = HERE / "results.csv"

# Conventions pre-registered in CRITERIA.md (before any paper was collected),
# amended 2026-07-31 to make the per-paper exponent primary (convention 9).
COMPARISON_BETA_EFF = LITERATURE_BETA_EFF   # 0.28, the old corpus-wide convention
PRIMARY_KAPPA_TARGET = 1e2       # paper's main-text convention (empirical.tex line 138)
BETA_EFF_GRID: tuple[float, ...] = (0.10, 0.20, 0.28, 0.35, 0.50)
KAPPA_TARGET_GRID: tuple[float, ...] = (1e2, 1e4, 1e5, 1e6)
NEAR_DUP_REL_TOL = 0.01          # CRITERIA.md convention 4


# ---------------------------------------------------------------------------
# Pre-registered conventions (CRITERIA.md)
# ---------------------------------------------------------------------------

def merge_near_duplicates(ks: tuple[float, ...],
                          rel_tol: float = NEAR_DUP_REL_TOL) -> tuple[float, ...]:
    """CRITERIA.md convention 4: adjacent ratios within rel_tol relative
    difference are one distinct ratio, merged to their geometric mean."""
    if not ks:
        raise ValueError("empty ratio list")
    if any(k <= 0 for k in ks):
        raise ValueError(f"non-positive ratio in {ks}")
    ordered = sorted(ks)
    groups: list[list[float]] = [[ordered[0]]]
    for k in ordered[1:]:
        if (k - groups[-1][-1]) / groups[-1][-1] <= rel_tol:
            groups[-1].append(k)
        else:
            groups.append([k])
    return tuple(float(np.exp(np.mean(np.log(g)))) for g in groups)


@dataclass(frozen=True)
class BetaVerdict:
    """The diagnostic evaluated at one beta_eff, at the primary kappa_target."""

    beta_eff: float
    V_K: float
    tau_K: float
    verdict_pass: bool
    kappa_verdicts: dict[float, bool]

    @classmethod
    def evaluate(cls, ks: tuple[float, ...], beta_eff: float) -> "BetaVerdict":
        vk = float(compute_VK(ks, beta_eff))
        tk = float(compute_tauK(ks, beta_eff, PRIMARY_KAPPA_TARGET))
        return cls(
            beta_eff=beta_eff,
            V_K=vk,
            tau_K=tk,
            verdict_pass=vk >= tk,
            kappa_verdicts={
                kt: bool(vk >= compute_tauK(ks, beta_eff, kt))
                for kt in KAPPA_TARGET_GRID
            },
        )

    @property
    def regime(self) -> str:
        return classify_regime(self.V_K, self.tau_K)

    @property
    def flips_on_kappa_grid(self) -> bool:
        return len(set(self.kappa_verdicts.values())) > 1

    @property
    def passes_at_most_lenient_kappa(self) -> bool:
        return self.kappa_verdicts[max(KAPPA_TARGET_GRID)]


@dataclass(frozen=True)
class Classification:
    """One included design, classified at its own beta_eff plus the 0.28 comparison."""

    design: Design
    k_distinct: tuple[float, ...]
    K: int
    own: BetaVerdict
    literature: BetaVerdict
    beta_probe_verdicts: dict[float, bool]

    @classmethod
    def from_design(cls, design: Design) -> "Classification":
        ks = merge_near_duplicates(design.k_list)
        own_beta = design.beta_eff.value
        # Sensitivity: the pre-registered grid, widened to include this design's
        # own exponent so a row whose beta_eff falls outside 0.10-0.50 is still
        # probed at the value that actually decides its primary verdict.
        probe_betas = sorted({*BETA_EFF_GRID, own_beta})
        return cls(
            design=design,
            k_distinct=ks,
            K=len(ks),
            own=BetaVerdict.evaluate(ks, own_beta),
            literature=BetaVerdict.evaluate(ks, COMPARISON_BETA_EFF),
            beta_probe_verdicts={
                b: BetaVerdict.evaluate(ks, b).verdict_pass for b in probe_betas
            },
        )

    @property
    def verdict_changed(self) -> bool:
        return self.own.verdict_pass != self.literature.verdict_pass

    @property
    def flips_on_beta_grid(self) -> bool:
        return len(set(self.beta_probe_verdicts.values())) > 1

    @property
    def fails_at_every_probed_beta(self) -> bool:
        return not any(self.beta_probe_verdicts.values())

    @property
    def exponent_circularity(self) -> bool:
        """Failing row whose beta_eff came from the very fit called ill-conditioned."""
        return (not self.own.verdict_pass) and self.design.beta_eff.from_classified_fit


# ---------------------------------------------------------------------------
# Candidate log reconciliation and outputs
# ---------------------------------------------------------------------------

def load_candidates() -> list[dict[str, str]]:
    with CANDIDATES_CSV.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def reconcile(candidates: list[dict[str, str]],
              designs: tuple[Design, ...]) -> None:
    """Every included candidate must have exactly one Design row and vice versa."""
    included_keys = {c["key"] for c in candidates if c["decision"] == "include"}
    design_keys = {d.key for d in designs if not d.is_calibration}
    if included_keys != design_keys:
        missing = included_keys - design_keys
        extra = design_keys - included_keys
        raise RuntimeError(
            f"candidates.csv and designs.py disagree: missing designs for "
            f"{sorted(missing)}; designs without include rows: {sorted(extra)}"
        )


def _verdict(passed: bool) -> str:
    return "pass" if passed else "fail"


def write_results(rows: list[Classification], out: Path = RESULTS_CSV) -> None:
    own_kappa_cols = [f"verdict_kappa_{kt:g}_own_beta" for kt in KAPPA_TARGET_GRID]
    lit_kappa_cols = [f"verdict_kappa_{kt:g}_beta0.28" for kt in KAPPA_TARGET_GRID]
    fieldnames = [
        "key", "citation", "arxiv_id", "K", "k_distinct", "k_list_source",
        "protocol", "protocol_source",
        "law_family", "beta_eff_symbol", "beta_eff", "beta_eff_reported",
        "beta_eff_from_classified_fit", "beta_eff_source", "beta_eff_range",
        "V_K_own_beta", "tau_K_own_beta", "verdict_at_own_beta", "regime_at_own_beta",
        "V_K_beta0.28", "tau_K_beta0.28", "verdict_at_0.28", "regime_at_0.28",
        "verdict_changed", *own_kappa_cols, *lit_kappa_cols,
        "beta_probe_grid", "beta_probe_verdicts", "flip_on_beta_grid",
        "fails_at_every_probed_beta", "flip_on_kappa_grid_own_beta",
        "passes_at_most_lenient_kappa_own_beta", "exponent_circularity",
        "is_calibration", "provenance",
    ]
    tmp = out.with_suffix(".csv.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            beta = r.design.beta_eff
            writer.writerow({
                "key": r.design.key,
                "citation": r.design.citation,
                "arxiv_id": r.design.arxiv_id,
                "K": r.K,
                "k_distinct": " ".join(f"{k:g}" for k in r.k_distinct),
                "k_list_source": r.design.k_list_source,
                "protocol": r.design.protocol,
                "protocol_source": r.design.protocol_source,
                "law_family": beta.law_family,
                "beta_eff_symbol": beta.symbol,
                "beta_eff": f"{beta.value:g}",
                "beta_eff_reported": beta.reported,
                "beta_eff_from_classified_fit": beta.from_classified_fit,
                "beta_eff_source": beta.source,
                "beta_eff_range": beta.range_note,
                "V_K_own_beta": f"{r.own.V_K:.6g}",
                "tau_K_own_beta": f"{r.own.tau_K:.6g}",
                "verdict_at_own_beta": _verdict(r.own.verdict_pass),
                "regime_at_own_beta": r.own.regime,
                "V_K_beta0.28": f"{r.literature.V_K:.6g}",
                "tau_K_beta0.28": f"{r.literature.tau_K:.6g}",
                "verdict_at_0.28": _verdict(r.literature.verdict_pass),
                "regime_at_0.28": r.literature.regime,
                "verdict_changed": r.verdict_changed,
                **{f"verdict_kappa_{kt:g}_own_beta": _verdict(r.own.kappa_verdicts[kt])
                   for kt in KAPPA_TARGET_GRID},
                **{f"verdict_kappa_{kt:g}_beta0.28":
                   _verdict(r.literature.kappa_verdicts[kt])
                   for kt in KAPPA_TARGET_GRID},
                "beta_probe_grid": " ".join(
                    f"{b:g}" for b in sorted(r.beta_probe_verdicts)),
                "beta_probe_verdicts": " ".join(
                    _verdict(r.beta_probe_verdicts[b])
                    for b in sorted(r.beta_probe_verdicts)),
                "flip_on_beta_grid": r.flips_on_beta_grid,
                "fails_at_every_probed_beta": r.fails_at_every_probed_beta,
                "flip_on_kappa_grid_own_beta": r.own.flips_on_kappa_grid,
                "passes_at_most_lenient_kappa_own_beta":
                    r.own.passes_at_most_lenient_kappa,
                "exponent_circularity": r.exponent_circularity,
                "is_calibration": r.design.is_calibration,
                "provenance": r.design.provenance,
            })
    tmp.replace(out)


def print_summary(candidates: list[dict[str, str]],
                  rows: list[Classification]) -> None:
    corpus = [r for r in rows if not r.design.is_calibration]
    n_screened = len(candidates)
    reasons: dict[str, int] = {}
    for c in candidates:
        if c["decision"] != "include":
            code = c["reason_detail"].split(":")[0].strip()
            if not code.startswith("excl-"):
                code = "excl-other"
            reasons[code] = reasons.get(code, 0) + 1
    fails = [r for r in corpus if not r.own.verdict_pass]
    passes = [r for r in corpus if r.own.verdict_pass]
    unreported = [r for r in corpus if not r.design.beta_eff.reported]
    flips = [r for r in corpus if r.verdict_changed]

    print("=" * 72)
    print("SUMMARY (primary: each design at ITS OWN paper's beta_eff, "
          f"kappa_target = {PRIMARY_KAPPA_TARGET:g})")
    print("=" * 72)
    print(f"candidates screened : {n_screened}")
    for reason in sorted(reasons):
        print(f"  excluded {reason:<16}: {reasons[reason]}")
    print(f"included (corpus)   : {len(corpus)}")
    print(f"  verdict FAIL (Regime A): {len(fails)}")
    print(f"    of which K = 1        : {sum(r.K == 1 for r in fails)}")
    print("    fail even at most lenient kappa_target: "
          f"{sum(not r.own.passes_at_most_lenient_kappa for r in fails)}")
    print("    non-K=1 fails passing at kappa_target 1e4: "
          f"{sum(r.K > 1 and r.own.kappa_verdicts[1e4] for r in fails)}")
    print(f"  verdict PASS (Regime B): {len(passes)}")
    print("    of which endpoints-mode k-list: "
          f"{sum(r.design.k_list_source == 'endpoints' for r in passes)}")
    print(f"beta_eff not reported by the paper (fell back to "
          f"{COMPARISON_BETA_EFF}): {len(unreported)}"
          + (f" [{', '.join(r.design.key for r in unreported)}]" if unreported else ""))

    print("-" * 72)
    lit_pass = sum(r.literature.verdict_pass for r in corpus)
    print(f"comparison at the old corpus-wide beta_eff = {COMPARISON_BETA_EFF}: "
          f"{lit_pass} pass / {len(corpus) - lit_pass} fail")
    print(f"verdict changes under per-paper exponents: {len(flips)}")
    for r in flips:
        beta = r.design.beta_eff
        print(f"  {r.design.key:<24s} {_verdict(r.literature.verdict_pass)} -> "
              f"{_verdict(r.own.verdict_pass)}   ({beta.symbol} = {beta.value:g})")

    print("-" * 72)
    circular = [r for r in fails if r.exponent_circularity]
    print(f"failing rows whose beta_eff came from the classified fit itself: "
          f"{len(circular)}")
    for r in circular:
        robust = "fails at every probed beta" if r.fails_at_every_probed_beta \
            else "VERDICT DEPENDS ON beta"
        print(f"  {r.design.key:<24s} beta_eff={r.design.beta_eff.value:<7g} {robust}")

    for r in (r for r in rows if r.design.is_calibration):
        print("-" * 72)
        print(f"calibration [{r.design.key}]: K={r.K} "
              f"verdict_at_own_beta={_verdict(r.own.verdict_pass)} "
              f"({r.design.beta_eff.symbol}={r.design.beta_eff.value:g}) "
              f"verdict_at_0.28={_verdict(r.literature.verdict_pass)}")
    print("=" * 72)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Classify published scaling-law designs (Prop. 2).")
    ap.add_argument("--out", type=Path, default=RESULTS_CSV, help="output CSV (default: results.csv here)")
    args = ap.parse_args(argv)
    candidates = load_candidates()
    reconcile(candidates, DESIGNS)
    rows = [Classification.from_design(d) for d in DESIGNS]
    write_results(rows, args.out)
    print_summary(candidates, rows)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
