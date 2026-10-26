"""Recovered experimental designs for every included paper (see CRITERIA.md).

Each entry records the distinct TPP ratio list k = D/N of the training runs
entering one pooled fit (or one slice, for per-slice protocols), its recovery
mode, the fitting protocol, the paper's OWN fitted data-side exponent
(beta_eff), and provenance. Populated from the candidate sweep; inclusion
decisions live in candidates.csv and are outcome-blind (CRITERIA.md).

Artifact-derived k-lists (Hoffmann/Besiroglu, Muennighoff, Farseer) are stored
in artifacts/*.txt with their derivation noted in each file header; the loader
below reads them so the raw run lists stay auditable.

beta_eff (added 2026-07-31, CRITERIA.md convention 9): tau_K depends on the
law-specific data-size exponent, and each paper fits its own law with its own
exponent, so every row now carries the exponent that paper actually reports for
the D term of the additive core it fits (formal_analysis.tex line 222: beta for
Chinchilla/repeated-data, alpha_D for Kaplan, gamma_D for Droppo-Elibol; the
exponent on the D term of the additive core for supersets). Papers that fit a
law but never print a numeric exponent are marked reported=False and fall back
to the corpus-wide LITERATURE_BETA_EFF, flagged as such.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

KListSource = Literal["exact", "axes-derived", "endpoints", "reconstruction"]
Protocol = Literal["pooled", "per-slice"]
ProtocolSource = Literal["stated", "artifact", "inferred"]

# Corpus-wide literature exponent. Was the sole primary convention until
# 2026-07-31 (CRITERIA.md convention 1); now retained as the comparison
# convention and as the fallback for papers that report no exponent.
LITERATURE_BETA_EFF = 0.28

_ART = Path(__file__).resolve().parent / "artifacts"


def _load_k(fname: str) -> tuple[float, ...]:
    lines = (_ART / fname).read_text().splitlines()
    return tuple(float(x) for x in lines if x and not x.startswith("#"))


def _cross(ns: tuple[float, ...], ds: tuple[float, ...]) -> tuple[float, ...]:
    return tuple(sorted(d / n for n in ns for d in ds))


@dataclass(frozen=True)
class BetaEff:
    """One paper's own data-side exponent for the law whose design is classified.

    `value` is the number fed to Prop. 2. `symbol` is the symbol as printed in
    that paper (notations collide across papers - Rosenfeld's data exponent is
    called alpha, Sedova's too, while most Chinchilla-form papers call it beta).
    `from_classified_fit` records whether the exponent was estimated from the
    very design under test: those rows are the ones where the circularity
    objection bites, and their robustness across the beta sensitivity grid is
    what makes them defensible.
    """

    value: float
    symbol: str
    law_family: str
    source: str
    range_note: str = ""
    reported: bool = True
    from_classified_fit: bool = True

    @classmethod
    def not_reported(cls, symbol: str, law_family: str, source: str) -> "BetaEff":
        """Paper fits the law but never prints the exponent: fall back, flagged."""
        return cls(
            value=LITERATURE_BETA_EFF,
            symbol=symbol,
            law_family=law_family,
            source=source,
            reported=False,
            from_classified_fit=False,
        )


@dataclass(frozen=True)
class Design:
    """One included paper's fitted-design record."""

    key: str
    citation: str
    arxiv_id: str
    k_list: tuple[float, ...]
    k_list_source: KListSource
    protocol: Protocol
    protocol_source: ProtocolSource
    beta_eff: BetaEff
    provenance: str
    is_calibration: bool = False


# Calibration row: this paper's own collinear (CO) fan - not part of the
# published-literature corpus or its denominator.
#
# beta_eff for this row: this paper's headline Prop.-2 numbers for the CO fan
# are the Kaplan ones (empirical.tex: V_K ~ 1.6e-3 < tau_K ~ 3.4e-2 at
# alpha_D ~ 0.095), so alpha_D = 0.095 is the matching per-law exponent. The
# verdict does not depend on the choice: the fan is Regime A at the primary
# kappa_target under alpha_D = 0.095 and under the old corpus-wide 0.28
# (V_K ~ 1.0e-2 < tau_K ~ 2.6e-2) alike. Per-kappa behavior: like the other narrow
# multi-ray fans in the corpus, the CO fan clears the criterion at the code
# package's lenient KAPPA_TARGETS; only the main-text kappa_target = 100
# convention fails it. Recorded in the per-kappa columns of results.csv.
_CALIBRATION = Design(
    key="this-paper-CO-fan",
    citation="This paper, collinear design (Section 4)",
    arxiv_id="",
    k_list=(1.0, 1.5, 1.9, 2.0, 2.5, 2.7, 3.0, 3.3, 3.5, 4.0, 4.5, 5.0),
    k_list_source="exact",
    protocol="pooled",
    protocol_source="stated",
    beta_eff=BetaEff(
        value=0.095,
        symbol="alpha_D",
        law_family="Kaplan",
        source="formal_analysis.tex line 79 (Definition 2, exponent gap) and the code "
               "package's LIT_BETA_EFF map (subset_analysis/per_law_winrate_ci.py "
               "line 33). This paper's headline CO-fan diagnostic is the Kaplan "
               "one (empirical.tex, Section 4), whose data-side exponent it takes as "
               "alpha_D ~ 0.095; this paper adopts published per-law exponents "
               "rather than re-fitting one on its own fan.",
        range_note="this paper's other two laws: beta = 0.28 (Chinchilla), "
                   "gamma_D = 0.08 (Droppo-Elibol)",
        from_classified_fit=False,
    ),
    provenance="empirical.tex, K_train definition (Section 4, Experimental designs)",
    is_calibration=True,
)

DESIGNS: tuple[Design, ...] = (
    _CALIBRATION,
    Design(
        key="kaplan2020",
        citation="Kaplan et al. 2020, Scaling Laws for Neural Language Models",
        arxiv_id="2001.08361",
        # Endpoint mode (CRITERIA convention 5): N endpoints 768..1.5e9 non-embedding
        # (stated model range), D endpoints 2.29e10-token WebText2 subsampled down by
        # factor 1024 to ~2.24e7 tokens (Sec 4.2). Corner ratios of the crossed axes.
        k_list=_cross((768.0, 1.5e9), (2.24e7, 2.29e10)),
        k_list_source="endpoints",
        protocol="pooled",
        protocol_source="inferred",
        beta_eff=BetaEff(
            value=0.103,
            symbol="alpha_D",
            law_family="Kaplan",
            source="arXiv:2001.08361 main.tex lines 476-487: the unnumbered 'Fits to "
                   "L(N,D)' table in Sec. 4.2 (Results), i.e. the joint 4-parameter fit "
                   "of Eq. 1.5 over the very (N, D) runs that constitute the classified "
                   "design (alpha_N = 0.076, alpha_D = 0.103).",
            range_note="the univariate L(D) fit (Eq. 1.5 / eq:crit_d, main.tex line 175) "
                       "gives alpha_D ~ 0.095, which is the value usually quoted from this "
                       "paper and the one this paper uses as its Kaplan literature "
                       "estimate.",
        ),
        provenance="Eq 1.5 fit over runs, Fig 9; model range 768-1.5B non-embed (abstract/Sec 3); "
                    "subsets of 2.29e10-token WebText2 reduced up to 1024x to ~2e7 tokens (Sec 4.2). "
                    "Endpoint corners (CRITERIA conv. 5 heuristic: endpoints-mode can over-credit "
                    "but not under-credit the design).",
    ),
    Design(
        key="hoffmann2022",
        citation="Hoffmann et al. 2022, Training Compute-Optimal Large Language Models (Chinchilla)",
        arxiv_id="2203.15556",
        k_list=_load_k("hoffmann_besiroglu_k.txt"),
        k_list_source="reconstruction",
        protocol="pooled",
        protocol_source="stated",
        beta_eff=BetaEff(
            value=0.2849,
            symbol="beta",
            law_family="Chinchilla",
            source="arXiv:2203.15556 main.tex line 894 (Appendix D.2): 'we find after "
                   "fitting that L(N,D) = E + A/N^0.34 + B/D^0.28'; the unrounded value "
                   "0.2849083 sits in the source comment at line 903 (this is the number "
                   "Besiroglu et al. recovered from the TeX source).",
            range_note="the paper's rounded printed value is 0.28 - the corpus-wide "
                       "literature convention descends from this row.",
        ),
        provenance="Approach 3 parametric L(N,D) fit pooled over all runs; run list is a PARTIAL "
                    "RECONSTRUCTION: Besiroglu et al.'s SVG digitization of the paper's Fig 4 scatter "
                    "(epoch-research/analyzing-chinchilla, data/svg_extracted_data.csv, 245 pts of the "
                    "~400 fitted runs); k = (FLOP/6N)/N. Digitization noise ~axis-resolution.",
    ),
    Design(
        key="besiroglu2024",
        citation="Besiroglu et al. 2024, Chinchilla Scaling: A Replication Attempt",
        arxiv_id="2404.10102",
        k_list=_load_k("hoffmann_besiroglu_k.txt"),
        k_list_source="exact",
        protocol="pooled",
        protocol_source="artifact",
        beta_eff=BetaEff(
            value=0.3658,
            symbol="beta",
            law_family="Chinchilla (replication refit)",
            source="arXiv:2404.10102 main.tex line 121 ('Our estimated model': "
                   "L = 1.8172 + 482.01/N^0.3478 + 2085.43/D^0.3658) and Table 1 at "
                   "line 147, which reports beta = 0.3658 with bootstrap SE 0.02 over "
                   "the 240 non-outlier points of the classified design.",
            range_note="the appendix variant that keeps all 245 points (main.tex line 300) "
                       "gives beta = 0.452 with SE 0.054 - a 24 percent swing in the "
                       "exponent from five points, on the same design.",
        ),
        provenance="Refit of Hoffmann Approach 3 on their own released reconstruction "
                    "(epoch-research/analyzing-chinchilla, data/svg_extracted_data.csv); "
                    "same 245-run pooled design.",
    ),
    Design(
        key="rosenfeld2019",
        citation="Rosenfeld et al. 2020, A Constructive Prediction of the Generalization Error Across Scales (ICLR 2020)",
        arxiv_id="1909.12673",
        # WikiText-103 / Transformer-XL fit (representative LM fit; per-dataset pooled fits).
        # Table 1a: data 100M words x 2^-k, k=0..5; model 41M non-embed params x 4^-k, k=0..6.
        k_list=_cross(
            tuple(4.1e7 * 4.0 ** (-j) for j in range(7)),
            tuple(1.0e8 * 2.0 ** (-k) for k in range(6)),
        ),
        k_list_source="axes-derived",
        protocol="pooled",
        protocol_source="stated",
        beta_eff=BetaEff(
            value=0.73505,
            symbol="alpha",
            law_family="Rosenfeld envelope (power-law core a n^-alpha + b m^-beta + c_inf)",
            source="arXiv:1909.12673 main.tex line 1077, Table 5(b) 'Language modeling', "
                   "WikiText-103 row of the fitted-theta table. NOTE the notation swap: in "
                   "this paper n is the DATA size and m the MODEL size (Eq. 5, main.tex "
                   "line 445), so alpha is the data-side exponent and beta the model-side "
                   "one - the reverse of the Chinchilla convention.",
            range_note="the other two LM corpora give alpha = 0.810 (PTB) and 1.008 "
                       "(WikiText-2); the six image-classification fits span 0.400-1.102.",
        ),
        provenance="Table 1a axes (paper PDF): WT103 100M words, scales 2^-k N k=0..5; TXL 41M "
                    "params (excl. word embeddings), scales 4^-k M k=0..6; Sec 6: least-squares fit "
                    "per dataset over all 42 (m,n) configurations. Words are the token unit (word-level LM).",
    ),
    Design(
        key="gordon2021",
        citation="Gordon, Duh, Kaplan 2021, Data and Parameter Scaling Laws for Neural Machine Translation (EMNLP 2021)",
        arxiv_id="",
        # Per-language-pair pooled fits; representative pair ru-en. Table 2 non-embed sizes;
        # data: random subsets 1/2^m of the corpus down to ~0.1% (m=0..10); full corpus
        # ~2.5e9 bytes (Fig 1 x-axis). D in bytes = the paper's fitted data unit.
        k_list=_cross(
            (3.93e5, 1.5e6, 6.3e6, 1.9e7, 3.8e7, 5.6e7),
            tuple(2.5e9 * 2.0 ** (-m) for m in range(11)),
        ),
        k_list_source="axes-derived",
        protocol="pooled",
        protocol_source="stated",
        beta_eff=BetaEff(
            value=0.38,
            symbol="alpha_D",
            law_family="Kaplan",
            source="ACL Anthology 2021.emnlp-main.478 (no arXiv e-print; read from the "
                   "anthology PDF), Table 1, Ru-En row - the per-language-pair fit of "
                   "their Eq. 1 over the classified design (alpha_N = 0.11, "
                   "alpha_D = 0.38).",
            range_note="the other two pairs give alpha_D = 0.35 (De-En) and 0.43 (Zh-En). "
                       "Their Table 3 also shows alpha_D moves by <= 0.03 when refit on "
                       "3-6 percent of the data.",
        ),
        provenance="Table 2 (393k-56M non-embed params); Sec 2: subsets 1/2, 1/4, ... with smallest "
                    "~0.1% => 11 fractions; full ru-en corpus ~2.5e9 bytes read from Fig 1 axis "
                    "(approximate; verdict insensitive to the byte-total within a factor ~4). "
                    "Eq 1 fit pooled per language pair (Fig 1 dotted fits).",
    ),
    Design(
        key="hu2024",
        citation="Hu et al. 2024, MiniCPM: Unveiling the Potential of Small Language Models (COLM 2024)",
        arxiv_id="2404.06395",
        k_list=(10.0, 20.0, 30.0, 40.0, 50.0, 60.0),
        k_list_source="axes-derived",
        protocol="pooled",
        protocol_source="stated",
        beta_eff=BetaEff(
            value=0.23,
            symbol="beta",
            law_family="Chinchilla",
            source="arXiv:2404.06395 tex/3.lrs.tex line 143: for their "
                   "L(N,D) = C_N N^-alpha + C_D D^-beta + L_0 fit, 'on average, we have "
                   "alpha=0.29, beta=0.23' across the evaluation corpora of the "
                   "classified design.",
            range_note="per-corpus values appear only inside the Figure 8 subplot text "
                       "boxes, not in the text; the paper states only that beta < alpha "
                       "on every corpus.",
        ),
        provenance="Scaling experiment: 6 WSD-decayed checkpoints from 10N to 60N tokens per size "
                    "(6 sizes 0.04B-2B), losses fit jointly with the Hoffmann form; interior "
                    "checkpoints read as evenly spaced 10N..60N. Concludes optimal D/N ~ 192.",
    ),
    Design(
        key="muennighoff2023",
        citation="Muennighoff et al. 2023, Scaling Data-Constrained Language Models (NeurIPS 2023)",
        arxiv_id="2305.16264",
        k_list=_load_k("muennighoff_k.txt"),
        k_list_source="exact",
        protocol="pooled",
        protocol_source="artifact",
        beta_eff=BetaEff(
            value=0.3527,
            symbol="beta",
            law_family="Chinchilla / repeated-data (effective N', D')",
            source="arXiv:2305.16264 main.tex line 1011 (Appendix, C4 scaling): 'Our fit "
                   "results in a=6.255414, b=7.3049974, e=0.6254804, alpha=beta=0.3526596'; "
                   "quoted again as beta = 0.353 at line 746.",
            range_note="alpha and beta were TIED in this fit, and were estimated on "
                       "Hoffmann's 54 C4 points, then held FIXED while only R_N* and R_D* "
                       "were fit on the 182 data-constrained runs (main.tex line 863) - so "
                       "this exponent does NOT come from the classified design, which is "
                       "why from_classified_fit is False here.",
            from_classified_fit=False,
        ),
        provenance="Data-constrained law fit pooled over 182 runs (paper Sec 3); 175 recovered by "
                    "replicating the run-name parsing of the released fit notebook "
                    "(huggingface/datablations utils/parametric_fit.ipynb); k = total tokens / params.",
    ),
    Design(
        key="aghajanyan2023",
        citation="Aghajanyan et al. 2023, Scaling Laws for Generative Mixed-Modal Language Models (ICML 2023)",
        arxiv_id="2301.03728",
        # Per-modality pooled fits share the same axes: D in {5,10,100}B tokens (stated),
        # N endpoints 8M and 6.7B (stated range, 7 sizes). Endpoint-mode on N.
        k_list=_cross((8.0e6, 6.7e9), (5.0e9, 1.0e10, 1.0e11)),
        k_list_source="endpoints",
        protocol="pooled",
        protocol_source="stated",
        beta_eff=BetaEff(
            value=0.22,
            symbol="beta",
            law_family="Chinchilla (per modality)",
            source="arXiv:2301.03728 iclr2023_conference.tex line 215, Table 3 (uni-modal "
                   "scaling-law parameters fit to their Eq. 2), Text column - the "
                   "language-modeling modality, matching the token-unit k-list. All 7 "
                   "modality fits share the identical (N, D) design.",
            range_note="across the 7 modalities beta spans 0.11 (Image-Text) to 0.32 "
                       "(Code); the causal-LM control fit gives Text beta = 0.23 and "
                       "Speech beta = 0.24 (Table 4).",
        ),
        provenance="Sec 4-5: 7 sizes 8M-6.7B, each trained on 5B/10B/100B tokens per modality; "
                    "Eq 2 params fit by pooled Huber objective per modality. N interior sizes "
                    "omitted under the conv. 5 endpoints heuristic (can over-credit, not "
                    "under-credit).",
    ),
    Design(
        key="frantar2023",
        citation="Frantar et al. 2023, Scaling Laws for Sparsely-Connected Foundation Models (ICLR 2024)",
        arxiv_id="2309.08520",
        # T5/C4 language grid (the token-axis fit): 4 dense-equivalent sizes x 3 lengths.
        # D from 250K/500K/1M steps x 65536 tokens/step ("up to ~65B tokens").
        k_list=_cross((1.3e6, 5.3e6, 2.12e7, 8.5e7), (1.64e10, 3.28e10, 6.55e10)),
        k_list_source="axes-derived",
        protocol="pooled",
        protocol_source="stated",
        beta_eff=BetaEff(
            value=0.203,
            symbol="b_D",
            law_family="sparsity superset (Chinchilla-style additive (a_D/D)^b_D term)",
            source="arXiv:2309.08520 arxiv.tex line 501, Table 5 (appendix 'Scaling "
                   "Coefficients'), T5/C4 unstructured row - the language grid that is "
                   "the classified design (a_D = 6.90e8, b_D = 0.203).",
            range_note="the ViT/JFT vision grid in the same table (not the classified "
                       "design) gives b_D = 0.890; the n:m row refits only the sparsity "
                       "coefficients and inherits b_D.",
        ),
        provenance="Sec 3.1 Table 1: T5/C4 sizes 1.3M..85M non-zero params (4x increments), "
                    "training lengths 250K/500K/1M steps, longest ~65B tokens => ~65536 tokens/step; "
                    "48 runs, single joint law pooled over sizes/lengths/sparsities.",
    ),
    Design(
        key="tao2024",
        citation="Tao et al. 2024, Scaling Laws with Vocabulary: Larger Models Deserve Larger Vocabularies (NeurIPS 2024)",
        arxiv_id="2407.13623",
        # Table 4: chars per non-vocab-param per size group (single budget per group);
        # k in the paper's own data unit (training characters / N_nv).
        k_list=(4.3e9 / 3.3e7, 1.11e10 / 8.5e7, 1.96e10 / 1.51e8,
                4.30e10 / 3.02e8, 1.016e11 / 6.31e8, 2.013e11 / 1.13e9),
        k_list_source="exact",
        protocol="pooled",
        protocol_source="stated",
        beta_eff=BetaEff(
            value=0.447,
            symbol="beta",
            law_family="vocabulary superset (Chinchilla core plus a vocab-parameter term)",
            source="arXiv:2407.13623 sec/3-3method.tex line 88 (Approach 3, their Eq. 5 "
                   "L_u = -E + A_1/N_nv^alpha_1 + A_2/N_v^alpha_2 + B/D^beta): 'Fitting "
                   "yields A_1=1.831, A_2=0.196, B=2.124, E=5.533, alpha_1=beta=0.447, "
                   "alpha_2=0.671'. D is training characters, the same unit as the k-list.",
            range_note="alpha_1 and beta were tied following Muennighoff et al.; no "
                       "per-size-group values are reported.",
        ),
        provenance="Table 4 (6 groups, 33M-1.13B non-vocab params; 4.3B-201.3B training chars); "
                    "Approach 3 fits all runs pooled (not only per-FLOP minima). Data axis is "
                    "characters (their vocab-independent unit), so k is chars per non-vocab param.",
    ),
    Design(
        key="bhagia2025",
        citation="Bhagia et al. 2025, Establishing Task Scaling Laws via Compute-Efficient Model Ladders (COLM 2025)",
        arxiv_id="2412.04403",
        k_list=(20.0, 40.0, 100.0, 200.0),
        k_list_source="exact",
        protocol="pooled",
        protocol_source="stated",
        beta_eff=BetaEff(
            value=0.24,
            symbol="beta",
            law_family="Chinchilla (task loss)",
            source="arXiv:2412.04403 floats/fitted_params.tex, Table 8 ('Step 1 Fitted "
                   "Function'), MMLU row: L(N,D) = 38.07/N^0.23 + 100.09/D^0.24 + 0.45. "
                   "MMLU is the paper's lead task (intro: 'the tasks that we care the most "
                   "about, e.g. MMLU and HellaSwag').",
            range_note="the identical 16-run design is fit once per task; beta spans 0.11 "
                       "(CommonsenseQA) to 0.48 (Social IQa) across the 8 tasks, with "
                       "HellaSwag at 0.26.",
        ),
        provenance="Ladder: sizes 190M/370M/760M/1.3B non-embed x D in {1,2,5,10}xC with 1xC = 20N "
                    "(Table 1); step-1 law L(N,D)=A/N^a+B/D^b+E fit on all 16 final checkpoints "
                    "(per task; identical 16-run design each fit).",
    ),
    Design(
        key="gadre2024",
        citation="Gadre et al. 2024, Language Models Scale Reliably with Over-Training and on Downstream Tasks",
        arxiv_id="2403.08540",
        k_list=(20.0, 20.0, 20.0, 20.0, 320.0),
        k_list_source="exact",
        protocol="pooled",
        protocol_source="stated",
        beta_eff=BetaEff(
            value=0.242,
            symbol="beta (= 2 eta)",
            law_family="Chinchilla with alpha tied to beta",
            source="arXiv:2403.08540 tables/fits.tex, Table 3 (scaling-law fit "
                   "parameters), C4 row: eta = 0.121. The paper imposes alpha = beta and "
                   "reparameterises with eta = alpha/2 (text/03_methods.tex lines 81-89), "
                   "so the data exponent is beta = 2 eta = 0.242.",
            range_note="the two other per-corpus fits over the same runs give beta = "
                       "0.272 (RedPajama) and 0.254 (RefinedWeb); this row is knife-edge "
                       "at the old convention, so the choice within that range matters.",
        ),
        provenance="Table 2 default fit: N in {0.011,0.079,0.154,0.411}B at M=20 plus N=0.011B at "
                    "M=320, where M = D/N by construction; three separate per-training-dataset fits "
                    "(C4, RedPajama, RefinedWeb), each pooled over these runs. Distinct ratios {20,320}. "
                    "KNIFE-EDGE PASS: V_K 0.0136 vs tau_K 0.0124 at the primary convention (~10% "
                    "margin); verdict flips to fail at beta_eff 0.10, 0.20 and 0.50 (passes only at "
                    "0.28 and 0.35).",
    ),
    Design(
        key="li2025farseer",
        citation="Li et al. 2025, Predictable Scale Part II - Farseer: A Refined Scaling Law in Large Language Models (NeurIPS 2025)",
        arxiv_id="2506.10972",
        k_list=_load_k("farseer_k.txt"),
        k_list_source="exact",
        protocol="pooled",
        protocol_source="artifact",
        beta_eff=BetaEff(
            value=0.327,
            symbol="A(N) = f_A(N; theta_A)",
            law_family="Farseer (additive, N-dependent data exponent)",
            source="arXiv:2506.10972 neurips_2025_new.tex line 705, Eq. 21 (the specific "
                   "fitted Farseer law): the data exponent is "
                   "A(N) = exp(-0.124 N^0.123 + 0.424). Farseer reports no single number, "
                   "so the row is evaluated at the geometric-mean N of the fitted range "
                   "(99.6M-6.37B), N_gm = 7.97e8, giving A = 0.327.",
            range_note="A(N) runs from 0.462 at N = 1e8 to 0.209 at N = 6.37e9, so no "
                       "single beta_eff is exact; the pre-registered sensitivity grid "
                       "brackets that whole range.",
        ),
        provenance="Released fit dataset (Farseer-Scaling-Law/Farseer ipynb/data/1222_full.csv): "
                    "404 runs, N 99.6M-6.37B, D 1e9-5.12e11; k = D/N spans 0.314-2570 on a ~sqrt(2) "
                    "geometric grid (27 distinct ratios); global pooled surface fit (paper Sec 3). "
                    "Additive-family note (conv. 8): their Eq. 15 is additive, L(N,D) = f(N) + "
                    "g(N) D^-b(N), but its data exponent b(N) varies with N (~0.46 at 1e8 params "
                    "to ~0.21 at 6.4e9), so no single beta_eff is exact; the pre-registered "
                    "beta_eff grid {0.10,...,0.50} brackets that range and the verdict does not "
                    "flip anywhere on it.",
    ),
    Design(
        key="krajewski2024",
        citation="Krajewski et al. 2024, Scaling Laws for Fine-Grained Mixture of Experts (ICML 2024)",
        arxiv_id="2402.07871",
        # Joint MoE law (Eq 9) pooled over >100 runs; D values stated {16,24,33,66,130}B,
        # N (total non-embedding, their convention) endpoints 129M / 3.7B (abstract).
        k_list=_cross((1.29e8, 3.7e9), (1.6e10, 2.4e10, 3.3e10, 6.6e10, 1.3e11)),
        k_list_source="endpoints",
        protocol="pooled",
        protocol_source="stated",
        beta_eff=BetaEff(
            value=0.147,
            symbol="beta",
            law_family="fine-grained MoE superset (Chinchilla core plus a granularity term)",
            source="arXiv:2402.07871 scaling_laws.tex line 387, Table 1 ('Values of the "
                   "fitted coefficients'), MoE row of their Eq. 9 "
                   "L(N,D,G) = c + (g/G^gamma + a)/N^alpha + b/D^beta.",
            range_note="the companion dense-transformer fit in the same table gives "
                       "beta = 0.127.",
        ),
        provenance="Sec 4 / Appendix A Tables 3-4: token counts 16B/24B/33B/66B/130B; models "
                    "129M-3.7B params; Eq 9 fit pooled over the MoE family. N endpoints x stated D axis.",
    ),
    Design(
        key="xu2026softq",
        citation="Xu et al. 2026, Data-Constrained Language Model Pretraining: Improved Regularization and Scaling Laws",
        arxiv_id="2606.06888",
        k_list=_cross((7.2e7, 1.4e8, 2.57e8, 6.64e8, 1.4e9),
                      (1.0e8, 2.0e8, 3.0e8, 4.0e8)),
        k_list_source="axes-derived",
        protocol="pooled",
        protocol_source="stated",
        beta_eff=BetaEff(
            value=0.1247,
            symbol="alpha/(1+alpha)",
            law_family="SoftQ (nested-power law, Kaplan-style)",
            source="arXiv:2606.06888 arxiv_version.tex line 1484, Table 10 ('Fitted "
                   "constants on all 20 DCLM baseline points'), SoftQ row: alpha = 0.1425, "
                   "rho = 0.7961. SoftQ's data-limited exponent is alpha/(1+alpha), stated "
                   "by the paper right after Eq. 6 (line 764), = 0.1247; it is also the "
                   "product of the two exponents in the boxed fitted law Eq. 24 "
                   "(0.69676 x 0.17906).",
            range_note="the held-out fit gives alpha = 0.1287 (beta_eff 0.114) and the "
                       "external Kim et al. grid alpha = 0.5905 (beta_eff 0.371); the "
                       "paper's own Chinchilla fit on these same 20 points has "
                       "beta = 0.2924.",
        ),
        provenance="5x4 grid: sizes {72M,140M,257M,664M,1.4B} x unique-token budgets "
                    "{100M,200M,300M,400M}; SoftQ fit pooled over the 20 (N,U,L) points with the "
                    "Hoffmann Approach-3 Huber objective. Data axis is unique tokens U (their law's axis).",
    ),
    Design(
        key="sedova2026",
        citation="Sedova et al. 2026, Scaling Laws for Mixture Pretraining Under Data Constraints",
        arxiv_id="2605.12715",
        k_list=(100.0,),
        k_list_source="axes-derived",
        protocol="pooled",
        protocol_source="stated",
        beta_eff=BetaEff.not_reported(
            symbol="alpha",
            law_family="mixture / repetition-aware superset (data term B N^delta / D_eff^alpha)",
            source="NOT REPORTED. arXiv:2605.12715 defines the data-side exponent as alpha "
                   "on D_eff (chapters/05_scaling_law.tex line 34, Eq. 3: L_size = E + "
                   "C/N^beta + B N^delta / D_eff^alpha + gamma h - note their beta is the "
                   "MODEL exponent) and lists the nine fitted parameters by name at line "
                   "39, but the paper and its appendix report only weighted R^2 and "
                   "prediction-error tables; no numeric fitted constant appears anywhere. "
                   "Fell back to the corpus-wide 0.28, flagged. Harmless here: the design "
                   "is K = 1, so V_1 = 0 < tau_1 at every beta_eff.",
        ),
        provenance="Primary multilingual L_size fit: sizes 101M-805M non-embed, each trained to "
                    "approximately 100xN total tokens (stated); joint fit across sizes => single "
                    "distinct TPP ray k~100. Multi-fit rule (conv. 7): the multilingual L_size fit "
                    "is the paper's headline bivariate fit and is the one classified; the "
                    "OpenWebMath variant (spanning ~24x-130x TPP) is a secondary setting, and its "
                    "own observation set is not stated run-by-run, so it would fall under rule (b) "
                    "if classified on its own.",
    ),
    Design(
        key="sardana2023",
        citation="Sardana et al. 2025, Beyond Chinchilla-Optimal: Accounting for Inference in "
                 "Language Model Scaling Laws (ICML 2024 version; cited as 2025 in this paper)",
        arxiv_id="2401.00448",
        # v3 Appendix C Table 4: TPP ratios per size - 150M gets all 12, larger sizes prefixes;
        # distinct-union of the ratios entering the All-Data pooled refit (Sec 5, Table 1).
        k_list=(10.0, 15.0, 20.0, 30.0, 50.0, 75.0, 100.0, 250.0, 500.0,
                1000.0, 5000.0, 10000.0),
        k_list_source="exact",
        protocol="pooled",
        protocol_source="stated",
        beta_eff=BetaEff(
            value=0.24,
            symbol="beta",
            law_family="Chinchilla",
            source="arXiv:2401.00448 main.tex line 318, Table 2 ('Parametric fitting on "
                   "data including extreme training runs'), 'All Data' row: alpha = 0.18, "
                   "beta = 0.24 - the headline pooled refit that is the classified fit.",
            range_note="the three nested-subset ablations of the same design give "
                       "beta = 0.13 (<=100 TPP), 0.16 (<=250) and 0.16 (<=500); the paper "
                       "itself reads this drift as evidence the Chinchilla form does not "
                       "extend to extreme ratios.",
        ),
        provenance="v3 (ICML 2024 version): 47 models, sizes 150M/370M/750M/1.3B/2.5B/6B, TPP 10 "
                    "to 10,000 (Table 4); Sec 5 refits the Chinchilla parametric law with the "
                    "Hoffmann Sec-D.2 objective on nested subsets (<=100, <=250, <=500 TPP) and All "
                    "Data. Multi-fit rule (conv. 7): the All-Data pooled fit is the paper's "
                    "headline bivariate fit (Table 1's reported constants) and is the one "
                    "classified; the three nested-subset fits are stated ablations of it, and each "
                    "is a strict prefix of the same ratio list. Re-included after the first audit: "
                    "the earlier excl-a call was based on v1, which only reused Hoffmann constants.",
    ),
    Design(
        key="magnusson2025",
        citation="Magnusson et al. 2025, DataDecide: How to Predict Best Pretraining Data with "
                 "Small Experiments (ICML 2025)",
        arxiv_id="2504.11393",
        k_list=(100.0,),
        k_list_source="exact",
        protocol="pooled",
        protocol_source="stated",
        beta_eff=BetaEff.not_reported(
            symbol="beta",
            law_family="Chinchilla",
            source="NOT REPORTED. arXiv:2504.11393 states the Appendix-C variant "
                   "L(N,D) = A/N^alpha + B/D^beta + E (example_paper.tex line 532) and "
                   "names its five parameters, but reports only per-variant prediction "
                   "errors (Table 5); no fitted coefficient is printed for any recipe or "
                   "task. Fell back to the corpus-wide 0.28, flagged. Harmless here: the "
                   "design is K = 1, so V_1 = 0 < tau_1 at every beta_eff.",
        ),
        provenance="Appendix C variant '5-parameter (N,D) fit' (their Eq. 5): "
                    "L(N,D)=A/N^a+B/D^b+E fit per data recipe on the 14-size ladder 4M-1B, every "
                    "size trained to D=100N (stated 'token to parameter ratio of 100') => K=1 at "
                    "k=100. This is NOT DataDecide's baseline: their baseline is the 3-parameter "
                    "L(C), and Sec 2.2 states why in their own words - 'as our suite only includes "
                    "one token-parameter ratio, we use the simplified 3 parameter baseline, L(C)'. "
                    "That sentence is the authors independently reporting K=1 and declining to fit "
                    "a bivariate law on it. Multi-fit rule (conv. 7): the Eq.-5 bivariate variant "
                    "is the paper's only N-and-D fit, so it is the one classified; the headline "
                    "L(C) fit is univariate in compute and not classifiable. The Eq.-5 variant's "
                    "observation set is not enumerated run by run, which would normally raise rule "
                    "(b)/(c); K is invariant to it here, because every run in the suite sits at the "
                    "same D = 100N ratio, so any subset of them gives K = 1. On checkpoints "
                    "specifically (the asymmetry vs choshen2024, which IS classified by pooling "
                    "intermediate checkpoints): Sec 2.3 states 'Following Bhagia et al. (2024) we "
                    "fit Equation 1 only on observations of final, fully trained checkpoints', and "
                    "Eq. 5 is the (N,D) replacement for Eq. 1, while intermediate checkpoints feed "
                    "only the Eq. 2 accuracy sigmoid. Robustness: even reading a last-10% "
                    "checkpoint window as k in {90,100}, the design still fails at kappa_target "
                    "1e2 and 1e4. Re-included after the "
                    "first audit: the earlier excl-a-nonparametric call was an abstract-level "
                    "error.",
    ),
    Design(
        key="choshen2024",
        citation="Choshen, Zhang, Andreas 2024, A Hitchhiker's Guide to Scaling Law Estimation "
                 "(ICML 2025)",
        arxiv_id="2410.11840",
        k_list=_load_k("choshen_pythia_k.txt"),
        k_list_source="exact",
        protocol="pooled",
        protocol_source="stated",
        beta_eff=BetaEff.not_reported(
            symbol="beta",
            law_family="Chinchilla (exponentiated coefficients)",
            source="NOT REPORTED for the Pythia slice. arXiv:2410.11840 fits >1000 laws of "
                   "the form e^E + e^A/N^alpha + e^B/D^beta (main.tex line 433) and shows "
                   "the per-family (B, beta) spread only as a scatter plot (Fig. 6, "
                   "fig:popts, main.tex line 525); the text gives no numeric beta for "
                   "Pythia, and the released ColPret artifact carries raw per-checkpoint "
                   "eval data, not fitted coefficients (checked the repo tree). Fell back "
                   "to the corpus-wide 0.28, flagged. The paper does note that one Pythia "
                   "70m intervention baseline is the outlier with beta > 4.",
        ),
        provenance="Meta-study fitting >1000 Chinchilla-form laws, one per scaled model family "
                    "(per-slice at family level, conv. 6); representative slice classified: the "
                    "Pythia family (8 sizes 70M-12B, 300B tokens, most-checkpointed family in the "
                    "released artifact). Fits pool the family's models AND intermediate checkpoints "
                    "('fit scaling laws using all these intermediate checkpoints'), ignoring the "
                    "first 10B tokens (their stated filter). EXACT per-checkpoint ratios: the "
                    "released ColPret artifact (IBM/ColPret aggregated_eval/pythia.csv, num_params "
                    "and steps columns) records every released Pythia checkpoint, so all 1112 "
                    "fitted (N, D) observations are enumerable; upgraded from endpoints-mode on "
                    "2026-07-31 (CRITERIA Amendments 2, item 2). Re-included after the first audit: "
                    "the ratio lists ARE recoverable from the artifact; other families derive the "
                    "same way.",
    ),
    Design(
        key="kumar2024",
        citation="Kumar et al. 2024, Scaling Laws for Precision (ICLR 2025)",
        arxiv_id="2411.04330",
        k_list=_cross((3.0e7, 6.0e7, 1.1e8, 2.2e8),
                      (1.5e9, 3.0e9, 6.0e9, 1.3e10, 2.6e10)),
        k_list_source="axes-derived",
        protocol="pooled",
        protocol_source="stated",
        beta_eff=BetaEff(
            value=0.4965,
            symbol="beta (tied to alpha)",
            law_family="precision superset (Chinchilla core with N -> N_eff)",
            source="arXiv:2411.04330 main.tex lines 688-712, Table 3 of Appendix "
                   "'Numerical Fits': alpha = 0.4965, with alpha tied to beta ('Following "
                   "Muennighoff et al. we tie alpha = beta', line 645).",
            range_note="an earlier fit left commented out in the same source gives "
                       "alpha = 0.1288; the paper explicitly warns that 'our numerical "
                       "constants are unlikely to be useful' because fitted constants "
                       "depend heavily on setup (line 113).",
        ),
        provenance="Sweep 'N in [30,60,110,220] million parameters (non-embedding) and D in "
                    "[1.5,3,6,13,26] billion tokens' (stated), 465 runs across precisions on the "
                    "crossed grid; single pooled precision-aware fit "
                    "L(N,D,P)=A N_eff^-a + B D^-b + E + delta_PTQ.",
    ),
    Design(
        key="shukor2025mixtures",
        citation="Shukor et al. 2025, Scaling Laws for Optimal Data Mixtures (arXiv 2507.09404)",
        arxiv_id="2507.09404",
        # LLM setting: fit runs at N in {412M, 834M, 1.1B, 1.4B}, D 4B-46B tokens (stated),
        # pooled over (N_j, D_j, h_j) runs. Endpoint corners; verdict robust: even the single
        # widest size column (412M, 4-46B) alone passes.
        k_list=_cross((4.12e8, 1.4e9), (4.0e9, 4.6e10)),
        k_list_source="endpoints",
        protocol="pooled",
        protocol_source="stated",
        beta_eff=BetaEff.not_reported(
            symbol="beta",
            law_family="data-mixture superset (Chinchilla core plus domain-weight terms)",
            source="NOT REPORTED. arXiv:2507.09404 defines the data term B/D^beta in both "
                   "the additive (Eq. 3, apple.tex line 286) and joint (Eq. 4, line 297) "
                   "laws and lists beta among the fitted parameters, but reports only MRE "
                   "tables, predicted-vs-observed plots and optimal domain weights; no "
                   "fitted exponent is printed. Fell back to the corpus-wide 0.28, flagged.",
        ),
        provenance="Additive law L=E+sum_i C_i h_i^-g + A/N^a + B/D^b fit by pooled Huber "
                    "objective over p runs (N_j,D_j,h_j) (stated); LLM grid 'Train: 412M, 834M, "
                    "1.1B, 1.4B parameters with 4-46B tokens'. Endpoint corners of the stated "
                    "axes.",
    ),
    Design(
        key="volkova2026optimizers",
        citation="Volkova et al. 2026, Towards Robust Scaling Laws for Optimizers (arXiv 2602.07712)",
        arxiv_id="2602.07712",
        k_list=(30.0, 50.0, 100.0, 200.0),
        k_list_source="exact",
        protocol="pooled",
        protocol_source="stated",
        beta_eff=BetaEff(
            value=0.38,
            symbol="beta",
            law_family="Chinchilla",
            source="arXiv:2602.07712 sections/experiments.tex line 45, Table 1 ('Scaling "
                   "law coefficients across optimizers'), AdamW row: beta = 0.38 +/- 0.04. "
                   "AdamW is the paper's reference optimizer, the one whose shared "
                   "(A, alpha, B, beta, E) the rest of the paper is built on "
                   "(experiments.tex line 192).",
            range_note="the other four optimizers, fit on the identical design, give "
                       "beta = 0.27 (Muon), 0.24 (Scion), 0.27 (Shampoo), 0.26 (SOAP). The "
                       "paper's own leave-one-out error bars are enormous (A = 4966 +/- "
                       "4612 for AdamW) and it states these per-optimizer fits are "
                       "ill-conditioned with highly correlated parameters - the exponent "
                       "here is exactly as identifiable as the criterion says it is.",
        ),
        provenance="Stated: 'For each model size N we train for a set of token-to-parameter "
                    "ratios D/N={30,50,100,200}'; sizes 50M-1.5B; Chinchilla-form Eq 1 fit "
                    "'across all model sizes and data budgets, independently for each optimizer'. "
                    "The paper itself reports these per-optimizer fits are 'ill-conditioned and "
                    "have highly correlated parameters' - an independent empirical report of the "
                    "predicted failure mode.",
    ),
    Design(
        key="panferov2025compressed",
        citation="Panferov et al. 2025, Unified Scaling Laws for Compressed Representations "
                 "(arXiv 2506.01863)",
        arxiv_id="2506.01863",
        k_list=(100.0,),
        k_list_source="exact",
        protocol="pooled",
        protocol_source="stated",
        beta_eff=BetaEff.not_reported(
            symbol="beta",
            law_family="compressed-representation superset (Chinchilla core with N -> N rho(R))",
            source="NOT REPORTED. arXiv:2506.01863 states "
                   "Loss(N,D) ~ A (N rho(R))^-alpha + B D^-beta + E (arxiv.tex line 175) "
                   "and calls A, B, alpha, beta fitted constants, but every table in the "
                   "paper reports representation-capacity values rho(R) and fit errors "
                   "only; no alpha or beta value is printed. Fell back to the corpus-wide "
                   "0.28, flagged. Harmless here: the design is K = 1, so V_1 = 0 < tau_1 "
                   "at every beta_eff.",
        ),
        provenance="Chinchilla-form Loss(N,D) ~ A(N rho(R))^-a + B D^-b + E fit over models "
                    "30M-200M non-embed, all trained at a stated single ratio ('100 training "
                    "tokens per model parameter') across compression formats => K=1 single-ray "
                    "design.",
    ),
    Design(
        key="sun2025fp",
        citation="Sun et al. 2025, Scaling Laws for Floating Point Quantization Training "
                 "(arXiv 2501.02423)",
        arxiv_id="2501.02423",
        k_list=_cross((4.1e7, 8.5e7, 1.54e8, 6.79e8),
                      (1.0e10, 2.0e10, 5.0e10, 1.0e11)),
        k_list_source="axes-derived",
        protocol="pooled",
        protocol_source="stated",
        beta_eff=BetaEff(
            value=0.5162,
            symbol="beta",
            law_family="FP-quantization superset (Chinchilla core n/N^alpha + d/D^beta + eps)",
            source="arXiv:2501.02423 src/appendix_fitting_details_of_scaling_laws.tex "
                   "line 33, Table 6 ('Fitted constants and their values in Eq. 15'), the "
                   "single pooled fit over the 358 experiments of the classified design "
                   "(n = 69.2343, alpha = 0.2368, d = 68973.06, beta = 0.5162).",
            range_note="single reported fit. Note 0.5162 sits just above the top of the "
                       "pre-registered beta_eff sensitivity grid (0.50).",
        ),
        provenance="FP-quantization law L(N,D,E,M,B) with Chinchilla core n/N^a + d/D^b + eps "
                    "plus precision terms; stated grid 'N in {41,85,154,679} million parameters "
                    "and D in {10,20,50,100} billion tokens'; pooled fit ('adopted all above 358 "
                    "experiments ... to obtain the specific hyper-parameters').",
    ),
)
