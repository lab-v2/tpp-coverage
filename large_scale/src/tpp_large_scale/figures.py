"""Report figures. Each builder is pure: results in, PNG path out."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Set before pyplot import: this runs headless.

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .config import CLEAN_RAY_BAND, FLOPS_PER_PARAM_TOKEN, AnalysisConfig  # noqa: E402
from .model.diversity import DiversityCriterion  # noqa: E402
from .model.fitting import CHINCHILLA  # noqa: E402
from .model.grid import TrainingGrid  # noqa: E402
from .model.splits import frontier_split  # noqa: E402

FIGURE_DPI = 130
FIGURE_FONT_SIZE = 9
ISOFLOP_CURVE_COUNT = 6


class FigureBuilder:
    """Writes every figure the report includes."""

    def __init__(self, config: AnalysisConfig, grid: TrainingGrid) -> None:
        self._config = config
        self._grid = grid
        plt.rcParams.update({"figure.dpi": FIGURE_DPI, "font.size": FIGURE_FONT_SIZE})

    def _save(self, figure: plt.Figure, name: str) -> Path:
        path = self._config.figures_dir / name
        figure.tight_layout()
        figure.savefig(path)
        plt.close(figure)
        return path

    def build_all(self, results: dict) -> list[Path]:
        """Build every figure and return the paths written."""
        return [
            self.loss_vs_n(),
            self.loss_vs_d(),
            self.kappa_by_design(results),
            self.holdout_scatter(results),
            self.exponent_gap_sweep(results),
            self.two_ray_threshold(results),
            self.isoflop_curves(results),
            SummaryFigure(self._config).build(results),
            # design_grid.png is produced by figures_tikz/build_design_grid.sh, a TikZ
            # port of the paper's appendix TPP-grid schematic. The matplotlib DesignScatter
            # class below is kept but no longer invoked, so the pipeline does not overwrite
            # the TikZ figure.
        ]

    def loss_vs_n(self) -> Path:
        arrays = self._grid.arrays
        tpp = self._grid.envelope["TPPd"].to_numpy(dtype=float)
        figure, axis = plt.subplots(figsize=(5, 3.6))
        scatter = axis.scatter(arrays.n, arrays.loss, c=np.log10(tpp), cmap="viridis", s=28)
        axis.set_xscale("log")
        axis.set_xlabel("N (non-embedding params)")
        axis.set_ylabel("val loss")
        axis.set_title("Envelope loss vs N (color = log10 design TPP)")
        figure.colorbar(scatter, ax=axis, label="log10 TPP")
        return self._save(figure, "loss_vs_N.png")

    def loss_vs_d(self) -> Path:
        arrays = self._grid.arrays
        figure, axis = plt.subplots(figsize=(5, 3.6))
        scatter = axis.scatter(arrays.d, arrays.loss, c=np.log10(arrays.n), cmap="plasma", s=28)
        axis.set_xscale("log")
        axis.set_xlabel("D (tokens)")
        axis.set_ylabel("val loss")
        axis.set_title("Envelope loss vs D (color = log10 N)")
        figure.colorbar(scatter, ax=axis, label="log10 N")
        return self._save(figure, "loss_vs_D.png")

    def kappa_by_design(self, results: dict) -> Path:
        rays = results["co_rays"]
        labels = ["full grid\n(NC)"] + [f"ray\nTPP~{b}" for b in rays]
        values = [results["full_grid_fits"]["Chinchilla"]["kappa_AB"]] + [
            rays[b].get("kappa_AB", np.nan) for b in rays
        ]
        figure, axis = plt.subplots(figsize=(5, 3.6))
        axis.bar(range(len(values)), values, color=["#2c7fb8"] + ["#d95f0e"] * (len(values) - 1))
        axis.set_yscale("log")
        axis.set_xticks(range(len(values)))
        axis.set_xticklabels(labels, fontsize=7)
        axis.set_ylabel(r"$\kappa_{A,B}(J^\top J)$")
        axis.set_title("Conditioning: 2D grid vs single-TPP rays")
        return self._save(figure, "kappa_AB.png")

    def holdout_scatter(self, results: dict) -> Path:
        holdout = results["co_vs_nc_holdout"]
        co, nc = holdout["rmse_co_samples"], holdout["rmse_nc_samples"]
        figure, axis = plt.subplots(figsize=(4, 4))
        axis.scatter(co, nc, s=14, alpha=0.5)
        limit = [0, max(max(co), max(nc)) * 1.05]
        axis.plot(limit, limit, "k--", lw=1)
        axis.set_xlim(limit)
        axis.set_ylim(limit)
        axis.set_xlabel("CO holdout RMSE")
        axis.set_ylabel("NC holdout RMSE")
        axis.set_title(
            f"CO vs NC (NC below line = NC wins; wr={holdout['nc_win_rate']:.2f})"
        )
        return self._save(figure, "co_vs_nc_rmse.png")

    def exponent_gap_sweep(self, results: dict) -> Path:
        sweep = results["formal_validation"]["prop1_cor1"]
        gaps = np.asarray(sweep["eps_grid"])
        kappas = np.asarray(sweep["kappa_AB"])
        ratios = np.asarray(sweep["ci_ratio"])
        figure, axes = plt.subplots(1, 2, figsize=(9, 3.7))
        axes[0].loglog(gaps, kappas, lw=2, label="measured")
        axes[0].loglog(
            gaps, kappas[0] * (gaps / gaps[0]) ** -2.0, ls="--", c="grey",
            label=r"$\varepsilon^{-2}$",
        )
        axes[1].loglog(gaps, ratios, lw=2, c="C1", label="measured")
        axes[1].loglog(
            gaps, ratios[0] * (gaps / gaps[0]) ** -1.0, ls="--", c="grey",
            label=r"$\varepsilon^{-1}$",
        )
        for entry in sweep["per_law"]:
            axes[0].scatter([entry["eps"]], [entry["kappa_AB_model"]], zorder=5)
            axes[1].scatter([entry["eps"]], [entry["ci_ratio_model"]], zorder=5)
            axes[0].annotate(entry["law"][:4], (entry["eps"], entry["kappa_AB_model"]), fontsize=7)
        axes[0].set_ylabel(r"$\kappa_{A,B}$")
        axes[1].set_ylabel(r"$\mathrm{CI}(A)/\mathrm{CI}(\psi)$")
        axes[0].set_title(r"Prop. 1: $\kappa_{A,B}\sim\varepsilon^{-2}$")
        axes[1].set_title(r"Cor. 1: CI inflation $\sim\varepsilon^{-1}$")
        for axis in axes:
            axis.set_xlabel(r"exponent gap $\varepsilon$")
            axis.legend(fontsize=8)
        figure.suptitle(
            "Conditioning and CI inflation vs exponent gap on the clean-ray grid "
            "(dots = fitted law gaps)"
        )
        return self._save(figure, "eps_scaling.png")

    def two_ray_threshold(self, results: dict) -> Path:
        prop2 = results["formal_validation"]["prop2_rmin"]
        criterion = DiversityCriterion(prop2["beta_eff"])
        spreads, v_values, tau_values = criterion.two_ray_curves(
            prop2["kappa_target"], points=self._config.rmin_sweep_points
        )
        r_min = prop2["R_min"]
        figure, axis = plt.subplots(figsize=(5.2, 3.8))
        axis.plot(spreads, v_values, lw=2, label=r"$V_2$ (achieved TPP diversity)")
        axis.plot(
            spreads, tau_values, lw=2, ls="--",
            label=r"$\tau_2$ ($\kappa_{\mathrm{target}}{=}10^4$)",
        )
        if np.isfinite(r_min):
            axis.axvline(
                r_min, color="k", ls=":", lw=1.2, label=rf"$R_{{\min}}\approx{r_min:.1f}$"
            )
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel(r"two-ray spread $R=k_2/k_1$")
        axis.set_ylabel("value")
        axis.set_title("Prop. 2: two-ray diversity threshold")
        axis.legend(fontsize=7, loc="lower right")
        return self._save(figure, "rmin_sweep.png")

    def isoflop_curves(self, results: dict) -> Path:
        isoflop = results["formal_validation"]["isoflop"]
        co_params = np.asarray(list(isoflop["co_params"].values()), dtype=float)
        nc_params = np.asarray(
            list(results["full_grid_fits"]["Chinchilla"]["params"].values()), dtype=float
        )
        # The holdout is everything off the clean ray, matching the section.
        ray_index = self._grid.band(CLEAN_RAY_BAND).index
        holdout = self._grid.envelope[~self._grid.envelope.index.isin(ray_index)]
        arrays = TrainingGrid.arrays_of(holdout)
        compute = np.asarray(isoflop["compute_budgets"], dtype=float)

        grid_n = np.logspace(
            np.log10(self._grid.arrays.n.min()), np.log10(self._grid.arrays.n.max()), 120
        )
        selected = np.argsort(compute)[
            np.linspace(0, len(compute) - 1, min(ISOFLOP_CURVE_COUNT, len(compute))).astype(int)
        ]
        figure, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
        for index in selected:
            budget = compute[index]
            tokens = budget / (FLOPS_PER_PARAM_TOKEN * grid_n)
            axes[0].plot(grid_n, CHINCHILLA.loss_fn(grid_n, tokens, *co_params), lw=1, alpha=0.85)
            axes[1].plot(grid_n, CHINCHILLA.loss_fn(grid_n, tokens, *nc_params), lw=1, alpha=0.85)
            for axis in axes:
                axis.scatter([arrays.n[index]], [arrays.loss[index]], c="k", s=20, zorder=5)
        for axis, title in zip(axes, ["CO fit: single-TPP ray", "NC fit: 2D grid"]):
            axis.set_xscale("log")
            axis.set_xlabel("N (non-embedding)")
            axis.set_title(title)
        axes[0].set_ylabel("predicted loss")
        figure.suptitle("Induced isoFLOP curves vs off-ray holdout points (black)")
        return self._save(figure, "isoflop.png")


#: Validated categorical slots 1-3 from the reference palette (all-pairs safe).
SERIES_DIVERSE = "#2a78d6"
SERIES_COLLINEAR = "#eb6834"
SERIES_REFERENCE = "#1baf7a"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
SURFACE = "#fcfcfb"
GRID_INK = "#d8d7d2"


class SummaryFigure:
    """One at-a-glance panel of the three headline results.

    Built for a README rather than the report: three panels answering what was
    measured, whether it matters for prediction, and why it happens. Colour
    encodes design type (diverse vs collinear) and nothing else.
    """

    def __init__(self, config: AnalysisConfig) -> None:
        self._config = config

    def build(self, results: dict) -> Path:
        figure, axes = plt.subplots(1, 3, figsize=(15, 4.6))
        figure.patch.set_facecolor(SURFACE)
        self._conditioning_panel(axes[0], results)
        self._holdout_panel(axes[1], results)
        self._rate_panel(axes[2], results)
        for axis in axes:
            axis.set_facecolor(SURFACE)
            for side in ("top", "right"):
                axis.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                axis.spines[side].set_color(GRID_INK)
            axis.tick_params(colors=INK_SECONDARY, labelsize=9)
        figure.suptitle(
            "Tokens-per-parameter coverage at scale: models to 354M non-embedding params, C4",
            fontsize=12, color=INK_PRIMARY, y=0.99,
        )
        path = self._config.figures_dir / "summary.png"
        figure.tight_layout(rect=(0, 0, 1, 0.95))
        figure.savefig(path, dpi=150, facecolor=SURFACE)
        plt.close(figure)
        return path

    def _conditioning_panel(self, axis, results: dict) -> None:
        rays = results["co_rays"]
        control = results["equal_n_control"]
        rows = [
            ("Full 2D grid\n(29 cells)", results["full_grid_fits"]["Chinchilla"]["kappa_AB"], False),
            (f"2D spread, {control['n_2d']} pts\n(size-matched control)",
             control["twod_spread_kappa_AB"], False),
            ("Ray TPP~120\n(2 sub-ratios)", rays["120"]["kappa_AB"], False),
            ("Ray TPP~37\n(2 sub-ratios)", rays["37"]["kappa_AB"], False),
            (f"Single-TPP ray\n({rays['67']['n_points']} pts, K=1)", rays["67"]["kappa_AB"], True),
        ]
        labels = [r[0] for r in rows]
        values = [r[1] for r in rows]
        colours = [SERIES_COLLINEAR if r[2] else SERIES_DIVERSE for r in rows]
        positions = np.arange(len(rows))
        axis.barh(positions, values, color=colours, height=0.62)
        axis.set_yticks(positions)
        axis.set_yticklabels(labels, fontsize=8.5, color=INK_PRIMARY)
        axis.set_xscale("log")
        axis.set_xlim(1, max(values) * 12)
        for pos, value in zip(positions, values):
            axis.text(value * 1.5, pos, f"{value:,.0f}" if value < 1e4 else f"{value:.1e}",
                      va="center", fontsize=8.5, color=INK_PRIMARY)
        axis.axvline(1e4, color=INK_SECONDARY, ls=":", lw=1.2)
        axis.text(1e4 * 1.4, -0.62, "target $10^4$", fontsize=8, color=INK_SECONDARY)
        axis.set_ylim(-0.9, len(rows) - 0.3)
        axis.set_xlabel(r"$\kappa_{A,B}$  (lower is better, log scale)",
                        fontsize=9.5, color=INK_PRIMARY)
        axis.set_title("Collinear designs are unidentifiable", fontsize=10.5,
                       color=INK_PRIMARY, loc="left")
        axis.grid(axis="x", color=GRID_INK, lw=0.6, alpha=0.7)
        axis.set_axisbelow(True)

    def _holdout_panel(self, axis, results: dict) -> None:
        holdout = results["co_vs_nc_holdout"]
        samples = [holdout["rmse_co_samples"], holdout["rmse_nc_samples"]]
        parts = axis.boxplot(
            samples, vert=False, widths=0.5, patch_artist=True, showfliers=False,
            medianprops={"color": INK_PRIMARY, "lw": 2},
            whiskerprops={"color": INK_SECONDARY}, capprops={"color": INK_SECONDARY},
        )
        for patch, colour in zip(parts["boxes"], (SERIES_COLLINEAR, SERIES_DIVERSE)):
            patch.set_facecolor(colour)
            patch.set_edgecolor(SURFACE)
            patch.set_linewidth(2)
        axis.set_yticks([1, 2])
        axis.set_yticklabels(
            ["Collinear\n(single TPP)", "Non-collinear\n(2D)"], fontsize=9, color=INK_PRIMARY
        )
        axis.set_xlabel("holdout RMSE (lower is better)", fontsize=9.5, color=INK_PRIMARY)
        significance = results["robustness"]["winrate_significance"]
        axis.set_title(
            f"Non-collinear predicts off-ray better\n"
            f"wins {holdout['nc_wins']}/{holdout['trials']} "
            f"({holdout['nc_win_rate']:.0%}), binomial p={significance['p_binom_gt_half']:.3f}",
            fontsize=10.5, color=INK_PRIMARY, loc="left",
        )
        axis.grid(axis="x", color=GRID_INK, lw=0.6, alpha=0.7)
        axis.set_axisbelow(True)

    def _rate_panel(self, axis, results: dict) -> None:
        sweep = results["formal_validation"]["prop1_cor1"]
        gaps = np.asarray(sweep["eps_grid"])
        kappas = np.asarray(sweep["kappa_AB"])
        axis.loglog(gaps, kappas, lw=2, color=SERIES_DIVERSE, label="measured")
        axis.loglog(
            gaps, kappas[0] * (gaps / gaps[0]) ** -2.0, lw=2, ls="--",
            color=SERIES_REFERENCE, label=r"theory: $\varepsilon^{-2}$",
        )
        for entry in sweep["per_law"]:
            axis.scatter([entry["eps"]], [entry["kappa_AB_model"]], s=42, zorder=5,
                         color=SERIES_COLLINEAR, edgecolor=SURFACE, linewidth=1.5)
            axis.annotate(entry["law"].split("-")[0], (entry["eps"], entry["kappa_AB_model"]),
                          textcoords="offset points", xytext=(8, 6), fontsize=8,
                          color=INK_SECONDARY)
        axis.set_xlabel(r"exponent gap $\varepsilon = |\alpha - \beta|$",
                        fontsize=9.5, color=INK_PRIMARY)
        axis.set_ylabel(r"$\kappa_{A,B}$", fontsize=9.5, color=INK_PRIMARY)
        axis.set_title(
            f"Why: conditioning blows up as $\\varepsilon^{{-2}}$\n"
            f"measured slope {sweep['kappa_slope_loglog']:.2f} vs theory -2",
            fontsize=10.5, color=INK_PRIMARY, loc="left",
        )
        axis.legend(fontsize=8.5, frameon=False, labelcolor=INK_SECONDARY)
        axis.grid(color=GRID_INK, lw=0.6, alpha=0.7)
        axis.set_axisbelow(True)


class DesignScatter:
    """Flat N vs D scatter of the design points, train vs holdout marked.

    Mirrors the paper's appendix geometry (Appendix D.2): the holdout is the
    extrapolation frontier, held out *beyond* the training range, not an
    interior subset. Here the training region is the four smaller model sizes
    and the holdout is the largest model (354M non-embedding), so the fit is
    asked to extrapolate to a scale it never saw.
    """

    TRAIN = "#2a78d6"
    HOLDOUT = "#eb6834"
    INK = "#0b0b0b"
    INK2 = "#52514e"
    SURFACE = "#fcfcfb"
    GRID = "#d8d7d2"

    def __init__(self, config: AnalysisConfig, grid: TrainingGrid) -> None:
        self._config = config
        self._grid = grid

    def build(self) -> Path:
        split = frontier_split(self._grid)
        train = split.train
        holdout = split.holdout
        figure, axis = plt.subplots(figsize=(6.6, 5.2))
        figure.patch.set_facecolor(self.SURFACE)
        axis.set_facecolor(self.SURFACE)

        axis.scatter(train["N"], train["Dt"], s=80, c=self.TRAIN,
                     edgecolor=self.SURFACE, linewidth=1.2,
                     label=f"Train, N <= 85M ({len(train)})", zorder=3)
        axis.scatter(holdout["N"], holdout["Dt"], s=150, marker="D", c=self.HOLDOUT,
                     edgecolor=self.SURFACE, linewidth=1.2,
                     label=f"Holdout, 354M frontier ({len(holdout)})", zorder=4)

        d_all = self._grid.arrays.d
        n_line = np.array([self._grid.arrays.n.min(), self._grid.arrays.n.max()])
        for tpp in (20, 67, 200):
            axis.plot(n_line, tpp * n_line, color=self.GRID, lw=0.8, ls="--", zorder=1)
            axis.annotate(f"TPP {tpp}", (n_line[1], tpp * n_line[1]),
                          textcoords="offset points", xytext=(4, 2),
                          fontsize=7.5, color=self.INK2)

        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_ylim(d_all.min() * 0.6, d_all.max() * 1.8)
        axis.set_xlabel("N (non-embedding parameters)", fontsize=10, color=self.INK)
        axis.set_ylabel("D (training tokens)", fontsize=10, color=self.INK)
        n_runs = len(self._grid.runs)
        axis.set_title(
            f"Design grid: {len(self._grid.envelope)}-cell envelope (best of {n_runs} "
            f"LR/WD runs)\ntrain on N <= 85M ({len(train)}), extrapolate to 354M ({len(holdout)})",
            fontsize=10.5, color=self.INK, loc="left",
        )
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            axis.spines[side].set_color(self.GRID)
        axis.tick_params(colors=self.INK2)
        axis.legend(fontsize=9, frameon=False, labelcolor=self.INK, loc="upper left")
        axis.grid(color=self.GRID, lw=0.5, alpha=0.5)
        axis.set_axisbelow(True)

        path = self._config.figures_dir / "design_grid.png"
        figure.tight_layout()
        figure.savefig(path, dpi=150, facecolor=self.SURFACE)
        plt.close(figure)
        return path
