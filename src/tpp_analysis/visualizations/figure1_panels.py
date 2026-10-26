#!/usr/bin/env python
# coding: utf-8

# # Individual Panel Figures: Surface · Convergence · IsoFLOP
# 
# Produces **separate** image files for each panel, suitable for Overleaf.  
# Font sizes, tick sizes, and panel dimensions are pinned to the **surface-plot baseline**
# (`axis_label=25`, `tick_label=16`, `legend=20`, panel ≈ 9 × 9).
# 
# **Inputs required (same as the viz notebook):**
# 1. `pipeline_results.pkl` - fitted params for surface + isoflop
# 2. `*_incremental.parquet` - R² convergence curves
# 3. CSV epoch-results files - for `load_and_prepare` (holdout scatter)
# 

# In[ ]:


# ============================================================
# 0 · SETUP
# ============================================================
import sys
import os
import glob
import importlib

_HERE = os.path.dirname(os.path.abspath(__file__))
_TPP = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, _TPP)  # resolves scaling_laws, incremental_coverage, AND seed_variance
sys.path.insert(0, _HERE)  # resolves the local isoflop_panel module

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family']      = 'STIXGeneral'

# ── Paths (script lives in src/tpp_analysis/visualizations/, repo root 3 levels up) ──
PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..', '..'))
# Inputs and outputs live at the repository root, like every other analysis script.
DATA_ROOT    = os.environ.get('TPP_DATA_ROOT', PROJECT_ROOT)
DATA_DIR     = os.path.join(DATA_ROOT, 'extracted_losses')
SAVE_DIR     = os.path.join(DATA_ROOT, 'full_coverage_results')
FIG_DIR      = os.path.join(SAVE_DIR, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

SV_SAVE_DIR  = os.path.join(DATA_ROOT, 'seed_variance_results')

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

print(f'PROJECT_ROOT: {PROJECT_ROOT}')
print(f'DATA_DIR:     {DATA_DIR}')
print(f'SAVE_DIR:     {SAVE_DIR}')
print(f'FIG_DIR:      {FIG_DIR}')
print(f'SV_SAVE_DIR:  {SV_SAVE_DIR}')


# In[ ]:


# ============================================================
# 1 · IMPORTS FROM PROJECT MODULES
# ============================================================
from scaling_laws import load_and_prepare
from scaling_laws.data import prepare_arrays
from scaling_laws.laws import LAWS

import incremental_coverage.viz as _viz_mod
importlib.reload(_viz_mod)
from incremental_coverage.viz import DESIGN_COLORS

from seed_variance import plot_convergence_with_errorbars

from isoflop_panel import (
    compute_flop_budgets,
    draw_isoflop_panel as _draw_isoflop_generic,
)
from panel_style import (
    SURF_W, SURF_H, PANEL_W, PANEL_H,
    AXIS_LABEL_FS, TICK_LABEL_FS, LEGEND_FS, LINE_WIDTH, MARKER_SIZE, LEGEND_ALPHA,
    DPI,
    _sv_tag,
    FixedOrderFormatter,
    style_ticks_2d, style_ticks_3d,
    _style_all_legends,
    _darken_errorbars,
)
from surface_helpers import (
    _holdout_style,
    _r2_key_for_holdout,
    _build_adaptive_rgba,
    _draw_boundary_edges_occlusion,
)

print('Imports OK')


# In[ ]:


# ============================================================
# 2 · LOAD DATA
# ============================================================

# ── Helper: reconstruct pipeline_results from full_coverage parquet ──
def load_pipeline_from_parquet(parquet_path):
    """
    Convert full_coverage_all_seeds.parquet → list of dicts matching
    the old pipeline_results.pkl format that the viz code expects.

    For each (dataset, law, mode, design), picks seed=0 and reconstructs
    the 'params' list in the correct order for loss_fn(N, D, *params).
    """
    df = pd.read_parquet(parquet_path)

    # Parameter column ordering per law (must match loss_fn signature)
    PARAM_ORDER = {
        'Chinchilla':    ['param_0', 'param_1', 'param_2', 'param_3', 'param_4'],  # E, A, α, B, β
        'Kaplan':        ['param_0', 'param_1', 'param_2', 'param_3'],              # Nc, Dc, αN, αD
        'Droppo-Elibol': ['param_0', 'param_1', 'param_2', 'param_3', 'param_4', 'param_5'],  # L∞, Nc, Dc, αN, αD, α
        'Parametric-Epoch': ['param_A', 'param_alpha', 'param_B', 'param_beta',
                             'param_E', 'param_R_star_D', 'param_R_star_N'],
    }

    DESIGN_MAP = {'CO': 'Colinear', 'NC': 'Non-Colinear'}
    DS_NAME_MAP = {'RedPajama': 'Red Pajama'}  # parquet → viz naming

    results = []
    # Pick seed=0 for deterministic single-fit results
    seed0 = df[df['seed'] == 0]

    for (ds, law, mode, design), row in seed0.groupby(['dataset', 'law', 'mode', 'design']):
        row = row.iloc[0]
        param_cols = PARAM_ORDER.get(law, [])
        params = [float(row[c]) for c in param_cols if c in row.index and pd.notna(row[c])]

        entry = {
            'dataset': DS_NAME_MAP.get(ds, ds),
            'law': law,
            'mode': mode.capitalize() if mode != 'Epoch-Expanded' else 'Epoch-Expanded',
            'design': DESIGN_MAP.get(design, design),
            'params': params,
            'r2_train': row.get('r2_train', np.nan),
            'r2_unified': row.get('r2_unified', np.nan),
            'r2_colinear': row.get('r2_colinear', np.nan),
            'r2_noncolinear': row.get('r2_noncolinear', np.nan),
            'rmse_train': row.get('rmse_train', np.nan),
            'rmse_unified': row.get('rmse_unified', np.nan),
            'rmse_colinear': row.get('rmse_colinear', np.nan),
            'rmse_noncolinear': row.get('rmse_noncolinear', np.nan),
        }
        results.append(entry)

    return results


# ── 2a. Incremental convergence data ──
# Convergence panel (Panel B) primarily uses seed_variance parquets.
# inc_results is a fallback for single-seed line plots; it requires
# the incremental_coverage analysis which is separate from full_coverage.
# If seed_variance_results/ exists, this fallback is not needed.
inc_results = {}
print('Convergence plots will use seed_variance_results/ (preferred)')


# ── 2b. Seed-variance parquets ──
sv_parquet_files = sorted(glob.glob(os.path.join(SV_SAVE_DIR, 'seed_variance_*.parquet')))
print(f'\nFound {len(sv_parquet_files)} seed-variance parquets:')

sv_results = {}
for f_path in sv_parquet_files:
    tag = os.path.basename(f_path).replace('seed_variance_', '').replace('.parquet', '')
    sv_results[tag] = pd.read_parquet(f_path)
    print(f'  {tag}: {len(sv_results[tag])} rows')

if not sv_results:
    print('  (none found - Panel B will fall back to single-seed lines)')


# ── 2c. Pipeline results (from full_coverage parquet, not pickle) ──
master_parquet = os.path.join(SAVE_DIR, 'full_coverage_all_seeds.parquet')
if os.path.exists(master_parquet):
    pipeline_results = load_pipeline_from_parquet(master_parquet)
    print(f'\nPipeline results: {len(pipeline_results)} entries (from parquet)')
else:
    pipeline_results = None
    print(f'⚠️  {master_parquet} not found - run run_analysis.py first')


# ── 2d. PreparedData from extracted CSVs ──
DATASETS = {
    'Wikipedia':  'wikipedia.csv',
    'Cosmopedia': 'cosmopedia.csv',
    'Red Pajama': 'redpajama.csv',
    'Pes2o':      'pes2o.csv',
    'C4':         'c4.csv',
}

prepared_data = {}
for name, csv_file in DATASETS.items():
    fpath = os.path.join(DATA_DIR, csv_file)
    if os.path.exists(fpath):
        prepared_data[name] = load_and_prepare(fpath, name, verbose=False)
    else:
        print(f'  ⚠️  {csv_file} not found, skipping {name}')

print(f'\nPreparedData: {list(prepared_data.keys())}')


# ---
# ## 3 · Shared Sizing Constants
# 
# Everything inherits from the surface-plot baseline so the panels
# look uniform at any scale-down.
# 

# In[ ]:


# ============================================================
# 3 · GLOBAL SIZING - surface plot is the baseline
# ============================================================
# All sizing/style constants and helpers (including _sv_tag) are
# imported from panel_style at the top of this file.
print('Sizing constants set.')


# ---
# ## 4 · Loss Functions, IsoFLOP Helpers, & Surface Helpers

# In[ ]:


# ============================================================
# 4 · LOSS FUNCTIONS + ISOFLOP HELPERS
# ============================================================
# LAW_LOSS_FNS, ISOFLOP_COLORS, generic_isoflop, generic_optimal_N
# and the renderer (_draw_isoflop_generic = draw_isoflop_panel)
# are imported from isoflop_panel at the top of this file.

print('Loss functions + isoFLOP helpers + _draw_isoflop_generic imported.')


# In[ ]:


# ============================================================
# 5 · SURFACE-PLOT HELPERS  (imported from surface_helpers)
# ============================================================
print('Surface-plot helpers imported.')


# ============================================================
# 6 · ISOFLOP BUDGET HELPER  (imported from isoflop_panel)
# ============================================================
print('compute_flop_budgets imported.')


# ============================================================
# 7 · TICK / LEGEND / ERRORBAR STYLING  (imported from panel_style)
# ============================================================
print('Tick + legend + errorbar styling helpers imported.')


# ---
# ## 8 · Pick Your Combo & Build Individual Panels
# 

# In[ ]:


# ============================================================
# 8 · CONFIGURATION - pick the (dataset, law, mode) combo
# ============================================================

DATASET_NAME = os.environ.get('FIG1_DATASET',  'Red Pajama')
LAW_NAME     = os.environ.get('FIG1_LAW',      'Kaplan')
MODE_NAME    = os.environ.get('FIG1_MODE',     'First')
OUT_PREFIX   = os.environ.get('FIG1_PREFIX',   '01_')

HOLDOUT_COL  = 'r2_unified'
HOLDOUT_KEY  = 'Unified'          # for surface plot
HOLDOUT_TYPE = 'unified'          # for isoflop thinning
MIN_COVERAGE = 0.15

# ── Surface-specific ──
SIDE_ELEV    = 15
SIDE_AZIM    = -35
CO_COLOR     = 'steelblue'
NC_COLOR     = 'indianred'

# ── IsoFLOP-specific (zoomed in to match Big TPP style) ──
N_RANGE           = np.logspace(np.log10(2e7), np.log10(2e8), 500)
Y_LIM_ISO         = None
SHOW_OPTIMAL      = False   # ridge is off-screen at this zoom
N_ISOFLOP_CURVES  = 5       # always exactly 5 curves = 5 colors

# ── Convergence-specific ──
Y_MIN_CONV   = -0.1
Y_MAX_CONV   = 1.02

print(f'Triptych: {DATASET_NAME} / {LAW_NAME} / {MODE_NAME}')


# In[ ]:


# ============================================================
# 9 · RESOLVE DATA FOR ALL THREE PANELS
# ============================================================

assert pipeline_results is not None, 'Need pipeline_results.pkl'
assert DATASET_NAME in prepared_data,  f'No PreparedData for {DATASET_NAME}'
# inc_results not needed if seed_variance parquets exist (convergence panel B)

# ── CO / NC fit results from pipeline ──
matches = [r for r in pipeline_results
           if r['law'] == LAW_NAME
           and r['mode'] == MODE_NAME
           and r['dataset'] == DATASET_NAME]
co_result = [r for r in matches if r['design'] == 'Colinear'][0]
nc_result = [r for r in matches if r['design'] == 'Non-Colinear'][0]

# ── Law object ──
law_obj = next(l for l in LAWS if l.name == LAW_NAME)

# ── PreparedData + holdout ──
pdata = prepared_data[DATASET_NAME]

holdout_map = {
    'Unified':    pdata.unified_holdout,
    'CO Holdout': pdata.colinear_holdout,
    'NC Holdout': pdata.noncolinear_holdout,
}
holdouts = {HOLDOUT_KEY: holdout_map[HOLDOUT_KEY]}

# ── IsoFLOP budgets: top 25% of C, every Nth ──
flop_budgets = compute_flop_budgets(
    pdata, holdout_type=HOLDOUT_TYPE, n_curves=N_ISOFLOP_CURVES)

print(f'IsoFLOP budgets: {len(flop_budgets)} values (top 25%, {N_ISOFLOP_CURVES} evenly spaced):')
for c in flop_budgets:
    print(f'  {c:.3e}')

print(f'\nFull holdout (no thinning): {len(pdata.unified_holdout)} unified, '
      f'{len(pdata.colinear_holdout)} CO, '
      f'{len(pdata.noncolinear_holdout)} NC')

# ── R² values for surface legend ──
_h_label = next(iter(holdouts))
_r2_key  = _r2_key_for_holdout(_h_label)
co_r2 = co_result.get(_r2_key, co_result.get('r2_train', float('nan')))
nc_r2 = nc_result.get(_r2_key, nc_result.get('r2_train', float('nan')))

print(f'\nCO R²={co_r2:.4f}  NC R²={nc_r2:.4f}')
print('Ready to plot.')


# In[ ]:


# ============================================================
# 10 · DRAW INDIVIDUAL PANEL FIGURES
# ============================================================

from matplotlib.ticker import ScalarFormatter, MultipleLocator

ds_tag   = DATASET_NAME.replace(' ', '_')
law_tag  = LAW_NAME.replace(' ', '_').replace('-', '_')
mode_tag = MODE_NAME.replace(' ', '_')
base_tag = f'{ds_tag}_{law_tag}_{mode_tag}'

# Extra padding around 3-D plots so rotated z-labels don't get cropped
SAVE_PAD_3D = 0.4
SAVE_PAD_2D = 0.1

# ══════════════════════════════════════════════════════════════
# PANEL A - Surface Plot (3-D)
# ══════════════════════════════════════════════════════════════
fig_surf = plt.figure(figsize=(SURF_W, SURF_H))
ax_surf = fig_surf.add_subplot(111, projection='3d')

co_params = co_result['params']
nc_params = nc_result['params']

all_N, all_D = np.array([]), np.array([])
prepped = {}
for h_label, h_list in holdouts.items():
    if not h_list: continue
    Nh, Dh, Lh = prepare_arrays(h_list, 'final')
    if len(Lh) > 0:
        prepped[h_label] = (Nh, Dh, Lh)
        all_N = np.concatenate([all_N, Nh])
        all_D = np.concatenate([all_D, Dh])

N_1d = np.logspace(np.log10(all_N.min()), np.log10(all_N.max()), 200)
D_1d = np.logspace(np.log10(all_D.min()), np.log10(all_D.max()), 200)
N_mesh, D_mesh = np.meshgrid(N_1d, D_1d)

L_co = law_obj.loss_fn(N_mesh, D_mesh, *co_params)
L_nc = law_obj.loss_fn(N_mesh, D_mesh, *nc_params)

all_L = np.concatenate([v[2] for v in prepped.values()])
L_lo  = np.nanpercentile(all_L, 1) * 0.5
L_hi  = np.nanpercentile(all_L, 99) * 1.5
L_co_clip = np.clip(L_co, L_lo, L_hi)
L_nc_clip = np.clip(L_nc, L_lo, L_hi)
z_lo = min(L_lo, L_co_clip.min(), L_nc_clip.min()) * 0.95
z_hi = max(L_hi, L_co_clip.max(), L_nc_clip.max()) * 1.05
n_lim = (N_mesh.min(), N_mesh.max())
d_lim = (D_mesh.min(), D_mesh.max())
z_lim = (z_lo, z_hi)

co_rgba = _build_adaptive_rgba(CO_COLOR, L_co_clip, L_nc_clip)
nc_rgba = _build_adaptive_rgba(NC_COLOR, L_nc_clip, L_co_clip)

ax_surf.plot_surface(N_mesh, D_mesh, L_co_clip,
                     facecolors=co_rgba, edgecolor='none',
                     rstride=2, cstride=2, shade=False, zorder=1)
ax_surf.plot_surface(N_mesh, D_mesh, L_nc_clip,
                     facecolors=nc_rgba, edgecolor='none',
                     rstride=2, cstride=2, shade=False, zorder=2)

_draw_boundary_edges_occlusion(ax_surf, N_mesh, D_mesh,
                                L_co_clip, L_nc_clip, CO_COLOR,
                                SIDE_ELEV, SIDE_AZIM, n_lim, d_lim, z_lim)
_draw_boundary_edges_occlusion(ax_surf, N_mesh, D_mesh,
                                L_nc_clip, L_co_clip, NC_COLOR,
                                SIDE_ELEV, SIDE_AZIM, n_lim, d_lim, z_lim)

ax_surf.plot([], [], [], color=CO_COLOR, lw=6, alpha=0.8,
             label=f'CO (R²={co_r2:.3f})')
ax_surf.plot([], [], [], color=NC_COLOR, lw=6, alpha=0.8,
             label=f'NC (R²={nc_r2:.3f})')

for h_label, (Nh, Dh, Lh) in prepped.items():
    sty = _holdout_style(h_label)
    ax_surf.scatter(Nh, Dh, Lh, c=sty['color'], s=sty['s'], alpha=0.85,
                    edgecolors='white', linewidths=0.5,
                    marker=sty['marker'], label=h_label.replace('Unified', 'Holdout').replace('unified', 'Holdout'), zorder=5)

bar_key  = next(iter(prepped))
N_b, D_b, L_b = prepped[bar_key]
L_pred_co = law_obj.loss_fn(N_b, D_b, *co_params)
L_pred_nc = law_obj.loss_fn(N_b, D_b, *nc_params)
tick_len  = (all_N.max() - all_N.min()) * 0.008
drawn_co = drawn_nc = False
for i in range(len(N_b)):
    ni, di, li = N_b[i], D_b[i], L_b[i]
    ax_surf.plot([ni,ni],[di,di],[li, L_pred_co[i]],
                color='green', lw=1.8, alpha=0.8, zorder=4,
                label='CO residual' if not drawn_co else None)
    ax_surf.plot([ni-tick_len,ni+tick_len],[di,di],
                [L_pred_co[i],L_pred_co[i]],
                color='green', lw=1.0, alpha=0.8, zorder=4)
    drawn_co = True
    ax_surf.plot([ni,ni],[di,di],[li, L_pred_nc[i]],
                color='black', lw=1.8, alpha=0.8, zorder=4,
                label='NC residual' if not drawn_nc else None)
    ax_surf.plot([ni-tick_len,ni+tick_len],[di,di],
                [L_pred_nc[i],L_pred_nc[i]],
                color='black', lw=1.0, alpha=0.8, zorder=4)
    drawn_nc = True

# Axis labels - plain "D (tokens)" with no embedded multiplier; matplotlib's
# ScalarFormatter (set below) renders the offset as a separate "1eN" text
# the same way the N axis already does.
# Plain 'N' / 'D' - the offset multiplier is rendered separately by the
# formatter, the same way N's "1e7" auto-offset reads.  labelpad bumped
# from 10 → 18 so the axis names sit clear of the tick numbers.
ax_surf.set_xlabel('N', fontsize=AXIS_LABEL_FS, labelpad=18)
ax_surf.set_ylabel('D', fontsize=AXIS_LABEL_FS, labelpad=18)
# Smaller labelpad keeps the rotated "Loss" text inside the savefig bbox.
# 'Loss' placed via text2D in axes-fraction coords - bypasses the 3-D
# rotated-zlabel bbox quirk that was causing 'Loss' to get cropped by
# bbox_inches='tight' even with generous pad_inches.
ax_surf.set_zlabel('')
ax_surf.text2D(1.08, 0.5, 'Loss',
               transform=ax_surf.transAxes,
               rotation=90, va='center', ha='left',
               fontsize=AXIS_LABEL_FS)

# N axis: standard auto-offset (gives "1e7" with integer-ish ticks).
_sf_n = ScalarFormatter(useMathText=False)
_sf_n.set_scientific(True)
_sf_n.set_powerlimits((0, 0))
ax_surf.xaxis.set_major_formatter(_sf_n)

# D axis: tick spacing fixed at 2e8 with integer labels (0, 2, ..., 14)
# and offset locked at 1e8, so it visually matches N's '1e7' offset.
ax_surf.yaxis.set_major_locator(MultipleLocator(2e8))
ax_surf.yaxis.set_major_formatter(FixedOrderFormatter(oom=8, fmt='%d'))

ax_surf.invert_yaxis()
ax_surf.set_zlim(z_lo, z_hi)
ax_surf.view_init(elev=SIDE_ELEV, azim=SIDE_AZIM)
ax_surf.legend(fontsize=LEGEND_FS, loc='upper left', framealpha=LEGEND_ALPHA)
style_ticks_3d(ax_surf)
_style_all_legends(ax_surf)

for ext in ['pdf', 'png']:
    out = os.path.join(FIG_DIR, f'{OUT_PREFIX}surface_{base_tag}.{ext}')
    fig_surf.savefig(out, dpi=DPI, bbox_inches='tight', pad_inches=SAVE_PAD_3D)
    print(f'Saved: {out}')
plt.show()
plt.close(fig_surf)


# ══════════════════════════════════════════════════════════════
# PANEL B - R² Convergence with Seed-Variance Error Bars (2-D)
# ══════════════════════════════════════════════════════════════
fig_conv = plt.figure(figsize=(PANEL_W, PANEL_H))
ax_conv = fig_conv.add_subplot(111)

sv_tag = _sv_tag(DATASET_NAME, LAW_NAME, MODE_NAME)
sv_path = os.path.join(SV_SAVE_DIR, f"seed_variance_{sv_tag}.parquet")

def _auto_ymin_from_sv(sv_df, anchor_cov=0.20, margin=0.05, floor=-0.5):
    if sv_df is None or sv_df.empty or 'coverage_frac' not in sv_df.columns:
        return Y_MIN_CONV
    r2_col = next(
        (c for c in ('r2_unified', 'r2_holdout_unified', 'r2_unified_holdout')
         if c in sv_df.columns),
        None,
    )
    if r2_col is None:
        cands = [c for c in sv_df.columns if 'r2' in c.lower()]
        r2_col = cands[0] if cands else None
    if r2_col is None:
        return Y_MIN_CONV
    sub = sv_df.loc[sv_df['coverage_frac'] >= anchor_cov, r2_col].dropna()
    if sub.empty:
        return Y_MIN_CONV
    return max(float(sub.min()) - margin, floor)

if os.path.exists(sv_path):
    sv_df = pd.read_parquet(sv_path)
    y_min_use = _auto_ymin_from_sv(sv_df) if Y_MIN_CONV is None else Y_MIN_CONV

    plot_convergence_with_errorbars(
        sv_df,
        holdout_split='unified',
        ax=ax_conv,
        y_min=y_min_use,
        y_max=Y_MAX_CONV,
        min_coverage=MIN_COVERAGE,
        band_type='percentile',
        axis_label_fs=AXIS_LABEL_FS,
        tick_label_fs=TICK_LABEL_FS,
        legend_fs=LEGEND_FS,
        marker_size=MARKER_SIZE,
        line_width=LINE_WIDTH,
    )
    leg = ax_conv.get_legend()
    if leg is not None:
        for txt in leg.get_texts():
            new = txt.get_text().replace('Non-Colinear', 'NC').replace('Colinear', 'CO')
            txt.set_text(new)
    ax_conv.set_ylabel(r'Holdout $R^2$', fontsize=AXIS_LABEL_FS)
    print(f'Panel B: seed-variance plot (tag={sv_tag}, y_min={y_min_use:.3f})')
else:
    print(f'Panel B: no seed-variance data for tag={sv_tag}, using single-seed fallback')
    if DATASET_NAME not in inc_results:
        print('  No incremental data - run seed_variance for convergence plots')
        ax_conv.text(0.5, 0.5, 'No convergence data\nRun seed_variance first',
                     transform=ax_conv.transAxes, ha='center', va='center', fontsize=14, color='gray')
    else:
        inc_df = inc_results[DATASET_NAME]
        conv_sub = inc_df[
            (inc_df['law'] == LAW_NAME)
            & (inc_df['mode'] == MODE_NAME)
            & inc_df['success']
        ].copy()
        if MIN_COVERAGE is not None:
            conv_sub = conv_sub[conv_sub['coverage_frac'] >= MIN_COVERAGE]
        if Y_MIN_CONV is None:
            anchor = conv_sub.loc[conv_sub['coverage_frac'] >= 0.20, HOLDOUT_COL].dropna()
            y_min_use = max(float(anchor.min()) - 0.05, -0.5) if not anchor.empty else -0.1
        else:
            y_min_use = Y_MIN_CONV

        _ABBREV = {'Colinear': 'CO', 'Non-Colinear': 'NC'}
        for design in ['Colinear', 'Non-Colinear']:
            dsub = conv_sub[conv_sub['design'] == design].sort_values('coverage_frac')
            if dsub.empty:
                continue
            ax_conv.plot(
                dsub['coverage_frac'], dsub[HOLDOUT_COL],
                color=DESIGN_COLORS.get(design, 'gray'),
                marker='o' if design == 'Colinear' else 's',
                markersize=MARKER_SIZE,
                linewidth=LINE_WIDTH,
                label=_ABBREV[design],
            )
        ax_conv.set_xlabel('Coverage Fraction', fontsize=AXIS_LABEL_FS)
        ax_conv.set_ylabel(r'Holdout $R^2$',    fontsize=AXIS_LABEL_FS)
        ax_conv.set_xlim(-0.02, 1.05)
        ax_conv.set_ylim(y_min_use, Y_MAX_CONV)
        ax_conv.legend(fontsize=LEGEND_FS, loc='lower right',
                       framealpha=LEGEND_ALPHA)
        ax_conv.grid(True, alpha=0.3)

style_ticks_2d(ax_conv)
# Darken the errorbars seed_variance drew at #333333/alpha=0.7 so they
# read clearly against the new GRID_BG_COLOR (#D8D8D8) background.
_darken_errorbars(ax_conv)
# Forces white face, black edge of width SPINE_LW, alpha=LEGEND_ALPHA on the
# convergence legend regardless of how plot_convergence_with_errorbars built it
_style_all_legends(ax_conv)

for ext in ['pdf', 'png']:
    out = os.path.join(FIG_DIR, f'{OUT_PREFIX}convergence_{base_tag}.{ext}')
    fig_conv.savefig(out, dpi=DPI, bbox_inches='tight', pad_inches=SAVE_PAD_2D)
    print(f'Saved: {out}')
plt.show()
plt.close(fig_conv)


# ══════════════════════════════════════════════════════════════
# PANEL C - IsoFLOP Curves (2-D)
# ══════════════════════════════════════════════════════════════
fig_iso = plt.figure(figsize=(PANEL_W, PANEL_H))
ax_iso = fig_iso.add_subplot(111)

iso_fits = [r for r in pipeline_results
            if r['law'] == LAW_NAME
            and r['mode'] == MODE_NAME
            and r['dataset'] == DATASET_NAME]

if HOLDOUT_TYPE == 'unified':
    h_thin = pdata.unified_holdout
elif HOLDOUT_TYPE == 'colinear':
    h_thin = pdata.colinear_holdout
else:
    h_thin = pdata.noncolinear_holdout

_draw_isoflop_generic(
    ax_iso, iso_fits, LAW_NAME, h_thin,
    'Holdout',
    flop_budgets, N_RANGE, Y_LIM_ISO,
    x_lim=(None, 1e8),
    show_optimal=SHOW_OPTIMAL,
    fontsize=AXIS_LABEL_FS,
    legend_fontsize=LEGEND_FS - 4,
    legend_loc='lower left',
    line_width=LINE_WIDTH,
    holdout_mode=MODE_NAME.lower(),
    style_legends_fn=_style_all_legends,
)
style_ticks_2d(ax_iso)
# _draw_isoflop_generic already calls _style_all_legends; this is a belt-and-
# suspenders second pass in case anything got reattached.
_style_all_legends(ax_iso)

for ext in ['pdf', 'png']:
    out = os.path.join(FIG_DIR, f'{OUT_PREFIX}isoflop_{base_tag}.{ext}')
    fig_iso.savefig(out, dpi=DPI, bbox_inches='tight', pad_inches=SAVE_PAD_2D)
    print(f'Saved: {out}')
plt.show()
plt.close(fig_iso)

print('Done.')


# ---
# ## 11 · Batch: Top-N Individual Panels
# 
# Optionally loop over the ranking table and generate individual panel figures for each top candidate.
# 

