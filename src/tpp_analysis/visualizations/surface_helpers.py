"""Shared helpers for the 3-D surface panel of Figure 1.

Used by ``figure1_panels.py``, ``visuals.ipynb`` and
``special_viz_notebook_bf16_v2.ipynb``.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba


_HOLDOUT_STYLES: dict[str, dict] = {
    'holdout':      {'color': 'black',      'marker': 'o', 's': 40},
    'unified':      {'color': 'black',      'marker': 'o', 's': 40},
    'non-colinear': {'color': 'darkorange', 'marker': '^', 's': 40},
    'nc':           {'color': 'darkorange', 'marker': '^', 's': 40},
    'colinear':     {'color': 'purple',     'marker': 's', 's': 35},
    'co':           {'color': 'purple',     'marker': 's', 's': 35},
}


def _holdout_style(label: str) -> dict:
    low = label.lower()
    for key in sorted(_HOLDOUT_STYLES, key=len, reverse=True):
        if key in low:
            return _HOLDOUT_STYLES[key]
    return {'color': 'gray', 'marker': 'x', 's': 30}


def _r2_key_for_holdout(holdout_label: str) -> str:
    low = holdout_label.lower()
    if 'non' in low or 'nc' in low:
        return 'r2_noncolinear'
    if 'co' in low:
        return 'r2_colinear'
    return 'r2_unified'


def _build_adaptive_rgba(
    base_color: str, L_self: np.ndarray, L_other: np.ndarray,
    alpha_top: float = 0.40, alpha_bottom: float = 0.12,
) -> np.ndarray:
    rgba = np.full(
        L_self.shape + (4,), to_rgba(base_color, alpha=alpha_bottom),
    )
    on_top = L_self >= L_other
    rgba[on_top, 3] = alpha_top
    return rgba


def _camera_depth(
    n_arr: np.ndarray, d_arr: np.ndarray, z_arr: np.ndarray,
    view_elev: float, view_azim: float,
    n_lim: tuple[float, float], d_lim: tuple[float, float],
    z_lim: tuple[float, float],
) -> np.ndarray:
    n_norm = (n_arr - n_lim[0]) / (n_lim[1] - n_lim[0] + 1e-30)
    d_norm = (d_arr - d_lim[0]) / (d_lim[1] - d_lim[0] + 1e-30)
    z_norm = (z_arr - z_lim[0]) / (z_lim[1] - z_lim[0] + 1e-30)
    elev_r = np.radians(view_elev)
    azim_r = np.radians(view_azim)
    cam = np.array([
        np.cos(elev_r) * np.cos(azim_r),
        np.cos(elev_r) * np.sin(azim_r),
        np.sin(elev_r),
    ])
    return n_norm * cam[0] + d_norm * cam[1] + z_norm * cam[2]


def _draw_boundary_edges_occlusion(
    ax, N_mesh: np.ndarray, D_mesh: np.ndarray,
    L_self_clip: np.ndarray, L_other_clip: np.ndarray,
    color: str, view_elev: float, view_azim: float,
    n_lim: tuple[float, float], d_lim: tuple[float, float],
    z_lim: tuple[float, float],
    lw_top: float = 1.8, lw_under: float = 1.0,
    alpha_top: float = 0.85, alpha_under: float = 0.35,
) -> None:
    """Draw the perimeter silhouette of ``L_self`` with the
    dashed-when-behind treatment on the back-facing edges and
    always-solid on the front-facing edges (for the fixed
    elev=15/azim=-35/invert_yaxis surface camera)."""
    back_edges = [
        (L_self_clip[:, 0],  L_other_clip[:, 0],  N_mesh[:, 0],  D_mesh[:, 0]),
        (L_self_clip[0, :],  L_other_clip[0, :],  N_mesh[0, :],  D_mesh[0, :]),
    ]
    for z_self, z_other, n_arr, d_arr in back_edges:
        in_front = z_self >= z_other
        changes = np.where(np.diff(in_front.astype(int)) != 0)[0]
        boundaries = np.concatenate(
            [[0], changes + 1, [len(in_front)]],
        )
        for k in range(len(boundaries) - 1):
            start = boundaries[k]
            end = min(boundaries[k + 1] + 1, len(in_front))
            if end - start < 2:
                continue
            slc = slice(start, end)
            if in_front[start]:
                ax.plot(
                    n_arr[slc], d_arr[slc], z_self[slc],
                    color=color, lw=lw_top, alpha=alpha_top,
                    linestyle='-', zorder=3,
                )
            else:
                ax.plot(
                    n_arr[slc], d_arr[slc], z_self[slc],
                    color=color, lw=lw_under, alpha=alpha_under,
                    linestyle='--', zorder=1,
                )

    front_edges = [
        (L_self_clip[:, -1], N_mesh[:, -1], D_mesh[:, -1]),
        (L_self_clip[-1, :], N_mesh[-1, :], D_mesh[-1, :]),
    ]
    for z_self, n_arr, d_arr in front_edges:
        ax.plot(
            n_arr, d_arr, z_self,
            color=color, lw=lw_top, alpha=alpha_top,
            linestyle='-', zorder=3,
        )


def _compute_intersection_3d(
    N_mesh: np.ndarray, D_mesh: np.ndarray,
    L_co: np.ndarray, L_nc: np.ndarray,
    law_obj, co_params: tuple | list,
    z_lo: float, z_hi: float,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    diff = L_co - L_nc
    fig_tmp, ax_tmp = plt.subplots()
    cs = ax_tmp.contour(
        np.log10(N_mesh), np.log10(D_mesh), diff, levels=[0],
    )
    plt.close(fig_tmp)
    segments = []
    for level_segs in cs.allsegs:
        for verts in level_segs:
            if len(verts) < 2:
                continue
            N_int = 10 ** verts[:, 0]
            D_int = 10 ** verts[:, 1]
            L_int = law_obj.loss_fn(N_int, D_int, *co_params)
            L_int = np.clip(L_int, z_lo, z_hi)
            segments.append((N_int, D_int, L_int))
    return segments
