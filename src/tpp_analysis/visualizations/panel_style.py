"""Shared sizing constants and style helpers used by every panel
(`figure1_panels.py`, `visuals.ipynb`, `special_viz_notebook_bf16_v2.ipynb`).

Central source of truth for fonts, line widths, tick parameters,
grid colors, legend styling and errorbar darkening so all three
panels (surface, convergence, isoFLOP) read at the same density.
"""
from matplotlib.legend import Legend
from matplotlib.container import ErrorbarContainer
from matplotlib.ticker import ScalarFormatter


SURF_W: int = 9
SURF_H: int = 9
PANEL_W: int = 9
PANEL_H: int = 9

AXIS_LABEL_FS: int = 25
TICK_LABEL_FS: int = 20
LEGEND_FS: int = 20
TITLE_FS: int = 22

TICK_MAJOR_LEN: int = 10
TICK_MAJOR_WID: float = 2.0
TICK_MINOR_LEN: int = 6
TICK_MINOR_WID: float = 1.2

LINE_WIDTH: float = 2.5
MARKER_SIZE: int = 8
SPINE_LW: float = 1.5

INTERSECTION_LW: float = 3.5
INTERSECTION_LS: str = ':'
INTERSECTION_ALPHA: float = 0.95

GRID_BG_COLOR: str = '#D8D8D8'
LEGEND_ALPHA: float = 0.5

GRID_LINE_COLOR: str = '#a8a8a8'
GRID_LINE_WIDTH: float = 1.6
GRID_LINE_ALPHA: float = 1.0

DPI: int = 300


def _sv_tag(dataset: str, law: str, mode: str) -> str:
    return f"{dataset.replace(' ', '')}_{law}_{mode.lower()}"


class FixedOrderFormatter(ScalarFormatter):
    """Lock the offset to a fixed power of 10 with integer tick labels."""

    def __init__(self, oom: int = 8, fmt: str = '%d', **kw):
        super().__init__(useMathText=False, useOffset=False, **kw)
        self._fixed_oom = oom
        self._fixed_fmt = fmt
        self.set_scientific(True)

    def set_locs(self, locs):
        super().set_locs(locs)
        self.orderOfMagnitude = self._fixed_oom
        self.format = self._fixed_fmt


def style_ticks_2d(ax) -> None:
    """Big visible ticks, full bounding box on all four sides, and a
    grid that visually matches the 3-D plot's pane gridlines."""
    ax.tick_params(
        axis='both', which='major',
        length=TICK_MAJOR_LEN, width=TICK_MAJOR_WID,
        labelsize=TICK_LABEL_FS, direction='in',
        top=True, right=True,
    )
    ax.tick_params(
        axis='both', which='minor',
        length=TICK_MINOR_LEN, width=TICK_MINOR_WID,
        direction='in', top=True, right=True,
    )
    for spine_name in ('top', 'bottom', 'left', 'right'):
        spine = ax.spines[spine_name]
        spine.set_visible(True)
        spine.set_linewidth(SPINE_LW)
        spine.set_color('black')
    ax.set_facecolor(GRID_BG_COLOR)
    ax.grid(
        True, which='major',
        color=GRID_LINE_COLOR,
        linewidth=GRID_LINE_WIDTH,
        alpha=GRID_LINE_ALPHA,
        zorder=0,
    )


def style_ticks_3d(ax) -> None:
    """Big visible ticks on a 3-D Axes, with bounding box."""
    ax.tick_params(axis='x', labelsize=TICK_LABEL_FS, pad=5)
    ax.tick_params(axis='y', labelsize=TICK_LABEL_FS, pad=5)
    ax.tick_params(axis='z', labelsize=TICK_LABEL_FS, pad=8)
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.pane.set_edgecolor('black')
        axis.pane.set_alpha(1.0)
        axis.line.set_linewidth(SPINE_LW)
        axis.pane.fill = True
        axis.pane.set_facecolor(GRID_BG_COLOR)


def _style_legend(legend, alpha: float | None = None) -> None:
    """Restyle a legend frame to a uniform white/black/square look."""
    if legend is None:
        return
    if alpha is None:
        alpha = LEGEND_ALPHA
    frame = legend.get_frame()
    frame.set_facecolor('white')
    frame.set_edgecolor('black')
    frame.set_linewidth(SPINE_LW)
    frame.set_alpha(alpha)
    try:
        frame.set_boxstyle('square', pad=0.4)
    except Exception:
        pass


def _style_all_legends(ax, alpha: float | None = None) -> None:
    """Restyle every legend attached to ``ax`` (handles isoFLOP's
    two-legend setup where the budget legend is added via
    ``ax.add_artist``)."""
    if alpha is None:
        alpha = LEGEND_ALPHA
    primary = ax.get_legend()
    if primary is not None:
        _style_legend(primary, alpha)
    for art in ax.get_children():
        if isinstance(art, Legend) and art is not primary:
            _style_legend(art, alpha)


def _darken_errorbars(
    ax, color: str = 'black', alpha: float = 1.0,
    linewidth: float = 2.0,
) -> None:
    """Restyle every existing errorbar so it pops against the
    ``GRID_BG_COLOR`` panel background."""
    for container in ax.containers:
        if not isinstance(container, ErrorbarContainer):
            continue
        _, caplines, barlinecols = container.lines
        for line in caplines or ():
            line.set_color(color)
            line.set_alpha(alpha)
            line.set_markeredgewidth(linewidth)
        for collection in barlinecols or ():
            collection.set_color(color)
            collection.set_alpha(alpha)
            collection.set_linewidth(linewidth)
