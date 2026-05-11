"""Shared plotting style helpers for analysis and illustration notebooks."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt


WHITE_STYLE = {
    "figure.facecolor": "white",
    "figure.edgecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#222222",
    "axes.labelcolor": "#222222",
    "axes.titlecolor": "#222222",
    "xtick.color": "#222222",
    "ytick.color": "#222222",
    "text.color": "#222222",
    "grid.color": "#c7c7c7",
    "grid.alpha": 0.45,
    "legend.facecolor": "white",
    "legend.edgecolor": "#bbbbbb",
    "savefig.facecolor": "white",
    "savefig.edgecolor": "white",
    "savefig.transparent": False,
}

STACK_COLORS = {
    "appliance": "#4E79A7",
    "hp": "#F28E2B",
    "ev": "#59A14F",
    "gas": "#B23A48",
    "total": "#111111",
    "tin": "#2F4B7C",
    "tank": "#8C564B",
    "band": "#7FB3D5",
    "hp_hw": "#FFBE7D",
    "bo_hw": "#D37295",
    "soc": "#6A3D9A",
    "price": "#0072B2",
    "p2_5": "#1F77B4",
    "p97_5": "#D62728",
}


def apply_white_style(*, reset: bool = True, overrides: dict[str, Any] | None = None) -> None:
    """Apply the repository's white-background plotting defaults."""

    if reset:
        import matplotlib as mpl

        mpl.rcParams.update(mpl.rcParamsDefault)
        plt.style.use("default")
    style = dict(WHITE_STYLE)
    if overrides:
        style.update(overrides)
    plt.rcParams.update(style)


def configure_inline_backend() -> None:
    """Keep inline notebook figures on a white background when IPython exists."""

    try:
        from IPython import get_ipython

        ip = get_ipython()
        if ip is not None:
            ip.run_line_magic(
                "config",
                "InlineBackend.print_figure_kwargs = {'facecolor': 'white', 'edgecolor': 'white'}",
            )
    except Exception:
        pass


def clean_axes(ax, *, grid_axis: str = "y", spine_color: str = "#222222"):
    """Apply the standard axes styling to one Matplotlib axis."""

    ax.set_facecolor("white")
    ax.tick_params(colors=spine_color)
    ax.xaxis.label.set_color(spine_color)
    ax.yaxis.label.set_color(spine_color)
    ax.title.set_color(spine_color)
    for spine in ax.spines.values():
        spine.set_color(spine_color)
    ax.grid(True, axis=grid_axis, color="#c7c7c7", alpha=0.45)
    return ax


def style_legend(legend):
    """Apply high-contrast legend styling and return the legend."""

    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("#bbbbbb")
    legend.get_frame().set_alpha(1.0)
    for text in legend.get_texts():
        text.set_color("#222222")
    return legend


def legend(ax, **kwargs):
    """Create and style an axes legend."""

    return style_legend(ax.legend(**kwargs))
