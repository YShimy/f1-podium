"""Shared matplotlib styling for the notebook's charts.

Colours are the first slots of a validated categorical palette, always used
in this fixed order so a series keeps its colour across charts.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator, PercentFormatter

INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRIDLINE = "#e4e3df"
SURFACE = "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # blue, orange, aqua, yellow


def new_figure(width: float = 10, height: float = 4.2):
    """A figure and axes on the chart surface colour."""
    fig, ax = plt.subplots(figsize=(width, height), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    return fig, ax


def style_axes(ax, title: str, xlabel: str | None = None, ylabel: str | None = None,
               percent: bool = False) -> None:
    """Left-aligned title, recessive y-grid, no top/right/left spines, optional % axis."""
    ax.set_title(title, loc="left", color=INK, fontsize=12, pad=14)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK_MUTED, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK_MUTED, fontsize=9)
    if percent:
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=None))
    ax.grid(axis="y", color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(INK_MUTED)
    ax.tick_params(colors=INK_MUTED, length=0, labelsize=9)


def plot_lines(ax, frame: pd.DataFrame, colors: list[str] | None = None, marker: bool = False,
               linestyles: list[str] | None = None, linewidths: list[float] | None = None) -> None:
    """One line per column of `frame` (index = x); a legend is drawn for 2+ series.

    `linestyles` / `linewidths` give a secondary encoding where series may overlap
    (e.g. a dashed line drawn over a thicker solid one).
    """
    n = frame.shape[1]
    colors = colors or SERIES[:n]
    linestyles = linestyles or ["-"] * n
    linewidths = linewidths or [2] * n
    for col, color, ls, lw in zip(frame.columns, colors, linestyles, linewidths):
        series = frame[col].astype(float)
        ax.plot(np.asarray(series.index, dtype=float), series.to_numpy(), color=color, linewidth=lw,
                linestyle=ls, label=col, zorder=3, marker="o" if marker else None, markersize=4.5,
                markeredgecolor=SURFACE, markeredgewidth=1)
    if n >= 2:
        ax.legend(frameon=False, fontsize=9, labelcolor=INK_MUTED, loc="best")


def mark_splits(ax, train_end_year: int, test_start_year: int) -> None:
    """Dashed lines at the train/validation and validation/test boundaries, labelled at the top."""
    val_start, val_end = train_end_year + 0.5, test_start_year - 0.5
    for x in (val_start, val_end):
        ax.axvline(x, color=INK_MUTED, linewidth=1, linestyle="--", zorder=1)
    top = ax.get_xaxis_transform()  # x in data units, y in axes fraction
    ax.text(val_start - 0.6, 1.0, "train", transform=top, ha="right", va="bottom", color=INK_MUTED, fontsize=8)
    ax.text((val_start + val_end) / 2, 1.0, "val", transform=top, ha="center", va="bottom", color=INK_MUTED, fontsize=8)
    ax.text(val_end + 0.6, 1.0, "test", transform=top, ha="left", va="bottom", color=INK_MUTED, fontsize=8)


def rate_dot_plot(ax, order: list[str], series: dict[str, pd.DataFrame], max_label: int = 30) -> None:
    """Horizontal dot plot with 95% interval bars.

    `order` lists the row labels from top to bottom. `series` maps a legend label to a
    frame indexed by those labels with columns rate, ci_low and ci_high. Labels missing
    from a frame are skipped. Series are drawn side by side within each row.
    """
    n = len(series)
    step = 0.3
    ypos = {label: i for i, label in enumerate(order)}
    for k, (name, frame) in enumerate(series.items()):
        offset = (k - (n - 1) / 2) * step
        color = SERIES[k]
        present = [c for c in order if c in frame.index]
        y = np.array([ypos[c] + offset for c in present])
        rate = frame.loc[present, "rate"].to_numpy(float)
        lo = frame.loc[present, "ci_low"].to_numpy(float)
        hi = frame.loc[present, "ci_high"].to_numpy(float)
        ax.hlines(y, lo, hi, color=color, linewidth=2, alpha=0.55, zorder=2)
        ax.plot(rate, y, "o", color=color, markersize=6, markeredgecolor=SURFACE,
                markeredgewidth=1, label=name, zorder=3)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([c if len(c) <= max_label else c[: max_label - 1] + "…" for c in order])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=None))
    if n >= 2:  # below the x-axis label, so it never covers a row or the title
        ax.legend(frameon=False, fontsize=9, labelcolor=INK_MUTED, loc="upper center",
                  bbox_to_anchor=(0.5, -0.17), ncol=n)


def identity_scatter(ax, x: pd.Series, y: pd.Series, label_top: int = 3, max_label: int = 22) -> None:
    """Scatter of y against x (same index) with a y = x line; labels the points farthest from it.

    Both axes use the same zoomed range (the data range plus a margin, in 5% steps) so the
    y = x line stays a true diagonal.
    """
    lo = max(0.0, np.floor((min(x.min(), y.min()) - 0.03) * 20) / 20)
    hi = min(1.0, np.ceil((max(x.max(), y.max()) + 0.03) * 20) / 20)
    ax.plot([lo, hi], [lo, hi], color=INK_MUTED, linewidth=1, linestyle="--", zorder=1)
    ax.scatter(x, y, s=34, color=SERIES[0], edgecolor=SURFACE, linewidth=1, zorder=3)
    far = (y - x).abs().sort_values(ascending=False).head(label_top).index
    # Different offsets per rank so neighbouring labels do not sit on top of each other.
    placements = [((6, 5), "left"), ((6, -12), "left"), ((6, 14), "left")]
    for rank, name in enumerate(far):
        label = str(name) if len(str(name)) <= max_label else str(name)[: max_label - 1] + "…"
        offset, ha = placements[rank % len(placements)]
        ax.annotate(label, (x[name], y[name]), xytext=offset, textcoords="offset points",
                    ha=ha, color=INK_MUTED, fontsize=8)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(MultipleLocator(0.1))
        axis.set_major_formatter(PercentFormatter(1.0, decimals=0))
