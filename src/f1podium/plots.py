"""Shared matplotlib styling for the notebook's charts.

Colours are the first slots of a validated categorical palette, always used
in this fixed order so a series keeps its colour across charts.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

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
