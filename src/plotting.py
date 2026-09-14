"""Shared plotting style and helpers for the manuscript figures.

Every figure is written as PNG (600 dpi), PDF and SVG, and the numbers behind
it are written to a ``source_data`` folder so that each panel can be verified
without re-running the analysis.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd


FIGURE_WIDTH_IN = 6.7          # single-column width of Applied Sciences
FIGURE_HEIGHT_IN = 8.3         # height of the workflow figure
PNG_DPI = 600

COLORS = {
    "ink": "#1A1A1A",
    "muted": "#5C6B73",
    "grid": "#C9D1D6",
    "blue": "#0072B2",
    "blue_fill": "#E7F1F8",
    "green": "#009E73",
    "green_fill": "#E6F5F0",
    "orange": "#E69F00",
    "orange_fill": "#FFF4DD",
    "red": "#D55E00",
    "red_fill": "#FCE9DF",
    "purple": "#7B61A8",
    "purple_fill": "#F0EBF8",
    "grey": "#7F7F7F",
    "grey_fill": "#EFF2F4",
    "water": "#2455D6",
    "pressure": "#D1495B",
    "rainfall": "#4C9BD6",
    "rainfall_fill": "#A8D5F0",
    "threshold": "#8E8E8E",
    "highlight": "#F0C808",
}

CAUSE_COLORS = {
    "residual": COLORS["red"],
    "rainfall": COLORS["blue"],
    "pressure": COLORS["green"],
    "mixed": COLORS["purple"],
    "borderline": COLORS["orange"],
}


def apply_style() -> None:
    """Apply a consistent, print-safe Matplotlib style."""
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9.5,
            "axes.titleweight": "bold",
            "axes.edgecolor": COLORS["muted"],
            "axes.linewidth": 0.8,
            "axes.labelcolor": COLORS["ink"],
            "xtick.color": COLORS["ink"],
            "ytick.color": COLORS["ink"],
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "legend.frameon": True,
            "legend.framealpha": 0.95,
            "legend.edgecolor": COLORS["grid"],
            "grid.color": COLORS["grid"],
            "grid.linewidth": 0.6,
            "grid.linestyle": ":",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.unicode_minus": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig, directory: Path, stem: str) -> list[Path]:
    """Save one figure as PNG, PDF and SVG and return the written paths."""
    directory.mkdir(parents=True, exist_ok=True)
    written = []
    for suffix, kwargs in [
        (".png", {"dpi": PNG_DPI}),
        (".pdf", {}),
        (".svg", {}),
    ]:
        path = directory / f"{stem}{suffix}"
        fig.savefig(path, bbox_inches="tight", pad_inches=0.03, **kwargs)
        written.append(path)
    plt.close(fig)
    return written


def write_source_data(directory: Path, name: str, frame: pd.DataFrame) -> Path:
    """Write the numeric source data for one panel or figure."""
    target_dir = directory / "source_data"
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / f"{name}.csv"
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def add_panel_label(
    ax, label: str, *, x: float = 0.015, y: float = 0.975, fontsize: float = 9.5
) -> None:
    """Place a bold (a)/(b)/... label inside an axes."""
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=fontsize,
        fontweight="bold",
        color=COLORS["ink"],
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.85),
        zorder=10,
    )
