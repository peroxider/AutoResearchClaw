"""Shared publication style for every Paper-5 figure."""
from __future__ import annotations

import matplotlib as mpl

NAVY = "#18324A"
BLUE = "#4477AA"
CYAN = "#66CCEE"
GREEN = "#228833"
RED = "#CC6677"
MAGENTA = "#AA3377"
GREY = "#6B7280"
PALETTE = [BLUE, CYAN, GREEN, RED, MAGENTA]


def apply_style() -> None:
    mpl.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.labelsize": 9,
        "axes.edgecolor": NAVY, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": NAVY, "ytick.color": NAVY, "text.color": NAVY,
        "legend.frameon": False, "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "savefig.dpi": 300,
        "svg.fonttype": "none", "pdf.fonttype": 42,
    })
