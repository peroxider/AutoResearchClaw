"""Create and insert the abstract input-method-output overview for Paper 5."""
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

from paper5_plot_style import BLUE, GREEN, MAGENTA, NAVY, RED, apply_style

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/paper5_trace_guard/full_run/deliverables"
apply_style()
fig, ax = plt.subplots(figsize=(15.5, 6.8))
ax.set_xlim(0, 15.5); ax.set_ylim(0, 6.8); ax.axis("off")


def frame(x, y, w, h, color, title):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=.03,rounding_size=.12",
                               facecolor="white", edgecolor=color, lw=1.7,
                               linestyle=(0, (6, 4)), zorder=0))
    ax.text(x+.22, y+h-.18, title, ha="left", va="top", color=color,
            fontsize=11, weight="bold", bbox=dict(facecolor="white", edgecolor="none", pad=1.3))


def card(x, y, w, h, title, subtitle, edge, fill, title_size=10.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=.035,rounding_size=.10",
                               facecolor=fill, edgecolor=edge, lw=1.5, zorder=2))
    ax.text(x+w/2, y+h*.64, title, ha="center", va="center", color=NAVY,
            fontsize=title_size, weight="bold", zorder=3)
    ax.text(x+w/2, y+h*.29, subtitle, ha="center", va="center", color="#40576B",
            fontsize=8.8, linespacing=1.12, zorder=3)


def arrow(a, b, color=NAVY, lw=1.8):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=15,
                                color=color, lw=lw, zorder=4))


ax.text(7.75, 6.50, "TRACE-Guard Clinical: study-level conceptual architecture",
        ha="center", va="top", fontsize=16.5, weight="bold", color=NAVY)
ax.text(7.75, 6.14, "From outcome-blind clinical inputs to selective, verifiable LLM outputs",
        ha="center", va="top", fontsize=9.7, color="#51697F")

# Three deliberately abstract macro layers.
frame(.25, .62, 3.65, 5.05, BLUE, "INPUT DATA")
frame(4.35, .62, 6.65, 5.05, MAGENTA, "LLM APPLICATION METHOD")
frame(11.45, .62, 3.80, 5.05, GREEN, "OUTPUT & EVALUATION")

# Input: visual table, baseline filter, and locked outcome.
card(.58, 3.78, 2.98, 1.24, "Structured clinical records", "", BLUE, "#EAF1F8")
# table glyph
gx, gy = .82, 4.04
for r in range(3):
    for c in range(4):
        ax.add_patch(Rectangle((gx+c*.13, gy+r*.11), .105, .085, facecolor="white",
                               edgecolor=BLUE, lw=.55, zorder=4))
ax.text(2.32, 4.19, "86 encounters\n89 source fields", ha="center", va="center",
        color="#40576B", fontsize=8.8, linespacing=1.1, zorder=5)
card(.58, 2.28, 2.98, 1.12, "Baseline-only projection", "identifiers removed\npost-index fields excluded", BLUE, "#EAF1F8")
arrow((2.07, 3.78), (2.07, 3.40), BLUE, 1.4)
card(.88, 1.05, 2.38, .82, "", "", RED, "#FAEDEF", 9.8)
# simple lock geometry, not an emoji
ax.add_patch(Rectangle((1.08, 1.27), .28, .23, facecolor="none", edgecolor=RED, lw=1.25, zorder=5))
ax.add_patch(plt.matplotlib.patches.Arc((1.22, 1.50), .22, .25, theta1=0, theta2=180,
                                       color=RED, lw=1.25, zorder=5))
ax.text(2.18, 1.54, "Outcome locked", ha="center", va="center",
        color=NAVY, fontsize=9.5, weight="bold", zorder=5)
ax.text(2.18, 1.27, "join after commitment", ha="center", va="center",
        color="#40576B", fontsize=8.4, zorder=5)

# Innovation wrapper: three highlighted innovations around one model core.
ax.add_patch(FancyBboxPatch((4.72, 1.00), 5.90, 4.15,
                            boxstyle="round,pad=.04,rounding_size=.20",
                            facecolor="#FCF4F8", edgecolor=MAGENTA, lw=2.2, zorder=1))
ax.text(7.67, 4.88, "TRACE-GUARD INNOVATION WRAPPER", ha="center", va="center",
        fontsize=11.2, color=MAGENTA, weight="bold", zorder=5)

# Central model node.
ax.add_patch(Circle((7.67, 3.02), .87, facecolor="#FFF2DD", edgecolor="#B7791F", lw=1.8, zorder=3))
ax.text(7.67, 3.16, "MiniMax-M3", ha="center", va="center", fontsize=11.0,
        weight="bold", color=NAVY, zorder=5)
ax.text(7.67, 2.78, "generator + critic", ha="center", va="center",
        fontsize=8.8, color="#40576B", zorder=5)

innovations = [
    (5.04, 3.88, "01", "Outcome-blind\ncommitment"),
    (8.78, 3.88, "02", "Critic-gated\nrelease"),
    (6.91, 1.34, "03", "Coverage +\nevidence trace"),
]
for x, y, num, label in innovations:
    ax.add_patch(FancyBboxPatch((x, y), 1.78, .82, boxstyle="round,pad=.025,rounding_size=.10",
                               facecolor="white", edgecolor=MAGENTA, lw=1.45, zorder=4))
    ax.add_patch(Circle((x+.24, y+.41), .17, facecolor=MAGENTA, edgecolor=MAGENTA, zorder=5))
    ax.text(x+.24, y+.41, num, ha="center", va="center", fontsize=7.2,
            color="white", weight="bold", zorder=6)
    ax.text(x+1.12, y+.41, label, ha="center", va="center", fontsize=8.0,
            color=NAVY, weight="bold", linespacing=1.06, zorder=6)

arrow((6.74, 3.47), (6.87, 3.33), MAGENTA, 1.25)
arrow((8.60, 3.33), (8.73, 3.47), MAGENTA, 1.25)
arrow((7.67, 2.15), (7.80, 2.16), MAGENTA, 1.25)
arrow((3.90, 3.02), (4.72, 3.02), NAVY, 2.0)

# Outputs: branching selective response and evaluation bundle.
card(11.82, 3.85, 3.06, 1.17, "Selective response", "risk score   |   abstain", GREEN, "#E8F4EE")
ax.add_patch(Circle((12.35, 4.16), .12, facecolor=GREEN, edgecolor=GREEN, zorder=5))
ax.add_patch(Rectangle((14.19, 4.05), .23, .23, facecolor="white", edgecolor=GREEN, lw=1.1, zorder=5))
card(11.82, 2.35, 3.06, 1.05, "Verifiable evidence", "committed hashes + decision trace", GREEN, "#E8F4EE")
card(11.82, 1.05, 3.06, .85, "Multi-dimensional profile", "performance · coverage · latency", GREEN, "#E8F4EE", 9.7)
arrow((11.00, 3.02), (11.82, 4.22), NAVY, 1.8)
arrow((11.00, 3.02), (11.82, 2.87), NAVY, 1.8)
arrow((13.35, 2.35), (13.35, 1.90), GREEN, 1.4)

# Innovation emphasis without prose paragraphs.
ax.text(5.98, .77, "NOVELTY", ha="center", va="center", color="white", fontsize=8.2,
        weight="bold", bbox=dict(boxstyle="round,pad=.32", facecolor=MAGENTA, edgecolor=MAGENTA))
ax.text(6.70, .77, "verifiable inference before outcome access",
        ha="left", va="center", fontsize=8.3, color=MAGENTA, weight="bold")

for suffix in ("png", "pdf", "svg"):
    fig.savefig(OUT / f"figure_conceptual_overview.{suffix}", dpi=300, bbox_inches="tight")
plt.close(fig)

# Insert immediately after Introduction and normalize the now five figure references.
md = OUT / "paper_final_corrected.md"
text = md.read_text(encoding="utf-8")
insert = '''Figure 1 provides a study-level view of the information flow. It separates the supplied clinical data, the TRACE-Guard LLM application wrapper, and the released analytical outputs. The three highlighted innovations are outcome-blind state commitment, critic-gated selective release, and joint reporting of coverage with a verifiable evidence trace.

![Study-level conceptual architecture of TRACE-Guard Clinical, showing outcome-blind input data, the innovation wrapper around the language model, and selective auditable outputs.](figure_conceptual_overview.png)

'''
intro_end = text.index("## Related Work and Methodological Positioning")
inserted_now = "figure_conceptual_overview.png" not in text
if inserted_now:
    text = text[:intro_end] + insert + text[intro_end:]

# Use temporary tokens to avoid cascading replacements.
mapping = {
    "Figure 1 summarizes outcome imbalance": "@@DATA_FIG@@ summarizes outcome imbalance",
    "Figure 2 formalizes the complete data and decision path": "@@ALGO_FIG@@ formalizes the complete data and decision path",
    "Figure 3A presents discrimination": "@@RESULT_FIG_A@@ presents discrimination",
    "(Figure 3B)": "(@@RESULT_FIG_B@@)",
    "Figure 3C separates": "@@RESULT_FIG_C@@ separates",
    "(Figure 3D)": "(@@RESULT_FIG_D@@)",
    "Figure 4A exposes": "@@CAL_FIG_A@@ exposes",
    "(Figure 4B)": "(@@CAL_FIG_B@@)",
}
if inserted_now:
    for old, token in mapping.items(): text = text.replace(old, token)
    for token, new in {
        "@@DATA_FIG@@": "Figure 2", "@@ALGO_FIG@@": "Figure 3",
        "@@RESULT_FIG_A@@": "Figure 4A", "@@RESULT_FIG_B@@": "Figure 4B",
        "@@RESULT_FIG_C@@": "Figure 4C", "@@RESULT_FIG_D@@": "Figure 4D",
        "@@CAL_FIG_A@@": "Figure 5A", "@@CAL_FIG_B@@": "Figure 5B",
    }.items(): text = text.replace(token, new)

# Semantic normalization makes repeated rendering idempotent.
text = re.sub(r"Calibration-related losses were uniformly large \(Figure \d+B\)\.",
              "Calibration-related losses were uniformly large (Figure 4B).", text)
text = re.sub(r"Figure \d+A exposes the source of this loss", "Figure 5A exposes the source of this loss", text)
text = re.sub(r"The empirical calibration curves \(Figure \d+B\)",
              "The empirical calibration curves (Figure 5B)", text)
md.write_text(text, encoding="utf-8")
print(OUT / "figure_conceptual_overview.png")
