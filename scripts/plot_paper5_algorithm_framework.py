"""Draw the grouped, journal-style TRACE-Guard algorithm framework."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from paper5_plot_style import GREEN, MAGENTA, NAVY, apply_style

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/paper5_trace_guard/full_run/deliverables"
apply_style()

fig, ax = plt.subplots(figsize=(16.2, 10.2))
ax.set_xlim(0, 16.2); ax.set_ylim(0, 10.2); ax.axis("off")

MODULES = {
    "data": {"edge": "#4477AA", "fill": "#EAF1F8", "title": "MODULE I  ·  DATA CONTRACT & STATE CONSTRUCTION"},
    "inference": {"edge": "#B7791F", "fill": "#FFF2DD", "title": "MODULE II  ·  LLM INFERENCE & ROLE-SEPARATED VERIFICATION"},
    "decision": {"edge": MAGENTA, "fill": "#FAEDEF", "title": "MODULE III  ·  SELECTIVE DECISION & IMMUTABLE EVIDENCE"},
    "evaluation": {"edge": GREEN, "fill": "#E8F4EE", "title": "MODULE IV  ·  DELAYED-OUTCOME EVALUATION"},
}


def group(x, y, w, h, key):
    m = MODULES[key]
    frame = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.025,rounding_size=0.08",
        facecolor="white", edgecolor=m["edge"], linewidth=1.7,
        linestyle=(0, (6, 4)), zorder=0,
    )
    ax.add_patch(frame)
    ax.text(x + .22, y + h - .18, m["title"], ha="left", va="top",
            fontsize=10.6, weight="bold", color=m["edge"],
            bbox=dict(facecolor="white", edgecolor="none", pad=1.5), zorder=5)


def box(x, y, w, h, title, lines, key, *, body_size=9.2, title_size=10.2):
    m = MODULES[key]
    patch = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.10",
        facecolor=m["fill"], edgecolor=m["edge"], linewidth=1.55, zorder=2,
    )
    ax.add_patch(patch)
    ax.text(x+w/2, y+h*.72, title, ha="center", va="center",
            fontsize=title_size, weight="bold", color=NAVY, linespacing=1.02, zorder=3)
    ax.text(x+w/2, y+h*.31, lines, ha="center", va="center",
            fontsize=body_size, color="#263746", linespacing=1.18, zorder=3)
    return (x, y, w, h)


def arrow(start, end, label="", *, bend=0.0, dashed=False, color=NAVY):
    p = FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=13, linewidth=1.45,
        linestyle="--" if dashed else "-", color=color,
        connectionstyle=f"arc3,rad={bend}", zorder=4,
    )
    ax.add_patch(p)
    if label:
        ax.text((start[0]+end[0])/2, (start[1]+end[1])/2+.10, label,
                ha="center", va="bottom", fontsize=8.1, color="#51697F",
                bbox=dict(facecolor="white", edgecolor="none", alpha=.94, pad=1.0), zorder=6)


ax.text(8.1, 9.92, "TRACE-Guard Clinical: grouped outcome-blind selective LLM inference",
        ha="center", va="top", fontsize=16.5, weight="bold", color=NAVY)
ax.text(8.1, 9.55,
        "The dashed module boundaries separate data preparation, model inference, decision control, and outcome evaluation",
        ha="center", va="top", fontsize=9.5, color="#51697F")

# Large dashed module boundaries.
group(.25, 6.75, 10.35, 2.45, "data")
group(.25, 3.85, 10.35, 2.45, "inference")
group(2.25, 1.15, 8.35, 2.15, "decision")
group(11.0, 1.15, 4.95, 8.05, "evaluation")

# Module I: deterministic state construction.
b1 = box(.55, 7.12, 2.15, 1.48, "Source record  $X_i$",
         "86 rows · 89 fields\nbaseline + post-index\nidentifiers + hidden\noutcome", "data", body_size=8.8)
b2 = box(3.05, 7.12, 2.15, 1.48, "Frozen data\ncontract",
         "allow-list $A$ · deny-list $P$\nindex time = admission\nno outcome derivatives", "data")
b3 = box(5.55, 7.12, 2.15, 1.48, "State-card\ncompiler  $C_A$",
         "$S_i=C_A(X_{i,A-P})$\ncanonical JSON\nexplicit missingness\nkey assertions", "data", body_size=8.8)
b4 = box(8.05, 7.12, 2.15, 1.48, "Pre-request\ncommitment",
         "$h_i=H(S_i)$\ncondition + row index\noutcome not joined", "data")
arrow((2.70, 7.86), (3.05, 7.86)); arrow((5.20, 7.86), (5.55, 7.86)); arrow((7.70, 7.86), (8.05, 7.86))

# Module II: generator/critic computation.
b5 = box(.55, 4.22, 2.15, 1.48, "Condition router  $T_c$",
         "Direct · Retrieval\nNoCritic · NoAbstention\nFullAuditedAgent", "inference", body_size=9.0)
b6 = box(3.05, 4.22, 2.15, 1.48, "MiniMax-M3\ngenerator  $G$",
         "$(P,A^G,R)=G(S,T_c)$\nprobability · abstain\nrationale\nzero local training", "inference", body_size=8.8)
b7 = box(5.55, 4.22, 2.15, 1.48, "Schema & retry\noperator",
         r"$P\in[0,1]$ or $P=\bot$" "\nvalidate condition + JSON\n≤4 attempts · backoff", "inference")
b8 = box(8.05, 4.22, 2.15, 1.48, "Prompted critic  $Q$",
         "$(U,F,Z)=Q(S,P,R)$\naccept · force abstain\ncritique\nfull condition only", "inference", body_size=8.8)
arrow((9.12, 7.12), (1.62, 5.70), "committed state + condition", bend=.08)
arrow((2.70, 4.96), (3.05, 4.96)); arrow((5.20, 4.96), (5.55, 4.96)); arrow((7.70, 4.96), (8.05, 4.96), "full arm")

# Module III: release and evidence commitment.
b9 = box(2.60, 1.52, 3.45, 1.32, "Selective release rule",
         r"$D=1(P\ne\bot)(1-A^G)(1-F)U$" "\nrelease iff $D=1$ · final abstention $A=1-D$", "decision", body_size=9.5)
b10 = box(6.55, 1.52, 3.70, 1.32, "Write-ahead inference event  $E_{ic}$",
          "hashes · score · verdict · latency · error\ncommit $H(E_{ic})$ before outcome access", "decision", body_size=9.4)
arrow((6.62, 4.22), (4.32, 2.84), "generator path", bend=.08)
arrow((9.12, 4.22), (5.25, 2.84), "critic gate", bend=-.06)
arrow((6.05, 2.18), (6.55, 2.18))

# Module IV: outcome is accessible only here.
b11 = box(11.42, 7.10, 4.10, 1.52, "Held-out outcome  $Y_i$",
          "post-anticoagulation bleeding\nSPSS: 1 = yes · 2 = no\n16/86 recorded events", "evaluation", body_size=9.5)
b12 = box(11.42, 4.52, 4.10, 1.52, "Post-commit outcome join",
          "attach $Y_i$ only after $H(E_{ic})$\nverify 430 unique $(c,i)$ keys\nreconstruct pre-outcome hashes", "evaluation", body_size=9.5)
b13 = box(11.42, 1.67, 4.10, 1.85, "Multi-objective evaluation",
          "$\kappa_s$ score coverage · $\kappa_r$ release coverage\nAUROC/AUPRC + 2,000-bootstrap CI\nBrier · ECE · latency · errors · critic calls\n$R_{sel}=E[D\ell]/E[D]$", "evaluation", body_size=9.2)
arrow((13.47, 7.10), (13.47, 6.04)); arrow((13.47, 4.52), (13.47, 3.52))
arrow((10.25, 2.18), (11.42, 5.20), "immutable event", bend=-.13)

# Strong visual information-flow boundary.
ax.plot([10.80, 10.80], [1.05, 9.25], color="#7C8B96", lw=1.45, ls=(0, (5, 4)), zorder=1)
ax.text(10.80, 6.45, "OUTCOME-ACCESS BOUNDARY", ha="center", va="center",
        rotation=90, fontsize=8.1, weight="bold", color="#657682",
        bbox=dict(facecolor="white", edgecolor="none", pad=1.2), zorder=6)

# Compact theorem strip uses the same dark ink and no competing module color.
ax.text(.35, .78, "FORMAL GUARANTEES", fontsize=10.3, weight="bold", color=NAVY)
ax.text(.35, .47,
        "Outcome-join invariance: $Y$ cannot alter a committed request   ·   "
        "Critic monotonicity: $D_1=D_0(1-F)U\leq D_0$   ·   "
        "Non-degeneracy: minimize $R_{sel}$ subject to $\kappa_r\geq\kappa_{min}>0$",
        fontsize=8.8, color=NAVY)
ax.text(8.1, .12, "Solid arrows show computation; dashed frames show functional modules; the vertical boundary blocks outcome access during inference.",
        ha="center", fontsize=8.2, color="#51697F")

for suffix in ("png", "pdf", "svg"):
    fig.savefig(OUT / f"trace_guard_algorithm_framework.{suffix}", dpi=300, bbox_inches="tight")
plt.close(fig)
print(OUT / "trace_guard_algorithm_framework.png")
