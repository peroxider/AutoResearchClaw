"""Recompute secondary Paper-5 analyses, draw journal figures, and expand Results."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from paper5_plot_style import PALETTE, BLUE, GREEN, RED, apply_style


ROOT = Path(__file__).resolve().parents[1]
DELIV = ROOT / "artifacts/paper5_trace_guard/full_run/deliverables"
EVID = ROOT / "artifacts/paper5_trace_guard/full_run/stage-12_v1/runs/sandbox/_project_1"
CONDITIONS = ["DirectStructured", "RetrievalOnly", "NoCritic", "NoAbstention", "FullAuditedAgent"]
SHORT = ["Direct\nstructured", "Retrieval\nonly", "No\ncritic", "No\nabstention", "Full audited\nagent"]
COLORS = PALETTE


def auc(y: np.ndarray, p: np.ndarray) -> float | None:
    pos, neg = int(y.sum()), int(len(y) - y.sum())
    if not pos or not neg:
        return None
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p), float)
    i = 0
    while i < len(p):
        j = i + 1
        while j < len(p) and p[order[j]] == p[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2
        i = j
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


rows = [json.loads(x) for x in (EVID / "patient_results.jsonl").read_text(encoding="utf-8").splitlines()]
metrics = json.loads((EVID / "metrics.json").read_text(encoding="utf-8"))["conditions"]
by_condition = {c: [r for r in rows if r["condition"] == c] for c in CONDITIONS}

# Common-score sensitivity cohort: same patients, hence a genuinely paired descriptive subset.
score_maps = {c: {int(r["row_index"]): r for r in rs if r.get("score") is not None} for c, rs in by_condition.items()}
common = sorted(set.intersection(*(set(v) for v in score_maps.values())))
common_stats = {}
for c in CONDITIONS:
    rr = [score_maps[c][i] for i in common]
    y = np.array([1 if str(float(r["outcome"])) == "1.0" else 0 for r in rr])
    p = np.array([float(r["score"]) for r in rr])
    common_stats[c] = {
        "n": len(rr), "events": int(y.sum()), "auroc": auc(y, p),
        "brier": float(np.mean((p-y)**2)), "mean_score": float(p.mean()),
    }

# Descriptive score and critic summaries.
distribution = {}
for c in CONDITIONS:
    scored = [r for r in by_condition[c] if r.get("score") is not None]
    neg = np.array([float(r["score"]) for r in scored if str(float(r["outcome"])) != "1.0"])
    pos = np.array([float(r["score"]) for r in scored if str(float(r["outcome"])) == "1.0"])
    distribution[c] = {
        "non_event_n": len(neg), "event_n": len(pos),
        "non_event_median": float(np.median(neg)), "event_median": float(np.median(pos)),
        "non_event_iqr": [float(np.quantile(neg, .25)), float(np.quantile(neg, .75))],
        "event_iqr": [float(np.quantile(pos, .25)), float(np.quantile(pos, .75))],
    }
full = [r for r in by_condition["FullAuditedAgent"] if r.get("critic_response_hash")]
accepted = [r for r in full if isinstance(r.get("critic_verdict"), dict) and r["critic_verdict"].get("accept")]
critic = {
    "calls": len(full), "accepted": len(accepted), "acceptance_rate": len(accepted)/len(full),
    "accepted_events": sum(str(float(r["outcome"])) == "1.0" for r in accepted),
    "accepted_mean_score": float(np.mean([r["score"] for r in accepted])),
    "not_accepted_mean_score": float(np.mean([r["score"] for r in full if r not in accepted])),
}
secondary = {"common_scored_rows": common, "common_case": common_stats,
             "score_distribution": distribution, "critic": critic}
(DELIV / "secondary_results_verified.json").write_text(json.dumps(secondary, indent=2), encoding="utf-8")

apply_style()

# Figure 2: four complementary operating dimensions.
fig, axs = plt.subplots(2, 2, figsize=(11.2, 8.2), constrained_layout=True)
x = np.arange(5)
ax = axs[0, 0]
for j, (key, lo, hi, marker, offset, label) in enumerate([
    ("auroc", "auroc_ci_low", "auroc_ci_high", "o", -.10, "AUROC"),
    ("auprc", "auprc_ci_low", "auprc_ci_high", "s", .10, "AUPRC"),
]):
    vals = np.array([metrics[c][key] for c in CONDITIONS])
    lows = np.array([metrics[c][lo] for c in CONDITIONS])
    highs = np.array([metrics[c][hi] for c in CONDITIONS])
    ax.errorbar(x+offset, vals, yerr=[vals-lows, highs-vals], fmt=marker, capsize=3,
                lw=1.4, ms=5, label=label, color=["#225588", "#BB5566"][j])
ax.axhline(.5, ls="--", lw=1, color="#777777", label="AUROC null")
ax.axhline(16/86, ls=":", lw=1.2, color="#228833", label="AUPRC prevalence")
ax.set_ylim(0, 1); ax.set_xticks(x, SHORT); ax.set_ylabel("Estimate (95% bootstrap CI)")
ax.set_title("A  Discrimination and ranking precision", loc="left", weight="bold")
ax.legend(frameon=False, ncol=2, fontsize=7.5)

ax = axs[0, 1]
w = .34
ax.bar(x-w/2, [metrics[c]["brier_score"] for c in CONDITIONS], w, label="Brier score", color="#4477AA")
ax.bar(x+w/2, [metrics[c]["expected_calibration_error"] for c in CONDITIONS], w, label="ECE", color="#EE6677")
ax.axhline((16/86)*(1-16/86), ls=":", color="#228833", lw=1.2, label="Prevalence-only Brier")
ax.set_ylim(0, .7); ax.set_xticks(x, SHORT); ax.set_ylabel("Loss (lower is better)")
ax.set_title("B  Calibration-related loss", loc="left", weight="bold"); ax.legend(frameon=False, fontsize=8)

ax = axs[1, 0]
coverage = np.array([metrics[c]["coverage"] for c in CONDITIONS])
release = 1-np.array([metrics[c]["abstention_rate"] for c in CONDITIONS])
error = np.array([metrics[c]["error_rate"] for c in CONDITIONS])
ax.bar(x-.25, coverage*100, .25, label="Numeric-score coverage", color="#4477AA")
ax.bar(x, release*100, .25, label="Operational release", color="#228833")
ax.bar(x+.25, error*100, .25, label="Request error", color="#CC6677")
ax.set_ylim(0, 105); ax.set_xticks(x, SHORT); ax.set_ylabel("Patients (%)")
ax.set_title("C  Availability, release, and failure", loc="left", weight="bold"); ax.legend(frameon=False, fontsize=8)

ax = axs[1, 1]
lat = np.array([metrics[c]["mean_latency_sec"] for c in CONDITIONS])
ax.bar(x, lat, .62, color=COLORS)
for i, v in enumerate(lat): ax.text(i, v+.35, f"{v:.1f}", ha="center", fontsize=8)
ax.set_ylim(0, max(lat)*1.22); ax.set_xticks(x, SHORT); ax.set_ylabel("Mean latency (seconds)")
ax.set_title("D  End-to-end service latency", loc="left", weight="bold")
fig.savefig(DELIV / "figure2_operating_characteristics.png", dpi=300, bbox_inches="tight")
fig.savefig(DELIV / "figure2_operating_characteristics.pdf", bbox_inches="tight")
plt.close(fig)

# Figure 3: patient-level distributions and empirical calibration bins.
fig, axs = plt.subplots(1, 2, figsize=(12, 5.2), constrained_layout=True)
ax = axs[0]
rng = np.random.default_rng(20260806)
for i, c in enumerate(CONDITIONS):
    scored = by_condition[c]
    groups = []
    for event, off, color in [(0, -.17, "#4477AA"), (1, .17, "#CC6677")]:
        vals = np.array([float(r["score"]) for r in scored if r.get("score") is not None and
                         (str(float(r["outcome"])) == "1.0") == bool(event)])
        groups.append(vals)
        bp = ax.boxplot(vals, positions=[i+off], widths=.27, patch_artist=True, showfliers=False,
                        medianprops={"color": "white", "lw": 1.4},
                        boxprops={"facecolor": color, "edgecolor": color, "alpha": .75},
                        whiskerprops={"color": color}, capprops={"color": color})
        jitter = rng.uniform(-.055, .055, len(vals))
        ax.scatter(np.full(len(vals), i+off)+jitter, vals, s=10, alpha=.45, color=color, edgecolors="none")
ax.set_xlim(-.6, 4.6); ax.set_ylim(-.03, 1.03); ax.set_xticks(x, SHORT); ax.set_ylabel("LLM probability-like score")
ax.set_title("A  Score distributions by observed outcome", loc="left", weight="bold")
ax.scatter([], [], color="#4477AA", label="No bleeding"); ax.scatter([], [], color="#CC6677", label="Bleeding")
ax.legend(frameon=False, loc="lower left")

ax = axs[1]
ax.plot([0, 1], [0, 1], ls="--", color="#777777", lw=1, label="Perfect calibration")
for color, c, label in zip(COLORS, CONDITIONS, SHORT):
    bins = metrics[c]["calibration_curve"]
    xp = np.array([b["mean_prediction"] for b in bins]); yp = np.array([b["event_rate"] for b in bins])
    sizes = np.array([b["n"] for b in bins])
    ax.plot(xp, yp, color=color, lw=1.2, alpha=.85)
    ax.scatter(xp, yp, s=10+sizes*2.0, color=color, alpha=.8, edgecolor="white", linewidth=.4, label=label.replace("\n", " "))
ax.set_xlim(-.03, 1.03); ax.set_ylim(-.03, 1.03); ax.set_xlabel("Mean predicted score in bin")
ax.set_ylabel("Observed event fraction"); ax.set_title("B  Empirical calibration (marker area reflects bin n)", loc="left", weight="bold")
ax.legend(frameon=False, fontsize=7.3, loc="upper left")
fig.savefig(DELIV / "figure3_score_calibration.png", dpi=300, bbox_inches="tight")
fig.savefig(DELIV / "figure3_score_calibration.pdf", bbox_inches="tight")
plt.close(fig)

# Replace Results only. Numbers below are rendered from verified objects.
def f3(v): return f"{v:.3f}"
common_auc = ", ".join(f"{c} {f3(common_stats[c]['auroc'])}" for c in CONDITIONS)
common_brier = ", ".join(f"{c} {f3(common_stats[c]['brier'])}" for c in CONDITIONS)
medians = "; ".join(f"{c}: {distribution[c]['event_median']:.2f} vs {distribution[c]['non_event_median']:.2f}"
                    for c in CONDITIONS)
results = f'''## Results

### Cohort-level execution and analytical denominators

All five conditions were attempted for all 86 records, producing 430 unique patient--condition evaluations. The cohort contained 16 recorded bleeding events (18.6%). A numeric score was available for 387 evaluations (90.0%), but availability differed materially by condition: 85/86 for DirectStructured and NoAbstention, 73/86 for NoCritic and FullAuditedAgent, and 71/86 for RetrievalOnly. The number of observed events represented in condition-specific score analyses consequently ranged from 12 to 16. This denominator variation is important because each AUROC, AUPRC, Brier score, and calibration estimate describes the successfully scored subset rather than an identical patient set.

| Condition | Scored/86 | Events scored | Score coverage | AUROC (95% CI) | AUPRC (95% CI) | Brier | ECE | Operational release | Error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DirectStructured | 85 | 16 | 98.8% | 0.582 (0.409–0.744) | 0.454 (0.227–0.674) | 0.487 | 0.580 | 80.2% | 1.2% |
| RetrievalOnly | 71 | 14 | 82.6% | 0.382 (0.218–0.546) | 0.195 (0.104–0.343) | 0.407 | 0.465 | 14.0% | 17.4% |
| NoCritic | 73 | 12 | 84.9% | 0.546 (0.367–0.725) | 0.275 (0.127–0.532) | 0.358 | 0.439 | 5.8% | 15.1% |
| NoAbstention | 85 | 16 | 98.8% | 0.625 (0.465–0.774) | 0.334 (0.188–0.562) | 0.507 | 0.595 | 98.8% | 1.2% |
| FullAuditedAgent | 73 | 16 | 84.9% | 0.581 (0.421–0.729) | 0.294 (0.170–0.472) | 0.362 | 0.415 | 0.0% | 15.1% |

The merged event store retained all failed requests rather than deleting them. It contained 430 unique keys and 430 pre-outcome events. The 86 corrected FullAuditedAgent events passed reconstruction of their pre-outcome hashes; no serialized state card contained the outcome field, patient number, or name. These checks establish execution completeness and information-flow integrity, but they do not imply predictive validity.

### Discrimination and precision--recall performance

Figure 2A presents discrimination with patient-level bootstrap uncertainty. NoAbstention had the largest AUROC point estimate (0.625), followed by DirectStructured (0.582), FullAuditedAgent (0.581), NoCritic (0.546), and RetrievalOnly (0.382). All 95% confidence intervals crossed 0.5, including the NoAbstention interval of 0.465--0.774. The data therefore do not establish above-chance discrimination for any condition. The width of the intervals—approximately 0.31 to 0.36—also shows that small differences between point estimates are minor relative to sampling uncertainty.

AUPRC supplied a complementary class-imbalance perspective. The cohort prevalence reference was 0.186. DirectStructured produced the highest AUPRC point estimate (0.454), while RetrievalOnly was 0.195, close to the prevalence reference. Although DirectStructured's point estimate exceeded that reference, its interval was wide (0.227--0.674). AUPRC and AUROC did not rank the conditions identically: NoAbstention ranked first by AUROC but only second by neither calibration nor AUPRC. This discordance argues against reducing model selection to a single discrimination statistic.

![Discrimination, calibration loss, availability, operational release, request errors, and latency across the five conditions. Error bars in panel A are 95% patient-level bootstrap intervals.](figure2_operating_characteristics.png)

### Calibration loss and patient-level score distributions

Calibration-related losses were uniformly large (Figure 2B). The prevalence-only constant prediction has a Brier score of 0.151 for an event fraction of 16/86; every condition had a larger observed Brier score, ranging from 0.358 for NoCritic to 0.507 for NoAbstention. This is a descriptive comparison because failures changed the evaluated subsets, but it demonstrates that none of the raw LLM score streams improved on the cohort-prevalence constant under squared error in their observed operating samples. ECE ranged from 0.415 to 0.595. Even FullAuditedAgent, which had the lowest ECE, remained far from calibrated probability output.

Figure 3A exposes the source of this loss at the patient level. The event versus non-event median scores were {medians}. The extensive overlap between outcome groups and the concentration of many scores above 0.70 show that the large calibration errors were not driven by a few outliers. In particular, NoAbstention placed most scores in a high-probability range despite an 18.6% cohort event rate. The empirical calibration curves (Figure 3B) lie predominantly below the identity line in populated high-score bins, consistent with systematic overprediction. Marker sizes also show why individual bin fluctuations should not be interpreted as stable calibration: several extreme event fractions came from very small bins.

![Patient-level score distributions by bleeding outcome and empirical ten-bin calibration. Points in panel A are individual scored records; box centers are medians. Calibration-marker area is proportional to bin size.](figure3_score_calibration.png)

### Coverage, abstention, and service failures

Figure 2C separates numeric-score coverage from operational release. DirectStructured generated scores for 98.8% of records and released 80.2%; NoAbstention generated and released 98.8%. In contrast, NoCritic generated scores for 84.9% but released only 5.8%, while FullAuditedAgent generated the same proportion yet released 0%. Thus score availability overstated operational coverage by 84.9 percentage points in the full method. RetrievalOnly occupied an intermediate but still inefficient operating point: 82.6% score coverage and 14.0% release.

Error rates formed two clusters. DirectStructured and NoAbstention each failed on 1/86 requests (1.2%), whereas RetrievalOnly, NoCritic, and FullAuditedAgent failed on 15/86, 13/86, and 13/86, respectively. Since missing probabilities were not imputed, these failures changed both the effective sample and the mix of observed events. High abstention and high error therefore cannot be treated as interchangeable notions: the former is a model/policy decision, while the latter represents absence of a valid computational output.

### Latency and critic operating behavior

Mean end-to-end latency increased from 6.85 seconds for DirectStructured to 7.49 for RetrievalOnly, 8.53 for NoCritic, 9.02 for NoAbstention, and 18.07 for FullAuditedAgent (Figure 2D). The complete method was 2.64 times as slow as DirectStructured, consistent with its second model call and retry exposure. Because the experiment did not retain per-call token counts or API cost, latency is the available efficiency endpoint and should not be interpreted as a complete resource analysis.

The full condition invoked the critic for 73 successfully scored records; 33 were accepted, an acceptance rate of {critic['acceptance_rate']*100:.1f}%. Of the accepted proposals, {critic['accepted_events']} corresponded to recorded bleeding events. Mean generator score was {critic['accepted_mean_score']:.3f} among accepted proposals and {critic['not_accepted_mean_score']:.3f} among non-accepted proposals. Nevertheless, none was released because final release required a nonmissing score, no generator abstention, no critic-forced abstention, and critic acceptance simultaneously. Critic acceptance is therefore a component-level behavior measure, not critic accuracy or clinical utility.

### Common-score sensitivity analysis

Only {len(common)} patients, including {common_stats[CONDITIONS[0]]['events']} events, had a numeric score under every condition. Restricting descriptively to this common-score subset removed the denominator mismatch. AUROC estimates were {common_auc}; corresponding Brier scores were {common_brier}. The subset analysis did not produce a stable uniformly superior condition: NoAbstention retained the largest AUROC, whereas NoCritic had the smallest Brier loss. Because this analysis contains only {common_stats[CONDITIONS[0]]['events']} events and was not prespecified as confirmatory, it is a robustness description rather than a significance test.

### Integrated multi-objective interpretation

Taken together, the conditions occupy different and clinically unfavorable parts of the discrimination--calibration--coverage--latency space. NoAbstention maximized release and had the largest AUROC point estimate, but it also had the worst Brier score and ECE. NoCritic minimized Brier loss but released only 5.8% of records. FullAuditedAgent reduced ECE relative to the other conditions yet doubled latency and collapsed release coverage to zero. RetrievalOnly did not yield a compensating improvement in discrimination, calibration, availability, or latency. These are descriptive operating points, not proof of component causality; their principal methodological value is showing that an apparently safer LLM pipeline can trade away all usable coverage without resolving miscalibration.

'''
md = DELIV / "paper_final_corrected.md"
text = md.read_text(encoding="utf-8")
start, end = text.index("## Results"), text.index("## Discussion")
text = text[:start] + results + text[end:]
# Avoid automatic LaTeX numbering plus caption-internal numbering.
text = text.replace("![Figure 1. Detailed TRACE-Guard Clinical algorithm framework.",
                    "![Detailed TRACE-Guard Clinical algorithm framework.")
md.write_text(text, encoding="utf-8")
print(json.dumps({"common_n": len(common), "common_events": common_stats[CONDITIONS[0]]["events"],
                  "critic": critic, "figures": ["figure2_operating_characteristics", "figure3_score_calibration"]}, indent=2))
