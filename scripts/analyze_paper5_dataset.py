"""Create verified cohort statistics/figure and insert the dataset-analysis subsection."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu

from paper5_plot_style import BLUE, CYAN, GREEN, MAGENTA, NAVY, RED, apply_style

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parent / "paper_5/data/data.csv"
DELIV = ROOT / "artifacts/paper5_trace_guard/full_run/deliverables"
OUTCOME = "抗凝药后是否出血"
ALLOW = ["性别", "年龄", "是否血透", "入院spO2", "是否使用抗凝药物", "has—bled", "caprini",
         "入血红蛋白", "入PLT血小板计数", "入WBC白细胞计数", "入RBC", "入PT", "入APTT", "入FIB",
         "入D—D", "入tbi", "入ALT", "入ALB", "入SCr", "入egfr", "入pct", "入LI—6", "入CRP",
         "入尿素", "入胆固醇", "入尿酸"]

df = pd.read_csv(DATA, encoding="utf-8-sig")
y = (pd.to_numeric(df[OUTCOME], errors="coerce") == 1).astype(int)
assert len(df) == 86 and int(y.sum()) == 16

continuous = [
    ("Age, years", "年龄"), ("HAS-BLED, recorded", "has—bled"), ("Caprini, recorded", "caprini"),
    ("Admission haemoglobin", "入血红蛋白"), ("Admission platelet count", "入PLT血小板计数"),
    ("Admission WBC count", "入WBC白细胞计数"), ("Admission APTT", "入APTT"),
    ("Admission creatinine", "入SCr"), ("Admission eGFR", "入egfr"), ("Admission CRP", "入CRP"),
]
categorical = [
    ("Male sex", "性别", 1), ("Haemodialysis", "是否血透", 1),
    ("Anticoagulant use", "是否使用抗凝药物", 1),
]

def summary(v: pd.Series) -> str:
    x = pd.to_numeric(v, errors="coerce").dropna().to_numpy(float)
    return f"{np.median(x):.2f} [{np.quantile(x,.25):.2f}–{np.quantile(x,.75):.2f}]"

records = []
for label, col in continuous:
    a = pd.to_numeric(df.loc[y == 1, col], errors="coerce").dropna()
    b = pd.to_numeric(df.loc[y == 0, col], errors="coerce").dropna()
    p = mannwhitneyu(a, b, alternative="two-sided", method="asymptotic").pvalue
    records.append({"variable": label, "type": "continuous", "overall": summary(df[col]),
                    "bleeding": summary(a), "no_bleeding": summary(b), "p_value": float(p),
                    "missing_n": int(df[col].isna().sum())})
for label, col, positive in categorical:
    x = pd.to_numeric(df[col], errors="coerce")
    counts = []
    for group in [1, 0]:
        mask = y == group; n = int(mask.sum()); k = int(((x == positive) & mask).sum()); counts.append((k, n))
    table = [[counts[0][0], counts[0][1]-counts[0][0]], [counts[1][0], counts[1][1]-counts[1][0]]]
    p = fisher_exact(table).pvalue
    overall_k = int((x == positive).sum())
    records.append({"variable": label, "type": "categorical", "overall": f"{overall_k} ({100*overall_k/86:.1f}%)",
                    "bleeding": f"{counts[0][0]} ({100*counts[0][0]/counts[0][1]:.1f}%)",
                    "no_bleeding": f"{counts[1][0]} ({100*counts[1][0]/counts[1][1]:.1f}%)",
                    "p_value": float(p), "missing_n": int(x.isna().sum())})

missing = (df[ALLOW].isna().mean()*100).sort_values(ascending=False)
artifact = {"schema": "paper5-dataset-profile/v1", "n": 86, "events": 16,
            "tests": "Mann-Whitney U for continuous; Fisher exact for binary; two-sided, unadjusted, exploratory",
            "variables": records, "allowed_field_missing_percent": {k: float(v) for k, v in missing.items()}}
(DELIV / "dataset_statistical_profile.json").write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
pd.DataFrame(records).to_csv(DELIV / "dataset_descriptive_statistics.csv", index=False, encoding="utf-8-sig")

apply_style()
fig, axs = plt.subplots(2, 2, figsize=(11.2, 8.1), constrained_layout=True)
ax = axs[0, 0]
bars = ax.bar([0, 1], [70, 16], color=[BLUE, RED], width=.62)
ax.set_xticks([0, 1], ["No recorded\nbleeding", "Recorded\nbleeding"]); ax.set_ylabel("Patients, n")
ax.set_title("A  Outcome composition", loc="left")
for b, n in zip(bars, [70, 16]): ax.text(b.get_x()+b.get_width()/2, n+1.2, f"{n} ({n/86*100:.1f}%)", ha="center", fontsize=9)
ax.set_ylim(0, 80)

ax = axs[0, 1]
bins = np.arange(np.floor(df["年龄"].min()/5)*5, np.ceil(df["年龄"].max()/5)*5+5, 5)
ax.hist(df.loc[y == 0, "年龄"], bins=bins, color=BLUE, alpha=.65, label="No bleeding", edgecolor="white")
ax.hist(df.loc[y == 1, "年龄"], bins=bins, color=RED, alpha=.75, label="Bleeding", edgecolor="white")
ax.axvline(df.loc[y == 0, "年龄"].median(), color=BLUE, lw=1.5, ls="--")
ax.axvline(df.loc[y == 1, "年龄"].median(), color=RED, lw=1.5, ls="--")
ax.set_xlabel("Age (years)"); ax.set_ylabel("Patients, n"); ax.set_title("B  Age distribution by outcome", loc="left"); ax.legend()

ax = axs[1, 0]
positions, values, cols = [], [], []
for i, col in enumerate(["has—bled", "caprini"]):
    for event, off, color in [(0, -.18, BLUE), (1, .18, RED)]:
        vals = pd.to_numeric(df.loc[y == event, col], errors="coerce").dropna().to_numpy()
        positions.append(i+off); values.append(vals); cols.append(color)
bp = ax.boxplot(values, positions=positions, widths=.3, patch_artist=True, showfliers=True,
                medianprops={"color": "white", "lw": 1.5})
for patch, color in zip(bp["boxes"], cols): patch.set(facecolor=color, edgecolor=color, alpha=.8)
for key in ["whiskers", "caps"]:
    for artist in bp[key]: artist.set(color=NAVY, lw=.8)
ax.set_xticks([0, 1], ["HAS-BLED", "Caprini"]); ax.set_ylabel("Recorded score")
ax.set_title("C  Recorded clinical risk scores", loc="left")
ax.scatter([], [], color=BLUE, label="No bleeding"); ax.scatter([], [], color=RED, label="Bleeding"); ax.legend()

ax = axs[1, 1]
top = missing.head(12).sort_values()
labels = {"入院spO2":"Admission SpO₂", "入PT":"Admission PT", "入FIB":"Admission fibrinogen",
          "入D—D":"Admission D-dimer", "入APTT":"Admission APTT", "入CRP":"Admission CRP",
          "入LI—6":"Admission IL-6", "入尿素":"Admission urea", "入egfr":"Admission eGFR",
          "入RBC":"Admission RBC", "入ALB":"Admission albumin", "入tbi":"Admission bilirubin"}
ax.barh(np.arange(len(top)), top.values, color=BLUE, edgecolor=NAVY, linewidth=.4)
ax.set_yticks(np.arange(len(top)), [labels.get(k, k) for k in top.index]); ax.set_xlabel("Missing records (%)")
ax.set_title("D  Missingness among frozen input fields", loc="left"); ax.set_xlim(0, max(top.values)*1.18)
for i, v in enumerate(top.values): ax.text(v+.35, i, f"{v:.1f}", va="center", fontsize=7.5)
fig.savefig(DELIV / "figure_dataset_profile.png", dpi=300, bbox_inches="tight")
fig.savefig(DELIV / "figure_dataset_profile.pdf", bbox_inches="tight")
plt.close(fig)

table_rows = []
for r in records:
    table_rows.append(f"| {r['variable']} | {r['overall']} | {r['bleeding']} | {r['no_bleeding']} | {r['missing_n']} | {r['p_value']:.3f} |")
subsection = f'''### Dataset characterization and statistical profile

The dataset was characterized before examining LLM performance. Continuous variables are reported as median [interquartile range] because of the small event group and visibly non-Gaussian laboratory distributions; categorical variables are n (%). Two-sided Mann--Whitney U and Fisher exact tests were used for exploratory outcome-group comparisons. The resulting $P$ values are unadjusted descriptive indices, not confirmatory evidence, and no variable screening or model fitting was based on them.

| Variable | Overall (n=86) | Bleeding (n=16) | No bleeding (n=70) | Missing, n | Exploratory P |
|---|---:|---:|---:|---:|---:|
{chr(10).join(table_rows)}

The median age was {summary(df['年龄'])} years overall and was similar in the bleeding and non-bleeding groups ({summary(df.loc[y==1,'年龄'])} versus {summary(df.loc[y==0,'年龄'])}). Recorded HAS-BLED was higher in the bleeding group ({summary(df.loc[y==1,'has—bled'])}) than in the non-bleeding group ({summary(df.loc[y==0,'has—bled'])}), whereas the corresponding Caprini distributions were {summary(df.loc[y==1,'caprini'])} and {summary(df.loc[y==0,'caprini'])}. These comparisons describe the supplied cohort only; they cannot validate either clinical score because the outcome definition, sampling process, and score collection timing were not independently adjudicated.

Across the 13 exploratory comparisons, recorded HAS-BLED was the only variable with an unadjusted $P<0.05$ ($P=0.0004$); age, Caprini, the selected admission laboratory values, sex, haemodialysis, and anticoagulant-use status did not show evidence of group separation in this small sample. This pattern should not be read as evidence that the other variables are unrelated to bleeding: with only 16 events, absence of a small $P$ value is compatible with low precision, nonlinear association, and residual confounding. Figure 1 summarizes outcome imbalance, distributional overlap, and input completeness.

![Cohort composition, age distribution, recorded clinical risk scores, and missingness among frozen model-input fields. Dashed lines in panel B indicate group medians.](figure_dataset_profile.png)

The frozen 26-field input set was complete for sex, age, haemodialysis, anticoagulant use, HAS-BLED, and Caprini. Missingness was concentrated in admission oxygen saturation (26/86; 30.2%), followed by PT (10/86; 11.6%), fibrinogen and D-dimer (9/86 each; 10.5%), and APTT (8/86; 9.3%). Missing values were retained explicitly rather than imputed. Consequently, the state cards represent both clinical measurements and heterogeneous measurement availability, a property that may influence LLM abstention and scoring behavior.

'''
md = DELIV / "paper_final_corrected.md"
text = md.read_text(encoding="utf-8")
if "### Dataset characterization and statistical profile" in text:
    a = text.index("### Dataset characterization and statistical profile")
    b = text.index("### Frozen predictor contract")
    text = text[:a] + subsection + text[b:]
else:
    at = text.index("### Frozen predictor contract")
    text = text[:at] + subsection + text[at:]
# The new dataset figure is first in manuscript order; normalize explicit references.
text = text.replace("Figure 1 formalizes the complete data and decision path", "Figure 2 formalizes the complete data and decision path")
text = text.replace("Figure 2A presents discrimination", "Figure 3A presents discrimination")
text = text.replace("Calibration-related losses were uniformly large (Figure 2B).", "Calibration-related losses were uniformly large (Figure 3B).")
text = text.replace("Calibration-related losses were uniformly large (Figure 4B).", "Calibration-related losses were uniformly large (Figure 3B).")
text = text.replace("Figure 2C separates", "Figure 3C separates").replace("(Figure 2D)", "(Figure 3D)")
text = text.replace("Figure 3A exposes the source", "Figure 4A exposes the source")
text = text.replace("The empirical calibration curves (Figure 3B)", "The empirical calibration curves (Figure 4B)")
md.write_text(text, encoding="utf-8")
print(json.dumps({"n": len(df), "events": int(y.sum()), "table_rows": len(records),
                  "max_missing": {missing.index[0]: int(df[missing.index[0]].isna().sum())}}, ensure_ascii=False))
