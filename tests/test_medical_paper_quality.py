from pathlib import Path

from researchclaw.pipeline.medical_paper_quality import audit_medical_ai_manuscript


def test_rejects_short_single_table_medical_paper():
    paper = """# Title
## Materials and Methods
### Data source
We used data.
## Results
### Main result
| Method | AUROC |
|---|---:|
| A | 0.5 |
## Discussion
Short.
"""
    report = audit_medical_ai_manuscript(paper)
    codes = {issue.code for issue in report.issues}
    assert not report.passed
    assert "paper.too_short" in codes
    assert "dataset.profile_missing" in codes
    assert "methods.no_theory" in codes
    assert "results.figures_insufficient" in codes


def test_current_corrected_paper_passes_structural_gate():
    root = Path(__file__).resolve().parents[1]
    deliverables = root / "artifacts/paper5_trace_guard/full_run/deliverables"
    paper = deliverables / "paper_final_corrected.md"
    if not paper.is_file():
        return
    report = audit_medical_ai_manuscript(paper.read_text(encoding="utf-8"), deliverables)
    assert report.passed, report.to_dict()


def test_requires_distinct_introduction_and_methods_architectures():
    filler = "word " * 4600
    paper = f"""# Title
## Introduction
{filler}
![Conceptual input model output architecture.](same.png)
## Related Work and Methodological Positioning
Prior work.
## Materials and Methods
### Dataset characterization and statistical profile
Median IQR and missingness are reported.
| Variable | Value |
|---|---:|
| Age | 60 |
$$a=b$$
$$c=d$$
$$e=f$$
### Proposition and proof
Proposition 1. A property holds. Proof: by construction.
![Detailed algorithm framework.](same.png)
## Results
### Discrimination
AUROC and AUPRC.
### Calibration
Brier and ECE.
### Coverage
Coverage, release, abstention, and error rate.
### Efficiency
Latency and critic cost.
| Method | Result |
|---|---:|
| A | 1 |
![Result one.](r1.png)
![Result two.](r2.png)
## Discussion
Discussion.
"""
    report = audit_medical_ai_manuscript(paper)
    assert "figures.architecture_reused" in {issue.code for issue in report.issues}
    assert not report.passed
