"""Deterministic quality gate for clinical-informatics LLM methods manuscripts."""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

Severity = Literal["error", "warning"]


@dataclass(frozen=True)
class MedicalPaperIssue:
    code: str
    severity: Severity
    message: str


@dataclass(frozen=True)
class MedicalPaperQualityReport:
    passed: bool
    word_count: int
    figure_count: int
    table_count: int
    results_subsection_count: int
    display_math_blocks: int
    issues: tuple[MedicalPaperIssue, ...]

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["issues"] = [asdict(issue) for issue in self.issues]
        return value


def _section(text: str, heading: str, next_heading: str | None = None) -> str:
    match = re.search(rf"(?im)^##\s+{re.escape(heading)}\s*$", text)
    if not match:
        return ""
    tail = text[match.end():]
    if next_heading:
        end = re.search(rf"(?im)^##\s+{re.escape(next_heading)}\s*$", tail)
    else:
        end = re.search(r"(?im)^##\s+", tail)
    return tail[:end.start()] if end else tail


def audit_medical_ai_manuscript(markdown: str, asset_dir: Path | None = None) -> MedicalPaperQualityReport:
    """Audit content/asset completeness without judging numerical correctness."""
    issues: list[MedicalPaperIssue] = []
    words = re.findall(r"\b[\w'-]+\b", re.sub(r"```.*?```", "", markdown, flags=re.S))
    images = re.findall(r"!\[([^\]]*)\]\(([^)]+)\)", markdown)
    tables = re.findall(r"(?m)^\|(?:[^\n]+)\|\s*$\n^\|(?:\s*:?-+:?\s*\|)+", markdown)
    math_blocks = len(re.findall(r"(?s)\$\$.*?\$\$", markdown))
    results = _section(markdown, "Results", "Discussion")
    methods = _section(markdown, "Materials and Methods", "Results")
    introduction = _section(markdown, "Introduction", "Related Work and Methodological Positioning")
    result_subsections = len(re.findall(r"(?m)^###\s+", results))

    def add(code: str, message: str, severity: Severity = "error") -> None:
        issues.append(MedicalPaperIssue(code, severity, message))

    if len(words) < 4500:
        add("paper.too_short", f"Full methods manuscript has {len(words)} words; expected at least 4500.")
    if not methods:
        add("methods.missing", "Materials and Methods section is missing.")
    if not results:
        add("results.missing", "Results section is missing.")
    dataset_terms = ("dataset characterization", "statistical profile", "cohort characteristics")
    if not any(term in methods.lower() for term in dataset_terms):
        add("dataset.profile_missing", "Add a dedicated dataset statistical-profile subsection.")
    if not re.search(r"(?i)missing(ness)?", methods) or not re.search(r"(?i)median|interquartile|IQR|mean", methods):
        add("dataset.statistics_incomplete", "Dataset analysis must report distribution summaries and missingness.")
    if math_blocks < 3:
        add("methods.insufficient_math", f"Found {math_blocks} display-math blocks; expected at least 3.")
    if not re.search(r"(?i)proposition|theorem|lemma", methods) or not re.search(r"(?i)proof|argument", methods):
        add("methods.no_theory", "Include a proposition/theorem and proof or explicit analytical argument.")
    intro_images = re.findall(r"!\[([^\]]*)\]\(([^)]+)\)", introduction)
    method_images = re.findall(r"!\[([^\]]*)\]\(([^)]+)\)", methods)
    conceptual_terms = ("conceptual", "overview", "architecture", "input data")
    detailed_terms = ("algorithm", "framework", "workflow", "data and decision path")
    conceptual = [(caption, path) for caption, path in intro_images
                  if any(term in caption.lower() for term in conceptual_terms)]
    detailed = [(caption, path) for caption, path in method_images
                if any(term in caption.lower() for term in detailed_terms)]
    if not conceptual:
        add("introduction.overview_figure_missing",
            "Introduction requires an abstract input-method-output conceptual architecture figure.")
    if not detailed:
        add("methods.framework_figure_missing",
            "Methods requires a separate detailed algorithm/framework figure.")
    if conceptual and detailed and {p for _, p in conceptual} & {p for _, p in detailed}:
        add("figures.architecture_reused",
            "Introduction overview and Methods framework must use different source assets.")
    if conceptual:
        caption_text = " ".join(c for c, _ in conceptual).lower()
        if not all(term in caption_text for term in ("input", "model", "output")):
            add("introduction.overview_scope_incomplete",
                "Overview caption should identify input, model/method, and output layers.", "warning")
    if result_subsections < 4:
        add("results.too_few_subsections", f"Results has {result_subsections} subsections; expected at least 4 analytical angles.")
    dimensions = {
        "discrimination": r"AUROC|AUPRC|discrimination|precision.?recall",
        "calibration": r"Brier|ECE|calibration",
        "coverage": r"coverage|abstention|release|failure|error rate",
        "efficiency": r"latency|cost|token|critic",
    }
    for name, pattern in dimensions.items():
        if not re.search(pattern, results, re.I):
            add(f"results.dimension_{name}_missing", f"Results lacks a {name} analysis.")
    result_images = [(c, p) for c, p in images if markdown.find(f"]({p})") > markdown.find("## Results")]
    if len(result_images) < 2:
        add("results.figures_insufficient", f"Found {len(result_images)} result figures; expected at least 2.")
    if len(images) < 4:
        add("figures.insufficient", f"Found {len(images)} figures; expected dataset, algorithm, and at least two result figures.")
    if len(tables) < 2:
        add("tables.insufficient", f"Found {len(tables)} Markdown tables; expected dataset and result tables.")
    for caption, path in images:
        if re.match(r"(?i)\s*(figure|fig\.)\s*\d+", caption):
            add("figures.manual_number", f"Caption for {path} manually includes a figure number.")
        if asset_dir is not None:
            source = asset_dir / path
            if not source.is_file():
                add("figures.asset_missing", f"Referenced figure does not exist: {source}")
            elif source.suffix.lower() == ".png" and not source.with_suffix(".pdf").is_file():
                add("figures.vector_pair_missing", f"PNG lacks vector PDF pair: {source.name}", "warning")
    passed = not any(issue.severity == "error" for issue in issues)
    return MedicalPaperQualityReport(passed, len(words), len(images), len(tables),
                                     result_subsections, math_blocks, tuple(issues))
