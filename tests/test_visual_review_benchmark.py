from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from researchclaw.agents.figure_agent.visual_review_benchmark import (
    VisualReviewBenchmarkError, run_visual_review_benchmark,
    verify_visual_review_benchmark,
)
from tests.test_framework_diagram import _reviewable_png


class Reviewer:
    def __init__(self):
        self.calls = 0

    def chat_image(self, prompt, image_bytes, **kwargs):
        self.calls += 1
        passed = b"good" in image_bytes
        value = {"status": "passed" if passed else "failed",
                 "semantic_score": 9 if passed else 3, "aesthetic_score": 8,
                 "issues": [] if passed else [{"dimension": "semantic",
                                                "severity": "critical",
                                                "message": "missing declared edge"}],
                 "repair_prompt": "" if passed else "Add the declared edge."}
        return SimpleNamespace(model="real-vision-model", content=json.dumps(value))


def fixture(root: Path):
    image = _reviewable_png()
    (root / "good.png").write_bytes(image + b"good")
    (root / "bad.png").write_bytes(image + b"bad")
    (root / "visual_review_benchmark_manifest.json").write_text(json.dumps({
        "schema_version": 1,
        "cases": [
            {"case_id": "control", "image": "good.png",
             "diagram_request": "A points to B", "expected_status": "passed"},
            {"case_id": "missing-edge", "image": "bad.png",
             "diagram_request": "A points to B", "expected_status": "failed"},
        ]}), encoding="utf-8")


def test_fixed_visual_benchmark_reports_rates_and_verifies(tmp_path: Path):
    fixture(tmp_path)
    reviewer = Reviewer()
    report = run_visual_review_benchmark(tmp_path, reviewer)
    assert reviewer.calls == 2
    assert report["totals"] == {
        "declared_cases": 2, "evaluated_cases": 2, "case_errors": 0,
        "defect_cases": 1, "false_accepts": 0, "false_accept_rate": 0.0,
        "control_cases": 1, "false_rejects": 0, "false_reject_rate": 0.0,
        "accuracy": 1.0,
    }
    assert verify_visual_review_benchmark(tmp_path) == report


def test_visual_benchmark_records_errors_without_inventing_verdicts(tmp_path: Path):
    fixture(tmp_path)
    report = run_visual_review_benchmark(tmp_path, object())
    assert report["cases"] == [] and report["totals"]["case_errors"] == 2
    assert report["totals"]["accuracy"] is None
    verify_visual_review_benchmark(tmp_path, report)


@pytest.mark.parametrize("mutation", [
    lambda root: (root / "bad.png").write_bytes(b"changed"),
    lambda root: (root / "visual_review_benchmark_manifest.json").write_text("{}"),
    lambda root: (root / "visual_review_benchmark_report.json").write_text("{}"),
])
def test_visual_benchmark_rejects_changed_evidence(tmp_path: Path, mutation):
    fixture(tmp_path)
    run_visual_review_benchmark(tmp_path, Reviewer())
    mutation(tmp_path)
    with pytest.raises(VisualReviewBenchmarkError):
        verify_visual_review_benchmark(tmp_path)


def test_visual_benchmark_rejects_unsafe_or_oversized_manifests(tmp_path: Path):
    (tmp_path / "visual_review_benchmark_manifest.json").write_text(json.dumps({
        "schema_version": 1, "cases": [{"case_id": "x", "image": "../x.png",
                                         "diagram_request": "x", "expected_status": "passed"}]
    }), encoding="utf-8")
    with pytest.raises(VisualReviewBenchmarkError):
        run_visual_review_benchmark(tmp_path, Reviewer())
