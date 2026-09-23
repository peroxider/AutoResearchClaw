"""Fixed-budget benchmark report: outcome classification and honest aggregation."""
import json
from pathlib import Path
from typing import Any

import pytest

from researchclaw.pipeline.benchmark_report import (
    BenchmarkReportError,
    build_benchmark_report,
    validate_benchmark_report,
    write_benchmark_report,
)
from researchclaw.pipeline.evidence_store import content_hash, file_hash
from researchclaw.pipeline.resource_ledger import build_resource_ledger
from researchclaw.llm.call_ledger import build_call_ledger


def write_acceptance(run_dir: Path, status: str,
                     issue_dimensions: list[str]) -> str:
    run_dir.mkdir(parents=True, exist_ok=True)
    verdict = {
        "schema_version": 1, "checker": "final_acceptance/v1",
        "input_version": "iv", "evidence_version": "ev", "review_version": "rv",
        "artifact_status": status, "target_status": "research_complete",
        "target_met": status != "exploratory", "dimensions": {},
        "issues": [{"dimension": dimension, "reason": f"seeded_{dimension}",
                    "artifact": "seed"} for dimension in issue_dimensions],
        "file_hashes": {}}
    path = run_dir / "final_acceptance.json"
    path.write_text(json.dumps(verdict), encoding="utf-8")
    return file_hash(path)


def write_costs(run_dir: Path, records: list[dict],
                extra: dict[str, Any] | None = None) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "llm_call_ledger.json").write_text(
        json.dumps(build_call_ledger(records)), encoding="utf-8")
    for name, document in (extra or {}).items():
        (run_dir / name).write_text(json.dumps(document), encoding="utf-8")
    (run_dir / "resource_ledger.json").write_text(
        json.dumps(build_resource_ledger(run_dir)), encoding="utf-8")


def case(case_id: str, kind: str, run_dir: str, digest: str,
         category: str | None = None, dimension: str | None = None) -> dict:
    entry = {"case_id": case_id, "kind": kind, "run_dir": run_dir,
             "acceptance_sha256": digest}
    if category is not None:
        entry["defect_category"] = category
    if dimension is not None:
        entry["expected_dimension"] = dimension
    return entry


def write_manifest(root: Path, cases: list[dict]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "benchmark_manifest.json").write_text(
        json.dumps({"schema_version": 1, "cases": cases}), encoding="utf-8")


def measured(tokens: int) -> dict:
    return {"status": "succeeded", "fallback_failures": [],
            "prompt_tokens": tokens - 1, "completion_tokens": 1,
            "total_tokens": tokens}


def unmeasured() -> dict:
    return {"status": "failed", "fallback_failures": [{"model": "a", "error": "e"}],
            "prompt_tokens": None, "completion_tokens": None, "total_tokens": None}


def test_defect_and_control_outcomes_classify_and_rate(tmp_path):
    d1 = write_acceptance(tmp_path / "runs" / "d1", "exploratory", ["citations"])
    d2 = write_acceptance(tmp_path / "runs" / "d2", "research_complete", [])
    c1 = write_acceptance(tmp_path / "runs" / "c1", "research_complete", [])
    c2 = write_acceptance(tmp_path / "runs" / "c2", "exploratory", ["numeric"])
    write_costs(tmp_path / "runs" / "d1", [measured(5)])
    write_costs(tmp_path / "runs" / "d2", [measured(7)])
    write_costs(tmp_path / "runs" / "c1", [measured(3)])
    write_costs(tmp_path / "runs" / "c2", [measured(3)])
    write_manifest(tmp_path, [
        case("d1", "defect", "runs/d1", d1, "fabricated_citation", "citations"),
        case("d2", "defect", "runs/d2", d2, "stale_result", "numeric"),
        case("c1", "control", "runs/c1", c1),
        case("c2", "control", "runs/c2", c2)])
    report = build_benchmark_report(tmp_path)
    assert [(r["case_id"], r["outcome"]) for r in report["cases"]] == \
        [("d1", "detected"), ("d2", "accepted_defect"),
         ("c1", "control_passed"), ("c2", "control_rejected")]
    assert report["totals"] == {
        "defect_cases": 2, "detected": 1, "accepted_defects": 1,
        "rejected_other_reason": 0, "error_acceptance_rate": 0.5,
        "detection_rate": 0.5, "control_cases": 2, "controls_rejected": 1,
        "control_rejection_rate": 0.5,
        "total_tokens": 18, "llm_chat_calls": 4, "llm_chat_failures": 0,
        "llm_review_calls": None, "writing_attempts": None}
    validate_benchmark_report(tmp_path, report)
    written = write_benchmark_report(tmp_path)
    assert written == report
    validate_benchmark_report(tmp_path, json.loads(
        (tmp_path / "benchmark_report.json").read_text()))


def test_rejected_other_reason_is_neither_detected_nor_accepted(tmp_path):
    d1 = write_acceptance(tmp_path / "runs" / "d1", "exploratory", ["numeric"])
    write_costs(tmp_path / "runs" / "d1", [measured(4)])
    write_manifest(tmp_path, [
        case("d1", "defect", "runs/d1", d1, "fabricated_citation", "citations")])
    (report,) = [build_benchmark_report(tmp_path)]
    assert report["cases"][0]["outcome"] == "rejected_other_reason"
    assert report["totals"]["detected"] == 0
    assert report["totals"]["accepted_defects"] == 0
    assert report["totals"]["rejected_other_reason"] == 1
    assert report["totals"]["error_acceptance_rate"] == 0.0
    assert report["totals"]["detection_rate"] == 0.0


def test_case_errors_exclude_cases_from_rates(tmp_path):
    d1 = write_acceptance(tmp_path / "runs" / "d1", "exploratory", ["citations"])
    c1 = write_acceptance(tmp_path / "runs" / "c1", "exploratory", ["numeric"])
    write_costs(tmp_path / "runs" / "d1", [measured(5)])
    write_manifest(tmp_path, [
        case("d1", "defect", "runs/d1", d1, "fabricated_citation", "citations"),
        case("d2", "defect", "runs/missing", "0" * 64, "stale_result", "numeric"),
        case("c1", "control", "runs/c1", c1),
        case("c2", "control", "runs/c2", "1" * 64)])
    report = build_benchmark_report(tmp_path)
    assert report["case_errors"] == [
        {"case_id": "d2", "reason": "missing_final_acceptance"},
        {"case_id": "c1", "reason": "missing_resource_ledger"},
        {"case_id": "c2", "reason": "missing_final_acceptance"}]
    assert report["totals"]["defect_cases"] == 1
    assert report["totals"]["detected"] == 1
    assert report["totals"]["error_acceptance_rate"] == 0.0
    assert report["totals"]["control_cases"] == 0
    assert report["totals"]["control_rejection_rate"] is None


def test_unreadable_acceptance_is_a_case_error(tmp_path):
    run_dir = tmp_path / "runs" / "d1"
    run_dir.mkdir(parents=True)
    (run_dir / "final_acceptance.json").write_text("{not json", encoding="utf-8")
    write_manifest(tmp_path, [
        case("d1", "defect", "runs/d1", "0" * 64, "x", "citations")])
    report = build_benchmark_report(tmp_path)
    assert report["case_errors"] == [
        {"case_id": "d1", "reason": "unreadable_final_acceptance:JSONDecodeError"}]
    assert report["totals"]["defect_cases"] == 0
    assert report["totals"]["error_acceptance_rate"] is None


def test_malformed_acceptance_shape_is_a_case_error(tmp_path):
    run_dir = tmp_path / "runs" / "d1"
    run_dir.mkdir(parents=True)
    (run_dir / "final_acceptance.json").write_text(
        json.dumps({"schema_version": 1, "checker": "other/v1",
                    "artifact_status": "banana", "issues": "no"}), encoding="utf-8")
    write_manifest(tmp_path, [
        case("d1", "defect", "runs/d1", file_hash(run_dir / "final_acceptance.json"),
             "x", "citations")])
    report = build_benchmark_report(tmp_path)
    assert report["case_errors"] == [
        {"case_id": "d1", "reason": "malformed_final_acceptance"}]


def test_digest_mismatch_with_existing_acceptance_is_a_case_error(tmp_path):
    d1 = write_acceptance(tmp_path / "runs" / "d1", "exploratory", ["citations"])
    write_costs(tmp_path / "runs" / "d1", [measured(5)])
    write_manifest(tmp_path, [
        case("d1", "defect", "runs/d1", "f" * 64, "fabricated_citation",
             "citations")])
    report = build_benchmark_report(tmp_path)
    assert report["case_errors"] == [
        {"case_id": "d1", "reason": "acceptance_digest_mismatch"}]
    assert report["cases"] == []
    assert report["totals"]["detection_rate"] is None


def test_tampered_resource_ledger_is_a_case_error(tmp_path):
    d1 = write_acceptance(tmp_path / "runs" / "d1", "exploratory", ["citations"])
    write_costs(tmp_path / "runs" / "d1", [measured(5)])
    write_manifest(tmp_path, [
        case("d1", "defect", "runs/d1", d1, "fabricated_citation", "citations")])
    assert build_benchmark_report(tmp_path)["case_errors"] == []
    (tmp_path / "runs" / "d1" / "resource_ledger.json").write_text(
        "{broken", encoding="utf-8")
    report = build_benchmark_report(tmp_path)
    assert report["case_errors"] == [
        {"case_id": "d1", "reason": "unreadable_resource_ledger:JSONDecodeError"}]
    assert report["totals"]["total_tokens"] is None


def test_cost_and_revision_proxies_aggregate(tmp_path):
    d1 = write_acceptance(tmp_path / "runs" / "d1", "exploratory", ["citations"])
    d2 = write_acceptance(tmp_path / "runs" / "d2", "exploratory", ["numeric"])
    write_costs(tmp_path / "runs" / "d1", [measured(5), unmeasured()], {
        "literature_evidence.json": {"review_budget": {"calls": 2, "limit": 3,
                                                       "failures": []}},
        "manuscript_ir.json": {"review_budget": {"calls": 1, "limit": 2,
                                                 "failures": []},
                               "sections": [{"attempts": ["a", "b"]}]}})
    write_costs(tmp_path / "runs" / "d2", [unmeasured()])
    write_manifest(tmp_path, [
        case("d1", "defect", "runs/d1", d1, "fabricated_citation", "citations"),
        case("d2", "defect", "runs/d2", d2, "stale_result", "numeric")])
    report = build_benchmark_report(tmp_path)
    (first, second) = report["cases"]
    assert first["cost"] == {"llm_chat_calls": 2, "llm_chat_failures": 1,
                             "total_tokens": 5, "llm_review_calls": 3,
                             "writing_attempts": 2}
    assert second["cost"]["total_tokens"] is None
    assert second["cost"]["llm_chat_calls"] == 1
    # The unmeasured case contributes no fabricated zero.
    assert report["totals"]["total_tokens"] == 5
    assert report["totals"]["llm_chat_calls"] == 3
    assert report["totals"]["llm_review_calls"] == 3
    assert report["totals"]["writing_attempts"] == 2


def test_zero_defect_manifest_has_null_defect_rates(tmp_path):
    c1 = write_acceptance(tmp_path / "runs" / "c1", "research_complete", [])
    write_costs(tmp_path / "runs" / "c1", [measured(2)])
    write_manifest(tmp_path, [case("c1", "control", "runs/c1", c1)])
    report = build_benchmark_report(tmp_path)
    assert report["totals"]["defect_cases"] == 0
    assert report["totals"]["error_acceptance_rate"] is None
    assert report["totals"]["detection_rate"] is None
    assert report["totals"]["control_rejection_rate"] == 0.0
    assert report["totals"]["total_tokens"] == 2


def test_malformed_manifest_fails_closed(tmp_path):
    root = tmp_path
    with pytest.raises(BenchmarkReportError, match="missing"):
        build_benchmark_report(root)
    (root / "benchmark_manifest.json").write_text("{oops", encoding="utf-8")
    with pytest.raises(BenchmarkReportError, match="unreadable"):
        build_benchmark_report(root)
    d1 = write_acceptance(root / "runs" / "d1", "exploratory", ["citations"])
    good = case("d1", "defect", "runs/d1", d1, "fabricated_citation", "citations")
    bad_manifests = [
        {"schema_version": 2, "cases": [good]},
        {"schema_version": 1, "cases": []},
        {"schema_version": 1, "cases": [good, dict(good)]},
        {"schema_version": 1, "cases": [dict(good, kind="banana")]},
        {"schema_version": 1, "cases": [dict(good, expected_dimension="banana")]},
        {"schema_version": 1, "cases": [{key: value for key, value in good.items()
                                         if key != "expected_dimension"}]},
        {"schema_version": 1, "cases": [dict(good, acceptance_sha256="")]},
        {"schema_version": 1, "cases": [case("c1", "control", "runs/c1", d1,
                                             "some_category")]},
        {"schema_version": 1, "cases": [case("", "control", "runs/c1", d1)]},
        {"schema_version": 1, "cases": ["not-a-mapping"]},
    ]
    for manifest in bad_manifests:
        (root / "benchmark_manifest.json").write_text(json.dumps(manifest),
                                                      encoding="utf-8")
        with pytest.raises(BenchmarkReportError, match="malformed"):
            build_benchmark_report(root)


def test_validation_rejects_tampering(tmp_path):
    d1 = write_acceptance(tmp_path / "runs" / "d1", "exploratory", ["citations"])
    write_costs(tmp_path / "runs" / "d1", [measured(5)])
    write_manifest(tmp_path, [
        case("d1", "defect", "runs/d1", d1, "fabricated_citation", "citations")])
    report = write_benchmark_report(tmp_path)
    stored = json.loads((tmp_path / "benchmark_report.json").read_text())
    validate_benchmark_report(tmp_path, stored)
    with pytest.raises(BenchmarkReportError, match="malformed"):
        validate_benchmark_report(tmp_path, {"schema_version": 2})
    with pytest.raises(BenchmarkReportError, match="version changed"):
        validate_benchmark_report(
            tmp_path, dict(stored, totals=dict(stored["totals"], detected=99)))
    rehashed = dict(stored, totals=dict(stored["totals"], detected=99))
    rehashed["version"] = content_hash(
        {key: value for key, value in rehashed.items() if key != "version"})
    with pytest.raises(BenchmarkReportError, match="differs from the frozen"):
        validate_benchmark_report(tmp_path, rehashed)
