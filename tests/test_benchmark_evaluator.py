from __future__ import annotations

import copy
import json

import pytest

from researchclaw.pipeline.benchmark_evaluator import (
    BenchmarkEvaluationError, build_benchmark_evaluation,
    verify_benchmark_evaluation,
)
from researchclaw.pipeline.benchmark_suite import freeze_benchmark_suite
from researchclaw.pipeline.evidence_store import content_hash, file_hash
from researchclaw.pipeline.resource_ledger import build_resource_ledger
from tests.test_benchmark_suite import suite_fixture, write_json


def evaluation_fixture(tmp_path):
    public, plan_path, gold_path, _ = suite_fixture(tmp_path)
    suite = freeze_benchmark_suite(public, plan_path, gold_path)
    gold = json.loads(gold_path.read_text(encoding="utf-8"))
    expected = {row["case_id"]: row["expected_disposition"] for row in gold["cases"]}
    results_root = public / "results"
    manifest_cases = []
    assessments = []
    for index, case in enumerate(suite["plan"]["cases"]):
        case_id = case["case_id"]
        run = results_root / case_id
        run.mkdir(parents=True)
        target_met = expected[case_id] != "reject"
        acceptance = {
            "schema_version": 1, "checker": "final_acceptance/v1",
            "artifact_status": "research_complete" if target_met else "exploratory",
            "target_status": "research_complete", "target_met": target_met,
            "issues": [], "dimensions": {},
        }
        write_json(run / "final_acceptance.json", acceptance)
        ledger = build_resource_ledger(run)
        write_json(run / "resource_ledger.json", ledger)
        receipt = {
            "schema_version": 1, "recorder": "arc-benchmark-runner/v1",
            "suite_version": suite["version"], "case_id": case_id,
            "input_sha256": case["input_sha256"], "budget": suite["plan"]["budget"],
            "started_at": float(index), "finished_at": float(index + 10),
            "wall_seconds": 10.0, "model_calls": 0, "total_tokens": 0,
        }
        write_json(run / "benchmark_case_receipt.json", receipt)
        manifest_cases.append({
            "case_id": case_id, "run_dir": case_id,
            "receipt_sha256": file_hash(run / "benchmark_case_receipt.json"),
            "acceptance_sha256": file_hash(run / "final_acceptance.json"),
            "resource_ledger_sha256": file_hash(run / "resource_ledger.json"),
        })
        assessments.append({
            "case_id": case_id,
            "acceptance_sha256": file_hash(run / "final_acceptance.json"),
            "observed_disposition": expected[case_id], "critical_errors": 0,
            "revision_minutes": 5.0, "revision_edits": 2,
            "rationale": "Blind fixture assessment.",
        })
    result_manifest_path = results_root / "benchmark_result_manifest.json"
    assessment_path = tmp_path / "private" / "assessment.json"
    write_json(result_manifest_path, {
        "schema_version": 1, "suite_version": suite["version"],
        "cases": manifest_cases})
    write_json(assessment_path, {
        "schema_version": 1, "suite_version": suite["version"],
        "assessor_id": "assessor-1", "blinded": True, "cases": assessments})
    args = {
        "suite_report": suite, "public_root": public, "plan_path": plan_path,
        "gold_path": gold_path, "results_root": results_root,
        "result_manifest_path": result_manifest_path,
        "assessment_path": assessment_path,
    }
    return args, manifest_cases, assessments


def test_complete_blind_evaluation_reports_grouped_rates_and_revision_burden(tmp_path):
    args, _, _ = evaluation_fixture(tmp_path)
    report = build_benchmark_evaluation(**args)
    assert report["overall"] == {
        "cases": 16, "correct_dispositions": 16, "disposition_accuracy": 1.0,
        "error_acceptances": 0, "error_acceptance_rate": 0.0,
        "reject_cases": 14, "accept_cases": 1, "false_rejections": 0,
        "false_rejection_rate": 0.0, "honest_negative_cases": 1,
        "honest_negative_accuracy": 1.0,
        "critical_errors": 0, "mean_revision_minutes": 5.0,
        "mean_revision_edits": 2.0, "budget_exceeded": 0,
    }
    assert set(report["by_task_family"]) == {
        "tabular_classification", "tabular_regression", "temporal", "image"}
    assert len(report["by_stress_scenario"]) == 8
    assert all(row["receipt_sha256"] for row in report["cases"])
    assert verify_benchmark_evaluation(report, **args) == report


def test_error_acceptance_and_budget_excess_are_not_hidden(tmp_path):
    args, manifest_cases, assessments = evaluation_fixture(tmp_path)
    assessments[0]["observed_disposition"] = "reject"
    assessments[1]["observed_disposition"] = "accept"
    write_json(args["assessment_path"], {
        "schema_version": 1, "suite_version": args["suite_report"]["version"],
        "assessor_id": "assessor-1", "blinded": True, "cases": assessments})
    run = args["results_root"] / manifest_cases[1]["run_dir"]
    receipt_path = run / "benchmark_case_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt.update(finished_at=302.0, wall_seconds=301.0)
    write_json(receipt_path, receipt)
    manifest_cases[1]["receipt_sha256"] = file_hash(receipt_path)
    write_json(args["result_manifest_path"], {
        "schema_version": 1, "suite_version": args["suite_report"]["version"],
        "cases": manifest_cases})
    report = build_benchmark_evaluation(**args)
    assert report["overall"]["error_acceptances"] == 1
    assert report["overall"]["error_acceptance_rate"] == pytest.approx(1 / 14)
    assert report["overall"]["false_rejections"] == 1
    assert report["overall"]["false_rejection_rate"] == 1.0
    assert report["overall"]["budget_exceeded"] == 1
    assert report["cases"][1]["budget_status"] == "exceeded"


def test_runner_receipt_cannot_undercount_visible_resource_evidence(tmp_path):
    args, manifest_cases, _ = evaluation_fixture(tmp_path)
    run = args["results_root"] / manifest_cases[0]["run_dir"]
    # A directly frozen ledger row is re-derived by build_resource_ledger;
    # monkeying only the aggregate would be rejected before receipt checks.
    from researchclaw.llm.call_ledger import build_call_ledger
    llm = build_call_ledger([{
        "provider": "fixture", "requested_model": "m", "served_model": "m",
        "endpoint": "fixture", "parameters": {}, "fallback_failures": [],
        "status": "succeeded", "prompt_tokens": 4, "completion_tokens": 6,
        "total_tokens": 10,
    }])
    write_json(run / "llm_call_ledger.json", llm)
    write_json(run / "resource_ledger.json", build_resource_ledger(run))
    manifest_cases[0]["resource_ledger_sha256"] = file_hash(run / "resource_ledger.json")
    write_json(args["result_manifest_path"], {
        "schema_version": 1, "suite_version": args["suite_report"]["version"],
        "cases": manifest_cases})
    with pytest.raises(BenchmarkEvaluationError, match="undercounts"):
        build_benchmark_evaluation(**args)


def test_one_run_directory_cannot_be_reused_for_two_cases(tmp_path):
    args, manifest_cases, _ = evaluation_fixture(tmp_path)
    manifest_cases[1]["run_dir"] = manifest_cases[0]["run_dir"]
    write_json(args["result_manifest_path"], {
        "schema_version": 1, "suite_version": args["suite_report"]["version"],
        "cases": manifest_cases})
    with pytest.raises(BenchmarkEvaluationError, match="reused"):
        build_benchmark_evaluation(**args)


def test_private_assessment_must_be_blind_complete_and_bound_to_output(tmp_path):
    args, _, assessments = evaluation_fixture(tmp_path)
    document = json.loads(args["assessment_path"].read_text(encoding="utf-8"))
    document["blinded"] = False
    write_json(args["assessment_path"], document)
    with pytest.raises(BenchmarkEvaluationError, match="not blinded"):
        build_benchmark_evaluation(**args)
    document["blinded"] = True
    document["cases"][0]["acceptance_sha256"] = "0" * 64
    write_json(args["assessment_path"], document)
    with pytest.raises(BenchmarkEvaluationError, match="assessment case"):
        build_benchmark_evaluation(**args)
    inside = args["public_root"] / "assessment.json"
    write_json(inside, document)
    with pytest.raises(BenchmarkEvaluationError, match="outside"):
        build_benchmark_evaluation(**dict(args, assessment_path=inside))


def test_evaluation_rejects_tamper_even_after_report_rehash(tmp_path):
    args, _, _ = evaluation_fixture(tmp_path)
    report = build_benchmark_evaluation(**args)
    changed = copy.deepcopy(report)
    changed["overall"]["correct_dispositions"] = 0
    changed["version"] = content_hash({key: value for key, value in changed.items()
                                       if key != "version"})
    with pytest.raises(BenchmarkEvaluationError, match="differs"):
        verify_benchmark_evaluation(changed, **args)
