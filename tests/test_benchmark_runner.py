from __future__ import annotations

import json

import pytest

from researchclaw.pipeline.benchmark_runner import (
    BenchmarkRunnerError, run_benchmark_suite,
)
from researchclaw.pipeline.benchmark_suite import freeze_benchmark_suite
from researchclaw.pipeline.evidence_store import content_hash, file_hash
from tests.test_benchmark_suite import suite_fixture, write_json


ADAPTER = r'''import json
import os
import sys
import time
from pathlib import Path

case = json.loads(os.environ["ARC_BENCHMARK_CASE"])
observation = {
    "case_id": case["case_id"],
    "allowed": os.environ.get("RC_BENCHMARK_ALLOWED"),
    "blocked_visible": "RC_BENCHMARK_BLOCKED" in os.environ,
    "input_exists": Path(os.environ["ARC_BENCHMARK_INPUT"]).is_file(),
}
Path("adapter_observation.json").write_text(json.dumps(observation), encoding="utf-8")
print(case["case_id"])
if case["case_id"] == "case-00" and os.environ.get("RC_TIMEOUT") == "1":
    time.sleep(3)
if case["case_id"] == "case-01" and os.environ.get("RC_FAIL") == "1":
    print("intentional fixture failure", file=sys.stderr)
    raise SystemExit(3)
'''


def runner_fixture(tmp_path, monkeypatch, *, wall_seconds=300):
    public, plan_path, gold_path, plan = suite_fixture(tmp_path)
    adapter = public / "runner_adapter.py"
    adapter.write_text(ADAPTER, encoding="utf-8")
    plan["budget"]["wall_seconds"] = wall_seconds
    plan["runner"] = {
        "adapter": "runner_adapter.py",
        "adapter_sha256": file_hash(adapter),
        "environment_allowlist": [
            "RC_BENCHMARK_ALLOWED", "RC_FAIL", "RC_TIMEOUT"],
    }
    write_json(plan_path, plan)
    suite = freeze_benchmark_suite(public, plan_path, gold_path)
    monkeypatch.setenv("RC_BENCHMARK_ALLOWED", "visible")
    monkeypatch.setenv("RC_BENCHMARK_BLOCKED", "secret")
    return public, plan_path, gold_path, suite


def test_public_runner_executes_every_case_once_and_preserves_failure(
        tmp_path, monkeypatch):
    public, plan_path, gold_path, suite = runner_fixture(tmp_path, monkeypatch)
    monkeypatch.setenv("RC_FAIL", "1")
    # The runner receives only the report, public root and public plan. Changing
    # private gold after freezing cannot affect or be detected by this execution.
    gold_path.write_text("private gold unavailable to runner", encoding="utf-8")
    results = tmp_path / "results"
    manifest = run_benchmark_suite(
        suite_report=suite, public_root=public, plan_path=plan_path,
        results_root=results)
    assert len(manifest["cases"]) == 16
    assert manifest["version"] == content_hash(
        {key: value for key, value in manifest.items() if key != "version"})
    statuses = []
    for item in manifest["cases"]:
        run = results / item["run_dir"]
        receipt = json.loads(
            (run / "benchmark_case_receipt.json").read_text(encoding="utf-8"))
        observation = json.loads(
            (run / "adapter_observation.json").read_text(encoding="utf-8"))
        statuses.append(receipt["status"])
        assert receipt["usage_scope"] == "audited_lower_bound"
        assert receipt["adapter_sha256"] == suite["plan"]["runner"]["adapter_sha256"]
        assert observation["allowed"] == "visible"
        assert observation["blocked_visible"] is False
        assert observation["input_exists"] is True
        assert (run / "final_acceptance.json").is_file()
        assert (run / "resource_ledger.json").is_file()
    assert statuses.count("failed") == 1
    assert statuses.count("succeeded") == 15
    failed = json.loads(
        (results / "case-01" / "benchmark_case_receipt.json")
        .read_text(encoding="utf-8"))
    assert failed["returncode"] == 3 and failed["timed_out"] is False
    with pytest.raises(BenchmarkRunnerError, match="not empty"):
        run_benchmark_suite(
            suite_report=suite, public_root=public, plan_path=plan_path,
            results_root=results)
    results_file = tmp_path / "results-file"
    results_file.write_text("not a directory", encoding="utf-8")
    with pytest.raises(BenchmarkRunnerError, match="not a directory"):
        run_benchmark_suite(
            suite_report=suite, public_root=public, plan_path=plan_path,
            results_root=results_file)


def test_timeout_is_recorded_and_later_cases_still_run(tmp_path, monkeypatch):
    public, plan_path, _, suite = runner_fixture(
        tmp_path, monkeypatch, wall_seconds=1)
    monkeypatch.setenv("RC_TIMEOUT", "1")
    results = tmp_path / "timeout-results"
    manifest = run_benchmark_suite(
        suite_report=suite, public_root=public, plan_path=plan_path,
        results_root=results)
    first = json.loads(
        (results / "case-00" / "benchmark_case_receipt.json")
        .read_text(encoding="utf-8"))
    last = json.loads(
        (results / "case-15" / "benchmark_case_receipt.json")
        .read_text(encoding="utf-8"))
    assert first["status"] == "timed_out" and first["timed_out"] is True
    assert last["status"] == "succeeded"
    assert len(manifest["cases"]) == 16


def test_changed_frozen_adapter_is_rejected(tmp_path, monkeypatch):
    public, plan_path, _, suite = runner_fixture(tmp_path, monkeypatch)
    (public / "runner_adapter.py").write_text("changed", encoding="utf-8")
    with pytest.raises(BenchmarkRunnerError, match="changed"):
        run_benchmark_suite(
            suite_report=suite, public_root=public, plan_path=plan_path,
            results_root=tmp_path / "results")
