from __future__ import annotations

import copy
import json

import pytest

from researchclaw.pipeline.benchmark_suite import (
    STRESS_SCENARIOS, TASK_FAMILIES, BenchmarkSuiteError,
    freeze_benchmark_suite, verify_benchmark_suite, verify_public_benchmark_suite,
)
from researchclaw.pipeline.evidence_store import file_hash


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def suite_fixture(tmp_path):
    public = tmp_path / "public"
    public.mkdir()
    cases = []
    gold_cases = []
    for index in range(16):
        case_id = f"case-{index:02d}"
        input_path = public / "inputs" / f"{case_id}.json"
        write_json(input_path, {"case": case_id, "prompt": "public task only"})
        cases.append({
            "case_id": case_id,
            "task_family": TASK_FAMILIES[index % len(TASK_FAMILIES)],
            "scenario": STRESS_SCENARIOS[index // 2],
            "repeat_index": index % 2,
            "input_bundle": input_path.relative_to(public).as_posix(),
            "input_sha256": file_hash(input_path),
        })
        gold_cases.append({
            "case_id": case_id,
            "expected_disposition": ("accept" if index == 0 else
                                     "honest_negative" if index == 4 else "reject"),
            "rubric": f"Private gold rubric for {case_id}",
        })
    plan = {"schema_version": 1, "min_repeats": 2,
            "budget": {"wall_seconds": 300, "model_calls": 20,
                       "total_tokens": 100_000},
            "cases": cases}
    plan_path = public / "benchmark_plan.json"
    gold_path = tmp_path / "private" / "gold.json"
    write_json(plan_path, plan)
    write_json(gold_path, {"schema_version": 1, "cases": gold_cases})
    return public, plan_path, gold_path, plan


def test_coverage_complete_blind_suite_freezes_without_gold_content(tmp_path):
    public, plan_path, gold_path, _ = suite_fixture(tmp_path)
    output = public / "benchmark_suite.json"
    report = freeze_benchmark_suite(public, plan_path, gold_path, output)
    assert report["status"] == "ready" and report["coverage"]["case_count"] == 16
    assert set(report["coverage"]["task_families"]) == set(TASK_FAMILIES)
    assert set(report["coverage"]["stress_scenarios"]) == set(STRESS_SCENARIOS)
    assert all(value >= 2 for value in report["coverage"]["task_families"].values())
    assert all(value >= 2 for value in report["coverage"]["stress_scenarios"].values())
    public_text = output.read_text(encoding="utf-8")
    assert "Private gold rubric" not in public_text
    assert str(gold_path) not in public_text
    assert verify_benchmark_suite(report, public, plan_path, gold_path) == report


def test_missing_repeat_or_category_fails_closed(tmp_path):
    public, plan_path, gold_path, plan = suite_fixture(tmp_path)
    plan["cases"].pop()
    write_json(plan_path, plan)
    with pytest.raises(BenchmarkSuiteError, match="coverage is incomplete"):
        freeze_benchmark_suite(public, plan_path, gold_path)


def test_private_gold_must_be_separate_and_cover_public_cases(tmp_path):
    public, plan_path, gold_path, _ = suite_fixture(tmp_path)
    inside = public / "gold.json"
    inside.write_bytes(gold_path.read_bytes())
    with pytest.raises(BenchmarkSuiteError, match="outside"):
        freeze_benchmark_suite(public, plan_path, inside)
    gold = json.loads(gold_path.read_text(encoding="utf-8"))
    gold["cases"].pop()
    write_json(gold_path, gold)
    with pytest.raises(BenchmarkSuiteError, match="every public case"):
        freeze_benchmark_suite(public, plan_path, gold_path)


def test_public_input_hash_and_safe_path_are_enforced(tmp_path):
    public, plan_path, gold_path, plan = suite_fixture(tmp_path)
    victim = public / plan["cases"][0]["input_bundle"]
    victim.write_text("changed", encoding="utf-8")
    with pytest.raises(BenchmarkSuiteError, match="changed"):
        freeze_benchmark_suite(public, plan_path, gold_path)
    plan["cases"][0]["input_bundle"] = "../private/gold.json"
    plan["cases"][0]["input_sha256"] = file_hash(gold_path)
    write_json(plan_path, plan)
    with pytest.raises(BenchmarkSuiteError, match="unsafe"):
        freeze_benchmark_suite(public, plan_path, gold_path)


@pytest.mark.parametrize("mutate", [
    lambda p: p["budget"].update(wall_seconds=True),
    lambda p: p["budget"].update(model_calls=0),
    lambda p: p.update(min_repeats=1),
    lambda p: p["cases"][0].update(case_id="../escape"),
    lambda p: p["cases"][1].update(
        task_family=p["cases"][0]["task_family"],
        scenario=p["cases"][0]["scenario"],
        repeat_index=p["cases"][0]["repeat_index"]),
])
def test_malformed_suite_plan_is_rejected(tmp_path, mutate):
    public, plan_path, gold_path, plan = suite_fixture(tmp_path)
    mutate(plan)
    write_json(plan_path, plan)
    with pytest.raises(BenchmarkSuiteError):
        freeze_benchmark_suite(public, plan_path, gold_path)


def test_report_is_recomputed_after_self_consistent_tamper(tmp_path):
    public, plan_path, gold_path, _ = suite_fixture(tmp_path)
    report = freeze_benchmark_suite(public, plan_path, gold_path)
    changed = copy.deepcopy(report)
    changed["coverage"]["case_count"] = 99
    from researchclaw.pipeline.evidence_store import content_hash
    changed["version"] = content_hash({key: value for key, value in changed.items()
                                       if key != "version"})
    with pytest.raises(BenchmarkSuiteError, match="differs"):
        verify_benchmark_suite(changed, public, plan_path, gold_path)
    with pytest.raises(BenchmarkSuiteError, match="overwrite"):
        freeze_benchmark_suite(public, plan_path, gold_path, plan_path)


def test_public_runner_identity_is_frozen_without_exposing_private_gold(tmp_path):
    public, plan_path, gold_path, plan = suite_fixture(tmp_path)
    adapter = public / "runner_adapter.py"
    adapter.write_text("print('fixture adapter')\n", encoding="utf-8")
    plan["runner"] = {
        "adapter": "runner_adapter.py", "adapter_sha256": file_hash(adapter),
        "environment_allowlist": ["OPENAI_API_KEY"],
    }
    write_json(plan_path, plan)
    report = freeze_benchmark_suite(public, plan_path, gold_path)
    assert verify_public_benchmark_suite(report, public, plan_path) == report
    gold_path.write_text("changed after public freeze", encoding="utf-8")
    # Public runner verification never reads or locates private gold.
    assert verify_public_benchmark_suite(report, public, plan_path) == report
    with pytest.raises(BenchmarkSuiteError):
        verify_benchmark_suite(report, public, plan_path, gold_path)


@pytest.mark.parametrize("runner", [
    {"adapter": "missing.py", "adapter_sha256": "0" * 64,
     "environment_allowlist": []},
    {"adapter": "runner_adapter.py", "adapter_sha256": "0" * 64,
     "environment_allowlist": ["bad-name"]},
    {"adapter": "runner_adapter.py", "adapter_sha256": "0" * 64,
     "environment_allowlist": ["TOKEN", "TOKEN"]},
    {"adapter": "runner_adapter.py", "adapter_sha256": "0" * 64,
     "environment_allowlist": ["ARC_BENCHMARK_CASE"]},
])
def test_runner_identity_and_environment_allowlist_fail_closed(tmp_path, runner):
    public, plan_path, gold_path, plan = suite_fixture(tmp_path)
    adapter = public / "runner_adapter.py"
    adapter.write_text("print('fixture')", encoding="utf-8")
    runner["adapter_sha256"] = (file_hash(adapter)
                                if runner["adapter"] == "runner_adapter.py" else "0" * 64)
    plan["runner"] = runner
    write_json(plan_path, plan)
    with pytest.raises(BenchmarkSuiteError):
        freeze_benchmark_suite(public, plan_path, gold_path)
