"""Freeze a coverage-complete, blind fixed-budget ARC benchmark suite."""
from __future__ import annotations

import argparse
import json
import re
import uuid
from collections import Counter
from pathlib import Path

from researchclaw.pipeline.evidence_store import content_hash, file_hash


TASK_FAMILIES = (
    "tabular_classification", "tabular_regression", "temporal", "image",
)
STRESS_SCENARIOS = (
    "leakage_trap", "invalid_idea", "null_or_negative_result",
    "external_download_failure", "incorrect_proof", "image_api_failure",
    "missing_template", "long_manuscript",
)
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_RUNNER_RESERVED_ENV = {
    "PYTHONIOENCODING", "ARC_BENCHMARK_CASE", "ARC_BENCHMARK_INPUT",
    "ARC_BENCHMARK_RUN_DIR", "ARC_BENCHMARK_BUDGET",
}
_SCOPE = (
    "coverage and blind-input fixture for fixed-budget model evaluation; "
    "no model outcomes are claimed"
)
_LIMITATIONS = [
    "Private-gold separation is a filesystem layout contract, not access-control proof.",
    "Ready means coverage-complete inputs; it does not mean any model was evaluated.",
    "Finite benchmark results cannot guarantee arbitrary-paper quality.",
]


class BenchmarkSuiteError(ValueError):
    pass


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _load(path: Path, name: str) -> dict:
    try:
        if path.stat().st_size > 10_000_000:
            raise BenchmarkSuiteError(f"{name} exceeds 10 MB")
        value = json.loads(path.read_text(encoding="utf-8"))
    except BenchmarkSuiteError:
        raise
    except (OSError, ValueError) as exc:
        raise BenchmarkSuiteError(f"{name} is unreadable: {type(exc).__name__}") from exc
    if not isinstance(value, dict):
        raise BenchmarkSuiteError(f"{name} must be an object")
    return value


def _validate_budget(value: object) -> dict:
    fields = {"wall_seconds", "model_calls", "total_tokens"}
    if not isinstance(value, dict) or set(value) != fields:
        raise BenchmarkSuiteError("Benchmark budget is malformed")
    limits = {"wall_seconds": 86_400, "model_calls": 1_000,
              "total_tokens": 10_000_000}
    if any(type(value[name]) is not int or not 1 <= value[name] <= limit
           for name, limit in limits.items()):
        raise BenchmarkSuiteError("Benchmark budget is outside bounded limits")
    return dict(value)


def _validate_runner(root: Path, value: object) -> dict:
    fields = {"adapter", "adapter_sha256", "environment_allowlist"}
    if not isinstance(value, dict) or set(value) != fields:
        raise BenchmarkSuiteError("Benchmark runner fields are malformed")
    adapter = _safe_input(root, value["adapter"], value["adapter_sha256"])
    allowlist = value["environment_allowlist"]
    if (not isinstance(allowlist, list) or len(allowlist) > 32
            or any(not isinstance(name, str)
                   or re.fullmatch(r"[A-Z][A-Z0-9_]{0,63}", name) is None
                   for name in allowlist)
            or len(set(allowlist)) != len(allowlist)
            or set(allowlist) & _RUNNER_RESERVED_ENV):
        raise BenchmarkSuiteError("Benchmark runner environment allowlist is malformed")
    return {"adapter": adapter, "adapter_sha256": value["adapter_sha256"],
            "environment_allowlist": allowlist}


def _safe_input(root: Path, relative: object, digest: object) -> str:
    if (not isinstance(relative, str) or not relative or "\\" in relative
            or not isinstance(digest, str) or _SHA256.fullmatch(digest) is None):
        raise BenchmarkSuiteError("Benchmark input reference is malformed")
    candidate = root / relative
    path = candidate.resolve()
    if (not path.is_relative_to(root) or candidate.is_symlink() or not path.is_file()
            or path.stat().st_size > 10_000_000 or file_hash(path) != digest):
        raise BenchmarkSuiteError("Benchmark input is missing, unsafe, oversized or changed")
    return path.relative_to(root).as_posix()


def _validate_plan(root: Path, plan: dict) -> tuple[dict, dict]:
    required = {"schema_version", "min_repeats", "budget", "cases"}
    if not required <= set(plan) or set(plan) - (required | {"runner"}):
        raise BenchmarkSuiteError("Benchmark plan fields are malformed")
    if type(plan["schema_version"]) is not int or plan["schema_version"] != 1:
        raise BenchmarkSuiteError("Unsupported benchmark plan schema")
    minimum = plan["min_repeats"]
    if type(minimum) is not int or not 2 <= minimum <= 20:
        raise BenchmarkSuiteError("Benchmark min_repeats must be in [2, 20]")
    budget = _validate_budget(plan["budget"])
    cases = plan["cases"]
    if not isinstance(cases, list) or not cases or len(cases) > 512:
        raise BenchmarkSuiteError("Benchmark cases are malformed or exceed 512")
    normalized, seen_ids, seen_repeats = [], set(), set()
    family_counts: Counter[str] = Counter()
    scenario_counts: Counter[str] = Counter()
    fields = {"case_id", "task_family", "scenario", "repeat_index",
              "input_bundle", "input_sha256"}
    for case in cases:
        if not isinstance(case, dict) or set(case) != fields:
            raise BenchmarkSuiteError("Benchmark case fields are malformed")
        case_id = case["case_id"]
        repeat = case["repeat_index"]
        if (not isinstance(case_id, str) or _SAFE_ID.fullmatch(case_id) is None
                or case_id in seen_ids or case["task_family"] not in TASK_FAMILIES
                or case["scenario"] not in STRESS_SCENARIOS
                or type(repeat) is not int or not 0 <= repeat < 20):
            raise BenchmarkSuiteError("Benchmark case identity is malformed")
        repeat_key = (case["task_family"], case["scenario"], repeat)
        if repeat_key in seen_repeats:
            raise BenchmarkSuiteError("Benchmark repeat identity is duplicated")
        seen_ids.add(case_id)
        seen_repeats.add(repeat_key)
        family_counts[case["task_family"]] += 1
        scenario_counts[case["scenario"]] += 1
        normalized.append({**case, "input_bundle": _safe_input(
            root, case["input_bundle"], case["input_sha256"])})
    missing_families = [name for name in TASK_FAMILIES if family_counts[name] < minimum]
    missing_scenarios = [name for name in STRESS_SCENARIOS if scenario_counts[name] < minimum]
    if missing_families or missing_scenarios:
        raise BenchmarkSuiteError("Benchmark category coverage is incomplete")
    coverage = {
        "task_families": {name: family_counts[name] for name in TASK_FAMILIES},
        "stress_scenarios": {name: scenario_counts[name] for name in STRESS_SCENARIOS},
        "minimum_repeats": minimum,
        "case_count": len(normalized),
    }
    document = {"schema_version": 1, "min_repeats": minimum,
                "budget": budget, "cases": normalized}
    if "runner" in plan:
        document["runner"] = _validate_runner(root, plan["runner"])
    return document, coverage


def _suite_report(plan: dict, coverage: dict, plan_sha256: str,
                  private_gold_sha256: str) -> dict:
    report = {
        "schema_version": 1,
        "checker": "arc-benchmark-suite/v1",
        "status": "ready",
        "scope": _SCOPE,
        "plan": plan,
        "plan_sha256": plan_sha256,
        "private_gold_sha256": private_gold_sha256,
        "coverage": coverage,
        "limitations": list(_LIMITATIONS),
    }
    report["version"] = content_hash(report)
    return report


def _validate_gold(gold: dict, case_ids: set[str]) -> dict:
    if (set(gold) != {"schema_version", "cases"}
            or type(gold.get("schema_version")) is not int or gold["schema_version"] != 1):
        raise BenchmarkSuiteError("Private gold fields are malformed")
    cases = gold["cases"]
    if not isinstance(cases, list):
        raise BenchmarkSuiteError("Private gold cases are malformed")
    seen = set()
    for case in cases:
        if (not isinstance(case, dict)
                or set(case) != {"case_id", "expected_disposition", "rubric"}
                or case.get("case_id") in seen
                or case.get("case_id") not in case_ids
                or case.get("expected_disposition") not in {"accept", "reject", "honest_negative"}
                or not isinstance(case.get("rubric"), str)
                or not 0 < len(case["rubric"]) <= 2000):
            raise BenchmarkSuiteError("Private gold case is malformed")
        seen.add(case["case_id"])
    if seen != case_ids:
        raise BenchmarkSuiteError("Private gold must cover every public case exactly")
    return gold


def freeze_benchmark_suite(public_root: Path, plan_path: Path, gold_path: Path,
                           output: Path | None = None) -> dict:
    """Validate public cases and private gold, freezing only a gold digest."""
    root = Path(public_root).resolve()
    plan_path, gold_path = Path(plan_path).resolve(), Path(gold_path).resolve()
    if not root.is_dir() or not plan_path.is_relative_to(root):
        raise BenchmarkSuiteError("Public plan must be inside the public suite root")
    if gold_path.is_relative_to(root):
        raise BenchmarkSuiteError("Private gold must be outside the public suite root")
    plan, coverage = _validate_plan(root, _load(plan_path, "Benchmark plan"))
    gold = _validate_gold(_load(gold_path, "Private gold"),
                          {case["case_id"] for case in plan["cases"]})
    report = _suite_report(plan, coverage, file_hash(plan_path), file_hash(gold_path))
    # The validated gold is intentionally not copied into this public artifact.
    _ = gold
    if output is not None:
        destination = Path(output).resolve()
        if not destination.is_relative_to(root):
            raise BenchmarkSuiteError("Public suite report must stay inside the public root")
        protected = {plan_path}
        protected.update((root / case["input_bundle"]).resolve()
                         for case in plan["cases"])
        if "runner" in plan:
            protected.add((root / plan["runner"]["adapter"]).resolve())
        if destination in protected:
            raise BenchmarkSuiteError("Public suite report cannot overwrite a frozen input")
        _write(destination, report)
    return report


def verify_public_benchmark_suite(report: dict, public_root: Path,
                                  plan_path: Path) -> dict:
    """Verify every public field without reading or locating private gold."""
    root, plan_path = Path(public_root).resolve(), Path(plan_path).resolve()
    if (not isinstance(report, dict) or not root.is_dir()
            or not plan_path.is_relative_to(root)
            or not isinstance(report.get("private_gold_sha256"), str)
            or _SHA256.fullmatch(report["private_gold_sha256"]) is None):
        raise BenchmarkSuiteError("Public benchmark suite report is malformed")
    plan, coverage = _validate_plan(root, _load(plan_path, "Benchmark plan"))
    expected = _suite_report(
        plan, coverage, file_hash(plan_path), report["private_gold_sha256"])
    if report != expected:
        raise BenchmarkSuiteError("Public benchmark suite differs from frozen inputs")
    return report


def verify_benchmark_suite(report: dict, public_root: Path,
                           plan_path: Path, gold_path: Path) -> dict:
    """Rebuild the public suite report from current public and private inputs."""
    expected = freeze_benchmark_suite(public_root, plan_path, gold_path)
    if not isinstance(report, dict) or report != expected:
        raise BenchmarkSuiteError("Benchmark suite report differs from frozen inputs")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Freeze a blind ARC benchmark suite")
    parser.add_argument("public_root", type=Path)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--private-gold", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = freeze_benchmark_suite(
        args.public_root, args.plan, args.private_gold, args.output)
    print(json.dumps({"status": report["status"],
                      "case_count": report["coverage"]["case_count"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
