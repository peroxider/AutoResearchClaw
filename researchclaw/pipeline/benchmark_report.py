"""Fixed-budget benchmark report: error acceptance, cost, revision proxies.

A benchmark manifest declares cases over prepared run directories. A defect
case seeds one known flaw, names the acceptance dimension expected to catch
it, and pins the run's frozen ``final_acceptance.json`` digest; a control
case seeds nothing. The report classifies each run from that frozen verdict
— detected (the expected dimension failed acceptance), rejected for another
reason, or the defect was accepted when the run still declared
research_complete — and aggregates per-case cost and revision proxies from
the run's own validated ``resource_ledger.json``.

Honest boundaries: rates describe only the declared finite case set and are
never a guarantee for arbitrary papers; costs are the auditable lower bound
the run's ledgers can prove; revision proxies count recorded machine
attempts and review calls, not human edits. Cases whose evidence is
missing, tampered with, or malformed are case errors excluded from rates —
never silently treated as detections or zeros.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from researchclaw.pipeline.evidence_store import content_hash, file_hash
from researchclaw.pipeline.final_acceptance import (
    PRESENTATION_DIMENSIONS,
    RESEARCH_DIMENSIONS,
)

BENCHMARK_DIMENSIONS = (*RESEARCH_DIMENSIONS, *PRESENTATION_DIMENSIONS,
                        "resources", "anonymity", "integrity")
_STATUS_RANK = {"exploratory": 0, "research_complete": 1, "submission_candidate": 2}


class BenchmarkReportError(ValueError):
    pass


def _validate_manifest(manifest: Any) -> list[dict]:
    if (not isinstance(manifest, dict) or manifest.get("schema_version") != 1
            or not isinstance(manifest.get("cases"), list) or not manifest["cases"]):
        raise BenchmarkReportError("Benchmark manifest is malformed")
    seen: set[str] = set()
    for case in manifest["cases"]:
        if (not isinstance(case, dict) or not isinstance(case.get("case_id"), str)
                or not case["case_id"] or case["case_id"] in seen
                or case.get("kind") not in ("defect", "control")
                or not isinstance(case.get("run_dir"), str) or not case["run_dir"]
                or not isinstance(case.get("acceptance_sha256"), str)
                or not case["acceptance_sha256"]):
            raise BenchmarkReportError("Benchmark manifest is malformed")
        seen.add(case["case_id"])
        if case["kind"] == "defect":
            if (not isinstance(case.get("defect_category"), str)
                    or not case["defect_category"]
                    or case.get("expected_dimension") not in BENCHMARK_DIMENSIONS):
                raise BenchmarkReportError("Benchmark manifest is malformed")
        elif "defect_category" in case or "expected_dimension" in case:
            raise BenchmarkReportError("Benchmark manifest is malformed")
    return manifest["cases"]


def _evaluate_case(case: dict, benchmark_root: Path) -> tuple[dict | None, str | None]:
    """Classify one case from its frozen acceptance verdict."""
    path = benchmark_root / case["run_dir"] / "final_acceptance.json"
    if not path.is_file():
        return None, "missing_final_acceptance"
    try:
        verdict = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return None, f"unreadable_final_acceptance:{type(exc).__name__}"
    if (not isinstance(verdict, dict) or verdict.get("schema_version") != 1
            or not isinstance(verdict.get("checker"), str)
            or not verdict["checker"].startswith("final_acceptance/")
            or verdict.get("artifact_status") not in _STATUS_RANK
            or not isinstance(verdict.get("issues"), list)
            or any(not isinstance(entry, dict)
                   or not isinstance(entry.get("dimension"), str)
                   for entry in verdict["issues"])):
        return None, "malformed_final_acceptance"
    if file_hash(path) != case["acceptance_sha256"]:
        return None, "acceptance_digest_mismatch"
    # Research-level acceptance failed: the run stayed exploratory.
    rejected = _STATUS_RANK[verdict["artifact_status"]] == 0
    issue_dimensions = sorted({entry["dimension"] for entry in verdict["issues"]})
    result = {"case_id": case["case_id"], "kind": case["kind"],
              "run_dir": case["run_dir"],
              "artifact_status": verdict["artifact_status"],
              "issue_dimensions": issue_dimensions}
    if case["kind"] == "defect":
        result["defect_category"] = case["defect_category"]
        result["expected_dimension"] = case["expected_dimension"]
        if not rejected:
            result["outcome"] = "accepted_defect"
        elif case["expected_dimension"] in issue_dimensions:
            result["outcome"] = "detected"
        else:
            # The run failed for unrelated reasons; the seeded defect itself
            # was neither caught nor accepted.
            result["outcome"] = "rejected_other_reason"
    else:
        result["outcome"] = "control_rejected" if rejected else "control_passed"
    return result, None


def _case_cost(run_dir: Path) -> tuple[dict | None, str | None]:
    """Cost and revision proxies from the run's validated resource ledger."""
    path = run_dir / "resource_ledger.json"
    if not path.is_file():
        return None, "missing_resource_ledger"
    try:
        ledger = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return None, f"unreadable_resource_ledger:{type(exc).__name__}"
    from researchclaw.pipeline.resource_ledger import (
        ResourceLedgerError,
        validate_resource_ledger,
    )

    try:
        validate_resource_ledger(run_dir, ledger)
    except (ResourceLedgerError, OSError, ValueError) as exc:
        return None, f"resource_ledger_mismatch:{type(exc).__name__}"
    cost: dict[str, int | None] = {"llm_chat_calls": None, "llm_chat_failures": None,
                                   "total_tokens": None, "llm_review_calls": None,
                                   "writing_attempts": None}

    def _add(field: str, value: Any) -> None:
        if type(value) is int:
            cost[field] = (cost[field] or 0) + value

    for row in ledger.get("rows", []):
        if not isinstance(row, dict):
            continue
        kind = row.get("kind")
        if kind == "llm_chat":
            _add("llm_chat_calls", row.get("calls"))
            _add("llm_chat_failures", row.get("failures"))
            _add("total_tokens", row.get("tokens"))
        elif kind == "llm_review":
            _add("llm_review_calls", row.get("calls"))
        elif kind == "writing_attempts":
            _add("writing_attempts", row.get("calls"))
    return cost, None


def _int_sum(values: list[Any]) -> int | None:
    if any(value is not None and type(value) is not int for value in values):
        return None
    if not any(value is not None for value in values):
        return None
    return sum(value for value in values if value is not None)


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def build_benchmark_report(benchmark_root: Path) -> dict:
    """Derive the report from benchmark_manifest.json and the frozen runs."""
    path = benchmark_root / "benchmark_manifest.json"
    if not path.is_file():
        raise BenchmarkReportError("Benchmark manifest is missing")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise BenchmarkReportError(
            f"Benchmark manifest is unreadable: {exc}") from exc
    cases = _validate_manifest(manifest)

    case_errors: list[dict] = []
    results: list[dict] = []
    for case in cases:
        run_dir = benchmark_root / case["run_dir"]
        result, problem = _evaluate_case(case, benchmark_root)
        if result is None:
            case_errors.append({"case_id": case["case_id"], "reason": problem})
            continue
        cost, cost_problem = _case_cost(run_dir)
        if cost is None:
            case_errors.append({"case_id": case["case_id"], "reason": cost_problem})
            continue
        result["cost"] = cost
        results.append(result)

    defect = [r for r in results if r["kind"] == "defect"]
    control = [r for r in results if r["kind"] == "control"]
    accepted = sum(1 for r in defect if r["outcome"] == "accepted_defect")
    detected = sum(1 for r in defect if r["outcome"] == "detected")
    rejected_other = sum(1 for r in defect if r["outcome"] == "rejected_other_reason")
    controls_rejected = sum(1 for r in control if r["outcome"] == "control_rejected")
    report = {
        "schema_version": 1,
        "cases": results,
        "case_errors": case_errors,
        "totals": {
            "defect_cases": len(defect),
            "detected": detected,
            "accepted_defects": accepted,
            "rejected_other_reason": rejected_other,
            "error_acceptance_rate": _rate(accepted, len(defect)),
            "detection_rate": _rate(detected, len(defect)),
            "control_cases": len(control),
            "controls_rejected": controls_rejected,
            "control_rejection_rate": _rate(controls_rejected, len(control)),
            "total_tokens": _int_sum([r["cost"]["total_tokens"] for r in results]),
            "llm_chat_calls": _int_sum([r["cost"]["llm_chat_calls"] for r in results]),
            "llm_chat_failures": _int_sum(
                [r["cost"]["llm_chat_failures"] for r in results]),
            "llm_review_calls": _int_sum(
                [r["cost"]["llm_review_calls"] for r in results]),
            "writing_attempts": _int_sum(
                [r["cost"]["writing_attempts"] for r in results]),
        },
    }
    report["version"] = content_hash(report)
    return report


def validate_benchmark_report(benchmark_root: Path, report: dict) -> None:
    """Fail closed unless the report re-derives from the frozen evidence."""
    if not isinstance(report, dict) or report.get("schema_version") != 1:
        raise BenchmarkReportError("Benchmark report is malformed")
    payload = {key: value for key, value in report.items() if key != "version"}
    if report.get("version") != content_hash(payload):
        raise BenchmarkReportError("Benchmark report version changed")
    if report != build_benchmark_report(benchmark_root):
        raise BenchmarkReportError(
            "Benchmark report differs from the frozen artifacts")


def write_benchmark_report(benchmark_root: Path) -> dict:
    report = build_benchmark_report(benchmark_root)
    (benchmark_root / "benchmark_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    return report
