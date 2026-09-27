"""Evaluate completed ARC benchmark runs against private blind assessments."""
from __future__ import annotations

import argparse
import json
import math
import re
import uuid
from collections import defaultdict
from pathlib import Path

from researchclaw.pipeline.benchmark_suite import (
    BenchmarkSuiteError, open_sealed_benchmark_gold, verify_benchmark_suite,
)
from researchclaw.pipeline.evidence_store import content_hash, file_hash
from researchclaw.pipeline.evidence_signature import (
    EvidenceSignatureError, verify_document,
)
from researchclaw.pipeline.resource_ledger import (
    ResourceLedgerError, validate_resource_ledger,
)


_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_DISPOSITIONS = {"accept", "reject", "honest_negative"}


class BenchmarkEvaluationError(ValueError):
    pass


def _load(path: Path, name: str) -> dict:
    try:
        if path.stat().st_size > 10_000_000:
            raise BenchmarkEvaluationError(f"{name} exceeds 10 MB")
        value = json.loads(path.read_text(encoding="utf-8"))
    except BenchmarkEvaluationError:
        raise
    except (OSError, ValueError) as exc:
        raise BenchmarkEvaluationError(f"{name} is unreadable: {type(exc).__name__}") from exc
    if not isinstance(value, dict):
        raise BenchmarkEvaluationError(f"{name} must be an object")
    return value


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _load_stable(path: Path, name: str) -> tuple[dict, str]:
    try:
        before = file_hash(path)
        value = _load(path, name)
        after = file_hash(path)
    except OSError as exc:
        raise BenchmarkEvaluationError(f"{name} is unreadable: {type(exc).__name__}") from exc
    if before != after:
        raise BenchmarkEvaluationError(f"{name} changed while it was being evaluated")
    return value, before


def _safe_run(results_root: Path, relative: object) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise BenchmarkEvaluationError("Benchmark run path is malformed")
    candidate = results_root / relative
    path = candidate.resolve()
    if not path.is_relative_to(results_root) or candidate.is_symlink() or not path.is_dir():
        raise BenchmarkEvaluationError("Benchmark run path is missing or unsafe")
    return path


def _validate_acceptance(path: Path, digest: object) -> dict:
    if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
        raise BenchmarkEvaluationError("Acceptance digest is malformed")
    value = _load(path, "Final acceptance")
    ranks = {"exploratory": 0, "research_complete": 1, "submission_candidate": 2}
    if (file_hash(path) != digest or type(value.get("schema_version")) is not int
            or value["schema_version"] != 1
            or value.get("checker") != "final_acceptance/v1"
            or value.get("artifact_status") not in ranks
            or value.get("target_status") not in ranks
            or type(value.get("target_met")) is not bool
            or value["target_met"] != (
                ranks[value["artifact_status"]] >= ranks[value["target_status"]])):
        raise BenchmarkEvaluationError("Final acceptance is malformed or changed")
    return value


def _visible_usage(ledger: dict) -> tuple[int, int]:
    calls = tokens = 0
    for row in ledger["rows"]:
        if row.get("kind") not in {"llm_chat", "llm_embeddings", "image_generation"}:
            continue
        if type(row.get("calls")) is int:
            calls += row["calls"]
        if type(row.get("tokens")) is int:
            tokens += row["tokens"]
    return calls, tokens


def _validate_receipt(path: Path, digest: object, *, suite: dict,
                      case: dict, run: Path,
                      visible_calls: int, visible_tokens: int) -> dict:
    if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
        raise BenchmarkEvaluationError("Runner receipt digest is malformed")
    receipt = _load(path, "Runner receipt")
    legacy_fields = {"schema_version", "recorder", "suite_version", "case_id",
              "input_sha256", "budget", "started_at", "finished_at",
              "wall_seconds", "model_calls", "total_tokens"}
    extended_fields = legacy_fields | {
        "adapter_sha256", "status", "returncode", "timed_out", "usage_scope",
        "environment_names", "stdout_sha256", "stderr_sha256"}
    runner = suite["plan"].get("runner")
    expected_fields = extended_fields if runner is not None else legacy_fields
    if (set(receipt) != expected_fields or file_hash(path) != digest
            or type(receipt["schema_version"]) is not int or receipt["schema_version"] != 1
            or receipt["recorder"] != "arc-benchmark-runner/v1"
            or receipt["suite_version"] != suite["version"]
            or receipt["case_id"] != case["case_id"]
            or receipt["input_sha256"] != case["input_sha256"]
            or receipt["budget"] != suite["plan"]["budget"]):
        raise BenchmarkEvaluationError("Runner receipt identity is malformed or changed")
    started, finished, wall = (receipt[name] for name in
                               ("started_at", "finished_at", "wall_seconds"))
    if (any(type(value) not in {int, float} or not math.isfinite(value)
            for value in (started, finished, wall))
            or started < 0 or finished < started or wall < 0
            or not math.isclose(wall, finished - started, rel_tol=0, abs_tol=1e-6)
            or type(receipt["model_calls"]) is not int or receipt["model_calls"] < 0
            or type(receipt["total_tokens"]) is not int or receipt["total_tokens"] < 0):
        raise BenchmarkEvaluationError("Runner receipt usage is malformed or undercounts evidence")
    if runner is None:
        if (receipt["model_calls"] < visible_calls
                or receipt["total_tokens"] < visible_tokens):
            raise BenchmarkEvaluationError("Runner receipt usage is malformed or undercounts evidence")
        receipt["usage_scope"] = "runner_enforced"
        return receipt
    statuses = {"succeeded", "failed", "timed_out"}
    internal_environment = {"PYTHONIOENCODING", "ARC_BENCHMARK_CASE",
                            "ARC_BENCHMARK_INPUT", "ARC_BENCHMARK_RUN_DIR",
                            "ARC_BENCHMARK_BUDGET"}
    allowed_environment = (set(runner["environment_allowlist"]) | internal_environment
                           | {"PATH", "SYSTEMROOT", "WINDIR", "TEMP", "TMP"})
    names = receipt["environment_names"]
    status, returncode, timed_out = (
        receipt["status"], receipt["returncode"], receipt["timed_out"])
    stdout, stderr = run / "stdout.txt", run / "stderr.txt"
    if (receipt["adapter_sha256"] != runner["adapter_sha256"]
            or status not in statuses or type(returncode) is not int
            or type(timed_out) is not bool
            or (status == "succeeded") != (returncode == 0 and not timed_out)
            or (status == "timed_out") != timed_out
            or receipt["usage_scope"] != "audited_lower_bound"
            or not isinstance(names, list) or names != sorted(set(names))
            or any(not isinstance(name, str) or name not in allowed_environment
                   for name in names)
            or not stdout.is_file() or not stderr.is_file()
            or any(not isinstance(receipt[name], str)
                   or _SHA256.fullmatch(receipt[name]) is None
                   for name in ("stdout_sha256", "stderr_sha256"))
            or file_hash(stdout) != receipt["stdout_sha256"]
            or file_hash(stderr) != receipt["stderr_sha256"]):
        raise BenchmarkEvaluationError("Extended runner receipt is malformed or changed")
    valid_usage = (receipt["model_calls"] == visible_calls
                   and receipt["total_tokens"] == visible_tokens)
    if not valid_usage:
        raise BenchmarkEvaluationError("Runner receipt usage is malformed or undercounts evidence")
    return receipt


def _validate_results(suite: dict, results_root: Path, manifest: dict) -> dict[str, dict]:
    fields = {"schema_version", "suite_version", "cases"}
    attestation = suite["plan"].get("attestation")
    expected_fields = fields | {"version", "signature"} if attestation else fields
    allowed_fields = ({frozenset(expected_fields)} if attestation else
                      {frozenset(fields), frozenset(fields | {"version"})})
    if (frozenset(manifest) not in allowed_fields
            or type(manifest.get("schema_version")) is not int
            or manifest["schema_version"] != 1
            or manifest.get("suite_version") != suite["version"]
            or not isinstance(manifest.get("cases"), list)):
        raise BenchmarkEvaluationError("Benchmark result manifest is malformed")
    if ("version" in manifest
            and manifest["version"] != content_hash(
                {key: value for key, value in manifest.items()
                 if key not in {"version", "signature"}})):
        raise BenchmarkEvaluationError("Benchmark result manifest version is malformed")
    if attestation:
        try:
            verify_document(
                manifest, signer=attestation["runner"],
                purpose="benchmark_result_manifest/v1")
        except EvidenceSignatureError as exc:
            raise BenchmarkEvaluationError(str(exc)) from exc
    declared = {case["case_id"]: case for case in suite["plan"]["cases"]}
    results = {}
    seen_runs: set[Path] = set()
    fields = {"case_id", "run_dir", "receipt_sha256", "acceptance_sha256",
              "resource_ledger_sha256"}
    for item in manifest["cases"]:
        if (not isinstance(item, dict) or set(item) != fields
                or item.get("case_id") not in declared or item["case_id"] in results):
            raise BenchmarkEvaluationError("Benchmark result case is malformed")
        run = _safe_run(results_root, item["run_dir"])
        if run in seen_runs:
            raise BenchmarkEvaluationError("Benchmark run directory is reused across cases")
        seen_runs.add(run)
        resource_path = run / "resource_ledger.json"
        if (not isinstance(item["resource_ledger_sha256"], str)
                or _SHA256.fullmatch(item["resource_ledger_sha256"]) is None
                or not resource_path.is_file()
                or file_hash(resource_path) != item["resource_ledger_sha256"]):
            raise BenchmarkEvaluationError("Resource ledger is missing or changed")
        ledger = _load(resource_path, "Resource ledger")
        try:
            validate_resource_ledger(run, ledger)
        except (ResourceLedgerError, OSError, ValueError) as exc:
            raise BenchmarkEvaluationError(
                f"Resource ledger failed portable validation: {type(exc).__name__}") from exc
        visible_calls, visible_tokens = _visible_usage(ledger)
        acceptance = _validate_acceptance(
            run / "final_acceptance.json", item["acceptance_sha256"])
        receipt = _validate_receipt(
            run / "benchmark_case_receipt.json", item["receipt_sha256"],
            suite=suite, case=declared[item["case_id"]], run=run,
            visible_calls=visible_calls, visible_tokens=visible_tokens)
        results[item["case_id"]] = {
            "manifest": item, "acceptance": acceptance, "receipt": receipt,
            "visible_calls": visible_calls, "visible_tokens": visible_tokens,
        }
    if set(results) != set(declared):
        raise BenchmarkEvaluationError("Result manifest must cover every suite case exactly")
    return results


def _validate_assessment(value: dict, suite: dict, results: dict[str, dict]) -> dict[str, dict]:
    fields = {"schema_version", "suite_version", "assessor_id", "blinded", "cases"}
    attestation = suite["plan"].get("attestation")
    if attestation:
        fields.add("signature")
    if (set(value) != fields
            or type(value.get("schema_version")) is not int or value["schema_version"] != 1
            or value.get("suite_version") != suite["version"]
            or not isinstance(value.get("assessor_id"), str)
            or _SAFE_ID.fullmatch(value["assessor_id"]) is None
            or value.get("blinded") is not True or not isinstance(value.get("cases"), list)):
        raise BenchmarkEvaluationError("Private assessment is malformed or not blinded")
    if attestation:
        if value["assessor_id"] != attestation["assessor"]["key_id"]:
            raise BenchmarkEvaluationError("Private assessment signer identity differs")
        try:
            verify_document(
                value, signer=attestation["assessor"],
                purpose="benchmark_private_assessment/v1")
        except EvidenceSignatureError as exc:
            raise BenchmarkEvaluationError(str(exc)) from exc
    assessments = {}
    fields = {"case_id", "acceptance_sha256", "observed_disposition",
              "critical_errors", "revision_minutes", "revision_edits", "rationale"}
    for row in value["cases"]:
        if (not isinstance(row, dict) or set(row) != fields
                or row.get("case_id") not in results or row["case_id"] in assessments
                or row.get("acceptance_sha256") != results[row["case_id"]]["manifest"]["acceptance_sha256"]
                or row.get("observed_disposition") not in _DISPOSITIONS
                or type(row.get("critical_errors")) is not int
                or not 0 <= row["critical_errors"] <= 1000
                or type(row.get("revision_minutes")) not in {int, float}
                or not math.isfinite(row["revision_minutes"])
                or not 0 <= row["revision_minutes"] <= 100_000
                or type(row.get("revision_edits")) is not int
                or not 0 <= row["revision_edits"] <= 1_000_000
                or not isinstance(row.get("rationale"), str)
                or not 0 < len(row["rationale"]) <= 2000):
            raise BenchmarkEvaluationError("Private assessment case is malformed")
        assessments[row["case_id"]] = row
    if set(assessments) != set(results):
        raise BenchmarkEvaluationError("Private assessment must cover every result exactly")
    return assessments


def _aggregate(rows: list[dict]) -> dict:
    count = len(rows)
    reject = [row for row in rows if row["expected_disposition"] == "reject"]
    accept = [row for row in rows if row["expected_disposition"] == "accept"]
    negative = [row for row in rows if row["expected_disposition"] == "honest_negative"]
    return {
        "cases": count,
        "correct_dispositions": sum(row["correct"] for row in rows),
        "disposition_accuracy": (sum(row["correct"] for row in rows) / count if count else None),
        "error_acceptances": sum(row["error_acceptance"] for row in rows),
        "reject_cases": len(reject),
        "error_acceptance_rate": (sum(row["error_acceptance"] for row in reject) / len(reject)
                                  if reject else None),
        "accept_cases": len(accept),
        "false_rejections": sum(row["false_rejection"] for row in accept),
        "false_rejection_rate": (sum(row["false_rejection"] for row in accept) / len(accept)
                                 if accept else None),
        "honest_negative_cases": len(negative),
        "honest_negative_accuracy": (sum(row["correct"] for row in negative) / len(negative)
                                     if negative else None),
        "critical_errors": sum(row["critical_errors"] for row in rows),
        "mean_revision_minutes": (sum(row["revision_minutes"] for row in rows) / count
                                  if count else None),
        "mean_revision_edits": (sum(row["revision_edits"] for row in rows) / count
                                if count else None),
        "budget_exceeded": sum(row["budget_status"] == "exceeded" for row in rows),
        "budget_unverified": sum(row["budget_status"] == "unverified" for row in rows),
    }


def build_benchmark_evaluation(*, suite_report: dict, public_root: Path,
                               plan_path: Path, gold_path: Path | None,
                               results_root: Path, result_manifest_path: Path,
                               assessment_path: Path,
                               sealed_gold_path: Path | None = None,
                               gold_decryption_key_path: Path | None = None) -> dict:
    """Build a scored report from complete runs and a bound private assessment."""
    public_root = Path(public_root).resolve()
    plan_path = Path(plan_path).resolve()
    results_root = Path(results_root).resolve()
    result_manifest_path = Path(result_manifest_path).resolve()
    assessment_path = Path(assessment_path).resolve()
    if not result_manifest_path.is_relative_to(results_root):
        raise BenchmarkEvaluationError("Result manifest must be inside the results root")
    if assessment_path.is_relative_to(public_root):
        raise BenchmarkEvaluationError("Private assessment must be outside the public root")
    confidentiality = suite_report.get("plan", {}).get("confidentiality")
    try:
        if confidentiality is not None:
            if (gold_path is not None or sealed_gold_path is None
                    or gold_decryption_key_path is None):
                raise BenchmarkEvaluationError(
                    "Sealed suite requires only sealed gold and its decryption key")
            gold_document = open_sealed_benchmark_gold(
                suite_report, public_root, plan_path, sealed_gold_path,
                gold_decryption_key_path)
            gold_sha256 = suite_report["private_gold_sha256"]
        else:
            if (gold_path is None or sealed_gold_path is not None
                    or gold_decryption_key_path is not None):
                raise BenchmarkEvaluationError(
                    "Plaintext suite requires only the private gold path")
            gold_path = Path(gold_path).resolve()
            verify_benchmark_suite(suite_report, public_root, plan_path, gold_path)
            gold_document, gold_sha256 = _load_stable(gold_path, "Private gold")
            if gold_sha256 != suite_report["private_gold_sha256"]:
                raise BenchmarkEvaluationError(
                    "Private gold differs from the frozen suite")
    except BenchmarkSuiteError as exc:
        raise BenchmarkEvaluationError(str(exc)) from exc
    gold = {row["case_id"]: row for row in gold_document["cases"]}
    manifest_document, manifest_sha256 = _load_stable(
        result_manifest_path, "Result manifest")
    results = _validate_results(
        suite_report, results_root, manifest_document)
    assessment_document, assessment_sha256 = _load_stable(
        assessment_path, "Private assessment")
    assessments = _validate_assessment(assessment_document, suite_report, results)
    cases_by_id = {case["case_id"]: case for case in suite_report["plan"]["cases"]}
    budget = suite_report["plan"]["budget"]
    rows = []
    for case_id in sorted(results):
        case, result, assessment = cases_by_id[case_id], results[case_id], assessments[case_id]
        expected = gold[case_id]["expected_disposition"]
        observed = assessment["observed_disposition"]
        receipt = result["receipt"]
        exceeded = (receipt["wall_seconds"] > budget["wall_seconds"]
                    or receipt["model_calls"] > budget["model_calls"]
                    or receipt["total_tokens"] > budget["total_tokens"])
        budget_status = ("exceeded" if exceeded else "within"
                         if receipt["usage_scope"] == "runner_enforced"
                         else "unverified")
        rows.append({
            "case_id": case_id, "task_family": case["task_family"],
            "scenario": case["scenario"], "repeat_index": case["repeat_index"],
            "expected_disposition": expected, "observed_disposition": observed,
            "correct": observed == expected,
            "error_acceptance": expected == "reject" and observed == "accept",
            "false_rejection": expected == "accept" and observed != "accept",
            "critical_errors": assessment["critical_errors"],
            "revision_minutes": assessment["revision_minutes"],
            "revision_edits": assessment["revision_edits"],
            "artifact_status": result["acceptance"]["artifact_status"],
            "wall_seconds": receipt["wall_seconds"],
            "model_calls": receipt["model_calls"], "total_tokens": receipt["total_tokens"],
            "visible_ledger_calls": result["visible_calls"],
            "visible_ledger_tokens": result["visible_tokens"],
            "usage_scope": receipt["usage_scope"],
            "budget_status": budget_status,
            "assessment_rationale": assessment["rationale"],
            "receipt_sha256": result["manifest"]["receipt_sha256"],
            "acceptance_sha256": result["manifest"]["acceptance_sha256"],
            "resource_ledger_sha256": result["manifest"]["resource_ledger_sha256"],
        })
    by_family: dict[str, list] = defaultdict(list)
    by_scenario: dict[str, list] = defaultdict(list)
    for row in rows:
        by_family[row["task_family"]].append(row)
        by_scenario[row["scenario"]].append(row)
    attestation = suite_report["plan"].get("attestation")
    identity_limitation = (
        "Ed25519 signatures prove possession of frozen keys, not physical "
        "identity, key custody, remote execution, or assessor blinding."
        if attestation else
        "Runner and assessor identities are self-declared; no signatures or "
        "remote attestation are provided.")
    gold_limitation = (
        "Authenticated gold decryption proves ciphertext integrity and key "
        "possession, not organizational access control or memory erasure."
        if confidentiality else
        "Plaintext private-gold separation remains a filesystem layout contract, "
        "not proof of access control.")
    report = {
        "schema_version": 1, "checker": "arc-benchmark-evaluation/v1",
        "suite_version": suite_report["version"],
        "assessor": {"id": assessment_document["assessor_id"],
                     "blinded": True, "assessment_sha256": assessment_sha256,
                     "signature_verified": attestation is not None},
        "attestation": {
            "required": attestation is not None,
            "curator_key_id": (attestation.get("curator", {}).get("key_id")
                               if attestation else None),
            "runner_key_id": (attestation["runner"]["key_id"]
                              if attestation else None),
            "assessor_key_id": (attestation["assessor"]["key_id"]
                                if attestation else None),
            "suite_signature_verified": bool(
                attestation and "curator" in attestation),
            "result_and_assessment_signatures_verified": attestation is not None,
        },
        "result_manifest_sha256": manifest_sha256,
        "private_gold_sha256": gold_sha256,
        "gold_confidentiality": {
            "sealed": confidentiality is not None,
            "recipient_key_id": (confidentiality["key_id"]
                                 if confidentiality else None),
            "authenticated_decryption": confidentiality is not None,
        },
        "cases": rows,
        "overall": _aggregate(rows),
        "by_task_family": {name: _aggregate(group) for name, group in sorted(by_family.items())},
        "by_stress_scenario": {name: _aggregate(group)
                               for name, group in sorted(by_scenario.items())},
        "limitations": [
            identity_limitation,
            "Audited-lower-bound receipts cannot prove model-call or token-budget "
            "compliance; only observed excess is conclusive.",
            "Final acceptance is schema- and digest-bound here; its source bundle "
            "must retain normal portable verification.",
            "Revision minutes and edits are blinded assessor records, not "
            "independently timed UI events.",
            gold_limitation,
            "Finite-suite rates do not guarantee arbitrary-paper quality.",
        ],
    }
    report["version"] = content_hash(report)
    return report


def verify_benchmark_evaluation(report: dict, **inputs) -> dict:
    expected = build_benchmark_evaluation(**inputs)
    if not isinstance(report, dict) or report != expected:
        raise BenchmarkEvaluationError("Benchmark evaluation differs from frozen evidence")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate completed ARC benchmark runs")
    parser.add_argument("--suite-report", type=Path, required=True)
    parser.add_argument("--public-root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--private-gold", type=Path)
    parser.add_argument("--sealed-gold", type=Path)
    parser.add_argument("--gold-decryption-key", type=Path)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--result-manifest", type=Path, required=True)
    parser.add_argument("--private-assessment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    suite = _load(args.suite_report, "Suite report")
    report = build_benchmark_evaluation(
        suite_report=suite, public_root=args.public_root, plan_path=args.plan,
        gold_path=args.private_gold, results_root=args.results_root,
        result_manifest_path=args.result_manifest,
        assessment_path=args.private_assessment,
        sealed_gold_path=args.sealed_gold,
        gold_decryption_key_path=args.gold_decryption_key)
    output = args.output.resolve()
    protected = {path.resolve() for path in (
        args.suite_report, args.plan, args.private_gold, args.sealed_gold,
        args.gold_decryption_key, args.result_manifest, args.private_assessment)
                 if path is not None}
    if output in protected:
        raise BenchmarkEvaluationError("Evaluation output cannot overwrite frozen input evidence")
    _write(output, report)
    print(json.dumps({"cases": report["overall"]["cases"],
                      "disposition_accuracy": report["overall"]["disposition_accuracy"]},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
