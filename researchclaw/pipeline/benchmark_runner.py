"""Blind, one-attempt-per-case host runner for frozen ARC benchmark suites."""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import uuid
from collections import Counter
from pathlib import Path

from researchclaw.pipeline.benchmark_suite import (
    BenchmarkSuiteError, verify_public_benchmark_suite,
)
from researchclaw.pipeline.evidence_store import content_hash, file_hash
from researchclaw.pipeline.evidence_signature import (
    EvidenceSignatureError, sign_document,
)
from researchclaw.pipeline.final_acceptance import assess_delivery
from researchclaw.pipeline.resource_ledger import build_resource_ledger


class BenchmarkRunnerError(ValueError):
    pass


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


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


def _terminate_tree(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL, timeout=10, check=False)
        else:
            os.killpg(process.pid, signal.SIGKILL)
    except (OSError, subprocess.SubprocessError):
        process.kill()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()


def _environment(runner: dict, case: dict, suite: dict,
                 input_path: Path, run_dir: Path) -> dict[str, str]:
    names = set(runner["environment_allowlist"])
    names.update({"PATH", "SYSTEMROOT", "WINDIR", "TEMP", "TMP"})
    environment = {name: os.environ[name] for name in names if name in os.environ}
    environment.update({
        "PYTHONIOENCODING": "utf-8",
        "ARC_BENCHMARK_CASE": json.dumps(case, sort_keys=True),
        "ARC_BENCHMARK_INPUT": str(input_path),
        "ARC_BENCHMARK_RUN_DIR": str(run_dir),
        "ARC_BENCHMARK_BUDGET": json.dumps(suite["plan"]["budget"], sort_keys=True),
    })
    return environment


def run_benchmark_suite(*, suite_report: dict, public_root: Path,
                        plan_path: Path, results_root: Path,
                        python_executable: str | None = None,
                        signing_key_path: Path | None = None) -> dict:
    """Run every public case once without reading private gold or assessments."""
    root = Path(public_root).resolve()
    try:
        verify_public_benchmark_suite(suite_report, root, plan_path)
    except BenchmarkSuiteError as exc:
        raise BenchmarkRunnerError(str(exc)) from exc
    runner = suite_report["plan"].get("runner")
    if runner is None:
        raise BenchmarkRunnerError("Benchmark plan does not freeze a runner adapter")
    attestation = suite_report["plan"].get("attestation")
    if (attestation is None) != (signing_key_path is None):
        raise BenchmarkRunnerError(
            "Runner signing key must match the frozen attestation policy")
    if attestation is not None:
        try:
            sign_document(
                {}, signer=attestation["runner"],
                private_key_path=Path(signing_key_path),
                purpose="benchmark_result_manifest/v1")
        except EvidenceSignatureError as exc:
            raise BenchmarkRunnerError(str(exc)) from exc
    results_root = Path(results_root).resolve()
    if results_root.exists() and not results_root.is_dir():
        raise BenchmarkRunnerError("Benchmark results root is not a directory")
    if results_root.exists() and any(results_root.iterdir()):
        raise BenchmarkRunnerError("Benchmark results root is not empty; start a new run")
    results_root.mkdir(parents=True, exist_ok=True)
    adapter = (root / runner["adapter"]).resolve()
    if file_hash(adapter) != runner["adapter_sha256"]:
        raise BenchmarkRunnerError("Frozen runner adapter changed")
    executable = python_executable or sys.executable
    budget = suite_report["plan"]["budget"]
    manifest_cases = []
    for case in suite_report["plan"]["cases"]:
        run_dir = results_root / case["case_id"]
        run_dir.mkdir(parents=False, exist_ok=False)
        stdout_path, stderr_path = run_dir / "stdout.txt", run_dir / "stderr.txt"
        started = time.monotonic()
        timed_out = False
        returncode = -1
        status = "failed"
        creationflags = (subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0)
        environment = _environment(
            runner, case, suite_report,
            (root / case["input_bundle"]).resolve(), run_dir)
        with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
            try:
                process = subprocess.Popen(
                    [executable, "-I", str(adapter)], cwd=run_dir,
                    env=environment, stdin=subprocess.DEVNULL,
                    stdout=stdout, stderr=stderr,
                    creationflags=creationflags,
                    start_new_session=os.name != "nt")
                try:
                    returncode = process.wait(timeout=budget["wall_seconds"])
                    status = "succeeded" if returncode == 0 else "failed"
                except subprocess.TimeoutExpired:
                    timed_out = True
                    status = "timed_out"
                    _terminate_tree(process)
                    returncode = (process.returncode
                                  if process.returncode is not None else -1)
            except OSError as exc:
                stderr.write(
                    f"benchmark runner could not start adapter: {type(exc).__name__}\n"
                    .encode("utf-8"))
        finished = time.monotonic()
        acceptance = assess_delivery(run_dir, target_status="research_complete")
        _write(run_dir / "final_acceptance.json", acceptance)
        ledger = build_resource_ledger(run_dir)
        _write(run_dir / "resource_ledger.json", ledger)
        calls, tokens = _visible_usage(ledger)
        receipt = {
            "schema_version": 1, "recorder": "arc-benchmark-runner/v1",
            "suite_version": suite_report["version"], "case_id": case["case_id"],
            "input_sha256": case["input_sha256"], "budget": budget,
            "adapter_sha256": runner["adapter_sha256"],
            "started_at": started, "finished_at": finished,
            "wall_seconds": finished - started,
            "status": status, "returncode": returncode, "timed_out": timed_out,
            "usage_scope": "audited_lower_bound",
            "environment_names": sorted(environment),
            "model_calls": calls, "total_tokens": tokens,
            "stdout_sha256": file_hash(stdout_path),
            "stderr_sha256": file_hash(stderr_path),
        }
        receipt_path = run_dir / "benchmark_case_receipt.json"
        _write(receipt_path, receipt)
        manifest_cases.append({
            "case_id": case["case_id"],
            "run_dir": run_dir.relative_to(results_root).as_posix(),
            "receipt_sha256": file_hash(receipt_path),
            "acceptance_sha256": file_hash(run_dir / "final_acceptance.json"),
            "resource_ledger_sha256": file_hash(run_dir / "resource_ledger.json"),
        })
    manifest = {"schema_version": 1, "suite_version": suite_report["version"],
                "cases": manifest_cases}
    manifest["version"] = content_hash(manifest)
    if attestation is not None:
        try:
            manifest["signature"] = sign_document(
                manifest, signer=attestation["runner"],
                private_key_path=Path(signing_key_path),
                purpose="benchmark_result_manifest/v1")
        except EvidenceSignatureError as exc:
            raise BenchmarkRunnerError(str(exc)) from exc
    _write(results_root / "benchmark_result_manifest.json", manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a frozen public ARC benchmark suite")
    parser.add_argument("--suite-report", type=Path, required=True)
    parser.add_argument("--public-root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--signing-key", type=Path)
    args = parser.parse_args(argv)
    suite = json.loads(args.suite_report.read_text(encoding="utf-8"))
    manifest = run_benchmark_suite(
        suite_report=suite, public_root=args.public_root,
        plan_path=args.plan, results_root=args.results_root,
        signing_key_path=args.signing_key)
    statuses = Counter(
        json.loads((args.results_root / case["run_dir"] / "benchmark_case_receipt.json")
                   .read_text(encoding="utf-8"))["status"] for case in manifest["cases"])
    print(json.dumps(dict(statuses), sort_keys=True))
    return 0 if set(statuses) <= {"succeeded"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
