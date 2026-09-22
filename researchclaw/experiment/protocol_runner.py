"""Host-owned, resumable execution of a frozen test matrix.

The ledger measures formal matrix invocations only. It does not measure LLM
calls, earlier code generation trials, hidden in-script tuning, or OS isolation.
Test metrics are released only after every required cell has finished.
"""
from __future__ import annotations

import json
import math
import os
import platform
import shutil
import time
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

from researchclaw.pipeline.evidence_store import EvidenceStore, content_hash, file_hash
from researchclaw.pipeline.experiment_protocol import ProtocolError, audit_coverage, load_protocol
from researchclaw.pipeline.independent_evaluator import evaluate_manifest
from researchclaw.research_inputs import verify_bundle_contract, bind_project_data


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


@contextmanager
def _lock(root: Path):
    """Kernel locks are released on process death, unlike a stale lock marker."""
    with (root / ".protocol.lock").open("a+b") as stream:
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise ProtocolError("Another process owns this experiment matrix") from exc
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def project_inventory(project: Path) -> dict[str, str]:
    inventory = {}
    for path in sorted(project.rglob("*")):
        name = path.relative_to(project)
        if name.parts[0] == "research_data" or any(p.startswith(".") or p == "__pycache__" for p in name.parts):
            continue
        if path.is_symlink():
            raise ProtocolError("Frozen execution project cannot contain symlinks")
        if path.is_file():
            inventory[name.as_posix()] = file_hash(path)
    if "main.py" not in inventory:
        raise ProtocolError("Protocol execution requires main.py")
    if "protocol_predictions.csv" in inventory:
        raise ProtocolError("Project must not contain precomputed protocol predictions")
    return inventory


def read_ledger(root: Path, protocol: dict) -> list[dict]:
    path = root / "protocol_execution.jsonl"
    if not path.exists():
        return []
    events, previous = [], ""
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            event = json.loads(line)
            digest = event.pop("hash")
            if (event["sequence"] != len(events) or event["previous"] != previous
                    or event["protocol_version"] != protocol["version"] or content_hash(event) != digest):
                raise ProtocolError("Execution ledger hash chain changed")
            event["hash"] = digest
            events.append(event)
            previous = digest
        except (TypeError, ValueError, KeyError) as exc:
            raise ProtocolError("Invalid or truncated execution ledger") from exc
    return events


def _append(root: Path, protocol: dict, events: list, **payload) -> None:
    event = {"sequence": len(events), "previous": events[-1]["hash"] if events else "",
             "protocol_version": protocol["version"], **payload}
    event["hash"] = content_hash(event)
    with (root / "protocol_execution.jsonl").open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(event, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    events.append(event)


def ledger_usage(root: Path, protocol: dict, events: list[dict]) -> dict:
    starts, finishes, used, success = {}, {}, {}, {}
    cells = {cell["cell_id"]: cell for cell in protocol["cells"]}
    for event in events:
        if event.get("kind") == "start":
            aid, cid = event["attempt"], event["cell_id"]
            if aid in starts or cid not in cells or type(event.get("allowance")) is not int or event["allowance"] <= 0:
                raise ProtocolError("Invalid execution reservation")
            starts[aid] = event
        elif event.get("kind") == "finish":
            aid = event["attempt"]
            if aid not in starts or aid in finishes:
                raise ProtocolError("Invalid execution completion")
            elapsed = event.get("charged_seconds")
            if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < 0:
                raise ProtocolError("Invalid measured duration")
            path = (root / event["receipt"]).resolve()
            if not path.is_relative_to(root.resolve()) or not path.is_file() or file_hash(path) != event["receipt_sha256"]:
                raise ProtocolError("Execution receipt changed or disappeared")
            receipt = json.loads(path.read_text(encoding="utf-8"))
            if (receipt["cell_id"] != starts[aid]["cell_id"] or receipt["attempt"] != aid
                    or receipt["protocol_version"] != protocol["version"]
                    or receipt["key"] != cells[receipt["cell_id"]]["key"]
                    or receipt.get("recorder") != "host_protocol_runner/v1"
                    or receipt.get("code_sha256") != starts[aid]["code_sha256"]
                    or receipt.get("charged_seconds") != event["charged_seconds"]):
                raise ProtocolError("Receipt identity differs from reservation")
            expected_logs = {f"evidence_artifacts/protocol_runs/{aid}/{name}" for name in ("stdout.txt", "stderr.txt")}
            if set(receipt.get("logs", {})) != expected_logs:
                raise ProtocolError("Execution receipt lacks stdout/stderr provenance")
            for name, digest in receipt["logs"].items():
                log = (root / name).resolve()
                if not log.is_relative_to(root.resolve()) or file_hash(log) != digest:
                    raise ProtocolError("Execution log changed or disappeared")
            finishes[aid] = event
            if receipt["status"] == "success":
                if type(receipt.get("returncode")) is not int or receipt["returncode"] != 0 or receipt.get("timed_out"):
                    raise ProtocolError("Failed execution cannot be marked successful")
                cid = receipt["cell_id"]
                if cid in success:
                    raise ProtocolError("Multiple successful test attempts for one cell")
                output = (root / receipt["predictions"]).resolve()
                if not output.is_relative_to(root.resolve()) or not output.is_file() or file_hash(output) != receipt["predictions_sha256"]:
                    raise ProtocolError("Frozen test predictions changed")
                success[cid] = receipt
        else:
            raise ProtocolError("Unknown ledger event")
    for aid, event in starts.items():
        cid = event["cell_id"]
        # An interrupted host process consumes its entire reserved allowance.
        charged = finishes[aid]["charged_seconds"] if aid in finishes else event["allowance"]
        used[cid] = used.get(cid, 0.0) + charged
    return {"used_seconds": sum(used.values()), "per_cell_seconds": used, "success": success,
            "attempts": len(starts), "interrupted_attempts": sorted(starts.keys() - finishes.keys())}


def run_matrix(root: Path, project: Path, config) -> dict:
    root, project = root.resolve(), project.resolve()
    protocol = load_protocol(root)
    if protocol is None:
        raise ProtocolError("Host matrix execution requires a frozen protocol")
    if config.mode not in {"sandbox", "docker"}:
        raise ProtocolError("Host matrix execution supports sandbox and docker only")
    from researchclaw.experiment.factory import create_sandbox
    with _lock(root):
        contract = verify_bundle_contract(root)
        bind_project_data(project)
        method = None
        if contract["brief"].get("method_spec_path"):
            from researchclaw.pipeline.research_workbench import check_implementation
            method = json.loads((root / "method_spec.json").read_text(encoding="utf-8"))
            method_report = check_implementation(method, project)
            _write(root / "method_implementation.json", method_report)
            if method_report["status"] != "mapped":
                raise ProtocolError("MethodSpec code mapping failed; inspect method_implementation.json")
        inventory = project_inventory(project)
        code_hash = content_hash(inventory)
        frozen_path = root / "protocol_code.json"
        identity = {"protocol_version": protocol["version"], "files": inventory,
                    "code_sha256": code_hash, "backend": config.mode,
                    "execution_config": json.loads(json.dumps(asdict(config.docker if config.mode == "docker" else config.sandbox)))}
        events = read_ledger(root, protocol)
        budget_path = root / "protocol_budget.json"
        if budget_path.exists():
            prior_budget = json.loads(budget_path.read_text(encoding="utf-8"))
            if prior_budget.get("ledger_tip") not in {e["hash"] for e in events}:
                raise ProtocolError("Execution ledger lost its previously recorded checkpoint")
        if frozen_path.exists():
            if json.loads(frozen_path.read_text(encoding="utf-8")) != identity:
                raise ProtocolError("Code or backend changed after test configuration freeze; start a new run")
        elif events:
            raise ProtocolError("Execution ledger has lost its frozen code identity")
        else:
            _write(frozen_path, identity)
        usage = ledger_usage(root, protocol, events)
        if len(usage["success"]) < len(protocol["cells"]):
            for name in ("trusted_evaluation.json", "evidence_store.json", "experiment_coverage.json"):
                if (root / name).exists():
                    _write(root / name, {})
        # Persist source for portable independent reproduction, excluding prepared data.
        source_root = root / "evidence_artifacts" / "protocol_source"
        for name, digest in inventory.items():
            dest = source_root / name
            if dest.exists() and file_hash(dest) != digest:
                raise ProtocolError("Frozen source archive changed")
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                shutil.copyfile(project / name, dest)
        if method is not None:
            from researchclaw.experiment.method_validation_runner import run_validation
            try:
                run_validation(root, method, identity, config)
            except ValueError as exc:
                raise ProtocolError("Pretest method validation failed: " + str(exc)) from exc
            if project_inventory(project) != inventory:
                raise ProtocolError("Experiment source changed during method validation")
        for cell in protocol["cells"]:
            cid = cell["cell_id"]
            if cid in usage["success"]:
                continue
            remaining = min(cell["budget"]["per_cell_seconds"] - usage["per_cell_seconds"].get(cid, 0),
                            protocol["spec"]["budget"]["max_total_seconds"] - usage["used_seconds"])
            allowance = min(math.floor(remaining), config.time_budget_sec)
            if allowance < 1:
                continue
            aid = f"attempt-{usage['attempts'] + 1:06d}"
            attempt_dir = root / "evidence_artifacts" / "protocol_runs" / aid
            sandbox = create_sandbox(config, root / "protocol_work" / aid)
            if config.mode == "docker" and type(sandbox).__name__ != "DockerSandbox":
                raise ProtocolError("Frozen Docker execution cannot fall back to a host subprocess")
            dataset = next(d for d in contract["datasets"] if d["manifest"]["dataset"] == cell["key"]["dataset"])
            method = next(m for m in protocol["spec"]["methods"] if m["id"] == cell["key"]["method"])
            request = {"schema_version": 1, "phase": "frozen_test", "cell_id": cid,
                       "protocol_version": protocol["version"], "key": cell["key"], "method": method,
                       "dataset": dataset["card"], "output": "protocol_predictions.csv",
                       "budget_seconds": allowance, "tuning_trials": cell["budget"]["tuning_trials"]}
            request["dataset"] = dict(request["dataset"], paths={
                n: f"research_data/{cell['key']['dataset']}/{n}.csv" for n in ("train", "validation", "test_features")})
            _append(root, protocol, events, kind="start", attempt=aid, cell_id=cid,
                    allowance=allowance, code_sha256=code_hash, started_at=time.time())
            attempt_dir.mkdir(parents=True, exist_ok=False)
            for name in ("stdout.txt", "stderr.txt"):
                (attempt_dir / name).write_text("", encoding="utf-8")
            start = time.monotonic()
            receipt = {"status": "failed", "returncode": -1, "attempt": aid, "cell_id": cid,
                       "key": cell["key"], "protocol_version": protocol["version"],
                       "code_commit": f"source-sha256:{code_hash}", "code_sha256": code_hash,
                       "recorder": "host_protocol_runner/v1", "environment": {
                           "backend": type(sandbox).__name__, "host_python": platform.python_version(),
                           "platform": platform.platform()}, "timeout_seconds": allowance}
            try:
                result = sandbox.run_project(project, timeout_sec=allowance,
                    env_overrides={"ARC_PROTOCOL_REQUEST": json.dumps(request, ensure_ascii=True)})
                receipt.update(returncode=result.returncode, timed_out=result.timed_out)
                (attempt_dir / "stdout.txt").write_text(result.stdout, encoding="utf-8")
                (attempt_dir / "stderr.txt").write_text(result.stderr, encoding="utf-8")
                # Only copy files from the concrete sandbox output location.
                output_root = Path(result.output_dir).resolve() if result.output_dir else None
                if result.returncode == 0 and not result.timed_out and output_root is not None:
                    output = output_root / "protocol_predictions.csv"
                    if (not output_root.is_relative_to(root / "protocol_work" / aid)
                            or not output.resolve().is_relative_to(output_root) or not output.is_file()):
                        raise ProtocolError("Predictions are missing or outside the sandbox output directory")
                    destination = attempt_dir / "predictions.csv"
                    shutil.copyfile(output, destination)
                    # Validate coverage and numeric format without computing a test metric yet.
                    from researchclaw.pipeline.independent_evaluator import _read_values
                    predicted = _read_values(destination, "prediction")
                    labels = _read_values(root / dataset["labels"], "label")
                    if predicted.keys() != labels.keys():
                        raise ProtocolError("Predictions do not cover the frozen test IDs")
                    if cell["key"]["metric"] == "accuracy" and any(not p.is_integer() for p in predicted.values()):
                        raise ProtocolError("Accuracy predictions must be discrete labels")
                    receipt.update(status="success", predictions=destination.relative_to(root).as_posix(),
                                   predictions_sha256=file_hash(destination))
            except Exception as exc:
                receipt.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            elapsed = time.monotonic() - start
            receipt["elapsed_seconds"] = elapsed
            # A timeout is charged at least its full allowance, including host overhead.
            charged = max(elapsed, allowance if receipt.get("timed_out") else 0)
            receipt["charged_seconds"] = charged
            receipt["logs"] = {(attempt_dir / name).relative_to(root).as_posix(): file_hash(attempt_dir / name)
                               for name in ("stdout.txt", "stderr.txt")}
            receipt_path = attempt_dir / "execution.json"
            _write(receipt_path, receipt)
            _append(root, protocol, events, kind="finish", attempt=aid, charged_seconds=charged,
                    receipt=receipt_path.relative_to(root).as_posix(), receipt_sha256=file_hash(receipt_path))
            usage = ledger_usage(root, protocol, events)
            if project_inventory(project) != inventory:
                raise ProtocolError("Experiment source changed while the frozen matrix was executing")
        complete = len(usage["success"]) == len(protocol["cells"])
        over_budget = (usage["used_seconds"] > protocol["spec"]["budget"]["max_total_seconds"]
                       or any(v > protocol["spec"]["budget"]["per_cell_seconds"]
                              for v in usage["per_cell_seconds"].values()))
        budget = {"schema_version": 1, "protocol_version": protocol["version"],
                  "ledger_tip": events[-1]["hash"] if events else None,
                  "ledger_sha256": file_hash(root / "protocol_execution.jsonl") if events else None,
                  "scope": "formal_matrix_host_wall_time", "used_seconds": usage["used_seconds"],
                  "per_cell_seconds": usage["per_cell_seconds"], "attempts": usage["attempts"],
                  "interrupted_attempts": usage["interrupted_attempts"],
                  "completed_cells": len(usage["success"]), "required_cells": len(protocol["cells"]),
                  "status": "budget_exceeded" if over_budget else "complete" if complete else "incomplete"}
        _write(root / "protocol_budget.json", budget)
        if complete:
            runs = []
            for cid, receipt in usage["success"].items():
                dataset = next(d for d in contract["datasets"] if d["manifest"]["dataset"] == receipt["key"]["dataset"])
                runs.append({"key": receipt["key"], "labels": dataset["labels"],
                             "predictions": receipt["predictions"],
                             "execution": f"evidence_artifacts/protocol_runs/{receipt['attempt']}/execution.json",
                             "expected_labels_sha256": contract["outputs"][dataset["labels"]]})
            manifest = {"schema_version": 1, "runs": runs}
            _write(root / "trusted_evaluation.json", manifest)
            evidence = evaluate_manifest(root, manifest)
            _write(root / "evidence_store.json", evidence)
            _write(root / "experiment_coverage.json", audit_coverage(root, protocol, EvidenceStore.from_dict(evidence)))
        return budget


def verify_execution_bundle(root: Path, protocol: dict) -> dict:
    """Verify portable code, all attempts and accounting without rerunning code."""
    try:
        code = json.loads((root / "protocol_code.json").read_text(encoding="utf-8"))
        if (code["protocol_version"] != protocol["version"] or not code["files"]
                or content_hash(code["files"]) != code["code_sha256"]):
            raise ProtocolError("Invalid frozen source inventory")
        for name, digest in code["files"].items():
            source = (root / "evidence_artifacts" / "protocol_source" / name).resolve()
            if not source.is_relative_to((root / "evidence_artifacts" / "protocol_source").resolve()) or file_hash(source) != digest:
                raise ProtocolError("Archived experiment source changed")
        method_validation = None
        if (root / "method_spec.json").is_file():
            from researchclaw.pipeline.research_workbench import check_implementation
            method = json.loads((root / "method_spec.json").read_text(encoding="utf-8"))
            if check_implementation(method, root / "evidence_artifacts" / "protocol_source")["status"] != "mapped":
                raise ProtocolError("Archived code does not map to MethodSpec")
            from researchclaw.pipeline.method_validation import verify_validation
            method_validation = verify_validation(root, method, code)
        events = read_ledger(root, protocol)
        if not events or any(e.get("code_sha256") != code["code_sha256"] for e in events if e["kind"] == "start"):
            raise ProtocolError("Execution ledger does not bind the frozen code")
        if method_validation and method_validation["status"] == "passed":
            starts = [e["started_at"] for e in events if e["kind"] == "start"]
            if not starts or min(starts) < method_validation["finished_at"]:
                raise ProtocolError("Method validation did not precede formal test execution")
        usage = ledger_usage(root, protocol, events)
        budget = json.loads((root / "protocol_budget.json").read_text(encoding="utf-8"))
        expected = {"protocol_version": protocol["version"], "ledger_sha256": file_hash(root / "protocol_execution.jsonl"),
                    "ledger_tip": events[-1]["hash"],
                    "used_seconds": usage["used_seconds"], "per_cell_seconds": usage["per_cell_seconds"],
                    "attempts": usage["attempts"], "completed_cells": len(usage["success"]),
                    "required_cells": len(protocol["cells"]), "interrupted_attempts": usage["interrupted_attempts"]}
        if any(budget.get(k) != v for k, v in expected.items()):
            raise ProtocolError("Budget report differs from execution ledger")
        if (usage["used_seconds"] > protocol["spec"]["budget"]["max_total_seconds"]
                or any(v > protocol["spec"]["budget"]["per_cell_seconds"] for v in usage["per_cell_seconds"].values())):
            raise ProtocolError("Measured formal experiment budget exceeded")
        if len(usage["success"]) != len(protocol["cells"]):
            raise ProtocolError("Host execution matrix is incomplete")
        return usage
    except (OSError, TypeError, ValueError, KeyError) as exc:
        if isinstance(exc, ProtocolError):
            raise
        raise ProtocolError(f"Invalid host execution evidence: {type(exc).__name__}") from exc
