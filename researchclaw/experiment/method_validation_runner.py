"""One reserved pretest probe run, with durable evidence and no implicit retries."""
from __future__ import annotations

import json
import shutil
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

from researchclaw.pipeline.method_validation import (
    WorkbenchError, assess, build_request, checker_identity, make_report, verify_validation,
)


def run_validation(root, method, code, config):
    # Called under protocol_runner's kernel lock, after the source freeze.
    from researchclaw.experiment.factory import create_sandbox
    from researchclaw.experiment.protocol_runner import _write
    from researchclaw.experiment import method_probe
    if "validation" not in method["spec"]:
        return {"status": "not_declared", "semantic_equivalence": "unresolved"}
    expected_config = json.loads(json.dumps(asdict(config.docker if config.mode == "docker" else config.sandbox)))
    if config.mode not in {"sandbox", "docker"} or code["backend"] != config.mode or code["execution_config"] != expected_config:
        raise WorkbenchError("Method validation backend differs from the frozen execution configuration")
    if (root / "method_validation.json").exists():
        return verify_validation(root, method, code)
    if (root / "protocol_execution.jsonl").exists():
        raise WorkbenchError("Cannot start method validation after formal test execution; evidence is missing")
    artifact = root / "evidence_artifacts" / "method_validation"
    if artifact.exists():
        raise WorkbenchError("Interrupted method validation reservation; no automatic retry, start a new run")
    timeout = min(method["spec"]["validation"]["timeout_seconds"], int(config.time_budget_sec))
    if timeout < 1:
        raise WorkbenchError("Method validation has no available execution time")
    reservation = {"method_version": method["version"], "code_sha256": code["code_sha256"],
                   "checker_identity": checker_identity(), "backend": code["backend"],
                   "execution_config": code["execution_config"], "request": build_request(method),
                   "timeout_seconds": timeout, "started_at": time.time()}
    artifact.mkdir(parents=True, exist_ok=False)
    _write(artifact / "reservation.json", reservation)
    shutil.copyfile(Path(method_probe.__file__), artifact / "driver.py")
    observations = {"schema_version": 1, "observations": []}
    receipt = {"returncode": -1, "timed_out": False, "elapsed_seconds": 0.0}
    for name in ("stdout.txt", "stderr.txt"):
        (artifact / name).write_text("", encoding="utf-8")
    workdir = root / "method_validation_work"
    start = time.monotonic()
    try:
        # Deliberately outside the run tree: sandbox.run_project normally binds
        # public research splits by searching ancestors for research_contract.json.
        with tempfile.TemporaryDirectory(prefix="arc-method-probe-") as staging_name:
            staging = Path(staging_name).resolve()
            if any((p / "research_contract.json").exists() for p in staging.parents):
                raise WorkbenchError("Synthetic probe staging must not inherit a research contract")
            source = root / "evidence_artifacts" / "protocol_source"
            for name in code["files"]:
                destination = staging / "source" / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source / name, destination)
            shutil.copyfile(artifact / "driver.py", staging / "arc_method_probe.py")
            _write(staging / "request.json", reservation["request"])
            sandbox = create_sandbox(config, workdir)
            if config.mode == "docker":
                from researchclaw.experiment.docker_sandbox import DockerSandbox
                if not isinstance(sandbox, DockerSandbox):
                    raise WorkbenchError("Docker probe execution cannot fall back to a host subprocess")
            result = sandbox.run_project(staging, entry_point="arc_method_probe.py", timeout_sec=timeout,
                                         env_overrides={"ARC_PROTOCOL_REQUEST": "", "ARC_METHOD_PROBE": "1"})
            receipt.update(returncode=result.returncode, timed_out=result.timed_out)
            for name, value in (("stdout.txt", result.stdout), ("stderr.txt", result.stderr)):
                (artifact / name).write_text(value, encoding="utf-8")
            if result.returncode == 0 and not result.timed_out:
                if not result.output_dir:
                    raise WorkbenchError("Method probe output directory is missing")
                output = (Path(result.output_dir) / "observations.json").resolve()
                if not output.is_relative_to(workdir.resolve()) or output.stat().st_size > 2_000_000:
                    raise WorkbenchError("Method probe output escaped its directory or exceeds byte budget")
                def reject_constant(value):
                    raise WorkbenchError("Nonfinite JSON probe output: " + value)
                observations = json.loads(output.read_text(encoding="utf-8"), parse_constant=reject_constant)
    except Exception as exc:
        receipt["error"] = f"{type(exc).__name__}: {exc}"[:2000]
    receipt["elapsed_seconds"] = time.monotonic() - start
    receipt["finished_at"] = time.time()
    receipt["charged_seconds"] = max(receipt["elapsed_seconds"], timeout if receipt["timed_out"] else 0)
    if receipt["elapsed_seconds"] > timeout:
        receipt["error"] = "Method validation wall-time budget exceeded"
    _write(artifact / "observations.json", observations)
    _write(artifact / "execution.json", receipt)
    try:
        checks = assess(method, observations)
    except (ValueError, TypeError, KeyError) as exc:
        checks = [{"id": "invalid_observations", "kind": "execution", "status": "failed", "error": str(exc)}]
    report = make_report(method, code, artifact, checks, receipt)
    _write(root / "method_validation.json", report)
    return verify_validation(root, method, code)
