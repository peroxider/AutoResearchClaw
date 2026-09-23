"""Frozen dependency manifests: host inventory and container digest pinning."""
import json
import subprocess

import pytest

from researchclaw.experiment.dependency_manifest import (
    build_dependency_manifest,
    validate_dependency_manifest,
)
from researchclaw.experiment.docker_sandbox import DockerSandbox
from researchclaw.experiment.protocol_runner import (
    read_ledger, run_matrix, verify_execution_bundle,
)
from researchclaw.experiment.sandbox import SandboxResult
from researchclaw.pipeline.evidence_store import content_hash, file_hash
from researchclaw.pipeline.experiment_protocol import ProtocolError
from tests.test_experiment_protocol import spec
from tests.test_research_inputs import inputs
from tests.test_protocol_runner import setup


DIGEST = "researchclaw/experiment@sha256:" + "a" * 64


def test_host_manifest_freezes_the_installed_inventory():
    manifest = build_dependency_manifest("host")
    validate_dependency_manifest(manifest)
    assert manifest["backend"] == "host" and manifest["image_digest"] is None
    assert manifest["python"] and manifest["platform"]
    assert "pytest" in manifest["distributions"]
    entry = manifest["distributions"]["pytest"]
    assert entry["version"]
    record = entry["record_sha256"]
    assert record is None or (len(record) == 64 and record == record.lower())


def test_host_manifest_is_deterministic():
    assert build_dependency_manifest("host") == build_dependency_manifest("host")


def test_docker_manifest_pins_the_image_and_refuses_unknown_backends():
    manifest = build_dependency_manifest("docker", image_digest=DIGEST)
    validate_dependency_manifest(manifest)
    assert manifest["distributions"] == {} and manifest["python"] is None
    assert manifest["image_digest"] == DIGEST
    with pytest.raises(ProtocolError, match="backend must be"):
        build_dependency_manifest("container")


def test_invalid_manifests_fail_closed():
    host = build_dependency_manifest("host")
    docker = build_dependency_manifest("docker", image_digest=DIGEST)
    bad = [
        None, [], {},
        dict(host, schema_version=2),
        dict(host, backend="container"),
        dict(host, image_digest="sha256:host_run_cannot_have_one"),
        {key: value for key, value in host.items() if key != "python"},
        dict(host, python=""),
        dict(host, platform=None),
        dict(host, distributions={}),
        dict(host, distributions={"pkg": {"version": "1.0", "record_sha256": "nothex"}}),
        dict(host, distributions={"pkg": {"version": "", "record_sha256": None}}),
        dict(host, distributions={"": {"version": "1.0", "record_sha256": None}}),
        dict(host, distributions={"pkg": "not-a-mapping"}),
        dict(host, image_digest=123),
        dict(docker, image_digest=None),
        dict(docker, image_digest=123),
        dict(docker, python="3.11.0"),
        dict(docker, distributions={"pkg": {"version": "1", "record_sha256": None}}),
    ]
    for manifest in bad:
        with pytest.raises(ProtocolError, match="manifest|Container runs"):
            validate_dependency_manifest(manifest)


def test_matrix_freezes_the_manifest_and_binds_it_everywhere(inputs, spec):
    root, project, protocol, cfg = setup(inputs, spec)
    assert run_matrix(root, project, cfg.experiment)["status"] == "complete"
    manifest = json.loads((root / "protocol_code.json").read_text())["dependency_manifest"]
    validate_dependency_manifest(manifest)
    assert manifest["backend"] == "host" and manifest["image_digest"] is None
    receipt = json.loads(
        (root / "evidence_artifacts/protocol_runs/attempt-000001/execution.json").read_text())
    assert receipt["dependency_sha256"] == content_hash(manifest)
    assert receipt["environment"]["host_python"] == manifest["python"]
    assert receipt["environment"]["platform"] == manifest["platform"]
    assert verify_execution_bundle(root, protocol)["success"]


def test_tampered_manifest_breaks_the_ledger_binding(inputs, spec):
    root, project, protocol, cfg = setup(inputs, spec)
    run_matrix(root, project, cfg.experiment)
    code_path = root / "protocol_code.json"
    code = json.loads(code_path.read_text())
    name = next(iter(code["dependency_manifest"]["distributions"]))
    code["dependency_manifest"]["distributions"][name]["version"] = "9.9.9-tampered"
    code_path.write_text(json.dumps(code, indent=2), encoding="utf-8")
    with pytest.raises(ProtocolError, match="does not bind the frozen dependency manifest"):
        verify_execution_bundle(root, protocol)


def _forge_chain(root, events):
    """Rebuild the hash chain and budget anchors after a tamper, exactly as
    a forger would; only the environment cross-check can then catch it."""
    previous = ""
    lines = []
    for event in events:
        stored = {key: value for key, value in event.items() if key != "hash"}
        stored["previous"] = previous
        stored["hash"] = content_hash(stored)
        previous = stored["hash"]
        lines.append(json.dumps(stored))
    (root / "protocol_execution.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    budget_path = root / "protocol_budget.json"
    budget = json.loads(budget_path.read_text())
    budget["ledger_sha256"] = file_hash(root / "protocol_execution.jsonl")
    budget["ledger_tip"] = previous
    budget_path.write_text(json.dumps(budget, indent=2), encoding="utf-8")


def test_forged_receipt_environment_breaks_acceptance(inputs, spec):
    root, project, protocol, cfg = setup(inputs, spec)
    run_matrix(root, project, cfg.experiment)
    receipt_path = root / "evidence_artifacts/protocol_runs/attempt-000001/execution.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["environment"]["host_python"] = "3.0.0-fabricated"
    receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    events = read_ledger(root, protocol)
    for event in events:
        if event["kind"] == "finish" and event["attempt"] == "attempt-000001":
            event["receipt_sha256"] = file_hash(receipt_path)
    _forge_chain(root, events)
    with pytest.raises(ProtocolError, match="does not match the recorded execution environment"):
        verify_execution_bundle(root, protocol)


def test_forged_failed_receipt_environment_is_rejected_too(inputs, spec):
    # Failed attempts carry no metrics, but their receipts are part of the
    # audited history: a fully forged chain must not launder an environment
    # tamper on a failed receipt either.
    from tests.test_protocol_runner import SCRIPT
    script = SCRIPT.replace('assert request["phase"] == "frozen_test"',
                            'assert request["phase"] == "frozen_test"\n'
                            'assert request["key"]["seed"] != "42", "seed 42 fails"')
    root, project, protocol, cfg = setup(inputs, spec, script)
    run_matrix(root, project, cfg.experiment)
    events = read_ledger(root, protocol)
    failed = [event for event in events if event["kind"] == "finish"
              and json.loads((root / event["receipt"]).read_text())["status"] != "success"]
    assert failed
    victim = failed[0]
    receipt_path = root / victim["receipt"]
    receipt = json.loads(receipt_path.read_text())
    receipt["environment"]["host_python"] = "1.2.3-fabricated"
    receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    for event in events:
        if event["kind"] == "finish" and event["attempt"] == receipt["attempt"]:
            event["receipt_sha256"] = file_hash(receipt_path)
    _forge_chain(root, events)
    with pytest.raises(ProtocolError, match="does not match the recorded execution environment"):
        verify_execution_bundle(root, protocol)


def test_docker_mode_requires_an_image_digest(inputs, spec, monkeypatch):
    from dataclasses import replace
    root, project, protocol, cfg = setup(inputs, spec)
    docker_cfg = replace(cfg.experiment, mode="docker",
                         docker=replace(cfg.experiment.docker, network_policy="none"))
    monkeypatch.setattr(DockerSandbox, "inspect_image_digest", lambda image: None)
    with pytest.raises(ProtocolError, match="cannot be pinned"):
        run_matrix(root, project, docker_cfg)
    assert not (root / "protocol_code.json").exists()


def test_docker_mode_freezes_the_image_digest_not_a_host_inventory(inputs, spec, monkeypatch):
    from dataclasses import replace
    from researchclaw.experiment import docker_sandbox
    root, project, protocol, cfg = setup(inputs, spec)
    docker_cfg = replace(cfg.experiment, mode="docker",
                         docker=replace(cfg.experiment.docker, network_policy="none"))
    monkeypatch.setattr(docker_sandbox.DockerSandbox, "inspect_image_digest",
                        lambda image: DIGEST)

    class DockerSandbox:  # The no-fallback guard admits this class name only.
        def run_project(self, project, **kwargs):
            return SandboxResult(1, "", "boom", 0, {}, timed_out=False)

    monkeypatch.setattr("researchclaw.experiment.factory.create_sandbox",
                        lambda *a, **k: DockerSandbox())
    report = run_matrix(root, project, docker_cfg)
    assert report["status"] == "incomplete"
    manifest = json.loads((root / "protocol_code.json").read_text())["dependency_manifest"]
    validate_dependency_manifest(manifest)
    assert manifest["backend"] == "docker" and manifest["image_digest"] == DIGEST
    assert manifest["python"] is None and manifest["distributions"] == {}


def test_inspect_image_digest_parses_inspect_output(monkeypatch):
    def inspect(payload, returncode=0):
        def fake_run(cmd, **kwargs):
            return subprocess.CompletedProcess(cmd, returncode,
                                               stdout=json.dumps(payload), stderr="")
        return fake_run

    full = [{"Id": "sha256:" + "d" * 64,
             "RepoDigests": ["reg/img@sha256:" + "e" * 64]}]
    monkeypatch.setattr("researchclaw.experiment.docker_sandbox.subprocess.run",
                        inspect(full))
    assert DockerSandbox.inspect_image_digest("img") == "reg/img@sha256:" + "e" * 64
    local = [{"Id": "sha256:" + "d" * 64, "RepoDigests": []}]
    monkeypatch.setattr("researchclaw.experiment.docker_sandbox.subprocess.run",
                        inspect(local))
    assert DockerSandbox.inspect_image_digest("img") == "sha256:" + "d" * 64
    monkeypatch.setattr("researchclaw.experiment.docker_sandbox.subprocess.run",
                        inspect([], returncode=1))
    assert DockerSandbox.inspect_image_digest("img") is None
    monkeypatch.setattr("researchclaw.experiment.docker_sandbox.subprocess.run",
                        inspect("{broken", returncode=0))
    assert DockerSandbox.inspect_image_digest("img") is None
