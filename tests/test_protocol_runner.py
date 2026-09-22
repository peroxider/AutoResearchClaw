import json
import sys
from dataclasses import replace

import pytest

from researchclaw.adapters import AdapterBundle
from researchclaw.experiment.protocol_runner import (
    _append, _lock, ledger_usage, read_ledger, run_matrix, verify_execution_bundle,
)
from researchclaw.experiment.sandbox import SandboxResult
from researchclaw.pipeline.evidence_store import EvidenceStore, file_hash
from researchclaw.pipeline.experiment_protocol import ProtocolError, audit_coverage
from researchclaw.pipeline.stages import StageStatus
from tests.test_experiment_protocol import spec, freeze
from tests.test_research_inputs import inputs


SCRIPT = '''import csv, json, os
from collections import Counter
from pathlib import Path
request = json.loads(os.environ["ARC_PROTOCOL_REQUEST"])
assert request["phase"] == "frozen_test"
data = request["dataset"]
train = list(csv.DictReader(Path(data["paths"]["train"]).open()))
test = list(csv.DictReader(Path(data["paths"]["test_features"]).open()))
assert data["label_column"] not in test[0]
assert not Path("research_data/demo/test_labels.csv").exists()
label = Counter(row[data["label_column"]] for row in train).most_common(1)[0][0]
with Path(request["output"]).open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "prediction"])
    writer.writerows([row[data["id_column"]], data["label_encoding"][label]] for row in test)
'''


def setup(inputs, spec, script=SCRIPT):
    root, contract, protocol, cfg = freeze(inputs, spec)
    cfg = replace(cfg, experiment=replace(cfg.experiment,
                  sandbox=replace(cfg.experiment.sandbox, python_path=sys.executable)))
    project = root / "stage-10" / "experiment"
    project.mkdir(parents=True)
    (project / "main.py").write_text(script, encoding="utf-8")
    return root, project, protocol, cfg


def test_real_host_matrix_evaluates_and_resume_never_reruns_success(inputs, spec, monkeypatch):
    root, project, protocol, cfg = setup(inputs, spec)
    report = run_matrix(root, project, cfg.experiment)
    assert report["status"] == "complete" and report["attempts"] == 8
    assert report["used_seconds"] > 0
    assert len(read_ledger(root, protocol)) == 16
    assert len(verify_execution_bundle(root, protocol)["success"]) == 8
    store = EvidenceStore.from_dict(json.loads((root / "evidence_store.json").read_text()))
    assert len(store.records) == 8
    assert {r.value for r in store.records.values()} == {0.5}
    assert audit_coverage(root, protocol, store)["status"] == "complete"
    def forbidden(*args, **kwargs):
        pytest.fail("Successful test cells must never execute on resume")
    monkeypatch.setattr("researchclaw.experiment.factory.create_sandbox", forbidden)
    before = file_hash(root / "protocol_execution.jsonl")
    assert run_matrix(root, project, cfg.experiment)["status"] == "complete"
    assert file_hash(root / "protocol_execution.jsonl") == before


def test_failure_history_preserved_and_no_test_metrics_released(inputs, spec):
    script = SCRIPT.replace('assert request["phase"] == "frozen_test"',
                            'assert request["key"]["seed"] != "42", "transient fixture failure"')
    root, project, protocol, cfg = setup(inputs, spec, script)
    report = run_matrix(root, project, cfg.experiment)
    assert report["status"] == "incomplete" and report["completed_cells"] == 4
    assert not (root / "trusted_evaluation.json").exists()
    assert not (root / "evidence_store.json").exists()
    first = (root / "protocol_execution.jsonl").read_text()
    again = run_matrix(root, project, cfg.experiment)
    assert again["attempts"] == 12 and again["used_seconds"] > report["used_seconds"]
    assert (root / "protocol_execution.jsonl").read_text().startswith(first)
    assert len(list((root / "evidence_artifacts/protocol_runs").glob("*/execution.json"))) == 12


def test_timed_out_cells_consume_cap_and_are_not_retried(inputs, spec, monkeypatch):
    root, project, protocol, cfg = setup(inputs, spec)
    calls = []
    class TimeoutSandbox:
        def run_project(self, project, **kwargs):
            calls.append(kwargs)
            return SandboxResult(-1, "", "timeout", 0, {}, timed_out=True)
    monkeypatch.setattr("researchclaw.experiment.factory.create_sandbox", lambda *a: TimeoutSandbox())
    assert run_matrix(root, project, cfg.experiment)["used_seconds"] == 80
    assert run_matrix(root, project, cfg.experiment)["attempts"] == 8
    assert len(calls) == 8
    assert all(call["timeout_sec"] == 10 for call in calls)


def test_interrupted_attempt_charges_reserved_time(inputs, spec):
    root, _, protocol, _ = setup(inputs, spec)
    events = []
    _append(root, protocol, events, kind="start", attempt="attempt-000001",
            cell_id=protocol["cells"][0]["cell_id"], allowance=10, code_sha256="fixture")
    usage = ledger_usage(root, protocol, read_ledger(root, protocol))
    assert usage["used_seconds"] == 10 and usage["interrupted_attempts"] == ["attempt-000001"]


@pytest.mark.parametrize("target", ["source", "predictions", "receipt", "ledger", "budget", "log"])
def test_delivery_rejects_tampered_host_artifacts(inputs, spec, target):
    root, project, protocol, cfg = setup(inputs, spec)
    run_matrix(root, project, cfg.experiment)
    names = {"source": "evidence_artifacts/protocol_source/main.py",
             "predictions": "evidence_artifacts/protocol_runs/attempt-000001/predictions.csv",
             "receipt": "evidence_artifacts/protocol_runs/attempt-000001/execution.json",
             "ledger": "protocol_execution.jsonl", "budget": "protocol_budget.json",
             "log": "evidence_artifacts/protocol_runs/attempt-000001/stdout.txt"}
    path = root / names[target]
    if target == "budget":
        data = json.loads(path.read_text())
        data["used_seconds"] = 0
        path.write_text(json.dumps(data))
    else:
        path.write_text(path.read_text() + "changed")
    with pytest.raises(ProtocolError):
        verify_execution_bundle(root, protocol)
    store = EvidenceStore.from_dict(json.loads((root / "evidence_store.json").read_text()))
    assert audit_coverage(root, protocol, store)["status"] == "incomplete"


def test_changed_code_after_test_freeze_is_rejected(inputs, spec):
    root, project, protocol, cfg = setup(inputs, spec)
    run_matrix(root, project, cfg.experiment)
    (project / "main.py").write_text(SCRIPT + "\n# changed")
    with pytest.raises(ProtocolError, match="Code or backend changed"):
        run_matrix(root, project, cfg.experiment)


def test_concurrent_matrix_run_rejected(inputs, spec):
    root, project, _, cfg = setup(inputs, spec)
    with _lock(root):
        with pytest.raises(ProtocolError, match="Another process"):
            run_matrix(root, project, cfg.experiment)


def test_stage12_and_stage13_use_frozen_host_execution(inputs, spec):
    from researchclaw.pipeline.stage_impls._execution import _execute_experiment_run, _execute_iterative_refine
    root, project, protocol, cfg = setup(inputs, spec)
    stage12, stage13 = root / "stage-12", root / "stage-13"
    stage12.mkdir()
    stage13.mkdir()
    result = _execute_experiment_run(stage12, root, cfg, AdapterBundle())
    assert result.status == StageStatus.DONE
    payload = json.loads((stage12 / "runs/run-001.json").read_text())
    assert len(payload["metrics"]) == 8
    assert _execute_iterative_refine(stage13, root, cfg, AdapterBundle()).decision == "frozen_test_protocol"
    assert len(read_ledger(root, protocol)) == 16


def test_docker_does_not_silently_change_execution_backend(inputs, spec, monkeypatch):
    root, project, _, cfg = setup(inputs, spec)
    class HostFallback:
        pass
    monkeypatch.setattr("researchclaw.experiment.factory.create_sandbox", lambda *args: HostFallback())
    with pytest.raises(ProtocolError, match="cannot fall back"):
        run_matrix(root, project, replace(cfg.experiment, mode="docker"))
    assert not (root / "protocol_execution.jsonl").exists()


def test_stale_metrics_invalidated_before_retry(inputs, spec):
    root, project, _, cfg = setup(inputs, spec, "raise RuntimeError('fixture failure')")
    for name in ("trusted_evaluation.json", "evidence_store.json", "experiment_coverage.json"):
        (root / name).write_text('{"status": "stale success"}')
    run_matrix(root, project, cfg.experiment)
    for name in ("trusted_evaluation.json", "evidence_store.json", "experiment_coverage.json"):
        assert json.loads((root / name).read_text()) == {}


def test_test_driven_research_rollback_is_paused(inputs, spec, monkeypatch):
    from researchclaw.pipeline import runner
    from researchclaw.pipeline._helpers import StageResult
    from researchclaw.pipeline.stages import Stage
    root, _, _, cfg = setup(inputs, spec)
    monkeypatch.setattr(runner, "STAGE_SEQUENCE", (Stage.RESEARCH_DECISION, Stage.PAPER_OUTLINE))
    calls = []
    def execute(stage, **kwargs):
        calls.append(stage)
        return StageResult(stage=stage, status=StageStatus.DONE, artifacts=(), decision="refine")
    monkeypatch.setattr(runner, "execute_stage", execute)
    results = runner.execute_pipeline(run_dir=root, run_id="freeze", config=cfg, adapters=AdapterBundle(),
                                      from_stage=Stage.RESEARCH_DECISION, auto_approve_gates=True)
    assert calls == [Stage.RESEARCH_DECISION]
    assert results[-1].decision == "new_protocol_required" and results[-1].status == StageStatus.PAUSED


def test_resume_rejects_a_truncated_but_valid_ledger_prefix(inputs, spec):
    root, project, protocol, cfg = setup(inputs, spec)
    run_matrix(root, project, cfg.experiment)
    path = root / "protocol_execution.jsonl"
    lines = path.read_text().splitlines()
    path.write_text("\n".join(lines[:-2]) + "\n")
    with pytest.raises(ProtocolError, match="checkpoint"):
        run_matrix(root, project, cfg.experiment)
