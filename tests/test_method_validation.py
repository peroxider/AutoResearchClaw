import copy
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from researchclaw.config import ExperimentConfig, SandboxConfig
from researchclaw.experiment.method_validation_runner import run_validation
from researchclaw.experiment.protocol_runner import run_matrix, verify_execution_bundle
from researchclaw.experiment.sandbox import SandboxResult
from researchclaw.pipeline.evidence_store import content_hash, file_hash
from researchclaw.pipeline.experiment_protocol import ProtocolError
from researchclaw.pipeline.method_validation import assess, build_request, verify_validation
from researchclaw.pipeline.research_workbench import WorkbenchError, compile_method
from tests.test_experiment_protocol import spec
from tests.test_protocol_runner import setup
from tests.test_research_inputs import inputs, dump


MODEL = '''from pathlib import Path
assert not Path("research_data").exists(), "probe received study data"
assert not Path("source/research_data").exists(), "probe received nested study data"
def loss(x):
    return sum(v * v for v in x)
def gradient(x):
    return [2 * v for v in x]
class Model:
    def __init__(self, factor=1): self.factor = factor
    def predict(self, x, enabled):
        return [v * self.factor + int(enabled) for v in x]
'''


@pytest.fixture
def runtime_method():
    variables = {key: {"shape": shape, "description": key} for key, shape in
                 (("x", ["d"]), ("y", ["d"]), ("g", ["d"]), ("loss_value", []), ("enabled", []))}
    steps = [{"id": sid, "phase": "inference" if sid == "predict" else "train",
              "description": sid, "inputs": ["x", "enabled"] if sid == "predict" else ["x"],
              "outputs": [output], "equations": [], "depends_on": [], "code_file": "model.py", "code_symbol": symbol}
             for sid, output, symbol in (("loss", "loss_value", "loss"), ("grad", "g", "gradient"),
                                         ("predict", "y", "Model.predict"))]
    return {"schema_version": 1, "method_id": "A_full", "description": "synthetic probe fixture",
            "variables": variables, "equations": [], "steps": steps, "losses": [],
            "stopping_rule": "one call", "complexity": {"time": "O(d)", "space": "O(d)"},
            "validation": {"schema_version": 1, "timeout_seconds": 10, "cases": [
                {"id": "golden", "kind": "microinstance", "atol": 0, "rtol": 0,
                 "call": {"step": "predict", "kwargs": {"x": [2, 3], "enabled": True}, "constructor_kwargs": {"factor": 2}},
                 "expected": [5, 7]},
                {"id": "derivative", "kind": "gradient", "atol": 1e-6, "rtol": 1e-6,
                 "call": {"step": "loss", "kwargs": {"x": [2.0, -3.0]}},
                 "gradient_call": {"step": "grad", "kwargs": {"x": [2.0, -3.0]}}, "argument": "x", "epsilon": 0.0001},
                {"id": "toggle", "kind": "component_toggle", "atol": 0, "rtol": 0,
                 "call": {"step": "predict", "kwargs": {"x": [2, 3], "enabled": True}}, "parameter": "enabled",
                 "expected_enabled": [3, 4], "expected_disabled": [2, 3]},
            ]}}


def prepared(tmp_path, runtime_method, model=MODEL):
    root = tmp_path / "probe-run"
    source = root / "evidence_artifacts" / "protocol_source"
    source.mkdir(parents=True)
    (source / "model.py").write_text(model, encoding="utf-8")
    (source / "main.py").write_text("raise RuntimeError('main must not execute in probes')", encoding="utf-8")
    method = compile_method(runtime_method)
    cfg = ExperimentConfig(sandbox=SandboxConfig(python_path=sys.executable))
    files = {p.name: file_hash(p) for p in source.iterdir()}
    code = {"files": files, "code_sha256": content_hash(files), "backend": "sandbox",
            "execution_config": json.loads(json.dumps(asdict(cfg.sandbox)))}
    dump(root / "method_spec.json", method)
    dump(root / "protocol_code.json", code)
    return root, method, code, cfg


def test_real_probes_call_source_and_host_checks_all_three_kinds(tmp_path, runtime_method, monkeypatch):
    root, method, code, cfg = prepared(tmp_path, runtime_method)
    report = run_validation(root, method, code, cfg)
    assert report["status"] == "passed"
    assert len(report["checks"]) == 3
    assert report["semantic_equivalence"] == "unresolved"
    assert report["coverage"]["untested_steps"] == []
    assert report["elapsed_seconds"] > 0
    request = build_request(method)
    assert len(request["calls"]) == 8
    assert "expected" not in json.dumps(request)
    rows = json.loads((root / "evidence_artifacts/method_validation/observations.json").read_text())["observations"]
    assert rows[0]["value"] == [5, 7] and rows[1]["value"] == [4, -6]
    monkeypatch.setattr("researchclaw.experiment.factory.create_sandbox", lambda *a: pytest.fail("must not rerun"))
    assert run_validation(root, method, code, cfg) == report


@pytest.mark.parametrize("mutation", [
    lambda m: m["validation"].update(schema_version=True),
    lambda m: m["validation"].update(timeout_seconds=121),
    lambda m: m["validation"].update(timeout_seconds=0),
    lambda m: m["validation"].update(cases=[]),
    lambda m: m["validation"]["cases"][0].update(expected=[5]),
    lambda m: m["validation"]["cases"][0].update(expected=[5, float("nan")]),
    lambda m: m["validation"]["cases"][0].update(atol=True),
    lambda m: m["validation"]["cases"][0].update(rtol=0.02),
    lambda m: m["validation"]["cases"][0].update(kind="self_report"),
    lambda m: m["validation"]["cases"][0].update(id="derivative"),
    lambda m: m["validation"]["cases"][0]["call"].update(step="not_mapped"),
    lambda m: m["validation"]["cases"][0]["call"]["kwargs"].update(file="test_labels.csv"),
    lambda m: m["validation"]["cases"][0]["call"]["kwargs"].update(x="test_labels.csv"),
    lambda m: m["validation"]["cases"][0]["call"]["kwargs"].update(x=[[1], [1, 2]]),
    lambda m: m["validation"]["cases"][1].update(epsilon=0),
    lambda m: m["validation"]["cases"][1].update(epsilon=True),
    lambda m: m["validation"]["cases"][1].update(argument="not_input"),
    lambda m: m["validation"]["cases"][1]["gradient_call"]["kwargs"].update(x=[3, 4]),
    lambda m: m["validation"]["cases"][2].update(parameter="x"),
    lambda m: m["variables"]["g"].update(shape=[]),
    lambda m: m["variables"]["loss_value"].update(shape=[1]),
    lambda m: m["steps"][0].update(code_symbol="Outer.Inner.loss"),
    lambda m: m["validation"]["cases"][0]["call"]["constructor_kwargs"].update(factor=float("inf")),
])
def test_invalid_plan_fails_before_execution(runtime_method, mutation):
    mutation(runtime_method)
    with pytest.raises(WorkbenchError):
        compile_method(runtime_method)


def test_gradient_invocation_budget_and_unrepresentable_perturbation(runtime_method):
    case = runtime_method["validation"]["cases"][1]
    case["call"]["kwargs"]["x"] = [1.0] * 64
    case["gradient_call"]["kwargs"]["x"] = [1.0] * 64
    runtime_method["validation"]["cases"] = [case, dict(copy.deepcopy(case), id="second")]
    with pytest.raises(WorkbenchError, match="invocation budget"):
        compile_method(runtime_method)
    runtime_method["validation"]["cases"] = [case]
    case["call"]["kwargs"]["x"] = [1e100]
    case["gradient_call"]["kwargs"]["x"] = [1e100]
    with pytest.raises(WorkbenchError, match="not representable"):
        compile_method(runtime_method)


@pytest.mark.parametrize("model,case_id", [
    (MODEL.replace("2 * v", "3 * v"), "derivative"),
    (MODEL.replace("int(enabled)", "1"), "toggle"),
    (MODEL.replace("v * self.factor", "v"), "golden"),
    (MODEL.replace("return [2 * v for v in x]", "return [[2 * v for v in x]]"), "derivative"),
    (MODEL.replace("return [2 * v for v in x]", "return [float('nan')] * len(x)"), "derivative"),
    (MODEL.replace("return [2 * v for v in x]", "raise RuntimeError('broken gradient')"), "derivative"),
])
def test_incorrect_implementation_preserves_failure_and_never_retries(tmp_path, runtime_method, model, case_id, monkeypatch):
    root, method, code, cfg = prepared(tmp_path, runtime_method, model)
    with pytest.raises(WorkbenchError):
        run_validation(root, method, code, cfg)
    report = json.loads((root / "method_validation.json").read_text())
    assert report["status"] == "failed"
    assert next(c for c in report["checks"] if c["id"] == case_id)["status"] == "failed"
    monkeypatch.setattr("researchclaw.experiment.factory.create_sandbox", lambda *a: pytest.fail("must not retry"))
    with pytest.raises(WorkbenchError):
        run_validation(root, method, code, cfg)


@pytest.mark.parametrize("filename", ["observations.json", "stdout.txt", "stderr.txt", "execution.json", "reservation.json", "driver.py"])
def test_archive_tampering_rejected(tmp_path, runtime_method, filename):
    root, method, code, cfg = prepared(tmp_path, runtime_method)
    run_validation(root, method, code, cfg)
    path = root / "evidence_artifacts/method_validation" / filename
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(WorkbenchError):
        verify_validation(root, method, code)


def test_rehashed_claim_cannot_override_raw_comparisons(tmp_path, runtime_method):
    root, method, code, cfg = prepared(tmp_path, runtime_method)
    report = run_validation(root, method, code, cfg)
    report["semantic_equivalence"] = "verified"
    report.pop("version")
    report["version"] = content_hash(report)
    dump(root / "method_validation.json", report)
    with pytest.raises(WorkbenchError, match="report changed"):
        verify_validation(root, method, code)


def test_interrupted_reservation_does_not_rerun(tmp_path, runtime_method, monkeypatch):
    root, method, code, cfg = prepared(tmp_path, runtime_method)
    (root / "evidence_artifacts/method_validation").mkdir()
    monkeypatch.setattr("researchclaw.experiment.factory.create_sandbox", lambda *a: pytest.fail("must not retry"))
    with pytest.raises(WorkbenchError, match="Interrupted"):
        run_validation(root, method, code, cfg)


def test_timeout_is_durable_and_blocks(tmp_path, runtime_method, monkeypatch):
    root, method, code, cfg = prepared(tmp_path, runtime_method)
    class Timeout:
        def run_project(self, project, **kwargs):
            assert kwargs["timeout_sec"] == 10
            return SandboxResult(-1, "partial", "timeout", 10, {}, timed_out=True)
    monkeypatch.setattr("researchclaw.experiment.factory.create_sandbox", lambda *a: Timeout())
    with pytest.raises(WorkbenchError):
        run_validation(root, method, code, cfg)
    assert (root / "evidence_artifacts/method_validation/stdout.txt").read_text() == "partial"
    report = json.loads((root / "method_validation.json").read_text())
    assert report["status"] == "failed" and report["charged_seconds"] == 10


def test_docker_fallback_is_rejected(tmp_path, runtime_method, monkeypatch):
    root, method, code, cfg = prepared(tmp_path, runtime_method)
    cfg = replace(cfg, mode="docker")
    code.update(backend="docker", execution_config=json.loads(json.dumps(asdict(cfg.docker))))
    monkeypatch.setattr("researchclaw.experiment.factory.create_sandbox", lambda *a: object())
    with pytest.raises(WorkbenchError):
        run_validation(root, method, code, cfg)
    receipt = json.loads((root / "evidence_artifacts/method_validation/execution.json").read_text())
    assert "cannot fall back" in receipt["error"]


def setup_matrix(inputs, spec, runtime_method):
    brief, _, _, data = inputs
    data["method_spec_path"] = "method.json"
    dump(brief, data)
    dump(brief.parent / "method.json", runtime_method)
    root, project, protocol, cfg = setup(inputs, spec)
    (project / "model.py").write_text(MODEL, encoding="utf-8")
    return root, project, protocol, cfg


def test_probes_precede_heldout_matrix_and_resume_is_read_only(inputs, spec, runtime_method, monkeypatch):
    root, project, protocol, cfg = setup_matrix(inputs, spec, runtime_method)
    assert run_matrix(root, project, cfg.experiment)["status"] == "complete"
    assert json.loads((root / "method_validation.json").read_text())["status"] == "passed"
    assert len(verify_execution_bundle(root, protocol)["success"]) == 8
    # Synthetic fixture asserts no research_data while formal execution requires it.
    assert (project / "research_data/demo/train.csv").exists()
    monkeypatch.setattr("researchclaw.experiment.factory.create_sandbox", lambda *a: pytest.fail("must not rerun"))
    assert run_matrix(root, project, cfg.experiment)["status"] == "complete"


def test_failed_probe_prevents_first_heldout_call(inputs, spec, runtime_method):
    root, project, protocol, cfg = setup_matrix(inputs, spec, runtime_method)
    (project / "model.py").write_text(MODEL.replace("2 * v", "0 * v"), encoding="utf-8")
    with pytest.raises(ProtocolError, match="Pretest method validation"):
        run_matrix(root, project, cfg.experiment)
    assert not (root / "protocol_execution.jsonl").exists()
    assert not (root / "trusted_evaluation.json").exists()


def test_writing_packet_binds_validation_and_rejects_missing_evidence(tmp_path, runtime_method):
    from researchclaw.pipeline.manuscript import evidence_catalog, ManuscriptError
    root, method, code, cfg = prepared(tmp_path, runtime_method)
    run_validation(root, method, code, cfg)
    catalog = evidence_catalog(root)
    assert catalog["entries"]["method_validation"]["data"]["semantic_equivalence"] == "unresolved"
    (root / "method_validation.json").unlink()
    with pytest.raises(ManuscriptError, match="Invalid method validation"):
        evidence_catalog(root)


def test_assessment_requires_exact_ordered_invocations(runtime_method):
    method = compile_method(runtime_method)
    with pytest.raises(WorkbenchError, match="exact ordered"):
        assess(method, {"schema_version": 1, "observations": []})


def test_legacy_method_does_not_gain_runtime_verification(tmp_path, runtime_method):
    del runtime_method["validation"]
    root, method, code, cfg = prepared(tmp_path, runtime_method)
    assert run_validation(root, method, code, cfg) == {"status": "not_declared", "semantic_equivalence": "unresolved"}
    assert not (root / "method_validation.json").exists()


def test_missing_probe_evidence_after_test_execution_never_triggers_new_run(tmp_path, runtime_method, monkeypatch):
    root, method, code, cfg = prepared(tmp_path, runtime_method)
    (root / "protocol_execution.jsonl").write_text("existing ledger", encoding="utf-8")
    monkeypatch.setattr("researchclaw.experiment.factory.create_sandbox", lambda *a: pytest.fail("must not run"))
    with pytest.raises(WorkbenchError, match="after formal test execution"):
        run_validation(root, method, code, cfg)


def test_changed_source_cannot_reuse_passed_report(tmp_path, runtime_method):
    root, method, code, cfg = prepared(tmp_path, runtime_method)
    run_validation(root, method, code, cfg)
    (root / "evidence_artifacts/protocol_source/model.py").write_text("changed", encoding="utf-8")
    with pytest.raises(WorkbenchError, match="source archive changed"):
        verify_validation(root, method, code)


def test_backend_config_must_match_actual_executor(tmp_path, runtime_method):
    root, method, code, cfg = prepared(tmp_path, runtime_method)
    cfg = replace(cfg, sandbox=replace(cfg.sandbox, python_path="another-python"))
    with pytest.raises(WorkbenchError, match="frozen execution configuration"):
        run_validation(root, method, code, cfg)
    assert not (root / "evidence_artifacts/method_validation").exists()


def test_nan_json_output_keeps_execution_failure_evidence(tmp_path, runtime_method, monkeypatch):
    root, method, code, cfg = prepared(tmp_path, runtime_method)
    class BadOutput:
        def run_project(self, project, **kwargs):
            work = root / "method_validation_work"
            work.mkdir()
            (work / "observations.json").write_text('{"observations": [NaN]}', encoding="utf-8")
            return SandboxResult(0, "partial output", "", 0.1, {}, output_dir=str(work))
    monkeypatch.setattr("researchclaw.experiment.factory.create_sandbox", lambda *a: BadOutput())
    with pytest.raises(WorkbenchError):
        run_validation(root, method, code, cfg)
    receipt = json.loads((root / "evidence_artifacts/method_validation/execution.json").read_text())
    assert "Nonfinite JSON" in receipt["error"]
    assert json.loads((root / "method_validation.json").read_text())["status"] == "failed"


def test_final_acceptance_marks_missing_runtime_evidence(tmp_path, runtime_method):
    from researchclaw.pipeline.final_acceptance import assess_delivery
    root, method, code, cfg = prepared(tmp_path, runtime_method)
    report = assess_delivery(root, target_status="research_complete")
    found = [i for i in report["issues"] if i["reason"] == "method_runtime_validation_failed"]
    assert len(found) == 1 and found[0]["repair_owner"] == "experiments"


def test_portable_report_rechecks_without_original_project_or_interpreter(tmp_path, runtime_method):
    import shutil
    root, method, code, cfg = prepared(tmp_path, runtime_method)
    report = run_validation(root, method, code, cfg)
    bundle = tmp_path / "portable"
    bundle.mkdir()
    shutil.copytree(root / "evidence_artifacts", bundle / "evidence_artifacts")
    shutil.copyfile(root / "method_validation.json", bundle / "method_validation.json")
    assert verify_validation(bundle, method, code) == report


def test_validation_recorded_after_test_is_rejected_even_with_fresh_hashes(inputs, spec, runtime_method):
    from researchclaw.pipeline.method_validation import make_report
    root, project, protocol, cfg = setup_matrix(inputs, spec, runtime_method)
    run_matrix(root, project, cfg.experiment)
    method = json.loads((root / "method_spec.json").read_text())
    code = json.loads((root / "protocol_code.json").read_text())
    artifact = root / "evidence_artifacts/method_validation"
    receipt = json.loads((artifact / "execution.json").read_text())
    receipt["finished_at"] += 10000
    dump(artifact / "execution.json", receipt)
    rows = json.loads((artifact / "observations.json").read_text())
    dump(root / "method_validation.json", make_report(method, code, artifact, assess(method, rows), receipt))
    with pytest.raises(ProtocolError, match="did not precede"):
        verify_execution_bundle(root, protocol)


@pytest.mark.parametrize("target", ["reservation.json", "execution.json", "report_files"])
def test_malformed_evidence_is_validation_failure_even_when_rehashed(tmp_path, runtime_method, target):
    root, method, code, cfg = prepared(tmp_path, runtime_method)
    report = run_validation(root, method, code, cfg)
    if target == "report_files":
        report["files"] = list(report["files"])
    else:
        path = root / "evidence_artifacts/method_validation" / target
        dump(path, [])
        report["files"][target] = file_hash(path)
    report.pop("version")
    report["version"] = content_hash(report)
    dump(root / "method_validation.json", report)
    with pytest.raises(WorkbenchError):
        verify_validation(root, method, code)
