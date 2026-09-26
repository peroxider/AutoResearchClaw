import copy
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from researchclaw.config import ExperimentConfig, SandboxConfig
from researchclaw.experiment.method_validation_runner import run_validation
from researchclaw.experiment.method_probe import apply_rng_policy
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


def prepared(tmp_path, runtime_method, model=MODEL, extra_sources=None):
    root = tmp_path / "probe-run"
    source = root / "evidence_artifacts" / "protocol_source"
    source.mkdir(parents=True)
    (source / "model.py").write_text(model, encoding="utf-8")
    (source / "main.py").write_text("raise RuntimeError('main must not execute in probes')", encoding="utf-8")
    for name, text in (extra_sources or {}).items():
        (source / name).write_text(text, encoding="utf-8")
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


STATE_MODEL = '''from pathlib import Path
assert not Path("research_data").exists(), "probe received study data"
CALLS = []
def loss(x):
    return sum(v * v for v in x)
class Model:
    def __init__(self, factor=1): self.factor = factor
    def predict(self, x, enabled):
        CALLS.append(1)
        return [v * self.factor + int(enabled) + len(CALLS) for v in x]
'''


# A minimal reverse-mode autodiff stand-in placed as torch.py inside the frozen
# source archive; the probe driver shadows any installed backend with it.
TORCH_STUB = '''"""Minimal reverse-mode autodiff stand-in for probe tests (not a real backend)."""

class _Node:
    def __init__(self, value, parents=(), requires_grad=False):
        self.value, self.parents, self.requires_grad, self.grad = value, parents, requires_grad, 0.0

class Tensor:
    def __init__(self, value=None, requires_grad=False, node=None):
        self.grad = None
        if node is not None:
            self._node, self._leaves, self.requires_grad = node, None, node.requires_grad
            return
        if isinstance(value, list):
            def walk(item):
                if isinstance(item, list):
                    return [walk(child) for child in item]
                return _Node(float(item), requires_grad=requires_grad)
            self._leaves, self._node, self.requires_grad = walk(value), None, requires_grad
        else:
            self._leaves, self._node = None, _Node(float(value), requires_grad=requires_grad)
            self.requires_grad = requires_grad

    def __iter__(self):
        for leaf in self._leaves:
            yield Tensor(node=leaf)

    def __mul__(self, other):
        a, b = self._node, other._node
        return Tensor(node=_Node(a.value * b.value, [(a, b.value), (b, a.value)],
                                 a.requires_grad or b.requires_grad))

    def __add__(self, other):
        a, b = self._node, other._node
        return Tensor(node=_Node(a.value + b.value, [(a, 1.0), (b, 1.0)],
                                 a.requires_grad or b.requires_grad))

    def __radd__(self, other):
        return self + Tensor(float(other))

    def dim(self):
        return 0 if self._leaves is None else 1

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self.grad

    def backward(self):
        order, seen = [], set()
        def visit(node):
            if id(node) in seen:
                return
            seen.add(id(node))
            for parent, _ in node.parents:
                visit(parent)
            order.append(node)
        visit(self._node)
        self._node.grad = 1.0
        for node in reversed(order):
            for parent, local in node.parents:
                parent.grad += node.grad * local
        for container in _CONTAINERS:
            container.grad = [leaf.grad if leaf.requires_grad else None for leaf in container._leaves]

_CONTAINERS = []

float64 = "float64"

def manual_seed(seed):
    pass

def tensor(value, dtype=None, requires_grad=False):
    result = Tensor(value, requires_grad=requires_grad)
    if result._leaves is not None:
        _CONTAINERS.append(result)
    return result
'''


def autodiff_case(**overrides):
    case = {"id": "autograd", "kind": "autodiff", "atol": 1e-6, "rtol": 1e-6,
            "call": {"step": "loss", "kwargs": {"x": [2.0, -3.0]}}, "argument": "x", "epsilon": 0.0001}
    case.update(overrides)
    return case


def determinism_case(**overrides):
    case = {"id": "stable", "kind": "determinism", "atol": 0, "rtol": 0,
            "call": {"step": "predict", "kwargs": {"x": [2, 3], "enabled": True},
                     "constructor_kwargs": {"factor": 2}}}
    case.update(overrides)
    return case


def process_determinism_case(**overrides):
    case = determinism_case(id="portable", kind="process_determinism")
    case.update(overrides)
    return case


def device_inventory_case(**overrides):
    case = {"id": "devices", "kind": "device_inventory", "required_devices": ["cpu"],
            "minimum_cuda_devices": 0, "require_torch_determinism": False}
    case.update(overrides)
    return case


def test_rng_policy_applies_torch_and_all_cuda_device_controls(monkeypatch):
    calls = []
    cudnn = SimpleNamespace(deterministic=False, benchmark=True)
    cuda = SimpleNamespace(
        is_available=lambda: True,
        manual_seed_all=lambda seed: calls.append(("cuda_all", seed)))
    torch = SimpleNamespace(
        manual_seed=lambda seed: calls.append(("torch", seed)), cuda=cuda,
        use_deterministic_algorithms=lambda enabled: calls.append(("deterministic", enabled)),
        backends=SimpleNamespace(cudnn=cudnn))
    monkeypatch.setitem(sys.modules, "torch", torch)
    result = apply_rng_policy({
        "runtime_policy": {"seed": 0, "enforce_torch_determinism": True}})
    assert result["torch"] == "applied" and result["cuda_all"] == "applied"
    assert calls == [("torch", 0), ("cuda_all", 0), ("deterministic", True)]
    assert cudnn.deterministic is True and cudnn.benchmark is False


def runtime_state(pid, *, torch_status="unavailable", cuda_count=0, deterministic=False):
    available = torch_status == "available"
    devices = [{"index": index, "name": f"GPU {index}", "capability": [8, 0],
                "total_memory": 1024} for index in range(cuda_count)] if available else []
    return {"process_id": pid, "python_version": "3.12.0", "implementation": "CPython",
            "platform": "test", "environment": {"CUDA_VISIBLE_DEVICES": None,
                                                   "CUBLAS_WORKSPACE_CONFIG": None,
                                                   "CUDA_LAUNCH_BLOCKING": None},
            "rng_policy": {"seed": 0, "python": "applied", "numpy": "applied",
                           "torch": "applied" if available else "unavailable",
                           "cuda_all": "applied" if cuda_count else (
                               "not_available" if available else "unavailable"),
                           "enforce_torch_determinism": deterministic},
            "torch": {"status": torch_status, "version": "test" if available else None,
                      "cuda_available": cuda_count > 0 if available else None,
                      "cuda_device_count": cuda_count if available else None,
                      "cuda_devices": devices, "mps_available": False if available else None,
                      "deterministic_algorithms": deterministic if available else None,
                      "cudnn_deterministic": False if available else None,
                      "cudnn_benchmark": False if available else None,
                      "cudnn_version": None}}


def runtime_inventory(controller=100, fresh=()):
    return {"schema_version": 1, "controller": runtime_state(controller),
            "fresh_processes": [{"call_id": call_id, "state": runtime_state(pid)}
                                for call_id, pid in fresh]}


@pytest.mark.parametrize("mutation", [
    lambda c: c["call"].update(step="predict"),
    lambda c: c.update(argument="not_input"),
    lambda c: c.update(epsilon=0.02),
    lambda c: c.update(epsilon=True),
    lambda c: c["call"]["kwargs"].update(x=[1.0] * 65),
    lambda c: c["call"]["kwargs"].update(x=[1e100]),
    lambda c: c.update(expected=[0]),
    lambda c: c["call"]["kwargs"].update(x=[[1.0], [2.0]]),
])
def test_invalid_autodiff_plan_fails_before_execution(runtime_method, mutation):
    runtime_method["validation"]["cases"] = [autodiff_case()]
    mutation(runtime_method["validation"]["cases"][0])
    with pytest.raises(WorkbenchError):
        compile_method(runtime_method)


def test_autodiff_invocation_budget(runtime_method):
    case = autodiff_case()
    case["call"]["kwargs"]["x"] = [1.0] * 64
    runtime_method["validation"]["cases"] = [case, dict(copy.deepcopy(case), id="second")]
    with pytest.raises(WorkbenchError, match="invocation budget"):
        compile_method(runtime_method)


def test_autodiff_probe_cross_checked_against_finite_differences(tmp_path, runtime_method, monkeypatch):
    runtime_method["validation"]["cases"] = [autodiff_case()]
    root, method, code, cfg = prepared(tmp_path, runtime_method, extra_sources={"torch.py": TORCH_STUB})
    report = run_validation(root, method, code, cfg)
    assert report["status"] == "passed" and report["coverage"]["kinds"] == ["autodiff"]
    assert any("not a derivative proof" in line for line in report["limitations"])
    request = build_request(method)
    assert [c["id"] for c in request["calls"]] == [
        "autograd:autodiff", "autograd:0:plus", "autograd:0:minus", "autograd:1:plus", "autograd:1:minus"]
    assert request["calls"][0]["mode"] == "autodiff" and request["calls"][0]["argument"] == "x"
    assert "expected" not in json.dumps(request)
    rows = json.loads((root / "evidence_artifacts/method_validation/observations.json").read_text())["observations"]
    by_id = {row["id"]: row.get("value") for row in rows}
    assert by_id["autograd:autodiff"] == [4.0, -6.0]
    assert by_id["autograd:0:plus"] == (2.0 + 0.0001) ** 2 + 9.0
    monkeypatch.setattr("researchclaw.experiment.factory.create_sandbox", lambda *a: pytest.fail("must not rerun"))
    assert run_validation(root, method, code, cfg) == report


def test_autodiff_without_backend_fails_without_degrading(tmp_path, runtime_method):
    import importlib.util
    if importlib.util.find_spec("torch") is not None:
        pytest.skip("a real torch backend is installed; the unavailable path needs one without it")
    runtime_method["validation"]["cases"] = [autodiff_case()]
    root, method, code, cfg = prepared(tmp_path, runtime_method)
    with pytest.raises(WorkbenchError):
        run_validation(root, method, code, cfg)
    report = json.loads((root / "method_validation.json").read_text())
    assert report["status"] == "failed"
    assert "PyTorch backend is required" in json.dumps(report["checks"])


def test_assess_autodiff_matches_exact_finite_difference(runtime_method):
    runtime_method["validation"]["cases"] = [autodiff_case()]
    method = compile_method(runtime_method)
    base, epsilon = [2.0, -3.0], 0.0001
    rows = [{"id": "autograd:autodiff", "value": [4.0, -6.0]}]
    for index in range(2):
        for sign, role in ((1, "plus"), (-1, "minus")):
            perturbed = list(base)
            perturbed[index] += sign * epsilon
            rows.append({"id": f"autograd:{index}:{role}", "value": sum(v * v for v in perturbed)})
    results = assess(method, {"schema_version": 1, "observations": rows})
    assert results[0]["status"] == "passed"
    assert results[0]["checks"][0]["comparison"] == "autograd_vs_central_finite_difference"


def test_assess_autodiff_wrong_gradient_and_shape_fail(runtime_method):
    runtime_method["validation"]["cases"] = [autodiff_case()]
    method = compile_method(runtime_method)
    epsilon = 0.0001
    def rows_with(autograd_value):
        rows = [{"id": "autograd:autodiff", "value": autograd_value}]
        for index in range(2):
            for sign, role in ((1, "plus"), (-1, "minus")):
                perturbed = [2.0, -3.0]
                perturbed[index] += sign * epsilon
                rows.append({"id": f"autograd:{index}:{role}", "value": sum(v * v for v in perturbed)})
        return rows
    wrong = assess(method, {"schema_version": 1, "observations": rows_with([4.0, -5.0])})
    assert wrong[0]["status"] == "failed" and wrong[0]["checks"][0]["status"] == "failed"
    shaped = assess(method, {"schema_version": 1, "observations": rows_with([[4.0, -6.0]])})
    assert shaped[0]["status"] == "failed" and "shape differs" in shaped[0]["error"]


def test_determinism_repeat_runs_last_and_passes_when_stateless(tmp_path, runtime_method, monkeypatch):
    runtime_method["validation"]["cases"] = [determinism_case()]
    root, method, code, cfg = prepared(tmp_path, runtime_method)
    report = run_validation(root, method, code, cfg)
    assert report["status"] == "passed" and report["coverage"]["kinds"] == ["determinism"]
    assert [c["id"] for c in build_request(method)["calls"]] == ["stable:first", "stable:repeat"]
    monkeypatch.setattr("researchclaw.experiment.factory.create_sandbox", lambda *a: pytest.fail("must not rerun"))
    assert run_validation(root, method, code, cfg) == report


def test_determinism_detects_state_pollution_across_calls(tmp_path, runtime_method, monkeypatch):
    runtime_method["validation"]["cases"] = [determinism_case()]
    root, method, code, cfg = prepared(tmp_path, runtime_method, model=STATE_MODEL)
    with pytest.raises(WorkbenchError):
        run_validation(root, method, code, cfg)
    report = json.loads((root / "method_validation.json").read_text())
    check = next(c for c in report["checks"] if c["id"] == "stable")
    assert check["status"] == "failed"
    assert check["checks"][0]["comparison"] == "repeat_after_other_calls"
    monkeypatch.setattr("researchclaw.experiment.factory.create_sandbox", lambda *a: pytest.fail("must not retry"))
    with pytest.raises(WorkbenchError):
        run_validation(root, method, code, cfg)


def test_determinism_reseeds_python_rng_before_each_declared_call(tmp_path, runtime_method):
    runtime_method["validation"]["cases"] = [determinism_case()]
    model = MODEL.replace(
        "from pathlib import Path", "import random\nfrom pathlib import Path").replace(
        "return [v * self.factor + int(enabled) for v in x]",
        "return [random.random() for _ in x]")
    root, method, code, cfg = prepared(tmp_path, runtime_method, model=model)
    report = run_validation(root, method, code, cfg)
    assert report["status"] == "passed"
    rows = json.loads((root / "evidence_artifacts/method_validation/observations.json").read_text())[
        "observations"]
    assert rows[0]["value"] == rows[1]["value"]


def test_determinism_repeat_is_appended_after_every_other_call(runtime_method):
    golden = runtime_method["validation"]["cases"][0]
    runtime_method["validation"]["cases"] = [determinism_case(), golden]
    request = build_request(compile_method(runtime_method))
    assert [c["id"] for c in request["calls"]] == ["stable:first", "golden:value", "stable:repeat"]


def test_assess_flags_determinism_mismatch(runtime_method):
    runtime_method["validation"]["cases"] = [determinism_case()]
    method = compile_method(runtime_method)
    rows = [{"id": "stable:first", "value": [5, 7]}, {"id": "stable:repeat", "value": [5, 8]}]
    results = assess(method, {"schema_version": 1, "observations": rows})
    assert results[0]["status"] == "failed"
    assert results[0]["checks"][0]["comparison"] == "repeat_after_other_calls"
    failed_invocation = assess(method, {"schema_version": 1, "observations": [
        {"id": "stable:first", "value": [5, 7]}, {"id": "stable:repeat", "error": "ValueError: broke"}]})
    assert failed_invocation[0]["status"] == "failed" and "Invocation failed" in failed_invocation[0]["error"]


def test_fresh_process_determinism_uses_distinct_interpreters(tmp_path, runtime_method):
    runtime_method["validation"]["cases"] = [process_determinism_case()]
    root, method, code, cfg = prepared(tmp_path, runtime_method, model=STATE_MODEL)
    report = run_validation(root, method, code, cfg)
    assert report["status"] == "passed" and report["coverage"]["kinds"] == ["process_determinism"]
    request = build_request(method)
    assert request["fresh_process_timeout_seconds"] == 5
    assert [call["mode"] for call in request["calls"]] == ["fresh_process", "fresh_process"]
    runtime = report["runtime_environment"]
    pids = [entry["state"]["process_id"] for entry in runtime["fresh_processes"]]
    assert len(set(pids)) == 2 and runtime["controller"]["process_id"] not in pids
    check = report["checks"][0]
    assert [item["comparison"] for item in check["checks"]] == [
        "fresh_python_process_repeat", "distinct_fresh_processes"]


def test_fresh_process_determinism_detects_process_identity(tmp_path, runtime_method):
    runtime_method["validation"]["cases"] = [process_determinism_case()]
    model = MODEL.replace("from pathlib import Path", "import os\nfrom pathlib import Path").replace(
        "return [v * self.factor + int(enabled) for v in x]", "return [os.getpid()]")
    root, method, code, cfg = prepared(tmp_path, runtime_method, model=model)
    with pytest.raises(WorkbenchError):
        run_validation(root, method, code, cfg)
    report = json.loads((root / "method_validation.json").read_text())
    assert report["checks"][0]["checks"][0]["comparison"] == "fresh_python_process_repeat"
    assert report["checks"][0]["checks"][0]["status"] == "failed"


def test_fresh_process_assessment_requires_complete_distinct_runtime(runtime_method):
    runtime_method["validation"]["cases"] = [process_determinism_case()]
    method = compile_method(runtime_method)
    rows = [{"id": "portable:first", "value": [5, 7]},
            {"id": "portable:repeat", "value": [5, 7]}]
    missing = assess(method, {"schema_version": 1, "observations": rows})
    assert missing[0]["status"] == "failed" and "runtime inventory" in missing[0]["error"]
    same = runtime_inventory(fresh=(("portable:first", 101), ("portable:repeat", 101)))
    result = assess(method, {"schema_version": 1, "observations": rows, "runtime": same})
    assert result[0]["status"] == "failed"
    assert result[0]["checks"][-1]["comparison"] == "distinct_fresh_processes"


@pytest.mark.parametrize("mutation", [
    lambda c: c.update(required_devices=[]),
    lambda c: c.update(required_devices=["cpu", "cpu"]),
    lambda c: c.update(required_devices=["tpu"]),
    lambda c: c.update(minimum_cuda_devices=True),
    lambda c: c.update(minimum_cuda_devices=-1),
    lambda c: c.update(minimum_cuda_devices=65),
    lambda c: c.update(require_torch_determinism=1),
    lambda c: c.update(extra=True),
])
def test_invalid_device_inventory_plan_fails_before_execution(runtime_method, mutation):
    case = device_inventory_case()
    mutation(case)
    runtime_method["validation"]["cases"] = [case]
    with pytest.raises(WorkbenchError):
        compile_method(runtime_method)


def test_device_inventory_records_pre_source_runtime_without_forged_torch(tmp_path, runtime_method):
    runtime_method["validation"]["cases"] = [device_inventory_case()]
    root, method, code, cfg = prepared(tmp_path, runtime_method, extra_sources={"torch.py": TORCH_STUB})
    report = run_validation(root, method, code, cfg)
    assert report["status"] == "passed" and report["coverage"]["tested_steps"] == []
    assert build_request(method)["calls"] == []
    torch = report["runtime_environment"]["controller"]["torch"]
    rng = report["runtime_environment"]["controller"]["rng_policy"]
    assert torch["version"] != "fixture"
    assert torch["status"] in {"available", "unavailable"} or torch["status"].startswith("error:")
    assert rng["seed"] == 0 and rng["python"] == "applied"
    assert rng["numpy"] in {"applied", "unavailable"} or rng["numpy"].startswith("error:")
    assert rng["enforce_torch_determinism"] is False
    assert report["checks"][0]["checks"][0]["available"]["cpu"] is True


def test_device_requirements_are_checked_from_runtime_inventory(runtime_method):
    runtime_method["validation"]["cases"] = [device_inventory_case(
        required_devices=["cpu", "cuda"], minimum_cuda_devices=2,
        require_torch_determinism=True)]
    method = compile_method(runtime_method)
    assert build_request(method)["runtime_policy"] == {
        "seed": 0, "enforce_torch_determinism": True}
    passing_runtime = runtime_inventory()
    passing_runtime["controller"] = runtime_state(100, torch_status="available", cuda_count=2,
                                                  deterministic=True)
    passed = assess(method, {"schema_version": 1, "observations": [], "runtime": passing_runtime})
    assert passed[0]["status"] == "passed"
    assert passed[0]["checks"][-1]["comparison"] == "rng_seed_policy"
    assert passed[0]["checks"][-1]["status"] == "passed"
    missing_cuda_seed = copy.deepcopy(passing_runtime)
    missing_cuda_seed["controller"]["rng_policy"]["cuda_all"] = "not_available"
    unseeded = assess(method, {"schema_version": 1, "observations": [],
                               "runtime": missing_cuda_seed})
    assert unseeded[0]["status"] == "failed"
    assert unseeded[0]["checks"][-1]["comparison"] == "rng_seed_policy"
    passing_runtime["controller"]["torch"]["deterministic_algorithms"] = False
    failed = assess(method, {"schema_version": 1, "observations": [], "runtime": passing_runtime})
    assert failed[0]["status"] == "failed"
    assert next(item for item in failed[0]["checks"]
                if item["comparison"] == "torch_deterministic_algorithms")["status"] == "failed"
    assert failed[0]["checks"][-1]["comparison"] == "rng_seed_policy"


def test_fresh_process_timeout_and_runtime_schema_are_strict(runtime_method):
    runtime_method["validation"]["cases"] = [process_determinism_case()]
    runtime_method["validation"]["timeout_seconds"] = 1
    with pytest.raises(WorkbenchError, match="at least two seconds"):
        compile_method(runtime_method)
    runtime_method["validation"]["timeout_seconds"] = 10
    method = compile_method(runtime_method)
    rows = [{"id": "portable:first", "value": [5, 7]},
            {"id": "portable:repeat", "value": [5, 7]}]
    malformed = runtime_inventory(fresh=(("portable:first", 101), ("portable:repeat", 102)))
    malformed["controller"]["torch"]["cuda_available"] = False
    with pytest.raises(WorkbenchError, match="device claims"):
        assess(method, {"schema_version": 1, "observations": rows, "runtime": malformed})
    wrong_seed = runtime_inventory(fresh=(("portable:first", 101), ("portable:repeat", 102)))
    wrong_seed["fresh_processes"][1]["state"]["rng_policy"]["seed"] = 7
    with pytest.raises(WorkbenchError, match="RNG policy"):
        assess(method, {"schema_version": 1, "observations": rows, "runtime": wrong_seed})
