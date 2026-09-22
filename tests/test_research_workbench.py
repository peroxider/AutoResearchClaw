import copy
import json

import pytest

from researchclaw.pipeline.evidence_store import content_hash, file_hash
from researchclaw.pipeline.research_workbench import (
    WorkbenchError, check_implementation, compile_method, compile_theory,
)
from researchclaw.research_inputs import prepare_inputs, verify_bundle_contract
from tests.test_research_inputs import inputs, dump
from tests.test_experiment_protocol import spec


@pytest.fixture
def method():
    return {"schema_version": 1, "method_id": "A_full", "description": "linear projection",
            "variables": {"X": {"shape": ["n", "d"], "description": "inputs"},
                          "W": {"shape": ["d", 1], "description": "weights"},
                          "Y": {"shape": ["n", 1], "description": "predictions"}},
            "equations": [{"id": "projection", "latex": "Y = X W", "inputs": ["X", "W"], "outputs": ["Y"]}],
            "steps": [{"id": "predict", "phase": "inference", "description": "Multiply inputs by weights",
                       "inputs": ["X", "W"], "outputs": ["Y"], "equations": ["projection"], "depends_on": [],
                       "code_file": "model.py", "code_symbol": "Linear.predict"}],
            "losses": [], "stopping_rule": "One inference pass", "complexity": {"time": "O(nd)", "space": "O(n+d)"},
            "shape_checks": [{"op": "matmul", "left": "X", "right": "W", "result": "Y"}]}


def theory(left="(x+y)**2", right="x**2 + 2*x*y + y**2"):
    return {"schema_version": 1, "definitions": {"domain": "x and y are real numbers"}, "obligations": [{
        "id": "identity", "required": True, "depends_on": [], "assumptions": [],
        "statement": {"kind": "polynomial_identity", "variables": ["x", "y"], "left": left, "right": right}}]}


def test_method_maps_exact_qualified_symbol_without_importing(tmp_path, method):
    document = compile_method(method)
    assert document["semantic_equivalence"] == "unresolved"
    (tmp_path / "model.py").write_text("raise RuntimeError('never imported')\nclass Linear:\n    def predict(self, x):\n        return x\n")
    report = check_implementation(document, tmp_path)
    assert report["status"] == "mapped" and report["bindings"][0]["line"] == 3
    # Existence only: this intentionally wrong body is NOT declared semantically verified.
    assert report["semantic_equivalence"] == "unresolved"
    (tmp_path / "model.py").write_text("def predict(x): return x\n")
    assert check_implementation(document, tmp_path)["status"] == "failed"


@pytest.mark.parametrize("mutation", [
    lambda s: s["variables"]["W"].update(shape=["other_dimension", 1]),
    lambda s: s["steps"][0].update(equations=["missing"]),
    lambda s: s["steps"][0].update(depends_on=["predict"]),
    lambda s: s["steps"][0].update(code_file="../model.py"),
    lambda s: s["steps"][0].update(code_symbol="model()"),
    lambda s: s["steps"][0].update(phase="both"),
    lambda s: s["steps"][0].update(equations=[]),
    lambda s: s["variables"]["X"].update(shape=[0, "d"]),
    lambda s: s["shape_checks"][0].update(op="invented"),
    lambda s: s["equations"].append(copy.deepcopy(s["equations"][0])),
    lambda s: s.update(steps=[]),
])
def test_invalid_method_contract_fails(method, mutation):
    mutation(method)
    with pytest.raises(WorkbenchError):
        compile_method(method)


@pytest.mark.parametrize("left,right", [
    ("(x+y)**2", "x**2+2*x*y+y**2"),
    ("x/3 + x/6", "x/2"), ("(x-y)*(x+y)", "x**2-y**2"),
    ("0*x", "0"), ("-(x+y)", "-x-y"),
])
def test_exact_polynomial_identities_are_machine_checked(left, right):
    bundle = compile_theory(theory(left, right))
    obligation = bundle["obligations"][0]
    assert obligation["status"] == "machine_checked"
    assert obligation["evidence"]["exact_arithmetic"] is True


def test_false_identity_has_exact_counterexample():
    obligation = compile_theory(theory("(x+y)**2", "x**2+y**2"))["obligations"][0]
    assert obligation["status"] == "disproved"
    witness = obligation["evidence"]["counterexample"]
    assert witness["left"] != witness["right"]


@pytest.mark.parametrize("expression", [
    "__import__('os').system('echo forbidden')", "x.real", "[x for x in y]", "x/y",
    "x/0", "x**10000", "True", "0.1*x", "z+1", "x**y",
])
def test_checker_rejects_code_execution_and_unsupported_expressions(expression):
    with pytest.raises(WorkbenchError):
        compile_theory(theory(expression, "x"))


def test_unreviewed_text_proof_stays_unresolved_and_blocks_dependents():
    spec = theory()
    spec["obligations"].insert(0, {"id": "general", "required": True, "depends_on": [], "assumptions": ["A hypothesis"],
                                 "statement": {"kind": "informal", "text": "A general convergence theorem"},
                                 "proof_text": "This is an unreviewed proof attempt."})
    spec["obligations"][1]["depends_on"] = ["general"]
    bundle = compile_theory(spec)
    assert all(o["status"] == "unresolved" for o in bundle["obligations"])
    assert bundle["obligations"][1]["evidence"]["unresolved_dependencies"] == ["general"]


def test_proof_dependency_cycle_and_missing_parent_fail():
    for depends in (["identity"], ["missing"]):
        spec = theory()
        spec["obligations"][0]["depends_on"] = depends
        with pytest.raises(WorkbenchError):
            compile_theory(spec)


def test_informal_review_is_not_machine_proof():
    spec = theory()
    item = spec["obligations"][0]
    item["statement"] = {"kind": "informal", "text": "A reviewed general statement"}
    item["proof_text"] = "Full proof attempt"
    item["review"] = {"checker": "fixture independent reviewer", "verdict": "accepted", "evidence": "fixture review rationale"}
    assert compile_theory(spec)["obligations"][0]["status"] == "reviewed_informal"
    item["review"]["verdict"] = "machine_checked"
    with pytest.raises(WorkbenchError):
        compile_theory(spec)


def test_informal_dependency_prevents_fully_machine_checked_label():
    spec = theory()
    spec["obligations"].insert(0, {"id": "informal", "required": True, "depends_on": [], "assumptions": [],
        "statement": {"kind": "informal", "text": "An informally reviewed prerequisite"}, "proof_text": "Proof text",
        "review": {"checker": "fixture", "verdict": "accepted", "evidence": "review rationale"}})
    spec["obligations"][1]["depends_on"] = ["informal"]
    assert compile_theory(spec)["obligations"][1]["status"] == "reviewed_informal"


def test_declared_method_mapping_is_checked_before_any_test_execution(inputs, spec, method):
    import sys
    from dataclasses import replace
    from researchclaw.experiment.protocol_runner import run_matrix
    from researchclaw.pipeline.experiment_protocol import ProtocolError
    from tests.test_research_inputs import make_config
    brief, root, _, data = inputs
    data.update(protocol_path="protocol.json", method_spec_path="method.json")
    dump(brief, data)
    dump(brief.parent / "protocol.json", spec)
    dump(brief.parent / "method.json", method)
    prepare_inputs(brief, root)
    project = root / "stage-10" / "experiment"
    project.mkdir(parents=True)
    (project / "main.py").write_text("raise RuntimeError('must not execute')")
    cfg = make_config(root, brief)
    cfg = replace(cfg.experiment, sandbox=replace(cfg.experiment.sandbox, python_path=sys.executable))
    with pytest.raises(ProtocolError, match="MethodSpec code mapping failed"):
        run_matrix(root, project, cfg)
    assert not (root / "protocol_execution.jsonl").exists()


def test_workbench_inputs_freeze_and_recompile_in_delivery(inputs, spec, method):
    brief, root, _, data = inputs
    data.update(protocol_path="protocol.json", method_spec_path="method.json", theory_bundle_path="theory.json")
    dump(brief, data)
    dump(brief.parent / "protocol.json", spec)
    dump(brief.parent / "method.json", method)
    dump(brief.parent / "theory.json", theory())
    contract = prepare_inputs(brief, root)
    assert {"method_spec.json", "theory_bundle.json"} <= contract["outputs"].keys()
    assert verify_bundle_contract(root) == contract
    bundle = json.loads((root / "theory_bundle.json").read_text())
    bundle["obligations"][0]["status"] = "reviewed_informal"
    bundle.pop("version")
    bundle["version"] = content_hash(bundle)
    dump(root / "theory_bundle.json", bundle)
    contract["outputs"]["theory_bundle.json"] = file_hash(root / "theory_bundle.json")
    contract.pop("version")
    contract["version"] = content_hash(contract)
    dump(root / "research_contract.json", contract)
    with pytest.raises(WorkbenchError, match="theory_bundle"):
        verify_bundle_contract(root)


def test_final_acceptance_rechecks_proof_instead_of_status_string(tmp_path):
    from researchclaw.pipeline.final_acceptance import assess_delivery
    bundle = compile_theory(theory("x+y", "x-y"))
    bundle["obligations"][0]["status"] = "machine_checked"
    dump(tmp_path / "theory_bundle.json", bundle)
    report = assess_delivery(tmp_path)
    assert report["dimensions"]["theory"] == "failed"
    assert any(i["reason"] == "changed_or_invalid_proof_check" for i in report["issues"])
