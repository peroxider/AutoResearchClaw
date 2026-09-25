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


def _extended(method):
    """Fixture variables covering every richer shape op, including rejection targets."""
    method["variables"].update({
        "A": {"shape": ["n", "d"], "description": "activations", "dtype": "float32"},
        "A2": {"shape": ["n", "d"], "description": "residual branch", "dtype": "float32"},
        "A64": {"shape": ["n", "d"], "description": "wide activations", "dtype": "float64"},
        "A64T": {"shape": ["d", "n"], "description": "transposed wide activations", "dtype": "float64"},
        "A_sum": {"shape": ["n"], "description": "row sum", "dtype": "float32"},
        "A_sum_keep": {"shape": ["n", 1], "description": "wrongly kept axis"},
        "S": {"shape": ["n", "d"], "description": "combined activations", "dtype": "float32"},
        "B": {"shape": ["d", "h"], "description": "projection", "dtype": "float32"},
        "B_t": {"shape": ["h", "d"], "description": "transposed projection", "dtype": "float32"},
        "V": {"shape": ["n"], "description": "one-dimensional activations"},
        "M": {"shape": ["m", "d"], "description": "other rows"},
        "F": {"shape": [2, 3, 4], "description": "features"},
        "G": {"shape": [6, 4], "description": "flattened features"},
        "G_bad": {"shape": [7, 4], "description": "wrong element count"},
        "G2": {"shape": [4], "description": "twice-reduced features"},
        "P": {"shape": [2, 3], "description": "left block"},
        "Q": {"shape": [4, 3], "description": "right block"},
        "PQ": {"shape": [6, 3], "description": "concatenated block"},
        "PQ_bad": {"shape": [5, 3], "description": "wrong concatenated length"},
        "Ps": {"shape": ["n", 3], "description": "symbolic left block"},
        "Qs": {"shape": ["m", 3], "description": "symbolic right block"},
        "PQs": {"shape": ["k", 3], "description": "symbolic concatenated block"},
        "Y2": {"shape": ["n", 1], "description": "biased predictions"},
        "bias": {"shape": [1, 1], "description": "scalar bias"},
    })
    return method


def test_richer_shape_ops_and_declared_dtypes_compile(method):
    method = _extended(method)
    method["shape_checks"] = [
        {"op": "elementwise_mul", "left": "A", "right": "A2", "result": "S"},
        {"op": "broadcast_add", "left": "A", "right": "A2", "result": "S"},
        {"op": "broadcast_add", "left": "Y", "right": "bias", "result": "Y2"},
        {"op": "transpose", "left": "B", "result": "B_t"},
        {"op": "reshape", "left": "F", "result": "G"},
        {"op": "concat", "left": "P", "right": "Q", "result": "PQ", "axis": 0},
        {"op": "reduce", "left": "A", "result": "A_sum", "axes": [1]},
        {"op": "cast", "left": "A", "result": "A64", "dtype": "float64"},
    ]
    document = compile_method(method)
    assert document["step_order"] == ["predict"]
    # Undeclared dtypes impose no constraint on the other operands.
    del method["variables"]["A2"]["dtype"], method["variables"]["S"]["dtype"]
    assert compile_method(method)["structure_status"] == "verified"


def _dtype_mismatch(s):
    s["variables"]["A2"].update(dtype="float64")
    s["shape_checks"].append({"op": "elementwise_mul", "left": "A", "right": "A2", "result": "S"})


@pytest.mark.parametrize("mutation", [
    lambda s: s["shape_checks"].append({"op": "broadcast_add", "left": "A", "right": "A2", "result": "B_t"}),
    lambda s: s["shape_checks"].append({"op": "broadcast_add", "left": "A", "right": "M", "result": "S"}),
    lambda s: s["shape_checks"].append({"op": "broadcast_add", "left": "A", "right": "V", "result": "S"}),
    lambda s: s["shape_checks"].append({"op": "concat", "left": "Ps", "right": "Qs", "result": "PQs", "axis": 0}),
    lambda s: s["shape_checks"].append({"op": "concat", "left": "P", "right": "Q", "result": "PQ_bad", "axis": 0}),
    lambda s: s["shape_checks"].append({"op": "concat", "left": "P", "right": "Q", "result": "PQ", "axis": 2}),
    lambda s: s["shape_checks"].append({"op": "transpose", "left": "V", "result": "V"}),
    lambda s: s["shape_checks"].append({"op": "transpose", "left": "B", "result": "B"}),
    lambda s: s["shape_checks"].append({"op": "transpose", "left": "B", "result": "B_t", "right": "A"}),
    lambda s: s["shape_checks"].append({"op": "reshape", "left": "F", "result": "G_bad"}),
    lambda s: s["shape_checks"].append({"op": "reshape", "left": "A", "result": "S"}),
    lambda s: s["shape_checks"].append({"op": "reduce", "left": "A", "result": "A_sum", "axes": [2]}),
    lambda s: s["shape_checks"].append({"op": "reduce", "left": "F", "result": "G2", "axes": [0, 0]}),
    lambda s: s["shape_checks"].append({"op": "reduce", "left": "A", "result": "A_sum_keep", "axes": [1]}),
    lambda s: s["shape_checks"].append({"op": "elementwise_mul", "left": "A", "right": "B", "result": "S"}),
    lambda s: s["variables"]["A"].update(dtype="float128"),
    _dtype_mismatch,
    lambda s: s["shape_checks"].append({"op": "cast", "left": "X", "result": "Y", "dtype": "float64"}),
    lambda s: s["shape_checks"].append({"op": "cast", "left": "A", "result": "A2", "dtype": "float64"}),
    lambda s: s["shape_checks"].append({"op": "cast", "left": "A", "result": "A64T", "dtype": "float64"}),
    lambda s: s["shape_checks"].append({"op": "cast", "left": "A", "result": "A64", "dtype": "float128"}),
])
def test_richer_shape_ops_reject_inconsistent_declarations(method, mutation):
    method = _extended(method)
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


def rational(left="(x+1/x)**2", right="x**2 + 2 + 1/x**2", variables=("x",)):
    return {"schema_version": 1, "definitions": {}, "obligations": [{
        "id": "rat", "required": True, "depends_on": [], "assumptions": [],
        "statement": {"kind": "rational_identity", "variables": list(variables), "left": left, "right": right}}]}


@pytest.mark.parametrize("left,right,variables", [
    ("(x+1/x)**2", "x**2 + 2 + 1/x**2", ("x",)),
    ("x/3 + x/6", "x/2", ("x",)), ("1/(x*y)", "1/x / y", ("x", "y")),
    ("x**-1", "1/x", ("x",)),
    ("(x-1)/(x**2-1)", "1/(x+1)", ("x",)),
])
def test_exact_rational_identities_are_machine_checked(left, right, variables):
    obligation = compile_theory(rational(left, right, variables))["obligations"][0]
    assert obligation["status"] == "machine_checked"
    assert obligation["checker"] == "rational_function_identity/v1"
    assert "denominators are nonzero" in obligation["evidence"]["domain"]


def test_false_rational_identity_has_counterexample_avoiding_denominator_zeros():
    obligation = compile_theory(rational("x/y", "y/x", ("x", "y")))["obligations"][0]
    assert obligation["status"] == "disproved"
    witness = obligation["evidence"]["counterexample"]
    assert witness["values"]["x"] != 0 and witness["values"]["y"] != 0
    assert witness["left"] != witness["right"]


def test_rational_grammar_and_prose_boundaries():
    with pytest.raises(WorkbenchError, match="zero polynomial"):
        compile_theory(rational("1/(x - x)", "0"))
    for expression in ("x.real", "0.5*x", "x**13", "[x for x in y]", "z+1"):
        with pytest.raises(WorkbenchError):
            compile_theory(rational(expression, "x"))
    with pytest.raises(WorkbenchError, match="assumptions"):
        assumed = rational("x", "x")
        assumed["obligations"][0]["assumptions"] = ["x is positive"]
        compile_theory(assumed)
    spec = rational("x", "x")
    spec["obligations"][0]["review"] = {"checker": "c", "verdict": "accepted", "evidence": "e"}
    with pytest.raises(WorkbenchError, match="prose proof overrides"):
        compile_theory(spec)


def symbolic(left="sin(x)**2 + cos(x)**2", right="1", variables=("x",)):
    return {"schema_version": 1, "definitions": {}, "obligations": [{
        "id": "sym", "required": True, "depends_on": [], "assumptions": [],
        "statement": {"kind": "symbolic_equality", "variables": list(variables), "left": left, "right": right}}]}


@pytest.mark.parametrize("left,right,variables", [
    ("sin(x)**2 + cos(x)**2", "1", ("x",)), ("(x+y)**2", "x**2 + 2*x*y + y**2", ("x", "y")),
    ("exp(x)**2", "exp(2*x)", ("x",)),
])
def test_symbolic_backend_proves_identities_machine_checked(left, right, variables):
    obligation = compile_theory(symbolic(left, right, variables))["obligations"][0]
    assert obligation["status"] == "machine_checked"
    assert obligation["checker"] == "sympy_symbolic_equality/v1"
    assert obligation["evidence"]["simplified_to_zero"] is True


def test_symbolic_nonzero_constant_difference_is_disproved():
    obligation = compile_theory(symbolic("sin(x)**2 + cos(x)**2", "2"))["obligations"][0]
    assert obligation["status"] == "disproved"
    assert obligation["evidence"]["difference_value"]


def test_symbolic_undecided_stays_unresolved():
    obligation = compile_theory(symbolic("log(x*y)", "log(x) + log(y)", ("x", "y")))["obligations"][0]
    assert obligation["status"] == "unresolved"
    assert obligation["evidence"]["decided"] is False


def test_symbolic_without_backend_stays_unresolved_but_validates_grammar(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "sympy", None)
    obligation = compile_theory(symbolic())["obligations"][0]
    assert obligation["status"] == "unresolved"
    assert obligation["evidence"]["backend"] == "unavailable"
    # Grammar is validated even when the backend is missing, so acceptance
    # cannot silently widen on machines without sympy.
    for expression in ("__import__('os').system('echo forbidden')", "x.real", "sin(x, y)", "x**13", "lambda x: x"):
        with pytest.raises(WorkbenchError):
            compile_theory(symbolic(expression, "x"))


def linear(*, domain="real", premises=None, conclusion=None, variables=("x", "y")):
    if premises is None:
        premises = [
            {"coefficients": {"x": 1, "y": 1}, "relation": ">=", "constant": 4},
            {"coefficients": {"x": 1}, "relation": ">=", "constant": 1},
        ]
    if conclusion is None:
        conclusion = {"coefficients": {"y": 1}, "relation": ">=", "constant": 0}
    return {"schema_version": 1, "definitions": {}, "obligations": [{
        "id": "linear", "required": True, "depends_on": [], "assumptions": [],
        "statement": {"kind": "linear_arithmetic", "domain": domain, "variables": list(variables),
                      "premises": premises, "conclusion": conclusion}}]}


def test_z3_linear_real_implication_is_machine_checked_with_consistent_premises():
    spec = linear(
        premises=[{"coefficients": {"x": 2, "y": "-1/2"}, "relation": "<=", "constant": 3},
                  {"coefficients": {"x": 1}, "relation": ">=", "constant": 2}],
        conclusion={"coefficients": {"y": 1}, "relation": ">=", "constant": 2})
    obligation = compile_theory(spec)["obligations"][0]
    assert obligation["status"] == "machine_checked"
    assert obligation["checker"] == "z3_linear_arithmetic_implication/v2"
    assert obligation["evidence"]["logic"] == "QF_LRA"
    assert obligation["evidence"]["premises_consistent"] is True
    assert obligation["evidence"]["negated_conclusion_result"] == "unsat"
    certificate = obligation["evidence"]["portable_certificate"]
    from researchclaw.pipeline.formal_proof import verify_farkas_certificate
    assert obligation["evidence"]["portable_certificate_verified"] is True
    assert verify_farkas_certificate(spec["obligations"][0]["statement"], certificate) is True


def test_farkas_certificate_proves_equality_and_rejects_tampering():
    statement = linear(
        variables=("x", "y"),
        premises=[{"coefficients": {"x": 1, "y": 1}, "relation": "==", "constant": 5},
                  {"coefficients": {"y": 1}, "relation": "==", "constant": 2}],
        conclusion={"coefficients": {"x": 1}, "relation": "==", "constant": 3})
    obligation = compile_theory(statement)["obligations"][0]
    certificate = obligation["evidence"]["portable_certificate"]
    assert [claim["target"] for claim in certificate["claims"]] == ["upper", "lower"]
    from researchclaw.pipeline.formal_proof import verify_farkas_certificate, FormalProofError
    original = statement["obligations"][0]["statement"]
    assert verify_farkas_certificate(original, certificate)
    for mutation in ("multiplier", "witness", "bound", "identity"):
        broken = copy.deepcopy(certificate)
        if mutation == "multiplier":
            broken["claims"][0]["multipliers"][0] = "-1"
        elif mutation == "witness":
            broken["premise_witness"]["x"] = "999"
        elif mutation == "bound":
            broken["claims"][0]["combined_constant"] = "999"
        else:
            broken["statement_hash"] = "0" * 64
        with pytest.raises(FormalProofError):
            verify_farkas_certificate(original, broken)


def test_frozen_farkas_certificate_recompiles_without_z3(monkeypatch):
    spec = linear(
        premises=[{"coefficients": {"x": 2, "y": "-1/2"}, "relation": "<=", "constant": 3},
                  {"coefficients": {"x": 1}, "relation": ">=", "constant": 2}],
        conclusion={"coefficients": {"y": 1}, "relation": ">=", "constant": 2})
    generated = compile_theory(spec)["obligations"][0]["evidence"]["portable_certificate"]
    portable = copy.deepcopy(spec)
    portable["obligations"][0]["statement"]["portable_certificate"] = generated
    import sys
    monkeypatch.setitem(sys.modules, "z3", None)
    obligation = compile_theory(portable)["obligations"][0]
    assert obligation["status"] == "machine_checked"
    assert obligation["checker"] == "exact_fraction_farkas/v1"
    assert obligation["evidence"]["backend"] == "portable_certificate"
    broken = copy.deepcopy(portable)
    broken["obligations"][0]["statement"]["portable_certificate"]["claims"][0]["multipliers"][0] = "-1"
    with pytest.raises(WorkbenchError):
        compile_theory(broken)


def test_portable_certificate_scope_does_not_overclaim_integer_or_strict_arithmetic():
    integer = compile_theory(linear(
        domain="integer", variables=("x",),
        premises=[{"coefficients": {"x": 1}, "relation": ">", "constant": 0}],
        conclusion={"coefficients": {"x": 1}, "relation": ">=", "constant": 1}))["obligations"][0]
    assert integer["status"] == "machine_checked" and integer["evidence"]["portable_certificate"] is None
    strict = compile_theory(linear(
        domain="real", variables=("x",),
        premises=[{"coefficients": {"x": 1}, "relation": ">", "constant": 1}],
        conclusion={"coefficients": {"x": 1}, "relation": ">", "constant": 0}))["obligations"][0]
    assert strict["status"] == "machine_checked" and strict["evidence"]["portable_certificate"] is None


def test_integer_implication_valid_over_reals_gets_stronger_portable_certificate(monkeypatch):
    spec = linear(
        domain="integer", variables=("x",),
        premises=[{"coefficients": {"x": 1}, "relation": ">=", "constant": 2}],
        conclusion={"coefficients": {"x": 1}, "relation": ">=", "constant": 1})
    generated = compile_theory(spec)["obligations"][0]["evidence"]["portable_certificate"]
    assert generated["schema_version"] == 2
    assert generated["kind"] == "farkas_linear_implication_over_reals"
    assert generated["proof_domain"] == "real_superset_of_integer_domain"
    from researchclaw.pipeline.formal_proof import verify_farkas_certificate
    assert verify_farkas_certificate(spec["obligations"][0]["statement"], generated)
    portable = copy.deepcopy(spec)
    portable["obligations"][0]["statement"]["portable_certificate"] = generated
    import sys
    monkeypatch.setitem(sys.modules, "z3", None)
    obligation = compile_theory(portable)["obligations"][0]
    assert obligation["status"] == "machine_checked"
    assert obligation["checker"] == "exact_fraction_farkas/v2"
    assert obligation["evidence"]["logic"] == "QF_LIA"


def test_bounded_integer_enumeration_certificate_covers_strict_and_not_equal(monkeypatch):
    spec = linear(
        domain="integer", variables=("x",),
        premises=[{"coefficients": {"x": 1}, "relation": ">", "constant": 0},
                  {"coefficients": {"x": 1}, "relation": "<=", "constant": 3},
                  {"coefficients": {"x": 1}, "relation": "!=", "constant": 2}],
        conclusion={"coefficients": {"x": 1}, "relation": ">=", "constant": 1})
    obligation = compile_theory(spec)["obligations"][0]
    certificate = obligation["evidence"]["portable_certificate"]
    assert certificate["kind"] == "exhaustive_bounded_integer_implication"
    assert certificate["bounds"] == {"x": {"lower": 1, "upper": 3}}
    assert certificate["checked_assignments"] == 3 and certificate["satisfying_assignments"] == 2
    from researchclaw.pipeline.formal_proof import verify_exhaustive_integer_certificate, FormalProofError
    statement = spec["obligations"][0]["statement"]
    assert verify_exhaustive_integer_certificate(statement, certificate)

    portable = copy.deepcopy(spec)
    portable["obligations"][0]["statement"]["portable_certificate"] = certificate
    import sys
    monkeypatch.setitem(sys.modules, "z3", None)
    replay = compile_theory(portable)["obligations"][0]
    assert replay["checker"] == "exact_bounded_integer_enumeration/v1"
    assert replay["evidence"]["certificate_proof_domain"] == "bounded_integer"
    for field in ("statement_hash", "bounds", "checked_assignments", "satisfying_assignments", "premise_witness"):
        broken = copy.deepcopy(certificate)
        if field == "statement_hash":
            broken[field] = "0" * 64
        elif field == "bounds":
            broken[field]["x"]["upper"] = 4
        elif field == "premise_witness":
            broken[field]["x"] = 2
        else:
            broken[field] += 1
        with pytest.raises(FormalProofError):
            verify_exhaustive_integer_certificate(statement, broken)


def test_z3_linear_integer_implication_uses_integrality():
    # Over the integers x > 0 entails x >= 1; the same statement is false over reals.
    premise = [{"coefficients": {"x": 1}, "relation": ">", "constant": 0}]
    conclusion = {"coefficients": {"x": 1}, "relation": ">=", "constant": 1}
    integer = compile_theory(linear(domain="integer", variables=("x",), premises=premise,
                                    conclusion=conclusion))["obligations"][0]
    real = compile_theory(linear(domain="real", variables=("x",), premises=premise,
                                 conclusion=conclusion))["obligations"][0]
    assert integer["status"] == "machine_checked" and integer["evidence"]["logic"] == "QF_LIA"
    assert real["status"] == "disproved" and real["evidence"]["counterexample_exactly_validated"] is True


@pytest.mark.parametrize("relation,constant", [("<=", 1), ("==", 1), ("!=", 0), ("<", 2)])
def test_z3_disproved_implications_return_exact_validated_counterexamples(relation, constant):
    premise = [{"coefficients": {"x": 1}, "relation": ">=", "constant": 0}]
    conclusion = {"coefficients": {"x": 1}, "relation": relation, "constant": constant}
    obligation = compile_theory(linear(variables=("x",), premises=premise,
                                       conclusion=conclusion))["obligations"][0]
    assert obligation["status"] == "disproved"
    assert obligation["evidence"]["counterexample_exactly_validated"] is True
    assert set(obligation["evidence"]["counterexample"]) == {"x"}


def test_inconsistent_linear_premises_are_not_accepted_vacuously():
    premises = [{"coefficients": {"x": 1}, "relation": ">", "constant": 0},
                {"coefficients": {"x": 1}, "relation": "<", "constant": 0}]
    obligation = compile_theory(linear(variables=("x",), premises=premises,
                                       conclusion={"coefficients": {"x": 1}, "relation": "==", "constant": 999}))["obligations"][0]
    assert obligation["status"] == "unresolved"
    assert obligation["evidence"]["premises_consistent"] is False
    assert obligation["evidence"]["vacuous_implication_rejected"] is True


def test_inconsistent_closed_real_premises_get_a_portable_inconsistency_certificate(monkeypatch):
    spec = linear(
        variables=("x",),
        premises=[{"coefficients": {"x": 1}, "relation": ">=", "constant": 1},
                  {"coefficients": {"x": 1}, "relation": "<=", "constant": 0}],
        conclusion={"coefficients": {"x": 1}, "relation": ">=", "constant": 0})
    obligation = compile_theory(spec)["obligations"][0]
    assert obligation["status"] == "unresolved"
    assert obligation["evidence"]["vacuous_implication_rejected"] is True
    certificate = obligation["evidence"]["portable_certificate"]
    assert certificate["kind"] == "farkas_linear_inconsistency"
    assert obligation["evidence"]["portable_certificate_verified"] is True
    from researchclaw.pipeline.formal_proof import verify_inconsistency_certificate
    statement = spec["obligations"][0]["statement"]
    assert verify_inconsistency_certificate(statement, certificate)
    portable = copy.deepcopy(spec)
    portable["obligations"][0]["statement"]["portable_certificate"] = certificate
    import sys
    monkeypatch.setitem(sys.modules, "z3", None)
    recompiled = compile_theory(portable)["obligations"][0]
    assert recompiled["status"] == "unresolved"
    assert recompiled["checker"] == "exact_fraction_inconsistency/v1"
    assert recompiled["evidence"]["backend"] == "portable_certificate"
    assert recompiled["evidence"]["vacuous_implication_rejected"] is True
    broken = copy.deepcopy(portable)
    broken["obligations"][0]["statement"]["portable_certificate"]["multipliers"][0] = "-1"
    with pytest.raises(WorkbenchError):
        compile_theory(broken)


def test_integer_inconsistency_over_reals_and_real_consistent_parity_stays_unresolved(monkeypatch):
    spec = linear(
        domain="integer", variables=("x",),
        premises=[{"coefficients": {"x": 1}, "relation": ">=", "constant": 1},
                  {"coefficients": {"x": 1}, "relation": "<=", "constant": 0}],
        conclusion={"coefficients": {"x": 1}, "relation": ">=", "constant": 0})
    obligation = compile_theory(spec)["obligations"][0]
    certificate = obligation["evidence"]["portable_certificate"]
    assert certificate["kind"] == "farkas_linear_inconsistency_over_reals"
    assert certificate["proof_domain"] == "real_superset_of_integer_domain"
    from researchclaw.pipeline.formal_proof import verify_inconsistency_certificate
    assert verify_inconsistency_certificate(spec["obligations"][0]["statement"], certificate)
    portable = copy.deepcopy(spec)
    portable["obligations"][0]["statement"]["portable_certificate"] = certificate
    import sys
    monkeypatch.setitem(sys.modules, "z3", None)
    assert compile_theory(portable)["obligations"][0]["status"] == "unresolved"
    # 2x == 1 is unsatisfiable over the integers but consistent over the
    # reals, so no real-relaxation certificate exists and the verdict
    # stays honestly unresolved rather than faking one.
    parity = compile_theory(linear(
        domain="integer", variables=("x",),
        premises=[{"coefficients": {"x": 2}, "relation": "==", "constant": 1}],
        conclusion={"coefficients": {"x": 1}, "relation": ">=", "constant": 0}))["obligations"][0]
    assert parity["status"] == "unresolved"
    assert parity["evidence"].get("portable_certificate") is None


def test_strict_premise_inconsistency_gets_a_portable_motzkin_certificate(monkeypatch):
    spec = linear(
        variables=("x",),
        premises=[{"coefficients": {"x": 1}, "relation": "<", "constant": 0},
                  {"coefficients": {"x": 1}, "relation": ">", "constant": 0}],
        conclusion={"coefficients": {"x": 1}, "relation": ">=", "constant": 0})
    obligation = compile_theory(spec)["obligations"][0]
    assert obligation["status"] == "unresolved"
    assert obligation["evidence"]["vacuous_implication_rejected"] is True
    certificate = obligation["evidence"]["portable_certificate"]
    assert certificate["kind"] == "motzkin_linear_inconsistency"
    assert obligation["evidence"]["portable_certificate_checker"] == "exact_fraction_motzkin/v1"
    assert obligation["evidence"]["portable_certificate_verified"] is True
    from researchclaw.pipeline.formal_proof import verify_motzkin_inconsistency_certificate
    statement = spec["obligations"][0]["statement"]
    assert verify_motzkin_inconsistency_certificate(statement, certificate)
    portable = copy.deepcopy(spec)
    portable["obligations"][0]["statement"]["portable_certificate"] = certificate
    import sys
    monkeypatch.setitem(sys.modules, "z3", None)
    recompiled = compile_theory(portable)["obligations"][0]
    assert recompiled["status"] == "unresolved"
    assert recompiled["checker"] == "exact_fraction_motzkin/v1"
    assert recompiled["evidence"]["backend"] == "portable_certificate"
    assert recompiled["evidence"]["vacuous_implication_rejected"] is True
    broken = copy.deepcopy(portable)
    broken["obligations"][0]["statement"]["portable_certificate"]["strict_multipliers"][0] = "-1"
    with pytest.raises(WorkbenchError):
        compile_theory(broken)


def test_mixed_closed_and_strict_inconsistency_gets_a_motzkin_certificate():
    spec = linear(
        variables=("x",),
        premises=[{"coefficients": {"x": 1}, "relation": ">=", "constant": 1},
                  {"coefficients": {"x": 1}, "relation": "<", "constant": 1}],
        conclusion={"coefficients": {"x": 1}, "relation": ">=", "constant": 0})
    obligation = compile_theory(spec)["obligations"][0]
    assert obligation["status"] == "unresolved"
    certificate = obligation["evidence"]["portable_certificate"]
    assert certificate["kind"] == "motzkin_linear_inconsistency"
    assert certificate["premise_rows"] == ["premise_1:lower"]
    assert certificate["strict_premise_rows"] == ["premise_2:strict"]
    from researchclaw.pipeline.formal_proof import verify_motzkin_inconsistency_certificate
    assert verify_motzkin_inconsistency_certificate(spec["obligations"][0]["statement"], certificate)


def test_equality_premises_keep_farkas_certificates_and_split_both_rows():
    # Regression for the round-55 verification: an elif in the strict/closed
    # split dropped the :lower row of == premises, silently removing Farkas
    # certificates from closed equality systems and crashing
    # check_linear_arithmetic on x==0 & x>=1 (the generated certificate's
    # premise_rows no longer matched the verifier's re-derived rows).
    crashed = compile_theory(linear(
        variables=("x",),
        premises=[{"coefficients": {"x": 1}, "relation": "==", "constant": 0},
                  {"coefficients": {"x": 1}, "relation": ">=", "constant": 1}],
        conclusion={"coefficients": {"x": 1}, "relation": ">=", "constant": 0}))["obligations"][0]
    assert crashed["status"] == "unresolved"
    assert crashed["evidence"]["portable_certificate"]["kind"] == "farkas_linear_inconsistency"
    mixed_spec = linear(
        variables=("x",),
        premises=[{"coefficients": {"x": 1}, "relation": "==", "constant": 0},
                  {"coefficients": {"x": 1}, "relation": "<", "constant": 0}],
        conclusion={"coefficients": {"x": 1}, "relation": ">=", "constant": 0})
    mixed = compile_theory(mixed_spec)["obligations"][0]
    certificate = mixed["evidence"]["portable_certificate"]
    assert certificate["kind"] == "motzkin_linear_inconsistency"
    assert certificate["premise_rows"] == ["premise_1:upper", "premise_1:lower"]
    assert certificate["strict_premise_rows"] == ["premise_2:strict"]
    from researchclaw.pipeline.formal_proof import verify_motzkin_inconsistency_certificate
    assert verify_motzkin_inconsistency_certificate(mixed_spec["obligations"][0]["statement"], certificate)


def test_integer_motzkin_over_reals_and_real_feasible_parity_stays_certless(monkeypatch):
    spec = linear(
        domain="integer", variables=("x",),
        premises=[{"coefficients": {"x": 1}, "relation": "<", "constant": 0},
                  {"coefficients": {"x": 1}, "relation": ">", "constant": 0}],
        conclusion={"coefficients": {"x": 1}, "relation": ">=", "constant": 0})
    obligation = compile_theory(spec)["obligations"][0]
    certificate = obligation["evidence"]["portable_certificate"]
    assert certificate["kind"] == "motzkin_linear_inconsistency_over_reals"
    assert certificate["proof_domain"] == "real_superset_of_integer_domain"
    from researchclaw.pipeline.formal_proof import verify_motzkin_inconsistency_certificate
    assert verify_motzkin_inconsistency_certificate(spec["obligations"][0]["statement"], certificate)
    portable = copy.deepcopy(spec)
    portable["obligations"][0]["statement"]["portable_certificate"] = certificate
    import sys
    monkeypatch.setitem(sys.modules, "z3", None)
    assert compile_theory(portable)["obligations"][0]["status"] == "unresolved"
    # x < 1 with 2x > 1 is unsatisfiable over the integers but feasible
    # over the reals (x = 3/4), so no real-relaxation certificate exists
    # and the verdict stays honestly unresolved rather than faking one.
    parity = compile_theory(linear(
        domain="integer", variables=("x",),
        premises=[{"coefficients": {"x": 1}, "relation": "<", "constant": 1},
                  {"coefficients": {"x": 2}, "relation": ">", "constant": 1}],
        conclusion={"coefficients": {"x": 1}, "relation": ">=", "constant": 0}))["obligations"][0]
    assert parity["status"] == "unresolved"
    assert parity["evidence"].get("portable_certificate") is None


def test_feasible_strict_system_rejects_all_zero_strict_motzkin_forgery():
    # x <= 5 with x >= 5 with x < 100 is feasible. With all-zero strict
    # multipliers the Motzkin combination would "refute" consistent
    # premises, which is exactly what the mu != 0 requirement forbids.
    spec = linear(
        variables=("x",),
        premises=[{"coefficients": {"x": 1}, "relation": "<=", "constant": 5},
                  {"coefficients": {"x": 1}, "relation": ">=", "constant": 5},
                  {"coefficients": {"x": 1}, "relation": "<", "constant": 100}],
        conclusion={"coefficients": {"x": 1}, "relation": ">=", "constant": 0})
    from researchclaw.pipeline.formal_proof import (verify_motzkin_inconsistency_certificate,
                                                    FormalProofError, _statement_hash)
    statement = spec["obligations"][0]["statement"]
    base = {"schema_version": 1, "kind": "motzkin_linear_inconsistency",
            "statement_hash": _statement_hash(statement),
            "premise_rows": ["premise_1:upper", "premise_2:lower"],
            "strict_premise_rows": ["premise_3:strict"],
            "closed_multipliers": ["1", "1"], "strict_multipliers": ["0"],
            "combined_constant": "0"}
    with pytest.raises(FormalProofError, match="positive strict multiplier"):
        verify_motzkin_inconsistency_certificate(statement, base)
    # Even with a positive strict multiplier, a feasible system cannot
    # reach a nonpositive combined constant: A^T*lambda + C^T*mu = 0 and
    # the feasible point force b^T*lambda + d^T*mu > 0.
    stretched = dict(base, closed_multipliers=["0", "1"], strict_multipliers=["1"],
                     combined_constant="95")
    with pytest.raises(FormalProofError, match="do not derive a contradiction"):
        verify_motzkin_inconsistency_certificate(statement, stretched)


def test_nequality_premise_inconsistency_stays_without_certificate():
    # x == 0 and x != 0 contradict, but the closed rows alone (x <= 0 and
    # -x <= 0) combine only to 0 <= 0: no Farkas refutation exists over the
    # reals, and the bounds replay needs the integer domain, so the
    # contradiction stays Z3-only and cert-less here.
    obligation = compile_theory(linear(
        variables=("x",),
        premises=[{"coefficients": {"x": 1}, "relation": "==", "constant": 0},
                  {"coefficients": {"x": 1}, "relation": "!=", "constant": 0}],
        conclusion={"coefficients": {"x": 1}, "relation": ">=", "constant": 0}))["obligations"][0]
    assert obligation["status"] == "unresolved"
    assert obligation["evidence"]["vacuous_implication_rejected"] is True
    assert obligation["evidence"].get("portable_certificate") is None


def test_motzkin_branch_falls_back_to_closed_farkas_when_strict_rows_cannot_participate(monkeypatch):
    # The strict row is the only one touching v0/v2, so any Motzkin
    # refutation is forced to mu = 0; the contradiction lives in the closed
    # rows alone (3*v1 <= -2 with -3*v1 <= 1 gives 0 <= -1). Fuzz trial 162.
    spec = linear(
        domain="integer", variables=("v0", "v1", "v2"),
        premises=[{"coefficients": {"v0": 1, "v1": 2, "v2": 1}, "relation": "<", "constant": 2},
                  {"coefficients": {"v1": 3}, "relation": ">=", "constant": -1},
                  {"coefficients": {"v1": 3}, "relation": "<=", "constant": -2}],
        conclusion={"coefficients": {"v0": 1}, "relation": ">=", "constant": 0})
    obligation = compile_theory(spec)["obligations"][0]
    assert obligation["status"] == "unresolved"
    assert obligation["evidence"]["premises_consistent"] is False
    assert obligation["evidence"]["vacuous_implication_rejected"] is True
    certificate = obligation["evidence"]["portable_certificate"]
    assert certificate["kind"] == "farkas_linear_inconsistency_over_reals"
    assert certificate["premise_rows"] == ["premise_2:lower", "premise_3:upper"]
    assert obligation["evidence"]["portable_certificate_verified"] is True
    from researchclaw.pipeline.formal_proof import (verify_inconsistency_certificate,
                                                    FormalProofError)
    statement = spec["obligations"][0]["statement"]
    assert verify_inconsistency_certificate(statement, certificate)
    strict_contaminated = copy.deepcopy(certificate)
    strict_contaminated["premise_rows"] = ["premise_1:strict", "premise_2:lower", "premise_3:upper"]
    with pytest.raises(FormalProofError):
        verify_inconsistency_certificate(statement, strict_contaminated)
    portable = copy.deepcopy(spec)
    portable["obligations"][0]["statement"]["portable_certificate"] = certificate
    import sys
    monkeypatch.setitem(sys.modules, "z3", None)
    replay = compile_theory(portable)["obligations"][0]
    assert replay["status"] == "unresolved"
    assert replay["checker"] == "exact_fraction_inconsistency/v1"
    assert replay["evidence"]["vacuous_implication_rejected"] is True


def test_nequality_premises_no_longer_blind_the_inconsistency_search():
    # A != premise is itself no obstacle: a contradiction carried by the
    # remaining closed rows still refutes the conjunction, and over the
    # integers the != premise participates in the bounds replay enumeration.
    # Fuzz trial 412 plus the integer-domain x == 0 /\ x != 0 shape.
    farkas_spec = linear(
        variables=("x",),
        premises=[{"coefficients": {"x": -2}, "relation": "!=", "constant": 0},
                  {"coefficients": {"x": 4}, "relation": "<=", "constant": -2},
                  {"coefficients": {"x": 1}, "relation": "==", "constant": 2}],
        conclusion={"coefficients": {"x": 1}, "relation": ">=", "constant": 0})
    obligation = compile_theory(farkas_spec)["obligations"][0]
    assert obligation["status"] == "unresolved"
    certificate = obligation["evidence"]["portable_certificate"]
    assert certificate["kind"] == "farkas_linear_inconsistency"
    assert certificate["premise_rows"] == ["premise_2:upper", "premise_3:upper", "premise_3:lower"]
    from researchclaw.pipeline.formal_proof import (verify_inconsistency_certificate,
                                                    verify_bounded_integer_infeasibility)
    assert verify_inconsistency_certificate(
        farkas_spec["obligations"][0]["statement"], certificate)

    bounds_spec = linear(
        domain="integer", variables=("x",),
        premises=[{"coefficients": {"x": 1}, "relation": "==", "constant": 0},
                  {"coefficients": {"x": 1}, "relation": "!=", "constant": 0}],
        conclusion={"coefficients": {"x": 1}, "relation": ">=", "constant": 0})
    bounds = compile_theory(bounds_spec)["obligations"][0]
    assert bounds["status"] == "unresolved"
    bounds_certificate = bounds["evidence"]["portable_certificate"]
    assert bounds_certificate["kind"] == "bounded_integer_infeasibility"
    assert bounds_certificate["checked_assignments"] == 1
    assert bounds["evidence"]["portable_certificate_verified"] is True
    assert verify_bounded_integer_infeasibility(
        bounds_spec["obligations"][0]["statement"], bounds_certificate)


def test_parity_equality_gets_a_bounded_integer_infeasibility_certificate(monkeypatch):
    spec = linear(
        domain="integer", variables=("x",),
        premises=[{"coefficients": {"x": 2}, "relation": "==", "constant": 1}],
        conclusion={"coefficients": {"x": 1}, "relation": ">=", "constant": 0})
    obligation = compile_theory(spec)["obligations"][0]
    assert obligation["status"] == "unresolved"
    certificate = obligation["evidence"]["portable_certificate"]
    assert certificate["kind"] == "bounded_integer_infeasibility"
    assert certificate["bounds"] == {"x": {"lower": 1, "upper": 0}}
    assert certificate["checked_assignments"] == 0
    assert obligation["evidence"]["portable_certificate_checker"] == "exact_bounded_integer_infeasibility/v1"
    assert obligation["evidence"]["portable_certificate_verified"] is True
    from researchclaw.pipeline.formal_proof import verify_bounded_integer_infeasibility
    statement = spec["obligations"][0]["statement"]
    assert verify_bounded_integer_infeasibility(statement, certificate)
    portable = copy.deepcopy(spec)
    portable["obligations"][0]["statement"]["portable_certificate"] = certificate
    import sys
    monkeypatch.setitem(sys.modules, "z3", None)
    recompiled = compile_theory(portable)["obligations"][0]
    assert recompiled["status"] == "unresolved"
    assert recompiled["checker"] == "exact_bounded_integer_infeasibility/v1"
    assert recompiled["evidence"]["vacuous_implication_rejected"] is True
    broken = copy.deepcopy(portable)
    broken["obligations"][0]["statement"]["portable_certificate"]["bounds"]["x"]["lower"] = 0
    with pytest.raises(WorkbenchError):
        compile_theory(broken)


def test_strict_parity_bounded_infeasibility_certificate():
    strict_spec = linear(
        domain="integer", variables=("x",),
        premises=[{"coefficients": {"x": 1}, "relation": "<", "constant": 1},
                  {"coefficients": {"x": 2}, "relation": ">", "constant": 1}],
        conclusion={"coefficients": {"x": 1}, "relation": ">=", "constant": 0})
    strict = compile_theory(strict_spec)["obligations"][0]
    certificate = strict["evidence"]["portable_certificate"]
    assert certificate["kind"] == "bounded_integer_infeasibility"
    assert certificate["checked_assignments"] == 0
    assert certificate["strict_premise_rows"] == ["premise_1:strict", "premise_2:strict"]
    from researchclaw.pipeline.formal_proof import verify_bounded_integer_infeasibility
    assert verify_bounded_integer_infeasibility(strict_spec["obligations"][0]["statement"], certificate)


def test_coupling_equality_gets_an_integer_congruence_certificate(monkeypatch):
    # The row space of 2x - 2y == 1 cannot pin either variable alone, so the
    # bounded-box certificate is out of reach; the congruence argument still
    # proves the contradiction: any solution satisfies 2x - 2y == 1, whose
    # left side is even at every integer point while the right side is odd.
    spec = linear(
        domain="integer", variables=("x", "y"),
        premises=[{"coefficients": {"x": 2, "y": -2}, "relation": "==", "constant": 1}],
        conclusion={"coefficients": {"x": 1}, "relation": ">=", "constant": 0})
    obligation = compile_theory(spec)["obligations"][0]
    assert obligation["status"] == "unresolved"
    certificate = obligation["evidence"]["portable_certificate"]
    assert certificate["kind"] == "integer_congruence_infeasibility"
    assert certificate["equality_premise_rows"] == [["premise_1:upper", "premise_1:lower"]]
    assert certificate["row_multipliers"] == ["1"]
    assert certificate["derived_coefficients"] == {"x": "2", "y": "-2"}
    assert certificate["derived_constant"] == "1"
    assert certificate["modulus"] == 2
    assert obligation["evidence"]["portable_certificate_checker"] == "exact_integer_congruence_infeasibility/v1"
    assert obligation["evidence"]["portable_certificate_verified"] is True
    from researchclaw.pipeline.formal_proof import verify_integer_congruence_infeasibility
    statement = spec["obligations"][0]["statement"]
    assert verify_integer_congruence_infeasibility(statement, certificate)
    portable = copy.deepcopy(spec)
    portable["obligations"][0]["statement"]["portable_certificate"] = certificate
    import sys
    monkeypatch.setitem(sys.modules, "z3", None)
    recompiled = compile_theory(portable)["obligations"][0]
    assert recompiled["status"] == "unresolved"
    assert recompiled["checker"] == "exact_integer_congruence_infeasibility/v1"
    assert recompiled["evidence"]["vacuous_implication_rejected"] is True
    broken = copy.deepcopy(portable)
    broken["obligations"][0]["statement"]["portable_certificate"]["modulus"] = 1
    with pytest.raises(WorkbenchError):
        compile_theory(broken)


def test_congruence_elimination_and_pair_form_certificates():
    from researchclaw.pipeline.formal_proof import (_integer_congruence_infeasibility_certificate,
                                                    validate_linear_statement)
    # The single-row shortcuts cannot see this contradiction; echelon
    # elimination derives (x+y) - (x-y) == 1, i.e. 2y == 1, whose primitive
    # row has a fractional constant.
    spec = linear(
        domain="integer", variables=("x", "y", "z"),
        premises=[{"coefficients": {"x": 1, "y": 1}, "relation": "==", "constant": 1},
                  {"coefficients": {"x": 1, "y": -1}, "relation": "==", "constant": 0},
                  {"coefficients": {"x": 2, "z": 2}, "relation": "==", "constant": 0}],
        conclusion={"coefficients": {"x": 1}, "relation": ">=", "constant": 0})
    statement = spec["obligations"][0]["statement"]
    certificate = _integer_congruence_infeasibility_certificate(
        statement, validate_linear_statement(statement))
    assert certificate["kind"] == "integer_congruence_infeasibility"
    assert certificate["equality_premise_rows"] == [["premise_1:upper", "premise_1:lower"],
                                                    ["premise_2:upper", "premise_2:lower"],
                                                    ["premise_3:upper", "premise_3:lower"]]
    assert certificate["row_multipliers"] == ["1", "-1", "0"]
    assert certificate["derived_coefficients"] == {"x": "0", "y": "2", "z": "0"}
    assert certificate["derived_constant"] == "1"
    assert certificate["modulus"] == 2
    # Opposite closed premises force an equality too: 2x - 2y <= 1 with
    # 2x - 2y >= 1 pins the coupled form from both sides, the bounded-box
    # search still fails, and the congruence certificate closes the system.
    pair = compile_theory(linear(
        domain="integer", variables=("x", "y"),
        premises=[{"coefficients": {"x": 2, "y": -2}, "relation": "<=", "constant": 1},
                  {"coefficients": {"x": 2, "y": -2}, "relation": ">=", "constant": 1}],
        conclusion={"coefficients": {"x": 1}, "relation": ">=", "constant": 0}))["obligations"][0]
    pair_certificate = pair["evidence"]["portable_certificate"]
    assert pair_certificate["kind"] == "integer_congruence_infeasibility"
    assert pair_certificate["equality_premise_rows"] == [["premise_1:upper", "premise_2:lower"]]
    assert pair["evidence"]["portable_certificate_checker"] == "exact_integer_congruence_infeasibility/v1"
    assert pair["status"] == "unresolved"


def test_congruence_survives_content_divided_rows_in_elimination():
    from researchclaw.pipeline.formal_proof import (_integer_congruence_infeasibility_certificate,
                                                    validate_linear_statement,
                                                    verify_integer_congruence_infeasibility)
    # 2x + 4y == -2 is stored content-divided (x + 2y == -1), so the firing
    # elimination mixes rows of different scales: 2*(4x - y) - 4*(2x + 4y)
    # == 10 gives -18y == 10, i.e. y == -5/9. The recorded multipliers must
    # reproduce that derived row exactly or the frozen certificate would not
    # re-verify.
    spec = linear(
        domain="integer", variables=("x", "y"),
        premises=[{"coefficients": {"x": 4, "y": -1}, "relation": "==", "constant": 1},
                  {"coefficients": {"x": 2, "y": 4}, "relation": "==", "constant": -2},
                  {"coefficients": {"x": 2, "y": 2}, "relation": "<=", "constant": -1}],
        conclusion={"coefficients": {"x": 1}, "relation": ">=", "constant": 0})
    statement = spec["obligations"][0]["statement"]
    certificate = _integer_congruence_infeasibility_certificate(
        statement, validate_linear_statement(statement))
    assert certificate is not None
    assert certificate["kind"] == "integer_congruence_infeasibility"
    assert certificate["equality_premise_rows"] == [["premise_1:upper", "premise_1:lower"],
                                                    ["premise_2:upper", "premise_2:lower"]]
    assert certificate["row_multipliers"] == ["2", "-4"]
    assert certificate["derived_coefficients"] == {"x": "0", "y": "-18"}
    assert certificate["derived_constant"] == "10"
    assert certificate["modulus"] == 18
    assert verify_integer_congruence_infeasibility(statement, certificate) is True


def test_congruence_forgeries_rejected():
    from researchclaw.pipeline.formal_proof import (FormalProofError,
                                                    _integer_congruence_infeasibility_certificate,
                                                    _statement_hash, validate_linear_statement,
                                                    verify_integer_congruence_infeasibility)
    infeasible = linear(
        domain="integer", variables=("x", "y"),
        premises=[{"coefficients": {"x": 2, "y": -2}, "relation": "==", "constant": 1}],
        conclusion={"coefficients": {"x": 1}, "relation": ">=", "constant": 0})
    statement = infeasible["obligations"][0]["statement"]
    good = _integer_congruence_infeasibility_certificate(statement, validate_linear_statement(statement))
    assert verify_integer_congruence_infeasibility(statement, good)
    # The derivation is genuine for the feasible system 2x - 2y == 0 as well,
    # but its scaled constant 0 is divisible by the modulus 2, so no
    # contradiction is derivable and the verifier refuses the certificate.
    feasible = linear(
        domain="integer", variables=("x", "y"),
        premises=[{"coefficients": {"x": 2, "y": -2}, "relation": "==", "constant": 0}],
        conclusion={"coefficients": {"x": 1}, "relation": ">=", "constant": 0})
    feasible_statement = feasible["obligations"][0]["statement"]
    # The declared row is the genuine derivation (2x - 2y == 0), so the
    # verifier reaches the divisibility test and finds nothing to reject.
    with pytest.raises(FormalProofError, match="no contradiction"):
        verify_integer_congruence_infeasibility(feasible_statement, {
            "schema_version": 1, "kind": "integer_congruence_infeasibility",
            "statement_hash": _statement_hash(feasible_statement),
            "equality_premise_rows": [["premise_1:upper", "premise_1:lower"]],
            "row_multipliers": ["1"],
            "derived_coefficients": {"x": "2", "y": "-2"}, "derived_constant": "0",
            "modulus": 2})
    # Over the reals the congruence argument proves nothing: 2x - 2y == 1 has
    # the real solution x = y + 1/2, so the certificate kind is out of place.
    real = linear(
        variables=("x", "y"),
        premises=[{"coefficients": {"x": 2, "y": -2}, "relation": "==", "constant": 1}],
        conclusion={"coefficients": {"x": 1}, "relation": ">=", "constant": 0})
    real_statement = real["obligations"][0]["statement"]
    with pytest.raises(FormalProofError, match="requires the integer domain"):
        verify_integer_congruence_infeasibility(
            real_statement, dict(good, statement_hash=_statement_hash(real_statement)))
    # A stale declared row: multiplier 2 recomputes to 4x - 4y == 2, which
    # differs from the declared 2x - 2y == 1 combination.
    with pytest.raises(FormalProofError, match="differs from the multiplier combination"):
        verify_integer_congruence_infeasibility(statement, dict(good, row_multipliers=["2"]))
    # Zero multipliers with a matching zero row derive no congruence at all.
    vacuous = {"schema_version": 1, "kind": "integer_congruence_infeasibility",
               "statement_hash": _statement_hash(statement),
               "equality_premise_rows": [["premise_1:upper", "premise_1:lower"]],
               "row_multipliers": ["0"],
               "derived_coefficients": {"x": "0", "y": "0"}, "derived_constant": "0",
               "modulus": 0}
    with pytest.raises(FormalProofError, match="derive no congruence"):
        verify_integer_congruence_infeasibility(statement, vacuous)
    # Statements without effective equalities (strict parity) admit no
    # congruence combination: the empty identity must be refused outright.
    strict = linear(
        domain="integer", variables=("x",),
        premises=[{"coefficients": {"x": 1}, "relation": "<", "constant": 1},
                  {"coefficients": {"x": 2}, "relation": ">", "constant": 1}],
        conclusion={"coefficients": {"x": 1}, "relation": ">=", "constant": 0})
    strict_statement = strict["obligations"][0]["statement"]
    with pytest.raises(FormalProofError, match="derive no congruence"):
        verify_integer_congruence_infeasibility(strict_statement, {
            "schema_version": 1, "kind": "integer_congruence_infeasibility",
            "statement_hash": _statement_hash(strict_statement),
            "equality_premise_rows": [], "row_multipliers": [],
            "derived_coefficients": {"x": "0"}, "derived_constant": "0", "modulus": 0})


def test_snf_search_closes_obstructions_the_echelon_misses():
    from researchclaw.pipeline.formal_proof import (_integer_congruence_infeasibility_certificate,
                                                    validate_linear_statement,
                                                    verify_integer_congruence_infeasibility)
    # Four equalities whose obstruction needs a combination outside the
    # elimination basis: the echelon derives no primitive row with a
    # fractional constant, but the Smith row transform combines the rows to
    # 257*v1 == 197, and 257 does not divide 197.
    spec = linear(
        domain="integer", variables=("v0", "v1", "v2", "v3"),
        premises=[{"coefficients": {"v0": -1, "v1": 3, "v2": 1, "v3": 1},
                   "relation": "==", "constant": 2},
                  {"coefficients": {"v0": 4, "v1": -1, "v2": 3, "v3": -2},
                   "relation": "==", "constant": 3},
                  {"coefficients": {"v0": -2, "v1": -2, "v2": 2, "v3": 1},
                   "relation": "==", "constant": -2},
                  {"coefficients": {"v0": -3, "v1": -3, "v2": 4, "v3": -3},
                   "relation": "==", "constant": -2}],
        conclusion={"coefficients": {"v0": 1}, "relation": ">=", "constant": 0})
    statement = spec["obligations"][0]["statement"]
    certificate = _integer_congruence_infeasibility_certificate(
        statement, validate_linear_statement(statement))
    assert certificate["kind"] == "integer_congruence_infeasibility"
    assert certificate["row_multipliers"] == ["63", "-1", "-44", "7"]
    assert certificate["derived_coefficients"] == {"v0": "0", "v1": "257", "v2": "0", "v3": "0"}
    assert certificate["derived_constant"] == "197"
    assert certificate["modulus"] == 257
    assert verify_integer_congruence_infeasibility(statement, certificate) is True
    # A dispatch-gap shape recorded by the round-60 fuzz (20000 random
    # systems): every layer missed it, SNF combines the rows to
    # 15*v0 - 15*v1 + 15*v2 == -5 with modulus 15.
    gap = linear(
        domain="integer", variables=("v0", "v1", "v2", "v3"),
        premises=[{"coefficients": {"v0": -3, "v1": 3, "v2": 2, "v3": 2},
                   "relation": "==", "constant": -3},
                  {"coefficients": {"v0": -2, "v1": -3, "v2": 3, "v3": 3},
                   "relation": "==", "constant": 3},
                  {"coefficients": {"v0": 2, "v1": -3, "v2": 4, "v3": 1},
                   "relation": "==", "constant": -1}],
        conclusion={"coefficients": {"v0": 1}, "relation": ">=", "constant": 0})
    gap_statement = gap["obligations"][0]["statement"]
    gap_certificate = _integer_congruence_infeasibility_certificate(
        gap_statement, validate_linear_statement(gap_statement))
    assert gap_certificate["row_multipliers"] == ["-1", "-1", "5"]
    assert gap_certificate["derived_coefficients"] == {"v0": "15", "v1": "-15",
                                                       "v2": "15", "v3": "0"}
    assert gap_certificate["derived_constant"] == "-5"
    assert gap_certificate["modulus"] == 15
    # Integer-feasible systems must stay certificate-less: an integer point
    # forces s_i | (U*d)_i on every Smith pivot, so no row can qualify.
    for feasible_rows in ([((2, 2), 2)], [((1, 1), 2), ((1, -1), 0)], [((2, 4), 0), ((4, 4), 0)]):
        feasible = linear(
            domain="integer", variables=("x", "y"),
            premises=[{"coefficients": {"x": coeffs[0], "y": coeffs[1]},
                       "relation": "==", "constant": const} for coeffs, const in feasible_rows],
            conclusion={"coefficients": {"x": 1}, "relation": ">=", "constant": 0})
        feasible_statement = feasible["obligations"][0]["statement"]
        assert _integer_congruence_infeasibility_certificate(
            feasible_statement, validate_linear_statement(feasible_statement)) is None


def test_snf_certificate_extends_dispatch_to_previously_unresolved_systems(monkeypatch):
    # End-to-end: this system left the dispatch honestly unresolved before
    # the SNF stage (no Farkas/Motzkin/bounds/echelon certificate); now the
    # congruence layer closes it.
    spec = linear(
        domain="integer", variables=("v0", "v1", "v2", "v3"),
        premises=[{"coefficients": {"v0": -3, "v1": 3, "v2": 2, "v3": 2},
                   "relation": "==", "constant": -3},
                  {"coefficients": {"v0": -2, "v1": -3, "v2": 3, "v3": 3},
                   "relation": "==", "constant": 3},
                  {"coefficients": {"v0": 2, "v1": -3, "v2": 4, "v3": 1},
                   "relation": "==", "constant": -1}],
        conclusion={"coefficients": {"v0": 1}, "relation": ">=", "constant": 0})
    obligation = compile_theory(spec)["obligations"][0]
    assert obligation["status"] == "unresolved"
    assert obligation["evidence"]["vacuous_implication_rejected"] is True
    assert obligation["evidence"]["portable_certificate_checker"] == \
        "exact_integer_congruence_infeasibility/v1"
    certificate = obligation["evidence"]["portable_certificate"]
    assert certificate["kind"] == "integer_congruence_infeasibility"
    assert certificate["modulus"] == 15
    # Determinism: the search is a pure function of the statement.
    replay = compile_theory(spec)["obligations"][0]
    assert json.dumps(replay["evidence"], sort_keys=True) == \
        json.dumps(obligation["evidence"], sort_keys=True)
    # Frozen-cert replay without z3: the portable dispatch still resolves
    # the checker from the certificate alone, and re-derives the identical
    # certificate.
    portable = copy.deepcopy(spec)
    portable["obligations"][0]["statement"]["portable_certificate"] = certificate
    import sys
    monkeypatch.setitem(sys.modules, "z3", None)
    replayed = compile_theory(portable)["obligations"][0]
    assert replayed["checker"] == "exact_integer_congruence_infeasibility/v1"
    assert replayed["evidence"]["backend"] == "portable_certificate"
    assert replayed["evidence"]["portable_certificate"] == certificate
    # Tampered certificates stay rejected: a shifted constant no longer
    # matches the multiplier combination, a swapped sign flips the row, and
    # a wrong modulus diverges from the derived gcd.
    from researchclaw.pipeline.formal_proof import (FormalProofError,
                                                    verify_integer_congruence_infeasibility)
    with pytest.raises(FormalProofError, match="differs from the multiplier combination"):
        verify_integer_congruence_infeasibility(
            spec["obligations"][0]["statement"],
            dict(certificate, derived_constant="-6"))
    with pytest.raises(FormalProofError, match="differs from the multiplier combination"):
        verify_integer_congruence_infeasibility(
            spec["obligations"][0]["statement"],
            dict(certificate, row_multipliers=["1", "1", "-5"]))
    with pytest.raises(FormalProofError, match="Declared modulus differs"):
        verify_integer_congruence_infeasibility(
            spec["obligations"][0]["statement"], dict(certificate, modulus=16))


def test_congruence_dispatch_prefers_the_echelon_certificate():
    from researchclaw.pipeline.formal_proof import (_integer_congruence_infeasibility_certificate,
                                                    validate_linear_statement)
    # When both searches fire, the echelon certificate is kept: 2x - 2y == 1
    # is caught by the initial primitivization test, multipliers ["1"], and
    # the SNF stage never runs.
    spec = linear(
        domain="integer", variables=("x", "y"),
        premises=[{"coefficients": {"x": 2, "y": -2}, "relation": "==", "constant": 1}],
        conclusion={"coefficients": {"x": 1}, "relation": ">=", "constant": 0})
    statement = spec["obligations"][0]["statement"]
    certificate = _integer_congruence_infeasibility_certificate(
        statement, validate_linear_statement(statement))
    assert certificate["row_multipliers"] == ["1"]
    assert certificate["derived_coefficients"] == {"x": "2", "y": "-2"}
    assert certificate["derived_constant"] == "1"
    assert certificate["modulus"] == 2


def test_cg_cut_certificate_closes_the_band_class():
    # The band 1/2 <= x + y <= 9/10 is real-feasible, integer-infeasible,
    # and carries no pinning bounds and no effective equalities, so Farkas,
    # Motzkin, the bounded box and the congruence layers all stay
    # certificate-less. One rounding cut closes it: x + y <= floor(9/10) = 0
    # holds at every integer point, and together with the original lower
    # row -x - y <= -1/2 the real relaxation of the derived system is
    # empty — Farkas territory over the cuts and the premise rows.
    spec = linear(
        domain="integer",
        premises=[{"coefficients": {"x": 1, "y": 1}, "relation": "<=", "constant": "9/10"},
                  {"coefficients": {"x": 1, "y": 1}, "relation": ">=", "constant": "1/2"}],
        conclusion={"coefficients": {"x": 1}, "relation": ">=", "constant": 0})
    obligation = compile_theory(spec)["obligations"][0]
    assert obligation["status"] == "unresolved"
    assert obligation["evidence"]["vacuous_implication_rejected"] is True
    assert obligation["evidence"]["portable_certificate_checker"] == \
        "exact_integer_cg_cut_infeasibility/v1"
    assert obligation["evidence"]["certificate_proof_domain"] == "integer"
    assert obligation["evidence"]["portable_certificate_verified"] is True
    certificate = obligation["evidence"]["portable_certificate"]
    assert certificate["kind"] == "integer_cg_cut_infeasibility"
    assert certificate["closed_premise_rows"] == ["premise_1:upper", "premise_2:lower"]
    assert certificate["strict_premise_rows"] == []
    assert certificate["cuts"] and all(
        cut["closed_multipliers"].count("1") + cut["strict_multipliers"].count("1") >= 1
        for cut in certificate["cuts"])
    from researchclaw.pipeline.formal_proof import verify_integer_cg_cut_infeasibility
    assert verify_integer_cg_cut_infeasibility(
        spec["obligations"][0]["statement"], certificate) is True


def test_cg_cut_dispatch_covers_strict_rows_and_spares_feasible_bands():
    from researchclaw.pipeline.formal_proof import verify_integer_cg_cut_infeasibility
    # Strict variant: x + y < 9/10 with x + y > 1/2 produces no closed rows,
    # so Motzkin has no closed multiplier to lean on. The strict cuts round
    # with ceil(d/g) - 1, landing on the same contradiction.
    strict_spec = linear(
        domain="integer",
        premises=[{"coefficients": {"x": 1, "y": 1}, "relation": "<", "constant": "9/10"},
                  {"coefficients": {"x": 1, "y": 1}, "relation": ">", "constant": "1/2"}],
        conclusion={"coefficients": {"x": 1}, "relation": ">=", "constant": 0})
    obligation = compile_theory(strict_spec)["obligations"][0]
    assert obligation["status"] == "unresolved"
    assert obligation["evidence"]["portable_certificate_checker"] == \
        "exact_integer_cg_cut_infeasibility/v1"
    certificate = obligation["evidence"]["portable_certificate"]
    assert certificate["closed_premise_rows"] == []
    assert certificate["strict_premise_rows"] == ["premise_1:strict", "premise_2:strict"]
    assert verify_integer_cg_cut_infeasibility(
        strict_spec["obligations"][0]["statement"], certificate) is True
    # The widened band 1/2 <= x + y <= 3/2 admits the integer point
    # x + y == 1: no CG certificate may exist, and the dispatch cannot
    # disguise a counterexample as an infeasibility proof.
    feasible_spec = linear(
        domain="integer",
        premises=[{"coefficients": {"x": 1, "y": 1}, "relation": "<=", "constant": "3/2"},
                  {"coefficients": {"x": 1, "y": 1}, "relation": ">=", "constant": "1/2"}],
        conclusion={"coefficients": {"x": 1}, "relation": ">=", "constant": 0})
    feasible = compile_theory(feasible_spec)["obligations"][0]
    feasible_certificate = feasible["evidence"].get("portable_certificate")
    assert feasible_certificate is not None  # the implication is disproved …
    assert feasible_certificate["kind"] == "linear_counterexample_witness"  # … by witness
    assert feasible["status"] == "disproved"


def test_cg_cut_certificate_freezes_without_z3(monkeypatch):
    spec = linear(
        domain="integer",
        premises=[{"coefficients": {"x": 1, "y": 1}, "relation": "<=", "constant": "9/10"},
                  {"coefficients": {"x": 1, "y": 1}, "relation": ">=", "constant": "1/2"}],
        conclusion={"coefficients": {"x": 1}, "relation": ">=", "constant": 0})
    from researchclaw.pipeline.formal_proof import verify_integer_cg_cut_infeasibility
    # The witness multipliers are solver-chosen and may differ between
    # runs; the layer's verdict is what must reproduce: the same checker
    # fires on the same premise rows and every certificate it mints
    # verifies exactly.
    for _ in range(2):
        obligation = compile_theory(spec)["obligations"][0]
        assert obligation["evidence"]["portable_certificate_checker"] == \
            "exact_integer_cg_cut_infeasibility/v1"
        assert obligation["evidence"]["certificate_proof_domain"] == "integer"
        assert verify_integer_cg_cut_infeasibility(
            spec["obligations"][0]["statement"],
            obligation["evidence"]["portable_certificate"]) is True
    certificate = compile_theory(spec)["obligations"][0]["evidence"]["portable_certificate"]
    # Frozen-cert replay without z3: the portable dispatch re-verifies the
    # certificate with the pure-Python CG verifier and echoes it unchanged.
    portable = copy.deepcopy(spec)
    portable["obligations"][0]["statement"]["portable_certificate"] = certificate
    import sys
    monkeypatch.setitem(sys.modules, "z3", None)
    replayed = compile_theory(portable)["obligations"][0]
    assert replayed["checker"] == "exact_integer_cg_cut_infeasibility/v1"
    assert replayed["evidence"]["backend"] == "portable_certificate"
    assert replayed["evidence"]["portable_certificate"] == certificate


def test_cg_cut_verifier_rejects_forgeries():
    spec = linear(
        domain="integer",
        premises=[{"coefficients": {"x": 1, "y": 1}, "relation": "<=", "constant": "9/10"},
                  {"coefficients": {"x": 1, "y": 1}, "relation": ">=", "constant": "1/2"}],
        conclusion={"coefficients": {"x": 1}, "relation": ">=", "constant": 0})
    statement = spec["obligations"][0]["statement"]
    certificate = compile_theory(spec)["obligations"][0]["evidence"]["portable_certificate"]
    from researchclaw.pipeline.formal_proof import (FormalProofError,
                                                    verify_integer_cg_cut_infeasibility)

    def forge(mutate, message):
        tampered = json.loads(json.dumps(certificate))
        mutate(tampered)
        with pytest.raises(FormalProofError, match=message):
            verify_integer_cg_cut_infeasibility(statement, tampered)

    # A cut that does not re-derive from its declared multipliers: wrong
    # rounding, negative or fractional derivation multipliers, a derived
    # row that disagrees with the combination.
    def wrong_rounding(c):
        c["cuts"][0]["derived_constant"] = "1"

    def negative_multiplier(c):
        c["cuts"][0]["closed_multipliers"][0] = "-1"

    def fractional_multiplier(c):
        c["cuts"][0]["closed_multipliers"][0] = "1/2"

    def wrong_derived_row(c):
        c["cuts"][0]["derived_coefficients"] = {"x": "1", "y": "2"}

    # A contradiction that never leaves the real relaxation is Farkas
    # territory, not CG; a witness constant that disagrees with the
    # multipliers is arithmetic theft.
    def cut_less_witness(c):
        c["cut_multipliers"] = ["0" for _ in c["cut_multipliers"]]

    def tampered_combined_constant(c):
        c["combined_constant"] = "-2"

    def hash_theft(c):
        c["statement_hash"] = "0" * 64

    forge(wrong_rounding, "differs from the multiplier combination")
    forge(negative_multiplier, "must be nonnegative and complete")
    forge(fractional_multiplier, "must be integers")
    forge(wrong_derived_row, "differs from the multiplier combination")
    forge(cut_less_witness, "must use at least one cut row")
    forge(tampered_combined_constant, "do not derive a CG contradiction")
    forge(hash_theft, "Invalid CG cut certificate identity")
    # The certificate is an integer-domain proof: the real statement it was
    # minted from does not accept it.
    with pytest.raises(FormalProofError, match="require the integer domain"):
        verify_integer_cg_cut_infeasibility(dict(statement, domain="real"), certificate)


def test_staged_cut_pool_closes_a_system_the_old_pool_misses():
    from researchclaw.pipeline.formal_proof import verify_integer_cg_cut_infeasibility
    # Real-feasible, integer-infeasible, no effective equality (congruence
    # blind), and the directions v0 >= / v1 <= have no row-space pinning
    # proof (bounded-box blind): every valid bounding box keeps integer
    # points. The old pool (combinations of at most three rows with
    # multipliers up to two) admits no refutation witness for this system;
    # the staged tiers (four rows with multipliers up to two, three rows
    # with multipliers up to three) supply the missing cut — the witness
    # needs a derivation whose multiplier reaches three, which is exactly
    # what the old cap excluded. Because the old pool's witness search is
    # unsat for this system, EVERY valid witness must use at least one cut
    # outside the old pool, so that property is asserted instead of any
    # solver-chosen multiplier.
    spec = linear(
        domain="integer",
        variables=("v0", "v1", "v2"),
        premises=[
            {"coefficients": {"v0": 3, "v1": 3, "v2": 1}, "relation": "<", "constant": 1},
            {"coefficients": {"v0": 2, "v1": 2, "v2": -3}, "relation": "<", "constant": 0},
            {"coefficients": {"v0": -3, "v1": -3, "v2": -1}, "relation": "<=", "constant": 1},
            {"coefficients": {"v0": 1, "v1": 1, "v2": -3}, "relation": ">=", "constant": -3},
            {"coefficients": {"v0": 2, "v1": 3, "v2": 1}, "relation": "<", "constant": 1},
        ],
        conclusion={"coefficients": {"v0": 1}, "relation": ">=", "constant": 0})
    obligation = compile_theory(spec)["obligations"][0]
    assert obligation["status"] == "unresolved"
    evidence = obligation["evidence"]
    assert evidence["vacuous_implication_rejected"] is True
    assert evidence["portable_certificate_checker"] == \
        "exact_integer_cg_cut_infeasibility/v1"
    assert evidence["certificate_proof_domain"] == "integer"
    assert evidence["portable_certificate_verified"] is True
    certificate = evidence["portable_certificate"]
    assert certificate["kind"] == "integer_cg_cut_infeasibility"
    assert certificate["closed_premise_rows"] == ["premise_3:upper", "premise_4:lower"]
    assert certificate["strict_premise_rows"] == \
        ["premise_1:strict", "premise_2:strict", "premise_5:strict"]

    def needs_staged_tier(cut):
        multipliers = [int(value) for value
                       in cut["closed_multipliers"] + cut["strict_multipliers"]]
        return max(multipliers) >= 3 or sum(1 for value in multipliers if value) >= 4

    assert certificate["cuts"]
    assert any(needs_staged_tier(cut) for cut in certificate["cuts"])
    assert verify_integer_cg_cut_infeasibility(
        spec["obligations"][0]["statement"], certificate) is True


def test_iterated_cut_closure_closes_the_free_z_diamond():
    from fractions import Fraction
    from researchclaw.pipeline.formal_proof import verify_integer_iterated_cg_cut_infeasibility
    # Real-feasible, integer-infeasible (the diamond bands force x + y = 1
    # and x - y = 0 for integers, so x = y = 1/2), and unbounded: z carries
    # no premises, so the relaxation has no finite box and the bounded-box
    # enumeration honestly refuses. Congruence is blind (no equality rows),
    # and the single-round cut pool admits no refutation witness — the
    # rank-one closure keeps the real point (1/2, 1/2, t). The iterated
    # layer feeds the single-row band cuts back as rows and closes the
    # class at rank two (x <= 0 and x >= 1). Because the single-round
    # witness search is unsat, EVERY valid iterated witness must weight a
    # round-two cut whose derivation consumes a round-one row — cuts over
    # premise rows alone plus round-one rows span exactly the single-round
    # search space — so that property is asserted instead of any
    # solver-chosen multiplier.
    spec = linear(
        domain="integer",
        variables=("x", "y", "z"),
        premises=[
            {"coefficients": {"x": 1, "y": 1}, "relation": "<=", "constant": "3/2"},
            {"coefficients": {"x": -1, "y": -1}, "relation": "<=", "constant": "-1/2"},
            {"coefficients": {"x": 1, "y": -1}, "relation": "<=", "constant": "1/2"},
            {"coefficients": {"x": -1, "y": 1}, "relation": "<=", "constant": "1/2"},
        ],
        conclusion={"coefficients": {"x": 1}, "relation": ">=", "constant": 0})
    obligation = compile_theory(spec)["obligations"][0]
    assert obligation["status"] == "unresolved"
    evidence = obligation["evidence"]
    assert evidence["vacuous_implication_rejected"] is True
    assert evidence["portable_certificate_checker"] == \
        "exact_integer_iterated_cg_cut_infeasibility/v1"
    assert evidence["certificate_proof_domain"] == "integer"
    assert evidence["portable_certificate_verified"] is True
    certificate = evidence["portable_certificate"]
    assert certificate["kind"] == "integer_iterated_cg_cut_infeasibility"
    assert certificate["closed_premise_rows"] == [
        "premise_1:upper", "premise_2:upper", "premise_3:upper", "premise_4:upper"]
    assert certificate["strict_premise_rows"] == []
    for entry in certificate["round1_cuts"]:
        assert sum(int(value) for value in entry["closed_multipliers"]
                   + entry["strict_multipliers"]) == 1
    round1_width = len(certificate["round1_cuts"])
    assert any(
        Fraction(multiplier) != 0
        and any(int(value) != 0 for value in cut["closed_multipliers"][:round1_width])
        for multiplier, cut in zip(certificate["cut_multipliers"], certificate["cuts"]))
    assert verify_integer_iterated_cg_cut_infeasibility(
        spec["obligations"][0]["statement"], certificate) is True


def test_iterated_cut_verifier_rejects_forgeries():
    spec = linear(
        domain="integer",
        variables=("x", "y", "z"),
        premises=[
            {"coefficients": {"x": 1, "y": 1}, "relation": "<=", "constant": "3/2"},
            {"coefficients": {"x": -1, "y": -1}, "relation": "<=", "constant": "-1/2"},
            {"coefficients": {"x": 1, "y": -1}, "relation": "<=", "constant": "1/2"},
            {"coefficients": {"x": -1, "y": 1}, "relation": "<=", "constant": "1/2"},
        ],
        conclusion={"coefficients": {"x": 1}, "relation": ">=", "constant": 0})
    statement = spec["obligations"][0]["statement"]
    certificate = compile_theory(spec)["obligations"][0]["evidence"]["portable_certificate"]
    from researchclaw.pipeline.formal_proof import (FormalProofError,
                                                    verify_integer_iterated_cg_cut_infeasibility)

    def forge(mutate, message):
        tampered = json.loads(json.dumps(certificate))
        mutate(tampered)
        with pytest.raises(FormalProofError, match=message):
            verify_integer_iterated_cg_cut_infeasibility(statement, tampered)

    # Round one is pinned to one unit multiplier on a single premise row;
    # its derived row must re-derive exactly.
    def non_unit_round1(c):
        c["round1_cuts"][0]["closed_multipliers"][0] = "2"

    def fractional_round1(c):
        c["round1_cuts"][0]["closed_multipliers"][0] = "1/2"

    def wrong_round1_row(c):
        c["round1_cuts"][0]["derived_constant"] = str(
            int(c["round1_cuts"][0]["derived_constant"]) - 1)

    # Round two multipliers index the merged space (round-one rows first,
    # then premises); they stay integral and must re-derive the declared cut.
    def wrong_round2_rounding(c):
        c["cuts"][0]["derived_constant"] = str(
            int(c["cuts"][0]["derived_constant"]) - 1)

    def fractional_round2(c):
        c["cuts"][0]["closed_multipliers"][0] = "3/2"

    def round2_length_mismatch(c):
        c["cuts"][0]["closed_multipliers"].append("0")

    # The witness must lean on the iterated structure and re-check exactly.
    def round2_less_witness(c):
        c["cut_multipliers"] = ["0" for _ in c["cut_multipliers"]]

    def tampered_combined_constant(c):
        c["combined_constant"] = "-2"

    def hash_theft(c):
        c["statement_hash"] = "0" * 64

    forge(non_unit_round1, "must be one unit multiplier")
    forge(fractional_round1, "must be nonnegative integers")
    forge(wrong_round1_row, "differs from the multiplier combination")
    forge(wrong_round2_rounding, "differs from the multiplier combination")
    forge(fractional_round2, "must be integers")
    forge(round2_length_mismatch, "must be nonnegative and complete")
    forge(round2_less_witness, "must use at least one round-two cut row")
    forge(tampered_combined_constant, "do not derive a CG contradiction")
    forge(hash_theft, "Invalid iterated CG cut certificate identity")
    with pytest.raises(FormalProofError, match="require the integer domain"):
        verify_integer_iterated_cg_cut_infeasibility(dict(statement, domain="real"), certificate)


def test_feasible_box_rejects_infeasibility_forgery():
    spec = linear(
        domain="integer", variables=("x",),
        premises=[{"coefficients": {"x": 2}, "relation": "==", "constant": 2}],
        conclusion={"coefficients": {"x": 1}, "relation": ">=", "constant": 0})
    from researchclaw.pipeline.formal_proof import (verify_bounded_integer_infeasibility,
                                                    FormalProofError, _statement_hash)
    statement = spec["obligations"][0]["statement"]
    proofs = {"lower": {"closed_multipliers": ["0", "1/2"], "strict_multipliers": [], "constant": "-1"},
              "upper": {"closed_multipliers": ["1/2", "0"], "strict_multipliers": [], "constant": "1"}}
    base = {"schema_version": 1, "kind": "bounded_integer_infeasibility",
            "statement_hash": _statement_hash(statement),
            "premise_rows": ["premise_1:upper", "premise_1:lower"],
            "strict_premise_rows": [], "bounds": {"x": {"lower": 1, "upper": 1}},
            "bound_proofs": {"x": proofs}, "checked_assignments": 1}
    # The bound proofs are genuine (2x == 2 forces x <= 1 and x >= 1), but
    # the declared box contains the satisfying point x = 1, so the
    # infeasibility claim fails the enumeration replay.
    with pytest.raises(FormalProofError, match="contains a satisfying point"):
        verify_bounded_integer_infeasibility(statement, base)
    with pytest.raises(FormalProofError, match="Declared integer bounds differ"):
        verify_bounded_integer_infeasibility(statement, dict(base, bounds={"x": {"lower": 0, "upper": 1}}))


def test_single_cut_fast_path_closes_the_wide_box_timeout_survivor():
    # Measured in the round-64 member search (2000 systems): this 8-row
    # wide-box system is real-feasible and integer-infeasible, but its
    # full cut pool (~2600 cuts) pushed the witness LP past the dispatch
    # budget, so the system stayed honestly unresolved although two
    # single-row cuts already refute it (v0+2v1+v2 <= 0 and
    # -(v0+v1+v2) <= -1 combine with -v1 <= 0 into 0 <= -1). The
    # single-cut fast pass closes it in milliseconds, and because that
    # pass searches only single-row derivations, every frozen cut carries
    # exactly one unit multiplier.
    from fractions import Fraction
    from researchclaw.pipeline.formal_proof import verify_integer_cg_cut_infeasibility
    spec = linear(
        domain="integer", variables=("v0", "v1", "v2"),
        premises=[
            {"coefficients": {"v0": 1}, "relation": "<=", "constant": 75},
            {"coefficients": {"v0": -1}, "relation": "<=", "constant": 1},
            {"coefficients": {"v1": 1}, "relation": "<=", "constant": 54},
            {"coefficients": {"v1": -1}, "relation": "<=", "constant": 0},
            {"coefficients": {"v2": 1}, "relation": "<=", "constant": 48},
            {"coefficients": {"v2": -1}, "relation": "<=", "constant": 5},
            {"coefficients": {"v0": 2, "v1": 4, "v2": 2}, "relation": "<=", "constant": 1},
            {"coefficients": {"v0": 4, "v1": 4, "v2": 4}, "relation": ">=", "constant": 1},
        ],
        conclusion={"coefficients": {"v0": 1}, "relation": ">=", "constant": 0})
    obligation = compile_theory(spec)["obligations"][0]
    assert obligation["status"] == "unresolved"
    evidence = obligation["evidence"]
    assert evidence["vacuous_implication_rejected"] is True
    assert evidence["portable_certificate_checker"] == \
        "exact_integer_cg_cut_infeasibility/v1"
    assert evidence["certificate_proof_domain"] == "integer"
    assert evidence["portable_certificate_verified"] is True
    certificate = evidence["portable_certificate"]
    assert certificate["kind"] == "integer_cg_cut_infeasibility"
    assert certificate["closed_premise_rows"] == \
        [f"premise_{index}:upper" for index in range(1, 8)] + ["premise_8:lower"]
    for cut in certificate["cuts"]:
        weights = [Fraction(value) for value in cut["closed_multipliers"]] \
            + [Fraction(value) for value in cut["strict_multipliers"]]
        assert sum(weights) == 1
        assert sum(1 for value in weights if value) == 1
    verify_integer_cg_cut_infeasibility(spec["obligations"][0]["statement"], certificate)


def test_witness_search_retries_after_resource_unknown(monkeypatch):
    # Round-64 hardening: a witness LP that answers unknown under the
    # caller's budget is retried once under the raised fixed budget,
    # while an unsat answer stays final. The staged-pool fixture's
    # witness needs combination cuts, so both the fast single-row pass
    # and the full-pool pass must run; a caller solver that always
    # answers unknown makes that deterministic, and only the retry
    # recovers the witness. With the retry budget itself exhausted the
    # search degrades to honest None — never a crash, never a claim.
    import z3
    from researchclaw.pipeline.formal_proof import (
        _alternative_rows, _cg_cut_pool, _cg_cut_witness, _cg_scale_row,
        validate_linear_statement,
    )

    class UnknownSolver:
        def __init__(self):
            self.constraints = []

        def add(self, *constraints):
            self.constraints.extend(constraints)

        def check(self):
            return z3.unknown

    spec = linear(
        domain="integer", variables=("v0", "v1", "v2"),
        premises=[
            {"coefficients": {"v0": 3, "v1": 3, "v2": 1}, "relation": "<", "constant": 1},
            {"coefficients": {"v0": 2, "v1": 2, "v2": -3}, "relation": "<", "constant": 0},
            {"coefficients": {"v0": -3, "v1": -3, "v2": -1}, "relation": "<=", "constant": 1},
            {"coefficients": {"v0": 1, "v1": 1, "v2": -3}, "relation": ">=", "constant": -3},
            {"coefficients": {"v0": 2, "v1": 3, "v2": 1}, "relation": "<", "constant": 1},
        ],
        conclusion={"coefficients": {"v0": 1}, "relation": ">=", "constant": 0})
    statement = spec["obligations"][0]["statement"]
    normalized = validate_linear_statement(statement)
    variables = normalized["variables"]
    closed_rows, strict_rows = _alternative_rows(normalized)
    closed_scaled = [_cg_scale_row(coefficients, bound, variables)
                     for _, coefficients, bound in closed_rows]
    strict_scaled = [_cg_scale_row(coefficients, bound, variables)
                     for _, coefficients, bound in strict_rows]
    pool = _cg_cut_pool(closed_scaled, strict_scaled)
    assert len(pool) > len(closed_rows) + len(strict_rows)

    witness = _cg_cut_witness(pool, closed_scaled, strict_scaled, variables,
                              UnknownSolver, _exact_z3_value, z3)
    assert witness is not None
    monkeypatch.setattr(
        "researchclaw.pipeline.formal_proof._CG_WITNESS_RETRY_TIMEOUT_MS", 1)
    monkeypatch.setattr(
        "researchclaw.pipeline.formal_proof._CG_WITNESS_RETRY_RLIMIT", 1)
    assert _cg_cut_witness(pool, closed_scaled, strict_scaled, variables,
                           UnknownSolver, _exact_z3_value, z3) is None


def _exact_z3_value(value):
    import z3
    return z3.IntVal(value.numerator) if value.denominator == 1 \
        else z3.RealVal(f"{value.numerator}/{value.denominator}")


def test_oversized_row_scale_certifies_exactly():
    # Round-64 probe: z3 model fractions extract exactly at any size
    # (the extraction converts numerator and denominator as strings, not
    # through machine words), so a band whose lower row carries 2**70
    # -scale coefficients still certifies and verifies — no exception
    # escapes the attach path, and the witness needs a multiplier whose
    # numerator or denominator exceeds any machine word.
    spec = linear(
        domain="integer", variables=("x", "y", "z"),
        premises=[
            {"coefficients": {"x": 1, "y": 1, "z": 1}, "relation": "<=",
             "constant": "9/10"},
            {"coefficients": {"x": -(2 ** 70), "y": -(2 ** 70), "z": -(2 ** 70)},
             "relation": "<=", "constant": -(2 ** 69)},
        ],
        conclusion={"coefficients": {"x": 1}, "relation": ">=", "constant": 0})
    obligation = compile_theory(spec)["obligations"][0]
    assert obligation["status"] == "unresolved"
    evidence = obligation["evidence"]
    assert evidence["vacuous_implication_rejected"] is True
    assert evidence["portable_certificate_verified"] is True
    certificate = evidence["portable_certificate"]
    assert certificate["kind"] == "integer_cg_cut_infeasibility"


def test_focused_iteration_pool_closes_the_extra_row_diamond():
    from fractions import Fraction
    from researchclaw.pipeline.formal_proof import verify_integer_iterated_cg_cut_infeasibility
    # Round-65 member: the free-z diamond plus one extra closed row on the
    # free variable. The merged iteration space is ten rows — beyond the
    # staged pool's eight-row enumeration limit — so round two used to
    # degenerate to a singles-only pool and the class stayed honestly
    # unresolved even with unlimited solver budget. The focused round-two
    # pool (combinations drawn from the fed-back cut rows) closes it. As in
    # the round-63 fixture, the single-round witness search is unsat, so
    # every valid iterated witness must weight a round-two cut whose
    # derivation consumes a round-one row; the merged-width property is
    # asserted too because the focused pool's multipliers index the full
    # merged space.
    spec = linear(
        domain="integer",
        variables=("x", "y", "z"),
        premises=[
            {"coefficients": {"x": 1, "y": 1}, "relation": "<=", "constant": "3/2"},
            {"coefficients": {"x": -1, "y": -1}, "relation": "<=", "constant": "-1/2"},
            {"coefficients": {"x": 1, "y": -1}, "relation": "<=", "constant": "5/2"},
            {"coefficients": {"x": -1, "y": 1}, "relation": "<=", "constant": "-3/2"},
            {"coefficients": {"z": 1}, "relation": ">=", "constant": "-1"},
        ],
        conclusion={"coefficients": {"x": 1}, "relation": ">=", "constant": 0})
    obligation = compile_theory(spec)["obligations"][0]
    assert obligation["status"] == "unresolved"
    evidence = obligation["evidence"]
    assert evidence["vacuous_implication_rejected"] is True
    assert evidence["portable_certificate_checker"] == \
        "exact_integer_iterated_cg_cut_infeasibility/v1"
    assert evidence["certificate_proof_domain"] == "integer"
    assert evidence["portable_certificate_verified"] is True
    certificate = evidence["portable_certificate"]
    assert certificate["kind"] == "integer_iterated_cg_cut_infeasibility"
    assert certificate["closed_premise_rows"] == [
        "premise_1:upper", "premise_2:upper", "premise_3:upper",
        "premise_4:upper", "premise_5:lower"]
    assert certificate["strict_premise_rows"] == []
    for entry in certificate["round1_cuts"]:
        assert sum(int(value) for value in entry["closed_multipliers"]
                   + entry["strict_multipliers"]) == 1
    round1_width = len(certificate["round1_cuts"])
    assert round1_width == 5
    for cut in certificate["cuts"]:
        assert len(cut["closed_multipliers"]) == round1_width + 5
    assert any(
        Fraction(multiplier) != 0
        and any(int(value) != 0 for value in cut["closed_multipliers"][:round1_width])
        for multiplier, cut in zip(certificate["cut_multipliers"], certificate["cuts"]))
    assert verify_integer_iterated_cg_cut_infeasibility(
        spec["obligations"][0]["statement"], certificate) is True


def test_focused_pool_enumerates_only_cut_row_combinations():
    from researchclaw.pipeline.formal_proof import (
        _alternative_rows, _cg_cut_pool, _cg_focused_cut_pool,
        _cg_rounded_cut, _cg_scale_row, validate_linear_statement)
    # The focused round-two pool keeps single-row cuts over the whole merged
    # space but draws multi-row derivations only from the round-one cut
    # rows, with multiplier vectors still indexing the full merged space.
    # The plain pool on the same too-wide space stays singles-only — the
    # degeneration that motivated the focused variant.
    premises = [
        {"coefficients": {"x": 1, "y": 1}, "relation": "<=", "constant": "3/2"},
        {"coefficients": {"x": -1, "y": -1}, "relation": "<=", "constant": "-1/2"},
        {"coefficients": {"x": 1, "y": -1}, "relation": "<=", "constant": "5/2"},
        {"coefficients": {"x": -1, "y": 1}, "relation": "<=", "constant": "-3/2"},
        {"coefficients": {"z": 1}, "relation": ">=", "constant": "-1"},
    ]
    statement = {"kind": "linear_arithmetic", "domain": "integer",
                 "variables": ["x", "y", "z"], "premises": premises,
                 "conclusion": {"coefficients": {"x": 1},
                                "relation": ">=", "constant": 0}}
    normalized = validate_linear_statement(statement)
    closed_rows, strict_rows = _alternative_rows(normalized)
    variables = normalized["variables"]
    closed_scaled = [_cg_scale_row(coefficients, constant, variables)
                     for _, coefficients, constant in closed_rows]
    strict_scaled = [_cg_scale_row(coefficients, constant, variables)
                     for _, coefficients, constant in strict_rows]
    premise_rows = ([(coefficients, bound, False)
                     for coefficients, bound in closed_scaled]
                    + [(coefficients, bound, True)
                       for coefficients, bound in strict_scaled])
    round1 = []
    for index in range(len(premise_rows)):
        multipliers = [0] * len(premise_rows)
        multipliers[index] = 1
        cut = _cg_rounded_cut(premise_rows, multipliers)
        if cut is not None:
            round1.append(cut)
    assert len(round1) == 5
    merged = [(cut[0], cut[1]) for cut in round1] + closed_scaled
    assert len(merged) + len(strict_scaled) > 8
    pool = _cg_focused_cut_pool(merged, strict_scaled, len(round1))
    assert pool
    assert any(sum(closed) + sum(strict) > 1 for closed, strict, _ in pool)
    for closed_mults, strict_mults, _ in pool:
        assert len(closed_mults) == len(merged)
        if sum(closed_mults) + sum(strict_mults) > 1:
            assert sum(closed_mults[len(round1):]) == 0
    degenerate = _cg_cut_pool(merged, strict_scaled)
    assert degenerate
    assert all(sum(closed) + sum(strict) == 1
               for closed, strict, _ in degenerate)


def test_iterated_gate_expansion_reaches_merged_sixteen():
    import z3
    from researchclaw.pipeline.formal_proof import (
        _alternative_rows, _cg_iterated_cut_infeasibility_certificate,
        validate_linear_statement,
        verify_integer_iterated_cg_cut_infeasibility)

    def unlimited():
        return z3.SolverFor("QF_LIA")

    def exact(value):
        return z3.IntVal(value.numerator) if value.denominator == 1 \
            else z3.RealVal(f"{value.numerator}/{value.denominator}")

    # Two independent free-z diamonds: eight closed premises, so the merged
    # space is sixteen rows and the iteration gate must admit it (the mint
    # runs under an unlimited-budget factory here because the focused-pool
    # witness LP is machine-dependently slow; the production-budget E2E
    # path stays uncovered by design, matching the round-64 wb176 decision).
    premises = [
        {"coefficients": {"x": 1, "y": 1}, "relation": "<=", "constant": "3/2"},
        {"coefficients": {"x": -1, "y": -1}, "relation": "<=", "constant": "-1/2"},
        {"coefficients": {"x": 1, "y": -1}, "relation": "<=", "constant": "5/2"},
        {"coefficients": {"x": -1, "y": 1}, "relation": "<=", "constant": "-3/2"},
        {"coefficients": {"u": 1, "v": 1}, "relation": "<=", "constant": "1/2"},
        {"coefficients": {"u": -1, "v": -1}, "relation": "<=", "constant": "-1/2"},
        {"coefficients": {"u": 1, "v": -1}, "relation": "<=", "constant": "3/2"},
        {"coefficients": {"u": -1, "v": 1}, "relation": "<=", "constant": "-1/2"},
    ]
    statement = {"kind": "linear_arithmetic", "domain": "integer",
                 "variables": ["x", "y", "u", "v", "z"], "premises": premises,
                 "conclusion": {"coefficients": {"x": 1},
                                "relation": ">=", "constant": 0}}
    normalized = validate_linear_statement(statement)
    split = _alternative_rows(normalized)
    certificate = _cg_iterated_cut_infeasibility_certificate(
        statement, normalized, split, unlimited, exact, z3)
    assert certificate is not None
    assert certificate["kind"] == "integer_iterated_cg_cut_infeasibility"
    assert len(certificate["round1_cuts"]) == 8
    assert len(certificate["cuts"][0]["closed_multipliers"]) == 16
    assert verify_integer_iterated_cg_cut_infeasibility(statement, certificate) is True

    # One premise pair past the gate: ten premises round up to a merged
    # space of twenty rows and the mint honestly refuses before any search.
    wide = dict(statement)
    wide["premises"] = premises + [
        {"coefficients": {"z": 1}, "relation": "<=", "constant": "7"},
        {"coefficients": {"z": -1}, "relation": "<=", "constant": "5"},
    ]
    wide_normalized = validate_linear_statement(wide)
    wide_split = _alternative_rows(wide_normalized)
    assert _cg_iterated_cut_infeasibility_certificate(
        wide, wide_normalized, wide_split, unlimited, exact, z3) is None


def test_disproved_implication_gets_a_portable_counterexample_witness(monkeypatch):
    spec = linear(
        variables=("x",),
        premises=[{"coefficients": {"x": 1}, "relation": ">=", "constant": 0}],
        conclusion={"coefficients": {"x": 1}, "relation": ">=", "constant": 1})
    obligation = compile_theory(spec)["obligations"][0]
    assert obligation["status"] == "disproved"
    assert obligation["evidence"]["counterexample_exactly_validated"] is True
    certificate = obligation["evidence"]["portable_certificate"]
    assert certificate["kind"] == "linear_counterexample_witness"
    assert obligation["evidence"]["portable_certificate_verified"] is True
    from researchclaw.pipeline.formal_proof import verify_counterexample_witness
    statement = spec["obligations"][0]["statement"]
    assert verify_counterexample_witness(statement, certificate)
    portable = copy.deepcopy(spec)
    portable["obligations"][0]["statement"]["portable_certificate"] = certificate
    import sys
    monkeypatch.setitem(sys.modules, "z3", None)
    recompiled = compile_theory(portable)["obligations"][0]
    assert recompiled["status"] == "disproved"
    assert recompiled["checker"] == "exact_fraction_counterexample/v1"
    assert recompiled["evidence"]["counterexample"] == obligation["evidence"]["counterexample"]
    assert recompiled["evidence"]["counterexample_exactly_validated"] is True
    broken = copy.deepcopy(portable)
    broken["obligations"][0]["statement"]["portable_certificate"]["witness"]["x"] = "999"
    with pytest.raises(WorkbenchError):
        compile_theory(broken)


def test_integer_counterexample_witness_rejects_rational_points():
    spec = linear(
        domain="integer", variables=("x",),
        premises=[{"coefficients": {"x": 1}, "relation": ">=", "constant": 0}],
        conclusion={"coefficients": {"x": 1}, "relation": ">=", "constant": 1})
    obligation = compile_theory(spec)["obligations"][0]
    assert obligation["status"] == "disproved"
    certificate = copy.deepcopy(obligation["evidence"]["portable_certificate"])
    certificate["witness"]["x"] = "1/2"
    from researchclaw.pipeline.formal_proof import verify_counterexample_witness, FormalProofError
    with pytest.raises(FormalProofError, match="integral"):
        verify_counterexample_witness(spec["obligations"][0]["statement"], certificate)


def test_unknown_portable_certificate_kind_fails_loudly():
    broken = linear()
    broken["obligations"][0]["statement"]["portable_certificate"] = {"kind": "bogus"}
    with pytest.raises(WorkbenchError):
        compile_theory(broken)


def test_z3_backend_absence_is_honest_and_grammar_still_validated(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "z3", None)
    obligation = compile_theory(linear())["obligations"][0]
    assert obligation["status"] == "unresolved" and obligation["evidence"]["backend"] == "unavailable"
    broken = linear()
    broken["obligations"][0]["statement"]["conclusion"]["constant"] = "1.5"
    with pytest.raises(WorkbenchError, match="canonical rational"):
        compile_theory(broken)


@pytest.mark.parametrize("mutation", [
    lambda s: s["obligations"][0]["statement"].update(domain="complex"),
    lambda s: s["obligations"][0]["statement"].update(variables=[]),
    lambda s: s["obligations"][0]["statement"].update(variables=["x", "x"]),
    lambda s: s["obligations"][0]["statement"].update(variables=["x; import os"]),
    lambda s: s["obligations"][0]["statement"]["premises"][0].update(extra="code"),
    lambda s: s["obligations"][0]["statement"]["premises"][0].update(relation="=>"),
    lambda s: s["obligations"][0]["statement"]["premises"][0].update(coefficients={"z": 1}),
    lambda s: s["obligations"][0]["statement"]["premises"][0].update(coefficients={"x": 0}),
    lambda s: s["obligations"][0]["statement"]["premises"][0].update(coefficients={"x": True}),
    lambda s: s["obligations"][0]["statement"]["premises"][0].update(coefficients={"x": "01"}),
    lambda s: s["obligations"][0]["statement"]["premises"][0].update(coefficients={"x": "1/0"}),
    lambda s: s["obligations"][0].update(assumptions=["x is probably positive"]),
    lambda s: s["obligations"][0].update(proof_text="Trust this proof"),
])
def test_linear_arithmetic_rejects_untyped_or_prose_escape_hatches(mutation):
    spec = linear()
    mutation(spec)
    with pytest.raises(WorkbenchError):
        compile_theory(spec)


def test_linear_arithmetic_resource_bounds_fail_before_solver_call():
    spec = linear(variables=tuple(f"x{i}" for i in range(17)))
    with pytest.raises(WorkbenchError, match="unique Python-style"):
        compile_theory(spec)
    spec = linear(variables=("x",), premises=[
        {"coefficients": {"x": 1}, "relation": ">=", "constant": i} for i in range(65)])
    with pytest.raises(WorkbenchError, match="premise bound"):
        compile_theory(spec)
    spec = linear(variables=("x",), premises=[
        {"coefficients": {"x": 2 ** 513}, "relation": ">=", "constant": 0}])
    with pytest.raises(WorkbenchError, match="arithmetic bound"):
        compile_theory(spec)


def test_linear_participates_in_conjunction_decomposition_and_dependency_downgrade():
    spec = {"schema_version": 1, "definitions": {}, "obligations": [
        {"id": "reviewed", "required": True, "depends_on": [], "assumptions": [],
         "statement": {"kind": "informal", "text": "Reviewed premise"}, "proof_text": "Proof",
         "review": {"checker": "named reviewer", "verdict": "accepted", "evidence": "reasoning"}},
        {"id": "compound", "required": True, "depends_on": ["reviewed"], "assumptions": [],
         "statement": {"kind": "conjunction", "parts": [
             linear(variables=("x",), premises=[], conclusion={"coefficients": {"x": 1},
                    "relation": "==", "constant": 0})["obligations"][0]["statement"],
             {"kind": "polynomial_identity", "variables": ["x"], "left": "(x+1)**2", "right": "x**2+2*x+1"},
         ]}},
    ]}
    bundle = compile_theory(spec)
    parts = {item["id"]: item for item in bundle["obligations"]}
    assert parts["compound_part1"]["status"] == "disproved"
    assert parts["compound_part2"]["status"] == "reviewed_informal"
    assert parts["compound"]["status"] == "disproved"


def decomposed(*, coverage="accepted"):
    coverage_review = None if coverage is None else {
        "checker": "named decomposition reviewer", "verdict": coverage,
        "evidence": "The two subclaims jointly cover the original conditional claim under its stated assumptions.",
    }
    statement = {"kind": "informal_decomposition",
                 "text": "Under regularity condition R, the update preserves feasibility and decreases the objective.",
                 "parts": [
                     {"id": "feasibility", "depends_on": [], "assumptions": [],
                      "statement": {"kind": "linear_arithmetic", "domain": "real", "variables": ["x"],
                                    "premises": [{"coefficients": {"x": 1}, "relation": ">=", "constant": 1}],
                                    "conclusion": {"coefficients": {"x": 1}, "relation": ">=", "constant": 0}}},
                     {"id": "descent", "depends_on": ["feasibility"], "assumptions": [],
                      "statement": {"kind": "polynomial_identity", "variables": ["d"],
                                    "left": "(d-1)**2", "right": "d**2-2*d+1"}},
                 ]}
    if coverage_review is not None:
        statement["coverage_review"] = coverage_review
    return {"schema_version": 1, "definitions": {"R": "The declared regularity conditions hold."}, "obligations": [{
        "id": "main_theorem", "required": True, "depends_on": [], "assumptions": ["Regularity condition R"],
        "statement": statement,
    }]}


def test_assisted_text_decomposition_emits_traceable_dependency_graph_with_status_cap():
    bundle = compile_theory(decomposed())
    indexed = {item["id"]: item for item in bundle["obligations"]}
    assert list(indexed) == ["main_theorem__feasibility", "main_theorem__descent", "main_theorem"]
    assert indexed["main_theorem__feasibility"]["status"] == "machine_checked"
    assert indexed["main_theorem__descent"]["depends_on"] == ["main_theorem__feasibility"]
    assert indexed["main_theorem__descent"]["status"] == "machine_checked"
    parent = indexed["main_theorem"]
    assert parent["status"] == "reviewed_informal"
    assert parent["checker"] == "informal_decomposition_review/v1"
    assert parent["evidence"]["machine_status_cap"] == "reviewed_informal"
    assert parent["depends_on"] == ["main_theorem__feasibility", "main_theorem__descent"]
    assert parent["evidence"]["coverage_review"]["checker"] == "named decomposition reviewer"


@pytest.mark.parametrize("verdict", [None, "rejected", "unresolved"])
def test_decomposition_without_accepted_coverage_never_completes_parent(verdict):
    bundle = compile_theory(decomposed(coverage=verdict))
    parent = bundle["obligations"][-1]
    assert parent["status"] == "unresolved"
    assert parent["evidence"]["coverage_review_required"] is True
    assert all(item["status"] == "machine_checked" for item in bundle["obligations"][:-1])


def test_failed_part_blocks_parent_but_does_not_claim_original_theorem_disproved():
    spec = decomposed()
    spec["obligations"][0]["statement"]["parts"][0]["statement"]["conclusion"]["constant"] = 2
    bundle = compile_theory(spec)
    indexed = {item["id"]: item for item in bundle["obligations"]}
    assert indexed["main_theorem__feasibility"]["status"] == "disproved"
    assert indexed["main_theorem__descent"]["status"] == "unresolved"
    assert indexed["main_theorem__descent"]["evidence"]["unresolved_dependencies"] == ["main_theorem__feasibility"]
    assert indexed["main_theorem"]["status"] == "unresolved"
    assert indexed["main_theorem"]["evidence"]["incomplete_parts"] == [
        "main_theorem__feasibility", "main_theorem__descent"]


def test_reviewed_informal_part_keeps_parent_informal_and_preserves_part_evidence():
    spec = decomposed()
    part = spec["obligations"][0]["statement"]["parts"][1]
    part["statement"] = {"kind": "informal", "text": "The update decreases the nonlinear objective."}
    part["proof_text"] = "A complete prose proof attempt for this bounded subclaim."
    part["review"] = {"checker": "named proof reviewer", "verdict": "accepted",
                      "evidence": "Reviewed line by line under the stated scope."}
    bundle = compile_theory(spec)
    assert bundle["obligations"][1]["status"] == "reviewed_informal"
    assert bundle["obligations"][1]["checker"] == "named proof reviewer"
    assert bundle["obligations"][-1]["status"] == "reviewed_informal"


def test_external_unresolved_dependency_propagates_through_decomposition():
    spec = decomposed()
    spec["obligations"].insert(0, {"id": "external", "required": True, "depends_on": [], "assumptions": [],
        "statement": {"kind": "informal", "text": "An unresolved external lemma."}})
    spec["obligations"][1]["depends_on"] = ["external"]
    bundle = compile_theory(spec)
    indexed = {item["id"]: item for item in bundle["obligations"]}
    assert indexed["main_theorem__feasibility"]["status"] == "unresolved"
    assert indexed["main_theorem"]["status"] == "unresolved"
    assert indexed["main_theorem"]["evidence"]["unresolved_dependencies"] == ["external"]


@pytest.mark.parametrize("mutation,pattern", [
    (lambda s: s["obligations"][0]["statement"].update(parts=[]), "2 to 16"),
    (lambda s: s["obligations"][0]["statement"]["parts"][1].update(id="feasibility"), "Duplicate"),
    (lambda s: s["obligations"][0]["statement"]["parts"][1].update(depends_on=["missing"]), "references"),
    (lambda s: s["obligations"][0]["statement"]["parts"][0].update(depends_on=["descent"]), "Cyclic"),
    (lambda s: s["obligations"][0]["statement"]["parts"][0].update(assumptions=[""]), "assumptions"),
    (lambda s: s["obligations"][0]["statement"]["coverage_review"].update(verdict="machine_checked"), "verdict"),
    (lambda s: s["obligations"][0]["statement"]["coverage_review"].update(evidence=""), "reasoning"),
    (lambda s: s["obligations"][0].update(proof_text="parent override"), "part proofs"),
    (lambda s: s["obligations"][0].update(review={"checker": "x", "verdict": "accepted", "evidence": "x"}), "part proofs"),
])
def test_invalid_informal_decomposition_fails_closed(mutation, pattern):
    spec = decomposed()
    mutation(spec)
    with pytest.raises(WorkbenchError, match=pattern):
        compile_theory(spec)


def test_generated_decomposition_id_collision_is_rejected():
    spec = decomposed()
    spec["obligations"].insert(0, {"id": "main_theorem__feasibility", "required": False,
        "depends_on": [], "assumptions": [], "statement": {"kind": "informal", "text": "Collision"}})
    with pytest.raises(WorkbenchError, match="collides"):
        compile_theory(spec)


def test_final_acceptance_distinguishes_reviewed_and_unreviewed_decomposition(tmp_path):
    from researchclaw.pipeline.final_acceptance import assess_delivery
    dump(tmp_path / "theory_bundle.json", compile_theory(decomposed()))
    reviewed = assess_delivery(tmp_path)
    assert not any(issue["reason"] == "unresolved_proof_obligation" for issue in reviewed["issues"])
    dump(tmp_path / "theory_bundle.json", compile_theory(decomposed(coverage=None)))
    unresolved = assess_delivery(tmp_path)
    assert any(issue["reason"] == "unresolved_proof_obligation" for issue in unresolved["issues"])


def test_conjunction_auto_split_is_machine_checked():
    spec = {"schema_version": 1, "definitions": {}, "obligations": [{
        "id": "pair", "required": True, "depends_on": [], "assumptions": [],
        "statement": {"kind": "conjunction", "parts": [
            {"kind": "polynomial_identity", "variables": ["x", "y"], "left": "(x+y)**2", "right": "x**2 + 2*x*y + y**2"},
            {"kind": "rational_identity", "variables": ["x"], "left": "x/3 + x/6", "right": "x/2"}]}}]}
    bundle = compile_theory(spec)
    by_id = {o["id"]: o for o in bundle["obligations"]}
    assert by_id["pair"]["status"] == "machine_checked"
    assert by_id["pair"]["checker"] == "conjunction_split/v1"
    assert by_id["pair"]["evidence"]["auto_split"] is True
    assert by_id["pair_part1"]["auto_split_from"] == "pair"
    assert by_id["pair_part1"]["required"] is True
    assert by_id["pair_part2"]["status"] == "machine_checked"


def test_conjunction_status_combinations():
    def bundle_with(parts):
        spec = {"schema_version": 1, "definitions": {}, "obligations": [{
            "id": "pair", "required": True, "depends_on": [], "assumptions": [],
            "statement": {"kind": "conjunction", "parts": parts}}]}
        return {o["id"]: o for o in compile_theory(spec)["obligations"]}

    by_id = bundle_with([
        {"kind": "polynomial_identity", "variables": ["x"], "left": "x", "right": "x"},
        {"kind": "polynomial_identity", "variables": ["x", "y"], "left": "(x+y)**2", "right": "x**2 + y**2"},
        {"kind": "polynomial_identity", "variables": ["x"], "left": "x", "right": "x"}])
    assert by_id["pair"]["status"] == "disproved"
    assert by_id["pair_part2"]["status"] == "disproved"
    assert by_id["pair_part1"]["status"] == "machine_checked"

    by_id = bundle_with([
        {"kind": "polynomial_identity", "variables": ["x"], "left": "x", "right": "x"},
        {"kind": "informal", "text": "An unreviewed part"}])
    assert by_id["pair"]["status"] == "unresolved"
    assert by_id["pair_part2"]["status"] == "unresolved"

    by_id = bundle_with([
        {"kind": "polynomial_identity", "variables": ["x"], "left": "x", "right": "x"},
        {"kind": "informal", "text": "A reviewed part", "proof_text": "Proof",
         "review": {"checker": "fixture", "verdict": "accepted", "evidence": "rationale"}}])
    assert by_id["pair"]["status"] == "reviewed_informal"


def test_conjunction_dependency_and_id_rules():
    def spec_with(depends=(), second_id="other", parts=None):
        return {"schema_version": 1, "definitions": {}, "obligations": [
            {"id": "general", "required": True, "depends_on": [], "assumptions": [],
             "statement": {"kind": "informal", "text": "Unreviewed prerequisite"}, "proof_text": "Attempt"},
            {"id": "pair", "required": True, "depends_on": list(depends), "assumptions": [],
             "statement": {"kind": "conjunction", "parts": parts or [
                 {"kind": "polynomial_identity", "variables": ["x"], "left": "x", "right": "x"}]}},
            {"id": second_id, "required": False, "depends_on": [], "assumptions": [],
             "statement": {"kind": "informal", "text": "Another obligation"}}]}

    bundle = {o["id"]: o for o in compile_theory(spec_with(depends=("general",)))["obligations"]}
    assert bundle["pair"]["status"] == "unresolved"
    assert bundle["pair_part1"]["status"] == "unresolved"
    assert bundle["pair_part1"]["evidence"]["unresolved_dependencies"] == ["general"]

    with pytest.raises(WorkbenchError, match="collides"):
        compile_theory(spec_with(second_id="pair_part1"))
    with pytest.raises(WorkbenchError, match="assumptions"):
        bad = spec_with()
        bad["obligations"][1]["assumptions"] = ["x is positive"]
        compile_theory(bad)
    nested = [{"kind": "conjunction", "parts": [{"kind": "informal", "text": "nested"}]}]
    with pytest.raises(WorkbenchError, match="Unsupported statement kind"):
        compile_theory(spec_with(parts=nested))
