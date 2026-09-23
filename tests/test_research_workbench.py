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
