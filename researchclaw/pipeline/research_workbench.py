"""Typed method and proof obligations with deliberately bounded checkers.

Code-symbol existence is not semantic equivalence. Exact polynomial identities
can be machine checked; arbitrary prose theorems cannot acquire that status.
No supplied expression or experiment module is evaluated or imported.
"""
from __future__ import annotations

import ast
import itertools
import json
import re
from fractions import Fraction
from pathlib import Path

from researchclaw.pipeline.evidence_store import content_hash, file_hash
from researchclaw.pipeline.experiment_protocol import ProtocolError, _object, _text, _identifier


class WorkbenchError(ValueError):
    pass


def _refs(value, allowed, label, *, nonempty=False):
    if (not isinstance(value, list) or (nonempty and not value)
            or any(not isinstance(v, str) for v in value) or len(set(value)) != len(value)
            or set(value) - set(allowed)):
        raise WorkbenchError(f"Invalid {label} references")
    return value


def _dag(nodes: dict[str, list[str]]) -> list[str]:
    pending, order = set(nodes), []
    if any(set(deps) - nodes.keys() for deps in nodes.values()):
        raise WorkbenchError("Unknown dependency")
    while pending:
        ready = sorted(n for n in pending if not set(nodes[n]) & pending)
        if not ready:
            raise WorkbenchError("Cyclic dependency")
        order.extend(ready)
        pending.difference_update(ready)
    return order


def compile_method(spec: dict) -> dict:
    try:
        _object(spec, {"schema_version", "method_id", "description", "variables", "equations", "steps",
                       "losses", "stopping_rule", "complexity", "shape_checks", "control_flow", "validation"},
                {"schema_version", "method_id", "description", "variables", "equations", "steps",
                 "losses", "stopping_rule", "complexity"}, "MethodSpec")
        if type(spec["schema_version"]) is not int or spec["schema_version"] != 1:
            raise WorkbenchError("Unsupported MethodSpec schema")
        _identifier(spec["method_id"])
        _text(spec["description"], "method description")
        _text(spec["stopping_rule"], "stopping rule")
        _object(spec["complexity"], {"time", "space"}, {"time", "space"}, "complexity")
        for value in spec["complexity"].values():
            _text(value, "complexity bound")
        variables = spec["variables"]
        if not isinstance(variables, dict) or not variables:
            raise WorkbenchError("MethodSpec needs typed variables")
        for name, variable in variables.items():
            _identifier(name)
            _object(variable, {"shape", "description"}, {"shape", "description"}, "variable")
            _text(variable["description"], "variable meaning")
            if not isinstance(variable["shape"], list) or any(
                not ((type(d) is int and d > 0) or (isinstance(d, str) and re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", d)))
                for d in variable["shape"]
            ):
                raise WorkbenchError("Shapes must contain positive dimensions or named dimensions")
        equations, steps = {}, {}
        if not isinstance(spec["equations"], list) or not isinstance(spec["steps"], list) or not spec["steps"]:
            raise WorkbenchError("Equations and nonempty steps must be lists")
        for equation in spec["equations"]:
            _object(equation, {"id", "latex", "inputs", "outputs"}, {"id", "latex", "inputs", "outputs"}, "equation")
            eid = _identifier(equation["id"])
            if eid in equations:
                raise WorkbenchError("Duplicate equation ID")
            _text(equation["latex"], "equation latex")
            _refs(equation["inputs"], variables, "equation inputs")
            _refs(equation["outputs"], variables, "equation outputs", nonempty=True)
            equations[eid] = equation
        for step in spec["steps"]:
            _object(step, {"id", "phase", "description", "inputs", "outputs", "equations", "depends_on", "code_file", "code_symbol"},
                    {"id", "phase", "description", "inputs", "outputs", "equations", "depends_on", "code_file", "code_symbol"}, "method step")
            sid = _identifier(step["id"])
            if sid in steps:
                raise WorkbenchError("Duplicate step ID")
            if step["phase"] not in {"train", "inference"}:
                raise WorkbenchError("Each step must distinguish train and inference")
            _text(step["description"], "pseudocode step")
            _refs(step["inputs"], variables, "step inputs")
            _refs(step["outputs"], variables, "step outputs", nonempty=True)
            _refs(step["equations"], equations, "step equations")
            filename = _text(step["code_file"], "code file")
            if Path(filename).is_absolute() or ".." in Path(filename).parts or Path(filename).suffix != ".py" or "\\" in filename:
                raise WorkbenchError("Code paths must be relative Python files")
            if not re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*", _text(step["code_symbol"], "code symbol")):
                raise WorkbenchError("Code symbols must be qualified Python names")
            steps[sid] = step
        for step in steps.values():
            _refs(step["depends_on"], steps, "step dependencies")
        order = _dag({sid: s["depends_on"] for sid, s in steps.items()})
        if "validation" in spec:
            from researchclaw.pipeline.method_validation import validate_plan
            validate_plan(spec["validation"], spec)
        if "control_flow" in spec:
            from researchclaw.pipeline.diagram_spec import validate_control_flow, DiagramError
            try:
                validate_control_flow(spec["control_flow"], steps)
            except DiagramError as exc:
                raise WorkbenchError(str(exc)) from exc
        _refs(spec["losses"], equations, "loss equations")
        unused = equations.keys() - {e for s in steps.values() for e in s["equations"]}
        if unused:
            raise WorkbenchError("Equations must map to at least one method step")
        shape_checks = spec.get("shape_checks", [])
        if not isinstance(shape_checks, list):
            raise WorkbenchError("shape_checks must be a list")
        for check in shape_checks:
            _object(check, {"op", "left", "right", "result"}, {"op", "left", "right", "result"}, "shape check")
            _refs([check["left"]], variables, "shape left")
            _refs([check["right"]], variables, "shape right")
            _refs([check["result"]], variables, "shape result")
            left, right, result = (variables[check[k]]["shape"] for k in ("left", "right", "result"))
            valid = (left == right == result if check["op"] == "add" else
                     len(left) == len(right) == len(result) == 2 and left[1] == right[0]
                     and result == [left[0], right[1]] if check["op"] == "matmul" else False)
            if not valid:
                raise WorkbenchError("Unsupported operation or inconsistent tensor shapes")
        document = {"schema_version": 1, "spec": spec, "step_order": order,
                    "structure_status": "verified", "semantic_equivalence": "unresolved"}
        document = json.loads(json.dumps(document, allow_nan=False))
        document["version"] = content_hash(document)
        return document
    except (TypeError, KeyError, ProtocolError) as exc:
        raise WorkbenchError(f"Invalid MethodSpec: {exc}") from exc


def check_implementation(method: dict, project: Path) -> dict:
    issues, bindings = [], []
    for step in method["spec"]["steps"]:
        path = (project / step["code_file"]).resolve()
        if not path.is_relative_to(project.resolve()) or not path.is_file():
            issues.append({"step": step["id"], "reason": "missing_code_file"})
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
            scope = tree.body
            symbol = None
            for part in step["code_symbol"].split("."):
                symbol = next((n for n in scope if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                               and n.name == part), None)
                if symbol is None:
                    break
                scope = symbol.body
            if symbol is None:
                issues.append({"step": step["id"], "reason": "missing_code_symbol"})
            else:
                bindings.append({"step": step["id"], "file": step["code_file"],
                                 "symbol": step["code_symbol"], "line": symbol.lineno, "sha256": file_hash(path)})
        except (OSError, SyntaxError, UnicodeError):
            issues.append({"step": step["id"], "reason": "unreadable_or_invalid_python"})
    return {"checker": "method_symbol_map/v1", "method_version": method["version"],
            "status": "mapped" if not issues else "failed", "bindings": bindings, "issues": issues,
            "semantic_equivalence": "unresolved"}


def _polynomial(expression: str, variables: list[str]) -> dict[tuple[int, ...], Fraction]:
    """Bounded rational polynomial algebra, with no eval/sympify/imports."""
    if not isinstance(expression, str) or len(expression) > 2000:
        raise WorkbenchError("Polynomial expression is too long")
    zero = (0,) * len(variables)
    def compact(poly):
        poly = {k: v for k, v in poly.items() if v}
        if len(poly) > 4096 or any(sum(k) > 48 or abs(v.numerator).bit_length() > 4096 or v.denominator.bit_length() > 4096 for k, v in poly.items()):
            raise WorkbenchError("Polynomial exceeds checker resource bounds")
        return poly
    def add(left, right, sign=1):
        result = dict(left)
        for term, value in right.items():
            result[term] = result.get(term, Fraction(0)) + sign * value
        return compact(result)
    def multiply(left, right):
        if len(left) * len(right) > 65536:
            raise WorkbenchError("Polynomial product exceeds checker resource bounds")
        result = {}
        for lterm, lvalue in left.items():
            for rterm, rvalue in right.items():
                term = tuple(a + b for a, b in zip(lterm, rterm))
                result[term] = result.get(term, Fraction(0)) + lvalue * rvalue
        return compact(result)
    def visit(node):
        if isinstance(node, ast.Constant) and type(node.value) is int and abs(node.value).bit_length() <= 256:
            return compact({zero: Fraction(node.value)})
        if isinstance(node, ast.Name) and node.id in variables:
            return {tuple(int(v == node.id) for v in variables): Fraction(1)}
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = visit(node.operand)
            return value if isinstance(node.op, ast.UAdd) else {k: -v for k, v in value.items()}
        if isinstance(node, ast.BinOp):
            left = visit(node.left)
            if isinstance(node.op, ast.Pow):
                if not isinstance(node.right, ast.Constant) or type(node.right.value) is not int or not 0 <= node.right.value <= 12:
                    raise WorkbenchError("Only small nonnegative integer powers are supported")
                result = {zero: Fraction(1)}
                for _ in range(node.right.value):
                    result = multiply(result, left)
                return result
            right = visit(node.right)
            if isinstance(node.op, ast.Add):
                return add(left, right)
            if isinstance(node.op, ast.Sub):
                return add(left, right, -1)
            if isinstance(node.op, ast.Mult):
                return multiply(left, right)
            if isinstance(node.op, ast.Div) and set(right) == {zero}:
                return compact({k: v / right[zero] for k, v in left.items()})
        raise WorkbenchError("Expression is outside rational polynomial grammar")
    try:
        tree = ast.parse(expression, mode="eval")
        if sum(1 for _ in ast.walk(tree)) > 500:
            raise WorkbenchError("Expression exceeds checker node limit")
        return visit(tree.body)
    except (SyntaxError, RecursionError, ZeroDivisionError) as exc:
        raise WorkbenchError("Invalid polynomial expression") from exc


def compile_theory(spec: dict) -> dict:
    try:
        _object(spec, {"schema_version", "definitions", "obligations"},
                {"schema_version", "definitions", "obligations"}, "TheoryBundle")
        if type(spec["schema_version"]) is not int or spec["schema_version"] != 1:
            raise WorkbenchError("Unsupported TheoryBundle schema")
        if not isinstance(spec["definitions"], dict) or not isinstance(spec["obligations"], list):
            raise WorkbenchError("Invalid definitions or proof obligations")
        for name, definition in spec["definitions"].items():
            _identifier(name)
            _text(definition, "definition")
        obligations = {}
        for item in spec["obligations"]:
            _object(item, {"id", "required", "depends_on", "assumptions", "statement", "proof_text", "review"},
                    {"id", "required", "depends_on", "assumptions", "statement"}, "proof obligation")
            oid = _identifier(item["id"])
            if oid in obligations or type(item["required"]) is not bool:
                raise WorkbenchError("Duplicate obligation or invalid required flag")
            if not isinstance(item["assumptions"], list) or any(not isinstance(a, str) or not a.strip() for a in item["assumptions"]):
                raise WorkbenchError("Assumptions must be explicit nonempty statements")
            obligations[oid] = item
        for item in obligations.values():
            _refs(item["depends_on"], obligations, "proof dependencies")
        order = _dag({oid: item["depends_on"] for oid, item in obligations.items()})
        checked = {}
        for oid in order:
            item = obligations[oid]
            statement = item["statement"]
            if not isinstance(statement, dict):
                raise WorkbenchError("Proof statement must be typed")
            status, checker, evidence = "unresolved", "", {}
            if statement.get("kind") == "polynomial_identity":
                _object(statement, {"kind", "variables", "left", "right"}, {"kind", "variables", "left", "right"}, "polynomial identity")
                variables = statement["variables"]
                if not isinstance(variables, list) or not 1 <= len(variables) <= 8:
                    raise WorkbenchError("Identity requires 1-8 declared variables over the rationals/reals")
                for variable in variables:
                    _identifier(variable)
                    if not variable.isidentifier():
                        raise WorkbenchError("Polynomial variables must be Python-style identifiers")
                if len(set(variables)) != len(variables):
                    raise WorkbenchError("Duplicate polynomial variable")
                left, right = (_polynomial(statement[key], variables) for key in ("left", "right"))
                status = "machine_checked" if left == right else "disproved"
                checker = "rational_polynomial_identity/v1"
                evidence = {"domain": "all rational or real substitutions", "exact_arithmetic": True,
                            "statement_hash": content_hash(statement), "identity_holds": left == right}
                if left != right:
                    # A witness is supplemental: unequal coefficient maps already refute identity.
                    for values in itertools.islice(itertools.product((-2, -1, 0, 1, 2), repeat=len(variables)), 625):
                        def evaluate(poly):
                            total = Fraction(0)
                            for powers, coefficient in poly.items():
                                value = coefficient
                                for x, power in zip(values, powers):
                                    value *= x ** power
                                total += value
                            return total
                        lvalue, rvalue = evaluate(left), evaluate(right)
                        if lvalue != rvalue:
                            evidence["counterexample"] = {"values": dict(zip(variables, values)), "left": str(lvalue), "right": str(rvalue)}
                            break
                if item["assumptions"]:
                    # This checker proves unconditional polynomial identities only.
                    raise WorkbenchError("Polynomial identity checker does not interpret prose assumptions")
                if "review" in item or "proof_text" in item:
                    raise WorkbenchError("Machine identities use structured statements, not prose proof overrides")
            elif statement.get("kind") == "informal":
                _object(statement, {"kind", "text"}, {"kind", "text"}, "informal statement")
                _text(statement["text"], "theorem statement")
                if "proof_text" in item:
                    _text(item["proof_text"], "proof text")
                if "review" in item:
                    review = _object(item["review"], {"checker", "verdict", "evidence"}, {"checker", "verdict", "evidence"}, "informal proof review")
                    _text(review["checker"], "proof reviewer")
                    _text(review["evidence"], "review reasoning")
                    if review["verdict"] not in {"accepted", "rejected", "unresolved"}:
                        raise WorkbenchError("Invalid informal proof review verdict")
                    if not item.get("proof_text"):
                        raise WorkbenchError("A review cannot stand in for a missing proof")
                    status = {"accepted": "reviewed_informal", "rejected": "disproved", "unresolved": "unresolved"}[review["verdict"]]
                    checker, evidence = review["checker"], {"review": review, "proof_hash": content_hash(item["proof_text"])}
            else:
                raise WorkbenchError("Unsupported statement kind; use informal for unresolved general proofs")
            unmet = [d for d in item["depends_on"] if checked[d]["status"] not in {"machine_checked", "reviewed_informal"}]
            if unmet and status != "disproved":
                status = "unresolved"
                evidence["unresolved_dependencies"] = unmet
            elif status == "machine_checked" and any(checked[d]["status"] == "reviewed_informal" for d in item["depends_on"]):
                status = "reviewed_informal"
                evidence["informal_dependencies"] = [d for d in item["depends_on"] if checked[d]["status"] == "reviewed_informal"]
            checked[oid] = {"id": oid, "required": item["required"], "statement": statement,
                            "assumptions": item["assumptions"], "depends_on": item["depends_on"],
                            "status": status, "checker": checker, "evidence": evidence}
        result = {"schema_version": 2, "spec": spec, "obligations": [checked[o] for o in order],
                  "scope": "typed_obligations_not_general_theorem_proving"}
        result["version"] = content_hash(result)
        return result
    except (TypeError, KeyError, ProtocolError) as exc:
        raise WorkbenchError(f"Invalid TheoryBundle: {exc}") from exc


def verify_workbench(root: Path, contract: dict) -> dict:
    documents = {}
    for field_name, filename, compiler in (
        ("method_spec_path", "method_spec.json", compile_method),
        ("theory_bundle_path", "theory_bundle.json", compile_theory),
    ):
        if not contract["brief"].get(field_name):
            continue
        if filename not in contract["outputs"]:
            raise WorkbenchError(f"Missing frozen {filename}")
        try:
            document = json.loads((root / filename).read_text(encoding="utf-8"))
            if compiler(document["spec"]) != document:
                raise WorkbenchError(f"Changed compiled {filename}")
        except (OSError, ValueError, TypeError, KeyError) as exc:
            raise WorkbenchError(f"Invalid frozen {filename}: {exc}") from exc
        documents[filename] = document
    return documents
