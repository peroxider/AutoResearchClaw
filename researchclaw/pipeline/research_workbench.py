"""Typed method and proof obligations with deliberately bounded checkers.

Code-symbol existence is not semantic equivalence. Exact polynomial identities
can be machine checked; arbitrary prose theorems cannot acquire that status.
No supplied expression or experiment module is evaluated or imported.
"""
from __future__ import annotations

import ast
import itertools
import json
import math
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


DTYPE_WHITELIST = ("float16", "bfloat16", "float32", "float64",
                   "int8", "int16", "int32", "int64", "bool")
_SHAPE_OP_FIELDS = {
    "add": {"op", "left", "right", "result"},
    "elementwise_mul": {"op", "left", "right", "result"},
    "matmul": {"op", "left", "right", "result"},
    "broadcast_add": {"op", "left", "right", "result"},
    "concat": {"op", "left", "right", "result", "axis"},
    "transpose": {"op", "left", "result"},
    "reshape": {"op", "left", "result"},
    "reduce": {"op", "left", "result", "axes"},
    "cast": {"op", "left", "result", "dtype"},
}


def _consistent_dtypes(variables, names, op):
    """Declared dtypes must agree; undeclared dtypes impose no constraint."""
    declared = {name: variables[name]["dtype"] for name in names if "dtype" in variables[name]}
    if len(set(declared.values())) > 1:
        raise WorkbenchError("Dtype mismatch for shape check " + op + ": "
                             + ", ".join(sorted(f"{name}={dtype}" for name, dtype in declared.items())))


def _check_shape_op(check, variables):
    """Bounded symbolic shape checker: dims compare by syntactic equality only.

    broadcast_add verifies an explicitly declared result against broadcast rules;
    it never invents one. reshape/concat need concrete integer dimensions where
    arithmetic (element counts, sums) would otherwise require guessing symbols.
    """
    fields = _SHAPE_OP_FIELDS.get(check.get("op") if isinstance(check, dict) else None)
    if fields is None:
        raise WorkbenchError("Unsupported operation or inconsistent tensor shapes")
    _object(check, fields, fields, "shape check")
    names = sorted(fields & {"left", "right", "result"})
    for key in names:
        _refs([check[key]], variables, f"shape {key}")
    shapes = {key: variables[check[key]]["shape"] for key in names}
    op = check["op"]
    if op in {"add", "elementwise_mul"}:
        if not shapes["left"] == shapes["right"] == shapes["result"]:
            raise WorkbenchError("Unsupported operation or inconsistent tensor shapes")
    elif op == "matmul":
        left, right, out = shapes["left"], shapes["right"], shapes["result"]
        if not (len(left) == len(right) == len(out) == 2 and left[1] == right[0]
                and out == [left[0], right[1]]):
            raise WorkbenchError("Unsupported operation or inconsistent tensor shapes")
    elif op == "broadcast_add":
        left, right, out = shapes["left"], shapes["right"], shapes["result"]
        if len(left) != len(right) or len(left) != len(out):
            raise WorkbenchError("Unsupported operation or inconsistent tensor shapes")
        expanded = []
        for left_dim, right_dim in zip(left, right):
            if left_dim == right_dim or right_dim == 1:
                expanded.append(left_dim)
            elif left_dim == 1:
                expanded.append(right_dim)
            else:
                raise WorkbenchError("Unsupported operation or inconsistent tensor shapes")
        if expanded != out:
            raise WorkbenchError("Unsupported operation or inconsistent tensor shapes")
    elif op == "concat":
        left, right, out = shapes["left"], shapes["right"], shapes["result"]
        axis = check["axis"]
        if (type(axis) is not int or len({len(left), len(right), len(out)}) != 1
                or not 0 <= axis < len(left)
                or any(left[i] != right[i] or left[i] != out[i] for i in range(len(left)) if i != axis)
                or not all(type(dim) is int for dim in (left[axis], right[axis], out[axis]))
                or out[axis] != left[axis] + right[axis]):
            raise WorkbenchError("Concat needs a valid axis and concrete integer dimensions on it")
    elif op == "transpose":
        if len(shapes["left"]) != 2 or shapes["result"] != [shapes["left"][1], shapes["left"][0]]:
            raise WorkbenchError("Transpose needs a two-dimensional input and its swapped shape")
    elif op == "reshape":
        left, out = shapes["left"], shapes["result"]
        if not all(type(dim) is int for dim in left + out) or math.prod(left) != math.prod(out):
            raise WorkbenchError("Reshape needs concrete integer shapes with equal element counts")
    elif op == "reduce":
        axes, left, out = check["axes"], shapes["left"], shapes["result"]
        if (not isinstance(axes, list) or not axes or any(type(a) is not int for a in axes)
                or len(set(axes)) != len(axes) or any(not 0 <= a < len(left) for a in axes)
                or out != [dim for i, dim in enumerate(left) if i not in axes]):
            raise WorkbenchError("Reduce needs valid unique axes and the shape without them")
    else:  # cast
        source, target = variables[check["left"]].get("dtype"), variables[check["result"]].get("dtype")
        if (check["dtype"] not in DTYPE_WHITELIST or source is None or target is None
                or target != check["dtype"] or shapes["left"] != shapes["result"]):
            raise WorkbenchError("Cast needs equal shapes, declared dtypes, and the result dtype equal to the cast target")
        return
    _consistent_dtypes(variables, [check[key] for key in names], op)


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
            _object(variable, {"shape", "description", "dtype"}, {"shape", "description"}, "variable")
            _text(variable["description"], "variable meaning")
            if "dtype" in variable and variable["dtype"] not in DTYPE_WHITELIST:
                raise WorkbenchError(f"Unknown variable dtype: {variable['dtype']}")
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
            _check_shape_op(check, variables)
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


def _poly_tools(variables: list[str]):
    """Bounded rational polynomial algebra shared by the identity checkers."""
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
    return zero, compact, add, multiply


def _evaluate_poly(poly, values):
    total = Fraction(0)
    for powers, coefficient in poly.items():
        value = coefficient
        for x, power in zip(values, powers):
            value *= x ** power
        total += value
    return total


def _int_exponent(node):
    """Signed integer exponent from a Constant or a negated Constant literal."""
    sign = 1
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        sign, node = -1, node.operand
    if isinstance(node, ast.Constant) and type(node.value) is int:
        return sign * node.value
    return None


def _polynomial(expression, variables):
    """Bounded rational polynomial algebra, with no eval/sympify/imports."""
    if not isinstance(expression, str) or len(expression) > 2000:
        raise WorkbenchError("Polynomial expression is too long")
    zero, compact, add, multiply = _poly_tools(variables)
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
                exponent = _int_exponent(node.right)
                if exponent is None or not 0 <= exponent <= 12:
                    raise WorkbenchError("Only small nonnegative integer powers are supported")
                result = {zero: Fraction(1)}
                for _ in range(exponent):
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


def _rational(expression, variables):
    """Bounded rational function as a (numerator, denominator) polynomial pair."""
    if not isinstance(expression, str) or len(expression) > 2000:
        raise WorkbenchError("Rational expression is too long")
    zero, compact, add, multiply = _poly_tools(variables)
    def visit(node):
        if isinstance(node, ast.Constant) and type(node.value) is int and abs(node.value).bit_length() <= 256:
            return compact({zero: Fraction(node.value)}), compact({zero: Fraction(1)})
        if isinstance(node, ast.Name) and node.id in variables:
            return {tuple(int(v == node.id) for v in variables): Fraction(1)}, compact({zero: Fraction(1)})
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            num, den = visit(node.operand)
            return (num, den) if isinstance(node.op, ast.UAdd) else ({k: -v for k, v in num.items()}, den)
        if isinstance(node, ast.BinOp):
            lnum, lden = visit(node.left)
            if isinstance(node.op, ast.Pow):
                exponent = _int_exponent(node.right)
                if exponent is None or not -12 <= exponent <= 12:
                    raise WorkbenchError("Only small integer powers are supported")
                num, den = compact({zero: Fraction(1)}), compact({zero: Fraction(1)})
                for _ in range(abs(exponent)):
                    if exponent >= 0:
                        num, den = multiply(num, lnum), multiply(den, lden)
                    else:
                        num, den = multiply(num, lden), multiply(den, lnum)
                return num, den
            rnum, rden = visit(node.right)
            if isinstance(node.op, ast.Add):
                return add(multiply(lnum, rden), multiply(rnum, lden)), multiply(lden, rden)
            if isinstance(node.op, ast.Sub):
                return add(multiply(lnum, rden), multiply(rnum, lden), -1), multiply(lden, rden)
            if isinstance(node.op, ast.Mult):
                return multiply(lnum, rnum), multiply(lden, rden)
            if isinstance(node.op, ast.Div):
                return multiply(lnum, rden), multiply(lden, rnum)
        raise WorkbenchError("Expression is outside rational function grammar")
    try:
        tree = ast.parse(expression, mode="eval")
        if sum(1 for _ in ast.walk(tree)) > 500:
            raise WorkbenchError("Expression exceeds checker node limit")
        num, den = visit(tree.body)
    except (SyntaxError, RecursionError, ZeroDivisionError) as exc:
        raise WorkbenchError("Invalid rational function expression") from exc
    if not den:
        raise WorkbenchError("Rational function denominator must not be the zero polynomial")
    return num, den


SYMBOLIC_FUNCTIONS = ("sin", "cos", "tan", "exp", "log", "sqrt")


def _symbolic_expression(expression, variables, backend):
    """Whitelist AST grammar for symbolic expressions; sympy construction only
    when the backend is importable. Grammar is validated either way so that
    acceptance does not depend on which optional packages happen to be installed."""
    if not isinstance(expression, str) or len(expression) > 600:
        raise WorkbenchError("Symbolic expression is too long")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise WorkbenchError("Invalid symbolic expression") from exc
    if sum(1 for _ in ast.walk(tree)) > 200:
        raise WorkbenchError("Symbolic expression exceeds node limit")
    if sum(1 for node in ast.walk(tree) if isinstance(node, ast.Call)) > 8:
        raise WorkbenchError("Symbolic expression exceeds the function call budget")
    def visit(node):
        if isinstance(node, ast.Constant):
            if type(node.value) is int and abs(node.value).bit_length() <= 128:
                return backend.Integer(node.value) if backend else None
            if type(node.value) is float and math.isfinite(node.value):
                return backend.Rational(str(node.value)) if backend else None
            raise WorkbenchError("Symbolic constants must be finite integers or decimal literals")
        if isinstance(node, ast.Name) and node.id in variables:
            return backend.Symbol(node.id, real=True) if backend else None
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = visit(node.operand)
            if backend is None:
                return None
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp):
            if isinstance(node.op, ast.Pow):
                exponent = _int_exponent(node.right)
                if exponent is None or not -12 <= exponent <= 12:
                    raise WorkbenchError("Only small integer symbolic powers are supported")
                base = visit(node.left)
                return None if backend is None else base ** exponent
            left, right = visit(node.left), visit(node.right)
            if backend is None:
                return None
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
        if isinstance(node, ast.Call):
            if (not isinstance(node.func, ast.Name) or node.func.id not in SYMBOLIC_FUNCTIONS
                    or node.keywords or len(node.args) != 1):
                raise WorkbenchError("Symbolic calls must be single-argument supported functions")
            argument = visit(node.args[0])
            return None if backend is None else getattr(backend, node.func.id)(argument)
        raise WorkbenchError("Expression is outside the symbolic grammar")
    return visit(tree.body)


def _symbolic_check(statement):
    _object(statement, {"kind", "variables", "left", "right"}, {"kind", "variables", "left", "right"}, "symbolic equality")
    variables = statement["variables"]
    if not isinstance(variables, list) or not 1 <= len(variables) <= 8:
        raise WorkbenchError("Symbolic equality requires 1-8 declared variables")
    for variable in variables:
        _identifier(variable)
        if not variable.isidentifier():
            raise WorkbenchError("Symbolic variables must be Python-style identifiers")
    if len(set(variables)) != len(variables):
        raise WorkbenchError("Duplicate symbolic variable")
    # Grammar is validated even without the backend so unavailable checkers stay honest.
    _symbolic_expression(statement["left"], variables, None)
    _symbolic_expression(statement["right"], variables, None)
    checker = "sympy_symbolic_equality/v1"
    evidence = {"statement_hash": content_hash(statement), "backend": "sympy"}
    try:
        import sympy
    except ImportError:
        evidence["backend"] = "unavailable"
        return "unresolved", checker, evidence
    evidence["sympy_version"] = sympy.__version__
    try:
        left = _symbolic_expression(statement["left"], variables, sympy)
        right = _symbolic_expression(statement["right"], variables, sympy)
        difference = sympy.simplify(left - right)
    except (RecursionError, MemoryError, ZeroDivisionError, TypeError, ValueError):
        evidence["decided"] = False
        return "unresolved", checker, evidence
    if difference.has(sympy.zoo, sympy.nan, sympy.oo):
        evidence["decided"] = False
        evidence["undefined_difference"] = True
        return "unresolved", checker, evidence
    if difference == 0:
        evidence.update({"exact_arithmetic": True, "simplified_to_zero": True})
        return "machine_checked", checker, evidence
    if not getattr(difference, "free_symbols", {None}):
        evidence["difference_value"] = str(difference)[:200]
        return "disproved", checker, evidence
    evidence["decided"] = False
    return "unresolved", checker, evidence


def _leaf_check(statement, *, assumptions=(), proof_text=None, review=None):
    """Check one typed leaf statement and return status, checker and evidence."""
    if not isinstance(statement, dict):
        raise WorkbenchError("Proof statement must be typed")
    status, checker, evidence = "unresolved", "", {}
    kind = statement.get("kind")
    if kind == "polynomial_identity":
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
                lvalue, rvalue = _evaluate_poly(left, values), _evaluate_poly(right, values)
                if lvalue != rvalue:
                    evidence["counterexample"] = {"values": dict(zip(variables, values)), "left": str(lvalue), "right": str(rvalue)}
                    break
        if assumptions:
            # This checker proves unconditional polynomial identities only.
            raise WorkbenchError("Polynomial identity checker does not interpret prose assumptions")
        if review is not None or proof_text is not None:
            raise WorkbenchError("Machine identities use structured statements, not prose proof overrides")
    elif kind == "rational_identity":
        _object(statement, {"kind", "variables", "left", "right"}, {"kind", "variables", "left", "right"}, "rational identity")
        variables = statement["variables"]
        if not isinstance(variables, list) or not 1 <= len(variables) <= 8:
            raise WorkbenchError("Rational identity requires 1-8 declared variables over the rationals/reals")
        for variable in variables:
            _identifier(variable)
            if not variable.isidentifier():
                raise WorkbenchError("Rational variables must be Python-style identifiers")
        if len(set(variables)) != len(variables):
            raise WorkbenchError("Duplicate rational variable")
        lnum, lden = _rational(statement["left"], variables)
        rnum, rden = _rational(statement["right"], variables)
        _, _, _, multiply = _poly_tools(variables)
        holds = multiply(lnum, rden) == multiply(rnum, lden)
        status = "machine_checked" if holds else "disproved"
        checker = "rational_function_identity/v1"
        evidence = {"domain": "all substitutions where both denominators are nonzero", "exact_arithmetic": True,
                    "statement_hash": content_hash(statement), "identity_holds": holds}
        if not holds:
            # Unequal cross products already refute identity; a witness avoiding
            # denominator zeros is supplemental and may not exist on the small grid.
            _, _, add, multiply = _poly_tools(variables)
            difference = add(multiply(lnum, rden), multiply(rnum, lden), -1)
            evidence["difference_numerator_is_nonzero_polynomial"] = bool(difference)
            for values in itertools.islice(itertools.product((-2, -1, 0, 1, 2), repeat=len(variables)), 625):
                if _evaluate_poly(lden, values) == 0 or _evaluate_poly(rden, values) == 0:
                    continue
                lvalue = _evaluate_poly(lnum, values) / _evaluate_poly(lden, values)
                rvalue = _evaluate_poly(rnum, values) / _evaluate_poly(rden, values)
                if lvalue != rvalue:
                    evidence["counterexample"] = {"values": dict(zip(variables, values)), "left": str(lvalue), "right": str(rvalue)}
                    break
        if assumptions:
            raise WorkbenchError("Rational identity checker does not interpret prose assumptions")
        if review is not None or proof_text is not None:
            raise WorkbenchError("Machine identities use structured statements, not prose proof overrides")
    elif kind == "symbolic_equality":
        status, checker, evidence = _symbolic_check(statement)
        if assumptions:
            raise WorkbenchError("Symbolic equality checker does not interpret prose assumptions")
        if review is not None or proof_text is not None:
            raise WorkbenchError("Machine identities use structured statements, not prose proof overrides")
    elif kind == "linear_arithmetic":
        from researchclaw.pipeline.formal_proof import check_linear_arithmetic, FormalProofError
        try:
            status, checker, evidence = check_linear_arithmetic(statement)
        except FormalProofError as exc:
            raise WorkbenchError(str(exc)) from exc
        if assumptions:
            raise WorkbenchError("Linear arithmetic uses structured premises, not prose assumptions")
        if review is not None or proof_text is not None:
            raise WorkbenchError("Machine implications use structured statements, not prose proof overrides")
    elif kind == "informal":
        _object(statement, {"kind", "text"}, {"kind", "text"}, "informal statement")
        _text(statement["text"], "theorem statement")
        if proof_text is not None:
            _text(proof_text, "proof text")
        if review is not None:
            review = _object(review, {"checker", "verdict", "evidence"}, {"checker", "verdict", "evidence"}, "informal proof review")
            _text(review["checker"], "proof reviewer")
            _text(review["evidence"], "review reasoning")
            if review["verdict"] not in {"accepted", "rejected", "unresolved"}:
                raise WorkbenchError("Invalid informal proof review verdict")
            if not proof_text:
                raise WorkbenchError("A review cannot stand in for a missing proof")
            status = {"accepted": "reviewed_informal", "rejected": "disproved", "unresolved": "unresolved"}[review["verdict"]]
            checker, evidence = review["checker"], {"review": review, "proof_hash": content_hash(proof_text)}
    else:
        raise WorkbenchError("Unsupported statement kind; use informal for unresolved general proofs")
    return status, checker, evidence


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
            if isinstance(statement, dict) and statement.get("kind") == "informal_decomposition":
                _object(statement, {"kind", "text", "parts", "coverage_review"},
                        {"kind", "text", "parts"}, "informal decomposition")
                _text(statement["text"], "decomposed theorem statement")
                parts = statement["parts"]
                if not isinstance(parts, list) or not 2 <= len(parts) <= 16:
                    raise WorkbenchError("Informal decomposition needs 2 to 16 typed parts")
                if item.get("proof_text") is not None or item.get("review") is not None:
                    raise WorkbenchError("Informal decomposition uses part proofs and a coverage review")
                part_specs = {}
                for part in parts:
                    _object(part, {"id", "depends_on", "assumptions", "statement", "proof_text", "review"},
                            {"id", "depends_on", "assumptions", "statement"}, "decomposition part")
                    part_id = _identifier(part["id"])
                    if part_id in part_specs:
                        raise WorkbenchError("Duplicate decomposition part ID")
                    if (not isinstance(part["assumptions"], list)
                            or any(not isinstance(value, str) or not value.strip() for value in part["assumptions"])):
                        raise WorkbenchError("Decomposition part assumptions must be explicit nonempty statements")
                    part_specs[part_id] = part
                for part in parts:
                    _refs(part["depends_on"], part_specs, "decomposition part dependencies")
                part_order = _dag({part_id: part_specs[part_id]["depends_on"] for part_id in part_specs})
                coverage = statement.get("coverage_review")
                if coverage is not None:
                    coverage = _object(coverage, {"checker", "verdict", "evidence"},
                                       {"checker", "verdict", "evidence"}, "decomposition coverage review")
                    _text(coverage["checker"], "decomposition reviewer")
                    _text(coverage["evidence"], "decomposition review reasoning")
                    if coverage["verdict"] not in {"accepted", "rejected", "unresolved"}:
                        raise WorkbenchError("Invalid decomposition coverage verdict")
                external_unmet = [d for d in item["depends_on"]
                                  if checked[d]["status"] not in {"machine_checked", "reviewed_informal"}]
                generated, statuses = {}, []
                for part_id in part_order:
                    part = part_specs[part_id]
                    generated_id = f"{oid}__{part_id}"
                    if generated_id in obligations or generated_id in checked:
                        raise WorkbenchError("Generated decomposition part ID collides with an existing obligation")
                    part_status, part_checker, part_evidence = _leaf_check(
                        part["statement"], assumptions=part["assumptions"],
                        proof_text=part.get("proof_text"), review=part.get("review"))
                    internal_dependencies = [generated[dependency] for dependency in part["depends_on"]]
                    dependencies = [*item["depends_on"], *internal_dependencies]
                    unmet = [*external_unmet, *(dependency for dependency in internal_dependencies
                                                if checked[dependency]["status"] not in
                                                {"machine_checked", "reviewed_informal"})]
                    informal = [dependency for dependency in dependencies
                                if checked[dependency]["status"] == "reviewed_informal"]
                    if unmet and part_status != "disproved":
                        part_status = "unresolved"
                        part_evidence["unresolved_dependencies"] = unmet
                    elif part_status == "machine_checked" and informal:
                        part_status = "reviewed_informal"
                        part_evidence["informal_dependencies"] = informal
                    checked[generated_id] = {
                        "id": generated_id, "required": item["required"], "statement": part["statement"],
                        "assumptions": part["assumptions"], "depends_on": dependencies,
                        "status": part_status, "checker": part_checker, "evidence": part_evidence,
                        "auto_split_from": oid, "decomposition_part": part_id,
                    }
                    generated[part_id] = generated_id
                    statuses.append(part_status)
                coverage_accepted = coverage is not None and coverage["verdict"] == "accepted"
                parent_status = ("reviewed_informal" if coverage_accepted and not external_unmet
                                 and all(status in {"machine_checked", "reviewed_informal"} for status in statuses)
                                 else "unresolved")
                parent_evidence = {
                    "assisted_decomposition": True,
                    "coverage_review": coverage,
                    "coverage_review_required": True,
                    "machine_status_cap": "reviewed_informal",
                    "parts": {generated[part_id]: checked[generated[part_id]]["status"] for part_id in part_order},
                    "statement_hash": content_hash(statement),
                }
                if external_unmet:
                    parent_evidence["unresolved_dependencies"] = external_unmet
                failed = [generated[part_id] for part_id in part_order
                          if checked[generated[part_id]]["status"] not in {"machine_checked", "reviewed_informal"}]
                if failed:
                    parent_evidence["incomplete_parts"] = failed
                checked[oid] = {
                    "id": oid, "required": item["required"], "statement": statement,
                    "assumptions": item["assumptions"],
                    "depends_on": [*item["depends_on"], *(generated[part_id] for part_id in part_order)],
                    "status": parent_status, "checker": "informal_decomposition_review/v1",
                    "evidence": parent_evidence,
                }
                continue
            if isinstance(statement, dict) and statement.get("kind") == "conjunction":
                _object(statement, {"kind", "parts"}, {"kind", "parts"}, "conjunction statement")
                parts = statement["parts"]
                if not isinstance(parts, list) or not 1 <= len(parts) <= 8:
                    raise WorkbenchError("Conjunction needs 1 to 8 typed parts")
                if item["assumptions"]:
                    raise WorkbenchError("Auto-split conjunctions declare no top-level assumptions")
                if item.get("proof_text") is not None or item.get("review") is not None:
                    raise WorkbenchError("Conjunction obligations use part-level proofs, not prose overrides")
                unmet = [d for d in item["depends_on"] if checked[d]["status"] not in {"machine_checked", "reviewed_informal"}]
                informal_deps = [d for d in item["depends_on"] if checked[d]["status"] == "reviewed_informal"]
                statuses = []
                for index, part in enumerate(parts, start=1):
                    part_id = f"{oid}_part{index}"
                    if not isinstance(part, dict):
                        raise WorkbenchError("Conjunction parts must be typed statements")
                    if part_id in obligations or part_id in checked:
                        raise WorkbenchError("Generated conjunction part ID collides with an existing obligation")
                    part_statement = {k: v for k, v in part.items() if k not in {"proof_text", "review"}}
                    part_status, part_checker, part_evidence = _leaf_check(
                        part_statement, proof_text=part.get("proof_text"), review=part.get("review"))
                    if unmet and part_status != "disproved":
                        part_status = "unresolved"
                        part_evidence["unresolved_dependencies"] = unmet
                    elif part_status == "machine_checked" and informal_deps:
                        part_status = "reviewed_informal"
                        part_evidence["informal_dependencies"] = informal_deps
                    checked[part_id] = {"id": part_id, "required": item["required"], "statement": part_statement,
                                        "assumptions": [], "depends_on": item["depends_on"],
                                        "status": part_status, "checker": part_checker, "evidence": part_evidence,
                                        "auto_split_from": oid}
                    statuses.append(part_status)
                if any(s == "disproved" for s in statuses):
                    parent_status = "disproved"
                elif all(s == "machine_checked" for s in statuses) and not informal_deps:
                    parent_status = "machine_checked"
                elif all(s in {"machine_checked", "reviewed_informal"} for s in statuses):
                    parent_status = "reviewed_informal"
                else:
                    parent_status = "unresolved"
                parent_evidence = {"auto_split": True, "parts": {f"{oid}_part{i}": s for i, s in enumerate(statuses, start=1)},
                                   "statement_hash": content_hash(statement)}
                if unmet and parent_status != "disproved":
                    parent_status = "unresolved"
                    parent_evidence["unresolved_dependencies"] = unmet
                checked[oid] = {"id": oid, "required": item["required"], "statement": statement,
                                "assumptions": item["assumptions"], "depends_on": item["depends_on"],
                                "status": parent_status, "checker": "conjunction_split/v1", "evidence": parent_evidence}
                continue
            status, checker, evidence = _leaf_check(statement, assumptions=item["assumptions"],
                                                    proof_text=item.get("proof_text"), review=item.get("review"))
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
        result = {"schema_version": 2, "spec": spec, "obligations": list(checked.values()),
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
