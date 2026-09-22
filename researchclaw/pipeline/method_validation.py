"""Bounded, predeclared method probes; comparisons run outside experiment code.

Passing pointwise checks never establishes semantic equivalence or scientific
validity. Host subprocess execution is not an operating-system security boundary.
"""
from __future__ import annotations

import copy
import json
import math
import re
from pathlib import Path

from researchclaw.pipeline.evidence_store import content_hash, file_hash
from researchclaw.pipeline.research_workbench import WorkbenchError

MAX_CALLS = 256
MAX_ELEMENTS = 256
LIMITATIONS = [
    "Only the predeclared synthetic instances and perturbations were checked.",
    "Finite differences are local numerical checks, not a derivative proof.",
    "Toggle checks compare outputs; they do not prove component removal or general effectiveness.",
    "Method-wide equivalence, stochastic state isolation and scientific validity remain unresolved.",
    "A host subprocess is not OS isolation; hashes detect changes, not a malicious host or code forgery.",
]


def _fields(value, required, optional=()):
    if not isinstance(value, dict) or set(value) - set(required) - set(optional) or set(required) - value.keys():
        raise WorkbenchError("Invalid method validation fields")


def _number(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def tensor(value, *, boolean=False, depth=0):
    """Return rectangular shape and finite flattened values, with bounded work."""
    if depth > 6:
        raise WorkbenchError("Probe tensor exceeds maximum rank")
    if _number(value) or (boolean and type(value) is bool):
        return [], [value]
    if not isinstance(value, list) or not 1 <= len(value) <= MAX_ELEMENTS:
        raise WorkbenchError("Probe tensors must be finite numeric scalars or nonempty arrays")
    shapes, flattened = [], []
    for item in value:
        shape, values = tensor(item, boolean=boolean, depth=depth + 1)
        shapes.append(shape)
        flattened.extend(values)
        if len(flattened) > MAX_ELEMENTS:
            raise WorkbenchError("Probe tensor exceeds element budget")
    if any(shape != shapes[0] for shape in shapes):
        raise WorkbenchError("Ragged probe tensor")
    return [len(value), *shapes[0]], flattened


def _shape(value, declared, dimensions):
    actual, _ = tensor(value, boolean=True)
    if len(actual) != len(declared):
        raise WorkbenchError("Probe rank differs from MethodSpec")
    for observed, expected in zip(actual, declared):
        if isinstance(expected, str):
            dimensions.setdefault(expected, observed)
            expected = dimensions[expected]
        if expected != observed:
            raise WorkbenchError("Probe dimensions differ from MethodSpec")


def _call(call, spec):
    _fields(call, {"step", "kwargs"}, {"constructor_kwargs"})
    step = next((s for s in spec["steps"] if s["id"] == call["step"]), None)
    if step is None or len(step["code_symbol"].split(".")) > 2 or len(step["outputs"]) != 1:
        raise WorkbenchError("Probe needs a mapped function or Class.method with one output")
    kwargs = call["kwargs"]
    if not isinstance(kwargs, dict) or set(kwargs) != set(step["inputs"]):
        raise WorkbenchError("Probe keyword arguments must match the mapped step inputs")
    dimensions = {}
    for key, value in kwargs.items():
        _shape(value, spec["variables"][key]["shape"], dimensions)
    constructor = call.get("constructor_kwargs", {})
    if (not isinstance(constructor, dict) or len(constructor) > 16
            or any(not isinstance(k, str) or not re.fullmatch(r"[A-Za-z_]\w*", k) for k in constructor)
            or (constructor and "." not in step["code_symbol"])):
        raise WorkbenchError("Invalid probe constructor keyword arguments")
    for value in constructor.values():
        tensor(value, boolean=True)
    return step, dimensions


def validate_plan(plan, spec):
    _fields(plan, {"schema_version", "timeout_seconds", "cases"})
    if type(plan["schema_version"]) is not int or plan["schema_version"] != 1:
        raise WorkbenchError("Unsupported method validation schema")
    if type(plan["timeout_seconds"]) is not int or not 1 <= plan["timeout_seconds"] <= 120:
        raise WorkbenchError("Method validation timeout must be between 1 and 120 seconds")
    if not isinstance(plan["cases"], list) or not 1 <= len(plan["cases"]) <= 32:
        raise WorkbenchError("Method validation needs 1 to 32 cases")
    ids, count = set(), 0
    for case in plan["cases"]:
        common = {"id", "kind", "call", "atol", "rtol"}
        if not isinstance(case, dict):
            raise WorkbenchError("Invalid probe case")
        kind = case.get("kind")
        extra = {"microinstance": {"expected"},
                 "gradient": {"gradient_call", "argument", "epsilon"},
                 "component_toggle": {"parameter", "expected_enabled", "expected_disabled"}}.get(kind)
        if extra is None:
            raise WorkbenchError("Unsupported method validation kind")
        _fields(case, common | extra)
        cid = case["id"]
        if not isinstance(cid, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,63}", cid) or cid in ids:
            raise WorkbenchError("Invalid or duplicate probe ID")
        ids.add(cid)
        for field in ("atol", "rtol"):
            if not _number(case[field]) or not 0 <= case[field] <= 0.01:
                raise WorkbenchError("Probe tolerances must be finite and between zero and 0.01")
        step, dimensions = _call(case["call"], spec)
        output_shape = spec["variables"][step["outputs"][0]]["shape"]
        if kind == "microinstance":
            tensor(case["expected"])
            _shape(case["expected"], output_shape, dimensions)
            count += 1
        elif kind == "component_toggle":
            parameter = case["parameter"]
            if not isinstance(parameter, str) or type(case["call"]["kwargs"].get(parameter)) is not bool:
                raise WorkbenchError("Component toggle requires a declared Boolean input")
            for key in ("expected_enabled", "expected_disabled"):
                tensor(case[key])
                _shape(case[key], output_shape, dimensions)
            count += 2
        else:
            gradient_step, gradient_dimensions = _call(case["gradient_call"], spec)
            arg = case["argument"]
            if not isinstance(arg, str) or arg not in case["call"]["kwargs"] or output_shape:
                raise WorkbenchError("Gradient probe requires a scalar loss and an input argument")
            if (case["call"]["kwargs"] != case["gradient_call"]["kwargs"]
                    or case["call"].get("constructor_kwargs", {}) != case["gradient_call"].get("constructor_kwargs", {})):
                raise WorkbenchError("Loss and gradient must use identical inputs and constructor arguments")
            _, values = tensor(case["call"]["kwargs"][arg])
            _shape(case["call"]["kwargs"][arg],
                   spec["variables"][gradient_step["outputs"][0]]["shape"], gradient_dimensions)
            if len(values) > 64 or not _number(case["epsilon"]) or not 1e-8 <= case["epsilon"] <= 0.01:
                raise WorkbenchError("Gradient probe exceeds coordinate or perturbation bounds")
            for value in values:
                if any(not math.isfinite(value + sign * case["epsilon"]) or value + sign * case["epsilon"] == value for sign in (-1, 1)):
                    raise WorkbenchError("Gradient perturbation is not representable")
            count += 1 + 2 * len(values)
    if count > MAX_CALLS:
        raise WorkbenchError("Method validation exceeds its invocation budget")
    if len(json.dumps(plan, allow_nan=False)) > 262144:
        raise WorkbenchError("Method validation exceeds its input byte budget")
    return plan


def _perturb(value, index, delta):
    position = 0
    def walk(item):
        nonlocal position
        if isinstance(item, list):
            return [walk(child) for child in item]
        result = item + delta if position == index else item
        position += 1
        return result
    return walk(value)


def build_request(method):
    plan = validate_plan(method["spec"]["validation"], method["spec"])
    calls = []
    def add(cid, role, call):
        step, _ = _call(call, method["spec"])
        calls.append({"id": f"{cid}:{role}", "code_file": step["code_file"], "code_symbol": step["code_symbol"],
                      "kwargs": copy.deepcopy(call["kwargs"]), "constructor_kwargs": call.get("constructor_kwargs", {})})
    for case in plan["cases"]:
        cid, call = case["id"], case["call"]
        if case["kind"] == "microinstance":
            add(cid, "value", call)
        elif case["kind"] == "component_toggle":
            for enabled in (True, False):
                changed = copy.deepcopy(call)
                changed["kwargs"][case["parameter"]] = enabled
                add(cid, "enabled" if enabled else "disabled", changed)
        else:
            add(cid, "gradient", case["gradient_call"])
            value = call["kwargs"][case["argument"]]
            for index in range(len(tensor(value)[1])):
                for sign, role in ((1, "plus"), (-1, "minus")):
                    changed = copy.deepcopy(call)
                    changed["kwargs"][case["argument"]] = _perturb(value, index, sign * case["epsilon"])
                    add(cid, f"{index}:{role}", changed)
    return {"schema_version": 1, "method_version": method["version"], "calls": calls}


def assess(method, observations):
    request = build_request(method)
    if (not isinstance(observations, dict) or set(observations) != {"schema_version", "observations"}
            or type(observations["schema_version"]) is not int or observations["schema_version"] != 1
            or not isinstance(observations["observations"], list)):
        raise WorkbenchError("Invalid method probe observations")
    rows = observations["observations"]
    if [r.get("id") for r in rows if isinstance(r, dict)] != [c["id"] for c in request["calls"]]:
        raise WorkbenchError("Probe observations do not cover the exact ordered invocation set")
    values = {}
    for row in rows:
        _fields(row, {"id", "value"} if "value" in row else {"id", "error"})
        values[row["id"]] = row
    results = []
    for case in method["spec"]["validation"]["cases"]:
        cid, details = case["id"], []
        def get(role):
            row = values[f"{cid}:{role}"]
            if "error" in row:
                raise WorkbenchError("Invocation failed: " + str(row["error"])[:500])
            tensor(row["value"])
            return row["value"]
        def compare(observed, expected, label):
            shape, actual = tensor(observed)
            expected_shape, wanted = tensor(expected)
            differences = [abs(a - b) for a, b in zip(actual, wanted)]
            passed = shape == expected_shape and all(math.isfinite(d) and d <= case["atol"] + case["rtol"] * abs(b)
                                                     for d, b in zip(differences, wanted))
            details.append({"comparison": label, "status": "passed" if passed else "failed",
                            "observed_shape": shape, "expected_shape": expected_shape,
                            "max_absolute_error": max(differences) if differences and all(math.isfinite(d) for d in differences) else None})
        try:
            if case["kind"] == "microinstance":
                compare(get("value"), case["expected"], "expected_output")
            elif case["kind"] == "component_toggle":
                for role in ("enabled", "disabled"):
                    compare(get(role), case["expected_" + role], role)
            else:
                arg_shape, coordinates = tensor(case["call"]["kwargs"][case["argument"]])
                derivative = []
                for index in range(len(coordinates)):
                    plus, minus = get(f"{index}:plus"), get(f"{index}:minus")
                    if not _number(plus) or not _number(minus):
                        raise WorkbenchError("Finite difference loss must be scalar")
                    derivative.append((plus - minus) / (2 * case["epsilon"]))
                gradient = get("gradient")
                gradient_shape, flattened = tensor(gradient)
                if gradient_shape != arg_shape:
                    raise WorkbenchError("Gradient shape differs from differentiated input")
                compare(flattened, derivative, "central_finite_difference")
            status = "passed" if all(d["status"] == "passed" for d in details) else "failed"
            results.append({"id": cid, "kind": case["kind"], "status": status, "checks": details})
        except (WorkbenchError, OverflowError) as exc:
            results.append({"id": cid, "kind": case["kind"], "status": "failed", "checks": details, "error": str(exc)})
    return results


def checker_identity():
    from researchclaw.experiment import method_probe
    return {"checker": file_hash(Path(__file__)), "driver": file_hash(Path(method_probe.__file__))}


def verify_validation(root, method, code):
    """Recompute comparisons from archived observations without executing code."""
    root = Path(root)
    if "validation" not in method["spec"]:
        return {"status": "not_declared", "semantic_equivalence": "unresolved"}
    try:
        from researchclaw.pipeline.research_workbench import compile_method
        if (compile_method(method["spec"]) != method or not isinstance(code["files"], dict) or not code["files"]
                or content_hash(code["files"]) != code["code_sha256"]):
            raise WorkbenchError("Invalid frozen method or source identity")
        source_root = (root / "evidence_artifacts" / "protocol_source").resolve()
        for name, digest in code["files"].items():
            path = (source_root / name).resolve()
            if not path.is_relative_to(source_root) or file_hash(path) != digest:
                raise WorkbenchError("Method validation source archive changed")
        artifact = root / "evidence_artifacts" / "method_validation"
        report = json.loads((root / "method_validation.json").read_text(encoding="utf-8"))
        reservation = json.loads((artifact / "reservation.json").read_text(encoding="utf-8"))
        expected = {"method_version": method["version"], "code_sha256": code["code_sha256"],
                    "checker_identity": checker_identity(), "backend": code["backend"],
                    "execution_config": code["execution_config"], "request": build_request(method)}
        _fields(reservation, set(expected) | {"timeout_seconds", "started_at"})
        if any(reservation.get(k) != v for k, v in expected.items()):
            raise WorkbenchError("Method probe reservation differs from frozen inputs")
        if type(reservation["timeout_seconds"]) is not int or not 1 <= reservation["timeout_seconds"] <= method["spec"]["validation"]["timeout_seconds"]:
            raise WorkbenchError("Invalid method probe time reservation")
        if not _number(reservation["started_at"]) or reservation["started_at"] <= 0:
            raise WorkbenchError("Invalid method probe start time")
        expected_files = {"reservation.json", "observations.json", "stdout.txt", "stderr.txt", "execution.json", "driver.py"}
        if (not isinstance(report["files"], dict) or set(report["files"]) != expected_files
                or any(file_hash(artifact / name) != digest for name, digest in report["files"].items())):
            raise WorkbenchError("Method probe artifacts changed or are incomplete")
        if file_hash(artifact / "driver.py") != checker_identity()["driver"]:
            raise WorkbenchError("Archived method probe driver changed")
        receipt = json.loads((artifact / "execution.json").read_text(encoding="utf-8"))
        _fields(receipt, {"returncode", "timed_out", "elapsed_seconds", "charged_seconds", "finished_at"}, {"error"})
        if (type(receipt["returncode"]) is not int or receipt["returncode"] != 0
                or receipt["timed_out"] is not False or receipt.get("error")
                or not _number(receipt["elapsed_seconds"]) or not 0 <= receipt["elapsed_seconds"] <= reservation["timeout_seconds"]
                or not _number(receipt["finished_at"]) or receipt["finished_at"] < reservation["started_at"]
                or not _number(receipt["charged_seconds"]) or receipt["charged_seconds"] != receipt["elapsed_seconds"]):
            raise WorkbenchError("Method probe execution failed, timed out or exceeded its budget")
        checks = assess(method, json.loads((artifact / "observations.json").read_text(encoding="utf-8")))
        expected_report = make_report(method, code, artifact, checks, receipt)
        if report != expected_report or report["status"] != "passed":
            raise WorkbenchError("Method probe comparisons failed or report changed")
        return report
    except (OSError, ValueError, TypeError, KeyError) as exc:
        if isinstance(exc, WorkbenchError):
            raise
        raise WorkbenchError("Missing or invalid method validation evidence") from exc


def make_report(method, code, artifact, checks, receipt):
    tested = sorted({call["step"] for case in method["spec"]["validation"]["cases"]
                     for call in (case["call"], case.get("gradient_call", case["call"]))})
    report = {"schema_version": 1, "method_version": method["version"], "code_sha256": code["code_sha256"],
              "checker_identity": checker_identity(), "checks": checks,
              "status": "passed" if checks and all(c["status"] == "passed" for c in checks)
              and receipt["returncode"] == 0 and not receipt["timed_out"] and not receipt.get("error") else "failed",
              "semantic_equivalence": "unresolved", "limitations": LIMITATIONS,
              "coverage": {"tested_steps": tested,
                           "untested_steps": sorted({s["id"] for s in method["spec"]["steps"]} - set(tested)),
                           "kinds": sorted({c["kind"] for c in checks})},
              "budget_scope": "pretest_method_validation_wall_time_separate_from_formal_matrix",
              "elapsed_seconds": receipt["elapsed_seconds"],
              "charged_seconds": receipt["charged_seconds"], "finished_at": receipt["finished_at"],
              "files": {name: file_hash(artifact / name) for name in
                        ("reservation.json", "observations.json", "stdout.txt", "stderr.txt", "execution.json", "driver.py")}}
    report["version"] = content_hash(report)
    return report
