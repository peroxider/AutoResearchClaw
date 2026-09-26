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
    "Autograd gradients are computed on the frozen forward code inside the probe process and cross-checked against host finite differences; this is not a derivative proof or a backend certification.",
    "In-process determinism checks cover repeated calls within one probe process only.",
    "Fresh-process determinism checks cover two new interpreters on the configured host with a frozen driver-owned RNG policy; they do not certify cross-host or cross-device reproducibility.",
    "Device inventory records runtime availability, applied seed backends, and deterministic settings before project imports; it does not prove that method code used a device or executed deterministically on it.",
    "Toggle checks compare outputs; they do not prove component removal or general effectiveness.",
    "Method-wide equivalence, adversarial mutation of process-global RNG state, and scientific validity remain unresolved.",
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
        if not isinstance(case, dict):
            raise WorkbenchError("Invalid probe case")
        cid = case.get("id")
        if not isinstance(cid, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,63}", cid) or cid in ids:
            raise WorkbenchError("Invalid or duplicate probe ID")
        ids.add(cid)
        kind = case.get("kind")
        if kind == "device_inventory":
            _fields(case, {"id", "kind", "required_devices", "minimum_cuda_devices",
                           "require_torch_determinism"})
            devices = case["required_devices"]
            if (not isinstance(devices, list) or not 1 <= len(devices) <= 3
                    or any(device not in {"cpu", "cuda", "mps"} for device in devices)
                    or len(set(devices)) != len(devices)
                    or type(case["minimum_cuda_devices"]) is not int
                    or not 0 <= case["minimum_cuda_devices"] <= 64
                    or type(case["require_torch_determinism"]) is not bool):
                raise WorkbenchError("Invalid device inventory requirements")
            continue
        common = {"id", "kind", "call", "atol", "rtol"}
        extra = {"microinstance": {"expected"},
                 "gradient": {"gradient_call", "argument", "epsilon"},
                 "component_toggle": {"parameter", "expected_enabled", "expected_disabled"},
                 "autodiff": {"argument", "epsilon"},
                 "determinism": set(),
                 "process_determinism": set()}.get(kind)
        if extra is None:
            raise WorkbenchError("Unsupported method validation kind")
        _fields(case, common | extra)
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
        elif kind in {"determinism", "process_determinism"}:
            if kind == "process_determinism" and plan["timeout_seconds"] < 2:
                raise WorkbenchError("Fresh-process determinism needs at least two seconds")
            count += 2
        elif kind == "autodiff":
            arg = case["argument"]
            if not isinstance(arg, str) or arg not in case["call"]["kwargs"] or output_shape:
                raise WorkbenchError("Autodiff probe requires a scalar loss and an input argument")
            _, values = tensor(case["call"]["kwargs"][arg])
            if len(values) > 64 or not _number(case["epsilon"]) or not 1e-8 <= case["epsilon"] <= 0.01:
                raise WorkbenchError("Autodiff probe exceeds coordinate or perturbation bounds")
            for value in values:
                if any(not math.isfinite(value + sign * case["epsilon"]) or value + sign * case["epsilon"] == value for sign in (-1, 1)):
                    raise WorkbenchError("Gradient perturbation is not representable")
            count += 1 + 2 * len(values)
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
    def add(cid, role, call, **extra):
        step, _ = _call(call, method["spec"])
        entry = {"id": f"{cid}:{role}", "code_file": step["code_file"], "code_symbol": step["code_symbol"],
                 "kwargs": copy.deepcopy(call["kwargs"]), "constructor_kwargs": call.get("constructor_kwargs", {})}
        entry.update(extra)
        calls.append(entry)
    def perturbed(cid, case):
        value = case["call"]["kwargs"][case["argument"]]
        for index in range(len(tensor(value)[1])):
            for sign, role in ((1, "plus"), (-1, "minus")):
                changed = copy.deepcopy(case["call"])
                changed["kwargs"][case["argument"]] = _perturb(value, index, sign * case["epsilon"])
                add(cid, f"{index}:{role}", changed)
    deferred = []
    for case in plan["cases"]:
        cid = case["id"]
        if case["kind"] == "device_inventory":
            continue
        call = case["call"]
        if case["kind"] == "microinstance":
            add(cid, "value", call)
        elif case["kind"] == "component_toggle":
            for enabled in (True, False):
                changed = copy.deepcopy(call)
                changed["kwargs"][case["parameter"]] = enabled
                add(cid, "enabled" if enabled else "disabled", changed)
        elif case["kind"] == "autodiff":
            add(cid, "autodiff", call, mode="autodiff", argument=case["argument"])
            perturbed(cid, case)
        elif case["kind"] == "determinism":
            add(cid, "first", call)
            deferred.append((cid, call))
        elif case["kind"] == "process_determinism":
            add(cid, "first", call, mode="fresh_process")
            add(cid, "repeat", call, mode="fresh_process")
        else:
            add(cid, "gradient", case["gradient_call"])
            perturbed(cid, case)
    # Determinism repeats are appended last so every other call ran in between.
    for cid, call in deferred:
        add(cid, "repeat", call)
    enforce_determinism = any(
        case["kind"] == "device_inventory" and case["require_torch_determinism"]
        for case in plan["cases"])
    request = {"schema_version": 1, "method_version": method["version"],
               "runtime_policy": {"seed": 0,
                                  "enforce_torch_determinism": enforce_determinism},
               "calls": calls}
    if any(case["kind"] == "process_determinism" for case in plan["cases"]):
        request["fresh_process_timeout_seconds"] = max(1, plan["timeout_seconds"] // 2)
    return request


def _bounded_text(value, maximum=1000):
    return isinstance(value, str) and len(value) <= maximum


def validate_runtime(value):
    """Validate the driver-owned pre-import runtime inventory and return it."""
    _fields(value, {"schema_version", "controller", "fresh_processes"})
    if type(value["schema_version"]) is not int or value["schema_version"] != 1:
        raise WorkbenchError("Unsupported runtime inventory schema")

    def state(item):
        _fields(item, {"process_id", "python_version", "implementation", "platform",
                       "environment", "torch", "rng_policy"})
        if (type(item["process_id"]) is not int or item["process_id"] <= 0
                or any(not _bounded_text(item[key]) for key in ("python_version", "implementation", "platform"))):
            raise WorkbenchError("Invalid runtime process identity")
        environment = item["environment"]
        names = {"CUDA_VISIBLE_DEVICES", "CUBLAS_WORKSPACE_CONFIG", "CUDA_LAUNCH_BLOCKING"}
        _fields(environment, names)
        if any(value is not None and not _bounded_text(value) for value in environment.values()):
            raise WorkbenchError("Invalid runtime environment inventory")
        rng = item["rng_policy"]
        _fields(rng, {"seed", "python", "numpy", "torch", "cuda_all",
                      "enforce_torch_determinism"})
        if (rng["seed"] != 0 or rng["python"] != "applied"
                or type(rng["enforce_torch_determinism"]) is not bool):
            raise WorkbenchError("Invalid runtime RNG policy")
        allowed_rng = {"applied", "unavailable", "not_available"}
        for key in ("numpy", "torch", "cuda_all"):
            status_value = rng[key]
            if (not _bounded_text(status_value, 100)
                    or not (status_value in allowed_rng
                            or re.fullmatch(r"error:[A-Za-z_]\w*", status_value))):
                raise WorkbenchError("Invalid runtime RNG backend status")
        torch = item["torch"]
        torch_fields = {"status", "version", "cuda_available", "cuda_device_count", "cuda_devices",
                        "mps_available", "deterministic_algorithms", "cudnn_deterministic",
                        "cudnn_benchmark", "cudnn_version"}
        _fields(torch, torch_fields)
        status = torch["status"]
        if not _bounded_text(status, 100) or not (status in {"available", "unavailable"}
                                                   or re.fullmatch(r"error:[A-Za-z_]\w*", status)):
            raise WorkbenchError("Invalid torch runtime status")
        if torch["version"] is not None and not _bounded_text(torch["version"], 200):
            raise WorkbenchError("Invalid torch runtime version")
        nullable_bools = ("cuda_available", "mps_available", "deterministic_algorithms",
                          "cudnn_deterministic", "cudnn_benchmark")
        if any(torch[key] is not None and type(torch[key]) is not bool for key in nullable_bools):
            raise WorkbenchError("Invalid torch runtime Boolean")
        if (torch["cuda_device_count"] is not None
                and (type(torch["cuda_device_count"]) is not int or not 0 <= torch["cuda_device_count"] <= 64)):
            raise WorkbenchError("Invalid CUDA device count")
        if (torch["cudnn_version"] is not None
                and (type(torch["cudnn_version"]) is not int or torch["cudnn_version"] < 0)):
            raise WorkbenchError("Invalid cuDNN version")
        devices = torch["cuda_devices"]
        if not isinstance(devices, list) or len(devices) > 64:
            raise WorkbenchError("Invalid CUDA device inventory")
        for index, device in enumerate(devices):
            _fields(device, {"index", "name", "capability", "total_memory"})
            capability = device["capability"]
            if (type(device["index"]) is not int or device["index"] != index
                    or not _bounded_text(device["name"], 500)
                    or not isinstance(capability, list) or len(capability) != 2
                    or any(type(part) is not int or part < 0 for part in capability)
                    or type(device["total_memory"]) is not int or device["total_memory"] < 0):
                raise WorkbenchError("Invalid CUDA device entry")
        if status == "available":
            if (torch["version"] is None or any(torch[key] is None for key in nullable_bools)
                    or torch["cuda_device_count"] is None
                    or len(devices) != torch["cuda_device_count"]
                    or (torch["cuda_available"] and torch["cuda_device_count"] == 0)
                    or (torch["cuda_device_count"] == 0 and torch["cuda_available"] is not False)):
                raise WorkbenchError("Incomplete available torch runtime inventory")
        elif (any(torch[key] is not None for key in nullable_bools)
              or torch["cuda_device_count"] is not None or devices or torch["cudnn_version"] is not None):
            raise WorkbenchError("Unavailable torch runtime has device claims")
        return item

    state(value["controller"])
    fresh = value["fresh_processes"]
    if not isinstance(fresh, list) or len(fresh) > MAX_CALLS:
        raise WorkbenchError("Invalid fresh-process runtime inventory")
    ids = set()
    for entry in fresh:
        _fields(entry, {"call_id", "state"})
        if not _bounded_text(entry["call_id"], 130) or entry["call_id"] in ids:
            raise WorkbenchError("Invalid fresh-process call identity")
        ids.add(entry["call_id"])
        state(entry["state"])
    return value


def assess(method, observations):
    request = build_request(method)
    if (not isinstance(observations, dict)
            or set(observations) - {"schema_version", "observations", "runtime"}
            or not {"schema_version", "observations"}.issubset(observations)
            or type(observations["schema_version"]) is not int or observations["schema_version"] != 1
            or not isinstance(observations["observations"], list)):
        raise WorkbenchError("Invalid method probe observations")
    runtime = validate_runtime(observations["runtime"]) if "runtime" in observations else None
    if runtime is not None:
        expected_policy = request["runtime_policy"]
        runtime_states = [runtime["controller"],
                          *(entry["state"] for entry in runtime["fresh_processes"])]
        if any(state["rng_policy"]["seed"] != expected_policy["seed"]
               or state["rng_policy"]["enforce_torch_determinism"]
               != expected_policy["enforce_torch_determinism"]
               for state in runtime_states):
            raise WorkbenchError("Runtime RNG policy differs from the declared request")
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
            elif case["kind"] == "determinism":
                compare(get("repeat"), get("first"), "repeat_after_other_calls")
            elif case["kind"] == "process_determinism":
                compare(get("repeat"), get("first"), "fresh_python_process_repeat")
                if runtime is None:
                    raise WorkbenchError("Fresh-process determinism needs runtime inventory")
                state_by_call = {entry["call_id"]: entry["state"] for entry in runtime["fresh_processes"]}
                wanted = [f"{cid}:first", f"{cid}:repeat"]
                if any(call_id not in state_by_call for call_id in wanted):
                    raise WorkbenchError("Fresh-process runtime inventory is incomplete")
                pids = [state_by_call[call_id]["process_id"] for call_id in wanted]
                distinct = len(set(pids)) == 2 and runtime["controller"]["process_id"] not in pids
                details.append({"comparison": "distinct_fresh_processes",
                                "status": "passed" if distinct else "failed",
                                "controller_process_id": runtime["controller"]["process_id"],
                                "fresh_process_ids": pids})
            elif case["kind"] == "device_inventory":
                if runtime is None:
                    raise WorkbenchError("Device inventory is missing")
                torch = runtime["controller"]["torch"]
                available = {"cpu": True,
                             "cuda": torch["status"] == "available" and torch["cuda_available"] is True,
                             "mps": torch["status"] == "available" and torch["mps_available"] is True}
                required = all(available[device] for device in case["required_devices"])
                cuda_count = torch["cuda_device_count"] if torch["status"] == "available" else 0
                enough_cuda = cuda_count >= case["minimum_cuda_devices"]
                deterministic = (not case["require_torch_determinism"]
                                 or (torch["status"] == "available"
                                     and torch["deterministic_algorithms"] is True))
                details.extend([
                    {"comparison": "required_devices", "status": "passed" if required else "failed",
                     "required": case["required_devices"], "available": available},
                    {"comparison": "minimum_cuda_devices", "status": "passed" if enough_cuda else "failed",
                     "required": case["minimum_cuda_devices"], "observed": cuda_count,
                     "devices": torch["cuda_devices"]},
                    {"comparison": "torch_deterministic_algorithms",
                     "status": "passed" if deterministic else "failed",
                     "required": case["require_torch_determinism"],
                     "observed": torch["deterministic_algorithms"], "torch_status": torch["status"]},
                    {"comparison": "rng_seed_policy",
                     "status": "passed" if (
                         runtime["controller"]["rng_policy"]["python"] == "applied"
                         and (torch["status"] != "available"
                              or runtime["controller"]["rng_policy"]["torch"] == "applied")
                         and (case["minimum_cuda_devices"] == 0
                              or runtime["controller"]["rng_policy"]["cuda_all"] == "applied")
                         and runtime["controller"]["rng_policy"]["enforce_torch_determinism"]
                         == case["require_torch_determinism"]
                     ) else "failed",
                     "policy": runtime["controller"]["rng_policy"]},
                ])
            elif case["kind"] == "autodiff":
                arg_shape, coordinates = tensor(case["call"]["kwargs"][case["argument"]])
                derivative = []
                for index in range(len(coordinates)):
                    plus, minus = get(f"{index}:plus"), get(f"{index}:minus")
                    if not _number(plus) or not _number(minus):
                        raise WorkbenchError("Finite difference loss must be scalar")
                    derivative.append((plus - minus) / (2 * case["epsilon"]))
                autograd = get("autodiff")
                autograd_shape, flattened = tensor(autograd)
                if autograd_shape != arg_shape:
                    raise WorkbenchError("Autograd gradient shape differs from differentiated input")
                compare(flattened, derivative, "autograd_vs_central_finite_difference")
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
    tested = sorted({call["step"] for case in method["spec"]["validation"]["cases"] if "call" in case
                     for call in (case["call"], case.get("gradient_call", case["call"]))})
    runtime = None
    try:
        observations = json.loads((artifact / "observations.json").read_text(encoding="utf-8"))
        if "runtime" in observations:
            runtime = validate_runtime(observations["runtime"])
    except (OSError, ValueError, TypeError, KeyError, WorkbenchError):
        runtime = None
    report = {"schema_version": 1, "method_version": method["version"], "code_sha256": code["code_sha256"],
              "checker_identity": checker_identity(), "checks": checks,
              "status": "passed" if checks and all(c["status"] == "passed" for c in checks)
              and runtime is not None and receipt["returncode"] == 0
              and not receipt["timed_out"] and not receipt.get("error") else "failed",
              "semantic_equivalence": "unresolved", "limitations": LIMITATIONS,
              "runtime_environment": runtime,
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
