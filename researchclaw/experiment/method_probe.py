"""Standalone invocation driver. No expected values or pass/fail decisions here."""
from __future__ import annotations

import importlib
import inspect
import json
import math
import os
import platform
import random
import subprocess
import sys
from pathlib import Path


def numeric(value, depth=0, budget=None):
    if budget is None:
        budget = [256]
    if depth > 6:
        raise ValueError("Output rank exceeds probe bounds")
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if type(value) in (int, float) and math.isfinite(value):
        budget[0] -= 1
        if budget[0] < 0:
            raise ValueError("Output exceeds element budget")
        return value
    if isinstance(value, (list, tuple)) and 1 <= len(value) <= 256:
        return [numeric(v, depth + 1, budget) for v in value]
    raise ValueError("Output is not a finite numeric scalar or array")


def autograd_gradient(call, target):
    """Compute the gradient of the declared loss with PyTorch autograd.

    The backend runs on the frozen forward code inside this probe process;
    conclusions are drawn outside, never here."""
    try:
        import torch
    except ImportError as exc:
        raise ValueError("PyTorch backend is required for autodiff probes and is unavailable") from exc
    inputs = {}
    for key, value in call["kwargs"].items():
        if isinstance(value, bool):
            inputs[key] = value
        else:
            inputs[key] = torch.tensor(value, dtype=torch.float64, requires_grad=key == call["argument"])
    output = target(**inputs)
    if not isinstance(output, torch.Tensor) or output.dim() != 0 or not output.requires_grad:
        raise ValueError("Autodiff probe needs a scalar differentiable tensor loss")
    output.backward()
    gradient = inputs[call["argument"]].grad
    if gradient is None:
        raise ValueError("Autodiff produced no gradient for the declared argument")
    return gradient


def runtime_state():
    """Inspect the interpreter before project source is added to ``sys.path``."""
    state = {"process_id": os.getpid(), "python_version": platform.python_version(),
             "implementation": platform.python_implementation(), "platform": platform.platform(),
             "environment": {name: os.environ.get(name) for name in
                             ("CUDA_VISIBLE_DEVICES", "CUBLAS_WORKSPACE_CONFIG", "CUDA_LAUNCH_BLOCKING")},
             "torch": {"status": "unavailable", "version": None, "cuda_available": None,
                       "cuda_device_count": None, "cuda_devices": [], "mps_available": None,
                       "deterministic_algorithms": None, "cudnn_deterministic": None,
                       "cudnn_benchmark": None, "cudnn_version": None}}
    try:
        import torch
    except ImportError:
        return state
    except Exception as exc:
        state["torch"]["status"] = "error:" + type(exc).__name__
        return state
    info = state["torch"]
    try:
        info.update(status="available", version=str(torch.__version__),
                    cuda_available=bool(torch.cuda.is_available()),
                    cuda_device_count=int(torch.cuda.device_count()),
                    deterministic_algorithms=bool(torch.are_deterministic_algorithms_enabled()),
                    cudnn_deterministic=bool(torch.backends.cudnn.deterministic),
                    cudnn_benchmark=bool(torch.backends.cudnn.benchmark),
                    cudnn_version=torch.backends.cudnn.version())
        mps = getattr(torch.backends, "mps", None)
        info["mps_available"] = bool(mps is not None and mps.is_available())
        for index in range(info["cuda_device_count"]):
            properties = torch.cuda.get_device_properties(index)
            capability = torch.cuda.get_device_capability(index)
            info["cuda_devices"].append({"index": index, "name": str(properties.name),
                                         "capability": [int(capability[0]), int(capability[1])],
                                         "total_memory": int(properties.total_memory)})
    except Exception as exc:
        info.update(status="error:" + type(exc).__name__, cuda_available=None,
                    cuda_device_count=None, cuda_devices=[], mps_available=None,
                    deterministic_algorithms=None, cudnn_deterministic=None,
                    cudnn_benchmark=None, cudnn_version=None)
    return state


def execute_calls(request, source):
    sys.path.insert(0, str(source))
    observations = []
    for call in request["calls"]:
        row = {"id": call["id"]}
        try:
            path = source / call["code_file"]
            module_name = ".".join(Path(call["code_file"]).with_suffix("").parts)
            module = importlib.import_module(module_name)
            if Path(module.__file__).resolve() != path.resolve():
                raise ValueError("Imported module does not match the mapped source file")
            # Common seeds reduce incidental differences; global state is not certified isolated.
            random.seed(0)
            if "numpy" in sys.modules:
                sys.modules["numpy"].random.seed(0)
            if "torch" in sys.modules:
                sys.modules["torch"].manual_seed(0)
            parts = call["code_symbol"].split(".")
            target = getattr(module, parts[0])
            if len(parts) == 2:
                target = getattr(target(**call["constructor_kwargs"]), parts[1])
            if call.get("mode") == "autodiff":
                row["value"] = numeric(autograd_gradient(call, target))
            else:
                output = target(**call["kwargs"])
                if inspect.isawaitable(output):
                    if inspect.iscoroutine(output):
                        output.close()
                    raise ValueError("Async method probes are unsupported")
                row["value"] = numeric(output)
            if len(json.dumps(row)) > 65536:
                raise ValueError("Probe output exceeds byte budget")
        except Exception as exc:
            row = {"id": call["id"], "error": f"{type(exc).__name__}: {exc}"[:1000]}
        observations.append(row)
    return observations


def _fresh_call(base, request, call, index):
    child_call = dict(call)
    child_call.pop("mode", None)
    child_request = {"schema_version": request["schema_version"], "method_version": request["method_version"],
                     "calls": [child_call]}
    request_path, output_path = base / f"fresh-{index}-request.json", base / f"fresh-{index}-output.json"
    request_path.write_text(json.dumps(child_request, allow_nan=False), encoding="utf-8")
    completed = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--child",
                                str(request_path), str(output_path)], capture_output=True, text=True,
                               encoding="utf-8", errors="replace",
                               timeout=request.get("fresh_process_timeout_seconds", 60),
                               cwd=base, check=False)
    if completed.stdout:
        print(f"[fresh process {index} stdout]\n{completed.stdout}", end="")
    if completed.stderr:
        print(f"[fresh process {index} stderr]\n{completed.stderr}", end="", file=sys.stderr)
    if completed.returncode != 0 or not output_path.is_file():
        return {"id": call["id"], "error": f"FreshProcessError: returncode={completed.returncode}"}, None
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    rows = payload.get("observations", [])
    if len(rows) != 1 or rows[0].get("id") != call["id"]:
        return {"id": call["id"], "error": "FreshProcessError: invalid child output"}, None
    return rows[0], payload.get("runtime")


def main():
    base = Path(__file__).resolve().parent
    if len(sys.argv) == 4 and sys.argv[1] == "--child":
        request = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
        state = runtime_state()
        observations = execute_calls(request, base / "source")
        Path(sys.argv[3]).write_text(json.dumps({"schema_version": 1, "observations": observations,
                                                "runtime": state}, allow_nan=False), encoding="utf-8")
        return
    request = json.loads((base / "request.json").read_text(encoding="utf-8"))
    state = runtime_state()
    normal = {**request, "calls": [call for call in request["calls"] if call.get("mode") != "fresh_process"]}
    normal_rows = {row["id"]: row for row in execute_calls(normal, base / "source")}
    fresh_states, observations = [], []
    for index, call in enumerate(request["calls"], start=1):
        if call.get("mode") == "fresh_process":
            row, child_state = _fresh_call(base, request, call, index)
            if child_state is not None:
                fresh_states.append({"call_id": call["id"], "state": child_state})
            observations.append(row)
        else:
            observations.append(normal_rows[call["id"]])
    (base / "observations.json").write_text(json.dumps({"schema_version": 1, "observations": observations,
                                                        "runtime": {"schema_version": 1,
                                                                    "controller": state,
                                                                    "fresh_processes": fresh_states}},
                                                       allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
