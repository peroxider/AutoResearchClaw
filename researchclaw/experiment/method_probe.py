"""Standalone invocation driver. No expected values or pass/fail decisions here."""
from __future__ import annotations

import importlib
import inspect
import json
import math
import random
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


def main():
    base = Path(__file__).resolve().parent
    request = json.loads((base / "request.json").read_text(encoding="utf-8"))
    source = base / "source"
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
    (base / "observations.json").write_text(json.dumps({"schema_version": 1, "observations": observations},
                                                      allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
