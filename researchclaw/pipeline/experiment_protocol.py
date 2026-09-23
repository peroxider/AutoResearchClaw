"""Predeclared comparisons and deterministic coverage of required experiments.

Budgets here are allocations, not measurements. Execution isolation, actual
resource accounting and the scientific adequacy of a comparison need separate
review; a complete matrix does not establish those properties.
"""
from __future__ import annotations

import json
import re
import statistics
from dataclasses import asdict
from pathlib import Path
from typing import Any

from researchclaw.pipeline.evidence_store import EvidenceKey, EvidenceStore, content_hash


class ProtocolError(ValueError):
    """An invalid, changed or incomplete experiment protocol."""


def _object(value: Any, fields: set[str], required: set[str], name: str) -> dict:
    if not isinstance(value, dict) or value.keys() - fields or required - value.keys():
        raise ProtocolError(f"Invalid {name} fields")
    return value


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProtocolError(f"{name} must be a nonempty string")
    return value


def _identifier(value: Any) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,63}", value):
        raise ProtocolError("Protocol IDs must be safe ASCII identifiers")
    return value


def _list(value: Any, name: str) -> list:
    if not isinstance(value, list) or not value:
        raise ProtocolError(f"{name} must be a nonempty list")
    return value


def _names(value: Any, name: str) -> list[str]:
    result = [_text(v, name) for v in _list(value, name)]
    if len(set(result)) != len(result):
        raise ProtocolError(f"Duplicate {name}")
    return result


def _positive(value: Any, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ProtocolError(f"{name} must be a positive integer")
    return value


def _json_value(value: Any) -> None:
    if isinstance(value, dict):
        if any(not isinstance(k, str) for k in value):
            raise ProtocolError("Parameter names must be strings")
        for child in value.values():
            _json_value(child)
    elif isinstance(value, list):
        for child in value:
            _json_value(child)
    elif value is not None and type(value) not in (str, int, float, bool):
        raise ProtocolError("Parameters must be JSON values")
    try:
        content_hash(value)
    except (TypeError, ValueError) as exc:
        raise ProtocolError("Parameters must be finite JSON values") from exc


def compile_protocol(spec: dict, datasets: list[dict], *, max_cell_seconds: int) -> dict:
    """Expand dataset x RQ x method x seed before any result is observed."""
    _object(spec, {"schema_version", "methods", "questions", "seeds", "budget"},
            {"schema_version", "methods", "questions", "seeds", "budget"}, "protocol")
    if type(spec["schema_version"]) is not int or spec["schema_version"] != 1:
        raise ProtocolError("Unsupported input protocol schema")
    seeds = _list(spec["seeds"], "seeds")
    if any(type(s) is not int or not 0 <= s < 2**32 for s in seeds) or len(set(seeds)) != len(seeds):
        raise ProtocolError("Seeds must be unique unsigned 32-bit integers")
    budget = _object(spec["budget"], {"per_cell_seconds", "max_total_seconds", "tuning_trials"},
                     {"per_cell_seconds", "max_total_seconds", "tuning_trials"}, "budget")
    for name, value in budget.items():
        _positive(value, name)
    if budget["per_cell_seconds"] > max_cell_seconds:
        raise ProtocolError("Per-cell allocation exceeds ResearchBrief time limit")
    methods = {}
    for method in _list(spec["methods"], "methods"):
        _object(method, {"id", "role", "description", "parameters", "parent", "disabled_components"},
                {"id", "role", "description", "parameters"}, "method")
        mid = _identifier(method["id"])
        if mid.casefold() in {m.casefold() for m in methods}:
            raise ProtocolError("Duplicate method ID")
        if method["role"] not in {"baseline", "proposed", "ablation"}:
            raise ProtocolError("Unknown method role")
        _text(method["description"], "method description")
        if not isinstance(method["parameters"], dict):
            raise ProtocolError("Method parameters must be an object")
        _json_value(method["parameters"])
        if method["role"] != "ablation" and ("parent" in method or "disabled_components" in method):
            raise ProtocolError("Only ablations may declare a parent or disabled components")
        methods[mid] = method
    for method in methods.values():
        if method["role"] == "ablation":
            parent = methods.get(_text(method.get("parent"), "ablation parent"))
            _names(method.get("disabled_components"), "disabled components")
            if parent is None or parent["role"] != "proposed":
                raise ProtocolError("Ablation parent must be a declared proposed method")
            if parent["parameters"] == method["parameters"]:
                raise ProtocolError("Ablation must declare a changed configuration")
    available = {d["manifest"]["dataset"]: d for d in datasets}
    cells, comparisons, questions, used_methods, used_data = [], [], set(), set(), set()
    analysis_draws = 0
    for question in _list(spec["questions"], "questions"):
        _object(question, {"id", "kind", "question", "datasets", "methods", "baseline", "analysis", "analysis_plan"},
                {"id", "kind", "question", "datasets", "methods", "baseline", "analysis"}, "question")
        qid = _identifier(question["id"])
        if qid.casefold() in questions:
            raise ProtocolError("Duplicate research question ID")
        questions.add(qid.casefold())
        if question["kind"] not in {"main", "ablation", "sensitivity", "generalization"}:
            raise ProtocolError("Unsupported question kind")
        _text(question["question"], "research question")
        _text(question["analysis"], "analysis and inference limitations")
        if "analysis_plan" in question:
            from researchclaw.pipeline.analysis_spec import validate_plan, AnalysisError
            try:
                if question["analysis_plan"] is None:
                    raise AnalysisError("Explicit analysis_plan cannot be null")
                validate_plan(question["analysis_plan"])
            except AnalysisError as exc:
                raise ProtocolError(str(exc)) from exc
        mids = _names(question["methods"], "question methods")
        names = _names(question["datasets"], "question datasets")
        baseline = _text(question["baseline"], "comparison baseline")
        if len(mids) < 2 or baseline not in mids or set(mids) - methods.keys():
            raise ProtocolError("Every question needs declared methods and an explicit comparison baseline")
        if question["kind"] != "ablation" and methods[baseline]["role"] != "baseline":
            raise ProtocolError("Main comparisons must reference a baseline method")
        if question["kind"] == "ablation":
            if methods[baseline]["role"] != "proposed" or any(
                methods[mid]["role"] != "ablation" or methods[mid].get("parent") != baseline
                for mid in mids if mid != baseline
            ):
                raise ProtocolError("Ablation comparisons require the full method and its declared ablations")
        if set(names) - available.keys():
            raise ProtocolError("Question references undeclared dataset")
        if "analysis_plan" in question and question["analysis_plan"]["interval"]["method"] != "none":
            from researchclaw.pipeline.analysis_spec import MAX_RESAMPLING_DRAWS
            interval = question["analysis_plan"]["interval"]
            if interval["method"] == "cluster_percentile_bootstrap" and any(
                    not available[name]["manifest"].get("group_column") for name in names):
                raise ProtocolError("Cluster bootstrap requires group_column on every question dataset")
            if interval["method"] == "moving_block_percentile_bootstrap":
                if any(not available[name]["manifest"].get("time_column") for name in names):
                    raise ProtocolError("Moving-block bootstrap requires time_column on every question dataset")
                if any(interval["block_length"] > available[name]["card"]["analysis_units"]["test_time_count"]
                       for name in names):
                    raise ProtocolError("Moving-block length exceeds a question dataset's frozen test time points")
            sample_factor = (sum(available[name]["card"]["split_sizes"]["test"] for name in names)
                             if interval["method"] in {"cluster_percentile_bootstrap",
                                                       "moving_block_percentile_bootstrap"} else len(names))
            analysis_draws += len(seeds) * sample_factor * (len(mids) - 1) * interval["replicates"]
            if analysis_draws > MAX_RESAMPLING_DRAWS:
                raise ProtocolError("Frozen analysis plans exceed the total resampling draw budget")
        used_methods.update(mids)
        used_data.update(names)
        for name in names:
            dataset = available[name]
            for seed in seeds:
                paired = {}
                for mid in mids:
                    key = EvidenceKey(name, dataset["manifest"]["version"], "test", mid,
                                      content_hash(methods[mid]), str(seed), dataset["manifest"]["metric"],
                                      "per_run", qid)
                    identity = asdict(key)
                    cell_id = content_hash(identity)
                    paired[mid] = cell_id
                    cells.append({"cell_id": cell_id, "key": identity,
                                  "split_ids_sha256": dataset["card"]["split_ids_sha256"],
                                  "budget": dict(budget)})
                for mid in mids:
                    if mid != baseline:
                        comparisons.append({"question": qid, "baseline_cell": paired[baseline],
                                            "candidate_cell": paired[mid], "unit": "seed"})
    if used_methods != methods.keys() or used_data != available.keys():
        raise ProtocolError("Every declared method and input dataset must occur in the required matrix")
    if not any(q["kind"] == "main" for q in spec["questions"]):
        raise ProtocolError("Protocol must declare a main comparison")
    reserved = len(cells) * budget["per_cell_seconds"]
    if reserved > budget["max_total_seconds"]:
        raise ProtocolError("Required matrix exceeds total allocation; revise the protocol, do not trim cells")
    document = {"schema_version": 2, "spec": spec, "cells": cells,
                "comparisons": comparisons, "required_keys": [c["key"] for c in cells],
                "allocation": {"required_cells": len(cells), "reserved_seconds": reserved,
                               "actual_usage_status": "unmeasured"},
                "selection_split": "validation", "preprocessing_fit_split": "train"}
    # Ensure YAML tuples/aliases cannot change the in-memory representation later.
    document = json.loads(json.dumps(document, allow_nan=False))
    document["version"] = content_hash(document)
    return document


def load_protocol(root: Path, contract: dict | None = None) -> dict | None:
    """Recompile instead of trusting a hand-edited list of required result keys."""
    path = root / "experiment_protocol.json"
    if contract is None and (root / "research_contract.json").is_file():
        from researchclaw.research_inputs import verify_bundle_contract
        contract = verify_bundle_contract(root)
    declared = bool(contract and contract["brief"].get("protocol_path"))
    if not path.is_file():
        if declared:
            raise ProtocolError("Frozen experiment protocol is missing")
        return None
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(document, dict):
            raise ProtocolError("Experiment protocol must be an object")
        if not declared and document.get("schema_version") != 2:
            return None  # P0 legacy required_keys remains handled by final acceptance.
        if not declared or "experiment_protocol.json" not in contract["outputs"]:
            raise ProtocolError("Protocol v2 must be frozen in the research input contract")
        rebuilt = compile_protocol(document.get("spec"), contract["datasets"],
                                   max_cell_seconds=contract["brief"]["max_experiment_seconds"])
        if rebuilt != document:
            raise ProtocolError("Frozen experiment matrix or version changed")
        return document
    except (OSError, TypeError, KeyError, ValueError) as exc:
        if isinstance(exc, ProtocolError):
            raise
        raise ProtocolError(f"Invalid experiment protocol: {type(exc).__name__}") from exc


def check_execution_binding(protocol: dict, key: EvidenceKey, execution: dict) -> None:
    cell_id = content_hash(asdict(key))
    if cell_id not in {c["cell_id"] for c in protocol["cells"]}:
        raise ProtocolError("Evaluation key is outside the frozen experiment matrix")
    if (execution.get("protocol_version") != protocol["version"]
            or execution.get("cell_id") != cell_id or execution.get("key") != asdict(key)):
        raise ProtocolError("Execution does not bind the declared protocol cell")


def audit_coverage(root: Path, protocol: dict, store: EvidenceStore) -> dict:
    host_error, host_runs = None, None
    if any((root / name).exists() for name in ("protocol_code.json", "protocol_execution.jsonl", "protocol_budget.json")):
        from researchclaw.experiment.protocol_runner import verify_execution_bundle
        try:
            host_runs = verify_execution_bundle(root, protocol)["success"]
        except ProtocolError as exc:
            host_error = str(exc)
    available = {content_hash(asdict(r.key)): rid for rid, r in store.records.items()}
    required = {c["cell_id"] for c in protocol["cells"]}
    complete, missing, invalid = [], [], []
    for cell in protocol["cells"]:
        cid = cell["cell_id"]
        rid = available.get(cid)
        if rid is None:
            missing.append({"cell_id": cid, "key": cell["key"]})
            continue
        errors = store.validate_record(rid, root)
        record = store.records[rid]
        # A record must carry the execution receipt, not merely a matching metric.
        receipts = [name for name, digest in record.artifacts if digest == record.execution]
        bound = False
        for name in receipts:
            path = (root / name).resolve()
            if not path.is_relative_to(root.resolve()):
                continue
            try:
                execution = json.loads(path.read_text(encoding="utf-8"))
                check_execution_binding(protocol, record.key, execution)
                if execution.get("recorder") == "host_protocol_runner/v1":
                    if host_runs is None or host_runs.get(cid) != execution:
                        continue
                if execution.get("status") == "success" and type(execution.get("returncode")) is int and execution["returncode"] == 0:
                    bound = True
            except (OSError, ValueError, TypeError, AttributeError):
                continue
        if not bound:
            errors.append("missing_or_unbound_execution_receipt")
        if errors:
            invalid.append({"cell_id": cid, "key": cell["key"], "errors": errors})
        else:
            complete.append(cid)
    unexpected = sorted(available.keys() - required)
    if host_error:
        invalid.append({"errors": [host_error], "artifact": "protocol_execution.jsonl"})
    # Use only complete, validated pairs; never pool different datasets or RQs.
    cell_index = {c["cell_id"]: c for c in protocol["cells"]}
    completed = set(complete)
    groups = {}
    for comparison in protocol["comparisons"]:
        base_id, candidate_id = comparison["baseline_cell"], comparison["candidate_cell"]
        base_key, candidate_key = cell_index[base_id]["key"], cell_index[candidate_id]["key"]
        group_key = (comparison["question"], base_key["dataset"], base_key["metric"],
                     base_key["method"], candidate_key["method"])
        group = groups.setdefault(group_key, {
            "question": group_key[0], "dataset": group_key[1], "metric": group_key[2],
            "baseline": group_key[3], "candidate": group_key[4],
            "difference": "candidate_minus_baseline", "resampling_unit": "training_seed",
            "expected_pairs": 0, "pairs": []})
        group["expected_pairs"] += 1
        if base_id in completed and candidate_id in completed:
            baseline_result, candidate_result = available[base_id], available[candidate_id]
            left, right = store.records[baseline_result], store.records[candidate_result]
            # Units must be identical; a percent value cannot silently become a fraction.
            if left.unit != right.unit or any(p["unit"] != left.unit for p in group["pairs"]):
                invalid.append({"cell_id": candidate_id, "key": candidate_key,
                                "errors": ["comparison_unit_mismatch"]})
                continue
            group["pairs"].append({"seed": base_key["seed"], "baseline_result": baseline_result,
                                   "candidate_result": candidate_result, "unit": left.unit,
                                   "difference": right.value - left.value})
    for group in groups.values():
        group["status"] = "complete" if len(group["pairs"]) == group["expected_pairs"] else "incomplete"
        # No summary over a selected subset of seeds.
        group["mean_difference"] = (statistics.mean(p["difference"] for p in group["pairs"])
                                    if group["status"] == "complete" else None)
    return {"schema_version": 1, "checker": "experiment_protocol/v2",
            "protocol_version": protocol["version"], "evidence_version": store.version,
            "status": "complete" if not (missing or invalid or unexpected) else "incomplete",
            "required_count": len(required), "completed_count": len(complete),
            "missing": missing, "invalid": invalid, "unexpected": unexpected,
            "comparisons": list(groups.values()),
            "allocation": protocol["allocation"],
            "limitations": ["Allocation is not measured resource use or proof of fair tuning.",
                            "Multiple seeds are not independent datasets or subjects.",
                            "Ablation configuration declarations do not prove implementation removal."]}


def protocol_context(protocol: dict) -> str:
    return ("\n## Frozen experiment protocol (binding)\n"
            "Complete every required cell; never trim methods, datasets or seeds. "
            "Use the explicit comparison baselines and validation-only model selection. "
            "Keep all failed attempts; never select the best test seed. "
            "main.py must handle ARC_PROTOCOL_REQUEST (a JSON environment variable): execute only its "
            "method/dataset/seed, use request.dataset.paths, and write request.output as id,prediction CSV. "
            "When request.learning_curve is present, also write its output path as exact step,value CSV using "
            "strictly increasing nonnegative integer steps, no more than max_points rows, and the declared split/metric. "
            "The host runs every cell and writes execution receipts; do not fabricate receipts or run the "
            "whole matrix inside one invocation. If this environment variable is absent during a code "
            "generation trial, use validation only, never export formal test results. "
            "Execution receipts must include protocol_version, cell_id and the exact key. "
            "Budget allocations are equal caps for each cell including validation tuning; "
            "they do not certify actual use. Report any unexecuted cells as missing.\n"
            + json.dumps(protocol, ensure_ascii=False, indent=2))


def apply_protocol_to_plan(plan: dict, protocol: dict) -> None:
    """Apply after model/HITL rewrites so frozen conditions cannot be trimmed."""
    methods = protocol["spec"]["methods"]
    for field, role in (("baselines", "baseline"), ("proposed_methods", "proposed"), ("ablations", "ablation")):
        plan[field] = [dict(m, name=m["id"]) for m in methods if m["role"] == role]
    plan["datasets"] = sorted({c["key"]["dataset"] for c in protocol["cells"]})
    plan["metrics"] = sorted({c["key"]["metric"] for c in protocol["cells"]})
    plan["objectives"] = [q["question"] for q in protocol["spec"]["questions"]]
    plan["seeds"] = protocol["spec"]["seeds"]
    plan["experiment_protocol_version"] = protocol["version"]
    plan["required_experiments"] = protocol["cells"]
    plan["comparisons"] = protocol["comparisons"]
    plan["compute_budget"] = protocol["spec"]["budget"]
