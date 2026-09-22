"""Host-side metrics computed from predictions and frozen ground-truth labels.

This evaluator never imports experiment code. Labels must be pinned before
experimentation; a generated script's reported metrics are not consumed.
CSV schemas: labels = id,label; predictions = id,prediction. Files must cover
the same unique IDs. AUROC uses binary labels and continuous scores.
"""
from __future__ import annotations

import csv
import json
import math
import platform
from pathlib import Path

from researchclaw.pipeline.evidence_store import EvidenceKey, EvidenceRecord, content_hash, file_hash


def _read_values(path: Path, column: str) -> dict[str, float]:
    values = {}
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            identity = row.get("id", "")
            if not identity or identity in values:
                raise ValueError("Missing or duplicate sample ID")
            value = float(row[column])
            if not math.isfinite(value):
                raise ValueError("Non-finite label or prediction")
            values[identity] = value
    if not values:
        raise ValueError("Empty evaluation data")
    return values


def evaluate_predictions(*, root: Path, key: EvidenceKey, labels: str, predictions: str,
                         expected_labels_sha256: str, execution: str) -> EvidenceRecord:
    """Return reproducible evidence from a successful execution, or reject it.

    execution JSON must contain status=success, returncode=0, code_commit,
    and environment. The label hash is an input contract, not inferred from
    model-generated outputs. Dataset/split provenance remains the caller's
    responsibility and is a separate final-acceptance dimension.
    """
    paths = [(root / name).resolve() for name in (labels, predictions, execution)]
    if not all(p.is_relative_to(root.resolve()) and p.is_file() for p in paths):
        raise ValueError("Evaluation paths must be files inside the evidence bundle")
    label_path, prediction_path, execution_path = paths
    if (root / "research_contract.json").is_file():
        from researchclaw.research_inputs import check_evaluation_binding, verify_bundle_contract
        check_evaluation_binding(verify_bundle_contract(root), key, labels, expected_labels_sha256)
    if file_hash(label_path) != expected_labels_sha256:
        raise ValueError("Frozen labels changed")
    run = json.loads(execution_path.read_text(encoding="utf-8"))
    from researchclaw.pipeline.experiment_protocol import check_execution_binding, load_protocol
    protocol = load_protocol(root)
    if protocol is not None:
        check_execution_binding(protocol, key, run)
    if run.get("status") != "success" or type(run.get("returncode")) is not int or run["returncode"] != 0:
        raise ValueError("Cannot evaluate a failed or incomplete run")
    if not run.get("code_commit") or not run.get("environment"):
        raise ValueError("Execution provenance is incomplete")
    truth = _read_values(label_path, "label")
    output = _read_values(prediction_path, "prediction")
    if truth.keys() != output.keys():
        raise ValueError("Predictions do not cover exactly the frozen evaluation split")
    ids = sorted(truth)
    pairs = [(truth[i], output[i]) for i in ids]
    unit = "raw"
    if key.metric == "accuracy":
        if any(not y.is_integer() or not p.is_integer() for y, p in pairs):
            raise ValueError("Accuracy requires discrete class labels, not scores")
        value = sum(y == p for y, p in pairs) / len(pairs)
        unit = "fraction"
    elif key.metric == "mse":
        value = math.fsum((y - p) ** 2 for y, p in pairs) / len(pairs)
    elif key.metric == "mae":
        value = math.fsum(abs(y - p) for y, p in pairs) / len(pairs)
    elif key.metric == "auroc":
        if {y for y, _ in pairs} != {0.0, 1.0}:
            raise ValueError("Binary AUROC requires both binary classes")
        ordered = sorted(pairs, key=lambda pair: pair[1])
        rank_sum, index = 0.0, 0
        while index < len(ordered):
            end = index + 1
            while end < len(ordered) and ordered[end][1] == ordered[index][1]:
                end += 1
            rank_sum += ((index + 1 + end) / 2) * sum(y for y, _ in ordered[index:end])
            index = end
        positives = sum(y for y, _ in pairs)
        negatives = len(pairs) - positives
        value = (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)
        unit = "fraction"
    else:
        raise ValueError(f"Unsupported trusted metric: {key.metric}")
    if key.aggregation != "per_run":
        raise ValueError("Prediction evaluator only produces per_run metrics")
    return EvidenceRecord(
        key=key, value=value, unit=unit, run_status="success", code_commit=run["code_commit"],
        environment=content_hash({"execution": run["environment"], "python": platform.python_version()}),
        execution=file_hash(execution_path), evaluator="host_predictions/v1:" + file_hash(Path(__file__)),
        evaluator_independent=True,
        artifacts=tuple((p.relative_to(root.resolve()).as_posix(), file_hash(p)) for p in paths),
    )


def evaluate_manifest(root: Path, manifest: dict) -> dict:
    """Import a predeclared evaluation manifest into an EvidenceStore document."""
    from researchclaw.pipeline.evidence_store import EvidenceStore
    store = EvidenceStore()
    if manifest.get("schema_version") != 1 or not manifest.get("runs"):
        raise ValueError("Evaluation manifest v1 must declare runs")
    for run in manifest["runs"]:
        spec = dict(run)
        for field in ("labels", "predictions", "execution"):
            if not spec[field].replace("\\", "/").startswith("evidence_artifacts/"):
                raise ValueError("Manifest outputs must be bundled under evidence_artifacts/")
        spec["key"] = EvidenceKey(**spec["key"])
        store.add(evaluate_predictions(root=root, **spec))
    return store.to_dict()
