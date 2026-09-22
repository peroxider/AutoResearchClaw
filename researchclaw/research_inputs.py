"""Frozen research briefs and local tabular data contracts.

Preflight uses labels for deterministic splitting and schema validation only.
Exploratory summaries are computed on the training partition. Holding labels
out of prompts is logical isolation; filesystem access control is the executor's
responsibility. No raw data is fetched from the network by this module.
"""
from __future__ import annotations

import csv
import io
import json
import math
import re
import shutil
import statistics
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from researchclaw.pipeline.evidence_store import content_hash, file_hash


class InputContractError(ValueError):
    """Invalid, missing, or changed research input; never silently repaired."""


def _fields(data: dict, allowed: set[str], name: str) -> None:
    if not isinstance(data, dict):
        raise InputContractError(f"{name} must be an object")
    unknown = data.keys() - allowed
    if unknown:
        raise InputContractError(f"Unknown {name} fields: {', '.join(sorted(unknown))}")


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InputContractError(f"{name} must be a nonempty string")
    return value


def _strings(value: Any, name: str, *, required: bool = False) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or (required and not value):
        raise InputContractError(f"{name} must be {'a nonempty' if required else 'a'} list")
    result = tuple(_text(v, name) for v in value)
    if len(result) != len(set(result)):
        raise InputContractError(f"{name} contains duplicates")
    return result


def _load(path: Path) -> dict:
    try:
        text = path.read_text(encoding="utf-8-sig")
        # YAML 1.1 interprets JSON exponent literals such as 1e-06 as strings.
        value = json.loads(text) if path.suffix.lower() == ".json" else yaml.safe_load(text)
    except (OSError, UnicodeError, yaml.YAMLError, json.JSONDecodeError) as exc:
        raise InputContractError(f"Cannot read input file {path.name}: {type(exc).__name__}") from exc
    if not isinstance(value, dict):
        raise InputContractError(f"{path.name} must contain an object")
    return value


@dataclass(frozen=True)
class SplitSpec:
    strategy: str = "stratified"
    seed: int = 42
    train_fraction: float = 0.6
    validation_fraction: float = 0.2

    @classmethod
    def from_dict(cls, data: dict) -> SplitSpec:
        _fields(data, set(cls.__dataclass_fields__), "split")
        result = cls(**data)
        if result.strategy not in {"random", "stratified", "group", "time"}:
            raise InputContractError("Unsupported split strategy")
        if type(result.seed) is not int:
            raise InputContractError("split.seed must be an integer")
        fractions = (result.train_fraction, result.validation_fraction)
        if any(type(v) not in (int, float) or not math.isfinite(v) or not 0 < v < 1 for v in fractions):
            raise InputContractError("Split fractions must be finite and strictly between 0 and 1")
        if sum(fractions) >= 1:
            raise InputContractError("A nonempty test fraction must remain")
        return result


@dataclass(frozen=True)
class DatasetManifest:
    dataset: str
    version: str
    path: str
    task: str
    id_column: str
    label_column: str
    features: tuple[str, ...]
    metric: str
    source: str
    license: str
    forbidden_features: tuple[str, ...] = ()
    group_column: str = ""
    time_column: str = ""
    split: SplitSpec = field(default_factory=SplitSpec)
    schema_version: int = 1

    @classmethod
    def from_dict(cls, data: dict) -> DatasetManifest:
        _fields(data, set(cls.__dataclass_fields__), "dataset manifest")
        try:
            values = dict(data)
            values["features"] = _strings(values.get("features"), "features", required=True)
            values["forbidden_features"] = _strings(values.get("forbidden_features", []), "forbidden_features")
            values["split"] = SplitSpec.from_dict(values.get("split", {}))
            result = cls(**values)
        except TypeError as exc:
            raise InputContractError("Dataset manifest is missing required fields") from exc
        for name in ("dataset", "version", "path", "task", "id_column", "label_column", "metric", "source", "license"):
            _text(getattr(result, name), name)
        if type(result.schema_version) is not int or result.schema_version != 1:
            raise InputContractError("Unsupported dataset manifest version")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", result.dataset):
            raise InputContractError("dataset must be a safe alphanumeric identifier")
        if result.dataset.upper() in {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}:
            raise InputContractError("Reserved dataset identifier")
        metrics = {"classification": {"accuracy", "auroc"}, "regression": {"mse", "mae"}}
        if result.task not in metrics or result.metric not in metrics[result.task]:
            raise InputContractError("Task and trusted metric are incompatible")
        for name in ("group_column", "time_column"):
            if not isinstance(getattr(result, name), str):
                raise InputContractError(f"{name} must be a string")
        metadata = [result.id_column, result.label_column, result.group_column, result.time_column]
        specified = [name for name in metadata if name]
        if len(specified) != len(set(specified)):
            raise InputContractError("ID, label, group and time columns must be distinct")
        if set(result.features) & (set(metadata) | set(result.forbidden_features)):
            raise InputContractError("Features include a label, identifier, split field or forbidden feature")
        if result.group_column and result.split.strategy not in {"group", "time"}:
            raise InputContractError("Entity data requires group or time splitting")
        if result.time_column and result.split.strategy != "time":
            raise InputContractError("Temporal data requires time splitting")
        if result.split.strategy == "group" and not result.group_column:
            raise InputContractError("Group splitting requires group_column")
        if result.split.strategy == "time" and not result.time_column:
            raise InputContractError("Time splitting requires time_column")
        if result.split.strategy == "stratified" and result.task != "classification":
            raise InputContractError("Stratified splitting is supported for classification only")
        return result


@dataclass(frozen=True)
class ResearchBrief:
    question: str
    ideas: tuple[str, ...]
    constraints: tuple[str, ...]
    datasets: tuple[str, ...]
    paper_type: str = "empirical"
    hypotheses: tuple[str, ...] = ()
    allow_external_data: bool = False
    max_experiment_seconds: int = 300
    protocol_path: str = ""
    method_spec_path: str = ""
    theory_bundle_path: str = ""
    schema_version: int = 1

    @classmethod
    def from_dict(cls, data: dict) -> ResearchBrief:
        _fields(data, set(cls.__dataclass_fields__), "research brief")
        try:
            values = dict(data)
            for name in ("ideas", "constraints", "datasets", "hypotheses"):
                values[name] = _strings(values.get(name, []), name, required=name in {"ideas", "datasets"})
            result = cls(**values)
        except TypeError as exc:
            raise InputContractError("Research brief is missing required fields") from exc
        _text(result.question, "question")
        if type(result.schema_version) is not int or result.schema_version != 1:
            raise InputContractError("Unsupported brief schema version")
        if result.paper_type not in {"empirical", "theoretical", "mixed"}:
            raise InputContractError("Unknown paper_type")
        if type(result.allow_external_data) is not bool:
            raise InputContractError("allow_external_data must be boolean")
        if type(result.max_experiment_seconds) is not int or result.max_experiment_seconds < 1:
            raise InputContractError("max_experiment_seconds must be a positive integer")
        if not isinstance(result.protocol_path, str) or (result.protocol_path and not result.protocol_path.strip()):
            raise InputContractError("protocol_path must be a path string")
        for name in ("method_spec_path", "theory_bundle_path"):
            value = getattr(result, name)
            if not isinstance(value, str) or (value and not value.strip()):
                raise InputContractError(f"{name} must be a path string")
        if result.method_spec_path and not result.protocol_path:
            raise InputContractError("MethodSpec code binding requires a frozen experiment protocol")
        return result


def _read_rows(path: Path, manifest: DatasetManifest) -> list[dict[str, str]]:
    try:
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            columns = reader.fieldnames or []
            if not columns or len(columns) != len(set(columns)) or any(not c.strip() for c in columns):
                raise InputContractError("CSV header is empty or contains duplicate/blank columns")
            needed = {manifest.id_column, manifest.label_column, *manifest.features,
                      *manifest.forbidden_features, manifest.group_column, manifest.time_column} - {""}
            if not needed.issubset(columns):
                raise InputContractError(f"Missing columns: {', '.join(sorted(needed - set(columns)))}")
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise InputContractError(f"Cannot read CSV {path.name}: {type(exc).__name__}") from exc
    ids = set()
    for row in rows:
        if None in row or any(v is None for v in row.values()):
            raise InputContractError("CSV row width does not match header")
        identity = row[manifest.id_column]
        if not identity.strip() or identity in ids:
            raise InputContractError("Missing or duplicate row ID")
        ids.add(identity)
        for name in (manifest.label_column, manifest.group_column, manifest.time_column):
            if name and not row[name].strip():
                raise InputContractError(f"Missing value in required column {name}")
        if manifest.task == "regression":
            try:
                value = float(row[manifest.label_column])
                if not math.isfinite(value):
                    raise ValueError
            except ValueError as exc:
                raise InputContractError("Regression labels must be finite numbers") from exc
    if len(rows) < 3:
        raise InputContractError("At least three records are required for train/validation/test")
    return rows


def _partition(items: list[Any], spec: SplitSpec) -> tuple[list, list, list]:
    n = len(items)
    if n < 3:
        raise InputContractError("Each split unit/stratum needs at least three members")
    a = max(1, min(n - 2, int(n * spec.train_fraction)))
    b = max(a + 1, min(n - 1, a + int(n * spec.validation_fraction)))
    return items[:a], items[a:b], items[b:]


def _timestamp(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        # Dates and naive timestamps are interpreted as UTC, explicitly.
        return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)
    except ValueError as exc:
        raise InputContractError("Time splitting requires ISO-8601 dates/timestamps") from exc


def split_rows(rows: list[dict[str, str]], manifest: DatasetManifest) -> dict[str, list[dict[str, str]]]:
    spec = manifest.split
    result = {"train": [], "validation": [], "test": []}

    def order(value: str) -> str:
        return content_hash([spec.seed, value])

    if spec.strategy == "stratified":
        classes: dict[str, list] = {}
        for row in rows:
            classes.setdefault(row[manifest.label_column], []).append(row)
        for label in sorted(classes):
            parts = _partition(sorted(classes[label], key=lambda r: order(r[manifest.id_column])), spec)
            for name, part in zip(result, parts):
                result[name].extend(part)
    elif spec.strategy in {"group", "time"}:
        units: dict[Any, list] = {}
        for row in rows:
            unit = row[manifest.group_column] if spec.strategy == "group" else _timestamp(row[manifest.time_column])
            units.setdefault(unit, []).append(row)
        ordered = sorted(units, key=order) if spec.strategy == "group" else sorted(units)
        for name, part in zip(result, _partition(ordered, spec)):
            result[name] = [row for unit in part for row in units[unit]]
    else:
        for name, part in zip(result, _partition(sorted(rows, key=lambda r: order(r[manifest.id_column])), spec)):
            result[name] = part
    if manifest.group_column:
        memberships = [set(r[manifest.group_column] for r in part) for part in result.values()]
        if any(memberships[i] & memberships[j] for i in range(3) for j in range(i + 1, 3)):
            raise InputContractError("Entity overlap across splits; use a compatible group/time protocol")
    return {name: sorted(part, key=lambda r: r[manifest.id_column]) for name, part in result.items()}


def _train_profile(rows: list[dict[str, str]], manifest: DatasetManifest) -> dict:
    features = {}
    labels = [r[manifest.label_column] for r in rows]
    for name in manifest.features:
        values = [r[name] for r in rows]
        present = [v for v in values if v.strip()]
        if values == labels and len(set(labels)) > 1:
            raise InputContractError(f"Feature {name} copies the target on the training split")
        profile: dict[str, Any] = {"missing": len(values) - len(present), "distinct": len(set(present))}
        try:
            numeric = [float(v) for v in present]
        except ValueError:
            numeric = []
        if numeric:
            if not all(math.isfinite(v) for v in numeric):
                raise InputContractError(f"Non-finite training feature: {name}")
            try:
                numeric_labels = [float(v) for v in labels]
            except ValueError:
                numeric_labels = []
            if numeric == numeric_labels and len(set(numeric_labels)) > 1:
                raise InputContractError(f"Feature {name} copies the numeric target on the training split")
            profile.update(type="numeric", minimum=min(numeric), maximum=max(numeric),
                           mean=statistics.mean(numeric))
        else:
            profile["type"] = "categorical" if present else "empty"
        features[name] = profile
    return {"scope": "train_only", "rows": len(rows), "features": features,
            "labels": dict(sorted(Counter(labels).items())) if manifest.task == "classification" else
            {"mean": statistics.mean(float(v) for v in labels)}}


def _csv(columns: list[str], rows: list[dict]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n", extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def prepare_inputs(brief_path: Path, run_dir: Path, *, runtime: dict | None = None) -> dict:
    """Validate everything before exporting, then freeze the contract last."""
    brief_path, run_dir = brief_path.resolve(), run_dir.resolve()
    if (run_dir / "research_contract.json").exists():
        contract = verify_inputs(brief_path, run_dir)
        if runtime is not None and contract.get("runtime") != runtime:
            raise InputContractError("Runtime constraints changed; start a new research run")
        return contract
    brief = ResearchBrief.from_dict(_load(brief_path))
    sources = {str(brief_path): file_hash(brief_path)}
    outputs: dict[str, str] = {}
    datasets = []
    seen = set()
    for manifest_name in brief.datasets:
        manifest_path = (brief_path.parent / manifest_name).resolve()
        manifest = DatasetManifest.from_dict(_load(manifest_path))
        if manifest.dataset.casefold() in seen:
            raise InputContractError("Dataset identifiers must be unique (case insensitive)")
        seen.add(manifest.dataset.casefold())
        path = (manifest_path.parent / manifest.path).resolve()
        if path.suffix.lower() != ".csv":
            raise InputContractError("This intake supports local CSV files only")
        sources[str(manifest_path)] = file_hash(manifest_path)
        sources[str(path)] = file_hash(path)
        rows = _read_rows(path, manifest)
        parts = split_rows(rows, manifest)
        profile = _train_profile(parts["train"], manifest)
        classes = sorted({r[manifest.label_column] for r in parts["train"]})
        encoding = {label: i for i, label in enumerate(classes)} if manifest.task == "classification" else {}
        if manifest.task == "classification":
            if len(classes) < 2 or any(r[manifest.label_column] not in encoding for r in rows):
                raise InputContractError("Training split must cover all label classes and contain at least two")
            if manifest.metric == "auroc" and (len(classes) != 2 or any(
                len({r[manifest.label_column] for r in part}) != 2 for part in parts.values()
            )):
                raise InputContractError("Binary AUROC requires both classes in every split")
        base = f"research_inputs/{manifest.dataset}"
        for name in ("train", "validation"):
            outputs[f"{base}/{name}.csv"] = _csv(
                [manifest.id_column, *manifest.features, manifest.label_column], parts[name])
        outputs[f"{base}/test_features.csv"] = _csv([manifest.id_column, *manifest.features], parts["test"])
        label_path = f"evidence_artifacts/{manifest.dataset}/test_labels.csv"
        outputs[label_path] = _csv(["id", "label"], [
            {"id": r[manifest.id_column], "label": encoding[r[manifest.label_column]] if encoding
             else r[manifest.label_column]} for r in parts["test"]])
        split_ids = {name: [r[manifest.id_column] for r in part] for name, part in parts.items()}
        outputs[f"{base}/split_ids.json"] = json.dumps(split_ids, indent=2)
        card = {"dataset": manifest.dataset, "version": manifest.version, "source": manifest.source,
                "license": manifest.license, "source_sha256": sources[str(path)],
                "task": manifest.task, "metric": manifest.metric, "features": manifest.features,
                "id_column": manifest.id_column, "label_column": manifest.label_column,
                "forbidden_features": manifest.forbidden_features, "label_encoding": encoding,
                "split": asdict(manifest.split), "split_sizes": {k: len(v) for k, v in parts.items()},
                "split_ids_sha256": content_hash(split_ids), "training_profile": profile,
                "preprocessing_fit_scope": "train_only",
                "paths": {name: f"{base}/{name}.csv" for name in ("train", "validation", "test_features")}}
        # Equal measurements from distinct subjects can be genuine repeats.
        # Report potential duplication; do not infer a data defect from values alone.
        signatures = {tuple(r[c] for c in (*manifest.features, manifest.label_column)) for r in rows}
        card["duplicate_feature_label_rows"] = len(rows) - len(signatures)
        card["warnings"] = (["Repeated feature/label rows require provenance review"]
                            if len(signatures) < len(rows) else [])
        outputs[f"{base}/dataset_card.json"] = json.dumps(card, indent=2)
        datasets.append({"manifest": asdict(manifest), "card": card, "labels": label_path})
    if brief.protocol_path:
        from researchclaw.pipeline.experiment_protocol import compile_protocol
        protocol_path = (brief_path.parent / brief.protocol_path).resolve()
        sources[str(protocol_path)] = file_hash(protocol_path)
        protocol = compile_protocol(_load(protocol_path), datasets,
                                    max_cell_seconds=brief.max_experiment_seconds)
        outputs["experiment_protocol.json"] = json.dumps(protocol, indent=2)
    from researchclaw.pipeline.research_workbench import compile_method, compile_theory
    for field_name, output_name, compiler in (
        ("method_spec_path", "method_spec.json", compile_method),
        ("theory_bundle_path", "theory_bundle.json", compile_theory),
    ):
        if getattr(brief, field_name):
            source = (brief_path.parent / getattr(brief, field_name)).resolve()
            sources[str(source)] = file_hash(source)
            document = compiler(_load(source))
            if output_name == "method_spec.json" and document["spec"]["method_id"] not in {
                m["id"] for m in protocol["spec"]["methods"]
            }:
                raise InputContractError("MethodSpec must reference a declared experiment method")
            outputs[output_name] = json.dumps(document, indent=2)
    # Avoid accepting a file that changed while it was being inspected.
    if any(file_hash(Path(path)) != digest for path, digest in sources.items()):
        raise InputContractError("Input changed during preflight")
    run_dir.mkdir(parents=True, exist_ok=True)
    for name, body in outputs.items():
        dest = run_dir / name
        if not dest.resolve().is_relative_to(run_dir):
            raise InputContractError("Prepared output resolves outside the run directory")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(body, encoding="utf-8", newline="")
    contract = {"schema_version": 1, "brief": asdict(brief), "source_files": sources,
                "runtime": runtime or {},
                "datasets": datasets, "outputs": {name: file_hash(run_dir / name) for name in outputs}}
    contract = json.loads(json.dumps(contract))  # One JSON shape on fresh runs and resume.
    contract["version"] = content_hash(contract)
    path = run_dir / "research_contract.json"
    staging = path.with_suffix(".json.tmp")
    staging.write_text(json.dumps(contract, indent=2), encoding="utf-8")
    staging.replace(path)
    return contract


def verify_bundle_contract(run_dir: Path) -> dict:
    contract_path = run_dir / "research_contract.json"
    contract = _load(contract_path)
    version = contract.pop("version", None)
    if version != content_hash(contract) or contract.get("schema_version") != 1:
        raise InputContractError("Research contract integrity mismatch")
    for field in ("source_files", "outputs"):
        entries = contract.get(field)
        if not isinstance(entries, dict) or not entries or any(
            not isinstance(name, str) or not isinstance(digest, str)
            or not re.fullmatch(r"[0-9a-f]{64}", digest) for name, digest in entries.items()
        ):
            raise InputContractError(f"Invalid contract {field}")
    ResearchBrief.from_dict(contract.get("brief"))
    if not isinstance(contract.get("datasets"), list) or not contract["datasets"]:
        raise InputContractError("Contract has no dataset manifests")
    for dataset in contract["datasets"]:
        if not isinstance(dataset, dict) or not isinstance(dataset.get("card"), dict):
            raise InputContractError("Invalid dataset card in contract")
        DatasetManifest.from_dict(dataset.get("manifest"))
        if dataset.get("labels") not in contract["outputs"]:
            raise InputContractError("Frozen labels missing from output inventory")
    for name, digest in contract["outputs"].items():
        path = (run_dir / name).resolve()
        if not path.is_relative_to(run_dir.resolve()) or not path.is_file() or file_hash(path) != digest:
            raise InputContractError(f"Prepared dataset changed or missing: {name}")
    if contract["brief"].get("protocol_path"):
        from researchclaw.pipeline.experiment_protocol import load_protocol
        load_protocol(run_dir, contract)
    from researchclaw.pipeline.research_workbench import verify_workbench
    verify_workbench(run_dir, contract)
    contract["version"] = version
    return contract


def verify_inputs(brief_path: Path, run_dir: Path) -> dict:
    contract = verify_bundle_contract(run_dir)
    if str(brief_path.resolve()) not in contract.get("source_files", {}):
        raise InputContractError("Research brief differs from the frozen contract")
    for name, digest in contract["source_files"].items():
        if not Path(name).is_file() or file_hash(Path(name)) != digest:
            raise InputContractError(f"Frozen source changed or missing: {Path(name).name}")
    return contract


def check_evaluation_binding(contract: dict, key: Any, labels: str, labels_hash: str) -> None:
    dataset = next((d for d in contract["datasets"] if d["manifest"]["dataset"] == key.dataset), None)
    if dataset is None:
        raise InputContractError("Evaluation uses an undeclared dataset")
    if key.dataset_version != dataset["manifest"]["version"] or key.metric != dataset["manifest"]["metric"]:
        raise InputContractError("Evaluation version or metric differs from the frozen dataset")
    if (key.split != "test" or labels.replace("\\", "/") != dataset["labels"]
            or labels_hash != contract["outputs"][dataset["labels"]]):
        raise InputContractError("Evaluation must use the frozen test split and label hash")


def public_context(contract: dict, run_dir: Path) -> str:
    """No full-data paths or held-out labels are included in model context."""
    brief = contract["brief"]
    public = {name: brief[name] for name in ("question", "ideas", "constraints", "hypotheses",
              "paper_type", "allow_external_data", "max_experiment_seconds")}
    public["datasets"] = []
    for dataset in contract["datasets"]:
        card = dict(dataset["card"])
        card["paths"] = {name: f"research_data/{card['dataset']}/{Path(path).name}"
                         for name, path in card["paths"].items()}
        public["datasets"].append(card)
    protocol_text = ""
    if brief.get("protocol_path"):
        from researchclaw.pipeline.experiment_protocol import load_protocol, protocol_context
        protocol_text = protocol_context(load_protocol(run_dir, contract))
    for field_name, output_name in (("method_spec_path", "method_spec.json"), ("theory_bundle_path", "theory_bundle.json")):
        if brief.get(field_name):
            protocol_text += (f"\n## Frozen {output_name}\nUse the declared IDs and proof statuses. "
                              "Structural mappings are not semantic proofs.\n"
                              + (run_dir / output_name).read_text(encoding="utf-8"))
    return ("\n## Frozen research inputs (binding)\n"
            "Use only the declared features, split IDs and metrics. Fit preprocessing on train only. "
            "Select models using validation only; export test predictions without inspecting test labels. "
            "Dataset paths are relative to the experiment project; the sandbox stages these files. "
            "Prediction CSV columns must be id,prediction; classification predictions use label_encoding. "
            "Do not replace datasets or reinterpret constraints to obtain a better result.\n"
            + json.dumps(public, ensure_ascii=False, indent=2) + protocol_text)


def contract_for_config(config: Any, run_dir: Path, *, initialize: bool = False) -> dict | None:
    brief = getattr(config.research, "brief_path", "")
    frozen = (run_dir / "research_contract.json").exists()
    if not brief:
        if frozen:
            raise InputContractError("Cannot remove research.brief_path from an initialized run")
        return None
    runtime = {"topic": config.research.topic, "metric_key": config.experiment.metric_key,
               "metric_direction": config.experiment.metric_direction,
               "time_budget_sec": config.experiment.time_budget_sec}
    try:
        if config.experiment.mode not in {"sandbox", "docker"}:
            raise InputContractError("Structured CSV intake currently supports sandbox and docker execution")
        declared = ResearchBrief.from_dict(_load(Path(brief)))
        if config.experiment.time_budget_sec > declared.max_experiment_seconds:
            raise InputContractError("Experiment time budget exceeds ResearchBrief constraint")
        if initialize:
            contract = prepare_inputs(Path(brief), run_dir, runtime=runtime)
        else:
            contract = verify_inputs(Path(brief), run_dir)
        if contract.get("runtime") != runtime:
            raise InputContractError("Runtime constraints changed; start a new research run")
        metrics = {d["manifest"]["metric"] for d in contract["datasets"]}
        if config.experiment.metric_key != "primary_metric" and config.experiment.metric_key not in metrics:
            raise InputContractError("Configured metric is not declared in DatasetManifest")
        directions = {"minimize" if m in {"mse", "mae"} else "maximize" for m in metrics}
        if len(directions) != 1 or config.experiment.metric_direction not in directions:
            raise InputContractError("Metric direction conflicts with the dataset task")
        return contract
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        if isinstance(exc, InputContractError):
            raise
        raise InputContractError(f"Invalid research input: {type(exc).__name__}") from exc


def bind_project_data(project_dir: Path) -> None:
    """Stage public splits into an experiment before sandbox copying.

    Called by the sandbox boundary, so initial execution, CodeAgent trials,
    refinement and automatic repairs all receive the same frozen inputs.
    """
    project_dir = project_dir.resolve()
    root = next((p for p in project_dir.parents if (p / "research_contract.json").is_file()), None)
    if root is None:
        return
    contract = verify_bundle_contract(root)
    # Also check original inputs while the run is on its producing host.
    for name, digest in contract["source_files"].items():
        if not Path(name).is_file() or file_hash(Path(name)) != digest:
            raise InputContractError(f"Frozen source changed or missing: {Path(name).name}")
    files = {}
    for dataset in contract["datasets"]:
        card = dataset["card"]
        for name, source in card["paths"].items():
            files[f"research_data/{card['dataset']}/{name}.csv"] = root / source
    data_dir = project_dir / "research_data"
    if data_dir.is_symlink():
        raise InputContractError("Frozen research_data cannot be a symlink")
    if data_dir.exists():
        for path in data_dir.rglob("*"):
            if path.is_symlink() or (path.is_file() and path.relative_to(project_dir).as_posix() not in files):
                raise InputContractError("Unexpected file or symlink in the frozen research_data namespace")
    for name, source in files.items():
        destination = project_dir / name
        if not destination.resolve().is_relative_to(project_dir):
            raise InputContractError("Staged data path escapes the experiment project")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
