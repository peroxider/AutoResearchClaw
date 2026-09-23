"""Frozen analysis intentions, exact result bindings and bounded computation.

Training seeds on one test split are not independent subjects or datasets.
An optional bootstrap is conditional on a declared exchangeability assumption;
neither that assumption nor scientific adequacy is certified by this checker.
"""
from __future__ import annotations

import copy
import json
import math
import platform
import random
import re
import statistics
from dataclasses import asdict
from pathlib import Path
from textwrap import fill

from researchclaw.literature.evidence import write_json
from researchclaw.pipeline.evidence_store import EvidenceStore, content_hash, file_hash


class AnalysisError(ValueError):
    pass


MAX_RESAMPLING_DRAWS = 5_000_000
DEFAULT_PLAN = {"schema_version": 1, "summary": "paired_seed_difference",
                "interval": {"method": "none"}, "figures": ["paired_seed"]}
SUPPORTED_FIGURES = ("paired_seed", "effect_summary", "calibration", "efficiency_pareto", "learning_curve")
CALIBRATION_BINS = 10
LIMITATIONS = [
    "Training seeds describe training randomness on one frozen test split, not independent subjects or datasets.",
    "A positive candidate-minus-baseline difference is not necessarily an improvement for a lower-is-better metric.",
    "No population generalization, causality, significance test, simultaneous confidence coverage or method ranking is established.",
    "Declared exchangeability and the scientific adequacy of the experiment require separate review.",
]
CALIBRATION_SCOPE = (
    "Expected calibration error compares predicted confidence against observed positive "
    "frequency in equal-width score bins, per training seed, on one frozen test split. "
    "Lower is better; it is not overall model quality and no population calibration is established.")
EFFICIENCY_SCOPE = (
    "Wall-clock seconds come from the frozen host execution ledger on one machine. "
    "No cross-hardware, monetary-cost or statistical ranking claim is made.")
LEARNING_CURVE_SCOPE = (
    "Step telemetry is emitted by the frozen experiment process and hash-bound to its execution receipt. "
    "The host validates its schema and completeness but does not independently recompute the declared training metric; "
    "it is descriptive training/validation telemetry, not frozen-test performance or a convergence guarantee.")


def validate_plan(value: dict | None) -> dict:
    """Validate before experiment execution; never infer a plan from outcomes."""
    if value is None:
        return copy.deepcopy(DEFAULT_PLAN)
    if (not isinstance(value, dict) or not set(DEFAULT_PLAN) <= set(value)
            or set(value) - set(DEFAULT_PLAN) - {"metric_direction", "learning_curve"}
            or type(value["schema_version"]) is not int or value["schema_version"] != 1
            or value["summary"] != "paired_seed_difference"):
        raise AnalysisError("Analysis plan requires schema_version, paired_seed_difference summary, interval and figures")
    figures = value["figures"]
    if (not isinstance(figures, list) or not figures or any(type(v) is not str for v in figures)
            or len(set(figures)) != len(figures) or set(figures) - set(SUPPORTED_FIGURES)):
        raise AnalysisError("Analysis figures must be unique supported plot types")
    # A Pareto frontier is meaningless without a declared optimization direction,
    # and a stray direction on plans that never draw one is a frozen-plan smell.
    if ("efficiency_pareto" in figures) != ("metric_direction" in value):
        raise AnalysisError("efficiency_pareto requires a declared metric_direction; no other figure may set one")
    if "metric_direction" in value and value["metric_direction"] not in {"minimize", "maximize"}:
        raise AnalysisError("Analysis metric_direction must be minimize or maximize")
    if ("learning_curve" in figures) != ("learning_curve" in value):
        raise AnalysisError("learning_curve figure requires an exact telemetry declaration")
    if "learning_curve" in value:
        curve = value["learning_curve"]
        if (not isinstance(curve, dict) or set(curve) != {"metric", "split", "direction", "max_points"}
                or not isinstance(curve["metric"], str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]{0,63}", curve["metric"])
                or curve["split"] not in {"train", "validation"}
                or curve["direction"] not in {"minimize", "maximize"}
                or type(curve["max_points"]) is not int or not 2 <= curve["max_points"] <= 2000):
            raise AnalysisError("Invalid learning-curve telemetry declaration")
    interval = value["interval"]
    if not isinstance(interval, dict):
        raise AnalysisError("Analysis interval must be an object")
    if interval == {"method": "none"}:
        return copy.deepcopy(value)
    common = {"method", "confidence", "replicates", "random_seed"}
    if (type(interval.get("confidence")) not in (int, float) or not 0.8 <= interval["confidence"] <= 0.99
            or type(interval.get("replicates")) is not int or not 1000 <= interval["replicates"] <= 20000
            or type(interval.get("random_seed")) is not int or not 0 <= interval["random_seed"] < 2**32):
        raise AnalysisError("Bootstrap requires bounded confidence, replicates and random seed")
    if interval["method"] == "paired_seed_percentile_bootstrap":
        if set(interval) != common | {"exchangeable_training_seeds"} or interval["exchangeable_training_seeds"] is not True:
            raise AnalysisError("Paired-seed bootstrap requires declared exchangeable training seeds")
    elif interval["method"] == "cluster_percentile_bootstrap":
        if set(interval) != common | {"resampling_unit"} or interval["resampling_unit"] != "group":
            raise AnalysisError("Cluster bootstrap requires resampling_unit=group")
    elif interval["method"] == "moving_block_percentile_bootstrap":
        if (set(interval) != common | {"resampling_unit", "block_length", "chronological_order_preserved"}
                or interval["resampling_unit"] != "time_point"
                or type(interval["block_length"]) is not int or not 2 <= interval["block_length"] <= 1000
                or interval["chronological_order_preserved"] is not True):
            raise AnalysisError("Moving-block bootstrap requires a bounded block and chronological time points")
    else:
        raise AnalysisError("Unsupported analysis interval method")
    return copy.deepcopy(value)


def _quantile(ordered: list[float], probability: float) -> float:
    index = probability * (len(ordered) - 1)
    low, high = math.floor(index), math.ceil(index)
    weight = index - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def summarize_pairs(rows: list[dict], plan: dict) -> dict:
    """Compute only matched, fully bound observations; all rows remain visible."""
    plan = validate_plan(plan)
    if (not isinstance(rows, list) or not rows or any(not isinstance(row, dict)
            or not isinstance(row.get("seed"), str) or not row["seed"]
            or any(type(row.get(name)) not in (int, float) or not math.isfinite(row[name])
                   for name in ("baseline", "candidate")) for row in rows)):
        raise AnalysisError("Analysis values must be finite numbers with explicit seed identities")
    if len({row["seed"] for row in rows}) != len(rows):
        raise AnalysisError("Analysis requires nonempty, unique matched seeds")
    samples = [row["candidate"] - row["baseline"] for row in rows]
    if not all(map(math.isfinite, samples)):
        raise AnalysisError("Analysis values and differences must be finite")
    interval = {"method": plan["interval"]["method"], "status": "not_requested", "low": None, "high": None,
                "resampling_unit": "matched_training_seed", "scope": "conditional_seed_mean_difference",
                "simultaneous_coverage": False, "assumption_verified": False}
    try:
        result = {"n_pairs": len(rows), "mean_baseline": statistics.mean(row["baseline"] for row in rows),
                  "mean_candidate": statistics.mean(row["candidate"] for row in rows),
                  "mean_difference": statistics.mean(samples),
                  "sd_difference": statistics.stdev(samples) if len(samples) > 1 else None,
                  "min_difference": min(samples), "max_difference": max(samples),
                  "negative_pairs": sum(v < 0 for v in samples), "zero_pairs": samples.count(0),
                  "positive_pairs": sum(v > 0 for v in samples), "interval": interval}
        if interval["method"] == "paired_seed_percentile_bootstrap":
            settings = plan["interval"]
            interval.update({k: settings[k] for k in ("confidence", "replicates", "random_seed")})
            interval["quantile_method"] = "linear_interpolation"
            if len(samples) < 3:
                interval["status"] = "unavailable_insufficient_seeds"
            elif min(samples) == max(samples):
                interval["status"] = "unavailable_constant_differences"
            else:
                if len(samples) * settings["replicates"] > MAX_RESAMPLING_DRAWS:
                    raise AnalysisError("Requested bootstrap exceeds resampling draw budget; no seed truncation is allowed")
                rng = random.Random(settings["random_seed"])
                # Resample whole pairs through their signed differences, never
                # independently shuffle candidate and baseline observations.
                boot = sorted(statistics.mean(rng.choices(samples, k=len(samples)))
                              for _ in range(settings["replicates"]))
                alpha = (1 - settings["confidence"]) / 2
                interval.update(status="computed_conditional", low=_quantile(boot, alpha), high=_quantile(boot, 1 - alpha))
        elif interval["method"] != "none":
            settings = plan["interval"]
            interval.update({k: settings[k] for k in ("confidence", "replicates", "random_seed")})
            interval.update(status="requires_frozen_sample_units", low=None, high=None,
                            resampling_unit=settings["resampling_unit"], scope="conditional_sample_structure")
        numbers = [v for v in result.values() if type(v) in (float, int)]
        numbers += [interval[k] for k in ("low", "high") if interval[k] is not None]
        if not all(map(math.isfinite, numbers)):
            raise AnalysisError("Nonfinite derived analysis statistic")
        return result
    except (OverflowError, statistics.StatisticsError) as exc:
        raise AnalysisError("Analysis arithmetic failed without a valid finite result") from exc


def expected_calibration_error(scores: dict[str, float], labels: dict[str, float],
                               bins: int = CALIBRATION_BINS) -> float:
    """Equal-width reliability bins over scores; empty bins contribute nothing."""
    total = len(scores)
    edges = [i / bins for i in range(bins + 1)]
    ece = 0.0
    for index in range(bins):
        low, high = edges[index], edges[index + 1]
        in_bin = [identity for identity, value in scores.items()
                  if (low <= value if index == 0 else low < value) and value <= high]
        if not in_bin:
            continue
        mean_score = statistics.fmean(scores[identity] for identity in in_bin)
        positive = statistics.fmean(labels[identity] for identity in in_bin)
        ece += len(in_bin) / total * abs(positive - mean_score)
    return ece


def _trusted_runs(root: Path) -> list[dict]:
    path = root / "trusted_evaluation.json"
    if not path.is_file():
        raise AnalysisError("Calibration figures require trusted_evaluation.json with the frozen per-run predictions")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (not isinstance(manifest, dict) or manifest.get("schema_version") != 1
            or not isinstance(manifest.get("runs"), list) or any(not isinstance(run, dict) for run in manifest["runs"])):
        raise AnalysisError("Trusted evaluation manifest is malformed")
    return manifest["runs"]


def _read_frozen_scores(root: Path, record, run: dict | None) -> tuple[dict[str, float], dict[str, float]]:
    """Read a run's predictions and labels only through the record's own hashes."""
    if run is None or run.get("key") != asdict(record.key) or run.get("labels") not in dict(record.artifacts) \
            or run.get("predictions") not in dict(record.artifacts):
        raise AnalysisError("Trusted evaluation manifest does not bind the analyzed record's frozen files")
    from researchclaw.pipeline.independent_evaluator import _read_values
    digests = dict(record.artifacts)
    for name in (run["labels"], run["predictions"]):
        if file_hash(root / name) != digests[name]:
            raise AnalysisError("Frozen predictions or labels changed; the calibration figure cannot be computed")
    try:
        scores = _read_values(root / run["predictions"], "prediction")
        labels = _read_values(root / run["labels"], "label")
    except (OSError, ValueError) as exc:
        raise AnalysisError(f"Calibration data is unreadable or malformed: {exc}") from exc
    if scores.keys() != labels.keys():
        raise AnalysisError("Predictions do not cover exactly the frozen labels")
    if any(not 0.0 <= value <= 1.0 for value in scores.values()):
        raise AnalysisError("Calibration requires probabilistic scores in [0, 1]")
    if any(value not in (0.0, 1.0) for value in labels.values()):
        raise AnalysisError("Calibration requires binary labels")
    return scores, labels


def calibration_rows(root: Path, store: EvidenceStore, rows: list[dict]) -> dict:
    """Per-seed ECE for both methods, computed from frozen per-sample predictions."""
    runs = _trusted_runs(root)
    rows_out = []
    for row in rows:
        entry, samples = {}, None
        for name in ("baseline_result", "candidate_result"):
            scores, labels = _read_frozen_scores(root, store.records[row[name]],
                                                 next((run for run in runs if run.get("key") == asdict(store.records[row[name]].key)), None))
            entry[name.replace("_result", "_ece")] = expected_calibration_error(scores, labels)
            samples = len(scores)
        rows_out.append({"seed": row["seed"], **entry, "difference": entry["candidate_ece"] - entry["baseline_ece"],
                         "test_samples": samples})
    return {"bins": CALIBRATION_BINS, "rows": rows_out, "resampling_unit": "training_seed", "scope": CALIBRATION_SCOPE}


def efficiency_rows(root: Path, protocol: dict, store: EvidenceStore, rows: list[dict], direction: str) -> dict:
    """Per-seed wall-clock seconds from the frozen host execution ledger."""
    path = root / "protocol_budget.json"
    if not path.is_file():
        raise AnalysisError("Efficiency figures require the frozen protocol budget record")
    budget = json.loads(path.read_text(encoding="utf-8"))
    if (not isinstance(budget, dict) or budget.get("schema_version") != 1
            or budget.get("protocol_version") != protocol["version"] or budget.get("status") != "complete"
            or not isinstance(budget.get("per_cell_seconds"), dict)):
        raise AnalysisError("Efficiency figures require a complete protocol budget for the same protocol version")
    rows_out = []
    for row in rows:
        entry = {"seed": row["seed"]}
        for name in ("baseline_result", "candidate_result"):
            record = store.records[row[name]]
            seconds = budget["per_cell_seconds"].get(content_hash(asdict(record.key)))
            if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds < 0:
                raise AnalysisError("The frozen protocol budget does not record finite seconds for every analyzed cell")
            entry[name.replace("_result", "_seconds")] = seconds
        rows_out.append(entry)
    return {"metric_direction": direction, "rows": rows_out, "scope": EFFICIENCY_SCOPE}


def read_learning_curve(path: Path, declaration: dict) -> list[dict]:
    """Read the exact bounded step,value telemetry schema."""
    path = Path(path)
    if not path.is_file() or path.stat().st_size > 1_000_000:
        raise AnalysisError("Learning curve telemetry is missing or exceeds its byte budget")
    import csv
    points = []
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["step", "value"]:
            raise AnalysisError("Learning curve CSV must have exactly step,value columns")
        for row in reader:
            try:
                step, value = int(row["step"]), float(row["value"])
            except (TypeError, ValueError) as exc:
                raise AnalysisError("Learning curve contains invalid numeric telemetry") from exc
            if str(step) != row["step"] or step < 0 or type(value) is not float or not math.isfinite(value):
                raise AnalysisError("Learning curve steps must be canonical nonnegative integers with finite values")
            points.append({"step": step, "value": value})
    if (not 2 <= len(points) <= declaration["max_points"]
            or any(right["step"] <= left["step"] for left, right in zip(points, points[1:]))):
        raise AnalysisError("Learning curve must contain bounded strictly increasing steps")
    return points


def _curve_file(root: Path, record, declaration: dict) -> list[dict]:
    receipts = [name for name, digest in record.artifacts if digest == record.execution]
    if len(receipts) != 1:
        raise AnalysisError("Learning curve needs one hash-bound execution receipt")
    receipt = json.loads((root / receipts[0]).read_text(encoding="utf-8"))
    relative, digest = receipt.get("learning_curve"), receipt.get("learning_curve_sha256")
    if not isinstance(relative, str) or not isinstance(digest, str):
        raise AnalysisError("Learning curve telemetry is missing from the execution receipt")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file() or file_hash(path) != digest:
        raise AnalysisError("Learning curve telemetry changed or escaped the bundle")
    return read_learning_curve(path, declaration)


def learning_curve_rows(root: Path, store: EvidenceStore, rows: list[dict], declaration: dict) -> dict:
    output = []
    for row in rows:
        baseline = _curve_file(root, store.records[row["baseline_result"]], declaration)
        candidate = _curve_file(root, store.records[row["candidate_result"]], declaration)
        if [point["step"] for point in baseline] != [point["step"] for point in candidate]:
            raise AnalysisError("Paired learning curves must report the same declared steps")
        output.extend({"seed": row["seed"], "step": left["step"], "baseline": left["value"],
                       "candidate": right["value"], "difference": right["value"] - left["value"]}
                      for left, right in zip(baseline, candidate))
    return {"metric": declaration["metric"], "split": declaration["split"],
            "direction": declaration["direction"], "rows": output, "scope": LEARNING_CURVE_SCOPE}


def _metric_on_ids(metric: str, labels: dict[str, float], predictions: dict[str, float], identities: list[str]) -> float:
    pairs = [(labels[identity], predictions[identity]) for identity in identities]
    if metric == "accuracy":
        return statistics.fmean(left == right for left, right in pairs)
    if metric == "mse":
        return statistics.fmean((left - right) ** 2 for left, right in pairs)
    if metric == "mae":
        return statistics.fmean(abs(left - right) for left, right in pairs)
    if metric == "auroc":
        positives = sum(left == 1 for left, _ in pairs)
        negatives = len(pairs) - positives
        if not positives or not negatives:
            raise AnalysisError("A structured bootstrap replicate lacks both AUROC classes")
        ordered = sorted(pairs, key=lambda pair: pair[1])
        rank_sum, index = 0.0, 0
        while index < len(ordered):
            end = index + 1
            while end < len(ordered) and ordered[end][1] == ordered[index][1]:
                end += 1
            rank_sum += ((index + 1 + end) / 2) * sum(left for left, _ in ordered[index:end])
            index = end
        return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)
    raise AnalysisError("Unsupported metric for structured bootstrap")


def structured_interval(root: Path, store: EvidenceStore, rows: list[dict], plan: dict, metric: str) -> dict:
    """Cluster or circular moving-block bootstrap over frozen test sample identities."""
    settings = plan["interval"]
    from researchclaw.research_inputs import verify_bundle_contract
    contract = verify_bundle_contract(root)
    dataset_name = store.records[rows[0]["baseline_result"]].key.dataset
    dataset = next((item for item in contract["datasets"] if item["manifest"]["dataset"] == dataset_name), None)
    relative = dataset.get("analysis_units") if dataset else None
    if not isinstance(relative, str):
        raise AnalysisError("Structured bootstrap requires frozen group/time analysis units")
    import csv
    unit_rows = list(csv.DictReader((root / relative).open(encoding="utf-8", newline="")))
    if not unit_rows or set(unit_rows[0]) != {"id", "group", "time", "time_order"} \
            or len({row["id"] for row in unit_rows}) != len(unit_rows):
        raise AnalysisError("Frozen analysis-unit mapping is malformed")
    runs = _trusted_runs(root)
    paired = []
    expected_ids = {row["id"] for row in unit_rows}
    for row in rows:
        values = []
        labels = None
        for result_name in ("baseline_result", "candidate_result"):
            record = store.records[row[result_name]]
            run = next((item for item in runs if item.get("key") == asdict(record.key)), None)
            prediction, truth = _read_frozen_scores(root, record, run)
            if set(prediction) != expected_ids:
                raise AnalysisError("Analysis units do not cover exactly the frozen test samples")
            labels = truth
            values.append(prediction)
        paired.append((labels, values[0], values[1]))
    if settings["method"] == "cluster_percentile_bootstrap":
        groups = {}
        for row in unit_rows:
            if not row["group"]:
                raise AnalysisError("Cluster bootstrap requires group analysis units")
            groups.setdefault(row["group"], []).append(row["id"])
        units = list(groups.values())
        def sample(rng):
            return [identity for group in rng.choices(units, k=len(units)) for identity in group]
        unit_label = "test_group"
    else:
        points = {}
        for row in unit_rows:
            try:
                order = int(row["time_order"])
            except ValueError as exc:
                raise AnalysisError("Time bootstrap requires canonical time order") from exc
            if not row["time"] or str(order) != row["time_order"] or order < 0:
                raise AnalysisError("Time bootstrap requires ordered time analysis units")
            points.setdefault(order, []).append(row["id"])
        if sorted(points) != list(range(len(points))) or settings["block_length"] > len(points):
            raise AnalysisError("Moving-block length exceeds the contiguous frozen time points")
        ordered = [points[index] for index in range(len(points))]
        length = settings["block_length"]
        def sample(rng):
            selected = []
            while len(selected) < len(ordered):
                start = rng.randrange(len(ordered))
                selected.extend(ordered[(start + offset) % len(ordered)] for offset in range(length))
            return [identity for point in selected[:len(ordered)] for identity in point]
        units, unit_label = ordered, "time_point_circular_block"
    if len(units) < 3:
        raise AnalysisError("Structured bootstrap requires at least three frozen resampling units")
    if len(unit_rows) * len(rows) * settings["replicates"] > MAX_RESAMPLING_DRAWS:
        raise AnalysisError("Structured bootstrap exceeds the frozen sample draw budget")
    rng, estimates = random.Random(settings["random_seed"]), []
    for _ in range(settings["replicates"]):
        identities = sample(rng)
        estimates.append(statistics.fmean(
            _metric_on_ids(metric, labels, candidate, identities)
            - _metric_on_ids(metric, labels, baseline, identities)
            for labels, baseline, candidate in paired))
    estimates.sort()
    alpha = (1 - settings["confidence"]) / 2
    return {"method": settings["method"], "status": "computed_conditional", "low": _quantile(estimates, alpha),
            "high": _quantile(estimates, 1 - alpha), "confidence": settings["confidence"],
            "replicates": settings["replicates"], "random_seed": settings["random_seed"],
            "resampling_unit": unit_label, "unit_count": len(units), "sample_count": len(unit_rows),
            "scope": "conditional_frozen_test_sample_structure", "simultaneous_coverage": False,
            "assumption_verified": False, "quantile_method": "linear_interpolation"}


def build_analysis(root: Path) -> dict | None:
    from researchclaw.pipeline.experiment_protocol import load_protocol, audit_coverage
    protocol = load_protocol(root)
    if protocol is None:
        return None
    store = EvidenceStore.from_dict(json.loads((root / "evidence_store.json").read_text(encoding="utf-8")))
    coverage = audit_coverage(root, protocol, store)
    if coverage["status"] != "complete":
        raise AnalysisError("Cannot analyze a selected subset of an incomplete protocol")
    questions = {q["id"]: q for q in protocol["spec"]["questions"]}
    analyses, reserved_draws = [], 0
    for comparison in coverage["comparisons"]:
        question = questions[comparison["question"]]
        plan = validate_plan(question.get("analysis_plan"))
        if plan["interval"]["method"] == "paired_seed_percentile_bootstrap":
            reserved_draws += len(comparison["pairs"]) * plan["interval"]["replicates"]
        if reserved_draws > MAX_RESAMPLING_DRAWS:
            raise AnalysisError("AnalysisSpec exceeds the total bootstrap draw budget; revise the frozen plan")
        rows = []
        for pair in sorted(comparison["pairs"], key=lambda p: int(p["seed"])):
            baseline, candidate = (store.records[pair[name]] for name in ("baseline_result", "candidate_result"))
            rows.append({"seed": pair["seed"], "baseline": baseline.value, "candidate": candidate.value,
                         "difference": candidate.value - baseline.value,
                         "baseline_result": pair["baseline_result"], "candidate_result": pair["candidate_result"]})
        first = store.records[rows[0]["baseline_result"]]
        context = {field: getattr(first.key, field) for field in
                   ("dataset", "dataset_version", "split", "metric", "aggregation", "regime")}
        identity = {**context, "question": question["id"], "baseline": comparison["baseline"], "candidate": comparison["candidate"]}
        analysis = {"analysis_id": content_hash(identity), **identity, "kind": question["kind"], "unit": first.unit,
                    "question_text": question["question"], "declared_analysis": question["analysis"],
                    "plan": plan, "plan_origin": "predeclared" if "analysis_plan" in question else "pipeline_descriptive_default",
                    "pairs": rows, "result_ids": list(dict.fromkeys(r[name] for r in rows for name in ("baseline_result", "candidate_result"))),
                    "statistics": summarize_pairs(rows, plan), "direction": "candidate_minus_baseline",
                    "permitted_scope": "Observed matched training-seed differences under the frozen conditions; optional intervals are conditional on declared exchangeability.",
                    "limitations": list(LIMITATIONS)}
        if plan["interval"]["method"] in {"cluster_percentile_bootstrap", "moving_block_percentile_bootstrap"}:
            analysis["statistics"]["interval"] = structured_interval(root, store, rows, plan, first.key.metric)
            reserved_draws += (analysis["statistics"]["interval"]["sample_count"] * len(rows)
                               * plan["interval"]["replicates"])
            if reserved_draws > MAX_RESAMPLING_DRAWS:
                raise AnalysisError("AnalysisSpec exceeds the total bootstrap draw budget; revise the frozen plan")
        # Figure data is frozen inside the report so recompute validation covers
        # it; a plan that requests a figure the data cannot support fails closed.
        if "calibration" in plan["figures"]:
            analysis["calibration"] = calibration_rows(root, store, rows)
        if "efficiency_pareto" in plan["figures"]:
            analysis["efficiency"] = efficiency_rows(root, protocol, store, rows, plan["metric_direction"])
        if "learning_curve" in plan["figures"]:
            analysis["learning_curve"] = learning_curve_rows(root, store, rows, plan["learning_curve"])
        analyses.append(analysis)
    report = {"schema_version": 1, "checker": "analysis-spec/v1", "protocol_version": protocol["version"],
              "implementation": {"source_sha256": file_hash(Path(__file__)), "python": platform.python_version()},
              "evidence_version": store.version, "sources": {name: file_hash(root / name) for name in
                  ("experiment_protocol.json", "evidence_store.json")},
              "status": "computed", "analyses": analyses,
              "budget": {"reserved_resampling_draws": reserved_draws, "max_resampling_draws": MAX_RESAMPLING_DRAWS},
              "scope": "recomputed numerical analysis; scientific assumptions remain unverified"}
    report["version"] = content_hash(report)
    return report


def prepare_analysis(root: Path) -> dict | None:
    report = build_analysis(root)
    path = root / "analysis_spec.json"
    if report is None:
        if path.exists():
            raise AnalysisError("AnalysisSpec exists without its frozen protocol")
        return None
    if path.is_file():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous == report:
            return report
        write_json(root / "evidence_artifacts/analysis_history" / f"{content_hash(previous)}.json", previous)
    write_json(path, report)
    return report


def analysis_figures(report: dict) -> list[dict]:
    """Select only frozen plot types; never select by observed effect size."""
    figures, effect_groups = [], {}
    for analysis in report["analyses"]:
        context = {name: analysis[name] for name in ("question", "dataset", "dataset_version", "split", "metric", "aggregation", "regime", "unit")}
        if "paired_seed" in analysis["plan"]["figures"]:
            figure = {**context, "kind": "paired_seed", "baseline": analysis["baseline"], "candidate": analysis["candidate"],
                      "analysis_ids": [analysis["analysis_id"]], "rows": analysis["pairs"],
                      "mean_difference": analysis["statistics"]["mean_difference"], "direction": analysis["direction"],
                      "resampling_unit": "training_seed", "inference_scope": analysis["permitted_scope"]}
            figures.append({"id": "results-" + content_hash(figure)[:24], **figure})
        if "calibration" in analysis["plan"]["figures"]:
            data = analysis.get("calibration")
            if not isinstance(data, dict) or not data.get("rows"):
                raise AnalysisError("Calibration figure requested but frozen per-seed ECE data is missing")
            figure = {**context, "kind": "calibration", "baseline": analysis["baseline"], "candidate": analysis["candidate"],
                      "analysis_ids": [analysis["analysis_id"]], "rows": data["rows"], "bins": data["bins"],
                      "mean_difference": statistics.fmean(row["difference"] for row in data["rows"]),
                      "direction": analysis["direction"], "resampling_unit": "training_seed",
                      "inference_scope": data["scope"]}
            figures.append({"id": "results-" + content_hash(figure)[:24], **figure})
        if "efficiency_pareto" in analysis["plan"]["figures"]:
            data = analysis.get("efficiency")
            if not isinstance(data, dict) or not data.get("rows"):
                raise AnalysisError("Efficiency figure requested but frozen wall-clock data is missing")
            seconds = {row["seed"]: row for row in data["rows"]}
            points = [{"seed": row["seed"], "baseline": row["baseline"], "candidate": row["candidate"],
                       "baseline_seconds": seconds[row["seed"]]["baseline_seconds"],
                       "candidate_seconds": seconds[row["seed"]]["candidate_seconds"]} for row in analysis["pairs"]]
            means = {method: {"metric": statistics.fmean(point[method] for point in points),
                              "seconds": statistics.fmean(point[f"{method}_seconds"] for point in points)}
                     for method in ("baseline", "candidate")}
            figure = {**context, "kind": "efficiency_pareto", "baseline": analysis["baseline"], "candidate": analysis["candidate"],
                      "analysis_ids": [analysis["analysis_id"]], "rows": points, "means": means,
                      "metric_direction": data["metric_direction"], "resampling_unit": "training_seed",
                      "inference_scope": data["scope"]}
            figures.append({"id": "results-" + content_hash(figure)[:24], **figure})
        if "learning_curve" in analysis["plan"]["figures"]:
            data = analysis.get("learning_curve")
            if not isinstance(data, dict) or not data.get("rows"):
                raise AnalysisError("Learning-curve figure requested but frozen telemetry is missing")
            figure = {**context, "kind": "learning_curve", "baseline": analysis["baseline"],
                      "candidate": analysis["candidate"], "analysis_ids": [analysis["analysis_id"]],
                      "rows": data["rows"], "curve_metric": data["metric"], "curve_split": data["split"],
                      "metric_direction": data["direction"], "resampling_unit": "training_seed",
                      "inference_scope": data["scope"]}
            figures.append({"id": "results-" + content_hash(figure)[:24], **figure})
        if "effect_summary" in analysis["plan"]["figures"]:
            effect_groups.setdefault(json.dumps(context, sort_keys=True), []).append(analysis)
    for encoded, analyses in effect_groups.items():
        # Bound physical figure height; all additional comparisons get figures.
        batches: list[list[dict]] = [[]]
        for analysis in analyses:
            current = batches[-1]
            if len(current) >= 6 or sum(effect_label_weight(a) for a in [*current, analysis]) > 22:
                batches.append([])
            batches[-1].append(analysis)
        for batch in batches:
            figure = {**json.loads(encoded), "kind": "effect_summary", "direction": "candidate_minus_baseline",
                      "analysis_ids": [a["analysis_id"] for a in batch],
                      "comparisons": [{name: a[name] for name in ("analysis_id", "baseline", "candidate", "pairs", "statistics")} for a in batch],
                      "resampling_unit": "training_seed", "inference_scope": list(LIMITATIONS)}
            figures.append({"id": "results-" + content_hash(figure)[:24], **figure})
    return figures


def effect_label_weight(analysis: dict) -> int:
    return max(3, fill(f"{analysis['candidate']} - {analysis['baseline']}", 23).count("\n") + 2)


def verify_analysis(root: Path) -> dict:
    report = json.loads((root / "analysis_spec.json").read_text(encoding="utf-8"))
    if report != build_analysis(root) or not isinstance(report, dict):
        raise AnalysisError("AnalysisSpec differs from the frozen plan and authoritative results")
    return report
