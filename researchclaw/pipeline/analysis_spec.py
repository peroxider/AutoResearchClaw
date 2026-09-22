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
import statistics
from pathlib import Path
from textwrap import fill

from researchclaw.literature.evidence import write_json
from researchclaw.pipeline.evidence_store import EvidenceStore, content_hash, file_hash


class AnalysisError(ValueError):
    pass


MAX_RESAMPLING_DRAWS = 5_000_000
DEFAULT_PLAN = {"schema_version": 1, "summary": "paired_seed_difference",
                "interval": {"method": "none"}, "figures": ["paired_seed"]}
LIMITATIONS = [
    "Training seeds describe training randomness on one frozen test split, not independent subjects or datasets.",
    "A positive candidate-minus-baseline difference is not necessarily an improvement for a lower-is-better metric.",
    "No population generalization, causality, significance test, simultaneous confidence coverage or method ranking is established.",
    "Declared exchangeability and the scientific adequacy of the experiment require separate review.",
]


def validate_plan(value: dict | None) -> dict:
    """Validate before experiment execution; never infer a plan from outcomes."""
    if value is None:
        return copy.deepcopy(DEFAULT_PLAN)
    if (not isinstance(value, dict) or set(value) != set(DEFAULT_PLAN)
            or type(value["schema_version"]) is not int or value["schema_version"] != 1
            or value["summary"] != "paired_seed_difference"):
        raise AnalysisError("Analysis plan requires schema_version, paired_seed_difference summary, interval and figures")
    figures = value["figures"]
    if (not isinstance(figures, list) or not figures or any(type(v) is not str for v in figures)
            or len(set(figures)) != len(figures) or set(figures) - {"paired_seed", "effect_summary"}):
        raise AnalysisError("Analysis figures must be unique supported plot types")
    interval = value["interval"]
    if not isinstance(interval, dict):
        raise AnalysisError("Analysis interval must be an object")
    if interval == {"method": "none"}:
        return copy.deepcopy(value)
    fields = {"method", "confidence", "replicates", "random_seed", "exchangeable_training_seeds"}
    if (set(interval) != fields or interval["method"] != "paired_seed_percentile_bootstrap"
            or type(interval["confidence"]) not in (int, float) or not 0.8 <= interval["confidence"] <= 0.99
            or type(interval["replicates"]) is not int or not 1000 <= interval["replicates"] <= 20000
            or type(interval["random_seed"]) is not int or not 0 <= interval["random_seed"] < 2**32
            or interval["exchangeable_training_seeds"] is not True):
        raise AnalysisError("Bootstrap requires bounded confidence/replicates/seed and declared exchangeable training seeds")
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
        if interval["method"] != "none":
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
        numbers = [v for v in result.values() if type(v) in (float, int)]
        numbers += [interval[k] for k in ("low", "high") if interval[k] is not None]
        if not all(map(math.isfinite, numbers)):
            raise AnalysisError("Nonfinite derived analysis statistic")
        return result
    except (OverflowError, statistics.StatisticsError) as exc:
        raise AnalysisError("Analysis arithmetic failed without a valid finite result") from exc


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
        if plan["interval"]["method"] != "none":
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
