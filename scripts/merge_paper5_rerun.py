"""Merge the verified Paper-5 FullAuditedAgent rerun into Stage-12 evidence."""
from __future__ import annotations

import hashlib
import json
import math
import random
import shutil
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "artifacts/paper5_trace_guard/full_run"
ORIGINAL = RUN / "stage-12_v1/runs/sandbox/_project_1"
RERUN = ROOT / "artifacts/paper5_trace_guard/full_audited_rerun"


def canonical_hash(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def auc(labels: list[int], scores: list[float]) -> float | None:
    pos = sum(labels)
    neg = len(labels) - pos
    if not pos or not neg:
        return None
    ranked = sorted(zip(scores, labels))
    rank_sum = 0.0
    i = 0
    while i < len(ranked):
        j = i + 1
        while j < len(ranked) and ranked[j][0] == ranked[i][0]:
            j += 1
        rank_sum += ((i + j + 1) / 2) * sum(v for _, v in ranked[i:j])
        i = j
    return (rank_sum - pos * (pos + 1) / 2) / (pos * neg)


def auprc(labels: list[int], scores: list[float]) -> float | None:
    positives = sum(labels)
    if not positives:
        return None
    ordered = sorted(zip(scores, labels), reverse=True)
    tp = 0
    area = 0.0
    previous_recall = 0.0
    for rank, (_, label) in enumerate(ordered, 1):
        tp += label
        if label:
            recall = tp / positives
            area += (recall - previous_recall) * (tp / rank)
            previous_recall = recall
    return area


def bootstrap(labels: list[int], scores: list[float], metric, seed: int) -> list[float | None]:
    rng = random.Random(seed)
    values = []
    for _ in range(2000):
        ix = [rng.randrange(len(labels)) for _ in labels]
        value = metric([labels[i] for i in ix], [scores[i] for i in ix])
        if value is not None and math.isfinite(value):
            values.append(value)
    values.sort()
    if len(values) < 100:
        return [None, None]
    return [values[int(0.025 * (len(values) - 1))], values[int(0.975 * (len(values) - 1))]]


def condition_metrics(items: list[dict], positives: set[str]) -> dict:
    usable = [x for x in items if x.get("score") is not None]
    labels = [1 if str(float(x["outcome"])) in positives else 0 for x in usable]
    scores = [float(x["score"]) for x in usable]
    roc = auc(labels, scores)
    pr = auprc(labels, scores)
    roc_ci = bootstrap(labels, scores, auc, 20260805) if usable else [None, None]
    pr_ci = bootstrap(labels, scores, auprc, 20260806) if usable else [None, None]
    brier = sum((s - y) ** 2 for y, s in zip(labels, scores)) / len(scores) if scores else None
    ece = 0.0 if scores else None
    bins = []
    if scores:
        for b in range(10):
            lo, hi = b / 10, (b + 1) / 10
            ix = [i for i, s in enumerate(scores) if lo <= s < hi or (b == 9 and s == 1)]
            if ix:
                predicted = sum(scores[i] for i in ix) / len(ix)
                observed = sum(labels[i] for i in ix) / len(ix)
                ece += len(ix) / len(scores) * abs(predicted - observed)
                bins.append({"lower": lo, "upper": hi, "mean_prediction": predicted,
                             "event_rate": observed, "n": len(ix)})
    return {
        "n": len(items), "n_scored": len(usable), "coverage": len(usable) / len(items),
        "n_events_scored": sum(labels), "auroc": roc, "auroc_ci_low": roc_ci[0],
        "auroc_ci_high": roc_ci[1], "auprc": pr, "auprc_ci_low": pr_ci[0],
        "auprc_ci_high": pr_ci[1], "brier_score": brier,
        "expected_calibration_error": ece, "calibration_curve": bins,
        "mean_latency_sec": sum(float(x.get("latency_sec", 0)) for x in items) / len(items),
        "abstention_rate": sum(bool(x.get("abstain")) for x in items) / len(items),
        "error_rate": sum(bool(x.get("error")) for x in items) / len(items),
        "critic_calls": sum(bool(x.get("critic_response_hash")) for x in items),
        "critic_accepted": sum(isinstance(x.get("critic_verdict"), dict)
                               and bool(x["critic_verdict"].get("accept")) for x in items),
    }


def main() -> None:
    original = read_jsonl(ORIGINAL / "patient_results.jsonl")
    rerun = read_jsonl(RERUN / "patient_results.jsonl")
    assert len(original) == 430 and len(rerun) == 86
    assert {x["condition"] for x in rerun} == {"FullAuditedAgent"}

    verified = 0
    for row in rerun:
        blinded = dict(row)
        blinded.pop("outcome", None)
        blinded["outcome_joined"] = False
        expected = blinded.pop("event_hash")
        assert canonical_hash(blinded) == expected
        assert "抗凝药后是否出血" not in row["state_card"]
        assert "编号" not in row["state_card"] and "姓名" not in row["state_card"]
        verified += 1

    merged = [x for x in original if x["condition"] != "FullAuditedAgent"] + rerun
    keys = [(x["condition"], x["row_index"]) for x in merged]
    assert len(merged) == 430 and len(set(keys)) == 430
    conditions_order = ["DirectStructured", "RetrievalOnly", "NoCritic", "NoAbstention", "FullAuditedAgent"]
    merged.sort(key=lambda x: (conditions_order.index(x["condition"]), x["row_index"]))

    positives = {"1.0"}
    metrics_by_condition = {
        condition: condition_metrics([x for x in merged if x["condition"] == condition], positives)
        for condition in conditions_order
    }
    metrics = {
        "schema_version": "paper5-merged/v1", "n_records": 86, "n_conditions": 5,
        "n_patient_condition_rows": 430,
        "n_scored": sum(v["n_scored"] for v in metrics_by_condition.values()),
        "coverage": sum(v["n_scored"] for v in metrics_by_condition.values()) / 430,
        "conditions": metrics_by_condition,
    }

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = ORIGINAL / f"premerge_backup_{stamp}"
    backup.mkdir()
    for name in ("patient_results.jsonl", "metrics.json", "results.json"):
        if (ORIGINAL / name).exists():
            shutil.copy2(ORIGINAL / name, backup / name)

    merged_text = "\n".join(json.dumps(x, ensure_ascii=False) for x in merged) + "\n"
    metrics_text = json.dumps(metrics, ensure_ascii=False, indent=2)
    results = {"metrics": metrics, "conditions": metrics_by_condition,
               "merge_manifest": "merge_manifest.json"}
    for target in (ORIGINAL / "patient_results.jsonl",):
        target.write_text(merged_text, encoding="utf-8")
    (ORIGINAL / "metrics.json").write_text(metrics_text, encoding="utf-8")
    (ORIGINAL / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    runs = ORIGINAL.parents[1]
    (runs / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    (runs / "run-1.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    canonical_ledger = ORIGINAL / "ledger_merged"
    canonical_ledger.mkdir(exist_ok=False)
    for row in merged:
        event = dict(row)
        event.pop("outcome", None)
        event["outcome_joined"] = False
        event_hash = event.pop("event_hash")
        assert canonical_hash(event) == event_hash
        event["event_hash"] = event_hash
        name = f'{row["condition"]}_{row["row_index"]}.json'
        (canonical_ledger / name).write_text(json.dumps(event, ensure_ascii=False), encoding="utf-8")

    manifest = {
        "schema_version": "paper5-merge-manifest/v1", "created_utc": stamp,
        "original_stage12_rows": len(original), "rerun_rows": len(rerun),
        "rerun_hash_verified": verified, "merged_rows": len(merged),
        "unique_condition_row_keys": len(set(keys)),
        "source_original": str(ORIGINAL / "patient_results.jsonl"),
        "source_rerun": str(RERUN / "patient_results.jsonl"),
        "merged_patient_results_sha256": hashlib.sha256(merged_text.encode()).hexdigest(),
        "metrics_sha256": hashlib.sha256(metrics_text.encode()).hexdigest(),
        "backup": str(backup), "canonical_ledger": str(canonical_ledger),
    }
    (ORIGINAL / "merge_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"manifest": manifest, "metrics": metrics}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
