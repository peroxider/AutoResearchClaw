"""Run auditable SCAFFOLD-R ablations on the public UCI diabetes cohort.

This study is a public, cross-task method validation for 30-day readmission;
it is not an external validation of the local bleeding-risk endpoint.  The
script excludes UCI identifiers before any model request and records every
request/response hash before outcomes are joined for evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
UCI_CSV = ROOT / "artifacts" / "paper3_cmpb_rebuild" / "data" / "uci" / "diabetic_data.csv"
OUT = ROOT / "artifacts" / "paper3_cmpb_rebuild" / "uci_experiment"
API_URL = "https://api.minimaxi.com/anthropic/v1/messages"
MODEL = "MiniMax-M3"
SEED = 20260727

# All identifiers, outcome fields, and explicit post-discharge fields are
# excluded. This is a discharge-time readmission-risk research task.
ID_AND_OUTCOME_COLUMNS = {"encounter_id", "patient_nbr", "readmitted"}
EXCLUDED_COLUMNS = ID_AND_OUTCOME_COLUMNS | {"discharge_disposition_id"}
FEATURE_COLUMNS = (
    "race", "gender", "age", "admission_type_id", "admission_source_id",
    "time_in_hospital", "medical_specialty", "num_lab_procedures",
    "num_procedures", "num_medications", "number_outpatient",
    "number_emergency", "number_inpatient", "diag_1", "diag_2", "diag_3",
    "number_diagnoses", "max_glu_serum", "A1Cresult", "metformin",
    "insulin", "change", "diabetesMed",
)

RETRIEVAL_POLICY = {
    "R1": "Use only fields present in the state card; do not infer undocumented diagnoses or social factors.",
    "R2": "A higher count of prior inpatient or emergency encounters may support a higher readmission-risk rank, but uncertainty must be stated.",
    "R3": "Medication and laboratory indicators are encounter descriptors, not treatment recommendations. Return a research score only.",
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def clean_value(value: object) -> str | int | float | None:
    if pd.isna(value) or value == "?":
        return None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return round(float(value), 4)
    return str(value)


def load_cohort(n: int) -> tuple[list[dict[str, Any]], np.ndarray, dict[str, Any]]:
    df = pd.read_csv(UCI_CSV)
    missing = set(FEATURE_COLUMNS) - set(df.columns)
    if missing:
        raise RuntimeError(f"UCI schema changed; missing expected columns: {sorted(missing)}")
    target = (df["readmitted"].astype(str) == "<30").astype(int)
    rng = np.random.default_rng(SEED)
    event_idx = np.flatnonzero(target.to_numpy() == 1)
    nonevent_idx = np.flatnonzero(target.to_numpy() == 0)
    # Preserve the cohort event rate for every requested sample size.  The
    # former lower bound of 30 distorted tiny smoke-test cohorts into an
    # almost-all-event case-control sample.
    n_event = max(1, round(n * len(event_idx) / len(df)))
    n_event = min(n_event, len(event_idx), n - 1)
    n_nonevent = n - n_event
    selected = np.concatenate((rng.choice(event_idx, n_event, replace=False), rng.choice(nonevent_idx, n_nonevent, replace=False)))
    rng.shuffle(selected)
    cards = []
    for row_id, row in df.iloc[selected].iterrows():
        fields = {col: clean_value(row[col]) for col in FEATURE_COLUMNS}
        cards.append({"study_row": len(cards), "state": fields})
    metadata = {
        "source": "UCI Diabetes 130-US Hospitals for Years 1999-2008",
        "doi": "10.24432/C5230J",
        "licence": "CC BY 4.0",
        "prediction_time": "discharge-time research assessment",
        "outcome": "readmitted == '<30'",
        "sampling": {"seed": SEED, "n": n, "events": int(target.iloc[selected].sum()), "non_events": int(n - target.iloc[selected].sum())},
        "excluded_columns": sorted(EXCLUDED_COLUMNS),
        "feature_allow_list": list(FEATURE_COLUMNS),
        "raw_csv_sha256": hashlib.sha256(UCI_CSV.read_bytes()).hexdigest(),
    }
    return cards, target.iloc[selected].to_numpy(dtype=int), metadata


def condition_instruction(condition: str) -> str:
    common = (
        "You are evaluating a public, de-identified diabetes hospital-record dataset for a research-only "
        "30-day readmission study. Do not make clinical recommendations. The outcome is withheld. "
        "Return exactly one assessment per card in input order. "
    )
    if condition == "direct_prompt":
        return common + "Return JSON {\"assessments\":[{\"risk_score\":0-100,\"uncertainty\":\"short text\"}]} only."
    if condition == "retrieval_prompt":
        return common + "Use only the supplied policy snippets. Return JSON {\"assessments\":[{\"risk_score\":0-100,\"evidence_ids\":[\"R1\"],\"uncertainty\":\"short text\"}]} only."
    if condition == "scaffold_no_critic":
        return common + "Use the schema and policy snippets. Return JSON {\"assessments\":[{\"risk_score\":0-100,\"evidence_fields\":[\"exact state field\"],\"evidence_ids\":[\"R1\"],\"uncertainty_flags\":[\"...\"]}]} only."
    if condition == "scaffold_no_abstention":
        return common + "Use the schema and policy snippets. Every card MUST receive a score. Return JSON {\"assessments\":[{\"risk_score\":0-100,\"evidence_fields\":[\"exact state field\"],\"evidence_ids\":[\"R1\"],\"uncertainty_flags\":[\"...\"]}]} only."
    if condition == "scaffold_r":
        return common + "Use the schema and policy snippets. You may abstain if the card is materially incomplete. Return JSON {\"assessments\":[{\"risk_score\":0-100|null,\"abstain\":true|false,\"evidence_fields\":[\"exact state field\"],\"evidence_ids\":[\"R1\"],\"uncertainty_flags\":[\"...\"]}]} only."
    raise ValueError(condition)


def call_model(payload: dict[str, Any], api_key: str) -> tuple[str, dict[str, Any]]:
    body = {"model": MODEL, "max_tokens": 4096, "temperature": 0, "messages": [{"role": "user", "content": canonical_json(payload)}]}
    with httpx.Client(timeout=120) as client:
        response = client.post(API_URL, headers={"x-api-key": api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"}, json=body)
    if response.status_code != 200:
        raise RuntimeError(f"MiniMax failed with HTTP {response.status_code}: {response.text[:300]}")
    raw = "".join(part.get("text", "") for part in response.json().get("content", []))
    match = re.search(r"\{.*\}", raw, flags=re.S)
    if not match:
        raise RuntimeError("MiniMax response did not contain a JSON object")
    return raw, json.loads(match.group(0))


def validate_assessments(response: dict[str, Any], count: int) -> list[dict[str, Any]]:
    items = response.get("assessments")
    if not isinstance(items, list) or len(items) != count:
        raise RuntimeError(f"Expected {count} assessments, received {type(items).__name__} length {len(items) if isinstance(items, list) else 'n/a'}")
    result = []
    for item in items:
        if not isinstance(item, dict):
            raise RuntimeError("Assessment must be an object")
        score = item.get("risk_score")
        if score is not None:
            score = max(0.0, min(100.0, float(score)))
        result.append({**item, "risk_score": score})
    return result


def critic(batch_cards: list[dict[str, Any]], assessments: list[dict[str, Any]], api_key: str) -> list[dict[str, Any]]:
    payload = {
        "role": "independent audit critic",
        "instruction": "Check each assessment against its state card. ACCEPT only if score is present, all evidence_fields are exact state keys, and evidence_ids are drawn from the policy snippets. Otherwise ABSTAIN. Return only JSON {\"verdicts\":[{\"decision\":\"ACCEPT|ABSTAIN\",\"reasons\":[\"...\"]}]}",
        "policy": RETRIEVAL_POLICY,
        "cards": batch_cards,
        "assessments": assessments,
    }
    _, response = call_model(payload, api_key)
    verdicts = response.get("verdicts")
    if not isinstance(verdicts, list) or len(verdicts) != len(batch_cards):
        raise RuntimeError("Critic returned an invalid verdict batch")
    return verdicts


def bootstrap(y: np.ndarray, score: np.ndarray, rng: np.random.Generator, draws: int = 1000) -> list[float] | None:
    if len(y) < 2 or len(np.unique(y)) < 2:
        return None
    values = []
    for _ in range(draws):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) == 2:
            values.append(float(roc_auc_score(y[idx], score[idx])))
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def metrics(records: list[dict[str, Any]], y: np.ndarray) -> dict[str, Any]:
    accepted = np.array([r["accepted"] for r in records], dtype=bool)
    scores = np.array([np.nan if r["score"] is None else r["score"] / 100 for r in records], dtype=float)
    valid = accepted & np.isfinite(scores)
    output: dict[str, Any] = {"n": int(len(y)), "accepted": int(valid.sum()), "abstained": int((~valid).sum()), "coverage": float(valid.mean())}
    if valid.sum() and len(np.unique(y[valid])) == 2:
        rng = np.random.default_rng(SEED)
        output.update({
            "auroc": float(roc_auc_score(y[valid], scores[valid])),
            "auroc_bootstrap_95ci": bootstrap(y[valid], scores[valid], rng),
            "auprc": float(average_precision_score(y[valid], scores[valid])),
            "brier": float(brier_score_loss(y[valid], scores[valid])),
        })
    else:
        output["metric_error"] = "No evaluable outcome variation after abstention"
    return output


def run_condition(
    condition: str,
    cards: list[dict[str, Any]],
    y: np.ndarray,
    api_key: str,
    batch_size: int,
    max_new_batches: int = 0,
    write_result: bool = True,
    batch_numbers: set[int] | None = None,
) -> dict[str, Any]:
    condition_dir = OUT / condition
    condition_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    new_batches = 0
    for start in range(0, len(cards), batch_size):
        batch = cards[start:start + batch_size]
        batch_no = start // batch_size + 1
        if batch_numbers is not None and batch_no not in batch_numbers:
            continue
        ledger_path = condition_dir / f"ledger_batch_{batch_no:03d}.json"
        if ledger_path.exists():
            try:
                cached = json.loads(ledger_path.read_text(encoding="utf-8"))
                cached_records = cached.get("records")
                if (
                    cached.get("condition") == condition
                    and isinstance(cached_records, list)
                    and len(cached_records) == len(batch)
                    and [record.get("study_row") for record in cached_records]
                    == [card["study_row"] for card in batch]
                ):
                    records.extend(cached_records)
                    print(f"{condition}: {min(start + batch_size, len(cards))}/{len(cards)} (resumed)", flush=True)
                    continue
            except (OSError, json.JSONDecodeError, TypeError):
                pass
        request = {"condition": condition, "instruction": condition_instruction(condition), "policy": RETRIEVAL_POLICY if condition != "direct_prompt" else {}, "state_cards": batch}
        request_hash = sha256(request)
        raw, response = call_model(request, api_key)
        assessments = validate_assessments(response, len(batch))
        verdicts: list[dict[str, Any]] = [{"decision": "ACCEPT", "reasons": []} for _ in batch]
        if condition == "scaffold_r":
            verdicts = critic(batch, assessments, api_key)
        response_hash = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        batch_records = []
        for card, assessment, verdict in zip(batch, assessments, verdicts, strict=True):
            accepted = assessment.get("risk_score") is not None and verdict.get("decision") == "ACCEPT"
            batch_records.append({"study_row": card["study_row"], "input_hash": sha256(card), "request_hash": request_hash, "response_hash": response_hash, "score": assessment.get("risk_score"), "accepted": accepted, "assessment": assessment, "critic": verdict})
        ledger = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "model": MODEL, "condition": condition, "batch": batch_no, "request_hash": request_hash, "response_hash": response_hash, "records": batch_records}
        ledger_path.write_text(json.dumps(ledger, ensure_ascii=False, indent=2), encoding="utf-8")
        records.extend(batch_records)
        new_batches += 1
        print(f"{condition}: {min(start + batch_size, len(cards))}/{len(cards)}", flush=True)
        time.sleep(0.2)
        if max_new_batches and new_batches >= max_new_batches:
            break
    records.sort(key=lambda x: x["study_row"])
    if not write_result:
        return {"condition": condition, "metrics": {"status": "partial_or_resumed"}, "records": []}
    result = {"condition": condition, "metrics": metrics(records, y), "records": records}
    (condition_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=300, help="Stratified public evaluation cohort size")
    parser.add_argument("--batch-size", type=int, default=15)
    parser.add_argument(
        "--max-new-batches", type=int, default=0,
        help="Process at most this many uncached batches per invocation (0 means all).",
    )
    parser.add_argument(
        "--no-summary", action="store_true",
        help="Skip metric calculation when running a short resumable batch segment.",
    )
    parser.add_argument(
        "--batch-numbers", nargs="*", type=int,
        help="Process only these 1-based batch numbers; used by the parallel resumer.",
    )
    parser.add_argument("--conditions", nargs="*", default=["direct_prompt", "retrieval_prompt", "scaffold_no_critic", "scaffold_no_abstention", "scaffold_r"])
    args = parser.parse_args()
    api_key = os.environ.get("MINIMAX_API_KEY")
    if not api_key:
        raise RuntimeError("MINIMAX_API_KEY is required and must not be saved in project files")
    OUT.mkdir(parents=True, exist_ok=True)
    cards, y, metadata = load_cohort(args.n)
    # This file deliberately includes no UCI ID column and no outcome value.
    (OUT / "frozen_state_cards.json").write_text(json.dumps(cards, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "protocol.json").write_text(json.dumps({**metadata, "model": MODEL, "conditions": args.conditions, "batch_size": args.batch_size, "policy": RETRIEVAL_POLICY}, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.no_summary:
        for condition in args.conditions:
            run_condition(
                condition, cards, y, api_key, args.batch_size, args.max_new_batches,
                write_result=False, batch_numbers=set(args.batch_numbers) if args.batch_numbers else None,
            )
        return
    all_results = [
        run_condition(
            condition, cards, y, api_key, args.batch_size, args.max_new_batches,
            batch_numbers=set(args.batch_numbers) if args.batch_numbers else None,
        )
        for condition in args.conditions
    ]
    summary = {"metadata": metadata, "conditions": [{"condition": r["condition"], **r["metrics"]} for r in all_results]}
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
