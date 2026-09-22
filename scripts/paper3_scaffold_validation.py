"""Blind, auditable MiniMax-M3 validation for the Paper 3 cohort."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import pyreadstat
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score

ROOT = Path(r"E:\Nodel\ExerciseProject\AutoResearchClaw")
CSV = Path(r"E:\Nodel\ExerciseProject\paper_3\data\data.csv")
SAV = Path(r"E:\Nodel\ExerciseProject\paper_3\data\data.sav")
OUT = ROOT / "artifacts" / "paper3_scaffold_clinical"
API_URL = "https://api.minimaxi.com/anthropic/v1/messages"
MODEL = "MiniMax-M3"
OUTCOME = "抗凝药后是否出血"
BASELINE_FIELDS = [
    "性别", "年龄", "学历", "婚姻情况", "身高（cm）", "入院体重（kg）", "入院spO2", "新冠疫苗接种情况", "是否血透", "血透年限", "是否有透析管道", "透析管道类型", "新冠分型", "Q高血压", "Q糖尿病", "Q冠心病", "Q脑梗", "Q肿瘤", "Q紫癜", "Q红斑狼疮", "QCOPD", "Q肾肝功能不全", "Q尿毒症", "是否使用抗凝药物", "Y速碧林", "Y肝素钠", "Y克赛", "Y利伐沙班", "Y黄达肝葵钠", "has—bled", "caprini", "入血红蛋白", "入PLT血小板计数", "入WBC白细胞计数", "入RBC", "入PT", "入APTT", "入FIB", "入D—D", "入tbi", "入ALT", "入ALB", "入SCr", "入egfr", "入pct", "入LI—6", "入CRP", "入尿素", "入胆固醇", "入尿酸",
]

def get_labels():
    _, meta = pyreadstat.read_sav(SAV, metadataonly=True)
    return {c: {str(k): str(v) for k, v in v.items()} for c, v in meta.variable_value_labels.items()}

def display(col, value, mappings):
    if pd.isna(value) or str(value).strip() == "":
        return "未记录"
    key = str(float(value)) if isinstance(value, (int, float, np.number)) else str(value)
    return mappings.get(col, {}).get(key, str(value))

def make_card(row, mappings):
    return {c: display(c, row[c], mappings) for c in BASELINE_FIELDS if c in row.index}

def make_prompt(cards):
    instruction = """You are the frozen inference component of SCAFFOLD-Clinical, an exploratory research-only zero-shot clinical-state audit. The outcome is withheld. Do not make treatment recommendations, diagnose, infer undocumented units, or use facts outside the state cards. Return ONLY JSON: {\"assessments\":[...]}. Preserve input order. Every assessment must contain risk_score (integer 0-100), risk_tier (low|intermediate|high), evidence_fields (2-6 exact field names from the state card), uncertainty_flags (array), and data_quality_flags (array). Scores are exploratory ranks, not clinical predictions. Do not mention drug doses or reference intervals.\nSTATE_CARDS="""
    return instruction + json.dumps(cards, ensure_ascii=False, separators=(",", ":"))

def ask(cards, key, audit_dir, batch_number):
    body = {"model": MODEL, "max_tokens": 4000, "temperature": 0, "messages": [{"role": "user", "content": make_prompt(cards)}]}
    with httpx.Client(timeout=90) as client:
        for attempt in range(3):
            response = client.post(API_URL, headers={"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"}, json=body)
            if response.status_code == 200:
                text = "".join(x.get("text", "") for x in response.json().get("content", []))
                match = re.search(r"\{.*\}", text, flags=re.S)
                try:
                    data = json.loads(match.group(0)) if match else {}
                except json.JSONDecodeError:
                    # An incomplete JSON response is invalid evidence, not a result.
                    # Retry the same outcome-blind request rather than repairing it.
                    time.sleep(2 ** attempt)
                    continue
                if isinstance(data.get("assessments"), list) and len(data["assessments"]) == len(cards):
                    # This immutable record deliberately contains no outcome field.  It is
                    # written before the evaluator is allowed to load the endpoint.
                    request_text = make_prompt(cards)
                    record = {
                        "batch": batch_number,
                        "created_at_utc": datetime.now(timezone.utc).isoformat(),
                        "model": MODEL,
                        "request_sha256": hashlib.sha256(request_text.encode("utf-8")).hexdigest(),
                        "response_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                        "state_cards": cards,
                        "response_text": text,
                    }
                    tmp = audit_dir / f"batch-{batch_number:02d}.json.tmp"
                    final = audit_dir / f"batch-{batch_number:02d}.json"
                    with tmp.open("w", encoding="utf-8") as fh:
                        json.dump(record, fh, ensure_ascii=False, indent=2)
                    os.replace(tmp, final)
                    return data["assessments"]
            time.sleep(2 ** attempt)
    raise RuntimeError(f"MiniMax request failed: {response.status_code} {response.text[:300]}")

def load_completed_batch(path, expected_count):
    """Resume only from a complete, hash-recorded response; never synthesize a batch."""
    if not path.exists():
        return None
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
        text = record["response_text"]
        match = re.search(r"\{.*\}", text, flags=re.S)
        data = json.loads(match.group(0)) if match else {}
        values = data.get("assessments")
        return values if isinstance(values, list) and len(values) == expected_count else None
    except (OSError, KeyError, TypeError, json.JSONDecodeError):
        return None

def auc_ci(y, scores, rng):
    point = float(roc_auc_score(y, scores)); values = []
    for _ in range(2000):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) == 2:
            values.append(roc_auc_score(y[idx], scores[idx]))
    return point, float(np.quantile(values, .025)), float(np.quantile(values, .975))

def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--dry-run", action="store_true"); args = parser.parse_args()
    key = os.getenv("MINIMAX_API_KEY")
    if not key and not args.dry_run: raise RuntimeError("MINIMAX_API_KEY is required; do not store it in files")
    OUT.mkdir(parents=True, exist_ok=True)
    audit_dir = OUT / "write_ahead_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(CSV, encoding="utf-8-sig"); mappings = get_labels()
    cards = [make_card(row, mappings) for _, row in df.iterrows()]
    with (OUT / "frozen_state_cards.json").open("w", encoding="utf-8") as f: json.dump(cards, f, ensure_ascii=False, indent=2)
    assessments = []
    for start in range(0, len(cards), 6):
        batch_cards = cards[start:start+6]
        batch_number = start // 6 + 1
        batch = ([{"risk_score": 50, "risk_tier": "intermediate", "evidence_fields": [], "uncertainty_flags": ["dry_run"], "data_quality_flags": []}] * len(batch_cards)) if args.dry_run else load_completed_batch(audit_dir / f"batch-{batch_number:02d}.json", len(batch_cards))
        if batch is None:
            batch = ask(batch_cards, key, audit_dir, batch_number)
        assessments.extend(batch); print(f"Scored {min(start+6, len(cards))}/{len(cards)}", flush=True)
    lock = []
    for i, a in enumerate(assessments):
        payload = {"row_index": i, "risk_score": int(a.get("risk_score", 50)), **a}
        lock.append({"row_index": i, "assessment_sha256": hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest(), "locked_at_utc": datetime.now(timezone.utc).isoformat(), "assessment": payload})
    with (OUT / "write_ahead_lock.json").open("w", encoding="utf-8") as f:
        json.dump({"protocol": "outcome-blind; endpoint is absent from every lock record", "records": lock}, f, ensure_ascii=False, indent=2)
    y = (pd.to_numeric(df[OUTCOME], errors="coerce") == 1).astype(int).to_numpy()
    scores = np.array([max(0, min(100, int(a.get("risk_score", 50)))) for a in assessments], dtype=float)
    cohort_fields = ["年龄", "性别", "是否血透", "新冠分型", "是否使用抗凝药物", "has—bled", "caprini", "住院天数", "入血红蛋白", "入PLT血小板计数", "入WBC白细胞计数", "入PT", "入APTT", "入FIB", "入D—D", "入ALB", "入SCr", "入egfr", "入CRP"]
    cohort_summary = {"n": int(len(df)), "bleeding_events": int(y.sum()), "non_bleeding": int((1-y).sum()), "variables": {}}
    categorical = {"性别", "是否血透", "新冠分型", "是否使用抗凝药物"}
    for col in cohort_fields:
        x = pd.to_numeric(df[col], errors="coerce")
        if col in categorical:
            cohort_summary["variables"][col] = {"bleeding_counts": {str(k): int(v) for k,v in x[y==1].value_counts(dropna=False).items()}, "non_bleeding_counts": {str(k): int(v) for k,v in x[y==0].value_counts(dropna=False).items()}, "missing": int(x.isna().sum())}
        else:
            cohort_summary["variables"][col] = {"bleeding_median_iqr": [float(x[y==1].median()), float(x[y==1].quantile(.25)), float(x[y==1].quantile(.75))], "non_bleeding_median_iqr": [float(x[y==0].median()), float(x[y==0].quantile(.25)), float(x[y==0].quantile(.75))], "missing": int(x.isna().sum())}
    with (OUT / "cohort_summary.json").open("w", encoding="utf-8") as f: json.dump(cohort_summary, f, ensure_ascii=False, indent=2)
    # Outcome linkage is intentionally a distinct, post-lock artifact.
    joined = [{"row_index": i, "assessment_sha256": lock[i]["assessment_sha256"], "recorded_bleeding": int(y[i])} for i in range(len(y))]
    with (OUT / "post_lock_outcome_join.json").open("w", encoding="utf-8") as f: json.dump(joined, f, ensure_ascii=False, indent=2)
    rng = np.random.default_rng(20260725); point, lo, hi = auc_ci(y, scores, rng)
    metrics = {"n": int(len(y)), "bleeding_events": int(y.sum()), "algorithm": "SCAFFOLD-Clinical", "model": MODEL, "scaffold_auc": {"estimate": point, "bootstrap_95ci": [lo, hi]}, "score_by_outcome": {"bleeding_median": float(np.median(scores[y == 1])), "non_bleeding_median": float(np.median(scores[y == 0])), "mann_whitney_p": float(mannwhitneyu(scores[y == 1], scores[y == 0], alternative="two-sided").pvalue)}}
    for col, name in [("has—bled", "has_bled"), ("caprini", "caprini")]:
        x = pd.to_numeric(df[col], errors="coerce"); valid = x.notna().to_numpy()
        if valid.sum() and len(np.unique(y[valid])) == 2:
            a, b, c = auc_ci(y[valid], x[valid].to_numpy(), rng); metrics[name + "_auc"] = {"n": int(valid.sum()), "estimate": a, "bootstrap_95ci": [b, c]}
    with (OUT / "validation_metrics.json").open("w", encoding="utf-8") as f: json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

if __name__ == "__main__": main()
