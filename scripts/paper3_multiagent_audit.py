"""Four-role outcome-blinded audit layer for SCAFFOLD-Clinical.

Roles are independent, schema-constrained MiniMax-M3 calls: clinical reasoner,
counterfactual quality controller, leakage auditor, and evidence adjudicator.
No role receives the recorded outcome or discharge data.
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import httpx
import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(r"E:\Nodel\ExerciseProject\AutoResearchClaw")
OUT = ROOT / "artifacts" / "paper3_scaffold_clinical"
API_URL = "https://api.minimaxi.com/anthropic/v1/messages"
MODEL = "MiniMax-M3"

COMMON = """You are a research-only audit agent. The input is a frozen, de-identified baseline state card. The bleeding outcome, thromboembolism outcome, discharge values, and medication-dose units are prohibited and absent. Do not recommend treatment, diagnose, assume undocumented units, or introduce external facts. Return valid JSON only."""

def call(prompt: str, key: str) -> dict:
    body = {"model": MODEL, "max_tokens": 3500, "temperature": 0, "messages": [{"role": "user", "content": prompt}]}
    with httpx.Client(timeout=120) as client:
        for attempt in range(3):
            r = client.post(API_URL, headers={"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"}, json=body)
            if r.status_code == 200:
                text = "".join(x.get("text", "") for x in r.json().get("content", []))
                hit = re.search(r"\{.*\}", text, flags=re.S)
                if hit:
                    return json.loads(hit.group(0))
            time.sleep(2 ** attempt)
    raise RuntimeError(f"MiniMax request failed: {r.status_code} {r.text[:200]}")

def role_prompt(role: str, cards: list[dict], upstream: list[dict] | None = None) -> str:
    payload = {"state_cards": cards}
    if upstream is not None: payload["upstream"] = upstream
    specs = {
        "clinical": "Return {assessments:[{candidate_score:integer_0_to_100,evidence_fields:array_of_exact_input_field_names,uncertainty_flags:array}]}, preserving order. Score is an exploratory risk rank, never a clinical prediction.",
        "counterfactual": "Return {assessments:[{stability:integer_0_to_100,masking_response:acceptable|insufficient,contradictions:array,uncertainty_flags:array}]}, preserving order. Audit whether removing/masking any cited evidence would require more uncertainty; do not calculate a clinical counterfactual or invent values.",
        "leakage": "Return {assessments:[{leakage_pass:boolean,forbidden_field_detected:array,unsupported_evidence:array}]}, preserving order. Fail if an output refers to a non-input, outcome, discharge, or undocumented dose/unit field.",
        "adjudicator": "Return {assessments:[{final_score:integer_0_to_100,decision:accept|abstain,evidence_fields:array,abstention_reasons:array}]}, preserving order. Accept only when leakage_pass is true, evidence fields are exact input fields, and uncertainty is acknowledged. Otherwise abstain and return final_score=50.",
    }
    return COMMON + "\nROLE=" + role + "\n" + specs[role] + "\nPAYLOAD=" + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

def run_role(role: str, cards: list[dict], key: str, upstream: list[dict] | None = None) -> list[dict]:
    out = []
    for start in range(0, len(cards), 24):
        data = call(role_prompt(role, cards[start:start+24], None if upstream is None else upstream[start:start+24]), key)
        values = data.get("assessments", [])
        if len(values) != len(cards[start:start+24]): raise RuntimeError(f"{role}: assessment length mismatch")
        out.extend(values); print(f"{role}: {min(start+24,len(cards))}/{len(cards)}", flush=True)
    return out

def boot_auc(y: np.ndarray, score: np.ndarray) -> dict:
    rng = np.random.default_rng(20260725); values=[]
    for _ in range(2000):
        idx=rng.integers(0,len(y),len(y))
        if len(np.unique(y[idx]))==2: values.append(roc_auc_score(y[idx],score[idx]))
    return {"estimate":float(roc_auc_score(y,score)),"bootstrap_95ci":[float(np.quantile(values,.025)),float(np.quantile(values,.975))]}

def main() -> None:
    key=os.environ.get("MINIMAX_API_KEY")
    if not key: raise RuntimeError("MINIMAX_API_KEY is required")
    cards=json.loads((OUT/"frozen_state_cards.json").read_text(encoding="utf-8"))
    clinical=run_role("clinical",cards,key)
    counterfactual=run_role("counterfactual",cards,key,clinical)
    leakage=run_role("leakage",cards,key,clinical)
    upstream=[{"clinical":c,"counterfactual":q,"leakage":l} for c,q,l in zip(clinical,counterfactual,leakage)]
    adjudicated=run_role("adjudicator",cards,key,upstream)
    prior=json.loads((OUT/"locked_blind_assessments.json").read_text(encoding="utf-8"))
    y=np.array([r["outcome_held_out_bleeding"] for r in prior]); scores=np.array([int(a.get("final_score",50)) for a in adjudicated])
    records=[{"row_index":i,"final_score":int(scores[i]),"outcome_held_out_bleeding":int(y[i]),"clinical_reasoner":clinical[i],"counterfactual_qc":counterfactual[i],"leakage_audit":leakage[i],"evidence_adjudicator":adjudicated[i]} for i in range(len(cards))]
    (OUT/"multiagent_locked_assessments.json").write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding="utf-8")
    metrics={"algorithm":"SCAFFOLD-Clinical-MA","model":MODEL,"n":int(len(y)),"bleeding_events":int(y.sum()),"auc":boot_auc(y,scores),"accepted":int(sum(a.get("decision")=="accept" for a in adjudicated)),"leakage_passed":int(sum(bool(a.get("leakage_pass")) for a in leakage)),"median_score_bleeding":float(np.median(scores[y==1])),"median_score_non_bleeding":float(np.median(scores[y==0]))}
    (OUT/"multiagent_validation_metrics.json").write_text(json.dumps(metrics,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(metrics,ensure_ascii=False,indent=2))

if __name__ == "__main__": main()
