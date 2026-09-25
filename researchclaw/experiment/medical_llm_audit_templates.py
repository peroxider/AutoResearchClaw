"""Deterministic Stage-10 templates for auditable clinical LLM API studies.

Templates intentionally replace free-form code synthesis.  Study-specific data
and method choices enter through a frozen JSON contract, never by editing this
module or by asking a model to invent an experiment project.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


TEMPLATE_IDS = frozenset({"binary_risk_audit", "multiclass_triage_audit", "extraction_audit"})


@dataclass(frozen=True)
class MedicalAuditTemplate:
    template_id: str
    required_contract_keys: tuple[str, ...]


CATALOG = {
    "binary_risk_audit": MedicalAuditTemplate("binary_risk_audit", ("source_paths", "runtime")),
    "multiclass_triage_audit": MedicalAuditTemplate("multiclass_triage_audit", ("source_paths", "runtime")),
    "extraction_audit": MedicalAuditTemplate("extraction_audit", ("source_paths", "runtime")),
}


def validate_contract(contract: dict[str, Any]) -> tuple[bool, str]:
    template_id = str(contract.get("executor_template", "binary_risk_audit"))
    template = CATALOG.get(template_id)
    if template is None:
        return False, f"unknown medical audit template: {template_id}"
    missing = [key for key in template.required_contract_keys if not contract.get(key)]
    if missing:
        return False, "missing contract keys: " + ", ".join(missing)
    runtime = contract.get("runtime")
    if not isinstance(runtime, dict):
        return False, "runtime must be an object"
    needed = ("outcome_field", "identifier_fields", "allowed_fields", "provider", "model", "api_key_env", "conditions")
    missing = [key for key in needed if not runtime.get(key)]
    return (not missing, "" if not missing else "missing runtime keys: " + ", ".join(missing))


def write_template(stage_dir: Path, contract: dict[str, Any]) -> tuple[str, ...]:
    """Write a compact, auditable API-evaluation program into ``experiment/``."""
    ok, detail = validate_contract(contract)
    if not ok:
        raise ValueError(f"medical LLM audit contract invalid: {detail}")
    exp_dir = stage_dir / "experiment"
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "data_contract.json").write_text(json.dumps(contract, ensure_ascii=False, indent=2), encoding="utf-8")
    (exp_dir / "main.py").write_text(_MAIN, encoding="utf-8")
    return ("experiment/",)


_MAIN = r'''# ARC_MEDICAL_LLM_AUDIT_TEMPLATE v2
import csv, hashlib, json, math, os, random, sys, time, urllib.request
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CFG = json.loads((ROOT / "data_contract.json").read_text(encoding="utf-8"))
R = CFG["runtime"]

def sha(obj):
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True).encode()).hexdigest()

def auc(labels, scores):
    pos=sum(labels); neg=len(labels)-pos
    if not pos or not neg: return None
    ranked=sorted(zip(scores,labels)); rank_sum=0.0; i=0
    while i<len(ranked):
      j=i+1
      while j<len(ranked) and ranked[j][0]==ranked[i][0]: j+=1
      avg=(i+j+1)/2
      rank_sum += avg*sum(v for _,v in ranked[i:j]); i=j
    return (rank_sum-pos*(pos+1)/2)/(pos*neg)

def bootstrap_auc(labels, scores, n_boot=500, seed=20260803):
    if not labels or not sum(labels) or sum(labels)==len(labels): return [None, None]
    rng=random.Random(seed); vals=[]; n=len(labels)
    for _ in range(n_boot):
      ix=[rng.randrange(n) for _ in range(n)]
      value=auc([labels[i] for i in ix],[scores[i] for i in ix])
      if value is not None: vals.append(value)
    if len(vals)<20: return [None, None]
    vals.sort(); return [vals[int(.025*(len(vals)-1))], vals[int(.975*(len(vals)-1))]]

def calibration(labels, scores, bins=10):
    if not labels: return {"brier":None,"ece":None,"bins":[]}
    brier=sum((s-y)**2 for y,s in zip(labels,scores))/len(labels); out=[]; ece=0.0
    for b in range(bins):
      lo=b/bins; hi=(b+1)/bins; ix=[i for i,s in enumerate(scores) if (lo<=s<hi or (b==bins-1 and s==hi))]
      if ix:
        p=sum(scores[i] for i in ix)/len(ix); obs=sum(labels[i] for i in ix)/len(ix)
        ece += len(ix)/len(labels)*abs(p-obs); out.append({"lower":lo,"upper":hi,"mean_prediction":p,"event_rate":obs,"n":len(ix)})
    return {"brier":brier,"ece":ece,"bins":out}

def decision_curve(labels, scores):
    rows=[]; n=max(1,len(labels)); prevalence=sum(labels)/n
    for threshold in (.10,.20,.30,.40,.50):
      tp=sum(1 for y,s in zip(labels,scores) if y and s>=threshold); fp=sum(1 for y,s in zip(labels,scores) if not y and s>=threshold)
      rows.append({"threshold":threshold,"net_benefit":tp/n-fp/n*threshold/(1-threshold),"treat_all":prevalence-(1-prevalence)*threshold/(1-threshold),"treat_none":0.0})
    return rows

CALLS=[]
def _write_calls():
  (ROOT/"model_call_ledger.json").write_text(json.dumps(CALLS,ensure_ascii=False),encoding="utf-8")

def api(prompt, required_key="risk_probability", meta=None):
    key = os.environ.get(R["api_key_env"])
    if not key: raise RuntimeError("missing API key environment variable")
    url = R.get("base_url", "https://api.minimaxi.com/anthropic").rstrip("/") + "/v1/messages"
    body = {"model": R["model"], "max_tokens": 1024, "messages":[{"role":"user","content":prompt}]}
    last_error=None
    for attempt in range(4):
      meta=meta or {}
      entry={"role":meta.get("role"),"condition":meta.get("condition"),"row_index":meta.get("row_index"),"attempt":attempt+1,"model":R["model"],"endpoint":url}
      started=time.time()
      try:
        req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"content-type":"application/json", "x-api-key":key, "anthropic-version":"2023-06-01"})
        with urllib.request.urlopen(req, timeout=float(R.get("timeout_sec", 60))) as resp:
            data=json.loads(resp.read().decode())
        usage=data.get("usage")
        entry.update({"status":"succeeded","error_type":None,"duration_sec":round(time.time()-started,3),"usage":usage if isinstance(usage,dict) else None,"response_sha":sha(data)})
        CALLS.append(entry); _write_calls()
        text="".join(x.get("text", "") for x in data.get("content", []) if isinstance(x, dict))
        parsed=json.loads(text[text.find("{"):text.rfind("}")+1])
        if required_key and required_key not in parsed: raise ValueError(required_key + " absent")
        return parsed, sha(data)
      except Exception as exc:
        if "status" not in entry:
          entry.update({"status":"failed","error_type":type(exc).__name__,"duration_sec":round(time.time()-started,3),"usage":None,"response_sha":None})
          CALLS.append(entry); _write_calls()
        last_error=exc
        if attempt < 3: time.sleep(2 ** attempt)
    raise RuntimeError("API request/JSON failed after 4 attempts: " + str(last_error))

def run():
    source = Path(CFG["source_paths"][0]); outcome=R["outcome_field"]; ids=set(R["identifier_fields"]); allowed=R["allowed_fields"]
    if not source.exists(): raise FileNotFoundError(source)
    rows=list(csv.DictReader(source.open("r", encoding=R.get("csv_encoding", "utf-8-sig"), newline="")))
    # Some clinical exports are byte-preserving Latin-1 CSVs whose original
    # headers/text were GBK.  The contract makes this conversion explicit.
    recode_to=R.get("text_recode_to", "")
    if recode_to:
      def recode(value):
        try: return value.encode("latin1").decode(recode_to)
        except (UnicodeError, AttributeError): return value
      rows=[{recode(k):recode(v) for k,v in row.items()} for row in rows]
    if outcome not in (rows[0] if rows else {}): raise ValueError("outcome field absent from source")
    ledger=ROOT/"ledger"; ledger.mkdir(exist_ok=True); scored=[]
    for condition in R["conditions"]:
      for index,row in enumerate(rows):
        card={k:row.get(k) for k in allowed if k in row and k not in ids and k != outcome}
        event={"condition":condition,"row_index":index,"state_card":card,"state_card_hash":sha(card),"outcome_joined":False}
        started=time.time()
        try:
          mode={"DirectStructured":"state card only", "RetrievalOnly":"state card plus allowed-field evidence list", "NoCritic":"state card/evidence without independent critique", "NoAbstention":"state card/evidence; abstention forbidden", "FullAuditedAgent":"state card/evidence followed by independent audit critique and abstention"}.get(condition, condition)
          prompt=("Return JSON only: {\\\"risk_probability\\\": number 0..1, \\\"abstain\\\": bool, \\\"rationale\\\": string}. "
                  f"Condition={condition}; protocol={mode}. Outcome is unavailable. State card={json.dumps(card,ensure_ascii=False)}")
          result,response_hash=api(prompt, meta={"role":"scorer","condition":condition,"row_index":index}); score=float(result["risk_probability"]); abstain=bool(result.get("abstain",False)); error=""; critic_hash=""; critic_verdict=None
          if condition=="FullAuditedAgent":
            critic_prompt=("Return JSON only: {\"accept\": bool, \"force_abstain\": bool, \"critique\": string}. "
                           "You are an independent clinical safety critic. You must not infer the hidden outcome. "
                           f"Audit this outcome-blind state card and proposed score. state_card={json.dumps(card,ensure_ascii=False)} proposal={json.dumps(result,ensure_ascii=False)}")
            critic,critic_hash=api(critic_prompt, required_key="accept", meta={"role":"critic","condition":condition,"row_index":index}); critic_verdict=critic
            abstain = abstain or bool(critic.get("force_abstain",False)) or not bool(critic.get("accept",False))
          if condition=="NoAbstention": abstain=False
        except Exception as exc: score=None; abstain=True; response_hash=""; critic_hash=""; critic_verdict=None; error=str(exc)
        event.update({"response_hash":response_hash,"critic_response_hash":critic_hash,"critic_verdict":critic_verdict,"score":score,"abstain":abstain,"latency_sec":time.time()-started,"error":error})
        event["event_hash"]=sha(event); (ledger/f"{condition}_{index}.json").write_text(json.dumps(event,ensure_ascii=False),encoding="utf-8")
        event["outcome"]=row[outcome]; event["outcome_joined"]=True; scored.append(event)
    (ROOT/"patient_results.jsonl").write_text("\n".join(json.dumps(x,ensure_ascii=False) for x in scored),encoding="utf-8")
    def norm_label(value):
      text=str(value).strip()
      try: return str(float(text))
      except ValueError: return text
    positive=set(norm_label(v) for v in R.get("positive_values", ["1"]))
    conditions={}
    for condition in R["conditions"]:
      items=[x for x in scored if x["condition"]==condition]; usable=[x for x in items if x["score"] is not None]
      labels=[1 if norm_label(x["outcome"]) in positive else 0 for x in usable]; scores=[x["score"] for x in usable]
      ci=bootstrap_auc(labels,scores); cal=calibration(labels,scores); dca=decision_curve(labels,scores)
      conditions[condition]={"n":len(items),"n_scored":len(usable),"coverage":len(usable)/max(1,len(items)),"auroc":auc(labels,scores),"auroc_ci_low":ci[0],"auroc_ci_high":ci[1],"brier_score":cal["brier"],"expected_calibration_error":cal["ece"],"calibration_curve":cal["bins"],"decision_curve":dca,"mean_latency_sec":sum(x["latency_sec"] for x in items)/max(1,len(items)),"abstention_rate":sum(bool(x["abstain"]) for x in items)/max(1,len(items)),"critic_calls":sum(1 for x in items if x.get("critic_response_hash")),"critic_accepted":sum(1 for x in items if isinstance(x.get("critic_verdict"),dict) and x["critic_verdict"].get("accept"))}
    metrics={"n_records":len(rows),"n_scored":sum(x["n_scored"] for x in conditions.values()),"coverage":sum(x["n_scored"] for x in conditions.values())/max(1,len(scored)),"conditions":conditions}
    (ROOT/"metrics.json").write_text(json.dumps(metrics,indent=2),encoding="utf-8")
    (ROOT/"results.json").write_text(json.dumps({"metrics":metrics,"conditions":conditions,"contract_hash":sha(CFG),"model_calls":{"attempts":len(CALLS),"failures":sum(1 for c in CALLS if c["status"]=="failed")}},indent=2),encoding="utf-8")
    print("coverage: " + str(metrics["coverage"]))

if __name__ == "__main__": run()
'''
