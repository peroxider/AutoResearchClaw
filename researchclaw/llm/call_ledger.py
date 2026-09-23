"""Run-level LLM call ledger.

Individual clients record what actually served each chat call: the
server-reported model (never assumed equal to the requested display name),
the endpoint, request parameters, token usage, fallback failures and
duration. The runner writes one frozen ``llm_call_ledger.json`` per run.
This is an auditable lower bound: calls made outside registered clients
(raw API callers) are not counted, the registry evicts the oldest clients
beyond a bound, and a run with zero recorded calls writes no artifact
rather than implying zero cost.
"""
from __future__ import annotations

import json
from collections import deque
from pathlib import Path

from researchclaw.pipeline.evidence_store import content_hash

# Strong refs to record lists, not client objects: stage-local clients may be
# garbage-collected before end-of-run collection, while their completed call
# records must survive. Keeping the list avoids extending network/session
# client lifetimes. The bound prevents unbounded process accumulation.
_CLIENTS: deque = deque(maxlen=512)


def register_client(client) -> None:
    records = getattr(client, "_call_records", None)
    if not isinstance(records, list):
        raise TypeError("Registered LLM client must expose a list _call_records")
    _CLIENTS.append(records)


def reset_call_ledger() -> None:
    """Drop every registered client and its records (test isolation)."""
    for records in _CLIENTS:
        records.clear()
    _CLIENTS.clear()


def collect_call_records() -> list[dict]:
    """Snapshot every recorded chat call from registered clients."""
    records: list[dict] = []
    for client_records in _CLIENTS:
        records.extend(dict(record) for record in client_records)
    return records


def _token_total(records: list[dict], field: str) -> int | None:
    values = [record.get(field) for record in records]
    if any(value is not None and type(value) is not int for value in values):
        return None
    if not any(value is not None for value in values):
        return None
    return sum(value for value in values if value is not None)


def call_ledger_totals(records: list[dict]) -> dict:
    failures = sum(1 for record in records if record.get("status") == "failed")
    return {"calls": len(records), "failures": failures,
            "prompt_tokens": _token_total(records, "prompt_tokens"),
            "completion_tokens": _token_total(records, "completion_tokens"),
            "total_tokens": _token_total(records, "total_tokens")}


def build_call_ledger(records: list[dict]) -> dict:
    """Freeze a validated ledger document; totals re-derive from the calls."""
    calls = [dict(record) for record in records]
    ledger = {"schema_version": 1, "calls": calls,
              "totals": call_ledger_totals(calls)}
    ledger["version"] = content_hash(ledger)
    return ledger


def validate_call_ledger(document: dict) -> None:
    if (not isinstance(document, dict) or document.get("schema_version") != 1
            or not isinstance(document.get("calls"), list)
            or any(not isinstance(record, dict) for record in document["calls"])):
        raise ValueError("LLM call ledger is malformed")
    for record in document["calls"]:
        if record.get("status") not in ("succeeded", "failed"):
            raise ValueError("LLM call ledger is malformed")
        if not isinstance(record.get("fallback_failures"), list):
            raise ValueError("LLM call ledger is malformed")
        for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = record.get(field)
            # bool is an int subclass; a boolean token count is malformed.
            if value is not None and type(value) is not int:
                raise ValueError("LLM call ledger is malformed")
    payload = {key: value for key, value in document.items() if key != "version"}
    if document.get("version") != content_hash(payload):
        raise ValueError("LLM call ledger version changed")
    if document.get("totals") != call_ledger_totals(document["calls"]):
        raise ValueError("LLM call ledger totals differ from the recorded calls")


def write_call_ledger(root: Path) -> dict | None:
    """Write llm_call_ledger.json for the run; None when nothing was recorded.

    Recorded calls are cleared only after the file is on disk, so a write
    failure cannot silently lose them.
    """
    records = collect_call_records()
    if not records:
        return None
    ledger = build_call_ledger(records)
    (root / "llm_call_ledger.json").write_text(
        json.dumps(ledger, indent=2), encoding="utf-8")
    for client_records in _CLIENTS:
        client_records.clear()
    return ledger
