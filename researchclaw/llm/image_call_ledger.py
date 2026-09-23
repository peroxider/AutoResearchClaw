"""Run-level image-model call ledger.

Stage 22's image-generation providers call their endpoints through raw
``urllib`` instead of the ledgered chat clients, so before this module their
attempts were invisible to run-level cost accounting: the stage-local
``framework_diagram_generation.json`` manifest froze provider/mode/status,
but nothing aggregated them per run. Every provider attempt — succeeded or
failed — is recorded here with the serving identity (provider, model,
endpoint), duration, returned image size and any usage block the endpoint
reported. The runner freezes one ``image_call_ledger.json`` per run beside
the chat call ledger, and final acceptance requires the bundle's diagram
manifest attempts to be covered by it.

This remains an auditable lower bound, not a billing report: endpoints that
do not report token usage leave it null — never inferred — and calls made
without going through a recording provider are still not counted.

The runner freezes the ledger unguarded at end of run, so a malformed record
fails the run there (fail-closed) instead of shipping an unvalidated file.
"""
from __future__ import annotations

import json
from collections import deque
from pathlib import Path

from researchclaw.pipeline.evidence_store import content_hash

# Bounded like the chat-call registry: process accumulation must not grow
# without limit across stage retries.
_RECORDS: deque = deque(maxlen=512)

# "provider" is the hybrid loop's tag for a failed plain generate — the
# manifest attempt ledger's four modes stay the ledger's four modes.
_MODES = ("prompt", "reference", "text_free", "provider")


def record_image_call(record: dict) -> None:
    _RECORDS.append(dict(record))


def reset_image_call_ledger() -> None:
    """Drop every recorded image call (test isolation)."""
    _RECORDS.clear()


def collect_image_call_records() -> list[dict]:
    return [dict(record) for record in _RECORDS]


def _validate_record(record: dict) -> None:
    if not isinstance(record, dict):
        raise ValueError("Image call ledger is malformed")
    provider = record.get("provider")
    if not isinstance(provider, str) or not provider:
        raise ValueError("Image call ledger is malformed")
    if record.get("mode") not in _MODES:
        raise ValueError("Image call ledger is malformed")
    status = record.get("status")
    if status not in ("succeeded", "failed"):
        raise ValueError("Image call ledger is malformed")
    for field in ("model", "endpoint", "error_type"):
        value = record.get(field)
        if value is not None and (not isinstance(value, str) or not value):
            raise ValueError("Image call ledger is malformed")
    # A failed attempt names the exception type; a succeeded one has none.
    if (status == "failed") != (record.get("error_type") is not None):
        raise ValueError("Image call ledger is malformed")
    # Duration is measured locally, so unlike endpoint-reported usage it is
    # always known and must be present.
    if type(record.get("duration_ms")) is not int or record["duration_ms"] < 0:
        raise ValueError("Image call ledger is malformed")
    image_bytes = record.get("image_bytes")
    if image_bytes is not None and (type(image_bytes) is not int or image_bytes < 0):
        raise ValueError("Image call ledger is malformed")
    usage = record.get("usage")
    if usage is not None and not isinstance(usage, dict):
        raise ValueError("Image call ledger is malformed")


def image_call_totals(records: list[dict]) -> dict:
    failures = sum(1 for record in records if record.get("status") == "failed")
    sizes = [record.get("image_bytes") for record in records]
    if any(value is not None and type(value) is not int for value in sizes):
        total_bytes = None
    elif not any(value is not None for value in sizes):
        total_bytes = None
    else:
        total_bytes = sum(value for value in sizes if value is not None)
    return {"calls": len(records), "failures": failures,
            "total_image_bytes": total_bytes}


def build_image_call_ledger(records: list[dict]) -> dict:
    """Freeze a validated ledger document; totals re-derive from the calls."""
    calls = [dict(record) for record in records]
    for record in calls:
        _validate_record(record)
    ledger = {"schema_version": 1, "calls": calls,
              "totals": image_call_totals(calls)}
    ledger["version"] = content_hash(ledger)
    return ledger


def validate_image_call_ledger(document: dict) -> None:
    if (not isinstance(document, dict) or document.get("schema_version") != 1
            or not isinstance(document.get("calls"), list)
            or any(not isinstance(record, dict) for record in document["calls"])):
        raise ValueError("Image call ledger is malformed")
    for record in document["calls"]:
        _validate_record(record)
    payload = {key: value for key, value in document.items() if key != "version"}
    if document.get("version") != content_hash(payload):
        raise ValueError("Image call ledger version changed")
    if document.get("totals") != image_call_totals(document["calls"]):
        raise ValueError("Image call ledger totals differ from the recorded calls")


def write_image_call_ledger(root: Path) -> dict | None:
    """Write image_call_ledger.json for the run; None when nothing recorded.

    Recorded calls are cleared only after the file is on disk, so a write
    failure cannot silently lose them.
    """
    records = collect_image_call_records()
    if not records:
        return None
    ledger = build_image_call_ledger(records)
    (root / "image_call_ledger.json").write_text(
        json.dumps(ledger, indent=2), encoding="utf-8")
    _RECORDS.clear()
    return ledger
