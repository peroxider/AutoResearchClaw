"""Run-level resource ledger.

Individual frozen artifacts record partial call budgets; this module
aggregates exactly what they record into one validated view. Fields the
source artifact does not record stay null — never inferred. The chat row
covers registered clients plus every raw caller that records itself (PRM
judge votes, session-end signals); embedding requests are split into their
own row by the records' call family. Beast-Mode subprocess invocations are
declared from the frozen stage logs — the model calls INSIDE the subprocess
are invisible to this process and stay uncounted rather than fabricated.
Stage wall clock aggregates the durations each stage's own stage_health
artifact recorded. Calls outside all recorded sources remain uncounted, so
the ledger is still an auditable lower bound, not a billing report.
"""
from __future__ import annotations

import json
from pathlib import Path

from researchclaw.pipeline.evidence_store import content_hash


class ResourceLedgerError(ValueError):
    pass


def _read(root: Path, name: str, errors: list) -> dict | None:
    path = root / name
    if not path.is_file():
        return None
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        errors.append({"source": name, "reason": str(exc)})
        return None
    if not isinstance(document, dict):
        errors.append({"source": name, "reason": "document is not a JSON mapping"})
        return None
    return document


def _review_row(name: str, document: dict) -> dict:
    budget = document.get("review_budget") or {}
    failures = budget.get("failures")
    return {"source": name, "kind": "llm_review",
            "calls": budget.get("calls"), "limit": budget.get("limit"),
            "failures": len(failures) if isinstance(failures, list) else None,
            "version": document.get("version")}


def _retrieval_row(document: dict) -> dict:
    entries = [entry for ledger in ("query_ledger", "web_ledger")
               if isinstance(document.get(ledger), list)
               for entry in document[ledger] if isinstance(entry, dict)]
    return {"source": "search_meta.json", "kind": "retrieval",
            "calls": len(entries), "limit": None,
            "failures": sum(1 for entry in entries if entry.get("status") == "failed"),
            "cache_only": sum(1 for entry in entries if entry.get("cache_only") is True),
            "version": None}


def _llm_rows(document: dict) -> tuple[list[dict], str | None]:
    """Re-derive the llm_chat and llm_embeddings rows; any self-inconsistency
    fails closed. Records without a call family are chat calls (client
    records predate the split); embeddings ride the same frozen artifact so
    one validation covers both."""
    from researchclaw.llm.call_ledger import call_ledger_totals, validate_call_ledger

    try:
        validate_call_ledger(document)
    except ValueError as exc:
        return [], str(exc)
    calls = document["calls"]
    chat = [record for record in calls
            if record.get("call_family") != "embeddings"]
    embeddings = [record for record in calls
                  if record.get("call_family") == "embeddings"]
    rows: list[dict] = []
    if chat or not embeddings:
        chat_totals = call_ledger_totals(chat)
        rows.append({"source": "llm_call_ledger.json", "kind": "llm_chat",
                     "calls": chat_totals["calls"], "limit": None,
                     "failures": chat_totals["failures"],
                     "tokens": chat_totals["total_tokens"],
                     "version": document.get("version")})
    if embeddings:
        totals = call_ledger_totals(embeddings)
        rows.append({"source": "llm_call_ledger.json", "kind": "llm_embeddings",
                     "calls": totals["calls"], "limit": None,
                     "failures": totals["failures"],
                     "tokens": totals["total_tokens"],
                     "version": document.get("version")})
    return rows, None


def _image_generation_row(document: dict) -> tuple[dict | None, str | None]:
    """Re-derive the image_generation row; any self-inconsistency fails closed."""
    from researchclaw.llm.image_call_ledger import (
        image_call_totals,
        validate_image_call_ledger,
    )

    try:
        validate_image_call_ledger(document)
    except ValueError as exc:
        return None, str(exc)
    totals = image_call_totals(document["calls"])
    return {"source": "image_call_ledger.json", "kind": "image_generation",
            "calls": totals["calls"], "limit": None,
            "failures": totals["failures"],
            "image_bytes": totals["total_image_bytes"],
            "version": document.get("version")}, None


def _stage_glob_rows(root: Path, errors: list, pattern: str,
                     kind: str) -> list[dict]:
    """One declaration row per frozen stage artifact matched by the pattern.

    The artifact records one invocation (beast_mode_log.json); the row
    declares it at run level — what the subprocess did internally is not
    recorded by the artifact and therefore stays null, never inferred."""
    rows: list[dict] = []
    for path in sorted(root.glob(pattern)):
        source = path.relative_to(root).as_posix()
        document = _read(root, source, errors)
        if document is None:
            continue
        success = document.get("success")
        if not isinstance(success, bool):
            continue
        rows.append({"source": source, "kind": kind,
                     "calls": 1, "limit": None,
                     "failures": 0 if success else 1, "version": None})
    return rows


def _stage_wall_clock(root: Path, errors: list) -> dict:
    """Aggregate the per-stage durations each stage_health artifact recorded."""
    entries: list[dict] = []
    for path in sorted(root.glob("stage-*/stage_health.json")):
        source = path.relative_to(root).as_posix()
        document = _read(root, source, errors)
        if document is None:
            continue
        duration = document.get("duration_sec")
        if type(duration) not in (int, float) or isinstance(duration, bool):
            errors.append({"source": source,
                           "reason": "stage_health duration_sec is not a number"})
            continue
        entries.append({"source": source, "duration_sec": duration,
                        "status": document.get("status")})
    total = round(sum(entry["duration_sec"] for entry in entries), 2) \
        if entries else None
    return {"entries": entries, "total_duration_sec": total}


def build_resource_ledger(root: Path) -> dict:
    rows: list[dict] = []
    errors: list[dict] = []

    literature = _read(root, "literature_evidence.json", errors)
    if literature is not None:
        rows.append(_review_row("literature_evidence.json", literature))

    novelty = _read(root, "novelty_matrix.json", errors)
    if novelty is not None:
        # novelty_matrix.json records only review_calls; limit and failures
        # are genuinely unrecorded, so they stay null.
        rows.append({"source": "novelty_matrix.json", "kind": "llm_review",
                     "calls": novelty.get("review_calls"), "limit": None,
                     "failures": None, "version": novelty.get("version")})

    support = _read(root, "citation_support.json", errors)
    if support is not None:
        rows.append(_review_row("citation_support.json", support))

    manuscript = _read(root, "manuscript_ir.json", errors)
    if manuscript is not None:
        rows.append(_review_row("manuscript_ir.json", manuscript))
        sections = manuscript.get("sections")
        attempts = (sum(len(section.get("attempts") or []) for section in sections
                        if isinstance(section, dict))
                    if isinstance(sections, list) else 0)
        rows.append({"source": "manuscript_ir.json", "kind": "writing_attempts",
                     "calls": attempts, "limit": None, "failures": None,
                     "version": manuscript.get("version")})

    peer_review = _read(root, "stage-18/manuscript_peer_review.json", errors)
    if peer_review is not None:
        # stage-18 records calls/limit at the top level; failures are unrecorded.
        rows.append({"source": "stage-18/manuscript_peer_review.json", "kind": "llm_review",
                     "calls": peer_review.get("calls"), "limit": peer_review.get("limit"),
                     "failures": None, "version": peer_review.get("ir_version")})

    search_meta = _read(root, "search_meta.json", errors)
    if search_meta is not None:
        rows.append(_retrieval_row(search_meta))

    validation = _read(root, "method_validation.json", errors)
    if validation is not None:
        calls = validation.get("calls")
        rows.append({"source": "method_validation.json", "kind": "method_probe",
                     "calls": len(calls) if isinstance(calls, list) else None,
                     "limit": None, "failures": None,
                     "version": validation.get("method_version")})

    protocol_budget = _read(root, "protocol_budget.json", errors)
    if protocol_budget is not None:
        rows.append({"source": "protocol_budget.json", "kind": "validation_tuning_trials",
                     "calls": protocol_budget.get("tuning_trials"),
                     "limit": protocol_budget.get("tuning_trial_limit"),
                     "failures": None, "version": protocol_budget.get("protocol_version")})

    llm_ledger = _read(root, "llm_call_ledger.json", errors)
    if llm_ledger is not None:
        llm_rows, problem = _llm_rows(llm_ledger)
        if problem is not None:
            errors.append({"source": "llm_call_ledger.json", "reason": problem})
        else:
            rows.extend(llm_rows)

    image_ledger = _read(root, "image_call_ledger.json", errors)
    if image_ledger is not None:
        row, problem = _image_generation_row(image_ledger)
        if row is None:
            errors.append({"source": "image_call_ledger.json", "reason": problem})
        else:
            rows.append(row)

    rows.extend(_stage_glob_rows(root, errors, "stage-*/beast_mode_log.json",
                                 "code_agent_subprocess"))

    ledger = {"schema_version": 1, "rows": rows, "errors": errors,
              "stage_wall_clock": _stage_wall_clock(root, errors)}
    ledger["version"] = content_hash(ledger)
    return ledger


def validate_resource_ledger(root: Path, ledger: dict) -> None:
    if not isinstance(ledger, dict) or ledger.get("schema_version") != 1:
        raise ResourceLedgerError("Resource ledger is malformed")
    document = dict(ledger)
    if document.pop("version", None) != content_hash(document):
        raise ResourceLedgerError("Resource ledger version changed")
    if ledger != build_resource_ledger(root):
        raise ResourceLedgerError("Resource ledger differs from the frozen artifacts")


def ledger_issues(ledger: dict) -> list[str]:
    """Acceptance-level problems: unreadable sources and impossible budgets."""
    issues = [f"unreadable_resource_source:{error['source']}" for error in ledger.get("errors", [])]
    for row in ledger.get("rows", []):
        calls, limit = row.get("calls"), row.get("limit")
        # bool is an int subclass; a malformed boolean count must not compare.
        if type(calls) is int and type(limit) is int and calls > limit:
            issues.append(f"budget_exceeded:{row['source']}")
    return issues
