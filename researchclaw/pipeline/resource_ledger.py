"""Run-level resource ledger.

Individual frozen artifacts record partial call budgets; this module
aggregates exactly what they record into one validated view. Fields the
source artifact does not record stay null — never inferred. Stage calls
outside these artifacts are not counted, so the ledger is not a full API
cost account; it is the auditable lower bound the pipeline can prove.
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


def _llm_chat_row(document: dict) -> tuple[dict | None, str | None]:
    """Re-derive the llm_chat row; any self-inconsistency fails closed."""
    from researchclaw.llm.call_ledger import call_ledger_totals, validate_call_ledger

    try:
        validate_call_ledger(document)
    except ValueError as exc:
        return None, str(exc)
    calls = document["calls"]
    totals = call_ledger_totals(calls)
    return {"source": "llm_call_ledger.json", "kind": "llm_chat",
            "calls": totals["calls"], "limit": None, "failures": totals["failures"],
            "tokens": totals["total_tokens"], "version": document.get("version")}, None


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
        row, problem = _llm_chat_row(llm_ledger)
        if row is None:
            errors.append({"source": "llm_call_ledger.json", "reason": problem})
        else:
            rows.append(row)

    ledger = {"schema_version": 1, "rows": rows, "errors": errors}
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
