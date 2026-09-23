"""Run-level resource ledger: the pipeline's accounted call budgets, one
validated view. Unrecorded fields stay null; every row must re-derive from
the frozen artifacts it names."""
import json

import pytest

from researchclaw.pipeline.evidence_store import content_hash
from researchclaw.pipeline.resource_ledger import (
    ResourceLedgerError, build_resource_ledger, ledger_issues, validate_resource_ledger,
)


def write(root, name, document):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document), encoding="utf-8")


def literature_row(**overrides):
    row = {"source": "literature_evidence.json", "kind": "llm_review",
           "calls": 5, "limit": 32, "failures": 1, "version": "lit-v1"}
    row.update(overrides)
    return row


def test_empty_run_yields_ledger_without_rows(tmp_path):
    ledger = build_resource_ledger(tmp_path)
    assert ledger["rows"] == [] and ledger["errors"] == []
    validate_resource_ledger(tmp_path, ledger)


def test_rows_aggregate_every_accounted_budget(tmp_path):
    write(tmp_path, "literature_evidence.json",
          {"version": "lit-v1", "review_budget": {"calls": 5, "limit": 32, "failures": ["a"]}})
    write(tmp_path, "novelty_matrix.json", {"version": "nov-v1", "review_calls": 3})
    write(tmp_path, "citation_support.json",
          {"version": "cit-v1", "review_budget": {"calls": 9, "limit": 128, "failures": []}})
    write(tmp_path, "manuscript_ir.json",
          {"version": "ir-v1", "review_budget": {"calls": 6, "limit": 96, "failures": ["y"]},
           "sections": [{"attempts": [1, 2]}, {"attempts": [3]}, "junk"]})
    write(tmp_path, "stage-18/manuscript_peer_review.json",
          {"ir_version": "ir-v1", "calls": 7, "limit": 40})
    write(tmp_path, "search_meta.json",
          {"status": "results_found", "query_ledger": [
              {"query": "q1", "provider": "openalex", "status": "results_found", "cache_only": False},
              {"query": "q2", "provider": "openalex", "status": "failed", "cache_only": False}],
           "web_ledger": [{"provider": "fetch", "status": "cache_only", "cache_only": True}]})
    write(tmp_path, "method_validation.json", {"method_version": "m-v1", "calls": [{"id": "c1"}]})

    ledger = build_resource_ledger(tmp_path)
    by_source = {(row["source"], row["kind"]): row for row in ledger["rows"]}
    assert by_source[("literature_evidence.json", "llm_review")] == literature_row()
    assert by_source[("novelty_matrix.json", "llm_review")] == {
        "source": "novelty_matrix.json", "kind": "llm_review",
        "calls": 3, "limit": None, "failures": None, "version": "nov-v1"}
    assert by_source[("citation_support.json", "llm_review")]["calls"] == 9
    assert by_source[("manuscript_ir.json", "llm_review")] == {
        "source": "manuscript_ir.json", "kind": "llm_review",
        "calls": 6, "limit": 96, "failures": 1, "version": "ir-v1"}
    assert by_source[("manuscript_ir.json", "writing_attempts")] == {
        "source": "manuscript_ir.json", "kind": "writing_attempts",
        "calls": 3, "limit": None, "failures": None, "version": "ir-v1"}
    assert by_source[("stage-18/manuscript_peer_review.json", "llm_review")] == {
        "source": "stage-18/manuscript_peer_review.json", "kind": "llm_review",
        "calls": 7, "limit": 40, "failures": None, "version": "ir-v1"}
    assert by_source[("search_meta.json", "retrieval")] == {
        "source": "search_meta.json", "kind": "retrieval",
        "calls": 3, "limit": None, "failures": 1, "cache_only": 1, "version": None}
    assert by_source[("method_validation.json", "method_probe")]["calls"] == 1


def test_protocol_budget_exposes_measured_validation_tuning_trials(tmp_path):
    (tmp_path / "protocol_budget.json").write_text(json.dumps(
        {"protocol_version": "p1", "tuning_trials": 7, "tuning_trial_limit": 12}))
    ledger = build_resource_ledger(tmp_path)
    row = ledger["rows"][0]
    assert row == {"source": "protocol_budget.json", "kind": "validation_tuning_trials",
                   "calls": 7, "limit": 12, "failures": None, "version": "p1"}
    validate_resource_ledger(tmp_path, ledger)
    validate_resource_ledger(tmp_path, ledger)


def test_unrecorded_fields_stay_null_not_inferred(tmp_path):
    write(tmp_path, "novelty_matrix.json", {"version": "nov-v1", "review_calls": 3})
    ledger = build_resource_ledger(tmp_path)
    row = ledger["rows"][0]
    assert row["limit"] is None and row["failures"] is None


def test_validate_rejects_tampered_or_stale_ledger(tmp_path):
    write(tmp_path, "literature_evidence.json",
          {"version": "lit-v1", "review_budget": {"calls": 5, "limit": 32, "failures": []}})
    ledger = build_resource_ledger(tmp_path)
    validate_resource_ledger(tmp_path, ledger)
    tampered = dict(ledger)
    tampered["rows"] = [dict(ledger["rows"][0], calls=1)]
    with pytest.raises(ResourceLedgerError):
        validate_resource_ledger(tmp_path, tampered)
    # Artifact changes behind a stored ledger are equally rejected.
    write(tmp_path, "literature_evidence.json",
          {"version": "lit-v2", "review_budget": {"calls": 6, "limit": 32, "failures": []}})
    with pytest.raises(ResourceLedgerError):
        validate_resource_ledger(tmp_path, ledger)
    with pytest.raises(ResourceLedgerError):
        validate_resource_ledger(tmp_path, {"schema_version": 1, "rows": [], "errors": []})


def test_unreadable_sources_fail_closed_into_errors(tmp_path):
    (tmp_path / "literature_evidence.json").write_text("{broken", encoding="utf-8")
    write(tmp_path, "novelty_matrix.json", "not-a-mapping")
    ledger = build_resource_ledger(tmp_path)
    assert [error["source"] for error in ledger["errors"]] == [
        "literature_evidence.json", "novelty_matrix.json"]
    assert ledger["rows"] == []
    assert ledger_issues(ledger) == ["unreadable_resource_source:literature_evidence.json",
                                     "unreadable_resource_source:novelty_matrix.json"]
    validate_resource_ledger(tmp_path, ledger)


def test_budget_exceeding_rows_are_flagged(tmp_path):
    write(tmp_path, "literature_evidence.json",
          {"version": "lit-v1", "review_budget": {"calls": 40, "limit": 32, "failures": []}})
    ledger = build_resource_ledger(tmp_path)
    assert ledger_issues(ledger) == ["budget_exceeded:literature_evidence.json"]
    write(tmp_path, "literature_evidence.json",
          {"version": "lit-v2", "review_budget": {"calls": 40, "limit": None, "failures": []}})
    assert ledger_issues(build_resource_ledger(tmp_path)) == []


def test_boolean_counts_are_not_compared_as_integers(tmp_path):
    write(tmp_path, "literature_evidence.json",
          {"version": "lit-v1", "review_budget": {"calls": True, "limit": 0, "failures": []}})
    assert ledger_issues(build_resource_ledger(tmp_path)) == []
