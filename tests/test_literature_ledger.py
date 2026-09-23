"""Per-(query, provider) retrieval ledger: cache fallbacks and failures must
stay visible instead of hiding behind one provider's aggregate success."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from researchclaw.literature import search as search_module
from researchclaw.literature.evidence import (
    LiteratureEvidenceError, coverage_report, validate_evidence, validate_search_log,
)
from researchclaw.literature.models import Author, Paper
from researchclaw.literature.search import search_papers, search_papers_multi_query, summarize_query_ledger
from researchclaw.pipeline.evidence_store import content_hash


def paper(source):
    return Paper(paper_id=f"id-{source}", title=f"Paper from {source}", authors=(Author(name="A", affiliation=""),),
                 year=2024, abstract="abstract", venue="venue", citation_count=1, doi="", arxiv_id="",
                 url="https://example.test", source=source)


def cached_paper(source):
    return {"paper_id": f"id-{source}", "title": f"Cached {source}", "authors": [{"name": "A", "affiliation": ""}],
            "year": 2024, "abstract": "abstract", "venue": "venue", "citation_count": 1, "doi": "",
            "arxiv_id": "", "url": "https://example.test", "source": source}


@pytest.fixture
def quiet(monkeypatch):
    monkeypatch.setattr(search_module, "time", SimpleNamespace(sleep=lambda seconds: None))
    monkeypatch.setattr(search_module, "_cache_api", lambda: (lambda *a: None, lambda *a: None))


def test_ledger_records_live_success_empty_and_failed_providers(quiet, monkeypatch):
    monkeypatch.setattr(search_module, "search_openalex", lambda query, **kwargs: [paper("openalex")])
    monkeypatch.setattr(search_module, "search_semantic_scholar", lambda query, **kwargs: [])
    def broken(query, **kwargs):
        raise RuntimeError("arxiv down")
    monkeypatch.setattr(search_module, "search_arxiv", broken)
    ledger = []
    papers = search_papers("gradient descent", sources=("openalex", "semantic_scholar", "arxiv"), ledger=ledger)
    assert [p.source for p in papers] == ["openalex"]
    assert ledger == [
        {"query": "gradient descent", "provider": "openalex", "status": "results_found",
         "candidate_count": 1, "cache_only": False, "error": ""},
        {"query": "gradient descent", "provider": "semantic_scholar", "status": "no_results",
         "candidate_count": 0, "cache_only": False, "error": ""},
        {"query": "gradient descent", "provider": "arxiv", "status": "failed",
         "candidate_count": 0, "cache_only": False, "error": "arxiv down"},
    ]
    summary = summarize_query_ledger(ledger)
    assert summary["status"] == "results_found" and summary["live_query_count"] == 1
    assert summary["failed_query_count"] == 2


def test_cache_fallback_is_recorded_and_never_counts_as_live(quiet, monkeypatch):
    monkeypatch.setattr(search_module, "_cache_api",
                        lambda: (lambda query, source, limit: [cached_paper(source)], lambda *a: None))
    def down(query, **kwargs):
        raise RuntimeError("rate limited")
    monkeypatch.setattr(search_module, "search_openalex", down)
    ledger = []
    papers = search_papers("q", sources=("openalex",), ledger=ledger)
    assert [p.source for p in papers] == ["openalex"]
    assert ledger[0]["status"] == "cache_only" and ledger[0]["cache_only"] is True
    assert ledger[0]["candidate_count"] == 1 and ledger[0]["error"] == "rate limited"
    assert summarize_query_ledger(ledger)["status"] == "cache_only"


def test_unknown_provider_is_recorded(quiet):
    ledger = []
    search_papers("q", sources=("web_of_science",), ledger=ledger)
    assert ledger == [{"query": "q", "provider": "web_of_science", "status": "unknown_provider",
                       "candidate_count": 0, "cache_only": False, "error": ""}]


def test_multi_query_ledger_covers_every_query(quiet, monkeypatch):
    monkeypatch.setattr(search_module, "search_arxiv", lambda query, **kwargs: [paper("arxiv")])
    ledger = []
    search_papers_multi_query(["a", "b"], sources=("arxiv",), inter_query_delay=0, ledger=ledger)
    assert [entry["query"] for entry in ledger] == ["a", "b"]
    assert all(entry["status"] == "results_found" for entry in ledger)


def test_validate_search_log_accepts_legacy_and_consistent_ledgers():
    assert validate_search_log({}) == "aggregate"
    assert validate_search_log({"status": "results_found"}) == "aggregate_backend_results"
    consistent = {"status": "results_found", "query_ledger": [
        {"query": "q", "provider": "openalex", "status": "results_found", "candidate_count": 3,
         "cache_only": False, "error": ""}]}
    assert validate_search_log(consistent) == "per_query_per_provider"


@pytest.mark.parametrize("entry", [
    {"query": "q", "provider": "p", "status": "failed", "candidate_count": 0, "cache_only": False, "error": "x"},
    {"query": "q", "provider": "p", "status": "cache_only", "candidate_count": 2, "cache_only": True, "error": "x"},
])
def test_aggregate_success_cannot_rest_on_failures_or_cache(entry):
    search_log = {"status": "results_found", "query_ledger": [entry]}
    with pytest.raises(LiteratureEvidenceError, match="contradicts the query ledger"):
        validate_search_log(search_log)


@pytest.mark.parametrize("entry,match", [
    ({"query": "", "provider": "p", "status": "results_found", "candidate_count": 1, "cache_only": False, "error": ""},
     "nonempty query"),
    ({"query": "q", "provider": "", "status": "results_found", "candidate_count": 1, "cache_only": False, "error": ""},
     "nonempty provider"),
    ({"query": "q", "provider": "p", "status": "worked", "candidate_count": 1, "cache_only": False, "error": ""},
     "Invalid query ledger status"),
    ({"query": "q", "provider": "p", "status": "results_found", "candidate_count": 0, "cache_only": False, "error": ""},
     "claims results without candidates"),
    ({"query": "q", "provider": "p", "status": "cache_only", "candidate_count": 0, "cache_only": True, "error": ""},
     "claims results without candidates"),
    ({"query": "q", "provider": "p", "status": "no_results", "candidate_count": 2, "cache_only": False, "error": ""},
     "carries candidates"),
    ({"query": "q", "provider": "p", "status": "failed", "candidate_count": 1, "cache_only": False, "error": ""},
     "carries candidates"),
    ({"query": "q", "provider": "p", "status": "results_found", "candidate_count": -1, "cache_only": False, "error": ""},
     "nonnegative integer"),
    ({"query": "q", "provider": "p", "status": "results_found", "candidate_count": "2", "cache_only": False, "error": ""},
     "nonnegative integer"),
])
def test_malformed_ledger_entries_fail_closed(entry, match):
    search_log = {"status": "empty_or_unavailable", "query_ledger": [entry]}
    with pytest.raises(LiteratureEvidenceError, match=match):
        validate_search_log(search_log)
    stored = {"schema_version": 1, "sources": [], "cards": [], "errors": [], "unprocessed_papers": [],
              "search_log": search_log}
    stored["version"] = content_hash(stored)
    with pytest.raises(LiteratureEvidenceError, match=match):
        validate_evidence(Path("."), stored)


def test_empty_ledger_reads_as_aggregate_scope():
    # Stage 4 writes query_ledger: [] when the search aborts before any
    # provider attempt; the evidence path must read that document instead of
    # crashing on it.
    assert validate_search_log({"status": "empty_or_unavailable", "query_ledger": []}) == "aggregate_backend_results"
    search_log = {"status": "empty_or_unavailable", "query_ledger": []}
    bundle = {"schema_version": 1, "sources": [], "cards": [], "errors": [], "unprocessed_papers": [],
              "version": "x", "search_log": search_log}
    report = coverage_report(bundle)
    assert report["retrieval_scope"] == "aggregate_backend_results"
    stored = {"schema_version": 1, "sources": [], "cards": [], "errors": [], "unprocessed_papers": [],
              "search_log": search_log}
    stored["version"] = content_hash(stored)
    validate_evidence(Path("."), stored)


def test_malformed_ledger_containers_still_fail_closed():
    for ledger in ("nope", [None]):
        with pytest.raises(LiteratureEvidenceError):
            validate_search_log({"status": "results_found", "query_ledger": ledger})


def test_coverage_report_marks_retrieval_scope_and_rejects_contradictions():
    bundle = {"schema_version": 1, "sources": [], "cards": [], "errors": [], "unprocessed_papers": [],
              "version": "x", "search_log": {"status": "results_found"}}
    report = coverage_report(bundle)
    assert report["retrieval_scope"] == "aggregate_backend_results"
    bundle["search_log"] = {"status": "results_found", "query_ledger": [
        {"query": "q", "provider": "openalex", "status": "failed", "candidate_count": 0,
         "cache_only": False, "error": "down"}]}
    with pytest.raises(LiteratureEvidenceError, match="contradicts the query ledger"):
        coverage_report(bundle)


class FakeResult:
    """Minimal search hit: WebSearchAgentResult.to_dict() calls item.to_dict()."""

    def to_dict(self):
        return {"title": "fixture"}


class FakeWebClient:
    def __init__(self, *, fail=False):
        self.fail = fail

    def search_multi(self, queries, *, max_results=10):
        if self.fail:
            raise RuntimeError("tavily unavailable")
        return [SimpleNamespace(results=[FakeResult()], answer="answer text") for _ in queries]


class FakeScholar:
    def __init__(self, *, fail=False):
        self.fail = fail

    def search(self, topic, *, limit=10):
        if self.fail:
            raise RuntimeError("scholar blocked")
        return [FakeResult()]


def web_agent(monkeypatch, *, scholar=True, web_fail=False, scholar_fail=False):
    from researchclaw.web.agent import WebSearchAgent
    agent = WebSearchAgent(enable_scholar=scholar, enable_crawling=False, enable_pdf=False)
    agent.web_client = FakeWebClient(fail=web_fail)
    if scholar:
        agent.scholar_client = FakeScholar(fail=scholar_fail)
        monkeypatch.setattr(type(agent.scholar_client), "available", property(lambda self: True), raising=False)
    return agent


def test_web_provider_log_records_successes(monkeypatch):
    result = web_agent(monkeypatch).search_and_extract("topic", search_queries=["q1", "q2"])
    statuses = {entry["provider"]: entry["status"] for entry in result.provider_log}
    assert statuses == {"web_search": "results_found", "google_scholar": "results_found"}
    web_entry = next(e for e in result.provider_log if e["provider"] == "web_search")
    assert web_entry["candidate_count"] == 2 and web_entry["queries"] == 2
    assert result.to_dict()["provider_log"] == result.provider_log


def test_web_provider_log_records_failures_and_unavailable(monkeypatch):
    result = web_agent(monkeypatch, scholar=False, web_fail=True).search_and_extract("topic", search_queries=["q"])
    statuses = {entry["provider"]: entry["status"] for entry in result.provider_log}
    assert statuses == {"web_search": "failed", "google_scholar": "unavailable"}
    web_entry = next(e for e in result.provider_log if e["provider"] == "web_search")
    assert web_entry["candidate_count"] == 0 and "tavily unavailable" in web_entry["error"]


def test_web_provider_log_records_scholar_failure(monkeypatch):
    result = web_agent(monkeypatch, scholar_fail=True).search_and_extract("topic", search_queries=["q"])
    scholar_entry = next(e for e in result.provider_log if e["provider"] == "google_scholar")
    assert scholar_entry["status"] == "failed" and "scholar blocked" in scholar_entry["error"]
