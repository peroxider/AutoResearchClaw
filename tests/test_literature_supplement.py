"""Incremental evidence supplement: gap-filling search must extend a validated
bundle without invalidating the whole evidence chain, and every retrieval
attempt — including failures — must stay visible in the per-query ledger."""
import json
from types import SimpleNamespace

import pytest

from researchclaw.adapters import AdapterBundle
from researchclaw.literature.evidence import (
    LiteratureEvidenceError, _merge_search_log, build_evidence, coverage_report,
    supplement_evidence, validate_evidence, write_json,
)
from researchclaw.pipeline.evidence_store import content_hash
from researchclaw.literature.models import Author, Paper
from researchclaw.pipeline.stages import StageStatus
from tests.test_literature_evidence import QUOTE, Reviewer, sources
from tests.test_submission_gates import config


def _entry(query, status, count=1):
    return {"query": query, "provider": "openalex", "status": status,
            "candidate_count": count, "cache_only": False, "error": ""}


def ready_bundle(sources):
    root, paper = sources
    return build_evidence(root, [paper], llm=Reviewer(), search_log={"status": "results_found"})


def add_second_local_source(root):
    directory = root / "literature_input"
    (directory / "second.txt").write_text(QUOTE + "\n" + "Follow-up study context. " * 12, encoding="utf-8")
    second = {"cite_key": "jones2023", "title": "Second fixture controlled intervention study",
              "path": "second.txt", "year": 2023, "url": "https://example.test/second", "abstract": QUOTE}
    papers = json.loads((directory / "sources.json").read_text(encoding="utf-8"))["papers"] + [second]
    write_json(directory / "sources.json", {"schema_version": 1, "papers": papers})
    return second


# ---------------------------------------------------------------------------
# supplement_evidence: incremental extension of a validated bundle
# ---------------------------------------------------------------------------


def test_supplement_extends_validated_bundle_incrementally(sources):
    root, _ = sources
    bundle = ready_bundle(sources)
    second = add_second_local_source(root)
    extended = supplement_evidence(root, [second], bundle=bundle, llm=Reviewer())
    validate_evidence(root, extended)
    assert [s["metadata"]["cite_key"] for s in extended["sources"]] == ["smith2024", "jones2023"]
    assert extended["version"] != bundle["version"]
    assert extended["review_budget"]["calls"] > bundle["review_budget"]["calls"]
    assert extended["review_budget"]["limit"] == bundle["review_budget"]["limit"] + 32
    stored = json.loads((root / "literature_evidence.json").read_text(encoding="utf-8"))
    assert stored["version"] == extended["version"]
    assert json.loads((root / "literature_coverage.json").read_text(encoding="utf-8"))["evidence_version"] == extended["version"]


def test_supplement_skips_already_evidenced_keys(sources):
    root, paper = sources
    bundle = ready_bundle(sources)
    once = supplement_evidence(root, [paper], bundle=bundle, llm=Reviewer())
    twice = supplement_evidence(root, [paper], bundle=once, llm=Reviewer())
    assert len(once["sources"]) == len(twice["sources"]) == 1
    assert once["cards"] == twice["cards"]
    validate_evidence(root, twice)


def test_supplement_rejects_tampered_bundle_before_archiving(sources):
    root, _ = sources
    ready_bundle(sources)
    stored = json.loads((root / "literature_evidence.json").read_text(encoding="utf-8"))
    stored["cards"][0]["claim"] = "Tampered claim"
    write_json(root / "literature_evidence.json", stored)
    with pytest.raises(LiteratureEvidenceError):
        supplement_evidence(root, [{"cite_key": "new2025", "title": "New"}])
    assert json.loads((root / "literature_evidence.json").read_text(encoding="utf-8"))["cards"][0]["claim"] == "Tampered claim"


def test_supplement_rejects_invalidated_placeholder(tmp_path):
    write_json(tmp_path / "literature_evidence.json", {"status": "invalidated_by_literature_refresh"})
    with pytest.raises(LiteratureEvidenceError):
        supplement_evidence(tmp_path, [])


def test_supplement_rejects_malformed_review_budget_before_archiving(sources):
    root, _ = sources
    ready_bundle(sources)
    stored = json.loads((root / "literature_evidence.json").read_text(encoding="utf-8"))
    stored["review_budget"]["calls"] = "corrupt"
    stored["version"] = content_hash({k: v for k, v in stored.items() if k != "version"})
    write_json(root / "literature_evidence.json", stored)
    with pytest.raises(LiteratureEvidenceError):
        supplement_evidence(root, [{"cite_key": "new2025", "title": "New"}])
    on_disk = json.loads((root / "literature_evidence.json").read_text(encoding="utf-8"))
    assert on_disk["review_budget"]["calls"] == "corrupt" and on_disk["version"] == stored["version"]


def test_supplement_rejects_non_mapping_papers_before_archiving(sources):
    root, _ = sources
    bundle = ready_bundle(sources)
    with pytest.raises(LiteratureEvidenceError):
        supplement_evidence(root, ["not-a-mapping"], bundle=bundle)
    on_disk = json.loads((root / "literature_evidence.json").read_text(encoding="utf-8"))
    assert on_disk["version"] == bundle["version"] and on_disk["sources"] == bundle["sources"]


def test_supplement_rejects_malformed_manifest_before_archiving(sources):
    root, _ = sources
    bundle = ready_bundle(sources)
    (root / "literature_input" / "sources.json").write_text("{broken", encoding="utf-8")
    with pytest.raises((LiteratureEvidenceError, ValueError)):
        supplement_evidence(root, [], bundle=bundle)
    on_disk = json.loads((root / "literature_evidence.json").read_text(encoding="utf-8"))
    assert on_disk["version"] == bundle["version"] and on_disk["sources"] == bundle["sources"]


def test_supplement_invalidates_downstream_and_archives_history(sources):
    root, paper = sources
    bundle = ready_bundle(sources)
    for name in ("novelty_matrix.json", "contribution_ledger.json", "citation_support.json"):
        write_json(root / name, {"stale": name})
    supplement_evidence(root, [paper], bundle=bundle, llm=Reviewer())
    for name in ("novelty_matrix.json", "contribution_ledger.json", "citation_support.json"):
        assert json.loads((root / name).read_text(encoding="utf-8")) == {"status": "invalidated_by_literature_refresh"}
        history = root / "evidence_artifacts" / "literature_history" / name
        assert any(archive.suffix == ".json" for archive in history.iterdir())
    evidence_history = root / "evidence_artifacts" / "literature_history" / "literature_evidence.json"
    archived = [json.loads(p.read_text(encoding="utf-8")) for p in evidence_history.iterdir()]
    assert any(old.get("version") == bundle["version"] for old in archived)


def test_supplement_records_missing_identity_and_caps_new_papers(tmp_path):
    paper = {"cite_key": "smith2024", "title": "Fixture", "abstract": QUOTE}
    bundle = build_evidence(tmp_path, [paper], llm=Reviewer(), search_log={"status": "results_found"})
    fresh = [{"title": "No identity"},
             {"cite_key": "a2024", "title": "First new", "abstract": QUOTE},
             {"cite_key": "b2024", "title": "Second new", "abstract": QUOTE}]
    extended = supplement_evidence(tmp_path, fresh, bundle=bundle, llm=Reviewer(), max_papers=1)
    assert {"cite_key": "", "reason": "missing_citation_identity"} in extended["errors"]
    assert len(extended["sources"]) == 2
    assert extended["unprocessed_papers"] == ["b2024"]


def test_supplement_clears_processed_pending_entries(tmp_path):
    first = {"cite_key": "smith2024", "title": "First", "abstract": QUOTE}
    second = {"cite_key": "jones2023", "title": "Second", "abstract": QUOTE}
    bundle = build_evidence(tmp_path, [first, second], llm=Reviewer(),
                            search_log={"status": "results_found"}, max_papers=1)
    assert bundle["unprocessed_papers"] == ["jones2023"]
    extended = supplement_evidence(tmp_path, [second], bundle=bundle, llm=Reviewer())
    assert extended["unprocessed_papers"] == []


def test_supplement_abstract_only_addition_cannot_flip_coverage(tmp_path):
    paper = {"cite_key": "smith2024", "title": "Fixture", "abstract": QUOTE}
    bundle = build_evidence(tmp_path, [paper], llm=Reviewer(), search_log={"status": "results_found"})
    extended = supplement_evidence(tmp_path, [{"cite_key": "a2024", "title": "More of the same", "abstract": QUOTE}],
                                   bundle=bundle, llm=Reviewer())
    assert extended["sources"][1]["scope"] == "abstract_only"
    assert coverage_report(extended)["status"] == "needs_more_evidence"


# ---------------------------------------------------------------------------
# retrieval ledger merging
# ---------------------------------------------------------------------------


def test_merge_search_log_unions_ledgers_and_rederives_status():
    old = {"status": "results_found", "query_ledger": [_entry("q1", "results_found")]}
    new = {"status": "results_found", "query_ledger": [_entry("q2", "results_found")]}
    merged = _merge_search_log(old, new)
    assert [e["query"] for e in merged["query_ledger"]] == ["q1", "q2"]
    assert merged["status"] == "results_found"
    demoted = _merge_search_log({"status": "results_found"},
                                {"status": "empty_or_unavailable", "query_ledger": [_entry("q3", "failed", count=0)]})
    assert demoted["status"] == "empty_or_unavailable"
    assert _merge_search_log(old, {"status": "failed"}) == old
    assert _merge_search_log(old, {}) == old
    assert _merge_search_log({}, {}) == {"status": "unknown"}


def test_supplement_merged_ledger_passes_validation(sources):
    root, paper = sources
    bundle = build_evidence(root, [paper], llm=Reviewer(),
                            search_log={"status": "results_found", "query_ledger": [_entry("q1", "results_found")]})
    extended = supplement_evidence(root, [paper], bundle=bundle, llm=Reviewer(),
                                   search_log={"status": "empty_or_unavailable",
                                               "query_ledger": [_entry("q2", "failed", count=0)]})
    validate_evidence(root, extended)
    assert extended["search_log"]["status"] == "results_found"
    assert coverage_report(extended)["retrieval_scope"] == "per_query_per_provider"
    fresh = build_evidence(root, [paper], llm=Reviewer(), search_log={"status": "results_found"})
    demoted = supplement_evidence(root, [paper], bundle=fresh, llm=Reviewer(),
                                  search_log={"status": "empty_or_unavailable",
                                              "query_ledger": [_entry("q3", "failed", count=0)]})
    validate_evidence(root, demoted)
    assert demoted["search_log"]["status"] == "empty_or_unavailable"


# ---------------------------------------------------------------------------
# Stage 6 bounded supplement loop
# ---------------------------------------------------------------------------

PAGE = QUOTE + "\n" + "Methods and study context. " * 12
SUPP_PAPER = Paper(paper_id="supp1", title="Supplemental full-text study",
                   authors=(Author(name="B", affiliation=""),), year=2023, abstract=QUOTE,
                   venue="venue", citation_count=1, doi="", arxiv_id="2401.00001",
                   url="https://arxiv.org/abs/2401.00001", source="openalex")


def fake_search(monkeypatch, papers):
    seen = []

    def search(queries, **kwargs):
        seen.extend(queries)
        ledger = kwargs["ledger"]
        status = "results_found" if papers else "no_results"
        for query in queries:
            ledger.append({"query": query, "provider": "openalex", "status": status,
                           "candidate_count": len(papers), "cache_only": False, "error": ""})
        return list(papers)

    monkeypatch.setattr("researchclaw.literature.search.search_papers_multi_query", search)
    return seen


def fake_fetch(monkeypatch, *, fail=False):
    from researchclaw.web.pdf_extractor import PDFContent
    calls = []

    def extract(self, url):
        calls.append(url)
        if fail:
            raise RuntimeError("network down")
        return PDFContent(path=url, success=True, page_texts=[PAGE], page_count=1,
                          document_sha256="fixture-sha")

    monkeypatch.setattr("researchclaw.web.pdf_extractor.PDFExtractor.extract_from_url", extract)
    return calls


def stage6(root, shortlist, *, status="submission_candidate", llm=None):
    from researchclaw.pipeline.stage_impls._literature import _execute_knowledge_extract
    for number in (4, 5, 6):
        (root / f"stage-{number:02d}").mkdir(exist_ok=True)
    (root / "stage-05/shortlist.jsonl").write_text(
        "\n".join(json.dumps(row) for row in shortlist), encoding="utf-8")
    write_json(root / "stage-04/search_meta.json",
               {"status": "results_found", "queries_used": ["fixture query"]})
    return _execute_knowledge_extract(root / "stage-06", root, config(root, status=status),
                                      AdapterBundle(), llm=llm if llm is not None else Reviewer())


ABSTRACT_ONLY_SHORTLIST = [{"cite_key": "smith2024", "title": "Abstract-only prior work", "abstract": QUOTE}]


def test_stage6_supplement_completes_coverage_and_records_round(tmp_path, monkeypatch):
    root = tmp_path
    fake_search(monkeypatch, [SUPP_PAPER])
    fetches = fake_fetch(monkeypatch)
    result = stage6(root, ABSTRACT_ONLY_SHORTLIST)
    assert result.status == StageStatus.DONE
    coverage = json.loads((root / "literature_coverage.json").read_text(encoding="utf-8"))
    assert coverage["status"] == "review_ready" and coverage["exhaustive"] is False
    supplement = json.loads((root / "literature_supplement.json").read_text(encoding="utf-8"))
    assert supplement["stopped_at"] == "review_ready" and supplement["exhaustive"] is False
    assert len(supplement["rounds"]) == 1
    round_record = supplement["rounds"][0]
    assert round_record["new_papers"] == 1
    assert round_record["coverage_status"] == "review_ready"
    assert round_record["retrieval"]["status"] == "results_found"
    assert round_record["pdf_fetches"] == [{"url": "https://arxiv.org/pdf/2401.00001", "status": "success"}]
    assert round_record["queries"] == ["fixture query survey", "fixture query review", "fixture query seminal"]
    assert fetches == ["https://arxiv.org/pdf/2401.00001"]
    bundle = json.loads((root / "literature_evidence.json").read_text(encoding="utf-8"))
    assert len(bundle["sources"]) == 2
    assert bundle["search_log"]["status"] == "results_found"
    assert {entry["query"] for entry in bundle["search_log"]["query_ledger"]} == set(round_record["queries"])
    validate_evidence(root, bundle)


def test_stage6_supplement_stops_after_two_rounds(tmp_path, monkeypatch):
    root = tmp_path
    fake_search(monkeypatch, [SUPP_PAPER])
    fake_fetch(monkeypatch, fail=True)
    result = stage6(root, ABSTRACT_ONLY_SHORTLIST)
    assert result.status == StageStatus.PAUSED
    assert result.decision == "literature_evidence_incomplete"
    supplement = json.loads((root / "literature_supplement.json").read_text(encoding="utf-8"))
    assert [r["new_papers"] for r in supplement["rounds"]] == [1, 0]
    assert supplement["stopped_at"] == "needs_more_evidence"
    assert supplement["rounds"][0]["queries"] == ["fixture query survey", "fixture query review", "fixture query seminal"]
    assert supplement["rounds"][1]["queries"] == ["fixture query foundational", "fixture query comparison", "fixture query benchmark"]


def test_stage6_supplement_stops_when_nothing_new_is_found(tmp_path, monkeypatch):
    root = tmp_path
    stub = SimpleNamespace(to_dict=lambda: {"cite_key": "smith2024", "title": "Already evidenced",
                                            "abstract": QUOTE, "arxiv_id": "", "url": ""})
    fake_search(monkeypatch, [stub])
    fake_fetch(monkeypatch)
    result = stage6(root, ABSTRACT_ONLY_SHORTLIST)
    assert result.status == StageStatus.PAUSED
    supplement = json.loads((root / "literature_supplement.json").read_text(encoding="utf-8"))
    assert [r["new_papers"] for r in supplement["rounds"]] == [0, 0]
    assert supplement["rounds"][1]["queries"] != supplement["rounds"][0]["queries"]


def test_stage6_supplement_records_failed_pdf_fetch(tmp_path, monkeypatch):
    root = tmp_path
    fake_search(monkeypatch, [SUPP_PAPER])
    calls = fake_fetch(monkeypatch, fail=True)
    result = stage6(root, ABSTRACT_ONLY_SHORTLIST)
    assert result.status == StageStatus.PAUSED
    supplement = json.loads((root / "literature_supplement.json").read_text(encoding="utf-8"))
    record = supplement["rounds"][0]["pdf_fetches"][0]
    assert record["status"] == "failed" and "network down" in record["error"]
    assert calls == ["https://arxiv.org/pdf/2401.00001"]


def local_fulltext_root(root):
    directory = root / "literature_input"
    directory.mkdir(exist_ok=True)
    (directory / "paper.txt").write_text(QUOTE + "\n" + "Methods and study context. " * 12, encoding="utf-8")
    paper = {"cite_key": "smith2024", "title": "Fixture controlled intervention study", "path": "paper.txt",
             "year": 2024, "url": "https://example.test/paper", "abstract": QUOTE}
    write_json(directory / "sources.json", {"schema_version": 1, "papers": [paper]})
    return paper


def test_stage6_skips_supplement_when_coverage_ready(tmp_path, monkeypatch):
    root = tmp_path
    paper = local_fulltext_root(root)

    def boom(*args, **kwargs):
        raise AssertionError("supplement search must not run when coverage is ready")

    monkeypatch.setattr("researchclaw.literature.search.search_papers_multi_query", boom)
    result = stage6(root, [paper])
    assert result.status == StageStatus.DONE
    assert not (root / "literature_supplement.json").exists()


def test_stage6_skips_supplement_for_exploratory_target(tmp_path, monkeypatch):
    root = tmp_path
    paper = local_fulltext_root(root)

    def boom(*args, **kwargs):
        raise AssertionError("supplement search must not run for exploratory targets")

    monkeypatch.setattr("researchclaw.literature.search.search_papers_multi_query", boom)
    result = stage6(root, [paper], status="exploratory", llm=Reviewer(status="not_supported"))
    assert result.status == StageStatus.DONE
    assert not (root / "literature_supplement.json").exists()


def test_supplement_tolerates_missing_or_non_dict_review_budget(sources):
    root, _ = sources
    for replacement in (None, "corrupt-string"):
        ready_bundle(sources)
        stored = json.loads((root / "literature_evidence.json").read_text(encoding="utf-8"))
        if replacement is None:
            stored.pop("review_budget")
        else:
            stored["review_budget"] = replacement
        stored["version"] = content_hash({k: v for k, v in stored.items() if k != "version"})
        write_json(root / "literature_evidence.json", stored)
        extended = supplement_evidence(root, [])
        validate_evidence(root, extended)
        assert extended["review_budget"] == {"calls": 0, "limit": 32, "failures": []}


# ---------------------------------------------------------------------------
# justified not-applicable coverage declarations
# ---------------------------------------------------------------------------

NA_REASON = "Fixture declaration: no external dataset protocol applies to synthetic study data."


def test_not_applicable_declarations_recorded_and_respected(sources):
    root, paper = sources
    bundle = build_evidence(root, [paper], llm=Reviewer(), search_log={"status": "results_found"},
                            not_applicable=[{"category": "dataset_protocol", "reason": NA_REASON}])
    validate_evidence(root, bundle)
    assert bundle["not_applicable"] == [{"category": "dataset_protocol", "reason": NA_REASON}]
    report = coverage_report(bundle)
    assert report["categories"]["dataset_protocol"]["status"] == "not_applicable"
    assert report["categories"]["dataset_protocol"]["reason"] == NA_REASON
    assert {"category": "dataset_protocol", "reason": NA_REASON} in report["not_applicable"]
    assert report["status"] == "review_ready" and report["exhaustive"] is False


def test_not_applicable_declarations_enable_review_ready(tmp_path):
    paper = {"cite_key": "smith2024", "title": "Abstract-only fixture", "abstract": QUOTE}
    bundle = build_evidence(tmp_path, [paper], llm=Reviewer(), search_log={"status": "results_found"})
    assert coverage_report(bundle)["status"] == "needs_more_evidence"
    declared = [{"category": category, "reason": NA_REASON} for category in
                ("topic", "foundational", "direct_competitors", "recent", "dataset_protocol")]
    declared_bundle = build_evidence(tmp_path, [paper], llm=Reviewer(),
                                     search_log={"status": "results_found"}, not_applicable=declared)
    report = coverage_report(declared_bundle)
    assert report["status"] == "review_ready"
    assert len(report["not_applicable"]) == 5 and report["exhaustive"] is False
    assert all(info["status"] == "not_applicable" for info in report["categories"].values())


def test_not_applicable_declarations_fail_closed(tmp_path):
    paper = {"cite_key": "smith2024", "title": "Fixture", "abstract": QUOTE}
    for bad in ("not-a-list", [{"category": "unknown_category", "reason": "x"}],
                [{"category": "topic", "reason": "   "}], [{"category": "topic"}],
                [{"category": "topic", "reason": 7}],
                [{"category": "topic", "reason": "x"}, {"category": "topic", "reason": "y"}]):
        with pytest.raises(LiteratureEvidenceError):
            build_evidence(tmp_path, [paper], llm=Reviewer(), search_log={"status": "results_found"},
                           not_applicable=bad)
    assert not (tmp_path / "literature_evidence.json").exists()


def test_validate_evidence_rejects_malformed_stored_declarations(sources):
    root, _ = sources
    ready_bundle(sources)
    stored = json.loads((root / "literature_evidence.json").read_text(encoding="utf-8"))
    stored["not_applicable"] = [{"category": "topic", "reason": ""}]
    stored["version"] = content_hash({k: v for k, v in stored.items() if k != "version"})
    write_json(root / "literature_evidence.json", stored)
    with pytest.raises(LiteratureEvidenceError):
        validate_evidence(root, stored)
    with pytest.raises(LiteratureEvidenceError):
        supplement_evidence(root, [{"cite_key": "new2025", "title": "New"}])
    on_disk = json.loads((root / "literature_evidence.json").read_text(encoding="utf-8"))
    assert on_disk["not_applicable"] == [{"category": "topic", "reason": ""}]


def test_supplement_carries_declarations_and_honors_override(sources):
    root, paper = sources
    declared = [{"category": "dataset_protocol", "reason": NA_REASON}]
    bundle = build_evidence(root, [paper], llm=Reviewer(), search_log={"status": "results_found"},
                            not_applicable=declared)
    carried = supplement_evidence(root, [paper], bundle=bundle, llm=Reviewer())
    assert carried["not_applicable"] == declared
    validate_evidence(root, carried)
    overridden = supplement_evidence(root, [paper], bundle=bundle, llm=Reviewer(), not_applicable=[])
    assert overridden["not_applicable"] == []
    validate_evidence(root, overridden)


def test_stage6_supplement_skips_not_applicable_categories(tmp_path, monkeypatch):
    from dataclasses import replace

    root = tmp_path
    fake_search(monkeypatch, [SUPP_PAPER])
    fake_fetch(monkeypatch, fail=True)
    real_config = config

    def patched(root_path, status="submission_candidate"):
        cfg = real_config(root_path, status=status)
        return replace(cfg, literature_search=replace(
            cfg.literature_search,
            not_applicable=({"category": "direct_competitors", "reason": NA_REASON},
                            {"category": "recent", "reason": NA_REASON})))

    monkeypatch.setattr("tests.test_literature_supplement.config", patched)
    result = stage6(root, ABSTRACT_ONLY_SHORTLIST)
    assert result.status == StageStatus.PAUSED
    supplement = json.loads((root / "literature_supplement.json").read_text(encoding="utf-8"))
    queries = [query for round_ in supplement["rounds"] for query in round_["queries"]]
    assert all("recent advances" not in query for query in queries)
    assert all("comparison" not in query and "benchmark" not in query for query in queries)
    assert "fixture query dataset" in queries and "fixture query protocol" in queries
    assert supplement["rounds"][1]["queries"] == ["fixture query foundational",
                                                  "fixture query dataset", "fixture query protocol"]
    coverage = json.loads((root / "literature_coverage.json").read_text(encoding="utf-8"))
    assert coverage["categories"]["recent"]["status"] == "not_applicable"
    assert {"category": "recent", "reason": NA_REASON} in coverage["not_applicable"]
