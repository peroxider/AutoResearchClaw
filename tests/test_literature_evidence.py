import json
from types import SimpleNamespace

import pytest

from researchclaw.adapters import AdapterBundle
from researchclaw.literature.evidence import (
    CATEGORIES, LiteratureEvidenceError, build_citation_support, build_evidence,
    citation_occurrences, citation_support_issues, coverage_report, read_local_sources,
    snapshot_source, validate_evidence, write_json,
)
from researchclaw.pipeline.evidence_store import content_hash, file_hash
from researchclaw.pipeline.stages import StageStatus
from tests.test_final_acceptance import delivery, review
from tests.test_submission_gates import config

QUOTE = "The intervention reduces loss in setting A only; no evidence is available for setting B."


class Reviewer:
    def __init__(self, *, status="supported", invented=False, unavailable=False):
        self.status, self.invented, self.unavailable = status, invented, unavailable
        self.calls = []

    def chat(self, messages, *, system="", **kwargs):
        payload = json.loads(messages[0]["content"])
        self.calls.append((system, payload))
        if self.unavailable:
            raise RuntimeError("fixture model unavailable")
        if system.startswith("Extract"):
            result = {"cards": [{"claim": "The intervention reduces loss in setting A only.",
                "excerpt": "Invented source text" if self.invented else QUOTE,
                "part": 1, "conditions": "Setting A only; B is untested.", "categories": list(CATEGORIES)}]}
        else:
            status = "contradicted" if "cures all diseases" in payload["claim"] else self.status
            result = {"status": status, "rationale": "Fixture semantic review, not a real expert judgment."}
        return SimpleNamespace(content=json.dumps(result), model="fixture-reviewer")


@pytest.fixture
def sources(tmp_path):
    root = tmp_path
    directory = root / "literature_input"
    directory.mkdir()
    (directory / "paper.txt").write_text(QUOTE + "\n" + "Methods and study context. " * 12, encoding="utf-8")
    paper = {"cite_key": "smith2024", "title": "Fixture controlled intervention study", "path": "paper.txt",
             "year": 2024, "url": "https://example.test/paper", "abstract": QUOTE}
    write_json(directory / "sources.json", {"schema_version": 1, "papers": [paper]})
    return root, paper


def build(sources, llm=None, **kwargs):
    root, paper = sources
    return build_evidence(root, [paper], llm=llm or Reviewer(), search_log={"status": "results_found"}, **kwargs)


def manuscripts(root, claim="The intervention reduces loss in setting A only."):
    (root / "paper_final.md").write_text(claim + " [smith2024]", encoding="utf-8")
    (root / "paper.tex").write_text(claim + r" \cite{smith2024}", encoding="utf-8")
    return {name: (root / name).read_text(encoding="utf-8") for name in ("paper_final.md", "paper.tex")}


def test_cards_bind_source_identity_spans_scope_and_review(sources):
    root, _ = sources
    bundle = build(sources)
    validate_evidence(root, bundle)
    assert bundle["sources"][0]["scope"] == "full_text"
    card = bundle["cards"][0]
    assert card["excerpt"] == QUOTE and card["review"]["status"] == "supported"
    assert card["locator"].startswith("text_part:1")
    assert coverage_report(bundle)["status"] == "review_ready"
    assert coverage_report(bundle)["exhaustive"] is False


def test_exact_quote_does_not_automatically_support_claim(sources):
    root, _ = sources
    bundle = build(sources, Reviewer(status="not_supported"))
    assert bundle["cards"][0]["excerpt"] == QUOTE
    assert coverage_report(bundle)["status"] == "needs_more_evidence"


@pytest.mark.parametrize("reviewer", [Reviewer(invented=True), Reviewer(unavailable=True)])
def test_invented_quote_or_failed_service_does_not_create_evidence(sources, reviewer):
    bundle = build(sources, reviewer)
    assert bundle["cards"] == []
    assert coverage_report(bundle)["status"] == "needs_more_evidence"


def test_abstract_only_cannot_pass_fulltext_coverage(tmp_path):
    paper = {"cite_key": "smith2024", "title": "Fixture", "abstract": QUOTE}
    bundle = build_evidence(tmp_path, [paper], llm=Reviewer(), search_log={"status": "results_found"})
    assert bundle["cards"][0]["review"]["status"] == "supported"
    assert bundle["sources"][0]["scope"] == "abstract_only"
    assert coverage_report(bundle)["status"] == "needs_more_evidence"
    text = manuscripts(tmp_path)
    support = build_citation_support(tmp_path, reviewer=Reviewer())
    assert support["status"] != "verified"
    assert citation_support_issues(tmp_path, support, text)


def test_search_failure_and_budget_exhaustion_remain_unknown(sources):
    root, paper = sources
    bundle = build_evidence(root, [paper], llm=Reviewer(), search_log={"status": "empty_or_unavailable"})
    assert coverage_report(bundle)["status"] == "needs_more_evidence"
    limited = build(sources, max_calls=1)
    assert limited["review_budget"]["calls"] == 1
    assert limited["cards"][0]["review"]["status"] == "unavailable"


@pytest.mark.parametrize("change", ["source", "claim", "offset", "paper", "review", "locator", "verdict"])
def test_changed_evidence_is_rejected_even_when_outer_version_rehashed(sources, change):
    root, _ = sources
    bundle = build(sources)
    card = bundle["cards"][0]
    if change == "source":
        (root / card["source"]).write_text("changed")
    elif change == "claim":
        card["claim"] = "A different claim"
    elif change == "offset":
        card["start"] += 1
    elif change == "paper":
        card["cite_key"] = "other2025"
    elif change == "locator":
        card["locator"] = "pdf_page:999"
    elif change == "verdict":
        card["review"]["status"] = "contradicted"
    else:
        card["review"]["input_hash"] = "wrong"
    if change != "source":
        card.pop("card_id")
        card["card_id"] = content_hash(card)
    bundle.pop("version")
    bundle["version"] = content_hash(bundle)
    with pytest.raises(LiteratureEvidenceError):
        validate_evidence(root, bundle)


def test_local_sources_cannot_escape_declared_directory(sources):
    root, paper = sources
    (root / "private.txt").write_text("not a literature source")
    write_json(root / "literature_input/sources.json", {"schema_version": 1, "papers": [{**paper, "path": "../private.txt"}]})
    with pytest.raises(LiteratureEvidenceError, match="within literature_input"):
        read_local_sources(root)


def test_web_fulltext_matches_by_identity_not_title(tmp_path):
    paper = {"cite_key": "smith2024", "title": "Same title", "url": "https://example.test/one", "abstract": QUOTE}
    wrong = {"path": "https://example.test/two", "title": "Same title", "success": True, "page_texts": ["Wrong paper"]}
    assert snapshot_source(tmp_path, paper, [wrong])["scope"] == "abstract_only"


def test_page_boundaries_are_preserved_in_fulltext(tmp_path):
    paper = {"cite_key": "smith2024", "title": "Fixture", "arxiv_id": "0000.00001v1"}
    fulltext = {"path": "https://arxiv.org/pdf/0000.00001v1", "success": True,
                "page_texts": ["First page", "Second page"], "page_count": 2, "document_sha256": "fixture"}
    source = snapshot_source(tmp_path, paper, [fulltext])
    assert source["total_pages"] == 2
    assert [p["kind"] for p in source["parts"]] == ["pdf_page", "pdf_page"]
    assert (tmp_path / source["parts"][1]["path"]).read_text() == "Second page"


def test_every_markdown_and_tex_occurrence_is_reviewed(sources):
    root, _ = sources
    build(sources)
    texts = manuscripts(root)
    reviewer = Reviewer()
    report = build_citation_support(root, reviewer=reviewer)
    assert report["status"] == "verified" and len(report["citations"]) == 2
    assert len(reviewer.calls) == 2
    assert citation_support_issues(root, report, texts) == []


def test_real_paper_can_be_used_for_supported_and_contradicted_sentences(sources):
    root, _ = sources
    build(sources)
    manuscripts(root)
    path = root / "paper_final.md"
    path.write_text(path.read_text() + "\n\nThe intervention cures all diseases. [smith2024]")
    report = build_citation_support(root, reviewer=Reviewer())
    assert report["status"] != "verified"
    assert {e["status"] for e in report["citations"]} == {"verified", "contradicted"}
    texts = {n: (root / n).read_text() for n in ("paper_final.md", "paper.tex")}
    assert any("claim_support_missing" in i for i in citation_support_issues(root, report, texts))


def test_citation_boundaries_include_predicate_after_inline_citation(sources):
    root, _ = sources
    build(sources)
    text = "This intervention [smith2024] cures all diseases."
    (root / "paper_final.md").write_text(text)
    (root / "paper.tex").write_text(text.replace("[smith2024]", r"\cite{smith2024}"))
    report = build_citation_support(root, reviewer=Reviewer())
    assert all(e["status"] == "contradicted" for e in report["citations"])


def test_post_review_manuscript_change_and_literature_version_change_invalidate_support(sources):
    root, _ = sources
    build(sources)
    texts = manuscripts(root)
    report = build_citation_support(root, reviewer=Reviewer())
    (root / "paper.tex").write_text(texts["paper.tex"] + " Changed.")
    assert "stale_citation_manuscript_bindings" in citation_support_issues(root, report, texts)


def test_citation_budget_does_not_turn_unreviewed_sentences_into_success(sources):
    root, _ = sources
    build(sources)
    text = manuscripts(root)
    report = build_citation_support(root, reviewer=Reviewer(), max_calls=1)
    assert report["review_budget"]["calls"] == 1 and report["status"] == "unavailable"
    assert citation_support_issues(root, report, text)


def test_final_legacy_review_cannot_authorize_unrelated_use_of_same_key(delivery):
    from researchclaw.pipeline.final_acceptance import assess_delivery
    path = delivery / "paper_final.md"
    path.write_text(path.read_text() + "\n\nAn unsupported medical claim. [smith2024]")
    support_path = delivery / "citation_support.json"
    support = json.loads(support_path.read_text())
    support["manuscript_hashes"]["paper_final.md"] = file_hash(path)
    write_json(support_path, support)
    review(delivery)
    report = assess_delivery(delivery)
    assert report["dimensions"]["citations"] == "failed"
    assert any(i["reason"].startswith("claim_support_missing:smith2024:paper_final.md") for i in report["issues"])


def test_formal_stage6_pauses_instead_of_inventing_template_cards(tmp_path):
    from researchclaw.pipeline.stage_impls._literature import _execute_knowledge_extract
    stage = tmp_path / "stage-06"
    stage.mkdir()
    prior = tmp_path / "stage-05"
    prior.mkdir()
    (prior / "shortlist.jsonl").write_text(json.dumps({"cite_key": "smith2024", "title": "Unavailable paper"}))
    result = _execute_knowledge_extract(stage, tmp_path, config(tmp_path), AdapterBundle())
    assert result.status == StageStatus.PAUSED and result.decision == "literature_evidence_incomplete"
    assert "Template key finding" not in (stage / "cards/evidence_index.md").read_text()
    assert json.loads((tmp_path / "literature_evidence.json").read_text())["cards"] == []


def test_stage6_success_records_semantic_review_without_claiming_exhaustiveness(sources, monkeypatch):
    from researchclaw.pipeline.stage_impls._literature import _execute_knowledge_extract
    root, paper = sources
    for number in (4, 5, 6):
        (root / f"stage-{number:02d}").mkdir()
    (root / "stage-05/shortlist.jsonl").write_text(json.dumps(paper))
    write_json(root / "stage-04/search_meta.json", {"status": "results_found", "queries_used": ["fixture query"]})
    monkeypatch.setattr("researchclaw.llm.build_reviewer_llm", lambda cfg: None)
    result = _execute_knowledge_extract(root / "stage-06", root, config(root), AdapterBundle(), llm=Reviewer())
    assert result.status == StageStatus.DONE
    assert json.loads((root / "literature_coverage.json").read_text())["exhaustive"] is False


def test_citation_parser_retains_repeated_and_grouped_occurrences():
    text = r"A \citep[p. 4]{one2024,two2023}. B [@one2024; @three2022]. C [one2024]."
    occurrences = citation_occurrences(text)
    assert len(occurrences) == 5
    assert sum(o["cite_key"] == "one2024" for o in occurrences) == 3


def test_refresh_preserves_previous_sources_and_invalidates_dependent_reviews(sources):
    root, _ = sources
    first = build(sources)
    manuscripts(root)
    build_citation_support(root, reviewer=Reviewer())
    path = root / "literature_input/paper.txt"
    path.write_text(path.read_text() + "\nAn updated paragraph.")
    second = build(sources)
    assert first["version"] != second["version"]
    validate_evidence(root, first)  # Old content-addressed source still exists.
    assert json.loads((root / "citation_support.json").read_text())["status"] == "invalidated_by_literature_refresh"
    assert list((root / "evidence_artifacts/literature_history/literature_evidence.json").glob("*.json"))


def test_partial_pdf_does_not_claim_complete_fulltext_coverage(tmp_path):
    paper = {"cite_key": "smith2024", "title": "Fixture", "url": "https://example.test/paper"}
    fulltexts = [{"path": paper["url"], "success": True, "page_texts": [QUOTE], "page_count": 10}]
    bundle = build_evidence(tmp_path, [paper], fulltexts=fulltexts, llm=Reviewer(), search_log={"status": "results_found"})
    assert bundle["sources"][0]["retrieval_scope"] == "partial_or_unmeasured"
    assert coverage_report(bundle)["status"] == "needs_more_evidence"


def test_real_pdf_extraction_preserves_pages_and_original_digest(tmp_path):
    fitz = pytest.importorskip("fitz")
    from researchclaw.web.pdf_extractor import PDFExtractor
    path = tmp_path / "fixture.pdf"
    with fitz.open() as doc:
        for text in ("First page evidence.", "Second page limitations."):
            page = doc.new_page()
            page.insert_text((72, 72), text)
        doc.save(path)
    complete = PDFExtractor().extract(path)
    partial = PDFExtractor(max_pages=1).extract(path)
    assert complete.success and partial.success
    assert complete.page_count == partial.page_count == 2
    assert len(complete.page_texts) == 2 and len(partial.page_texts) == 1
    assert "Second page" in complete.page_texts[1]
    assert complete.document_sha256 == partial.document_sha256 == file_hash(path)


def test_stage4_keeps_local_bibliography_when_remote_search_succeeds(sources, monkeypatch):
    from dataclasses import replace
    from researchclaw.pipeline.stage_impls._literature import _execute_literature_collect
    from researchclaw.literature.models import Paper
    root, paper = sources
    remote = Paper(paper_id="fixture-remote", title="Remote result", year=2025, source="fixture")
    monkeypatch.setattr("researchclaw.literature.search.search_papers_multi_query", lambda *a, **k: [remote])
    monkeypatch.setattr("researchclaw.data.load_seminal_papers", lambda *a: [])
    stage = root / "stage-04"
    stage.mkdir()
    cfg = config(root)
    cfg = replace(cfg, web_search=replace(cfg.web_search, enabled=False))
    _execute_literature_collect(stage, root, cfg, AdapterBundle())
    bibliography = (stage / "references.bib").read_text(encoding="utf-8")
    assert "{" + paper["cite_key"] + "," in bibliography
    assert bibliography.count("{" + remote.cite_key + ",") == 1


@pytest.mark.parametrize("claim,expected", [
    ("The intervention reduces loss in setting A only.", StageStatus.DONE),
    ("The intervention cures all diseases.", StageStatus.FAILED),
])
def test_stage23_requires_claim_support_even_when_metadata_is_verified(sources, monkeypatch, claim, expected):
    from researchclaw.pipeline.stage_impls import _review_publish
    from researchclaw.literature.verify import CitationResult, VerificationReport, VerifyStatus
    root, paper = sources
    build(sources)
    prior, stage = root / "stage-22", root / "stage-23"
    prior.mkdir()
    stage.mkdir()
    (prior / "references.bib").write_text("@article{smith2024,title={Fixture}}")
    (prior / "paper_final.md").write_text(claim + " [smith2024]")
    (prior / "paper.tex").write_text(claim + r" \cite{smith2024}")
    metadata = VerificationReport(total=1, verified=1, results=[
        CitationResult("smith2024", paper["title"], VerifyStatus.VERIFIED, 1.0, "fixture")])
    monkeypatch.setattr("researchclaw.literature.verify.verify_citations", lambda *a, **k: metadata)
    monkeypatch.setattr("researchclaw.llm.build_reviewer_llm", lambda cfg: None)
    monkeypatch.setattr(_review_publish, "_check_citation_relevance", lambda *a: {"smith2024": 1.0})
    result = _review_publish._execute_citation_verify(stage, root, config(root), AdapterBundle(), llm=Reviewer())
    assert result.status == expected
    support = json.loads((root / "citation_support.json").read_text())
    assert len(support["citations"]) == 2
    assert (support["status"] == "verified") is (expected == StageStatus.DONE)
