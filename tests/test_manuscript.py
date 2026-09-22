import json
from types import SimpleNamespace

import pytest

from researchclaw.adapters import AdapterBundle
from researchclaw.literature.evidence import build_evidence, write_json
from researchclaw.literature.positioning import build_novelty_matrix, contribution_ledger
from researchclaw.pipeline.evidence_store import content_hash
from researchclaw.pipeline.manuscript import (
    CHECK_SYSTEM, SECTION_SYSTEM, WRITE_SYSTEM, ManuscriptError, build_manuscript,
    evidence_catalog, export_manuscript, package_manuscript, quality_report,
    render_manuscript, review_manuscript, section_tasks, validate_manuscript, verify_exports,
)
from researchclaw.pipeline.stages import StageStatus
from tests.test_research_inputs import inputs
from tests.test_experiment_protocol import spec, freeze, evaluate_cells
from tests.test_literature_evidence import QUOTE, Reviewer
from tests.test_literature_positioning import PositionReviewer


@pytest.fixture
def study(inputs, spec):
    root, contract, protocol, cfg = freeze(inputs, spec)
    store = evaluate_cells(root, contract, protocol)
    write_json(root / "evidence_store.json", store.to_dict())
    directory = root / "literature_input"
    directory.mkdir()
    (directory / "paper.txt").write_text(QUOTE, encoding="utf-8")
    paper = {"cite_key": "smith2024", "title": "Fixture intervention", "path": "paper.txt"}
    write_json(directory / "sources.json", {"schema_version": 1, "papers": [paper]})
    build_evidence(root, [paper], llm=Reviewer(), search_log={"status": "results_found"})
    build_novelty_matrix(root, ["Test the intervention"], reviewer=PositionReviewer())
    contribution_ledger(root)
    return root, cfg


class Writer:
    def __init__(self, *, unsupported=False, omit=False, bad_ref=False, long=False, low_section=None):
        self.calls = []
        self.unsupported, self.omit, self.bad_ref = unsupported, omit, bad_ref
        self.long, self.low_section = long, low_section

    def chat(self, messages, *, system, **kwargs):
        payload = json.loads(messages[0]["content"])
        self.calls.append((system, payload))
        if system == WRITE_SYSTEM:
            text = "These observations apply only to the declared conditions; alternative mechanisms remain untested."
            if self.long:
                text = "A contextual limitation remains. " * 155 + " TAIL_MARKER"
            if self.unsupported:
                text += " Unsupported universal cure."
            ids = list(payload["evidence"])
            if self.omit:
                ids = ids[:1]
            if self.bad_ref:
                ids = ["result:invented"]
            response = {"blocks": [{"text": text, "kind": "limitation", "evidence_ids": ids}]}
            if self.long:
                response["blocks"] *= 2
        elif system == CHECK_SYSTEM:
            response = {"status": "contradicted" if "Unsupported universal cure" in payload["text"] else "supported",
                        "rationale": "Controlled fixture judgment, not real scientific review."}
        elif system == SECTION_SYSTEM:
            response = {"status": "passed", "score": 4 if payload["contract"]["role"] == self.low_section else 8,
                        "issues": []}
        else:
            raise AssertionError(system)
        return SimpleNamespace(content=json.dumps(response), model="fixture-writer-reviewer")


def build(study, writer=None, **kwargs):
    root, _ = study
    return build_manuscript(root, "Fixture study", llm=writer or Writer(), **kwargs)


def rehash(report):
    report.pop("version")
    report["version"] = content_hash(report)


def test_shared_export_binds_every_result_and_paragraph(study):
    root, _ = study
    report = build(study)
    validate_manuscript(root, report)
    export_manuscript(root, root)
    verify_exports(root)
    claims = json.loads((root / "numeric_claims.json").read_text())
    results = json.loads((root / "evidence_store.json").read_text())["records"]
    assert {c["result_id"] for c in claims["claims"]} == {r["result_id"] for r in results}
    for claim in claims["claims"]:
        for name, span in claim["spans"].items():
            assert (root / name).read_text(encoding="utf-8")[span["start"]:span["end"]] == claim["rendered"]
    assert "[@smith2024]" in (root / "paper_final.md").read_text(encoding="utf-8")
    assert r"\cite{smith2024}" in (root / "paper.tex").read_text(encoding="utf-8")
    assert quality_report(root, 7)["verdict"] == "proceed"


@pytest.mark.parametrize("writer", [Writer(unsupported=True), Writer(omit=True), Writer(bad_ref=True)])
def test_unsupported_omitted_or_invented_evidence_stays_incomplete(study, writer):
    root, _ = study
    report = build(study, writer)
    assert report["status"] == "incomplete"
    with pytest.raises(ManuscriptError):
        render_manuscript(root, report)


def test_missing_reviewer_or_budget_cannot_earn_reviewed_state(study):
    root, _ = study
    report = build_manuscript(root, "Fixture study", llm=None)
    assert report["status"] == "incomplete" and report["review_budget"]["calls"] == 0
    report = build(study, max_calls=3)
    assert report["status"] == "incomplete" and report["review_budget"]["calls"] == 3
    complete = build(study)
    assert complete["status"] == "reviewed"
    cached = Writer()
    assert build(study, cached)["status"] == "reviewed"
    assert cached.calls == []


def test_long_manuscript_reviews_last_paragraph_without_truncation(study):
    root, _ = study
    writer = Writer(long=True)
    report = build(study, writer)
    text, _ = render_manuscript(root, report)
    assert len(text["paper_final.md"]) > 80000
    paragraphs = [p for system, p in writer.calls if system == CHECK_SYSTEM]
    assert paragraphs and all(p["text"].endswith("TAIL_MARKER") for p in paragraphs)
    assert len(paragraphs) == sum(len(s["blocks"]) for s in report["sections"])


@pytest.mark.parametrize("mutation", ["text", "evidence", "verdict", "title"])
def test_rehashed_manuscript_cannot_reuse_changed_review(study, mutation):
    root, _ = study
    report = build(study)
    block = report["sections"][0]["blocks"][0]
    if mutation == "text":
        block["text"] = "A completely changed scientific conclusion."
    elif mutation == "evidence":
        block["evidence_ids"] = block["evidence_ids"][:1]
    elif mutation == "title":
        report["title"] = "A universal cure"
    else:
        block["review"]["verdict"]["rationale"] = "New verdict without a review"
    rehash(report)
    with pytest.raises(ManuscriptError):
        validate_manuscript(root, report)


def test_raw_evidence_changes_invalidate_entire_manuscript(study):
    root, _ = study
    report = build(study)
    store = json.loads((root / "evidence_store.json").read_text())
    path = root / store["records"][0]["artifacts"][1][0]
    path.write_text("Changed raw evidence")
    with pytest.raises(ManuscriptError, match="Invalid result"):
        validate_manuscript(root, report)


def test_exported_text_and_numeric_attribution_cannot_change(study):
    root, _ = study
    build(study)
    export_manuscript(root, root)
    tex = root / "paper.tex"
    tex.write_text(tex.read_text(encoding="utf-8") + "Extra unsupported conclusion", encoding="utf-8")
    with pytest.raises(ManuscriptError, match="shared ManuscriptIR"):
        verify_exports(root)
    export_manuscript(root, root)
    path = root / "numeric_claims.json"
    claims = json.loads(path.read_text())
    claims["claims"][0]["key"]["method"] = "A wrong method"
    write_json(path, claims)
    with pytest.raises(ManuscriptError, match="Numeric"):
        verify_exports(root)


def test_export_newline_rewrite_invalidates_exact_byte_contract(study):
    root, _ = study
    build(study)
    export_manuscript(root, root)
    path = root / "paper.tex"
    path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))
    with pytest.raises(ManuscriptError, match="shared ManuscriptIR"):
        verify_exports(root)


def test_one_low_quality_section_cannot_be_averaged_away(study):
    root, _ = study
    build(study, Writer(low_section="conclusion"))
    report = quality_report(root, 7)
    assert report["verdict"] == "reject" and report["score_1_to_10"] == 4


def test_pipeline_writing_revision_gate_and_export_use_shared_source(study, monkeypatch):
    from researchclaw.pipeline.stage_impls._paper_writing import _execute_paper_draft
    from researchclaw.pipeline.stage_impls._review_publish import (
        _execute_peer_review, _execute_paper_revision, _execute_quality_gate, _execute_export_publish,
    )
    root, cfg = study
    monkeypatch.setattr("researchclaw.llm.build_reviewer_llm", lambda cfg: None)
    for number in (17, 18, 19, 20, 22):
        (root / f"stage-{number}").mkdir()
    writer = Writer()
    assert _execute_paper_draft(root / "stage-17", root, cfg, AdapterBundle(), llm=writer).status == StageStatus.DONE
    assert _execute_peer_review(root / "stage-18", root, cfg, AdapterBundle(), llm=writer).status == StageStatus.DONE
    assert _execute_paper_revision(root / "stage-19", root, cfg, AdapterBundle(), llm=writer).status == StageStatus.DONE
    assert _execute_quality_gate(root / "stage-20", root, cfg, AdapterBundle()).status == StageStatus.DONE
    assert _execute_export_publish(root / "stage-22", root, cfg, AdapterBundle()).status == StageStatus.DONE
    revised = root / "stage-19/paper_revised.md"
    revised.write_text(revised.read_text(encoding="utf-8") + "\nUnreviewed edit", encoding="utf-8")
    assert _execute_quality_gate(root / "stage-20", root, cfg, AdapterBundle()).status == StageStatus.FAILED


def test_packaging_preserves_exact_exports_and_disables_compile_rewrites(study, monkeypatch):
    root, cfg = study
    build(study)
    for number in (20, 22, 23):
        (root / f"stage-{number}").mkdir()
    export_manuscript(root, root / "stage-22")
    (root / "stage-23/paper_final_verified.md").write_bytes((root / "stage-22/paper_final.md").read_bytes())
    (root / "stage-23/references_verified.bib").write_text("@article{smith2024,title={Fixture}}")
    write_json(root / "stage-23/verification_report.json", {"status": "verified"})
    write_json(root / "stage-20/quality_report.json", quality_report(root, cfg.research.quality_threshold))
    calls = []
    def compile_fixture(path, **kwargs):
        calls.append(kwargs)
        (path.parent / "paper.pdf").write_bytes(b"%PDF-fixture")
        return SimpleNamespace(success=True, errors=[], warnings=[], fixes_applied=[])
    monkeypatch.setattr("researchclaw.templates.compiler.compile_latex", compile_fixture)
    dest = package_manuscript(root, "fixture", cfg)
    verify_exports(dest)
    assert calls == [{"max_attempts": 1, "timeout": 120, "allow_repairs": False}]
    assert (dest / "paper.tex").read_bytes() == (root / "stage-22/paper.tex").read_bytes()
    assert json.loads((dest / "final_acceptance.json").read_text())["artifact_status"] != "submission_candidate"


def test_results_subsections_follow_every_declared_research_question(study):
    root, _ = study
    catalog = evidence_catalog(root)
    results = [t for t in section_tasks(catalog) if t["role"] == "results"]
    assert {t["question_id"] for t in results} == {"rq_main", "rq_ablation"}
    assert all(any(key.startswith("contribution:") for key in t["evidence_ids"]) for t in results)


def test_compiler_readonly_mode_never_sanitizes_or_repairs_sources(tmp_path, monkeypatch):
    from researchclaw.templates import compiler
    tex, bib = tmp_path / "paper.tex", tmp_path / "references.bib"
    tex.write_text(r"\documentclass{article}\begin{document}\badcommand \end{document}")
    bib.write_text("@article{test,title={A & B}}")
    originals = (tex.read_bytes(), bib.read_bytes())
    monkeypatch.setattr(compiler.shutil, "which", lambda name: "fixture")
    monkeypatch.setattr(compiler, "_run_pdflatex", lambda *a: ("! Undefined control sequence.", False))
    monkeypatch.setattr(compiler, "_run_bibtex", lambda *a, **k: None)
    result = compiler.compile_latex(tex, allow_repairs=False)
    assert not result.success and result.fixes_applied == []
    assert (tex.read_bytes(), bib.read_bytes()) == originals


def test_peer_review_uses_complete_sections_and_budget_absence_is_unknown(study):
    root, _ = study
    report = build(study, Writer(long=True))
    reviewer = Writer()
    peer = review_manuscript(root, reviewer=reviewer)
    assert peer["status"] == "reviewed"
    assert len(peer["sections"]) == len(report["sections"])
    paragraphs = [payload for system, payload in reviewer.calls if system == CHECK_SYSTEM]
    assert all(p["text"].endswith("TAIL_MARKER") for p in paragraphs)
    unknown = review_manuscript(root, reviewer=Writer(), max_calls=1)
    assert unknown["status"] == "needs_revision" and unknown["calls"] == 1


def test_changed_authoritative_evidence_discards_cached_sections(study):
    root, _ = study
    first = build(study)
    # New comparison idea legitimately changes the evidence graph.
    build_novelty_matrix(root, ["A different intervention"], reviewer=PositionReviewer())
    contribution_ledger(root)
    writer = Writer()
    second = build(study, writer)
    assert second["catalog_version"] != first["catalog_version"] and writer.calls
    with pytest.raises(ManuscriptError, match="dependencies"):
        validate_manuscript(root, first)


def test_model_cannot_insert_even_plausible_raw_metric_literals(study):
    class NumericWriter(Writer):
        def chat(self, messages, *, system, **kwargs):
            response = super().chat(messages, system=system, **kwargs)
            if system == WRITE_SYSTEM:
                data = json.loads(response.content)
                data["blocks"][0]["text"] = "The accuracy is 0.999 under the declared conditions."
                response.content = json.dumps(data)
            return response
    report = build(study, NumericWriter())
    assert report["status"] == "incomplete"
    assert any("generated result records" in error for section in report["sections"]
               for attempt in section["attempts"] for error in attempt["errors"])
