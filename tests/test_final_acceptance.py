import json
from dataclasses import asdict

import pytest

from researchclaw.pipeline.evidence_store import EvidenceKey, EvidenceRecord, EvidenceStore, content_hash, file_hash
from researchclaw.pipeline.final_acceptance import (
    RESEARCH_DIMENSIONS, PRESENTATION_DIMENSIONS, assess_delivery, compilation_inputs,
    inventory, seal_delivery, validate_seal,
)


def write_json(root, name, value):
    (root / name).write_text(json.dumps(value), encoding="utf-8")


def review(root):
    write_json(root, "final_reviews.json", {
        "input_version": content_hash(inventory(root)),
        "dimensions": {d: {"status": "not_applicable" if d == "theory" else "passed",
                            "checker": "fixture-expert", "evidence": "fixture-review-log"}
                       for d in (*RESEARCH_DIMENSIONS, *PRESENTATION_DIMENSIONS)},
    })


@pytest.fixture
def delivery(tmp_path):
    root = tmp_path
    (root / "paper.tex").write_text(r"\section{Results} Baseline achieves 81.00. \cite{smith2024}", encoding="utf-8")
    (root / "paper_final.md").write_text("# Results\nBaseline achieves 81.00. [smith2024]", encoding="utf-8")
    (root / "references.bib").write_text("@article{smith2024,title={A paper}}", encoding="utf-8")
    (root / "paper.pdf").write_bytes(b"%PDF-1.4\nfixture")
    (root / "raw.csv").write_text("label,prediction\n1,1\n", encoding="utf-8")
    (root / "source.txt").write_text("Baseline reference evidence.", encoding="utf-8")
    key = EvidenceKey("data", "v1", "test", "Baseline", "config", "42", "accuracy", "mean")
    store = EvidenceStore()
    result_id = store.add(EvidenceRecord(key, 81.0, "percent", "success", "commit", "env",
                                         "run", "evaluator", True, (("raw.csv", file_hash(root / "raw.csv")),)))
    write_json(root, "evidence_store.json", store.to_dict())
    write_json(root, "experiment_protocol.json", {"required_keys": [asdict(key)]})
    papers = {name: file_hash(root / name) for name in ("paper.tex", "paper_final.md")}
    spans = {}
    for name in papers:
        start = (root / name).read_text(encoding="utf-8").index("81.00")
        spans[name] = {"start": start, "end": start + 5}
    write_json(root, "numeric_claims.json", {"evidence_version": store.version,
               "manuscript_hashes": papers, "claims": [{"result_id": result_id, "key": asdict(key),
               "value": 81.0, "unit": "percent", "decimals": 2, "rendered": "81.00", "spans": spans}]})
    write_json(root, "citation_support.json", {"manuscript_hashes": papers, "citations": [{
        "cite_key": "smith2024", "status": "verified", "checker": "fixture-expert",
        "source": "source.txt", "source_sha256": file_hash(root / "source.txt"),
        "locator": "p.1", "excerpt": "Baseline reference evidence.", "claim": "Baseline achieves 81.00.",
    }]})
    write_json(root, "quality_report.json", {"score_1_to_10": 8})
    write_json(root, "verification_report.json", {"status": "verified"})
    write_json(root, "compilation.json", {"success": True, "inputs": compilation_inputs(root),
                                          "pdf_sha256": file_hash(root / "paper.pdf")})
    review(root)
    return root


def test_valid_bound_bundle_can_pass_and_seal(delivery):
    report = seal_delivery(delivery, target_status="submission_candidate")
    assert report["artifact_status"] == "submission_candidate", report["issues"]
    assert report["target_met"] and validate_seal(delivery)


@pytest.mark.parametrize("name", ["paper.tex", "paper_final.md", "references.bib", "raw.csv", "paper.pdf"])
def test_post_review_mutation_invalidates_acceptance_and_seal(delivery, name):
    seal_delivery(delivery, target_status="submission_candidate")
    with (delivery / name).open("ab") as stream:
        stream.write(b" changed")
    assert not validate_seal(delivery)
    assert not assess_delivery(delivery, target_status="submission_candidate")["target_met"]


@pytest.mark.parametrize("filename,payload,dimension", [
    ("quality_report.json", {"score_1_to_10": 2}, "quality"),
    ("compilation.json", {"success": False}, "layout"),
    ("verification_report.json", {"status": "unavailable"}, "citations"),
    ("verification_report.json", {"status": "contradicted"}, "citations"),
    ("citation_support.json", {}, "citations"),
    ("theory_bundle.json", {"obligations": [{"status": "unresolved", "required": True}]}, "theory"),
    ("experiment_protocol.json", {"required_keys": [{"method": "missing"}]}, "experiments"),
    ("pipeline_blockers.json", {"issues": ["retry_exhausted"]}, "experiments"),
])
def test_failed_dimension_cannot_be_overridden_by_review_score(delivery, filename, payload, dimension):
    write_json(delivery, filename, payload)
    review(delivery)  # Even a positive fresh review cannot override deterministic failures.
    report = assess_delivery(delivery, target_status="submission_candidate")
    assert not report["target_met"]
    assert report["dimensions"][dimension] == "failed"


def test_missing_figure_and_placeholder_are_not_accepted(delivery):
    with (delivery / "paper.tex").open("a", encoding="utf-8") as stream:
        stream.write(r"\includegraphics{charts/missing.png} TODO \ref{missing}")
    review(delivery)
    report = assess_delivery(delivery, target_status="submission_candidate")
    assert report["dimensions"]["figures"] == "failed"
    assert report["dimensions"]["consistency"] == "failed"


def test_empty_delivery_is_exploratory_not_submission(tmp_path):
    report = assess_delivery(tmp_path, target_status="submission_candidate")
    assert report["artifact_status"] == "exploratory"
    assert not report["target_met"] and report["issues"]


def test_new_file_invalidates_seal(delivery):
    seal_delivery(delivery)
    (delivery / "new.txt").write_text("new", encoding="utf-8")
    assert not validate_seal(delivery)


def test_removed_citation_requires_explicit_claim_resolution(delivery):
    write_json(delivery, "verification_report.json", {
        "status": "verified", "removed_citation_keys": ["removed2024"], "requires_claim_review": True,
    })
    review(delivery)
    report = assess_delivery(delivery, target_status="submission_candidate")
    assert not report["target_met"]
    assert any(i["reason"] == "removed_claim_unresolved:removed2024" for i in report["issues"])


def test_invented_excerpt_does_not_support_real_citation(delivery):
    data = json.loads((delivery / "citation_support.json").read_text())
    data["citations"][0]["excerpt"] = "This excerpt is absent from the source."
    write_json(delivery, "citation_support.json", data)
    review(delivery)
    assert assess_delivery(delivery)["dimensions"]["citations"] == "failed"


def test_malformed_review_fails_closed(delivery):
    write_json(delivery, "final_reviews.json", {"input_version": content_hash(inventory(delivery)),
                                               "dimensions": ["passed"]})
    report = assess_delivery(delivery, target_status="submission_candidate")
    assert not report["target_met"] and report["artifact_status"] == "exploratory"
