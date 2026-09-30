"""End-to-end acceptance of structured exports, with offline review fixtures.

The PDF is real and parseable; it is a layout fixture, not a production paper
or evidence that a real compiler/expert/model was used.
"""
import json

import pytest

from researchclaw.literature.evidence import build_citation_support
from researchclaw.pipeline.evidence_store import file_hash
from researchclaw.pipeline.final_acceptance import assess_delivery, compilation_inputs, seal_delivery, validate_seal
from researchclaw.pipeline.manuscript import quality_report
from researchclaw.pipeline.submission_bundle import prepare_submission
from researchclaw.templates.bundle import inspect_constraints
from tests.test_final_acceptance import review, write_json
from tests.test_literature_evidence import Reviewer
from tests.test_research_inputs import inputs
from tests.test_experiment_protocol import spec
from tests.test_manuscript import study
from tests.test_submission_bundle import submission


@pytest.fixture
def formal_delivery(submission):
    root = submission
    contract = json.loads((root / "research_contract.json").read_text())
    write_json(root, "data_preflight.json", {"status": "verified", "contract_version": contract["version"]})
    build_citation_support(root, reviewer=Reviewer())
    write_json(root, "quality_report.json", quality_report(root, 7))
    write_json(root, "verification_report.json", {"status": "verified"})
    write_json(root, "template_constraints.json", inspect_constraints(root))
    assert prepare_submission(root)["status"] == "prepared"
    review(root)
    reviews = json.loads((root / "final_reviews.json").read_text())
    reviews["dimensions"]["anonymity"] = {
        "status": "passed", "checker": "fixture-expert", "evidence": "offline-fixture-review",
    }
    write_json(root, "final_reviews.json", reviews)
    return root


def test_structured_bundle_passes_full_acceptance_and_seal(formal_delivery):
    report = seal_delivery(formal_delivery, target_status="submission_candidate")
    assert report["artifact_status"] == "submission_candidate", report["issues"]
    assert report["issues"] == []
    assert report["target_met"] and validate_seal(formal_delivery)


@pytest.mark.parametrize("filename", ["research_contract.json", "experiment_protocol.json", "manuscript_ir.json"])
def test_removing_structured_source_cannot_downgrade_into_legacy_pass(formal_delivery, filename):
    (formal_delivery / filename).unlink()
    review(formal_delivery)
    report = assess_delivery(formal_delivery, target_status="research_complete")
    assert report["artifact_status"] == "exploratory"
    assert not report["target_met"]


def test_structured_unbound_number_fails_even_after_hashes_and_reviews_refresh(formal_delivery):
    root = formal_delivery
    for name in ("paper.tex", "paper_final.md"):
        with (root / name).open("a", encoding="utf-8", newline="") as stream:
            stream.write("\nProposed achieves 99.99 accuracy on an unseen dataset.\n")
    hashes = {name: file_hash(root / name) for name in ("paper.tex", "paper_final.md")}
    for name in ("numeric_claims.json", "citation_support.json"):
        document = json.loads((root / name).read_text())
        document["manuscript_hashes"] = hashes
        write_json(root, name, document)
    compilation = json.loads((root / "compilation.json").read_text())
    compilation["inputs"] = compilation_inputs(root)
    write_json(root, "compilation.json", compilation)
    review(root)
    report = assess_delivery(root, target_status="research_complete")
    assert report["artifact_status"] == "exploratory"
    assert any(i["reason"] == "unverified_manuscript_claim_coverage" for i in report["issues"])


def test_pdf_header_alone_cannot_earn_submission_status(formal_delivery):
    root = formal_delivery
    (root / "paper.pdf").write_bytes(b"%PDF-1.4\nfixture")
    compilation = json.loads((root / "compilation.json").read_text())
    compilation["pdf_sha256"] = file_hash(root / "paper.pdf")
    write_json(root, "compilation.json", compilation)
    review(root)
    report = assess_delivery(root, target_status="submission_candidate")
    assert not report["target_met"]
    assert any(i["reason"] == "invalid_pdf" for i in report["issues"])
