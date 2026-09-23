import json
import sys

import pytest

from researchclaw.pipeline.evidence_store import content_hash, file_hash
from researchclaw.pipeline.method_semantics import (
    PACKET_NAME, REVIEW_NAME, build_packet, compile_review, reported_equivalence,
    verify_semantic_review, write_packet,
)
from researchclaw.pipeline.research_workbench import WorkbenchError, check_implementation, compile_method
from tests.test_research_inputs import dump


SOURCE = "raise RuntimeError('never imported')\nclass Linear:\n    def predict(self, x):\n        return x\n"


def method_spec():
    return {"schema_version": 1, "method_id": "A_full", "description": "linear projection",
            "variables": {"X": {"shape": ["n", "d"], "description": "inputs"},
                          "W": {"shape": ["d", 1], "description": "weights"},
                          "Y": {"shape": ["n", 1], "description": "predictions"}},
            "equations": [{"id": "projection", "latex": "Y = X W", "inputs": ["X", "W"], "outputs": ["Y"]}],
            "steps": [{"id": "predict", "phase": "inference", "description": "Multiply inputs by weights",
                       "inputs": ["X", "W"], "outputs": ["Y"], "equations": ["projection"], "depends_on": [],
                       "code_file": "model.py", "code_symbol": "Linear.predict"}],
            "losses": [], "stopping_rule": "One inference pass", "complexity": {"time": "O(nd)", "space": "O(n+d)"},
            "shape_checks": [{"op": "matmul", "left": "X", "right": "W", "result": "Y"}]}


def prepared(tmp_path, spec=None, model=SOURCE):
    root = tmp_path / "review-root"
    source = root / "evidence_artifacts" / "protocol_source"
    source.mkdir(parents=True)
    (source / "model.py").write_text(model, encoding="utf-8")
    method = compile_method(spec or method_spec())
    files = {"model.py": file_hash(source / "model.py")}
    code = {"files": files, "code_sha256": content_hash(files),
            "backend": "sandbox", "execution_config": {}}
    dump(root / "method_spec.json", method)
    dump(root / "protocol_code.json", code)
    return root, method, code


def review(method, packet, *, step_verdict="consistent", equation_verdict="consistent",
           spec_quote="linear projection", code_quote="return x", notes="", conclusion=None):
    worst = ("inconsistent" if "inconsistent" in (step_verdict, equation_verdict)
             else "unverifiable" if "unverifiable" in (step_verdict, equation_verdict) else "consistent")
    return {
        "schema_version": 1, "method_version": method["version"], "code_sha256": packet["code_sha256"],
        "packet_version": packet["packet_version"],
        "reviewer": {"checker": "fixture independent reviewer", "evidence": "reviewed packet and archived source"},
        "steps": [{"step": "predict", "verdict": step_verdict, "spec_evidence": spec_quote,
                   "code_evidence": code_quote, "notes": notes}],
        "equations": [{"equation": "projection", "verdict": equation_verdict,
                       "spec_evidence": "Y = X W", "notes": ""}],
        "conclusion": conclusion or {"verdict": worst, "evidence": "derived from the recorded verdicts"},
    }


def test_accepted_review_upgrades_reported_status_only_to_reviewed_informal(tmp_path):
    root, method, code = prepared(tmp_path)
    packet = write_packet(root, method, code)
    assert packet["steps"][0]["source"].splitlines()[-1].strip() == "return x"
    assert packet["packet_version"] and packet["validation"] is None
    document = compile_review(review(method, packet), method, packet)
    assert document["semantic_equivalence"] == "reviewed_informal"
    dump(root / REVIEW_NAME, document)
    verified = verify_semantic_review(root, method, code)
    assert verified["semantic_equivalence"] == "reviewed_informal"
    assert reported_equivalence(root, method) == {"status": "reviewed_informal",
                                                  "checker": "fixture independent reviewer"}
    # The frozen MethodSpec itself keeps the unresolved machine status; the
    # mapping check never declares semantic equivalence.
    assert method["semantic_equivalence"] == "unresolved"
    mapping = check_implementation(method, root / "evidence_artifacts" / "protocol_source")
    assert mapping["semantic_equivalence"] == "unresolved"


def test_review_is_bound_to_packet_and_source_identity(tmp_path):
    root, method, code = prepared(tmp_path)
    packet = write_packet(root, method, code)
    stale_binding = review(method, packet)
    stale_binding["packet_version"] = "stale"
    with pytest.raises(WorkbenchError, match="different frozen version"):
        compile_review(stale_binding, method, packet)
    tampered = dict(packet)
    tampered["steps"] = []
    tampered.pop("packet_version")
    tampered["packet_version"] = content_hash(tampered)
    dump(root / PACKET_NAME, tampered)
    dump(root / REVIEW_NAME, review(method, packet))
    with pytest.raises(WorkbenchError, match="differs from the recomputed packet"):
        verify_semantic_review(root, method, code)
    (root / "evidence_artifacts/protocol_source/model.py").write_text("changed\n", encoding="utf-8")
    with pytest.raises(WorkbenchError, match="source archive changed"):
        build_packet(root, method, code)


def test_verbatim_quotes_are_required_from_spec_and_source(tmp_path):
    root, method, code = prepared(tmp_path)
    packet = build_packet(root, method, code)
    with pytest.raises(WorkbenchError, match="Spec evidence quote"):
        compile_review(review(method, packet, spec_quote="equivalent to the reference implementation"),
                       method, packet)
    with pytest.raises(WorkbenchError, match="Code evidence quote"):
        compile_review(review(method, packet, code_quote="return reference_output()"), method, packet)
    # The empty string is a substring of everything; it is not evidence.
    with pytest.raises(WorkbenchError, match="Code evidence quote"):
        compile_review(review(method, packet, code_quote=""), method, packet)
    with pytest.raises(WorkbenchError, match="Code evidence quote"):
        compile_review(review(method, packet, code_quote="   "), method, packet)


def test_step_without_archived_source_cannot_be_reviewed(tmp_path):
    root, method, code = prepared(tmp_path)
    spec = method_spec()
    spec["steps"].append({"id": "aux", "phase": "inference", "description": "extra step",
                          "inputs": ["X"], "outputs": ["Y"], "equations": ["projection"],
                          "depends_on": [], "code_file": "missing.py", "code_symbol": "aux_fn"})
    unmapped = compile_method(spec)
    unmapped_packet = build_packet(root, unmapped, code)
    assert unmapped_packet["mapping_issues"]
    incomplete = review(unmapped, unmapped_packet)
    incomplete["steps"] = [
        {"step": "predict", "verdict": "consistent", "spec_evidence": "linear projection",
         "code_evidence": "return x", "notes": ""},
        {"step": "aux", "verdict": "consistent", "spec_evidence": "linear projection",
         "code_evidence": "", "notes": ""}]
    with pytest.raises(WorkbenchError, match="no archived source"):
        compile_review(incomplete, unmapped, unmapped_packet)


def test_conclusion_follows_from_verdicts_and_coverage_is_enforced(tmp_path):
    root, method, code = prepared(tmp_path)
    packet = build_packet(root, method, code)
    with pytest.raises(WorkbenchError, match="does not follow"):
        compile_review(review(method, packet, step_verdict="unverifiable", notes="cannot verify without execution",
                              conclusion={"verdict": "consistent", "evidence": "contradicts the records"}),
                       method, packet)
    with pytest.raises(WorkbenchError, match="notes"):
        compile_review(review(method, packet, step_verdict="unverifiable"), method, packet)
    unverifiable = review(method, packet, step_verdict="unverifiable", notes="cannot verify without execution")
    assert compile_review(unverifiable, method, packet)["semantic_equivalence"] == "unresolved"
    inconsistent = review(method, packet, step_verdict="inconsistent", notes="code contradicts the declared projection")
    assert compile_review(inconsistent, method, packet)["semantic_equivalence"] == "contradicted"
    incomplete = review(method, packet)
    incomplete["steps"] = []
    with pytest.raises(WorkbenchError, match="misses step entries"):
        compile_review(incomplete, method, packet)
    unknown = review(method, packet)
    unknown["equations"][0]["equation"] = "other"
    with pytest.raises(WorkbenchError, match="Duplicate or unknown"):
        compile_review(unknown, method, packet)
    invalid = review(method, packet)
    invalid["steps"][0]["verdict"] = "approved"
    with pytest.raises(WorkbenchError, match="Invalid semantic review verdict"):
        compile_review(invalid, method, packet)
    wrong_schema = review(method, packet)
    wrong_schema["schema_version"] = 2
    with pytest.raises(WorkbenchError, match="Unsupported method semantic review schema"):
        compile_review(wrong_schema, method, packet)


def test_missing_or_invalid_review_falls_back_to_frozen_status(tmp_path):
    root, method, code = prepared(tmp_path)
    assert reported_equivalence(root, method) == {"status": "unresolved", "checker": None}
    dump(root / REVIEW_NAME, {"schema_version": 1})
    assert reported_equivalence(root, method) == {"status": "unresolved", "checker": None}
    (root / "protocol_code.json").unlink()
    assert reported_equivalence(root, method) == {"status": "unresolved", "checker": None}
    # A tampered MethodSpec fails closed instead of trusting a stale review.
    packet = write_packet(root, method, code)
    dump(root / REVIEW_NAME, review(method, packet))
    tampered = dict(method)
    tampered["version"] = "tampered"
    with pytest.raises(WorkbenchError, match="Frozen MethodSpec changed"):
        verify_semantic_review(root, tampered, code)
    # A re-signed but edited review is rejected: the version binds every verdict.
    stored = json.loads((root / REVIEW_NAME).read_text())
    stored["conclusion"]["evidence"] = "edited after the fact"
    stored["version"] = "re-signed"
    dump(root / REVIEW_NAME, stored)
    with pytest.raises(WorkbenchError, match="Recorded method semantic review changed"):
        verify_semantic_review(root, method, code)


def test_cli_writes_packet_and_verifies_review(tmp_path, monkeypatch, capsys):
    from researchclaw.pipeline import method_semantics
    root, method, code = prepared(tmp_path)
    monkeypatch.setattr(sys, "argv", ["method_semantics", str(root), "--write-packet"])
    assert method_semantics.main() == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["reviewed_steps"] == 1 and summary["mapping_issues"] == 0
    packet = json.loads((root / PACKET_NAME).read_text())
    dump(root / REVIEW_NAME, review(method, packet))
    monkeypatch.setattr(sys, "argv", ["method_semantics", str(root)])
    assert method_semantics.main() == 0
    assert json.loads(capsys.readouterr().out)["semantic_equivalence"] == "reviewed_informal"


def test_final_acceptance_reflects_review_outcomes(tmp_path):
    from researchclaw.pipeline.final_acceptance import assess_delivery
    root, method, code = prepared(tmp_path)
    packet = write_packet(root, method, code)
    dump(root / REVIEW_NAME, compile_review(review(method, packet), method, packet))
    report = assess_delivery(root, target_status="research_complete")
    assert not any("semantic_review" in issue["reason"] for issue in report["issues"])
    contradicted = review(method, packet, step_verdict="inconsistent",
                          notes="code contradicts the declared projection")
    dump(root / REVIEW_NAME, compile_review(contradicted, method, packet))
    report = assess_delivery(root, target_status="research_complete")
    assert any(issue["reason"] == "method_semantic_review_contradicted" for issue in report["issues"])
    stale = review(method, packet)
    stale["packet_version"] = "stale"
    dump(root / REVIEW_NAME, stale)
    report = assess_delivery(root, target_status="research_complete")
    assert any(issue["reason"] == "invalid_or_stale_method_semantic_review" for issue in report["issues"])


def test_evidence_catalog_carries_review_scope_and_fails_closed(tmp_path):
    from researchclaw.pipeline.manuscript import ManuscriptError, evidence_catalog
    root, method, code = prepared(tmp_path)
    packet = write_packet(root, method, code)
    dump(root / REVIEW_NAME, compile_review(review(method, packet), method, packet))
    entry = evidence_catalog(root)["entries"]["method_semantics"]
    assert entry["data"]["semantic_equivalence"] == "reviewed_informal"
    assert "never a machine proof" in entry["scope"]
    dump(root / REVIEW_NAME, {"schema_version": 1})
    with pytest.raises(ManuscriptError, match="Invalid method semantic review"):
        evidence_catalog(root)
