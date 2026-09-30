"""Offline fault-injection harness: seeded defects meet the real acceptance."""
import json

import pytest

from researchclaw.pipeline.benchmark_harness import (
    DEFECT_OPERATORS,
    BenchmarkHarnessError,
    prepare_benchmark,
)
from researchclaw.pipeline.benchmark_report import (
    build_benchmark_report,
    validate_benchmark_report,
    write_benchmark_report,
)
from researchclaw.pipeline.evidence_store import content_hash
from researchclaw.pipeline.final_acceptance import (
    PRESENTATION_DIMENSIONS, RESEARCH_DIMENSIONS, inventory,
)
from researchclaw.llm.call_ledger import build_call_ledger
from tests.test_formal_acceptance import formal_delivery, submission, study, inputs, spec


def write_json(root, name, value):
    (root / name).write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def template(formal_delivery):
    """A structured research_complete bundle, with unavailable figure review."""
    root = formal_delivery
    write_json(root, "llm_call_ledger.json", build_call_ledger([
        {"status": "succeeded", "fallback_failures": [],
         "prompt_tokens": 4, "completion_tokens": 1, "total_tokens": 5}]))
    reviews = json.loads((root / "final_reviews.json").read_text())
    reviews["input_version"] = content_hash(inventory(root))
    reviews["dimensions"]["figures"]["status"] = "unknown"
    write_json(root, "final_reviews.json", reviews)
    return root


def defect(case_id, category, dimension):
    return {"case_id": case_id, "kind": "defect", "defect_category": category,
            "expected_dimension": dimension}


def outcomes(root):
    report = build_benchmark_report(root)
    return report, {case["case_id"]: case["outcome"] for case in report["cases"]}


def test_control_copies_inherit_the_template_verdict(tmp_path, template):
    manifest = prepare_benchmark(template, tmp_path / "bench",
                                 [{"case_id": "c1", "kind": "control"}])
    assert manifest["cases"][0]["run_dir"] == "runs/c1"
    report, mapped = outcomes(tmp_path / "bench")
    assert mapped == {"c1": "control_passed"}
    assert report["cases"][0]["artifact_status"] == "research_complete"
    assert report["totals"]["control_rejection_rate"] == 0.0
    assert report["totals"]["error_acceptance_rate"] is None
    # The template's own llm ledger flows through the rebuilt resource ledger.
    assert report["cases"][0]["cost"]["total_tokens"] == 5


def test_detectable_defects_are_classified_by_their_dimension(tmp_path, template):
    cases = [defect("d1", "fabricated_citation", "citations"),
             defect("d2", "missing_figure", "figures"),
             defect("d3", "unresolved_placeholder", "consistency"),
             defect("d4", "stale_manuscript", "numeric"),
             defect("d5", "pipeline_blocker", "experiments"),
             defect("d6", "corrupt_call_ledger", "resources")]
    prepare_benchmark(template, tmp_path / "bench", cases)
    report, mapped = outcomes(tmp_path / "bench")
    assert mapped == {f"d{i}": "detected" for i in range(1, 7)}
    assert report["totals"]["detection_rate"] == 1.0
    assert report["totals"]["error_acceptance_rate"] == 0.0
    by_id = {case["case_id"]: case for case in report["cases"]}
    assert by_id["d1"]["expected_dimension"] == "citations"
    assert "citations" in by_id["d1"]["issue_dimensions"]
    assert "figures" in by_id["d2"]["issue_dimensions"]
    assert "consistency" in by_id["d3"]["issue_dimensions"]
    assert "numeric" in by_id["d4"]["issue_dimensions"]
    assert "experiments" in by_id["d5"]["issue_dimensions"]
    assert "resources" in by_id["d6"]["issue_dimensions"]
    # The corrupted ledger stops measuring cost instead of faking a zero.
    assert by_id["d6"]["cost"]["total_tokens"] is None


def test_blind_spot_defect_is_accepted(tmp_path, template):
    prepare_benchmark(template, tmp_path / "bench",
                      [defect("d1", "unbound_claim", "citations"),
                       {"case_id": "c1", "kind": "control"}])
    report, mapped = outcomes(tmp_path / "bench")
    assert mapped == {"d1": "accepted_defect", "c1": "control_passed"}
    # One accepted defect of one defect case is a rate of 1.0.
    assert report["totals"]["error_acceptance_rate"] == 1.0
    assert report["totals"]["detection_rate"] == 0.0
    assert (tmp_path / "bench" / "runs" / "d1" / "response_to_reviewers.txt").is_file()


def test_prepared_benchmark_round_trips_through_the_report_layer(tmp_path, template):
    prepare_benchmark(template, tmp_path / "bench",
                      [defect("d1", "fabricated_citation", "citations"),
                       {"case_id": "c1", "kind": "control"}])
    report = write_benchmark_report(tmp_path / "bench")
    validate_benchmark_report(tmp_path / "bench", report)


def test_harness_is_deterministic_across_runs(tmp_path, template):
    first = prepare_benchmark(template, tmp_path / "a",
                              [defect("d1", "fabricated_citation", "citations"),
                               {"case_id": "c1", "kind": "control"}])
    second = prepare_benchmark(template, tmp_path / "b",
                               [defect("d1", "fabricated_citation", "citations"),
                                {"case_id": "c1", "kind": "control"}])
    assert first == second


def test_malformed_cases_fail_closed(tmp_path, template):
    out = tmp_path / "bench"
    bad = [
        [],
        ["not-a-mapping"],
        [{"case_id": "", "kind": "control"}],
        [{"case_id": "c1", "kind": "control"}, {"case_id": "c1", "kind": "control"}],
        [{"case_id": "c1", "kind": "mystery"}],
        [defect("d1", "unknown_category", "citations")],
        [defect("d1", "fabricated_citation", "banana")],
        [dict(defect("d1", "fabricated_citation", "citations"), kind="control")],
        [{"case_id": "c1", "kind": "control", "defect_category": "unbound_claim"}],
    ]
    for cases in bad:
        with pytest.raises(BenchmarkHarnessError, match="malformed"):
            prepare_benchmark(template, out, cases)


def test_hostile_case_ids_fail_closed(tmp_path, template):
    # A case_id becomes a path component under runs/: traversal, whitespace,
    # and over-long ids must be rejected before any copy happens.
    out = tmp_path / "bench"
    hostile = [
        [{"case_id": "../escape", "kind": "control"}],
        [{"case_id": "deep/../..//escape", "kind": "defect",
          "defect_category": "fabricated_citation",
          "expected_dimension": "citations"}],
        [{"case_id": "  ", "kind": "control"}],
        [{"case_id": "a" * 65, "kind": "control"}],
        [{"case_id": ".hidden", "kind": "control"}],
    ]
    for cases in hostile:
        with pytest.raises(BenchmarkHarnessError, match="malformed"):
            prepare_benchmark(template, out, cases)
    assert not out.exists()


def test_template_and_output_guards_fail_closed(tmp_path, template):
    with pytest.raises(BenchmarkHarnessError, match="already exists"):
        prepare_benchmark(template, tmp_path / "bench", [defect("d1", "fabricated_citation", "citations")])
        prepare_benchmark(template, tmp_path / "bench", [defect("d1", "fabricated_citation", "citations")])
    with pytest.raises(BenchmarkHarnessError, match="inside the template"):
        prepare_benchmark(template, template / "bench", [{"case_id": "c1", "kind": "control"}])
    bare = tmp_path / "bare"
    bare.mkdir()
    with pytest.raises(BenchmarkHarnessError, match="final_reviews"):
        prepare_benchmark(bare, tmp_path / "bench2", [{"case_id": "c1", "kind": "control"}])
    with pytest.raises(BenchmarkHarnessError, match="missing"):
        prepare_benchmark(tmp_path / "nope", tmp_path / "bench3",
                          [{"case_id": "c1", "kind": "control"}])


def test_operator_seeds_require_their_template_files(tmp_path, template):
    (template / "paper_final.md").unlink()
    (template / "final_reviews.json").write_text(json.dumps({
        "input_version": content_hash(inventory(template)),
        "dimensions": {d: {"status": "passed", "checker": "c", "evidence": "e"}
                       for d in (*RESEARCH_DIMENSIONS, *PRESENTATION_DIMENSIONS)}}),
        encoding="utf-8")
    with pytest.raises(BenchmarkHarnessError, match="expects"):
        prepare_benchmark(template, tmp_path / "bench",
                          [defect("d1", "unresolved_placeholder", "consistency")])
    assert set(DEFECT_OPERATORS) == {"fabricated_citation", "missing_figure",
                                     "unresolved_placeholder", "stale_manuscript",
                                     "pipeline_blocker", "corrupt_call_ledger",
                                     "unbound_claim"}
