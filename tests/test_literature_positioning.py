import json
from types import SimpleNamespace

import pytest

from researchclaw.literature.evidence import LiteratureEvidenceError, build_evidence, write_json
from researchclaw.literature.positioning import build_novelty_matrix, contribution_ledger, validate_novelty
from researchclaw.pipeline.evidence_store import content_hash
from tests.test_literature_evidence import sources, build, Reviewer
from tests.test_research_inputs import inputs
from tests.test_experiment_protocol import spec, freeze, evaluate_cells


class PositionReviewer:
    def __init__(self, kind="mechanism", status="candidate_difference", wrong_id=False):
        self.kind, self.status, self.wrong_id = kind, status, wrong_id

    def chat(self, messages, **kwargs):
        payload = json.loads(messages[0]["content"])
        return SimpleNamespace(model="fixture-positioning", content=json.dumps({
            "status": self.status, "nearest_card_ids": ["invented" if self.wrong_id else payload["prior_work"][0]["card_id"]],
            "difference_kind": self.kind, "difference": "Fixture mechanism comparison",
            "limitations": "Only the supplied works were compared; originality remains provisional."}))


def test_supported_comparison_is_provisional_not_guaranteed_novelty(sources):
    root, _ = sources
    literature = build(sources)
    matrix = build_novelty_matrix(root, ["Change the intervention mechanism"], reviewer=PositionReviewer())
    validate_novelty(root, matrix, literature)
    assert matrix["status"] == "reviewed" and matrix["novelty_guaranteed"] is False
    assert matrix["rows"][0]["nearest_card_ids"] == [literature["cards"][0]["card_id"]]


@pytest.mark.parametrize("reviewer", [PositionReviewer(kind="naming"), PositionReviewer(wrong_id=True), None])
def test_renaming_unknown_work_and_unavailable_review_are_unresolved(sources, reviewer):
    root, _ = sources
    build(sources)
    matrix = build_novelty_matrix(root, ["A renamed method"], reviewer=reviewer)
    assert matrix["status"] == "unresolved"


def test_no_candidates_cannot_be_high_novelty(tmp_path):
    build_evidence(tmp_path, [{"cite_key": "missing2024", "title": "No source"}])
    matrix = build_novelty_matrix(tmp_path, ["Some idea"], reviewer=PositionReviewer())
    assert matrix["status"] == "unresolved" and matrix["review_calls"] == 0


def test_literature_version_change_invalidates_positioning(sources):
    root, _ = sources
    literature = build(sources)
    matrix = build_novelty_matrix(root, ["An idea"], reviewer=PositionReviewer())
    literature["version"] = "changed"
    with pytest.raises(LiteratureEvidenceError, match="changed"):
        validate_novelty(root, matrix, literature)


def test_contribution_ledger_reports_zero_effect_and_binds_every_result(inputs, spec):
    root, contract, protocol, cfg = freeze(inputs, spec)
    directory = root / "literature_input"
    directory.mkdir()
    from tests.test_literature_evidence import QUOTE
    (directory / "source.txt").write_text(QUOTE)
    paper = {"cite_key": "smith2024", "title": "Fixture", "path": "source.txt"}
    write_json(directory / "sources.json", {"schema_version": 1, "papers": [paper]})
    build_evidence(root, [paper], llm=Reviewer(), search_log={"status": "results_found"})
    build_novelty_matrix(root, ["A method variant"], reviewer=PositionReviewer())
    store = evaluate_cells(root, contract, protocol)
    write_json(root / "evidence_store.json", store.to_dict())
    ledger = contribution_ledger(root)
    assert len(ledger["entries"]) == 2
    assert all(entry["observation"] == "zero_difference" for entry in ledger["entries"])
    assert {rid for e in ledger["entries"] for rid in e["result_ids"]} == set(store.records)
    assert contribution_ledger(root, write=False) == ledger
    # Removing a seed invalidates its comparison, rather than cherry-picking the rest.
    store.records.pop(next(iter(store.records)))
    write_json(root / "evidence_store.json", store.to_dict())
    changed = contribution_ledger(root, write=False)
    assert changed["version"] != ledger["version"]
    assert changed["entries"][0]["status"] == "unresolved"


def test_rehashed_claim_of_guaranteed_novelty_is_rejected(sources):
    root, _ = sources
    literature = build(sources)
    matrix = build_novelty_matrix(root, ["Idea"], reviewer=PositionReviewer())
    matrix["novelty_guaranteed"] = True
    matrix.pop("version")
    matrix["version"] = content_hash(matrix)
    with pytest.raises(LiteratureEvidenceError, match="override"):
        validate_novelty(root, matrix, literature)


@pytest.mark.parametrize("mutation", ["idea", "verdict", "trace"])
def test_rehashed_novelty_cannot_reuse_review_for_changed_idea_or_verdict(sources, mutation):
    root, _ = sources
    literature = build(sources)
    matrix = build_novelty_matrix(root, ["Change intervention mechanism"], reviewer=PositionReviewer())
    row = matrix["rows"][0]
    if mutation == "idea":
        row["idea"] = "An entirely different intervention"
        row["idea_id"] = content_hash(row["idea"])
    elif mutation == "verdict":
        row["status"] = "already_known"
    else:
        row["review_trace"]["response"] = "{}"
    matrix.pop("version")
    matrix["version"] = content_hash(matrix)
    with pytest.raises(LiteratureEvidenceError, match="review"):
        validate_novelty(root, matrix, literature)
