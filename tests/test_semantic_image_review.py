from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from researchclaw.agents.figure_agent.semantic_image_review import (
    SemanticImageReviewError, review_image_semantics, verify_semantic_image_review,
)


class Reviewer:
    def __init__(self, result, model="independent-vision"):
        self.result = result
        self.model = model
        self.calls = []

    def chat_image(self, prompt, image_bytes, **kwargs):
        self.calls.append((prompt, image_bytes, kwargs))
        return SimpleNamespace(content=json.dumps(self.result), model=self.model)


def verdict(**changes):
    base = {"status": "passed", "semantic_score": 9, "aesthetic_score": 8,
            "issues": [], "repair_prompt": ""}
    return {**base, **changes}


def test_independent_reviewer_receives_actual_bytes_and_freezes_identity():
    reviewer = Reviewer(verdict())
    result = review_image_semantics(reviewer=reviewer, image_bytes=b"actual-png",
                                    diagram_request="A -> B", generator_model="image-generator")
    assert result["status"] == "passed" and result["reviewer_model"] == "independent-vision"
    assert result["generator_model"] == "image-generator"
    assert reviewer.calls[0][1] == b"actual-png"
    assert reviewer.calls[0][2]["temperature"] == 0 and reviewer.calls[0][2]["json_mode"] is True
    assert verify_semantic_image_review(result, image_bytes=b"actual-png",
                                        diagram_request="A -> B",
                                        generator_model="image-generator") == result


def test_failed_review_requires_and_preserves_bounded_repair_instruction():
    issue = {"dimension": "semantic", "severity": "critical", "message": "Missing edge A to B"}
    result = review_image_semantics(
        reviewer=Reviewer(verdict(status="failed", semantic_score=4, issues=[issue],
                                  repair_prompt="Add only the declared A to B edge.")),
        image_bytes=b"png", diagram_request="A -> B", generator_model="generator")
    assert result["status"] == "failed"
    assert result["repair_prompt"] == "Add only the declared A to B edge."


@pytest.mark.parametrize("result", [
    verdict(status="passed", semantic_score=4),
    verdict(status="failed", semantic_score=4, repair_prompt=""),
    verdict(semantic_score=True),
    verdict(issues=[{"dimension": "semantic", "severity": "critical", "message": "x"}]),
    {"status": "passed"},
])
def test_contradictory_or_malformed_verdicts_fail_closed(result):
    with pytest.raises(SemanticImageReviewError):
        review_image_semantics(reviewer=Reviewer(result), image_bytes=b"png",
                               diagram_request="A -> B", generator_model="generator")


def test_generator_model_cannot_review_its_own_image():
    with pytest.raises(SemanticImageReviewError, match="own output"):
        review_image_semantics(reviewer=Reviewer(verdict(), model="same-model"), image_bytes=b"png",
                               diagram_request="A -> B", generator_model="same-model")


def test_missing_visual_capability_and_invalid_json_fail_closed():
    with pytest.raises(SemanticImageReviewError, match="unavailable"):
        review_image_semantics(reviewer=object(), image_bytes=b"png",
                               diagram_request="A -> B", generator_model="generator")
    reviewer = Reviewer(verdict())
    reviewer.chat_image = lambda *args, **kwargs: SimpleNamespace(content="not-json", model="vision")
    with pytest.raises(SemanticImageReviewError, match="invalid JSON"):
        review_image_semantics(reviewer=reviewer, image_bytes=b"png",
                               diagram_request="A -> B", generator_model="generator")


def test_frozen_review_rejects_tampering_and_changed_image():
    result = review_image_semantics(reviewer=Reviewer(verdict()), image_bytes=b"png",
                                    diagram_request="A -> B", generator_model="generator")
    with pytest.raises(SemanticImageReviewError):
        verify_semantic_image_review({**result, "semantic_score": 10}, image_bytes=b"png",
                                     diagram_request="A -> B", generator_model="generator")
    with pytest.raises(SemanticImageReviewError):
        verify_semantic_image_review(result, image_bytes=b"changed",
                                     diagram_request="A -> B", generator_model="generator")


@pytest.mark.parametrize("mutation", [
    {"semantic_score": 11},
    {"aesthetic_score": True},
    {"repair_prompt": None},
    {"issues": [{"dimension": "semantic", "severity": "warning", "message": ""}]},
    {"issues": [{"dimension": "unknown", "severity": "warning", "message": "x"}]},
    {"issues": [{"dimension": "semantic", "severity": "minor", "message": "x"}]},
])
def test_frozen_review_rejects_self_consistent_invalid_fields(mutation):
    from researchclaw.agents.figure_agent import semantic_image_review as module

    result = review_image_semantics(reviewer=Reviewer(verdict()), image_bytes=b"png",
                                    diagram_request="A -> B", generator_model="generator")
    forged = {**result, **mutation}
    payload = dict(forged)
    payload.pop("version")
    forged["version"] = module._version(payload)
    with pytest.raises(SemanticImageReviewError):
        verify_semantic_image_review(forged, image_bytes=b"png",
                                     diagram_request="A -> B", generator_model="generator")
