"""Independent semantic and aesthetic review for image-model output."""
from __future__ import annotations

import hashlib
import json
from typing import Any


class SemanticImageReviewError(ValueError):
    pass


def _version(value: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode("utf-8")).hexdigest()


_SYSTEM = """You are an independent reviewer of a scientific framework diagram.
Inspect the attached image itself against the declared diagram request. Return JSON only:
{"status":"passed|failed","semantic_score":1,"aesthetic_score":1,
 "issues":[{"dimension":"semantic|aesthetic","severity":"warning|critical","message":"..."}],
 "repair_prompt":"..."}
Scores are integers from 1 to 10. Pass only when both scores are at least 7 and there are
no critical issues. Check missing, added, renamed, or misleading nodes/arrows and whether
the diagram remains legible, restrained, and publication suitable. A repair_prompt is
required on failure and must describe only changes supported by the declared request."""


def _document(content: str) -> dict[str, Any]:
    if not isinstance(content, str):
        raise SemanticImageReviewError("Image reviewer returned non-text content")
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            text = "\n".join(lines[1:-1])
            if text.lstrip().startswith("json"):
                text = text.lstrip()[4:].lstrip()
    try:
        value = json.loads(text)
    except (json.JSONDecodeError, TypeError) as exc:
        raise SemanticImageReviewError("Image reviewer returned invalid JSON") from exc
    if not isinstance(value, dict):
        raise SemanticImageReviewError("Image reviewer response must be an object")
    return value


def review_image_semantics(*, reviewer: Any, image_bytes: bytes, diagram_request: str,
                           generator_model: str | None) -> dict[str, Any]:
    """Call a distinct vision reviewer and normalize its fail-closed verdict."""
    if reviewer is None or not callable(getattr(reviewer, "chat_image", None)):
        raise SemanticImageReviewError("Independent vision reviewer is unavailable")
    if not isinstance(diagram_request, str) or not diagram_request.strip():
        raise SemanticImageReviewError("Diagram request is unavailable")
    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        mime_type = "image/png"
    elif image_bytes.startswith(b"\xff\xd8\xff"):
        mime_type = "image/jpeg"
    elif image_bytes.startswith(b"RIFF") and image_bytes[8:12] == b"WEBP":
        mime_type = "image/webp"
    else:
        raise SemanticImageReviewError("Reviewed image has an unsupported signature")
    response = reviewer.chat_image(
        "Declared diagram request:\n" + diagram_request,
        image_bytes,
        mime_type=mime_type,
        max_tokens=2048,
        temperature=0,
        json_mode=True,
        system=_SYSTEM,
    )
    reviewer_model = str(getattr(response, "model", "") or "").strip()
    if not reviewer_model:
        raise SemanticImageReviewError("Reviewer did not report its served model")
    if generator_model and reviewer_model.casefold() == str(generator_model).strip().casefold():
        raise SemanticImageReviewError("Image generator cannot review its own output")
    value = _document(getattr(response, "content", ""))
    if set(value) != {"status", "semantic_score", "aesthetic_score", "issues", "repair_prompt"}:
        raise SemanticImageReviewError("Image reviewer response has an invalid schema")
    if value["status"] not in {"passed", "failed"}:
        raise SemanticImageReviewError("Image reviewer status is invalid")
    for field in ("semantic_score", "aesthetic_score"):
        if type(value[field]) is not int or not 1 <= value[field] <= 10:
            raise SemanticImageReviewError("Image reviewer score is invalid")
    if not isinstance(value["issues"], list):
        raise SemanticImageReviewError("Image reviewer issues must be a list")
    issues = []
    for issue in value["issues"]:
        if (not isinstance(issue, dict)
                or set(issue) != {"dimension", "severity", "message"}
                or issue.get("dimension") not in {"semantic", "aesthetic"}
                or issue.get("severity") not in {"warning", "critical"}
                or not isinstance(issue.get("message"), str) or not issue["message"].strip()):
            raise SemanticImageReviewError("Image reviewer issue is invalid")
        issues.append({**issue, "message": issue["message"].strip()})
    repair_prompt = value["repair_prompt"]
    if not isinstance(repair_prompt, str):
        raise SemanticImageReviewError("Image reviewer repair prompt is invalid")
    objectively_passed = (value["semantic_score"] >= 7 and value["aesthetic_score"] >= 7
                          and not any(issue["severity"] == "critical" for issue in issues))
    if (value["status"] == "passed") != objectively_passed:
        raise SemanticImageReviewError("Image reviewer verdict contradicts its scores or issues")
    if not objectively_passed and not repair_prompt.strip():
        raise SemanticImageReviewError("Failed image review needs a repair prompt")
    record = {
        "schema_version": 1,
        "checker": "independent-vision-review/v1",
        "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
        "request_sha256": hashlib.sha256(diagram_request.encode("utf-8")).hexdigest(),
        "generator_model": generator_model,
        "reviewer_model": reviewer_model,
        "status": value["status"],
        "semantic_score": value["semantic_score"],
        "aesthetic_score": value["aesthetic_score"],
        "issues": issues,
        "repair_prompt": repair_prompt.strip(),
    }
    return {**record, "version": _version(record)}


def verify_semantic_image_review_record(record: dict[str, Any], *, image_bytes: bytes,
                                        diagram_request: str,
                                        generator_model: str | None) -> dict[str, Any]:
    """Verify any frozen verdict and its binding to the reviewed inputs."""
    if not isinstance(record, dict) or set(record) != {
            "schema_version", "checker", "image_sha256", "request_sha256", "generator_model",
            "reviewer_model", "status", "semantic_score", "aesthetic_score", "issues",
            "repair_prompt", "version"}:
        raise SemanticImageReviewError("Frozen image review has an invalid schema")
    data = dict(record)
    version = data.pop("version")
    if version != _version(data):
        raise SemanticImageReviewError("Frozen image review changed")
    issues = record["issues"]
    valid_issues = (isinstance(issues, list) and all(
        isinstance(issue, dict)
        and set(issue) == {"dimension", "severity", "message"}
        and issue.get("dimension") in {"semantic", "aesthetic"}
        and issue.get("severity") in {"warning", "critical"}
        and isinstance(issue.get("message"), str)
        and bool(issue["message"].strip())
        for issue in issues
    ))
    if (record["schema_version"] != 1 or record["checker"] != "independent-vision-review/v1"
            or record["image_sha256"] != hashlib.sha256(image_bytes).hexdigest()
            or record["request_sha256"] != hashlib.sha256(diagram_request.encode("utf-8")).hexdigest()
            or record["generator_model"] != generator_model
            or record["status"] not in {"passed", "failed"}
            or not isinstance(record["reviewer_model"], str) or not record["reviewer_model"]
            or (generator_model and record["reviewer_model"].casefold() == generator_model.casefold())
            or type(record["semantic_score"]) is not int or not 1 <= record["semantic_score"] <= 10
            or type(record["aesthetic_score"]) is not int or not 1 <= record["aesthetic_score"] <= 10
            or not valid_issues
            or not isinstance(record["repair_prompt"], str)):
        raise SemanticImageReviewError("Frozen image review is invalid")
    objectively_passed = (record["semantic_score"] >= 7 and record["aesthetic_score"] >= 7
                          and not any(issue["severity"] == "critical" for issue in issues))
    if ((record["status"] == "passed") != objectively_passed
            or (not objectively_passed and not record["repair_prompt"].strip())):
        raise SemanticImageReviewError("Frozen image review verdict is contradictory")
    return record


def verify_semantic_image_review(record: dict[str, Any], *, image_bytes: bytes,
                                 diagram_request: str, generator_model: str | None) -> dict[str, Any]:
    """Verify that a frozen, input-bound review supports publication."""
    verify_semantic_image_review_record(
        record, image_bytes=image_bytes, diagram_request=diagram_request,
        generator_model=generator_model)
    if record["status"] != "passed":
        raise SemanticImageReviewError("Frozen image review does not support publication")
    return record
