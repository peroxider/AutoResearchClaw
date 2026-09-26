"""Fixed, auditable benchmark runner for an independent vision reviewer."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path, PurePosixPath
from typing import Any

from researchclaw.agents.figure_agent.semantic_image_review import (
    SemanticImageReviewError, review_image_semantics, verify_semantic_image_review_record,
)
from researchclaw.pipeline.evidence_store import content_hash, file_hash


class VisualReviewBenchmarkError(ValueError):
    pass


_CASE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}")
_EXPECTED = {"passed", "failed"}


def _load_manifest(root: Path) -> tuple[dict, list[dict]]:
    path = root / "visual_review_benchmark_manifest.json"
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise VisualReviewBenchmarkError("Visual benchmark manifest is unavailable") from exc
    if (not isinstance(manifest, dict) or set(manifest) != {"schema_version", "cases"}
            or manifest["schema_version"] != 1 or not isinstance(manifest["cases"], list)
            or not 1 <= len(manifest["cases"]) <= 32):
        raise VisualReviewBenchmarkError("Visual benchmark manifest is malformed")
    seen = set()
    for case in manifest["cases"]:
        if (not isinstance(case, dict)
                or set(case) != {"case_id", "image", "diagram_request", "expected_status"}
                or not isinstance(case["case_id"], str) or not _CASE_ID.fullmatch(case["case_id"])
                or case["case_id"] in seen
                or not isinstance(case["diagram_request"], str)
                or not case["diagram_request"].strip()
                or len(case["diagram_request"]) > 20_000
                or case["expected_status"] not in _EXPECTED):
            raise VisualReviewBenchmarkError("Visual benchmark case is malformed")
        seen.add(case["case_id"])
        if not isinstance(case["image"], str):
            raise VisualReviewBenchmarkError("Visual benchmark image path is malformed")
        relative = PurePosixPath(case["image"])
        if (relative.is_absolute() or not relative.parts or "\\" in case["image"]
                or any(part in {".", ".."} for part in relative.parts)
                or relative.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}):
            raise VisualReviewBenchmarkError("Visual benchmark image path is unsafe")
        image = root / relative
        if (not image.is_file() or image.is_symlink()
                or not image.resolve().is_relative_to(root.resolve())
                or not 0 < image.stat().st_size <= 10_000_000):
            raise VisualReviewBenchmarkError("Visual benchmark image is unavailable")
        try:
            from PIL import Image
            with Image.open(image) as opened:
                expected = {".png": "PNG", ".jpg": "JPEG", ".jpeg": "JPEG", ".webp": "WEBP"}[
                    relative.suffix.lower()]
                if opened.format != expected or opened.width * opened.height > 50_000_000:
                    raise VisualReviewBenchmarkError("Visual benchmark image encoding is invalid")
                opened.verify()
        except (OSError, ValueError, Image.DecompressionBombError) as exc:
            raise VisualReviewBenchmarkError("Visual benchmark image is not decodable") from exc
    return manifest, manifest["cases"]


def _totals(results: list[dict], errors: list[dict]) -> dict:
    defects = [row for row in results if row["expected_status"] == "failed"]
    controls = [row for row in results if row["expected_status"] == "passed"]
    false_accepts = sum(row["observed_status"] == "passed" for row in defects)
    false_rejects = sum(row["observed_status"] == "failed" for row in controls)
    correct = sum(row["observed_status"] == row["expected_status"] for row in results)
    rate = lambda numerator, denominator: numerator / denominator if denominator else None
    return {
        "declared_cases": len(results) + len(errors), "evaluated_cases": len(results),
        "case_errors": len(errors), "defect_cases": len(defects),
        "false_accepts": false_accepts,
        "false_accept_rate": rate(false_accepts, len(defects)),
        "control_cases": len(controls), "false_rejects": false_rejects,
        "false_reject_rate": rate(false_rejects, len(controls)),
        "accuracy": rate(correct, len(results)),
    }


def run_visual_review_benchmark(root: Path, reviewer: Any) -> dict:
    """Run exactly one independent review call per fixed case and freeze results."""
    root = Path(root)
    manifest, cases = _load_manifest(root)
    results, errors = [], []
    for case in cases:
        image_path = root / case["image"]
        image_bytes = image_path.read_bytes()
        try:
            review = review_image_semantics(
                reviewer=reviewer, image_bytes=image_bytes,
                diagram_request=case["diagram_request"],
                generator_model="fixed-benchmark-fixture")
            results.append({
                "case_id": case["case_id"], "image": case["image"],
                "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
                "request_sha256": hashlib.sha256(
                    case["diagram_request"].encode("utf-8")).hexdigest(),
                "expected_status": case["expected_status"],
                "observed_status": review["status"],
                "review": review,
            })
        except (SemanticImageReviewError, OSError, ValueError, TypeError) as exc:
            errors.append({"case_id": case["case_id"], "error_type": type(exc).__name__})
    report = {
        "schema_version": 1,
        "manifest_sha256": file_hash(root / "visual_review_benchmark_manifest.json"),
        "cases": results, "case_errors": errors,
        "totals": _totals(results, errors),
        "scope": "finite fixed cases; not a guarantee of arbitrary diagram review quality",
    }
    report["version"] = content_hash(report)
    (root / "visual_review_benchmark_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def verify_visual_review_benchmark(root: Path, report: dict | None = None) -> dict:
    """Rebind a frozen report to its manifest and current sample bytes."""
    root = Path(root)
    _, cases = _load_manifest(root)
    if report is None:
        try:
            report = json.loads((root / "visual_review_benchmark_report.json").read_text("utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise VisualReviewBenchmarkError("Visual benchmark report is unavailable") from exc
    if (not isinstance(report, dict) or set(report) != {
            "schema_version", "manifest_sha256", "cases", "case_errors", "totals",
            "scope", "version"} or report.get("schema_version") != 1):
        raise VisualReviewBenchmarkError("Visual benchmark report is malformed")
    payload = dict(report)
    version = payload.pop("version", None)
    if version != content_hash(payload):
        raise VisualReviewBenchmarkError("Visual benchmark report changed")
    if report.get("manifest_sha256") != file_hash(root / "visual_review_benchmark_manifest.json"):
        raise VisualReviewBenchmarkError("Visual benchmark manifest changed")
    by_id = {case["case_id"]: case for case in cases}
    results, errors = report.get("cases"), report.get("case_errors")
    if not isinstance(results, list) or not isinstance(errors, list):
        raise VisualReviewBenchmarkError("Visual benchmark result lists are malformed")
    ids = [row.get("case_id") for row in results + errors if isinstance(row, dict)]
    if len(ids) != len(cases) or set(ids) != set(by_id) or len(ids) != len(set(ids)):
        raise VisualReviewBenchmarkError("Visual benchmark cases are incomplete")
    for row in results:
        case = by_id.get(row.get("case_id")) if isinstance(row, dict) else None
        if (case is None or set(row) != {"case_id", "image", "image_sha256",
                                        "request_sha256", "expected_status",
                                        "observed_status", "review"}
                or row.get("image") != case["image"]
                or row.get("expected_status") != case["expected_status"]
                or row.get("observed_status") not in _EXPECTED
                or row.get("image_sha256") != file_hash(root / case["image"])
                or row.get("request_sha256") != hashlib.sha256(
                    case["diagram_request"].encode("utf-8")).hexdigest()):
            raise VisualReviewBenchmarkError("Visual benchmark case binding is invalid")
        try:
            review = verify_semantic_image_review_record(
                row.get("review"), image_bytes=(root / case["image"]).read_bytes(),
                diagram_request=case["diagram_request"],
                generator_model="fixed-benchmark-fixture")
        except (SemanticImageReviewError, OSError, TypeError, ValueError) as exc:
            raise VisualReviewBenchmarkError("Visual benchmark review is invalid") from exc
        if row["observed_status"] != review["status"]:
            raise VisualReviewBenchmarkError("Visual benchmark observed status differs")
    for row in errors:
        if (not isinstance(row, dict) or set(row) != {"case_id", "error_type"}
                or not isinstance(row["error_type"], str) or not row["error_type"]):
            raise VisualReviewBenchmarkError("Visual benchmark error entry is invalid")
    if report.get("totals") != _totals(results, errors):
        raise VisualReviewBenchmarkError("Visual benchmark totals differ")
    return report


def main(argv: list[str] | None = None) -> int:
    """Run the fixed suite with the review model declared by an RC config."""
    import argparse
    parser = argparse.ArgumentParser(description="Run a fixed visual-review benchmark")
    parser.add_argument("benchmark_root", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    from researchclaw.config import RCConfig
    from researchclaw.llm import build_reviewer_llm
    from researchclaw.llm.call_ledger import reset_call_ledger, write_call_ledger
    config = RCConfig.load(args.config, check_paths=False)
    reviewer = build_reviewer_llm(config)
    if reviewer is None:
        raise VisualReviewBenchmarkError(
            "Config must provide an OpenAI-compatible independent reviewer_model")
    reset_call_ledger()
    report = run_visual_review_benchmark(args.benchmark_root, reviewer)
    write_call_ledger(args.benchmark_root)
    verify_visual_review_benchmark(args.benchmark_root, report)
    print(json.dumps(report["totals"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
