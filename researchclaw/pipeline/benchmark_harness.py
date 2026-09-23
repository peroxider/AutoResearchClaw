"""Offline fault-injection harness for the fixed-budget benchmark.

``prepare_benchmark`` copies a reviewed template bundle once per declared
case, seeds the named defect into each defective copy, re-runs the real
final acceptance in every copy (defective and control alike), and writes
the ``benchmark_manifest.json`` that ``benchmark_report`` consumes.

The harness simulates a reviewer who approved the mutated bundle: the
copy's ``final_reviews.json`` is rebound to the injected inventory with the
template's original verdicts, so the mechanical acceptance gates — and
only they — get the chance to catch the seeded defect. Defects the gates
cannot see therefore surface as accepted defects in the report; that is
the measurement, not a harness failure. Controls are unmodified copies and
inherit the template's verdict.
"""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

from researchclaw.pipeline.benchmark_report import BENCHMARK_DIMENSIONS
from researchclaw.pipeline.evidence_store import content_hash, file_hash
from researchclaw.pipeline.final_acceptance import assess_delivery, inventory


class BenchmarkHarnessError(ValueError):
    pass


def _append(run_dir: Path, name: str, text: str) -> None:
    path = run_dir / name
    if not path.is_file():
        raise BenchmarkHarnessError(f"Defect seed expects {name} in the template")
    path.write_text(path.read_text(encoding="utf-8") + text, encoding="utf-8")


def _replace(run_dir: Path, name: str, old: str, new: str) -> None:
    path = run_dir / name
    if not path.is_file() or old not in path.read_text(encoding="utf-8"):
        raise BenchmarkHarnessError(f"Defect seed expects {old!r} in {name}")
    path.write_text(path.read_text(encoding="utf-8").replace(old, new),
                    encoding="utf-8")


def _seed_fabricated_citation(run_dir: Path) -> None:
    _append(run_dir, "paper.tex", " Ghost results~\\cite{ghost2026}.")
    _append(run_dir, "paper_final.md", " Ghost results [ghost2026].")


def _seed_missing_figure(run_dir: Path) -> None:
    _append(run_dir, "paper.tex", "\n\\includegraphics{ghost_figure}.")


def _seed_unresolved_placeholder(run_dir: Path) -> None:
    _append(run_dir, "paper_final.md", " TODO fix this section.")


def _seed_stale_manuscript(run_dir: Path) -> None:
    # Rewriting the reported value invalidates the frozen numeric claim
    # spans and manuscript bindings without touching the evidence store.
    for name in ("paper.tex", "paper_final.md"):
        _replace(run_dir, name, "81.00", "82.00")


def _seed_pipeline_blocker(run_dir: Path) -> None:
    (run_dir / "pipeline_blockers.json").write_text(
        json.dumps({"issues": [{"artifact": "seed", "reason": "seeded_blocker"}]}),
        encoding="utf-8")


def _seed_corrupt_call_ledger(run_dir: Path) -> None:
    (run_dir / "llm_call_ledger.json").write_text('{"calls": "not-a-list"}',
                                                  encoding="utf-8")


def _seed_unbound_claim(run_dir: Path) -> None:
    # A fabricated claim in a file no acceptance check reads: an explicit
    # acceptance blind-spot probe for the benchmark.
    (run_dir / "response_to_reviewers.txt").write_text(
        "Reviewers independently confirm the effect is 99.00.\n",
        encoding="utf-8")


DEFECT_OPERATORS = {
    "fabricated_citation": _seed_fabricated_citation,
    "missing_figure": _seed_missing_figure,
    "unresolved_placeholder": _seed_unresolved_placeholder,
    "stale_manuscript": _seed_stale_manuscript,
    "pipeline_blocker": _seed_pipeline_blocker,
    "corrupt_call_ledger": _seed_corrupt_call_ledger,
    "unbound_claim": _seed_unbound_claim,
}


# case_id becomes a path component under runs/: a safe charset rejects
# traversal, whitespace-only, and over-long ids before any copy happens.
_SAFE_CASE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}")


def _validate_cases(cases: Any) -> list[dict]:
    if not isinstance(cases, list) or not cases:
        raise BenchmarkHarnessError("Benchmark harness cases are malformed")
    seen: set[str] = set()
    for case in cases:
        if (not isinstance(case, dict) or not isinstance(case.get("case_id"), str)
                or not _SAFE_CASE_ID.fullmatch(case["case_id"])
                or case["case_id"] in seen
                or case.get("kind") not in ("defect", "control")):
            raise BenchmarkHarnessError("Benchmark harness cases are malformed")
        seen.add(case["case_id"])
        if case["kind"] == "defect":
            if (case.get("defect_category") not in DEFECT_OPERATORS
                    or case.get("expected_dimension") not in BENCHMARK_DIMENSIONS):
                raise BenchmarkHarnessError("Benchmark harness cases are malformed")
        elif "defect_category" in case or "expected_dimension" in case:
            raise BenchmarkHarnessError("Benchmark harness cases are malformed")
    return cases


def _rebind_reviews(run_dir: Path) -> None:
    """Carry the template's review verdicts onto the mutated inventory.

    This models a reviewer approving the defective bundle; the mechanical
    gates remain the only chance to catch the seeded defect.
    """
    path = run_dir / "final_reviews.json"
    reviews = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(reviews, dict):
        raise BenchmarkHarnessError("Template final_reviews.json is malformed")
    reviews["input_version"] = content_hash(inventory(run_dir))
    path.write_text(json.dumps(reviews, indent=2), encoding="utf-8")


def prepare_benchmark(template: Path, out_root: Path, cases: list[dict]) -> dict:
    """Prepare defective/control runs from a reviewed template and freeze
    the benchmark manifest the report layer consumes."""
    _validate_cases(cases)
    template = Path(template)
    out_root = Path(out_root)
    if not template.is_dir():
        raise BenchmarkHarnessError("Benchmark template is missing")
    if not (template / "final_reviews.json").is_file():
        raise BenchmarkHarnessError(
            "Benchmark template must carry final_reviews.json")
    resolved_out = out_root.resolve()
    resolved_template = template.resolve()
    if resolved_out == resolved_template or resolved_template in resolved_out.parents:
        raise BenchmarkHarnessError(
            "Benchmark output root must not live inside the template")
    for case in cases:
        if (out_root / "runs" / case["case_id"]).exists():
            raise BenchmarkHarnessError(
                f"Benchmark case directory already exists: {case['case_id']}")

    manifest_cases: list[dict] = []
    for case in cases:
        run_dir = out_root / "runs" / case["case_id"]
        shutil.copytree(template, run_dir)
        if case["kind"] == "defect":
            DEFECT_OPERATORS[case["defect_category"]](run_dir)
            _rebind_reviews(run_dir)
        report = assess_delivery(run_dir, target_status="research_complete")
        (run_dir / "final_acceptance.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8")
        entry: dict[str, Any] = {"case_id": case["case_id"], "kind": case["kind"],
                                 "run_dir": f"runs/{case['case_id']}",
                                 "acceptance_sha256": file_hash(
                                     run_dir / "final_acceptance.json")}
        if case["kind"] == "defect":
            entry["defect_category"] = case["defect_category"]
            entry["expected_dimension"] = case["expected_dimension"]
        manifest_cases.append(entry)

    manifest = {"schema_version": 1, "cases": manifest_cases}
    (out_root / "benchmark_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
