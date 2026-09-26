"""Compare two independently executed formal experiment bundles."""
from __future__ import annotations

import argparse
import json
import math
import re
import uuid
from dataclasses import asdict
from pathlib import Path

from researchclaw.experiment.protocol_runner import verify_execution_bundle
from researchclaw.pipeline.evidence_store import EvidenceStore, content_hash, file_hash
from researchclaw.pipeline.experiment_protocol import (
    ProtocolError, audit_coverage, get_reproduction_policy, load_protocol,
)
from researchclaw.pipeline.independent_evaluator import evaluate_manifest
from researchclaw.research_inputs import verify_bundle_contract


_CHECKER = "independent-reproduction/v1"
_SCOPE = (
    "exact metric reproduction across two independently recorded formal bundles; "
    "site labels are self-declared and do not authenticate physical hosts"
)
_LABEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}")


class ReproductionComparisonError(ValueError):
    pass


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _records_by_key(store: EvidenceStore) -> dict[str, tuple[str, object]]:
    return {content_hash(asdict(record.key)): (result_id, record)
            for result_id, record in store.records.items()}


def compare_reproduction_values(policy: dict, metric: str,
                                left: float, right: float) -> tuple[bool, float | None, float]:
    """Apply an exact or execution-predeclared symmetric numeric tolerance."""
    difference = abs(left - right)
    if policy["mode"] == "exact":
        allowed = 0.0
    else:
        rule = policy["metrics"][metric]
        allowed = (rule["absolute_tolerance"]
                   + rule["relative_tolerance"] * max(abs(left), abs(right)))
    if not math.isfinite(difference):
        return False, None, allowed
    return difference <= allowed, difference, allowed


def _load_bundle(root: Path) -> dict:
    root = Path(root).resolve()
    protocol = load_protocol(root)
    if protocol is None:
        raise ReproductionComparisonError("A frozen experiment protocol is required")
    try:
        contract = verify_bundle_contract(root)
        verify_execution_bundle(root, protocol)
        code = json.loads((root / "protocol_code.json").read_text(encoding="utf-8"))
        budget = json.loads((root / "protocol_budget.json").read_text(encoding="utf-8"))
        manifest = json.loads((root / "trusted_evaluation.json").read_text(encoding="utf-8"))
        frozen = EvidenceStore.from_dict(
            json.loads((root / "evidence_store.json").read_text(encoding="utf-8")))
        recalculated = EvidenceStore.from_dict(evaluate_manifest(root, manifest))
        frozen_by_key = _records_by_key(frozen)
        recalculated_by_key = _records_by_key(recalculated)
        if set(frozen_by_key) != set(recalculated_by_key):
            raise ReproductionComparisonError("Frozen and recalculated evidence keys differ")
        for key_hash, (_, record) in frozen_by_key.items():
            _, fresh = recalculated_by_key[key_hash]
            frozen_fields = asdict(record)
            fresh_fields = asdict(fresh)
            # The evaluator host's Python version is intentionally part of
            # EvidenceRecord.environment, so portable replay may differ only
            # there. Every scientific and artifact field must still agree.
            frozen_fields.pop("environment")
            fresh_fields.pop("environment")
            if frozen_fields != fresh_fields:
                raise ReproductionComparisonError("Frozen metric differs from independent recalculation")
            issues = frozen.validate_record(frozen_by_key[key_hash][0], root)
            if issues or record.evaluator_independent is not True:
                raise ReproductionComparisonError("Frozen evidence is invalid: " + ",".join(issues))
        coverage = audit_coverage(root, protocol, frozen)
        if coverage["status"] != "complete":
            raise ReproductionComparisonError("Experiment coverage is incomplete")
        frozen_coverage = json.loads(
            (root / "experiment_coverage.json").read_text(encoding="utf-8"))
        if frozen_coverage != coverage:
            raise ReproductionComparisonError("Frozen coverage differs from portable recomputation")
    except ReproductionComparisonError:
        raise
    except (OSError, TypeError, ValueError, KeyError, ProtocolError) as exc:
        raise ReproductionComparisonError(
            f"Invalid reproduction bundle: {type(exc).__name__}: {exc}") from exc
    portable_contract = {
        "brief": contract["brief"],
        "runtime": contract["runtime"],
        "datasets": contract["datasets"],
        "outputs": contract["outputs"],
        "source_content": sorted(
            (Path(name).name, digest) for name, digest in contract["source_files"].items()),
    }
    return {
        "root": root,
        "protocol": protocol,
        "contract_version": contract["version"],
        "contract_identity": content_hash(portable_contract),
        "code_sha256": code["code_sha256"],
        "dependency_sha256": content_hash(code["dependency_manifest"]),
        "ledger_sha256": budget["ledger_sha256"],
        "records": frozen_by_key,
    }


def _build_report(left: dict, right: dict, left_site: str, right_site: str) -> dict:
    shared = sorted(set(left["records"]) & set(right["records"]))
    missing_left = sorted(set(right["records"]) - set(left["records"]))
    missing_right = sorted(set(left["records"]) - set(right["records"]))
    identities_match = (
        left["protocol"]["version"] == right["protocol"]["version"]
        and left["contract_identity"] == right["contract_identity"]
        and left["code_sha256"] == right["code_sha256"]
    )
    same_protocol = left["protocol"]["version"] == right["protocol"]["version"]
    policy = (get_reproduction_policy(left["protocol"])
              if same_protocol else {"mode": "exact", "metrics": {}})
    pairs = []
    mismatches = []
    for key_hash in shared:
        left_id, left_record = left["records"][key_hash]
        right_id, right_record = right["records"][key_hash]
        units_match = left_record.unit == right_record.unit
        value_match, difference, allowed = compare_reproduction_values(
            policy, left_record.key.metric, left_record.value, right_record.value)
        matched_value = units_match and value_match
        row = {
            "key": asdict(left_record.key),
            "left_result_id": left_id,
            "right_result_id": right_id,
            "left_value": left_record.value,
            "right_value": right_record.value,
            "unit": left_record.unit if left_record.unit == right_record.unit else None,
            "absolute_difference": difference,
            "allowed_difference": allowed,
            "value_match": matched_value,
        }
        pairs.append(row)
        if not matched_value:
            mismatches.append(key_hash)
    left_receipts = {record.execution for _, record in left["records"].values()}
    right_receipts = {record.execution for _, record in right["records"].values()}
    execution_evidence_distinct = (
        left["ledger_sha256"] != right["ledger_sha256"]
        and left_receipts.isdisjoint(right_receipts)
    )
    matched = (identities_match and execution_evidence_distinct and not missing_left
               and not missing_right and not mismatches and bool(pairs))
    report = {
        "schema_version": 1,
        "checker": _CHECKER,
        "status": "matched" if matched else "mismatched",
        "scope": _SCOPE,
        "protocol_version": (left["protocol"]["version"]
                             if same_protocol else None),
        "reproduction_policy": policy if same_protocol else None,
        "identities_match": identities_match,
        "execution_evidence_distinct": execution_evidence_distinct,
        "environment_relation": (
            "same" if left["dependency_sha256"] == right["dependency_sha256"] else "different"),
        "sites": [
            {"label": left_site, "label_provenance": "self_declared",
             "contract_version": left["contract_version"],
             "contract_identity": left["contract_identity"],
             "code_sha256": left["code_sha256"],
             "dependency_sha256": left["dependency_sha256"],
             "ledger_sha256": left["ledger_sha256"]},
            {"label": right_site, "label_provenance": "self_declared",
             "contract_version": right["contract_version"],
             "contract_identity": right["contract_identity"],
             "code_sha256": right["code_sha256"],
             "dependency_sha256": right["dependency_sha256"],
             "ledger_sha256": right["ledger_sha256"]},
        ],
        "record_count": len(pairs),
        "pairs": pairs,
        "missing_left": missing_left,
        "missing_right": missing_right,
        "mismatches": mismatches,
        "limitations": [
            "Exact equality is required unless a bounded tolerance and rationale were frozen before execution.",
            "Distinct hash-bound receipts reject a copied bundle but are not host authentication.",
            "This comparison does not establish scientific validity or generalization.",
        ],
    }
    report["version"] = content_hash(report)
    return report


def compare_reproduction_bundles(left_root: Path, right_root: Path, *,
                                 left_site: str, right_site: str,
                                 output: Path | None = None) -> dict:
    """Verify and compare two complete bundles, optionally freezing the report."""
    if (not isinstance(left_site, str) or not isinstance(right_site, str)
            or _LABEL.fullmatch(left_site) is None or _LABEL.fullmatch(right_site) is None
            or left_site == right_site):
        raise ReproductionComparisonError("Site labels must be distinct safe identifiers")
    left_path, right_path = Path(left_root).resolve(), Path(right_root).resolve()
    if left_path == right_path:
        raise ReproductionComparisonError("Reproduction roots must be distinct")
    report = _build_report(
        _load_bundle(left_path), _load_bundle(right_path), left_site, right_site)
    if output is not None:
        _write(Path(output).resolve(), report)
    return report


def verify_reproduction_report(report: dict, left_root: Path, right_root: Path) -> dict:
    """Recompute a report from its bound bundles and reject any changed field."""
    if not isinstance(report, dict) or report.get("checker") != _CHECKER:
        raise ReproductionComparisonError("Malformed reproduction report")
    if report.get("scope") != _SCOPE or report.get("schema_version") != 1:
        raise ReproductionComparisonError("Invalid reproduction report identity")
    sites = report.get("sites")
    if (not isinstance(sites, list) or len(sites) != 2
            or any(not isinstance(site, dict) for site in sites)):
        raise ReproductionComparisonError("Invalid reproduction sites")
    expected = compare_reproduction_bundles(
        left_root, right_root, left_site=sites[0].get("label", ""),
        right_site=sites[1].get("label", ""))
    if report != expected:
        raise ReproductionComparisonError("Reproduction report differs from verified bundles")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare two formal reproduction bundles")
    parser.add_argument("left_root", type=Path)
    parser.add_argument("right_root", type=Path)
    parser.add_argument("--left-site", required=True)
    parser.add_argument("--right-site", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = compare_reproduction_bundles(
        args.left_root, args.right_root, left_site=args.left_site,
        right_site=args.right_site, output=args.output)
    print(json.dumps({"status": report["status"], "record_count": report["record_count"]},
                     sort_keys=True))
    return 0 if report["status"] == "matched" else 1


if __name__ == "__main__":
    raise SystemExit(main())
