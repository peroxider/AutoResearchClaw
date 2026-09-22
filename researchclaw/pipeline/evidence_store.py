"""Versioned, typed numerical evidence. A global number match is not evidence.

Records are immutable and addressed by their content. Legacy summaries may
still feed exploratory writing, but cannot silently acquire verified status.
All file references are relative to the evidence bundle's root.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


def content_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class EvidenceKey:
    dataset: str
    dataset_version: str
    split: str
    method: str
    config: str
    seed: str
    metric: str
    aggregation: str
    regime: str = "default"

    def __post_init__(self) -> None:
        if any(not isinstance(v, str) or not v.strip() for v in asdict(self).values()):
            raise ValueError("Every evidence identity field must be a nonempty string")


@dataclass(frozen=True)
class EvidenceRecord:
    key: EvidenceKey
    value: float
    unit: str  # fraction | percent | percentage_point | raw
    run_status: str
    code_commit: str
    environment: str
    execution: str
    evaluator: str
    evaluator_independent: bool
    artifacts: tuple[tuple[str, str], ...]  # (relative output/prediction path, sha256)

    def __post_init__(self) -> None:
        if type(self.value) not in (int, float) or not math.isfinite(self.value):
            raise ValueError("Evidence value must be finite")
        if self.unit not in {"fraction", "percent", "percentage_point", "raw"}:
            raise ValueError("Unknown evidence unit")
        if self.unit == "fraction" and not 0 <= self.value <= 1:
            raise ValueError("Fraction must be in [0, 1]")
        if self.unit == "percent" and not 0 <= self.value <= 100:
            raise ValueError("Percent must be in [0, 100]")
        if self.run_status not in {"success", "failed", "unavailable"}:
            raise ValueError("Invalid run status")
        if type(self.evaluator_independent) is not bool:
            raise ValueError("evaluator_independent must be boolean")
        if any(not isinstance(value, str) for value in (
            self.code_commit, self.environment, self.execution, self.evaluator,
        )):
            raise ValueError("Provenance fields must be strings")
        if any(not isinstance(name, str) or not isinstance(digest, str)
               or re.fullmatch(r"[0-9a-f]{64}", digest) is None for name, digest in self.artifacts):
            raise ValueError("Artifacts require relative paths and SHA-256 hashes")

    @property
    def result_id(self) -> str:
        return content_hash(asdict(self))


class EvidenceStore:
    schema_version = 1

    def __init__(self) -> None:
        self.records: dict[str, EvidenceRecord] = {}
        self._keys: dict[EvidenceKey, str] = {}

    def add(self, record: EvidenceRecord) -> str:
        result_id = record.result_id
        existing = self._keys.get(record.key)
        if existing is not None and existing != result_id:
            raise ValueError("Conflicting evidence for the same complete experiment key")
        self.records[result_id] = record
        self._keys[record.key] = result_id
        return result_id

    @property
    def version(self) -> str:
        return content_hash(sorted(self.records))

    def validate_record(self, result_id: str, root: Path) -> list[str]:
        record = self.records.get(result_id)
        if record is None:
            return ["unknown_result"]
        issues = []
        if record.run_status != "success":
            issues.append("run_not_successful")
        if record.evaluator_independent is not True:
            issues.append("independent_evaluation_missing")
        if not all((record.code_commit, record.environment, record.execution, record.evaluator)):
            issues.append("provenance_missing")
        if not record.artifacts:
            issues.append("raw_output_missing")
        for name, digest in record.artifacts:
            path = (root / name).resolve()
            if not path.is_relative_to(root.resolve()) or not path.is_file():
                issues.append(f"artifact_missing:{name}")
            elif file_hash(path) != digest:
                issues.append(f"artifact_changed:{name}")
        return issues

    def verify_claim(self, claim: dict[str, Any], root: Path) -> list[str]:
        """Require the full identity and explicit units for every numerical claim."""
        result_id = claim.get("result_id", "")
        issues = self.validate_record(result_id, root)
        if issues:
            return issues
        record = self.records[result_id]
        if claim.get("key") != asdict(record.key):
            issues.append("identity_mismatch")
        unit = claim.get("unit")
        expected = record.value
        if unit != record.unit:
            if (record.unit, unit) == ("fraction", "percent"):
                expected *= 100
            elif (record.unit, unit) == ("percent", "fraction"):
                expected /= 100
            else:
                issues.append("unit_mismatch")
        value = claim.get("value")
        decimals = claim.get("decimals", 6)
        if type(decimals) is not int or not 0 <= decimals <= 12:
            issues.append("invalid_precision")
        elif (type(value) not in (int, float) or not math.isfinite(value)
              or abs(value - round(expected, decimals)) > 1e-12):
            issues.append("value_mismatch")
        return issues

    def derive(self, left: str, right: str, operation: str, root: Path) -> dict[str, Any]:
        """Calculate signed differences/relative gains, retaining the source IDs."""
        for result_id in (left, right):
            issues = self.validate_record(result_id, root)
            if issues:
                raise ValueError(", ".join(issues))
        a, b = self.records[left], self.records[right]
        ak, bk = asdict(a.key), asdict(b.key)
        for field in ("method", "config"):
            ak.pop(field)
            bk.pop(field)
        if ak != bk or a.unit != b.unit:
            raise ValueError("Cannot compare different datasets, versions, splits, metrics or units")
        if operation == "difference":
            value, unit = a.value - b.value, "raw"
        elif operation == "percentage_point" and a.unit in {"fraction", "percent"}:
            value = (a.value - b.value) * (100 if a.unit == "fraction" else 1)
            unit = "percentage_point"
        elif operation == "relative_change" and b.value != 0:
            value, unit = (a.value - b.value) / abs(b.value) * 100, "percent"
        else:
            raise ValueError("Invalid derivation or zero denominator")
        return {"sources": [left, right], "operation": operation, "value": value, "unit": unit}

    def render(self, result_id: str, root: Path, decimals: int = 3) -> str:
        if type(decimals) is not int or not 0 <= decimals <= 12:
            raise ValueError("Precision must be an integer in [0, 12]")
        issues = self.validate_record(result_id, root)
        if issues:
            raise ValueError(", ".join(issues))
        return f"{self.records[result_id].value:.{decimals}f}"

    def render_table(self, result_ids: list[str], root: Path, *, latex: bool = False) -> str:
        """Render identities and values together; the model cannot swap labels."""
        headers = ["Dataset", "Version", "Split", "Method", "Regime", "Config",
                   "Seed", "Metric", "Aggregation", "Value", "Unit"]
        rows = []
        for result_id in result_ids:
            value = self.render(result_id, root)
            record = self.records[result_id]
            key = record.key
            rows.append([key.dataset, key.dataset_version, key.split, key.method, key.regime,
                         key.config, key.seed, key.metric, key.aggregation, value, record.unit])
        if latex:
            escapes = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
                       "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
                       "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
            def escape(cell: str) -> str:
                return "".join(escapes.get(char, char) for char in cell.replace("\n", " "))
            lines = [" & ".join(escape(cell) for cell in row) + r" \\" for row in [headers, *rows]]
            return "\\begin{tabular}{" + "l" * len(headers) + "}\n" + "\n".join(lines) + "\n\\end{tabular}\n"
        def md_row(row: list[str]) -> str:
            return "| " + " | ".join(cell.replace("|", r"\|").replace("\n", " ") for cell in row) + " |"
        return "\n".join([md_row(headers), md_row(["---"] * len(headers)), *(md_row(r) for r in rows)]) + "\n"

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": self.schema_version, "version": self.version,
                "records": [{"result_id": key, **asdict(record)}
                            for key, record in sorted(self.records.items())]}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvidenceStore:
        if data.get("schema_version") != cls.schema_version:
            raise ValueError("Unsupported evidence schema")
        store = cls()
        for item in data["records"]:
            item = dict(item)
            expected_id = item.pop("result_id")
            item["key"] = EvidenceKey(**item["key"])
            item["artifacts"] = tuple(tuple(a) for a in item["artifacts"])
            if store.add(EvidenceRecord(**item)) != expected_id:
                raise ValueError("Evidence content hash mismatch")
        if store.version != data.get("version"):
            raise ValueError("Evidence version mismatch")
        return store
