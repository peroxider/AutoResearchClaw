"""Structured semantic review of the MethodSpec against the frozen source.

A named reviewer inspects a recomputable evidence packet — the spec, the
archived code of every mapped step, and the runtime validation summary — and
records per-step verdicts. Quotes must exist verbatim in the frozen artifacts,
and the conclusion is recomputed from the recorded verdicts. An accepted review
upgrades the reported status only to reviewed_informal; it is never a machine
proof, and the frozen MethodSpec keeps semantic_equivalence unresolved.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

from researchclaw.pipeline.evidence_store import content_hash, file_hash
from researchclaw.pipeline.experiment_protocol import ProtocolError, _object, _text
from researchclaw.pipeline.research_workbench import WorkbenchError, check_implementation, compile_method

VERDICTS = ("consistent", "inconsistent", "unverifiable")
EQUIVALENCE = {"consistent": "reviewed_informal", "unverifiable": "unresolved", "inconsistent": "contradicted"}
PACKET_NAME = "method_semantic_review_packet.json"
REVIEW_NAME = "method_semantic_review.json"


def _spec_contains(spec, quote: str) -> bool:
    """A quote must occur in the serialized spec or inside one of its strings."""
    if not isinstance(quote, str) or not quote.strip():
        return False
    if quote in json.dumps(spec, ensure_ascii=False, sort_keys=True):
        return True
    stack = [spec]
    while stack:
        item = stack.pop()
        if isinstance(item, str) and quote in item:
            return True
        if isinstance(item, dict):
            stack.extend(item.values())
        elif isinstance(item, list):
            stack.extend(item)
    return False


def _symbol_source(path: Path, qualified: str) -> str:
    try:
        text = path.read_text(encoding="utf-8")
        node = None
        scope = ast.parse(text).body
        for part in qualified.split("."):
            node = next((n for n in scope if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                         and n.name == part), None)
            if node is None:
                return ""
            scope = node.body
        return "\n".join(text.splitlines()[node.lineno - 1: node.end_lineno])
    except (OSError, SyntaxError, UnicodeError):
        return ""


def build_packet(root, method, code) -> dict:
    """Recompute the reviewer's evidence packet from frozen artifacts only."""
    root = Path(root)
    if not isinstance(code, dict) or not isinstance(code.get("files"), dict) or not code["files"] \
            or content_hash(code["files"]) != code.get("code_sha256"):
        raise WorkbenchError("Invalid frozen source identity")
    source_root = (root / "evidence_artifacts" / "protocol_source").resolve()
    for name, digest in code["files"].items():
        path = (source_root / name).resolve()
        if not path.is_relative_to(source_root) or file_hash(path) != digest:
            raise WorkbenchError("Method semantic review source archive changed")
    implementation = check_implementation(method, source_root)
    sources = {}
    for binding in implementation["bindings"]:
        if binding["step"] in sources:
            continue
        sources[binding["step"]] = {"step": binding["step"], "file": binding["file"], "symbol": binding["symbol"],
                                    "line": binding["line"], "sha256": binding["sha256"],
                                    "source": _symbol_source(source_root / binding["file"], binding["symbol"])}
    validation = None
    report_path = root / "method_validation.json"
    if report_path.is_file():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            validation = {"status": report.get("status"), "coverage": report.get("coverage"),
                          "checks": report.get("checks"), "report_sha256": file_hash(report_path)}
        except (OSError, ValueError):
            validation = {"status": "unreadable", "report_sha256": file_hash(report_path)}
    packet = {"schema_version": 1, "method_version": method["version"], "code_sha256": code["code_sha256"],
              "spec": method["spec"],
              "steps": [sources[step["id"]] for step in method["spec"]["steps"] if step["id"] in sources],
              "mapping_issues": implementation["issues"], "validation": validation}
    packet["packet_version"] = content_hash(packet)
    return packet


def write_packet(root, method, code) -> dict:
    """Persist the packet for an isolated reviewer; verification recomputes it."""
    packet = build_packet(root, method, code)
    (Path(root) / PACKET_NAME).write_text(json.dumps(packet, indent=2, ensure_ascii=False), encoding="utf-8")
    return packet


def compile_review(review, method, packet) -> dict:
    """Validate one review against the recomputed packet; no verdict is trusted."""
    try:
        fields = {"schema_version", "method_version", "code_sha256", "packet_version",
                  "reviewer", "steps", "equations", "conclusion"}
        # The persisted form carries the derived fields; they are re-verified below.
        _object(review, fields | {"semantic_equivalence", "version"}, fields, "method semantic review")
        if type(review["schema_version"]) is not int or review["schema_version"] != 1:
            raise WorkbenchError("Unsupported method semantic review schema")
        if (review["method_version"] != packet["method_version"] or review["code_sha256"] != packet["code_sha256"]
                or review["packet_version"] != packet["packet_version"]):
            raise WorkbenchError("Method semantic review is bound to a different frozen version")
        reviewer = _object(review["reviewer"], {"checker", "evidence"}, {"checker", "evidence"}, "semantic reviewer")
        _text(reviewer["checker"], "semantic reviewer identity")
        _text(reviewer["evidence"], "review context evidence")
        sources = {step_item["step"]: step_item["source"] for step_item in packet.get("steps", [])}

        def check_items(items, key_name, targets, require_code):
            fields = ({key_name, "verdict", "spec_evidence", "code_evidence", "notes"} if require_code
                      else {key_name, "verdict", "spec_evidence", "notes"})
            seen = {}
            for item in items:
                _object(item, fields, fields, f"semantic {key_name} review")
                target = item[key_name]
                if not isinstance(target, str) or target in seen or target not in targets:
                    raise WorkbenchError(f"Duplicate or unknown {key_name} review")
                if item["verdict"] not in VERDICTS:
                    raise WorkbenchError("Invalid semantic review verdict")
                if not _spec_contains(packet["spec"], item["spec_evidence"]):
                    raise WorkbenchError(f"Spec evidence quote does not exist in the frozen MethodSpec ({key_name} {target})")
                if item["verdict"] != "consistent" and not (isinstance(item["notes"], str) and item["notes"].strip()):
                    raise WorkbenchError(f"Non-consistent verdicts require notes ({key_name} {target})")
                if require_code:
                    archived = sources.get(target)
                    if not archived:
                        # An empty segment means the mapping or source extraction
                        # failed; no reviewer verdict can bind to missing code.
                        raise WorkbenchError(f"Step has no archived source; fix the code mapping before review ({key_name} {target})")
                    if not isinstance(item["code_evidence"], str) or not item["code_evidence"].strip() \
                            or item["code_evidence"] not in archived:
                        raise WorkbenchError(f"Code evidence quote does not exist in the archived source ({key_name} {target})")
                seen[target] = item["verdict"]
            missing = sorted(set(targets) - set(seen))
            if missing:
                raise WorkbenchError(f"Method semantic review misses {key_name} entries: " + ", ".join(missing))
            return list(seen.values())

        verdicts = check_items(review["steps"], "step", tuple(step["id"] for step in method["spec"]["steps"]), True)
        equation_verdicts = check_items(review["equations"], "equation",
                                        tuple(equation["id"] for equation in method["spec"]["equations"]), False)
        all_verdicts = verdicts + equation_verdicts
        derived = ("inconsistent" if "inconsistent" in all_verdicts
                   else "unverifiable" if "unverifiable" in all_verdicts else "consistent")
        conclusion = _object(review["conclusion"], {"verdict", "evidence"}, {"verdict", "evidence"}, "semantic conclusion")
        if conclusion["verdict"] != derived:
            raise WorkbenchError("Semantic conclusion does not follow from the recorded verdicts")
        if not (isinstance(conclusion["evidence"], str) and conclusion["evidence"].strip()):
            raise WorkbenchError("Semantic conclusion requires evidence")
        # Recompute the derived fields over the canonical key set, so the stored
        # hash cannot be folded into its own recomputation.
        document = {key: value for key, value in review.items() if key not in {"semantic_equivalence", "version"}}
        document["semantic_equivalence"] = EQUIVALENCE[derived]
        document["version"] = content_hash(document)
        if "semantic_equivalence" in review and review["semantic_equivalence"] != document["semantic_equivalence"]:
            raise WorkbenchError("Recorded semantic_equivalence does not follow from the recorded verdicts")
        if "version" in review and review["version"] != document["version"]:
            raise WorkbenchError("Recorded method semantic review changed")
        return document
    except (TypeError, KeyError, ProtocolError) as exc:
        raise WorkbenchError(f"Invalid method semantic review: {exc}") from exc


def verify_semantic_review(root, method, code) -> dict:
    """Recompute the packet and recheck every verdict binding; fail closed."""
    root = Path(root)
    try:
        if compile_method(method.get("spec")) != method:
            raise WorkbenchError("Frozen MethodSpec changed")
    except (TypeError, KeyError, ProtocolError) as exc:
        raise WorkbenchError(f"Invalid MethodSpec: {exc}") from exc
    packet = build_packet(root, method, code)
    packet_path = root / PACKET_NAME
    if packet_path.is_file():
        try:
            recorded = json.loads(packet_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            recorded = None
        if recorded != packet:
            raise WorkbenchError("Recorded semantic review packet differs from the recomputed packet")
    try:
        review = json.loads((root / REVIEW_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise WorkbenchError("Missing or unreadable method semantic review") from exc
    return compile_review(review, method, packet)


def reported_equivalence(root, method) -> dict:
    """Display status: frozen spec stays unresolved; a verified review upgrades it."""
    code_path = Path(root) / "protocol_code.json"
    try:
        code = json.loads(code_path.read_text(encoding="utf-8")) if code_path.is_file() else None
        review = verify_semantic_review(root, method, code)
    except (OSError, ValueError, TypeError, KeyError):
        return {"status": method["semantic_equivalence"], "checker": None}
    return {"status": review["semantic_equivalence"], "checker": review["reviewer"]["checker"]}


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(
        description="Prepare or verify the evidence packet for the method semantic review")
    parser.add_argument("directory", type=Path, help="run or delivery root containing method_spec.json and protocol_code.json")
    parser.add_argument("--write-packet", action="store_true",
                        help=f"write {PACKET_NAME} for an isolated reviewer")
    args = parser.parse_args()
    if not args.directory.is_dir():
        parser.error("directory does not exist")
    try:
        method = json.loads((args.directory / "method_spec.json").read_text(encoding="utf-8"))
        code = json.loads((args.directory / "protocol_code.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        parser.error(f"cannot read frozen method inputs: {exc}")
    if args.write_packet:
        packet = write_packet(args.directory, method, code)
        print(json.dumps({"packet_version": packet["packet_version"],
                          "reviewed_steps": len(packet["steps"]),
                          "mapping_issues": len(packet["mapping_issues"])}))
        return 0
    reviewed = verify_semantic_review(args.directory, method, code)
    print(json.dumps({"semantic_equivalence": reviewed["semantic_equivalence"],
                      "reviewer": reviewed["reviewer"]["checker"], "version": reviewed["version"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
