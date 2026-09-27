"""Build and verify least-privilege runner bundles for blind benchmarks."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import uuid
import zipfile
from pathlib import Path, PurePosixPath

from researchclaw.pipeline.benchmark_suite import (
    BenchmarkSuiteError, verify_public_benchmark_suite,
)
from researchclaw.pipeline.evidence_store import content_hash, file_hash
from researchclaw.pipeline.evidence_signature import (
    EvidenceSignatureError, verify_document,
)


_SHA256 = re.compile(r"[0-9a-f]{64}")
_MANIFEST = "_arc/runner_bundle_manifest.json"
_SUITE_REPORT = "_arc/benchmark_suite.json"
_MAX_ENTRIES = 520
_MAX_FILE_BYTES = 15_000_000
_MAX_TOTAL_BYTES = 6_000_000_000


class BenchmarkDistributionError(ValueError):
    pass


def _json_bytes(value: dict) -> bytes:
    return json.dumps(
        value, indent=2, sort_keys=True, allow_nan=False).encode("utf-8")


def _safe_name(value: object) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise BenchmarkDistributionError("Runner bundle path is malformed")
    pure = PurePosixPath(value)
    if (pure.is_absolute() or not pure.parts
            or any(part in {".", ".."} or ":" in part
                   or any(ord(character) < 32 for character in part)
                   for part in pure.parts)
            or str(pure) != value):
        raise BenchmarkDistributionError("Runner bundle path is unsafe")
    return value


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = (stat.S_IFREG | 0o644) << 16
    info.create_system = 3
    return info


def _source(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if (not path.is_relative_to(root) or not path.is_file()
            or (root / relative).is_symlink()
            or path.stat().st_size > _MAX_FILE_BYTES):
        raise BenchmarkDistributionError(
            "Runner bundle source is missing, unsafe or oversized")
    return path


def _hash_stream(stream) -> tuple[str, int]:
    digest, size = hashlib.sha256(), 0
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        size += len(chunk)
        if size > _MAX_FILE_BYTES:
            raise BenchmarkDistributionError("Runner bundle entry exceeds size limit")
        digest.update(chunk)
    return digest.hexdigest(), size


def build_runner_bundle(*, suite_report: dict, public_root: Path,
                        plan_path: Path, output_path: Path) -> dict:
    """Create a deterministic explicit-file runner ZIP with no private key or gold."""
    root, plan_path = Path(public_root).resolve(), Path(plan_path).resolve()
    try:
        verify_public_benchmark_suite(suite_report, root, plan_path)
    except BenchmarkSuiteError as exc:
        raise BenchmarkDistributionError(str(exc)) from exc
    plan = suite_report["plan"]
    if ("confidentiality" not in plan
            or "curator" not in plan.get("attestation", {})
            or "signature" not in suite_report):
        raise BenchmarkDistributionError(
            "Runner distribution requires curator-signed sealed gold")
    plan_relative = plan_path.relative_to(root).as_posix()
    roles: dict[str, str] = {plan_relative: "public_plan"}
    for case in plan["cases"]:
        roles[case["input_bundle"]] = "public_case_input"
    roles[plan["runner"]["adapter"]] = "runner_adapter"
    roles[plan["confidentiality"]["sealed_gold"]] = "sealed_gold"
    reserved = {_MANIFEST, _SUITE_REPORT}
    if set(roles) & reserved:
        raise BenchmarkDistributionError("Public input collides with runner metadata path")
    sources = {name: _source(root, _safe_name(name)) for name in roles}
    suite_bytes = _json_bytes(suite_report)
    files = [{
        "path": name, "sha256": file_hash(path),
        "size": path.stat().st_size, "role": roles[name],
    } for name, path in sorted(sources.items())]
    files.append({
        "path": _SUITE_REPORT,
        "sha256": hashlib.sha256(suite_bytes).hexdigest(),
        "size": len(suite_bytes), "role": "signed_suite_report",
    })
    manifest = {
        "schema_version": 1, "checker": "arc-runner-bundle/v1",
        "suite_version": suite_report["version"],
        "entrypoints": {"suite_report": _SUITE_REPORT, "plan": plan_relative},
        "files": files,
        "scope": "explicit public runner inputs and sealed gold only",
        "limitations": [
            "Bundle separation does not control copies retained outside this archive.",
            "Adapter source may itself contain operator-provided literals.",
        ],
    }
    manifest["version"] = content_hash(manifest)
    manifest_bytes = _json_bytes(manifest)
    output = Path(output_path).resolve()
    if output.exists():
        raise BenchmarkDistributionError("Runner bundle output already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")
    try:
        with zipfile.ZipFile(
                temporary, "w", compression=zipfile.ZIP_DEFLATED,
                compresslevel=9, allowZip64=True) as archive:
            for name, path in sorted(sources.items()):
                with path.open("rb") as source, archive.open(
                        _zip_info(name), "w") as target:
                    shutil.copyfileobj(source, target, length=1024 * 1024)
            archive.writestr(_zip_info(_SUITE_REPORT), suite_bytes)
            archive.writestr(_zip_info(_MANIFEST), manifest_bytes)
        verify_runner_bundle(temporary)
        temporary.replace(output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return manifest


def _load_manifest(archive: zipfile.ZipFile) -> dict:
    try:
        info = archive.getinfo(_MANIFEST)
        if info.file_size > 1_000_000:
            raise BenchmarkDistributionError("Runner bundle manifest is oversized")
        value = json.loads(archive.read(info).decode("utf-8"))
    except BenchmarkDistributionError:
        raise
    except (KeyError, UnicodeDecodeError, ValueError, OSError) as exc:
        raise BenchmarkDistributionError("Runner bundle manifest is unreadable") from exc
    if not isinstance(value, dict):
        raise BenchmarkDistributionError("Runner bundle manifest must be an object")
    return value


def verify_runner_bundle(bundle_path: Path) -> dict:
    """Verify exact archive membership, paths, sizes, hashes and manifest version."""
    bundle = Path(bundle_path).resolve()
    try:
        if not bundle.is_file():
            raise BenchmarkDistributionError("Runner bundle is missing")
        with zipfile.ZipFile(bundle, "r") as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if (not infos or len(infos) > _MAX_ENTRIES or len(set(names)) != len(names)
                    or any(_safe_name(name) != name for name in names)
                    or any(info.is_dir() or info.flag_bits & 0x1 for info in infos)
                    or any(stat.S_ISLNK((info.external_attr >> 16) & 0xFFFF)
                           for info in infos)
                    or sum(info.file_size for info in infos) > _MAX_TOTAL_BYTES):
                raise BenchmarkDistributionError(
                    "Runner bundle membership is unsafe or exceeds limits")
            manifest = _load_manifest(archive)
            fields = {"schema_version", "checker", "suite_version", "entrypoints",
                      "files", "scope", "limitations", "version"}
            unsigned = {key: value for key, value in manifest.items()
                        if key != "version"}
            if (set(manifest) != fields or manifest.get("schema_version") != 1
                    or manifest.get("checker") != "arc-runner-bundle/v1"
                    or manifest.get("version") != content_hash(unsigned)
                    or not isinstance(manifest.get("files"), list)
                    or not isinstance(manifest.get("entrypoints"), dict)
                    or set(manifest["entrypoints"]) != {"suite_report", "plan"}
                    or manifest.get("scope") != (
                        "explicit public runner inputs and sealed gold only")
                    or manifest.get("limitations") != [
                        "Bundle separation does not control copies retained "
                        "outside this archive.",
                        "Adapter source may itself contain operator-provided literals.",
                    ]):
                raise BenchmarkDistributionError("Runner bundle manifest is malformed")
            declared = {}
            file_fields = {"path", "sha256", "size", "role"}
            for row in manifest["files"]:
                if (not isinstance(row, dict) or set(row) != file_fields
                        or _safe_name(row.get("path")) in declared
                        or row.get("path") == _MANIFEST
                        or not isinstance(row.get("sha256"), str)
                        or _SHA256.fullmatch(row["sha256"]) is None
                        or type(row.get("size")) is not int
                        or not 0 <= row["size"] <= _MAX_FILE_BYTES
                        or not isinstance(row.get("role"), str) or not row["role"]):
                    raise BenchmarkDistributionError(
                        "Runner bundle file declaration is malformed")
                declared[row["path"]] = row
            expected_names = set(declared) | {_MANIFEST}
            if set(names) != expected_names:
                raise BenchmarkDistributionError(
                    "Runner bundle contains missing or undeclared files")
            if (manifest["entrypoints"]["suite_report"] != _SUITE_REPORT
                    or _SUITE_REPORT not in declared
                    or manifest["entrypoints"]["plan"] not in declared):
                raise BenchmarkDistributionError("Runner bundle entrypoints are malformed")
            for name, row in declared.items():
                info = archive.getinfo(name)
                if info.file_size != row["size"] or info.file_size > _MAX_FILE_BYTES:
                    raise BenchmarkDistributionError("Runner bundle file size differs")
                with archive.open(info, "r") as stream:
                    digest, size = _hash_stream(stream)
                if digest != row["sha256"] or size != row["size"]:
                    raise BenchmarkDistributionError("Runner bundle file digest differs")
            suite = json.loads(archive.read(_SUITE_REPORT).decode("utf-8"))
            if (not isinstance(suite, dict)
                    or suite.get("version") != manifest["suite_version"]):
                raise BenchmarkDistributionError("Runner bundle suite identity differs")
            plan = suite.get("plan")
            attestation = plan.get("attestation", {}) if isinstance(plan, dict) else {}
            if (not isinstance(plan, dict) or "curator" not in attestation
                    or "confidentiality" not in plan or "signature" not in suite
                    or suite["version"] != content_hash({
                        key: value for key, value in suite.items()
                        if key not in {"version", "signature"}})):
                raise BenchmarkDistributionError(
                    "Runner bundle suite is not curator-signed sealed evidence")
            try:
                verify_document(
                    suite, signer=attestation["curator"],
                    purpose="benchmark_suite_report/v1")
            except EvidenceSignatureError as exc:
                raise BenchmarkDistributionError(str(exc)) from exc
            plan_name = manifest["entrypoints"]["plan"]
            plan_document = json.loads(archive.read(plan_name).decode("utf-8"))
            if (not isinstance(plan_document, dict)
                    or declared[plan_name]["sha256"] != suite["plan_sha256"]):
                raise BenchmarkDistributionError("Runner bundle plan differs from suite")
            required_roles = {plan_name: "public_plan", _SUITE_REPORT: "signed_suite_report"}
            required_hashes = {
                plan_name: suite["plan_sha256"], _SUITE_REPORT: declared[_SUITE_REPORT]["sha256"]}
            for case in plan["cases"]:
                required_roles[case["input_bundle"]] = "public_case_input"
                required_hashes[case["input_bundle"]] = case["input_sha256"]
            adapter = plan["runner"]["adapter"]
            required_roles[adapter] = "runner_adapter"
            required_hashes[adapter] = plan["runner"]["adapter_sha256"]
            sealed = plan["confidentiality"]["sealed_gold"]
            required_roles[sealed] = "sealed_gold"
            required_hashes[sealed] = suite["sealed_gold"]["sha256"]
            if set(declared) != set(required_roles):
                raise BenchmarkDistributionError(
                    "Runner bundle declarations differ from signed suite inputs")
            for name, role in required_roles.items():
                if (declared[name]["role"] != role
                        or declared[name]["sha256"] != required_hashes[name]):
                    raise BenchmarkDistributionError(
                        "Runner bundle role or digest differs from signed suite")
            return manifest
    except BenchmarkDistributionError:
        raise
    except (OSError, zipfile.BadZipFile, ValueError, KeyError) as exc:
        raise BenchmarkDistributionError("Runner bundle is unreadable") from exc


def extract_runner_bundle(bundle_path: Path, destination: Path) -> dict:
    """Verify then extract to a new or empty directory without following links."""
    manifest = verify_runner_bundle(bundle_path)
    target = Path(destination).resolve()
    if target.exists() and (not target.is_dir() or any(target.iterdir())):
        raise BenchmarkDistributionError("Runner bundle destination is not empty")
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(Path(bundle_path).resolve(), "r") as archive:
        for info in archive.infolist():
            output = (target / info.filename).resolve()
            if not output.is_relative_to(target):
                raise BenchmarkDistributionError("Runner bundle extraction path is unsafe")
            output.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info, "r") as source, output.open("xb") as sink:
                shutil.copyfileobj(source, sink, length=1024 * 1024)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build or inspect a benchmark runner bundle")
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--suite-report", type=Path, required=True)
    build.add_argument("--public-root", type=Path, required=True)
    build.add_argument("--plan", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("bundle", type=Path)
    extract = commands.add_parser("extract")
    extract.add_argument("bundle", type=Path)
    extract.add_argument("destination", type=Path)
    args = parser.parse_args(argv)
    if args.command == "build":
        suite = json.loads(args.suite_report.read_text(encoding="utf-8"))
        report = build_runner_bundle(
            suite_report=suite, public_root=args.public_root,
            plan_path=args.plan, output_path=args.output)
    elif args.command == "verify":
        report = verify_runner_bundle(args.bundle)
    else:
        report = extract_runner_bundle(args.bundle, args.destination)
    print(json.dumps({"suite_version": report["suite_version"],
                      "files": len(report["files"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
