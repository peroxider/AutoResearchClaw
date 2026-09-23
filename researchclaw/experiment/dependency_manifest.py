"""Frozen dependency inventory for auditable host execution.

For host runs the manifest records the exact installed-distribution
inventory: name, version, and a sha256 of each distribution's RECORD —
which itself pins the per-file hashes of everything the distribution
installed. For container runs the interior inventory is deliberately
not re-recorded: the pinned image digest IS the dependency evidence,
and host python/platform strings would lie about the container interior.

Nothing here recomputes against the live host at acceptance time — a
frozen bundle must stay auditable on a different machine. Integrity
comes from binding: the manifest's content hash rides in every ledger
start event (hash-chained) and every execution receipt, so any
post-freeze edit breaks verification.
"""
from __future__ import annotations

import hashlib
import platform
import re
from typing import Any

from researchclaw.pipeline.evidence_store import content_hash
from researchclaw.pipeline.experiment_protocol import ProtocolError

_RECORD_DIGEST = re.compile(r"[0-9a-f]{64}")


def _record_digest(dist: Any) -> str | None:
    text = dist.read_text("RECORD")
    return hashlib.sha256(text.encode("utf-8")).hexdigest() if text is not None else None


def build_dependency_manifest(backend: str, image_digest: str | None = None) -> dict[str, Any]:
    """Freeze the execution environment's dependency evidence."""
    if backend not in ("host", "docker"):
        raise ProtocolError("Dependency manifest backend must be host or docker")
    if backend == "docker":
        return {"schema_version": 1, "backend": "docker", "python": None,
                "platform": None, "image_digest": image_digest, "distributions": {}}
    import importlib.metadata as metadata
    discovered = []
    for dist in metadata.distributions():
        name = dist.metadata.get("Name") or ""
        if name:
            discovered.append((name, str(getattr(dist, "_path", "")), dist))
    distributions: dict[str, dict[str, Any]] = {}
    for name, _path, dist in sorted(discovered, key=lambda item: (item[0].lower(), item[1])):
        if name in distributions:
            continue  # First hit in path order shadows later duplicates.
        distributions[name] = {"version": dist.version, "record_sha256": _record_digest(dist)}
    if not distributions:
        raise ProtocolError("Dependency manifest found no installed distributions")
    return {"schema_version": 1, "backend": "host", "python": platform.python_version(),
            "platform": platform.platform(), "image_digest": None,
            "distributions": distributions}


def validate_dependency_manifest(manifest: Any) -> None:
    """Fail closed on any shape that cannot be the manifest run_matrix froze."""
    if (not isinstance(manifest, dict) or manifest.get("schema_version") != 1
            or manifest.get("backend") not in ("host", "docker")):
        raise ProtocolError("Invalid frozen dependency manifest")
    digest = manifest.get("image_digest")
    if manifest["backend"] == "docker":
        if not isinstance(digest, str) or not digest:
            raise ProtocolError("Invalid frozen dependency manifest")
        # A container run pins the image digest and must not pretend the
        # host inventory describes the container interior.
        if (manifest.get("python") is not None or manifest.get("platform") is not None
                or manifest.get("distributions") != {}):
            raise ProtocolError("Container runs pin the image digest, not a host inventory")
        return
    if digest is not None:
        raise ProtocolError("Invalid frozen dependency manifest")
    if (not isinstance(manifest.get("python"), str) or not manifest["python"]
            or not isinstance(manifest.get("platform"), str) or not manifest["platform"]
            or not isinstance(manifest.get("distributions"), dict)
            or not manifest["distributions"]):
        raise ProtocolError("Invalid frozen dependency manifest")
    for name, entry in manifest["distributions"].items():
        record = entry.get("record_sha256") if isinstance(entry, dict) else None
        version = entry.get("version") if isinstance(entry, dict) else None
        if (not isinstance(name, str) or not name or not isinstance(entry, dict)
                or not isinstance(version, str) or not version
                or not (record is None or (isinstance(record, str)
                        and _RECORD_DIGEST.fullmatch(record)))):
            raise ProtocolError("Invalid frozen dependency manifest")
