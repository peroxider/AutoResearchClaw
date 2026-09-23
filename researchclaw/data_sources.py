"""Content-addressed acquisition of declared external dataset files.

An external source is admitted only when the manifest pins its exact content:
the declaration carries a lowercase SHA-256 of the bytes, acquisition verifies
the fetched payload against that pin before anything is written, and every
cache reuse re-verifies payload and provenance. Integrity comes from the pin,
not from the transport; a pinned-but-wrong dataset is a provenance error this
module cannot detect and does not pretend to. The origin URL lives only in
the run-local provenance sidecar — it is deliberately not propagated into
model-facing dataset cards.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path

from researchclaw.pipeline.evidence_store import file_hash

PROVENANCE_ADAPTER = "data-sources/v1"
DEFAULT_TIMEOUT_SECONDS = 30
MAX_SIZE_CAP = 1024 * 1024 * 1024
_DIGEST = re.compile(r"[0-9a-f]{64}")
# Concurrent Windows renames can transiently lock or hide a path for a
# reader; a bounded retry keeps such races from failing operations whose
# final state is already correct. Integrity is re-checked after every retry.
_RETRY_ATTEMPTS = 250
_RETRY_PAUSE = 0.001


def _retry_transient(operation):
    for attempt in range(_RETRY_ATTEMPTS):
        try:
            return operation()
        except OSError:
            if attempt + 1 == _RETRY_ATTEMPTS:
                raise
            time.sleep(_RETRY_PAUSE)


def _hash_with_retry(path: Path) -> str:
    return _retry_transient(lambda: file_hash(path))


class DataAcquisitionError(ValueError):
    pass


def _check_provenance(sidecar: Path, sha: str, payload: Path) -> None:
    if not sidecar.is_file():
        raise DataAcquisitionError("Cached external data lacks its provenance record")
    try:
        record = json.loads(_retry_transient(
            lambda: sidecar.read_text(encoding="utf-8")))
    except ValueError as exc:
        raise DataAcquisitionError("Cached external data provenance is unreadable") from exc
    size = _retry_transient(lambda: payload.stat().st_size)
    if (not isinstance(record, dict) or record.get("adapter") != PROVENANCE_ADAPTER
            or record.get("sha256") != sha or type(record.get("size")) is not int
            or record.get("size") != size
            or not isinstance(record.get("url"), str) or not record["url"]):
        raise DataAcquisitionError("Cached external data provenance is inconsistent with the payload")


def normalize_fetch(declaration) -> dict:
    """Fail closed on any shape that is not a pinned, bounded acquisition."""
    if not isinstance(declaration, dict) or set(declaration) != {"url", "sha256", "size_cap"}:
        raise DataAcquisitionError("External source requires exactly url, sha256 and size_cap")
    url, sha, cap = declaration["url"], declaration["sha256"], declaration["size_cap"]
    if not isinstance(url, str) or not url:
        raise DataAcquisitionError("External source URL must be a non-empty string")
    # Python's urlparse silently strips surrounding whitespace; reject it
    # explicitly so a padded origin can never masquerade as a clean URL.
    if any(char.isspace() for char in url):
        raise DataAcquisitionError("External source URL must not contain whitespace")
    if not isinstance(sha, str) or not _DIGEST.fullmatch(sha):
        raise DataAcquisitionError("External source requires a lowercase 64-hex sha256 content pin")
    if type(cap) is not int or not 1 <= cap <= MAX_SIZE_CAP:
        raise DataAcquisitionError(f"External source size_cap must be an integer in 1..{MAX_SIZE_CAP}")
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("https", "file"):
        raise DataAcquisitionError("External source URL scheme must be https or file")
    if parsed.scheme == "https" and not parsed.netloc:
        raise DataAcquisitionError("External https source requires a host")
    if parsed.scheme == "file":
        if parsed.netloc not in ("", "localhost"):
            raise DataAcquisitionError("External file source must be local")
        if not parsed.path or not Path(urllib.request.url2pathname(parsed.path)).is_absolute():
            raise DataAcquisitionError("External file source must be an absolute file URL")
    if parsed.username is not None or parsed.password is not None:
        raise DataAcquisitionError("External source URL must not embed credentials")
    if parsed.fragment:
        raise DataAcquisitionError("External source URL must not carry a fragment")
    return {"url": url, "sha256": sha, "size_cap": cap}


def _read_capped(stream, size_cap: int, url: str) -> bytes:
    chunks, remaining = [], size_cap + 1
    while remaining > 0:
        chunk = stream.read(min(1 << 20, remaining))
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    raw = b"".join(chunks)
    if len(raw) > size_cap:
        raise DataAcquisitionError(f"External source exceeds its declared size cap: {url}")
    return raw


def _fetch_bytes(url: str, size_cap: int) -> bytes:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme == "file":
        local = Path(urllib.request.url2pathname(parsed.path))
        if not local.is_file():
            raise DataAcquisitionError("External file source does not exist")
        with local.open("rb") as stream:
            return _read_capped(stream, size_cap, url)
    request = urllib.request.Request(url, method="GET",
                                     headers={"User-Agent": "AutoResearchClaw data-sources/v1"})
    try:
        with urllib.request.urlopen(request, timeout=DEFAULT_TIMEOUT_SECONDS) as response:
            return _read_capped(response, size_cap, url)
    except DataAcquisitionError:
        raise
    except (OSError, ValueError) as exc:
        raise DataAcquisitionError(f"External source is unreachable: {exc}") from exc


def materialize_fetch(declaration, cache_dir, expected_name: str | None = None) -> Path:
    """Return the cached payload for a pinned declaration, fetching on miss."""
    declaration = normalize_fetch(declaration)
    sha = declaration["sha256"]
    # The content-addressed payload name is the only accepted alias; it keeps
    # the manifest path and the pinned digest mutually consistent.
    if expected_name is not None and expected_name != f"{sha}.csv":
        raise DataAcquisitionError("External payload name must be the pinned digest '<sha256>.csv'")
    cache_dir = Path(cache_dir)
    payload = cache_dir / f"{sha}.csv"
    sidecar = cache_dir / f"{sha}.provenance.json"
    if payload.is_file():
        if _hash_with_retry(payload) != sha:
            raise DataAcquisitionError(
                "Cached external data no longer matches its pinned digest; investigate instead of refetching")
        _check_provenance(sidecar, sha, payload)
        return payload
    raw = _fetch_bytes(declaration["url"], declaration["size_cap"])
    if hashlib.sha256(raw).hexdigest() != sha:
        raise DataAcquisitionError("Fetched external data does not match its pinned sha256; nothing was cached")
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Provenance becomes visible before the payload. A concurrent reader that
    # finds the payload must never miss its provenance record, so the
    # sidecar-first ordering is what makes the cache fast path atomic: the
    # reverse order exposed a window where the payload was already readable
    # and the fast path failed with a missing provenance record.
    record = {"adapter": PROVENANCE_ADAPTER, "sha256": sha, "size": len(raw), "url": declaration["url"]}
    staging = cache_dir / f"{sha}.{os.getpid()}.{threading.get_ident()}.provenance.staging"
    try:
        staging.write_text(json.dumps(record, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        try:
            staging.replace(sidecar)
        except OSError:
            # The record is deterministic; a concurrent writer's copy is
            # equivalent. An absent sidecar is rechecked briefly for the same
            # rename-window reason as the payload.
            for _ in range(_RETRY_ATTEMPTS):
                if sidecar.is_file():
                    break
                time.sleep(_RETRY_PAUSE)
            else:
                raise
    finally:
        staging.unlink(missing_ok=True)
    # Staging names are unique per writer: shared names would let one
    # writer's cleanup unlink the file another writer is about to replace.
    # Every writer to one content address carries the same verified bytes,
    # so concurrent staging writes cannot tear anything.
    staging = cache_dir / f"{sha}.{os.getpid()}.{threading.get_ident()}.staging"
    try:
        staging.write_bytes(raw)
        try:
            staging.replace(payload)
        except OSError:
            # On Windows a concurrent writer to the same content address wins
            # the replace; accept its payload only if it is exactly the pinned
            # bytes. A rename can transiently hide or lock the destination, so
            # an absent payload is rechecked briefly before giving up.
            accepted = False
            for _ in range(_RETRY_ATTEMPTS):
                if payload.is_file() and _hash_with_retry(payload) == sha:
                    accepted = True
                    break
                time.sleep(_RETRY_PAUSE)
            if not accepted:
                raise
    finally:
        staging.unlink(missing_ok=True)
    _check_provenance(sidecar, sha, payload)
    return payload
