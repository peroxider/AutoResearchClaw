"""Ed25519 signatures for canonical JSON evidence documents."""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import uuid
from pathlib import Path

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey, Ed25519PublicKey,
)

from researchclaw.pipeline.evidence_store import content_hash


_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}")
_PURPOSES = {
    "benchmark_result_manifest/v1",
    "benchmark_private_assessment/v1",
}


class EvidenceSignatureError(ValueError):
    pass


def _canonical(value: dict) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise EvidenceSignatureError("Signed payload is not canonical JSON") from exc


def _decode64(value: object, *, length: int, name: str) -> bytes:
    if not isinstance(value, str) or len(value) > 512:
        raise EvidenceSignatureError(f"{name} is malformed")
    try:
        decoded = base64.b64decode(value, validate=True)
    except (ValueError, TypeError) as exc:
        raise EvidenceSignatureError(f"{name} is malformed") from exc
    if len(decoded) != length or base64.b64encode(decoded).decode("ascii") != value:
        raise EvidenceSignatureError(f"{name} is malformed")
    return decoded


def validate_signer(value: object) -> dict:
    fields = {"algorithm", "key_id", "public_key"}
    if (not isinstance(value, dict) or set(value) != fields
            or value.get("algorithm") != "ed25519"
            or not isinstance(value.get("key_id"), str)
            or _SAFE_ID.fullmatch(value["key_id"]) is None):
        raise EvidenceSignatureError("Signer identity is malformed")
    _decode64(value.get("public_key"), length=32, name="Signer public key")
    return dict(value)


def generate_private_key(path: Path, *, key_id: str) -> dict:
    if not isinstance(key_id, str) or _SAFE_ID.fullmatch(key_id) is None:
        raise EvidenceSignatureError("Signer key ID is malformed")
    destination = Path(path).resolve()
    if destination.exists():
        raise EvidenceSignatureError("Private key destination already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    private_key = Ed25519PrivateKey.generate()
    data = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption())
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if os.name != "nt":
        flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(destination, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    public = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw)
    return {"algorithm": "ed25519", "key_id": key_id,
            "public_key": base64.b64encode(public).decode("ascii")}


def _load_private(path: Path) -> Ed25519PrivateKey:
    source = Path(path).resolve()
    try:
        if not source.is_file() or source.stat().st_size > 10_000:
            raise EvidenceSignatureError("Private signing key is missing or oversized")
        key = serialization.load_pem_private_key(source.read_bytes(), password=None)
    except EvidenceSignatureError:
        raise
    except (OSError, ValueError, TypeError) as exc:
        raise EvidenceSignatureError("Private signing key is unreadable") from exc
    if not isinstance(key, Ed25519PrivateKey):
        raise EvidenceSignatureError("Private signing key is not Ed25519")
    return key


def sign_document(document: dict, *, signer: dict, private_key_path: Path,
                  purpose: str) -> dict:
    identity = validate_signer(signer)
    if purpose not in _PURPOSES or "signature" in document:
        raise EvidenceSignatureError("Signature purpose or payload is malformed")
    private_key = _load_private(private_key_path)
    actual_public = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw)
    expected_public = _decode64(
        identity["public_key"], length=32, name="Signer public key")
    if actual_public != expected_public:
        raise EvidenceSignatureError("Private signing key does not match frozen public key")
    signature = private_key.sign(_canonical(document))
    return {
        "algorithm": "ed25519", "key_id": identity["key_id"],
        "purpose": purpose, "payload_sha256": content_hash(document),
        "value": base64.b64encode(signature).decode("ascii"),
    }


def verify_document(document: dict, *, signer: dict, purpose: str) -> dict:
    identity = validate_signer(signer)
    signature = document.get("signature")
    fields = {"algorithm", "key_id", "purpose", "payload_sha256", "value"}
    payload = {key: value for key, value in document.items() if key != "signature"}
    if (purpose not in _PURPOSES or not isinstance(signature, dict)
            or set(signature) != fields
            or signature.get("algorithm") != "ed25519"
            or signature.get("key_id") != identity["key_id"]
            or signature.get("purpose") != purpose
            or signature.get("payload_sha256") != content_hash(payload)):
        raise EvidenceSignatureError("Evidence signature envelope is malformed")
    raw_signature = _decode64(
        signature.get("value"), length=64, name="Evidence signature")
    public_key = Ed25519PublicKey.from_public_bytes(_decode64(
        identity["public_key"], length=32, name="Signer public key"))
    try:
        public_key.verify(raw_signature, _canonical(payload))
    except InvalidSignature as exc:
        raise EvidenceSignatureError("Evidence signature is invalid") from exc
    return document


def _write(path: Path, value: dict) -> None:
    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(destination)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sign canonical benchmark evidence")
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate")
    generate.add_argument("--private-key", type=Path, required=True)
    generate.add_argument("--key-id", required=True)
    sign = commands.add_parser("sign-json")
    sign.add_argument("--input", type=Path, required=True)
    sign.add_argument("--output", type=Path, required=True)
    sign.add_argument("--private-key", type=Path, required=True)
    sign.add_argument("--signer", type=Path, required=True)
    sign.add_argument("--purpose", choices=sorted(_PURPOSES), required=True)
    args = parser.parse_args(argv)
    if args.command == "generate":
        print(json.dumps(generate_private_key(
            args.private_key, key_id=args.key_id), sort_keys=True))
        return 0
    document = json.loads(args.input.read_text(encoding="utf-8"))
    signer = json.loads(args.signer.read_text(encoding="utf-8"))
    protected = {args.input.resolve(), args.private_key.resolve(), args.signer.resolve()}
    if args.output.resolve() in protected:
        raise EvidenceSignatureError("Signed output cannot overwrite an input or private key")
    if not isinstance(document, dict):
        raise EvidenceSignatureError("Signed JSON document must be an object")
    signed = dict(document)
    signed["signature"] = sign_document(
        document, signer=signer, private_key_path=args.private_key,
        purpose=args.purpose)
    _write(args.output, signed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
