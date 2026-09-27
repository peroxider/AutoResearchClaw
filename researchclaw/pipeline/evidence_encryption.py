"""Recipient-bound authenticated encryption for private benchmark evidence."""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
from pathlib import Path

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric.x25519 import (
    X25519PrivateKey, X25519PublicKey,
)
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from researchclaw.pipeline.evidence_store import content_hash


_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}")
_PURPOSE = "benchmark_private_gold/v1"


class EvidenceEncryptionError(ValueError):
    pass


def _canonical(value: dict) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise EvidenceEncryptionError("Encrypted payload is not canonical JSON") from exc


def _encode64(value: bytes) -> str:
    return base64.b64encode(value).decode("ascii")


def _decode64(value: object, *, length: int | None, name: str,
              maximum: int = 15_000_000) -> bytes:
    if not isinstance(value, str) or len(value) > maximum * 2:
        raise EvidenceEncryptionError(f"{name} is malformed")
    try:
        decoded = base64.b64decode(value, validate=True)
    except (ValueError, TypeError) as exc:
        raise EvidenceEncryptionError(f"{name} is malformed") from exc
    if ((length is not None and len(decoded) != length)
            or len(decoded) > maximum or _encode64(decoded) != value):
        raise EvidenceEncryptionError(f"{name} is malformed")
    return decoded


def validate_recipient(value: object) -> dict:
    fields = {"algorithm", "key_id", "public_key"}
    if (not isinstance(value, dict) or set(value) != fields
            or value.get("algorithm") != "x25519-aes256-gcm"
            or not isinstance(value.get("key_id"), str)
            or _SAFE_ID.fullmatch(value["key_id"]) is None):
        raise EvidenceEncryptionError("Encryption recipient is malformed")
    _decode64(value.get("public_key"), length=32, name="Recipient public key")
    return dict(value)


def generate_private_key(path: Path, *, key_id: str) -> dict:
    if not isinstance(key_id, str) or _SAFE_ID.fullmatch(key_id) is None:
        raise EvidenceEncryptionError("Encryption key ID is malformed")
    destination = Path(path).resolve()
    if destination.exists():
        raise EvidenceEncryptionError("Private key destination already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    private_key = X25519PrivateKey.generate()
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
    return {"algorithm": "x25519-aes256-gcm", "key_id": key_id,
            "public_key": _encode64(public)}


def _load_private(path: Path) -> X25519PrivateKey:
    source = Path(path).resolve()
    try:
        if not source.is_file() or source.stat().st_size > 10_000:
            raise EvidenceEncryptionError("Private decryption key is missing or oversized")
        key = serialization.load_pem_private_key(source.read_bytes(), password=None)
    except EvidenceEncryptionError:
        raise
    except (OSError, ValueError, TypeError) as exc:
        raise EvidenceEncryptionError("Private decryption key is unreadable") from exc
    if not isinstance(key, X25519PrivateKey):
        raise EvidenceEncryptionError("Private decryption key is not X25519")
    return key


def _derive(shared: bytes, salt: bytes) -> bytes:
    return HKDF(
        algorithm=hashes.SHA256(), length=32, salt=salt,
        info=b"researchclaw/benchmark-private-gold/v1").derive(shared)


def seal_document(document: dict, *, recipient: dict) -> dict:
    identity = validate_recipient(recipient)
    plaintext = _canonical(document)
    if len(plaintext) > 10_000_000:
        raise EvidenceEncryptionError("Private gold exceeds 10 MB")
    ephemeral = X25519PrivateKey.generate()
    ephemeral_public = ephemeral.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw)
    recipient_public = X25519PublicKey.from_public_bytes(_decode64(
        identity["public_key"], length=32, name="Recipient public key"))
    salt, nonce = os.urandom(16), os.urandom(12)
    header = {
        "algorithm": "x25519-aes256-gcm", "key_id": identity["key_id"],
        "purpose": _PURPOSE, "ephemeral_public_key": _encode64(ephemeral_public),
        "salt": _encode64(salt), "nonce": _encode64(nonce),
        "plaintext_sha256": content_hash(document),
    }
    try:
        key = _derive(ephemeral.exchange(recipient_public), salt)
    except ValueError as exc:
        raise EvidenceEncryptionError("Recipient public key is unusable") from exc
    ciphertext = AESGCM(key).encrypt(nonce, plaintext, _canonical(header))
    return {**header, "ciphertext": _encode64(ciphertext)}


def validate_envelope(envelope: object, *, recipient: dict) -> dict:
    identity = validate_recipient(recipient)
    fields = {"algorithm", "key_id", "purpose", "ephemeral_public_key",
              "salt", "nonce", "plaintext_sha256", "ciphertext"}
    if (not isinstance(envelope, dict) or set(envelope) != fields
            or envelope.get("algorithm") != "x25519-aes256-gcm"
            or envelope.get("key_id") != identity["key_id"]
            or envelope.get("purpose") != _PURPOSE
            or not isinstance(envelope.get("plaintext_sha256"), str)
            or re.fullmatch(r"[0-9a-f]{64}", envelope["plaintext_sha256"]) is None):
        raise EvidenceEncryptionError("Sealed evidence envelope is malformed")
    _decode64(envelope["ephemeral_public_key"], length=32,
              name="Ephemeral public key")
    _decode64(envelope["salt"], length=16, name="Encryption salt")
    _decode64(envelope["nonce"], length=12, name="Encryption nonce")
    ciphertext = _decode64(
        envelope["ciphertext"], length=None, name="Encrypted evidence")
    if len(ciphertext) < 16:
        raise EvidenceEncryptionError("Encrypted evidence is malformed")
    return dict(envelope)


def open_document(envelope: dict, *, recipient: dict,
                  private_key_path: Path) -> dict:
    value = validate_envelope(envelope, recipient=recipient)
    private_key = _load_private(private_key_path)
    actual_public = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw)
    identity = validate_recipient(recipient)
    expected_public = _decode64(
        identity["public_key"], length=32, name="Recipient public key")
    if actual_public != expected_public:
        raise EvidenceEncryptionError(
            "Private decryption key does not match frozen public key")
    ephemeral = X25519PublicKey.from_public_bytes(_decode64(
        value["ephemeral_public_key"], length=32, name="Ephemeral public key"))
    salt = _decode64(value["salt"], length=16, name="Encryption salt")
    nonce = _decode64(value["nonce"], length=12, name="Encryption nonce")
    ciphertext = _decode64(
        value["ciphertext"], length=None, name="Encrypted evidence")
    header = {key: item for key, item in value.items() if key != "ciphertext"}
    try:
        plaintext = AESGCM(_derive(private_key.exchange(ephemeral), salt)).decrypt(
            nonce, ciphertext, _canonical(header))
        document = json.loads(plaintext.decode("utf-8"))
    except (InvalidTag, UnicodeDecodeError, ValueError) as exc:
        raise EvidenceEncryptionError("Sealed evidence authentication failed") from exc
    if (not isinstance(document, dict)
            or content_hash(document) != value["plaintext_sha256"]):
        raise EvidenceEncryptionError("Sealed evidence plaintext digest differs")
    return document


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate a private-evidence encryption recipient")
    parser.add_argument("generate", nargs="?")
    parser.add_argument("--private-key", type=Path, required=True)
    parser.add_argument("--key-id", required=True)
    args = parser.parse_args(argv)
    if args.generate not in {None, "generate"}:
        raise EvidenceEncryptionError("Unknown encryption command")
    print(json.dumps(generate_private_key(
        args.private_key, key_id=args.key_id), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
