from __future__ import annotations

import base64
import copy
import json

import pytest

from researchclaw.pipeline.evidence_encryption import (
    EvidenceEncryptionError, generate_private_key, main, open_document,
    seal_document, validate_envelope,
)


def test_sealed_document_round_trip_and_public_validation(tmp_path):
    private = tmp_path / "evaluator.pem"
    recipient = generate_private_key(private, key_id="evaluator-1")
    document = {"schema_version": 1, "cases": [{"case_id": "a"}]}
    envelope = seal_document(document, recipient=recipient)
    assert "case_id" not in json.dumps(envelope)
    assert validate_envelope(envelope, recipient=recipient) == envelope
    assert open_document(
        envelope, recipient=recipient, private_key_path=private) == document


def test_ciphertext_and_authenticated_header_tamper_fail_closed(tmp_path):
    private = tmp_path / "evaluator.pem"
    recipient = generate_private_key(private, key_id="evaluator-1")
    envelope = seal_document({"secret": "gold"}, recipient=recipient)
    for field, value in (("plaintext_sha256", "0" * 64),
                         ("nonce", "A" * 16)):
        changed = copy.deepcopy(envelope)
        changed[field] = value
        with pytest.raises(EvidenceEncryptionError):
            open_document(changed, recipient=recipient, private_key_path=private)
    changed = copy.deepcopy(envelope)
    raw = bytearray(base64.b64decode(changed["ciphertext"]))
    raw[0] ^= 1
    changed["ciphertext"] = base64.b64encode(raw).decode("ascii")
    with pytest.raises(EvidenceEncryptionError, match="authentication"):
        open_document(changed, recipient=recipient, private_key_path=private)


def test_wrong_private_key_is_rejected(tmp_path):
    recipient = generate_private_key(tmp_path / "right.pem", key_id="evaluator-1")
    generate_private_key(tmp_path / "wrong.pem", key_id="other")
    envelope = seal_document({"secret": "gold"}, recipient=recipient)
    with pytest.raises(EvidenceEncryptionError, match="does not match"):
        open_document(
            envelope, recipient=recipient,
            private_key_path=tmp_path / "wrong.pem")


def test_key_generation_cli_emits_only_public_recipient(tmp_path, capsys):
    private = tmp_path / "evaluator.pem"
    assert main([
        "generate", "--private-key", str(private),
        "--key-id", "evaluator-1"]) == 0
    recipient = json.loads(capsys.readouterr().out)
    assert recipient["key_id"] == "evaluator-1"
    assert "PRIVATE" not in json.dumps(recipient)
    assert "PRIVATE" in private.read_text(encoding="ascii")
