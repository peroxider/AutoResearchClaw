from __future__ import annotations

import copy
import json

import pytest

from researchclaw.pipeline.evidence_signature import (
    EvidenceSignatureError, generate_private_key, main, sign_document,
    validate_signer, verify_document,
)


def test_ed25519_document_signature_round_trip_and_tamper(tmp_path):
    private = tmp_path / "private.pem"
    signer = generate_private_key(private, key_id="runner-1")
    document = {"schema_version": 1, "cases": [{"case_id": "a"}]}
    signed = dict(document)
    signed["signature"] = sign_document(
        document, signer=signer, private_key_path=private,
        purpose="benchmark_result_manifest/v1")
    assert verify_document(
        signed, signer=signer,
        purpose="benchmark_result_manifest/v1") == signed
    changed = copy.deepcopy(signed)
    changed["cases"][0]["case_id"] = "b"
    with pytest.raises(EvidenceSignatureError, match="envelope"):
        verify_document(
            changed, signer=signer,
            purpose="benchmark_result_manifest/v1")


def test_wrong_private_key_and_identity_fail_closed(tmp_path):
    first = generate_private_key(tmp_path / "first.pem", key_id="runner-1")
    generate_private_key(tmp_path / "second.pem", key_id="runner-2")
    with pytest.raises(EvidenceSignatureError, match="does not match"):
        sign_document(
            {"value": 1}, signer=first,
            private_key_path=tmp_path / "second.pem",
            purpose="benchmark_result_manifest/v1")
    malformed = dict(first, public_key="A" * 44)
    with pytest.raises(EvidenceSignatureError, match="public key"):
        validate_signer(malformed)


def test_key_generation_refuses_overwrite_and_private_material_is_not_public(
        tmp_path):
    private = tmp_path / "private.pem"
    signer = generate_private_key(private, key_id="assessor-1")
    assert set(signer) == {"algorithm", "key_id", "public_key"}
    assert "PRIVATE" in private.read_text(encoding="ascii")
    assert "PRIVATE" not in json.dumps(signer)
    with pytest.raises(EvidenceSignatureError, match="already exists"):
        generate_private_key(private, key_id="assessor-1")


def test_sign_cli_refuses_to_overwrite_any_input(tmp_path):
    private = tmp_path / "private.pem"
    signer = generate_private_key(private, key_id="assessor-1")
    signer_path = tmp_path / "signer.json"
    input_path = tmp_path / "assessment.json"
    signer_path.write_text(json.dumps(signer), encoding="utf-8")
    input_path.write_text(json.dumps({"schema_version": 1}), encoding="utf-8")
    with pytest.raises(EvidenceSignatureError, match="cannot overwrite"):
        main([
            "sign-json", "--input", str(input_path), "--output", str(input_path),
            "--private-key", str(private), "--signer", str(signer_path),
            "--purpose", "benchmark_private_assessment/v1",
        ])


@pytest.mark.parametrize("key_id", ["", "../escape", "x" * 65])
def test_key_id_is_bounded_and_safe(tmp_path, key_id):
    with pytest.raises(EvidenceSignatureError, match="key ID"):
        generate_private_key(tmp_path / "private.pem", key_id=key_id)
