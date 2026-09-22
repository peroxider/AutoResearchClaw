from dataclasses import asdict, replace

import pytest

from researchclaw.pipeline.evidence_store import EvidenceKey, EvidenceRecord, EvidenceStore, file_hash


@pytest.fixture
def evidence(tmp_path):
    raw = tmp_path / "predictions.csv"
    raw.write_text("id,label,prediction\n1,1,1\n", encoding="utf-8")
    key = EvidenceKey("dataset-A", "v1", "test", "Baseline", "config-hash", "42", "accuracy", "mean")
    record = EvidenceRecord(key, 0.81, "fraction", "success", "commit", "env-hash",
                            "run-42", "frozen-evaluator/v1", True, ((raw.name, file_hash(raw)),))
    store = EvidenceStore()
    result_id = store.add(record)
    return store, record, {"result_id": result_id, "key": asdict(key), "value": 81.0,
                           "unit": "percent", "decimals": 2}


@pytest.mark.parametrize("field,value", [
    ("method", "Proposed"), ("dataset", "dataset-B"), ("dataset_version", "v0"),
    ("split", "train"), ("metric", "AUROC"), ("seed", "43"),
    ("config", "different-config"), ("aggregation", "max"), ("regime", "shifted"),
])
def test_reject_wrong_attribution(evidence, tmp_path, field, value):
    store, _, claim = evidence
    claim["key"][field] = value
    assert "identity_mismatch" in store.verify_claim(claim, tmp_path)


def test_units_and_rounding_are_explicit(evidence, tmp_path):
    store, _, claim = evidence
    assert store.verify_claim(claim, tmp_path) == []
    claim["unit"] = "percentage_point"
    assert "unit_mismatch" in store.verify_claim(claim, tmp_path)
    claim.update(unit="percent", value=81.8)
    assert "value_mismatch" in store.verify_claim(claim, tmp_path)


@pytest.mark.parametrize("changes,reason", [
    ({"run_status": "failed"}, "run_not_successful"),
    ({"evaluator_independent": False}, "independent_evaluation_missing"),
    ({"artifacts": ()}, "raw_output_missing"),
    ({"code_commit": ""}, "provenance_missing"),
])
def test_failed_or_untraceable_run(evidence, tmp_path, changes, reason):
    _, record, _ = evidence
    store = EvidenceStore()
    result_id = store.add(replace(record, **changes))
    assert reason in store.validate_record(result_id, tmp_path)


def test_changed_raw_output_rejected(evidence, tmp_path):
    store, _, claim = evidence
    (tmp_path / "predictions.csv").write_text("modified", encoding="utf-8")
    assert store.verify_claim(claim, tmp_path) == ["artifact_changed:predictions.csv"]


def test_roundtrip_and_tampering(evidence):
    store, _, _ = evidence
    data = store.to_dict()
    assert EvidenceStore.from_dict(data).version == store.version
    data["records"][0]["value"] = 0.91
    with pytest.raises(ValueError, match="hash mismatch"):
        EvidenceStore.from_dict(data)


def test_conflicting_complete_identity_rejected(evidence):
    store, record, _ = evidence
    with pytest.raises(ValueError, match="Conflicting"):
        store.add(replace(record, value=0.91))


def test_derived_percentage_points_differ_from_relative_percent(evidence, tmp_path):
    store, record, claim = evidence
    proposed = store.add(replace(record, key=replace(record.key, method="Proposed"), value=0.91))
    assert store.derive(proposed, claim["result_id"], "percentage_point", tmp_path)["value"] == pytest.approx(10)
    assert store.derive(proposed, claim["result_id"], "relative_change", tmp_path)["value"] == pytest.approx(12.345679)


def test_derived_cross_version_comparison_rejected(evidence, tmp_path):
    store, record, claim = evidence
    old = store.add(replace(record, key=replace(record.key, dataset_version="v0")))
    with pytest.raises(ValueError, match="different datasets"):
        store.derive(old, claim["result_id"], "difference", tmp_path)


def test_tables_render_bound_identity_and_escape_labels(evidence, tmp_path):
    store, record, claim = evidence
    other_id = store.add(replace(record, key=replace(record.key, method="A&B_1"), value=0.91))
    markdown = store.render_table([claim["result_id"], other_id], tmp_path)
    rows = markdown.splitlines()
    assert "Baseline" in rows[2] and "0.810" in rows[2]
    assert "A&B_1" in rows[3] and "0.910" in rows[3]
    latex = store.render_table([other_id], tmp_path, latex=True)
    assert r"A\&B\_1" in latex and "0.910" in latex
