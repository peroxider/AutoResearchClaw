import json

import pytest

from researchclaw.pipeline.evidence_store import EvidenceKey, file_hash
from researchclaw.pipeline.independent_evaluator import evaluate_predictions


@pytest.fixture
def evaluation(tmp_path):
    (tmp_path / "labels.csv").write_text("id,label\na,0\nb,1\nc,1\nd,0\n", encoding="utf-8")
    (tmp_path / "predictions.csv").write_text("id,prediction\nd,0\nc,0\nb,1\na,0\n", encoding="utf-8")
    (tmp_path / "run.json").write_text(json.dumps({"status": "success", "returncode": 0,
                                                  "code_commit": "commit", "environment": "env"}), encoding="utf-8")
    return dict(root=tmp_path, key=EvidenceKey("d", "v1", "test", "m", "c", "1", "accuracy", "per_run"),
                labels="labels.csv", predictions="predictions.csv", execution="run.json",
                expected_labels_sha256=file_hash(tmp_path / "labels.csv"))


def test_recompute_accuracy_from_predictions_not_reported_number(evaluation):
    record = evaluate_predictions(**evaluation)
    assert record.value == 0.75 and record.unit == "fraction"
    assert record.evaluator_independent and record.evaluator.startswith("host_predictions/v1:")
    assert len(record.artifacts) == 3


@pytest.mark.parametrize("predictions,expected", [([0.1, 0.8, 0.9, 0.2], 1.0),
                                                            ([0.5] * 4, 0.5),
                                                            ([0.9, 0.2, 0.1, 0.8], 0.0)])
def test_auroc_with_ties_and_reversed_ranking(evaluation, predictions, expected):
    from dataclasses import replace
    evaluation["key"] = replace(evaluation["key"], metric="auroc")
    (evaluation["root"] / "predictions.csv").write_text(
        "id,prediction\n" + "\n".join(f"{i},{p}" for i, p in zip("abcd", predictions)), encoding="utf-8")
    assert evaluate_predictions(**evaluation).value == expected


def test_label_mutation_rejected(evaluation):
    (evaluation["root"] / "labels.csv").write_text("id,label\na,1", encoding="utf-8")
    with pytest.raises(ValueError, match="Frozen labels changed"):
        evaluate_predictions(**evaluation)


@pytest.mark.parametrize("rows", ["a,0\na,0", "a,0", "a,nan\nb,1\nc,0\nd,0"])
def test_invalid_predictions_rejected(evaluation, rows):
    (evaluation["root"] / "predictions.csv").write_text("id,prediction\n" + rows, encoding="utf-8")
    with pytest.raises(ValueError):
        evaluate_predictions(**evaluation)


def test_failed_execution_cannot_be_promoted(evaluation):
    (evaluation["root"] / "run.json").write_text('{"status":"failed","returncode":1}', encoding="utf-8")
    with pytest.raises(ValueError, match="failed or incomplete"):
        evaluate_predictions(**evaluation)
