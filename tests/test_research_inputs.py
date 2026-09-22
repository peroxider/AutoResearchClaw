import csv
import json
from dataclasses import replace
from pathlib import Path

import pytest

from researchclaw.pipeline.evidence_store import EvidenceKey, file_hash
from researchclaw.research_inputs import (
    DatasetManifest, InputContractError, ResearchBrief, SplitSpec, check_evaluation_binding,
    contract_for_config, prepare_inputs, public_context, split_rows, verify_bundle_contract, verify_inputs,
)


def dump(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def inputs(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    with (source / "records.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["row", "feature", "target", "forbidden", "person", "time"])
        writer.writeheader()
        writer.writerows({"row": f"r{i:03d}", "feature": i + 0.125, "target": i % 2,
                          "forbidden": i % 2, "person": f"p{i // 4}",
                          "time": f"2026-01-{i // 4 + 1:02d}"} for i in range(60))
    manifest = {"dataset": "demo", "version": "v1", "path": "records.csv", "task": "classification",
                "id_column": "row", "label_column": "target", "features": ["feature"], "metric": "accuracy",
                "source": "local study", "license": "private research", "forbidden_features": ["forbidden"],
                "split": {"strategy": "stratified", "seed": 42}}
    dump(source / "manifest.json", manifest)
    brief = {"question": "Does the proposed method help?", "ideas": ["Compare a simple baseline"],
             "constraints": ["No outside data"], "datasets": ["manifest.json"]}
    dump(source / "brief.json", brief)
    return source / "brief.json", tmp_path / "run", manifest, brief


def test_prepare_freezes_files_and_only_profiles_training(inputs):
    brief, root, _, _ = inputs
    contract = prepare_inputs(brief, root)
    assert verify_inputs(brief, root) == contract
    assert prepare_inputs(brief, root) == contract  # Resume does not reshuffle.
    card = contract["datasets"][0]["card"]
    assert card["split_sizes"] == {"train": 36, "validation": 12, "test": 12}
    assert card["training_profile"]["scope"] == "train_only"
    assert card["label_encoding"] == {"0": 0, "1": 1}
    test_rows = list(csv.DictReader((root / card["paths"]["test_features"]).open(encoding="utf-8")))
    assert all(set(row) == {"row", "feature"} for row in test_rows)
    context = public_context(contract, root)
    assert "test_labels.csv" not in context and "records.csv" not in context
    assert "No outside data" in context and "label_encoding" in context
    assert "forbidden" not in list(csv.DictReader((root / card["paths"]["train"]).open()).fieldnames)


@pytest.mark.parametrize("name", ["records.csv", "manifest.json", "brief.json"])
def test_source_mutation_invalidates_frozen_contract(inputs, name):
    brief, root, _, _ = inputs
    prepare_inputs(brief, root)
    with (brief.parent / name).open("a", encoding="utf-8") as stream:
        stream.write(" ")
    with pytest.raises(InputContractError, match="Frozen source changed"):
        verify_inputs(brief, root)


@pytest.mark.parametrize("name", ["research_inputs/demo/split_ids.json", "research_inputs/demo/train.csv",
                                  "research_inputs/demo/test_features.csv", "evidence_artifacts/demo/test_labels.csv"])
def test_prepared_data_mutation_rejected(inputs, name):
    brief, root, _, _ = inputs
    prepare_inputs(brief, root)
    with (root / name).open("a", encoding="utf-8") as stream:
        stream.write("changed")
    with pytest.raises(InputContractError, match="Prepared dataset changed"):
        prepare_inputs(brief, root)


@pytest.mark.parametrize("change", [
    {"features": ["target"]}, {"features": ["forbidden"]}, {"features": ["row"]},
    {"dataset": "../escape"}, {"dataset": "NUL"}, {"metric": "mse"},
    {"split": {"strategy": "group"}}, {"split": {"strategy": "time"}},
    {"group_column": "person"}, {"typo": "value"}, {"features": "feature"},
    {"split": {"train_fraction": 0.9, "validation_fraction": 0.2}},
    {"split": {"train_fraction": float("nan")}},
])
def test_manifest_rejects_invalid_or_leaky_schema(inputs, change):
    _, _, manifest, _ = inputs
    with pytest.raises(InputContractError):
        DatasetManifest.from_dict({**manifest, **change})


@pytest.mark.parametrize("change", [{"ideas": []}, {"datasets": []}, {"allow_external_data": "false"},
                                   {"max_experiment_seconds": 0}, {"unknown": 1}])
def test_brief_is_strict(inputs, change):
    _, _, _, brief = inputs
    with pytest.raises(InputContractError):
        ResearchBrief.from_dict({**brief, **change})


def test_group_split_keeps_entities_together(inputs):
    brief, _, manifest, _ = inputs
    manifest = DatasetManifest.from_dict({**manifest, "group_column": "person", "split": {"strategy": "group"}})
    rows = list(csv.DictReader((brief.parent / "records.csv").open()))
    parts = split_rows(rows, manifest)
    groups = [set(row["person"] for row in part) for part in parts.values()]
    assert all(not groups[i] & groups[j] for i in range(3) for j in range(i + 1, 3))
    assert sum(map(len, parts.values())) == len(rows)
    assert split_rows(list(reversed(rows)), manifest) == parts


def test_time_split_is_chronological_and_keeps_timestamp_ties(inputs):
    brief, _, manifest, _ = inputs
    manifest = DatasetManifest.from_dict({**manifest, "time_column": "time", "split": {"strategy": "time"}})
    rows = list(csv.DictReader((brief.parent / "records.csv").open()))
    parts = split_rows(rows, manifest)
    assert max(r["time"] for r in parts["train"]) < min(r["time"] for r in parts["validation"])
    assert max(r["time"] for r in parts["validation"]) < min(r["time"] for r in parts["test"])


def test_time_split_rejects_cross_partition_entities(inputs):
    brief, _, manifest, _ = inputs
    manifest = DatasetManifest.from_dict({**manifest, "group_column": "person", "time_column": "time",
                                          "split": {"strategy": "time"}})
    rows = list(csv.DictReader((brief.parent / "records.csv").open()))
    rows[-1]["person"] = rows[0]["person"]
    with pytest.raises(InputContractError, match="Entity overlap"):
        split_rows(rows, manifest)


def test_raw_label_copy_is_rejected_before_writing(inputs):
    brief, root, manifest, _ = inputs
    manifest["features"] = ["forbidden"]
    manifest["forbidden_features"] = []
    dump(brief.parent / "manifest.json", manifest)
    # Add a varying feature so duplicate-row checks do not mask the label-copy check.
    manifest["features"].append("feature")
    dump(brief.parent / "manifest.json", manifest)
    with pytest.raises(InputContractError, match="copies the target"):
        prepare_inputs(brief, root)
    assert not (root / "research_contract.json").exists()


def test_duplicate_ids_and_malformed_rows_are_rejected(inputs):
    brief, root, _, _ = inputs
    source = brief.parent / "records.csv"
    lines = source.read_text().splitlines()
    source.write_text("\n".join(lines + [lines[1]]))
    with pytest.raises(InputContractError, match="duplicate row ID"):
        prepare_inputs(brief, root)


def test_portable_bundle_validation_does_not_need_original_dataset(inputs):
    brief, root, _, _ = inputs
    contract = prepare_inputs(brief, root)
    (brief.parent / "records.csv").rename(brief.parent / "renamed.csv")
    assert verify_bundle_contract(root) == contract
    with pytest.raises(InputContractError):
        verify_inputs(brief, root)


def test_evaluator_is_bound_to_frozen_test_labels(inputs):
    brief, root, _, _ = inputs
    contract = prepare_inputs(brief, root)
    key = EvidenceKey("demo", "v1", "test", "Baseline", "config", "42", "accuracy", "per_run")
    labels = contract["datasets"][0]["labels"]
    digest = file_hash(root / labels)
    check_evaluation_binding(contract, key, labels, digest)
    for wrong_key in (replace(key, dataset="other"), replace(key, dataset_version="v0"),
                      replace(key, metric="auroc"), replace(key, split="validation")):
        with pytest.raises(InputContractError):
            check_evaluation_binding(contract, wrong_key, labels, digest)
    with pytest.raises(InputContractError):
        check_evaluation_binding(contract, key, "other_labels.csv", digest)


def make_config(root, brief):
    from tests.test_submission_gates import config
    cfg = config(root, "exploratory")
    return replace(cfg, research=replace(cfg.research, brief_path=str(brief)),
                   experiment=replace(cfg.experiment, metric_direction="maximize"))


def test_runtime_changes_or_removed_brief_cannot_bypass_contract(inputs):
    brief, root, _, _ = inputs
    cfg = make_config(root, brief)
    contract_for_config(cfg, root, initialize=True)
    changed = replace(cfg, experiment=replace(cfg.experiment, metric_key="auroc"))
    with pytest.raises(InputContractError, match="Runtime constraints changed"):
        contract_for_config(changed, root)
    with pytest.raises(InputContractError, match="Cannot remove"):
        contract_for_config(replace(cfg, research=replace(cfg.research, brief_path="")), root)


def test_resume_without_initialization_is_blocked(inputs):
    brief, root, _, _ = inputs
    with pytest.raises(InputContractError):
        contract_for_config(make_config(root, brief), root)


def test_budget_is_not_silently_increased(inputs):
    brief, root, _, _ = inputs
    cfg = make_config(root, brief)
    cfg = replace(cfg, experiment=replace(cfg.experiment, time_budget_sec=600))
    with pytest.raises(InputContractError, match="budget exceeds"):
        contract_for_config(cfg, root, initialize=True)


def test_runner_stops_if_stage_modifies_frozen_data(inputs, monkeypatch):
    from researchclaw.adapters import AdapterBundle
    from researchclaw.pipeline import runner
    from researchclaw.pipeline._helpers import StageResult
    from researchclaw.pipeline.stages import Stage, StageStatus
    brief, root, _, _ = inputs
    cfg = make_config(root, brief)
    contract_for_config(cfg, root, initialize=True)
    monkeypatch.setattr(runner, "STAGE_SEQUENCE", (Stage.CODE_GENERATION, Stage.RESOURCE_PLANNING))
    calls = []
    def execute(stage, **kwargs):
        calls.append(stage)
        with (root / "research_inputs/demo/train.csv").open("a") as stream:
            stream.write("changed")
        return StageResult(stage=stage, status=StageStatus.DONE, artifacts=())
    monkeypatch.setattr(runner, "execute_stage", execute)
    results = runner.execute_pipeline(run_dir=root, run_id="input-test", config=cfg,
                                      adapters=AdapterBundle(), from_stage=Stage.CODE_GENERATION,
                                      skip_noncritical=True)
    assert calls == [Stage.CODE_GENERATION]
    assert results[0].status == StageStatus.FAILED
    assert results[0].decision == "input_contract_invalid"
    assert json.loads((root / "data_preflight.json").read_text())["status"] == "failed"


def test_final_acceptance_rechecks_prepared_data(inputs):
    from researchclaw.pipeline.final_acceptance import assess_delivery
    brief, root, _, _ = inputs
    contract = prepare_inputs(brief, root)
    dump(root / "data_preflight.json", {"status": "verified", "contract_version": contract["version"]})
    assert assess_delivery(root)["dimensions"]["data"] == "unknown"  # Preflight is not a full research review.
    (root / "research_inputs/demo/test_features.csv").write_text("changed")
    report = assess_delivery(root)
    assert report["dimensions"]["data"] == "failed"
    assert any(i["reason"] == "invalid_or_changed_data_contract" for i in report["issues"])


def test_independent_evaluator_rejects_relabelled_test_as_validation(inputs):
    from researchclaw.pipeline.independent_evaluator import evaluate_predictions
    brief, root, _, _ = inputs
    contract = prepare_inputs(brief, root)
    label_name = contract["datasets"][0]["labels"]
    labels = list(csv.DictReader((root / label_name).open()))
    predictions = root / "evidence_artifacts/demo/predictions.csv"
    predictions.write_text("id,prediction\n" + "\n".join(f"{r['id']},{r['label']}" for r in labels))
    execution = root / "evidence_artifacts/demo/execution.json"
    dump(execution, {"status": "success", "returncode": 0, "code_commit": "commit", "environment": "env"})
    key = EvidenceKey("demo", "v1", "test", "Baseline", "config", "42", "accuracy", "per_run")
    kwargs = dict(root=root, key=key, labels=label_name, expected_labels_sha256=file_hash(root / label_name),
                  predictions=predictions.relative_to(root).as_posix(), execution=execution.relative_to(root).as_posix())
    assert evaluate_predictions(**kwargs).value == 1.0
    with pytest.raises(InputContractError, match="frozen test split"):
        evaluate_predictions(**{**kwargs, "key": replace(key, split="validation")})


def test_regression_random_split_produces_numeric_frozen_labels(inputs):
    brief, root, manifest, _ = inputs
    manifest.update(task="regression", metric="mse", split={"strategy": "random"})
    dump(brief.parent / "manifest.json", manifest)
    contract = prepare_inputs(brief, root)
    card = contract["datasets"][0]["card"]
    assert card["label_encoding"] == {}
    assert "mean" in card["training_profile"]["labels"]


def test_config_resolves_brief_path_from_project_root(inputs):
    from researchclaw.config import RCConfig
    brief, root, _, _ = inputs
    cfg = make_config(root, brief)
    data = json.loads(json.dumps(cfg.to_dict()))
    data["research"]["brief_path"] = "brief.json"
    loaded = RCConfig.from_dict(data, project_root=brief.parent, check_paths=False)
    assert Path(loaded.research.brief_path) == brief.resolve()


def test_real_sandbox_receives_public_data_but_no_test_labels(inputs):
    import sys
    from researchclaw.config import SandboxConfig
    from researchclaw.experiment.sandbox import ExperimentSandbox
    brief, root, _, _ = inputs
    prepare_inputs(brief, root)
    project = root / "stage-10" / "experiment"
    project.mkdir(parents=True)
    (project / "main.py").write_text(
        "import csv\nfrom pathlib import Path\n"
        "data = Path('research_data/demo')\n"
        "assert not (data / 'test_labels.csv').exists()\n"
        "train = list(csv.DictReader((data / 'train.csv').open()))\n"
        "test = list(csv.DictReader((data / 'test_features.csv').open()))\n"
        "assert len(train) == 36 and len(test) == 12\n"
        "assert 'target' in train[0] and 'target' not in test[0]\n"
        "assert 'forbidden' not in train[0]\n"
        "print('primary_metric: 1.0')\n", encoding="utf-8")
    sandbox = ExperimentSandbox(SandboxConfig(python_path=sys.executable), root / "stage-12" / "sandbox")
    result = sandbox.run_project(project, timeout_sec=20)
    assert result.returncode == 0, result.stderr
    assert result.metrics["primary_metric"] == 1.0


def test_staging_does_not_accept_extra_dataset_files(inputs):
    from researchclaw.research_inputs import bind_project_data
    brief, root, _, _ = inputs
    prepare_inputs(brief, root)
    project = root / "stage-10" / "experiment"
    data_dir = project / "research_data/demo"
    data_dir.mkdir(parents=True)
    (data_dir / "test_labels.csv").write_text("unauthorized labels")
    with pytest.raises(InputContractError, match="Unexpected file"):
        bind_project_data(project)


def test_equal_observations_are_reported_not_called_invalid(inputs):
    brief, root, _, _ = inputs
    source = brief.parent / "records.csv"
    rows = list(csv.DictReader(source.open()))
    # Distinct IDs with equal feature/label values may be genuine subjects.
    rows[2]["feature"] = rows[0]["feature"]
    with source.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    card = prepare_inputs(brief, root)["datasets"][0]["card"]
    assert card["duplicate_feature_label_rows"] == 1 and card["warnings"]


def test_unseen_holdout_class_is_not_silently_encoded(inputs):
    brief, root, manifest, _ = inputs
    manifest["split"] = {"strategy": "random"}
    dump(brief.parent / "manifest.json", manifest)
    source = brief.parent / "records.csv"
    rows = list(csv.DictReader(source.open()))
    heldout = split_rows(rows, DatasetManifest.from_dict(manifest))["test"][0]["row"]
    for row in rows:
        if row["row"] == heldout:
            row["target"] = "new-class"
    with source.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(InputContractError, match="cover all label classes"):
        prepare_inputs(brief, root)


@pytest.mark.parametrize("corruption", ["extra_column", "missing_label", "duplicate_header"])
def test_csv_schema_failures(inputs, corruption):
    brief, root, _, _ = inputs
    path = brief.parent / "records.csv"
    lines = path.read_text().splitlines()
    if corruption == "extra_column":
        lines[1] += ",extra"
    elif corruption == "missing_label":
        fields = lines[1].split(",")
        fields[2] = ""
        lines[1] = ",".join(fields)
    else:
        lines[0] = lines[0].replace("forbidden", "feature")
    path.write_text("\n".join(lines))
    with pytest.raises(InputContractError):
        prepare_inputs(brief, root)


def test_rehashed_malformed_contract_is_still_rejected(inputs):
    from researchclaw.pipeline.evidence_store import content_hash
    brief, root, _, _ = inputs
    contract = prepare_inputs(brief, root)
    contract.pop("version")
    contract["outputs"] = ["not a mapping"]
    contract["version"] = content_hash(contract)
    dump(root / "research_contract.json", contract)
    with pytest.raises(InputContractError, match="Invalid contract outputs"):
        verify_bundle_contract(root)
