import copy
import json
from dataclasses import replace

import pytest
import yaml

from researchclaw.adapters import AdapterBundle
from researchclaw.pipeline.evidence_store import EvidenceKey, EvidenceStore, content_hash, file_hash
from researchclaw.pipeline.experiment_protocol import (
    ProtocolError, audit_coverage, compile_protocol, get_reproduction_policy,
    load_protocol,
)
from researchclaw.pipeline.independent_evaluator import evaluate_predictions
from researchclaw.pipeline.stages import StageStatus
from researchclaw.research_inputs import InputContractError, prepare_inputs, public_context, verify_inputs
from tests.test_research_inputs import inputs, dump, make_config


@pytest.fixture
def spec():
    return {"schema_version": 1, "seeds": [42, 7],
            "budget": {"per_cell_seconds": 10, "max_total_seconds": 100, "tuning_trials": 2},
            "methods": [
                {"id": "Z_base", "role": "baseline", "description": "fixed baseline", "parameters": {"depth": 2}},
                {"id": "A_full", "role": "proposed", "description": "new component", "parameters": {"component": True}},
                {"id": "A_off", "role": "ablation", "description": "disable component", "parameters": {"component": False},
                 "parent": "A_full", "disabled_components": ["component"]}],
            "questions": [
                {"id": "rq_main", "kind": "main", "question": "Does A improve on Z?",
                 "datasets": ["demo"], "methods": ["Z_base", "A_full"], "baseline": "Z_base",
                 "analysis": "Compare matched seeds; report uncertainty and null results."},
                {"id": "rq_ablation", "kind": "ablation", "question": "Does component contribute?",
                 "datasets": ["demo"], "methods": ["A_full", "A_off"], "baseline": "A_full",
                 "analysis": "Report component effect without treating zero difference as an error."}]}


def freeze(inputs, spec):
    brief, root, _, brief_data = inputs
    brief_data["protocol_path"] = "protocol.yaml"
    dump(brief, brief_data)
    (brief.parent / "protocol.yaml").write_text(yaml.safe_dump(spec), encoding="utf-8")
    cfg = make_config(root, brief)
    from researchclaw.research_inputs import contract_for_config
    contract = contract_for_config(cfg, root, initialize=True)
    return root, contract, load_protocol(root), cfg


def evaluate_cells(root, contract, protocol):
    store = EvidenceStore()
    labels = contract["datasets"][0]["labels"]
    for cell in protocol["cells"]:
        cid = cell["cell_id"]
        predictions = f"evidence_artifacts/{cid}.csv"
        (root / predictions).write_text((root / labels).read_text().replace("id,label", "id,prediction"))
        execution = f"evidence_artifacts/{cid}.json"
        dump(root / execution, {"status": "success", "returncode": 0, "code_commit": "fixture-code",
                               "environment": {"python": "fixture"}, "protocol_version": protocol["version"],
                               "cell_id": cid, "key": cell["key"]})
        store.add(evaluate_predictions(root=root, key=EvidenceKey(**cell["key"]), labels=labels,
                  predictions=predictions, execution=execution, expected_labels_sha256=file_hash(root / labels)))
    return store


def test_protocol_freezes_full_product_and_explicit_baselines(inputs, spec):
    root, contract, protocol, _ = freeze(inputs, spec)
    assert protocol["allocation"]["required_cells"] == 8
    assert protocol["allocation"]["reserved_seconds"] == 80
    assert protocol["allocation"]["actual_usage_status"] == "unmeasured"
    assert "reproduction" not in protocol["spec"]
    assert get_reproduction_policy(protocol) == {"mode": "exact", "metrics": {}}
    assert len({c["cell_id"] for c in protocol["cells"]}) == 8
    assert len(protocol["comparisons"]) == 4
    index = {c["cell_id"]: c for c in protocol["cells"]}
    for comparison in protocol["comparisons"]:
        base, candidate = index[comparison["baseline_cell"]], index[comparison["candidate_cell"]]
        assert base["budget"] == candidate["budget"]
        assert base["key"]["seed"] == candidate["key"]["seed"]
        assert base["split_ids_sha256"] == candidate["split_ids_sha256"]
        assert base["key"]["method"] == ("Z_base" if comparison["question"] == "rq_main" else "A_full")
    assert prepare_inputs(inputs[0], root) == contract
    assert "Frozen experiment protocol" in public_context(contract, root)


def test_protocol_freezes_bounded_reproduction_tolerance_before_execution(inputs, spec):
    spec["reproduction"] = {"metrics": {"accuracy": {
        "absolute_tolerance": 0.001,
        "relative_tolerance": 0,
        "rationale": "Allow documented floating-point variation across accelerators.",
    }}}
    _, _, protocol, _ = freeze(inputs, spec)
    policy = get_reproduction_policy(protocol)
    assert policy["mode"] == "tolerance"
    assert policy["metrics"]["accuracy"]["absolute_tolerance"] == 0.001


@pytest.mark.parametrize("policy", [
    {"metrics": {}},
    {"metrics": {"accuracy": {"absolute_tolerance": 0, "relative_tolerance": 0,
                                "rationale": "Both zero"}}},
    {"metrics": {"accuracy": {"absolute_tolerance": True, "relative_tolerance": 0,
                                "rationale": "Boolean masquerade"}}},
    {"metrics": {"accuracy": {"absolute_tolerance": float("nan"), "relative_tolerance": 0,
                                "rationale": "Non-finite"}}},
    {"metrics": {"accuracy": {"absolute_tolerance": 0, "relative_tolerance": 1.01,
                                "rationale": "Unbounded relative tolerance"}}},
    {"metrics": {"accuracy": {"absolute_tolerance": -0.1, "relative_tolerance": 0,
                                "rationale": "Negative absolute tolerance"}}},
    {"metrics": {"accuracy": {"absolute_tolerance": 1_000_001, "relative_tolerance": 0,
                                "rationale": "Unbounded absolute tolerance"}}},
    {"metrics": {"accuracy": {"absolute_tolerance": 0.1, "relative_tolerance": 0,
                                "rationale": ""}}},
    {"metrics": {"accuracy": {"absolute_tolerance": 0.1, "relative_tolerance": 0,
                                "rationale": "x" * 1001}}},
    {"metrics": {"accuracy": {"absolute_tolerance": 0.1, "relative_tolerance": 0,
                                "rationale": "valid"},
                 "mse": {"absolute_tolerance": 0.1, "relative_tolerance": 0,
                         "rationale": "undeclared metric"}}},
])
def test_invalid_reproduction_policy_is_rejected_before_execution(inputs, spec, policy):
    contract = prepare_inputs(inputs[0], inputs[1])
    spec["reproduction"] = policy
    with pytest.raises(ProtocolError, match="Reproduction|reproduction"):
        compile_protocol(spec, contract["datasets"], max_cell_seconds=300)


@pytest.mark.parametrize("mutation", [
    lambda s: s.update(seeds=[42, 42]), lambda s: s.update(seeds=[True]),
    lambda s: s.update(seeds=[]), lambda s: s.update(extra="typo"),
    lambda s: s["budget"].update(max_total_seconds=79),
    lambda s: s["budget"].update(per_cell_seconds=301),
    lambda s: s["budget"].update(tuning_trials=0),
    lambda s: s["budget"].update(tuning_trials=1.5),
    lambda s: s["methods"].append(copy.deepcopy(s["methods"][0])),
    lambda s: s["methods"][0].update(id="../escape"),
    lambda s: s["methods"][0].update(parameters={"x": float("nan")}),
    lambda s: s["methods"][2].update(parameters={"component": True}),
    lambda s: s["methods"][2].update(parent="Z_base"),
    lambda s: s["methods"][2].update(disabled_components=[]),
    lambda s: s["questions"][0].update(baseline="A_full"),
    lambda s: s["questions"][0].update(datasets=["other"]),
    lambda s: s["questions"][0].update(methods=["Z_base"]),
    lambda s: s["questions"][0].update(analysis=""),
    lambda s: s["questions"][1].update(id="rq_main"),
    lambda s: s["questions"].pop(),
])
def test_invalid_matrix_is_rejected_before_any_output(inputs, spec, mutation):
    # Obtain valid dataset descriptors without freezing this protocol.
    contract = prepare_inputs(inputs[0], inputs[1])
    mutation(spec)
    with pytest.raises(ProtocolError):
        compile_protocol(spec, contract["datasets"], max_cell_seconds=300)


@pytest.mark.parametrize("target", ["source", "matrix", "deleted"])
def test_protocol_changes_invalidate_resume(inputs, spec, target):
    root, _, _, _ = freeze(inputs, spec)
    path = inputs[0].parent / "protocol.yaml" if target == "source" else root / "experiment_protocol.json"
    if target == "deleted":
        path.unlink()
    else:
        path.write_text(path.read_text() + "\n ")
    with pytest.raises(InputContractError):
        verify_inputs(inputs[0], root)


def test_rehashed_shortened_matrix_is_recompiled(inputs, spec):
    root, _, protocol, _ = freeze(inputs, spec)
    protocol["cells"].pop()
    protocol.pop("version")
    protocol["version"] = content_hash(protocol)
    dump(root / "experiment_protocol.json", protocol)
    contract = json.loads((root / "research_contract.json").read_text())
    contract["outputs"]["experiment_protocol.json"] = file_hash(root / "experiment_protocol.json")
    contract.pop("version")
    contract["version"] = content_hash(contract)
    dump(root / "research_contract.json", contract)
    with pytest.raises(ProtocolError, match="matrix or version changed"):
        load_protocol(root)


def test_zero_improvement_complete_matrix_passes_coverage(inputs, spec):
    root, contract, protocol, _ = freeze(inputs, spec)
    store = evaluate_cells(root, contract, protocol)
    report = audit_coverage(root, protocol, store)
    assert report["status"] == "complete" and report["completed_count"] == 8
    # All fixture methods have identical values; no effect-size heuristic rejects them.
    assert {record.value for record in store.records.values()} == {1.0}
    assert len(report["comparisons"]) == 2
    assert all(c["status"] == "complete" and c["mean_difference"] == 0
               and len(c["pairs"]) == 2 for c in report["comparisons"])


def test_missing_seed_wrong_regime_and_failed_execution_do_not_complete_matrix(inputs, spec):
    root, contract, protocol, _ = freeze(inputs, spec)
    full = evaluate_cells(root, contract, protocol)
    records = list(full.records.values())
    store = EvidenceStore()
    for record in records[1:]:
        store.add(record)
    store.add(replace(records[0], key=replace(records[0].key, regime="wrong_question")))
    result = audit_coverage(root, protocol, store)
    assert result["status"] == "incomplete" and len(result["missing"]) == len(result["unexpected"]) == 1
    assert result["comparisons"][0]["mean_difference"] is None
    failed = EvidenceStore()
    for record in records:
        failed.add(replace(record, run_status="failed"))
    assert len(audit_coverage(root, protocol, failed)["invalid"]) == 8


@pytest.mark.parametrize("change", [{"method": "invented"}, {"seed": "99"}, {"config": "changed"},
                                    {"regime": "rq_ablation"}])
def test_evaluator_refuses_relabeling_execution(inputs, spec, change):
    root, contract, protocol, _ = freeze(inputs, spec)
    evaluate_cells(root, contract, protocol)
    cell = protocol["cells"][0]
    key = replace(EvidenceKey(**cell["key"]), **change)
    labels = contract["datasets"][0]["labels"]
    with pytest.raises(ProtocolError):
        evaluate_predictions(root=root, key=key, labels=labels,
            predictions=f"evidence_artifacts/{cell['cell_id']}.csv",
            execution=f"evidence_artifacts/{cell['cell_id']}.json", expected_labels_sha256=file_hash(root / labels))


def test_audit_rejects_bound_key_without_execution_receipt(inputs, spec):
    root, contract, protocol, _ = freeze(inputs, spec)
    store = evaluate_cells(root, contract, protocol)
    changed = EvidenceStore()
    for record in store.records.values():
        changed.add(replace(record, artifacts=tuple((n, h) for n, h in record.artifacts if h != record.execution)))
    assert len(audit_coverage(root, protocol, changed)["invalid"]) == 8


def test_stage9_preserves_more_than_legacy_condition_limit(inputs, spec):
    from researchclaw.pipeline.stage_impls._experiment_design import _execute_experiment_design
    for i in range(8):
        mid = f"base_{i}"
        spec["methods"].append({"id": mid, "role": "baseline", "description": "extra baseline", "parameters": {}})
        spec["questions"][0]["methods"].append(mid)
    spec["budget"]["max_total_seconds"] = 300
    root, _, protocol, cfg = freeze(inputs, spec)
    stage = root / "stage-09"
    stage.mkdir()
    result = _execute_experiment_design(stage, root, cfg, AdapterBundle())
    assert result.status == StageStatus.DONE
    plan = yaml.safe_load((stage / "exp_plan.yaml").read_text(encoding="utf-8"))
    assert len(plan["baselines"]) == 9
    assert len(plan["required_experiments"]) == 24
    assert plan["experiment_protocol_version"] == protocol["version"]
    assert plan["datasets"] == ["demo"]
    assert plan["metrics"] == ["accuracy"]
    assert plan["objectives"] == [q["question"] for q in spec["questions"]]


def test_stage14_fails_missing_matrix_without_evaluation_manifest(inputs, spec):
    from researchclaw.pipeline.stage_impls._analysis import _execute_result_analysis
    root, _, _, cfg = freeze(inputs, spec)
    stage = root / "stage-14"
    stage.mkdir()
    cfg = replace(cfg, research=replace(cfg.research, target_status="submission_candidate"))
    result = _execute_result_analysis(stage, root, cfg, AdapterBundle())
    assert result.status == StageStatus.FAILED and result.decision == "experiment_matrix_incomplete"
    coverage = json.loads((root / "experiment_coverage.json").read_text())
    assert len(coverage["missing"]) == 8


def test_final_acceptance_recomputes_instead_of_trusting_coverage_report(inputs, spec):
    from researchclaw.pipeline.final_acceptance import assess_delivery
    root, contract, protocol, _ = freeze(inputs, spec)
    store = evaluate_cells(root, contract, protocol)
    store.records.pop(next(iter(store.records)))
    dump(root / "evidence_store.json", store.to_dict())
    dump(root / "experiment_coverage.json", {"status": "complete"})
    report = assess_delivery(root, target_status="submission_candidate")
    assert report["dimensions"]["experiments"] == "failed"
    assert any(i["reason"] == "frozen_experiment_matrix_incomplete" for i in report["issues"])


def test_mismatched_units_cannot_produce_a_comparison(inputs, spec):
    root, contract, protocol, _ = freeze(inputs, spec)
    store = evaluate_cells(root, contract, protocol)
    changed = EvidenceStore()
    for record in store.records.values():
        changed.add(replace(record, unit="percent", value=100.0) if record.key.method == "Z_base" else record)
    report = audit_coverage(root, protocol, changed)
    assert report["status"] == "incomplete"
    assert report["comparisons"][0]["status"] == "incomplete"
    assert report["comparisons"][0]["mean_difference"] is None


def test_stage14_emits_bound_comparisons_in_summary(inputs, spec):
    from researchclaw.pipeline.stage_impls._analysis import _execute_result_analysis
    root, contract, protocol, cfg = freeze(inputs, spec)
    store = evaluate_cells(root, contract, protocol)
    dump(root / "evidence_store.json", store.to_dict())
    stage = root / "stage-14"
    stage.mkdir()
    result = _execute_result_analysis(stage, root, cfg, AdapterBundle())
    assert result.status == StageStatus.DONE
    summary = json.loads((stage / "experiment_summary.json").read_text())
    coverage = summary["protocol_coverage"]
    assert coverage["status"] == "complete"
    assert coverage["comparisons"][0]["baseline"] == "Z_base"
    assert coverage["comparisons"][1]["baseline"] == "A_full"


def test_multiple_datasets_are_separate_comparison_units(inputs, spec):
    brief, _, manifest, brief_data = inputs
    dump(brief.parent / "second.json", {**manifest, "dataset": "second", "version": "v2"})
    brief_data["datasets"].append("second.json")
    for question in spec["questions"]:
        question["datasets"].append("second")
    spec["budget"]["max_total_seconds"] = 200
    root, contract, protocol, _ = freeze(inputs, spec)
    assert len(protocol["cells"]) == 16
    # Empty evidence still reports each dataset/RQ comparison separately.
    report = audit_coverage(root, protocol, EvidenceStore())
    assert len(report["comparisons"]) == 4
    assert {c["dataset"] for c in report["comparisons"]} == {"demo", "second"}
    assert all(c["expected_pairs"] == 2 and c["mean_difference"] is None for c in report["comparisons"])
