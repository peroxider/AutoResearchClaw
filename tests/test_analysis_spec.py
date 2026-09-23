import copy
import csv
import json
import sys
from dataclasses import replace

import pytest

from researchclaw.adapters import AdapterBundle
from researchclaw.experiment.protocol_runner import run_matrix
from researchclaw.literature.evidence import write_json
from researchclaw.pipeline.analysis_spec import (
    AnalysisError, DEFAULT_PLAN, analysis_figures, build_analysis, calibration_rows,
    efficiency_rows, expected_calibration_error, prepare_analysis, summarize_pairs,
    read_learning_curve, validate_plan, verify_analysis,
)
from researchclaw.pipeline.evidence_store import EvidenceStore, content_hash, file_hash
from researchclaw.pipeline.experiment_protocol import ProtocolError, load_protocol
from researchclaw.pipeline.independent_evaluator import evaluate_predictions
from researchclaw.pipeline.manuscript import build_manuscript, evidence_catalog, section_tasks, validate_manuscript
from researchclaw.pipeline.publication_assets import AssetError, chart_bytes, chart_csv, prepare_assets, verify_assets
from researchclaw.pipeline.stages import StageStatus
from tests.test_experiment_protocol import spec, freeze, evaluate_cells
from tests.test_manuscript import study, Writer
from tests.test_research_inputs import inputs, dump


def bootstrap_plan(**changes):
    plan = copy.deepcopy(DEFAULT_PLAN)
    plan["figures"] = ["paired_seed", "effect_summary"]
    plan["interval"] = {"method": "paired_seed_percentile_bootstrap", "confidence": 0.95,
                        "replicates": 1000, "random_seed": 19, "exchangeable_training_seeds": True, **changes}
    return plan


def pairs(values):
    return [{"seed": str(i), "baseline": 0.5, "candidate": 0.5 + v} for i, v in enumerate(values)]


@pytest.mark.parametrize("plan", [
    {}, {**DEFAULT_PLAN, "extra": True}, {**DEFAULT_PLAN, "schema_version": True},
    {**DEFAULT_PLAN, "figures": []}, {**DEFAULT_PLAN, "figures": ["paired_seed", "paired_seed"]},
    {**DEFAULT_PLAN, "figures": ["fabricated_learning_curve"]}, {**DEFAULT_PLAN, "interval": {"method": "paired_t"}},
    bootstrap_plan(replicates=999), bootstrap_plan(replicates=20001), bootstrap_plan(replicates=True),
    bootstrap_plan(confidence=1), bootstrap_plan(confidence=float("nan")), bootstrap_plan(confidence=True),
    bootstrap_plan(random_seed=-1), bootstrap_plan(random_seed=True), bootstrap_plan(exchangeable_training_seeds=False),
    bootstrap_plan(exchangeable_training_seeds=1),
])
def test_invalid_or_undeclared_analysis_cannot_be_admitted(plan):
    with pytest.raises(AnalysisError):
        validate_plan(plan)


def test_default_plan_is_descriptive_and_returns_an_independent_copy():
    plan = validate_plan(None)
    assert plan == DEFAULT_PLAN
    plan["figures"].append("changed")
    assert validate_plan(None) == DEFAULT_PLAN
    report = summarize_pairs(pairs([-0.5, 0, 0.5]), DEFAULT_PLAN)
    assert report["mean_difference"] == 0 and report["sd_difference"] == 0.5
    assert report["negative_pairs"] == report["zero_pairs"] == report["positive_pairs"] == 1
    assert report["interval"]["status"] == "not_requested"
    assert "p_value" not in report


@pytest.mark.parametrize("values,status", [
    ([0.1], "unavailable_insufficient_seeds"), ([0, 0], "unavailable_insufficient_seeds"),
    ([0, 0, 0], "unavailable_constant_differences"), ([-0.5] * 4, "unavailable_constant_differences"),
])
def test_insufficient_and_degenerate_data_do_not_gain_confidence(values, status):
    report = summarize_pairs(pairs(values), bootstrap_plan())
    assert report["interval"]["status"] == status
    assert report["interval"]["low"] is report["interval"]["high"] is None
    assert report["n_pairs"] == len(values)


def test_bootstrap_is_paired_reproducible_bounded_and_conditional():
    rows = pairs([-0.5, 0, 0.5])
    first = summarize_pairs(rows, bootstrap_plan())
    assert first == summarize_pairs(rows, bootstrap_plan())
    interval = first["interval"]
    assert interval["status"] == "computed_conditional"
    # For this three-point sample, each extreme mean has mass 1/27,
    # exceeding the 2.5% tail; the known percentile endpoints are +/- 0.5.
    assert (interval["low"], interval["high"]) == (-0.5, 0.5)
    assert interval["assumption_verified"] is interval["simultaneous_coverage"] is False
    # Adding the same seed-specific offset to each member of the pair cannot
    # change a paired interval; independently resampling arms would change it.
    shifted = [{**r, "baseline": r["baseline"] + 100 * i, "candidate": r["candidate"] + 100 * i} for i, r in enumerate(rows)]
    assert summarize_pairs(shifted, bootstrap_plan())["interval"] == interval


@pytest.mark.parametrize("rows", [[], pairs([0.2]) * 2, pairs([float("inf")]), pairs([float("nan")]),
    [{"seed": "0", "baseline": -1e308, "candidate": 1e308}],
    [{"seed": "0", "baseline": True, "candidate": 1.0}],
    [{"seed": "0", "baseline": "0.5", "candidate": 1.0}], [{"seed": "", "baseline": 0, "candidate": 1}], [{}]])
def test_invalid_samples_fail_without_silent_filtering(rows):
    with pytest.raises(AnalysisError):
        summarize_pairs(rows, bootstrap_plan())


def test_plan_is_frozen_before_data_and_budget_is_checked_before_execution(inputs, spec):
    spec["questions"][0]["analysis_plan"] = bootstrap_plan()
    root, _, protocol, _ = freeze(inputs, spec)
    assert protocol["spec"]["questions"][0]["analysis_plan"] == bootstrap_plan()
    path = inputs[0].parent / "protocol.yaml"
    changed = copy.deepcopy(spec)
    changed["questions"][0]["analysis_plan"]["interval"]["confidence"] = 0.9
    dump(path, changed)
    # Published bundles depend on the frozen snapshot, while resuming against
    # changed original input must fail instead of changing the analysis plan.
    assert load_protocol(root) == protocol
    from researchclaw.research_inputs import prepare_inputs
    with pytest.raises(ValueError):
        prepare_inputs(inputs[0], root)


def test_excessive_bootstrap_work_rejected_when_freezing(inputs, spec):
    spec["seeds"] = list(range(251))
    spec["budget"]["max_total_seconds"] = 20000
    spec["questions"][0]["analysis_plan"] = bootstrap_plan(replicates=20000)
    with pytest.raises(ValueError) as failure:
        freeze(inputs, spec)
    assert isinstance(failure.value.__cause__, ProtocolError)
    assert "resampling draw budget" in str(failure.value.__cause__)


def test_default_report_binds_every_pair_and_exact_versions(study):
    root, _ = study
    report = prepare_analysis(root)
    assert verify_analysis(root) == report
    store = EvidenceStore.from_dict(json.loads((root / "evidence_store.json").read_text()))
    assert report["evidence_version"] == store.version
    assert {rid for a in report["analyses"] for rid in a["result_ids"]} == set(store.records)
    assert all(a["plan_origin"] == "pipeline_descriptive_default" for a in report["analyses"])
    assert all(a["statistics"]["mean_difference"] == 0 for a in report["analyses"])


@pytest.mark.parametrize("mutation", ["mean", "source", "plan", "scope", "drop", "unit"])
def test_rehashing_analysis_cannot_forge_results_or_inference_scope(study, mutation):
    root, _ = study
    report = prepare_analysis(root)
    row = report["analyses"][0]
    if mutation == "mean":
        row["statistics"]["mean_difference"] = 1.0
    elif mutation == "source":
        row["result_ids"].reverse()
    elif mutation == "plan":
        row["plan"] = bootstrap_plan()
    elif mutation == "scope":
        row["limitations"] = []
        row["statistics"]["interval"]["assumption_verified"] = True
    elif mutation == "unit":
        row["unit"] = "percent"
    else:
        report["analyses"].pop()
    report.pop("version")
    report["version"] = content_hash(report)
    write_json(root / "analysis_spec.json", report)
    with pytest.raises(AnalysisError):
        verify_analysis(root)


def test_incomplete_matrix_cannot_be_analyzed_and_stale_raw_data_fails(study):
    root, _ = study
    prepare_analysis(root)
    store = EvidenceStore.from_dict(json.loads((root / "evidence_store.json").read_text()))
    (root / next(iter(store.records.values())).artifacts[1][0]).write_text("Changed raw observations")
    with pytest.raises(AnalysisError, match="incomplete protocol"):
        verify_analysis(root)


def test_formal_stage_emits_analysis_without_calling_legacy_analysis_model(inputs, spec):
    from researchclaw.pipeline.stage_impls._analysis import _execute_result_analysis
    root, contract, protocol, cfg = freeze(inputs, spec)
    write_json(root / "evidence_store.json", evaluate_cells(root, contract, protocol).to_dict())
    stage = root / "stage-14"
    stage.mkdir()
    class ForbiddenModel:
        def __getattr__(self, name):
            raise AssertionError("Formal analysis must use frozen AnalysisSpec")
    result = _execute_result_analysis(stage, root, cfg, AdapterBundle(), llm=ForbiddenModel())
    assert result.status == StageStatus.DONE
    assert (stage / "analysis_spec.json").read_bytes() == (root / "analysis_spec.json").read_bytes()
    summary = json.loads((stage / "experiment_summary.json").read_text())
    assert summary["analysis_spec_version"] == verify_analysis(root)["version"]
    assert "paired_comparisons" not in summary  # No undeclared legacy t-tests.


def test_analysis_is_in_every_matching_results_packet_and_invalidates_manuscript(study):
    root, _ = study
    report = build_manuscript(root, "Fixture", llm=Writer())
    catalog = evidence_catalog(root)
    for task in section_tasks(catalog):
        if task["role"] == "results":
            assert len([key for key in task["evidence_ids"] if key.startswith("analysis:")]) == 1
    path = root / "analysis_spec.json"
    document = json.loads(path.read_text())
    document["analyses"][0]["statistics"]["mean_difference"] = 1
    write_json(path, document)
    with pytest.raises(ValueError):
        validate_manuscript(root, report)


def test_final_acceptance_reports_missing_analysis_with_a_specific_repair_owner(study):
    from researchclaw.pipeline.final_acceptance import assess_delivery
    from researchclaw.pipeline.manuscript import export_manuscript
    root, _ = study
    build_manuscript(root, "Fixture", llm=Writer())
    export_manuscript(root, root)
    (root / "analysis_spec.json").unlink()
    report = assess_delivery(root, target_status="submission_candidate")
    assert any(i["reason"] == "invalid_missing_or_stale_analysis_spec" and i["repair_owner"] == "analysis"
               for i in report["issues"])
    assert not report["target_met"]


def varied_study(inputs, spec):
    spec["seeds"] = [7, 19, 42]
    spec["budget"]["max_total_seconds"] = 200
    for question in spec["questions"]:
        question["analysis_plan"] = bootstrap_plan()
    root, contract, protocol, cfg = freeze(inputs, spec)
    store = evaluate_cells(root, contract, protocol)
    changed = EvidenceStore()
    for record in store.records.values():
        if record.key.method != "A_full":
            changed.add(record)
            continue
        labels, predictions, execution = [name for name, _ in record.artifacts]
        with (root / labels).open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        incorrect = {"7": 0, "19": len(rows) // 2, "42": len(rows)}[record.key.seed]
        with (root / predictions).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["id", "prediction"])
            writer.writeheader()
            writer.writerows({"id": row["id"], "prediction": 1 - float(row["label"]) if i < incorrect else row["label"]}
                             for i, row in enumerate(rows))
        changed.add(evaluate_predictions(root=root, key=record.key, labels=labels, predictions=predictions,
                    execution=execution, expected_labels_sha256=file_hash(root / labels)))
    write_json(root / "evidence_store.json", changed.to_dict())
    return root, cfg


def test_effect_plot_preserves_negative_null_and_positive_observations_and_all_ids(inputs, spec):
    root, _ = varied_study(inputs, spec)
    assets = prepare_assets(root)
    assert verify_assets(root) == assets
    analysis = verify_analysis(root)
    assert all(a["plan_origin"] == "predeclared" for a in analysis["analyses"])
    assert all(a["statistics"]["interval"]["status"] == "computed_conditional" for a in analysis["analyses"])
    figures = [f for f in assets["spec"]["figures"] if f["kind"] == "effect_summary"]
    assert len(figures) == 2
    assert figures[0]["comparisons"][0]["statistics"]["mean_difference"] < 0
    assert figures[1]["comparisons"][0]["statistics"]["mean_difference"] > 0
    for figure in figures:
        source_csv = chart_csv(figure)
        for item in figure["comparisons"]:
            for row in item["pairs"]:
                assert row["baseline_result"] in source_csv and row["candidate_result"] in source_csv
        assert chart_bytes(figure, "png").startswith(b"\x89PNG")
        pdf_bytes = chart_bytes(figure, "pdf")
        assert pdf_bytes == chart_bytes(figure, "pdf")
        import fitz
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            page = doc[0]
            assert page.search_for("(fraction)")[0].y1 < page.search_for("Matched seed differences")[0].y0
            assert page.search_for("Observed paired mean")[0].y1 < page.search_for("One frozen test split")[0].y0
    (root / "analysis_spec.json").unlink()
    with pytest.raises(AssetError, match="AnalysisSpec"):
        verify_assets(root)


def test_effect_figures_split_large_groups_without_sorting_or_dropping_comparisons(study):
    root, _ = study
    original = prepare_analysis(root)["analyses"][0]
    analyses = []
    for i in range(13):
        item = copy.deepcopy(original)
        item.update(candidate=f"method_{i}", analysis_id=f"analysis_{i}")
        item["plan"]["figures"] = ["effect_summary"]
        item["statistics"]["mean_difference"] = (-1) ** i * i
        analyses.append(item)
    figures = analysis_figures({"analyses": analyses})
    assert len(figures) == 3
    assert [c["candidate"] for f in figures for c in f["comparisons"]] == [f"method_{i}" for i in range(13)]


def test_effect_figures_budget_long_labels_without_clipping_or_dropping(study):
    root, _ = study
    original = prepare_analysis(root)["analyses"][0]
    analyses = []
    for i in range(7):
        item = copy.deepcopy(original)
        item.update(candidate="Candidate_" + str(i) * 54, baseline="Baseline_" + "b" * 55, analysis_id=str(i))
        item["plan"]["figures"] = ["effect_summary"]
        analyses.append(item)
    figures = analysis_figures({"analyses": analyses})
    assert len(figures) >= 3
    assert [c["analysis_id"] for f in figures for c in f["comparisons"]] == [str(i) for i in range(7)]
    from researchclaw.pipeline.analysis_spec import effect_label_weight
    for figure in figures:
        assert sum(effect_label_weight(c) for c in figure["comparisons"]) <= 22
        assert chart_bytes(figure, "pdf").startswith(b"%PDF-")


def test_efficiency_plan_requires_declared_direction_and_rejects_stray_fields():
    assert validate_plan({**DEFAULT_PLAN, "figures": ["calibration"]})["figures"] == ["calibration"]
    pareto = {**DEFAULT_PLAN, "figures": ["paired_seed", "efficiency_pareto"], "metric_direction": "maximize"}
    assert validate_plan(pareto) == pareto
    for bad in ({**DEFAULT_PLAN, "figures": ["efficiency_pareto"]},
                {**DEFAULT_PLAN, "figures": ["paired_seed"], "metric_direction": "maximize"},
                {**DEFAULT_PLAN, "figures": ["efficiency_pareto"], "metric_direction": "up"},
                {**DEFAULT_PLAN, "figures": ["efficiency_pareto"], "metric_direction": True},
                {**DEFAULT_PLAN, "figures": ["paired_seed", "fabricated_learning_curve"]},
                {**DEFAULT_PLAN, "figures": ["calibration"], "metric_direction": "minimize"}):
        with pytest.raises(AnalysisError):
            validate_plan(bad)


def learning_plan(**changes):
    declaration = {"metric": "validation_loss", "split": "validation",
                   "direction": "minimize", "max_points": 20}
    declaration.update(changes)
    return {**DEFAULT_PLAN, "figures": ["paired_seed", "learning_curve"],
            "learning_curve": declaration}


def structured_plan(method, **changes):
    interval = {"method": method, "confidence": 0.95, "replicates": 1000, "random_seed": 23,
                "resampling_unit": "group" if method == "cluster_percentile_bootstrap" else "time_point"}
    if method == "moving_block_percentile_bootstrap":
        interval.update(block_length=2, chronological_order_preserved=True)
    interval.update(changes)
    return {**DEFAULT_PLAN, "figures": ["paired_seed", "effect_summary"], "interval": interval}


@pytest.mark.parametrize("plan", [
    {**DEFAULT_PLAN, "figures": ["learning_curve"]},
    {**DEFAULT_PLAN, "figures": ["paired_seed"], "learning_curve": learning_plan()["learning_curve"]},
    learning_plan(metric="bad metric"), learning_plan(split="test"), learning_plan(direction="up"),
    learning_plan(max_points=1), learning_plan(max_points=True), learning_plan(extra=True),
])
def test_learning_curve_plan_is_exact_and_predeclared(plan):
    with pytest.raises(AnalysisError):
        validate_plan(plan)
    assert validate_plan(learning_plan()) == learning_plan()


@pytest.mark.parametrize("content", [
    "step,value\n0,1\n0,0.5\n", "step,value\n0,1\n2,nan\n",
    "epoch,value\n0,1\n1,0.5\n", "step,value\n00,1\n1,0.5\n",
    "step,value\n0,1\n", "step,value,extra\n0,1,x\n1,0.5,x\n",
])
def test_learning_curve_reader_rejects_malformed_or_cherry_picked_points(tmp_path, content):
    path = tmp_path / "curve.csv"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(AnalysisError):
        read_learning_curve(path, learning_plan()["learning_curve"])


LEARNING_SCRIPT = '''import csv, json, os
from collections import Counter
from pathlib import Path
request = json.loads(os.environ["ARC_PROTOCOL_REQUEST"])
Path(request["tuning"]["output"]).write_text(json.dumps({"schema_version": 1,
    "metric": request["tuning"]["metric"], "trials": [], "selected_trial_id": None}))
data = request["dataset"]
train = list(csv.DictReader(Path(data["paths"]["train"]).open()))
test = list(csv.DictReader(Path(data["paths"]["test_features"]).open()))
label = Counter(row[data["label_column"]] for row in train).most_common(1)[0][0]
with Path(request["output"]).open("w", newline="") as stream:
    writer = csv.writer(stream); writer.writerow(["id", "prediction"])
    writer.writerows([row[data["id_column"]], data["label_encoding"][label]] for row in test)
if "learning_curve" in request:
    assert request["learning_curve"]["columns"] == ["step", "value"]
    offset = 0.1 if request["key"]["method"] == "A_full" else 0.0
    with Path(request["learning_curve"]["output"]).open("w", newline="") as stream:
        writer = csv.writer(stream); writer.writerow(["step", "value"])
        writer.writerows((step, 1.0 / (step + 1) + offset) for step in (0, 1, 2))
'''


def test_learning_curve_is_host_archived_recomputed_rendered_and_tamper_evident(inputs, spec):
    from tests.test_protocol_runner import setup
    spec["questions"][0]["analysis_plan"] = learning_plan()
    root, project, protocol, cfg = setup(inputs, spec, LEARNING_SCRIPT)
    assert run_matrix(root, project, cfg.experiment)["status"] == "complete"
    report = prepare_analysis(root)
    curves = [item for item in report["analyses"] if "learning_curve" in item]
    assert len(curves) == 1 and len(curves[0]["learning_curve"]["rows"]) == 6
    assert {row["step"] for row in curves[0]["learning_curve"]["rows"]} == {0, 1, 2}
    figure = next(item for item in analysis_figures(report) if item["kind"] == "learning_curve")
    assert "step,baseline,candidate,difference" in chart_csv(figure)
    assert chart_bytes(figure, "png").startswith(b"\x89PNG")
    assert chart_bytes(figure, "pdf").startswith(b"%PDF")
    assets = prepare_assets(root)
    assert verify_assets(root) == assets
    receipts = list((root / "evidence_artifacts/protocol_runs").glob("*/execution.json"))
    with_curve = [json.loads(path.read_text()) for path in receipts if "learning_curve" in json.loads(path.read_text())]
    assert len(with_curve) == 4  # two methods x two seeds for rq_main only
    curve_path = root / with_curve[0]["learning_curve"]
    curve_path.write_text(curve_path.read_text().replace("1.0", "9.0", 1), encoding="utf-8")
    with pytest.raises(ProtocolError, match="telemetry changed"):
        from researchclaw.experiment.protocol_runner import verify_execution_bundle
        verify_execution_bundle(root, protocol)


@pytest.mark.parametrize("method,manifest_changes,expected_unit", [
    ("cluster_percentile_bootstrap", {"group_column": "person", "split": {"strategy": "group"}}, "test_group"),
    ("moving_block_percentile_bootstrap", {"time_column": "time", "split": {"strategy": "time"}},
     "time_point_circular_block"),
])
def test_structured_bootstrap_uses_private_frozen_test_units(inputs, spec, method, manifest_changes, expected_unit):
    from tests.test_protocol_runner import setup, SCRIPT
    manifest = inputs[2]
    manifest.update(manifest_changes)
    dump(inputs[0].parent / "manifest.json", manifest)
    spec["questions"][0]["analysis_plan"] = structured_plan(method)
    root, project, protocol, cfg = setup(inputs, spec, SCRIPT)
    assert run_matrix(root, project, cfg.experiment)["status"] == "complete"
    report = prepare_analysis(root)
    analyzed = next(item for item in report["analyses"] if item["question"] == "rq_main")
    interval = analyzed["statistics"]["interval"]
    assert interval["status"] == "computed_conditional" and interval["resampling_unit"] == expected_unit
    assert interval["unit_count"] >= 3 and interval["sample_count"] > interval["unit_count"]
    assert interval["low"] == interval["high"] == 0
    assert interval["assumption_verified"] is interval["simultaneous_coverage"] is False
    contract = json.loads((root / "research_contract.json").read_text())
    units = contract["datasets"][0]["analysis_units"]
    assert file_hash(root / units) == contract["outputs"][units]
    assert not any(path.name == "test_analysis_units.csv" for path in project.rglob("*"))
    before = verify_analysis(root)
    assert before == report
    (root / units).write_text((root / units).read_text() + "tamper", encoding="utf-8")
    with pytest.raises(ValueError):
        verify_analysis(root)


@pytest.mark.parametrize("plan", [
    structured_plan("cluster_percentile_bootstrap", resampling_unit="seed"),
    structured_plan("moving_block_percentile_bootstrap", block_length=1),
    structured_plan("moving_block_percentile_bootstrap", chronological_order_preserved=False),
])
def test_structured_bootstrap_plan_rejects_wrong_unit_or_block(plan):
    with pytest.raises(AnalysisError):
        validate_plan(plan)


def test_structured_bootstrap_requires_matching_dataset_structure(inputs, spec):
    spec["questions"][0]["analysis_plan"] = structured_plan("cluster_percentile_bootstrap")
    with pytest.raises(ValueError) as failure:
        freeze(inputs, spec)
    assert "group_column" in str(failure.value.__cause__)


def test_expected_calibration_error_matches_known_bin_values():
    scores = {str(i): 0.9 for i in range(10)}
    labels = {str(i): (1.0 if i < 5 else 0.0) for i in range(10)}
    assert abs(expected_calibration_error(scores, labels) - 0.4) < 1e-12
    # Perfectly calibrated bin and empty bins contribute nothing.
    assert expected_calibration_error({"a": 0.5, "b": 0.5}, {"a": 1.0, "b": 0.0}) == 0.0
    assert expected_calibration_error({}, {}) == 0.0


SCORE_SCRIPT = '''import csv, json, os
from pathlib import Path
request = json.loads(os.environ["ARC_PROTOCOL_REQUEST"])
assert request["phase"] == "frozen_test"
Path(request["tuning"]["output"]).write_text(json.dumps({"schema_version": 1,
    "metric": request["tuning"]["metric"], "trials": [], "selected_trial_id": None}))
data = request["dataset"]
test = list(csv.DictReader(Path(data["paths"]["test_features"]).open()))
assert data["label_column"] not in test[0]
if request["key"]["method"] == "Z_base":
    scores = {row[data["id_column"]]: 0.5 for row in test}
else:
    # Deliberately overconfident and inverted: high scores on label-0 rows.
    scores = {row[data["id_column"]]: (0.99 if int(round(float(row["feature"]) - 0.125)) % 2 == 0 else 0.01)
              for row in test}
with Path(request["output"]).open("w", newline="") as stream:
    writer = csv.writer(stream)
    writer.writerow(["id", "prediction"])
    writer.writerows(scores.items())
'''


@pytest.fixture
def score_study(inputs, spec):
    """A real host-executed frozen study whose predictions are scores."""
    # AUROC accepts continuous scores against the binary frozen labels.
    manifest = inputs[2]
    manifest["metric"] = "auroc"
    dump(inputs[0].parent / "manifest.json", manifest)
    spec["seeds"] = [7, 19]
    spec["questions"][0]["analysis_plan"] = {
        **DEFAULT_PLAN, "figures": ["paired_seed", "calibration", "efficiency_pareto"], "metric_direction": "maximize"}
    root, contract, protocol, cfg = freeze(inputs, spec)
    cfg = replace(cfg, experiment=replace(cfg.experiment,
                  sandbox=replace(cfg.experiment.sandbox, python_path=sys.executable)))
    project = root / "stage-10" / "experiment"
    project.mkdir(parents=True)
    (project / "main.py").write_text(SCORE_SCRIPT, encoding="utf-8")
    report = run_matrix(root, project, cfg.experiment)
    assert report["status"] == "complete"
    return root


def test_calibration_and_efficiency_data_are_frozen_and_recomputed(score_study):
    report = prepare_analysis(score_study)
    assert verify_analysis(score_study) == report
    analyzed = [a for a in report["analyses"] if "calibration" in a["plan"]["figures"]]
    assert len(analyzed) == 1  # Only the question with a predeclared plan requests figures.
    analysis = analyzed[0]
    calibration = analysis["calibration"]
    assert calibration["bins"] == 10 and len(calibration["rows"]) == 2
    for row in calibration["rows"]:
        assert row["test_samples"] > 0
        assert abs(row["difference"] - (row["candidate_ece"] - row["baseline_ece"])) < 1e-12
        # Constant-0.5 baseline scores are perfectly calibrated on the
        # balanced split; the inverted overconfident candidate is not.
        assert row["baseline_ece"] == 0.0 and row["candidate_ece"] > 0.9
    efficiency = analysis["efficiency"]
    assert efficiency["metric_direction"] == "maximize" and len(efficiency["rows"]) == 2
    assert all(row["baseline_seconds"] > 0 and row["candidate_seconds"] > 0
               for row in efficiency["rows"])


def test_requested_figures_project_frozen_data_into_assets(score_study):
    report = prepare_analysis(score_study)
    figures = analysis_figures(report)
    calibration = next(f for f in figures if f["kind"] == "calibration")
    assert calibration["bins"] == 10 and calibration["rows"]
    csv_text = chart_csv(calibration)
    assert "baseline_ece,candidate_ece,difference,test_samples" in csv_text
    png = chart_bytes(calibration, "png")
    assert png.startswith(b"\x89PNG") and png == chart_bytes(calibration, "png")
    pareto = next(f for f in figures if f["kind"] == "efficiency_pareto")
    assert pareto["metric_direction"] == "maximize" and set(pareto["means"]) == {"baseline", "candidate"}
    assert "baseline_seconds" in chart_csv(pareto)
    pdf = chart_bytes(pareto, "pdf")
    assert pdf.startswith(b"%PDF") and pdf == chart_bytes(pareto, "pdf")
    assets = prepare_assets(score_study)
    assert verify_assets(score_study) == assets
    assert [f["kind"] for f in assets["spec"]["figures"] if f["kind"] in ("calibration", "efficiency_pareto")] \
        == ["calibration", "efficiency_pareto"]


def test_calibration_fails_closed_without_frozen_or_consistent_scores(score_study):
    (score_study / "trusted_evaluation.json").unlink()
    with pytest.raises(AnalysisError, match="trusted_evaluation"):
        build_analysis(score_study)
    write_json(score_study / "trusted_evaluation.json", {"schema_version": 1, "runs": []})
    with pytest.raises(AnalysisError, match="does not bind"):
        build_analysis(score_study)  # No manifest run binds the analyzed records.


def test_calibration_rejects_out_of_range_or_non_binary_frozen_data(score_study):
    analysis = prepare_analysis(score_study)["analyses"][0]
    rows = analysis["pairs"]
    store = EvidenceStore.from_dict(json.loads((score_study / "evidence_store.json").read_text()))
    record = store.records[rows[0]["candidate_result"]]
    predictions = score_study / record.artifacts[1][0]
    original = predictions.read_text()
    # Out-of-range scores with a consistent stored digest reach the range check.
    predictions.write_text(original.replace("0.99", "1.5"), encoding="utf-8")
    store.records[rows[0]["candidate_result"]] = replace(
        record, artifacts=record.artifacts[:1]
        + ((record.artifacts[1][0], file_hash(predictions)),) + record.artifacts[2:])
    with pytest.raises(AnalysisError, match="probabilistic scores"):
        calibration_rows(score_study, store, rows)
    # Non-binary labels with a consistent digest must also fail closed; both
    # records share the labels file, so both digests move together.
    predictions.write_text(original, encoding="utf-8")
    store = EvidenceStore.from_dict(json.loads((score_study / "evidence_store.json").read_text()))
    labels = score_study / store.records[rows[0]["candidate_result"]].artifacts[0][0]
    labels.write_text(labels.read_text().replace(",1", ",0.5"), encoding="utf-8")
    digest = file_hash(labels)
    for name in ("baseline_result", "candidate_result"):
        shared = store.records[rows[0][name]]
        store.records[rows[0][name]] = replace(
            shared, artifacts=((shared.artifacts[0][0], digest),) + shared.artifacts[1:])
    with pytest.raises(AnalysisError, match="binary labels"):
        calibration_rows(score_study, store, rows)


def test_tampered_frozen_predictions_reject_calibration(score_study):
    analysis = prepare_analysis(score_study)["analyses"][0]
    rows = analysis["pairs"]
    store = EvidenceStore.from_dict(json.loads((score_study / "evidence_store.json").read_text()))
    record = store.records[rows[0]["candidate_result"]]
    predictions = score_study / record.artifacts[1][0]
    document = predictions.read_text()
    predictions.write_text(document.replace("0.99", "0.42", 1), encoding="utf-8")
    with pytest.raises(AnalysisError, match="changed"):
        calibration_rows(score_study, store, rows)
    # Through the full pipeline the same tampering is caught even earlier, by
    # the coverage audit's artifact digest validation.
    with pytest.raises(AnalysisError, match="incomplete protocol"):
        build_analysis(score_study)


def test_efficiency_fails_closed_without_complete_matching_budget(score_study):
    analysis = prepare_analysis(score_study)["analyses"][0]
    rows = analysis["pairs"]
    store = EvidenceStore.from_dict(json.loads((score_study / "evidence_store.json").read_text()))
    protocol = load_protocol(score_study)
    budget = json.loads((score_study / "protocol_budget.json").read_text())
    assert efficiency_rows(score_study, protocol, store, rows, "maximize")["rows"]
    # A missing budget cannot support the figure; the full pipeline fails
    # closed even earlier, at the coverage audit.
    (score_study / "protocol_budget.json").unlink()
    with pytest.raises(AnalysisError, match="protocol budget record"):
        efficiency_rows(score_study, protocol, store, rows, "maximize")
    with pytest.raises(AnalysisError, match="incomplete protocol"):
        build_analysis(score_study)
    write_json(score_study / "protocol_budget.json", budget)
    for mutation, pattern in (({"status": "incomplete"}, "complete protocol budget"),
                              ({"schema_version": 2}, "complete protocol budget"),
                              ({"protocol_version": "changed"}, "same protocol version")):
        write_json(score_study / "protocol_budget.json", {**budget, **mutation})
        with pytest.raises(AnalysisError, match=pattern):
            efficiency_rows(score_study, protocol, store, rows, "maximize")
    first_cell = next(iter(budget["per_cell_seconds"]))
    write_json(score_study / "protocol_budget.json",
               {**budget, "per_cell_seconds": {**budget["per_cell_seconds"], first_cell: True}})
    with pytest.raises(AnalysisError, match="finite seconds"):
        efficiency_rows(score_study, protocol, store, rows, "maximize")
