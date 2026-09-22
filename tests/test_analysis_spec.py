import copy
import csv
import json

import pytest

from researchclaw.adapters import AdapterBundle
from researchclaw.literature.evidence import write_json
from researchclaw.pipeline.analysis_spec import (
    AnalysisError, DEFAULT_PLAN, analysis_figures, build_analysis, prepare_analysis,
    summarize_pairs, validate_plan, verify_analysis,
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
