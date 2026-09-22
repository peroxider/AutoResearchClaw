import json
from dataclasses import replace

import pytest

from researchclaw.adapters import AdapterBundle
from researchclaw.config import RCConfig
from researchclaw.pipeline import runner
from researchclaw.pipeline._helpers import StageResult
from researchclaw.pipeline.stages import Stage, StageStatus
from researchclaw.pipeline.stage_impls import _analysis, _review_publish


def config(root, status="submission_candidate"):
    return RCConfig.from_dict({
        "project": {"name": "quality-test"}, "research": {"topic": "test", "target_status": status},
        "runtime": {"timezone": "UTC"}, "notifications": {"channel": "local"},
        "knowledge_base": {"root": str(root / "kb")}, "llm": {"provider": "openai-compatible",
        "base_url": "http://localhost:1234/v1", "api_key": "test", "api_key_env": "RC_TEST_KEY"},
    }, project_root=root, check_paths=False)


def test_missing_citations_have_no_perfect_score_and_fail_strict(tmp_path):
    stage = tmp_path / "stage-23"
    stage.mkdir()
    result = _review_publish._execute_citation_verify(stage, tmp_path, config(tmp_path), AdapterBundle())
    report = json.loads((stage / "verification_report.json").read_text())
    assert result.status == StageStatus.FAILED
    assert report["status"] == "missing" and report["summary"]["integrity_score"] is None


def test_strict_quality_gate_does_not_degrade(tmp_path, monkeypatch):
    stage = tmp_path / "stage-20"
    stage.mkdir()
    monkeypatch.setattr(_review_publish, "_default_quality_report", lambda *a, **kw: {
        "score_1_to_10": 2, "verdict": "reject", "weaknesses": ["missing evidence"]})
    result = _review_publish._execute_quality_gate(stage, tmp_path, config(tmp_path), AdapterBundle())
    assert result.status == StageStatus.FAILED and result.decision != "degraded"


def test_retry_budget_exhaustion_pauses_and_preserves_resume(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "STAGE_SEQUENCE", (Stage.RESEARCH_DECISION, Stage.PAPER_OUTLINE))
    monkeypatch.setattr(runner, "_read_pivot_count", lambda p: runner.MAX_DECISION_PIVOTS)
    calls = []
    def execute(stage, **kwargs):
        calls.append(stage)
        return StageResult(stage=stage, status=StageStatus.DONE, artifacts=(), decision="refine")
    monkeypatch.setattr(runner, "execute_stage", execute)
    results = runner.execute_pipeline(run_dir=tmp_path, run_id="strict", config=config(tmp_path),
                                      adapters=AdapterBundle(), from_stage=Stage.RESEARCH_DECISION,
                                      auto_approve_gates=True)
    assert calls == [Stage.RESEARCH_DECISION]
    assert results[0].status == StageStatus.PAUSED
    checkpoint = json.loads((tmp_path / "checkpoint.json").read_text())
    assert checkpoint["last_completed_stage"] == int(Stage.RESULT_ANALYSIS)
    assert json.loads((tmp_path / "pipeline_summary.json").read_text())["target_met"] is False


def test_analysis_keeps_regimes_and_declared_baseline(tmp_path, monkeypatch):
    stage = tmp_path / "stage-14"
    stage.mkdir()
    cfg = config(tmp_path)
    cfg = replace(cfg, experiment=replace(cfg.experiment, comparison_baseline="Z", metric_key="accuracy"))
    metrics = {f"{method}/{regime}/{seed}/accuracy": value + seed
               for method, value in (("Z", 10), ("A", 12))
               for regime in ("easy", "hard") for seed in range(3)}
    monkeypatch.setattr(_analysis, "_collect_experiment_results", lambda *a, **kw: {
        "metrics_summary": {}, "runs": [{}], "best_run": {"metrics": metrics}, "latex_table": ""})
    _analysis._execute_result_analysis(stage, tmp_path, cfg, AdapterBundle())
    report = json.loads((stage / "experiment_summary.json").read_text())
    assert set(report["condition_summaries"]) == {"Z/easy", "Z/hard", "A/easy", "A/hard"}
    assert len(report["paired_comparisons"]) == 2
    assert all(c["baseline"] == "Z" for c in report["paired_comparisons"])


def test_no_global_citation_cap_or_hallucination_resurrection(tmp_path, monkeypatch):
    from researchclaw.literature.verify import CitationResult, VerificationReport, VerifyStatus
    stage = tmp_path / "stage-23"
    stage.mkdir()
    prior = tmp_path / "stage-22"
    prior.mkdir()
    keys = [f"paper2024x{i}" for i in range(65)]
    (prior / "references.bib").write_text("\n".join(f"@article{{{k},title={{Title}}}}" for k in keys))
    (prior / "paper_final.md").write_text(" ".join(f"[{k}]" for k in keys))
    report = VerificationReport(total=65, verified=65, results=[
        CitationResult(k, "Title", VerifyStatus.VERIFIED, 1.0, "doi") for k in keys])
    monkeypatch.setattr("researchclaw.literature.verify.verify_citations", lambda *a, **kw: report)
    _review_publish._execute_citation_verify(stage, tmp_path, config(tmp_path), AdapterBundle())
    assert (stage / "references_verified.bib").read_text().count("@article") == 65
    report.verified, report.hallucinated = 1, 64
    for entry in report.results[1:]:
        entry.status = VerifyStatus.HALLUCINATED
    result = _review_publish._execute_citation_verify(stage, tmp_path, config(tmp_path), AdapterBundle())
    assert result.status == StageStatus.FAILED
    assert (stage / "references_verified.bib").read_text().count("@article") == 1


def test_agent_requirements_exhaustion_stays_paused(tmp_path, monkeypatch):
    stage = tmp_path / "stage-15"
    stage.mkdir()
    monkeypatch.setattr(_analysis, "_read_requirements_from_manifest", lambda p: [{"id": "required"}])
    monkeypatch.setattr(_analysis, "_read_experiment_summary", lambda p: {})
    monkeypatch.setattr(_analysis, "_read_agent_results_canonical", lambda p: {})
    monkeypatch.setattr(_analysis, "_read_retry_count", lambda p: _analysis._REQUIREMENTS_MAX_RETRIES)
    monkeypatch.setattr("researchclaw.pipeline.requirements_judge.judge_requirements",
                        lambda *a, **kw: {"verdict": "reject", "per_requirement": []})
    result = _analysis._agent_requirements_decision(stage_dir=stage, run_dir=tmp_path,
                                                   config=config(tmp_path), llm=object())
    assert result.status == StageStatus.PAUSED and result.decision == "blocked"


def test_stage14_independent_evaluation_failure_invalidates_prior_store(tmp_path):
    stage = tmp_path / "stage-14"
    stage.mkdir()
    (tmp_path / "evidence_store.json").write_text('{"old":"success"}')
    (tmp_path / "trusted_evaluation.json").write_text('{"schema_version":1,"runs":[]}')
    result = _analysis._execute_result_analysis(stage, tmp_path, config(tmp_path), AdapterBundle())
    assert result.status == StageStatus.FAILED
    assert json.loads((tmp_path / "evidence_store.json").read_text()) == {}
    assert json.loads((stage / "independent_evaluation.json").read_text())["status"] == "failed"


@pytest.mark.parametrize("section,field,value", [
    ("research", "target_status", "accepted"), ("research", "quality_threshold", 11),
    ("export", "max_citations", 0), ("export", "max_citations", True),
    ("experiment", "comparison_baseline", ["Baseline"]),
])
def test_invalid_quality_configuration_is_rejected(section, field, value):
    from researchclaw.config import validate_config
    validation = validate_config({section: {field: value}}, check_paths=False)
    assert any(f"{section}.{field}" in error for error in validation.errors)
