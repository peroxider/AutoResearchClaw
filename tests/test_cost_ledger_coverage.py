"""Round 66: full-pipeline cost ledger coverage.

Every outbound model-call path the pipeline drives outside the registered
LLM clients must land in a run-level ledger: PRM judge votes, the MetaClaw
session-end signal, embedding requests, Nano Banana generation attempts and
the generated medical-audit experiment's API attempts. Beast-Mode
subprocess invocations are declared (their internals stay uncounted), the
resource ledger aggregates per-stage wall clock, and final acceptance
cross-checks Nano Banana results against the image call ledger.
"""
from __future__ import annotations

import base64
import io
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import pytest

from researchclaw.llm.call_ledger import (
    build_call_ledger,
    collect_call_records,
    record_raw_chat_call,
    reset_call_ledger,
    validate_call_ledger,
    write_call_ledger,
)
from researchclaw.llm.image_call_ledger import (
    build_image_call_ledger,
    collect_image_call_records,
    reset_image_call_ledger,
)
from researchclaw.pipeline.resource_ledger import (
    build_resource_ledger,
    ledger_issues,
    validate_resource_ledger,
)


@pytest.fixture(autouse=True)
def _clean_ledgers():
    reset_call_ledger()
    reset_image_call_ledger()
    yield
    reset_call_ledger()
    reset_image_call_ledger()


class _FakeResponse:
    def __init__(self, payload: bytes):
        self._payload = payload

    def read(self) -> bytes:
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _raw_chat_record(**overrides):
    record = {
        "status": "succeeded", "requested_chain": ["m"],
        "served_model": "served-m", "endpoint": "http://stub/v1/x",
        "adapter": "raw_urllib", "call_family": "chat",
        "purpose": "test", "max_tokens": 8, "temperature": 0.0,
        "json_mode": False, "prompt_tokens": 1, "completion_tokens": 2,
        "total_tokens": 3, "finish_reason": "", "truncated": False,
        "fallback_failures": [], "error_type": None,
        "duration_seconds": 0.01,
    }
    record.update(overrides)
    return record


# ---------------------------------------------------------------------------
# Raw chat-call records
# ---------------------------------------------------------------------------


def test_raw_records_flow_into_the_frozen_chat_ledger(tmp_path):
    record_raw_chat_call(_raw_chat_record())
    record_raw_chat_call(_raw_chat_record(status="failed",
                                          error_type="URLError",
                                          served_model=None,
                                          prompt_tokens=None,
                                          completion_tokens=None,
                                          total_tokens=None))
    frozen = write_call_ledger(tmp_path)
    assert frozen is not None and len(frozen["calls"]) == 2
    assert frozen["totals"]["calls"] == 2
    assert frozen["totals"]["failures"] == 1
    assert frozen["totals"]["total_tokens"] == 3
    validate_call_ledger(frozen)
    assert collect_call_records() == []


def test_raw_chat_record_validation_fails_closed():
    for malformed in (
        _raw_chat_record(status="pending"),
        _raw_chat_record(fallback_failures="no"),
        _raw_chat_record(total_tokens=True),
        _raw_chat_record(call_family="images"),
        _raw_chat_record(call_family=None),
    ):
        with pytest.raises(ValueError, match="malformed"):
            record_raw_chat_call(malformed)


def test_records_without_family_count_as_chat_in_the_resource_ledger(tmp_path):
    record_raw_chat_call(_raw_chat_record())
    record_raw_chat_call(_raw_chat_record(call_family="embeddings",
                                          purpose="embedding",
                                          completion_tokens=None,
                                          total_tokens=1))
    frozen = write_call_ledger(tmp_path)
    ledger = build_resource_ledger(tmp_path)
    rows = {row["kind"]: row for row in ledger["rows"]}
    assert rows["llm_chat"]["calls"] == 1
    assert rows["llm_chat"]["tokens"] == 3
    assert rows["llm_embeddings"]["calls"] == 1
    assert rows["llm_embeddings"]["tokens"] == 1
    assert ledger_issues(ledger) == []
    validate_resource_ledger(tmp_path, ledger)
    assert frozen is not None


# ---------------------------------------------------------------------------
# PRM judge gate
# ---------------------------------------------------------------------------


def _patch_urlopen(monkeypatch, results):
    """Patch urllib.request.urlopen to pop scripted results.

    A result may be a dict (served payload) or an Exception (raised)."""
    scripted = list(results)
    seen: list[object] = []

    def fake_urlopen(req, timeout=None):
        seen.append(req)
        outcome = scripted.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return _FakeResponse(json.dumps(outcome).encode("utf-8"))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    return seen


def test_prm_judge_votes_are_ledgered_per_attempt(monkeypatch):
    from researchclaw.metaclaw_bridge.prm_gate import ResearchPRMGate

    _patch_urlopen(monkeypatch, [
        {"model": "served-judge",
         "choices": [{"message": {"content": "Score: 1\nok"}}],
         "usage": {"prompt_tokens": 11, "completion_tokens": 2,
                   "total_tokens": 13}},
        {"model": "served-judge",
         "choices": [{"message": {"content": "Score: 1\nfine"}}]},
        {"model": "served-judge",
         "choices": [{"message": {"content": "no score here"}}]},
    ] * 1)
    gate = ResearchPRMGate(api_base="http://stub", api_key="k", model="m",
                           votes=3)
    score = gate.evaluate_stage(5, "output text")
    assert score == 1.0  # two parseable votes agree
    records = [record for record in collect_call_records()
               if record.get("purpose") == "prm_judge"]
    assert len(records) == 3
    assert all(record["status"] == "succeeded" for record in records)
    assert all(record["served_model"] == "served-judge" for record in records)
    assert {record["total_tokens"] for record in records} == {13, None}
    assert all(record["endpoint"] == "http://stub/chat/completions"
               for record in records)
    assert all(record["call_family"] == "chat" for record in records)
    frozen = build_call_ledger(collect_call_records())
    assert frozen["totals"]["calls"] == 3


def test_failed_prm_judge_calls_are_ledgered_as_failures(monkeypatch):
    from researchclaw.metaclaw_bridge.prm_gate import ResearchPRMGate

    _patch_urlopen(monkeypatch, [urllib.error.URLError("down")] * 2)
    gate = ResearchPRMGate(api_base="http://stub", api_key="k", model="m",
                           votes=2)
    assert gate.evaluate_stage(9, "text") == 0.0
    records = [record for record in collect_call_records()
               if record.get("purpose") == "prm_judge"]
    assert len(records) == 2
    assert all(record["status"] == "failed" for record in records)
    assert all(record["error_type"] == "URLError" for record in records)
    assert all(record["served_model"] is None for record in records)


# ---------------------------------------------------------------------------
# MetaClaw session-end signal
# ---------------------------------------------------------------------------


def _post_pipeline_config():
    return SimpleNamespace(
        metaclaw_bridge=SimpleNamespace(
            enabled=True, proxy_url="http://proxy:30000",
            lesson_to_skill=None, use_memory=False, use_message=False,
            skills_dir="", prm=None),
    )


def test_session_end_signal_is_ledgered(monkeypatch, tmp_path):
    from researchclaw.pipeline.runner import _metaclaw_post_pipeline

    monkeypatch.setattr(
        "researchclaw.metaclaw_bridge.session.MetaClawSession",
        lambda run_id: SimpleNamespace(end=lambda: {"X-Session": "s"}),
    )
    _patch_urlopen(monkeypatch, [{"model": "session-end"}])
    _metaclaw_post_pipeline(_post_pipeline_config(), [], [], "run-1", tmp_path)
    (record,) = [record for record in collect_call_records()
                 if record.get("purpose") == "session_end_signal"]
    assert record["status"] == "succeeded"
    assert record["requested_chain"] == ["session-end"]
    assert record["endpoint"] == "http://proxy:30000/v1/chat/completions"
    assert record["total_tokens"] is None


def test_failed_session_end_signal_is_ledgered_as_a_failure(monkeypatch, tmp_path):
    from researchclaw.pipeline.runner import _metaclaw_post_pipeline

    monkeypatch.setattr(
        "researchclaw.metaclaw_bridge.session.MetaClawSession",
        lambda run_id: SimpleNamespace(end=lambda: {}),
    )
    _patch_urlopen(monkeypatch, [urllib.error.URLError("proxy down")])
    _metaclaw_post_pipeline(_post_pipeline_config(), [], [], "run-1", tmp_path)
    (record,) = [record for record in collect_call_records()
                 if record.get("purpose") == "session_end_signal"]
    assert record["status"] == "failed"
    assert record["error_type"] == "URLError"


# ---------------------------------------------------------------------------
# Embedding provider
# ---------------------------------------------------------------------------


def _embedding_provider():
    from researchclaw.memory.embeddings import EmbeddingProvider

    return EmbeddingProvider(api_base_url="http://stub", api_key="k")


def test_embedding_api_calls_are_ledgered(monkeypatch):
    provider = _embedding_provider()
    _patch_urlopen(monkeypatch, [{
        "data": [{"embedding": [0.1, 0.2]}],
        "usage": {"prompt_tokens": 4, "total_tokens": 4},
    }])
    vector = provider.embed("hello")
    assert vector == [0.1, 0.2]
    (record,) = [record for record in collect_call_records()
                 if record.get("purpose") == "embedding"]
    assert record["status"] == "succeeded"
    assert record["call_family"] == "embeddings"
    assert record["endpoint"] == "http://stub/embeddings"
    assert record["prompt_tokens"] == 4 and record["total_tokens"] == 4
    assert record["completion_tokens"] is None


def test_failed_embedding_call_is_ledgered_and_falls_back(monkeypatch):
    from researchclaw.memory.embeddings import _TFIDF_DIM

    provider = _embedding_provider()
    _patch_urlopen(monkeypatch, [urllib.error.URLError("down")])
    vector = provider.embed("hello world")
    assert len(vector) == _TFIDF_DIM
    (record,) = [record for record in collect_call_records()
                 if record.get("purpose") == "embedding"]
    assert record["status"] == "failed"
    assert record["error_type"] == "URLError"
    assert record["prompt_tokens"] is None


# ---------------------------------------------------------------------------
# Nano Banana generation attempts
# ---------------------------------------------------------------------------


_GEMINI_IMAGE = base64.b64encode(b"\x89PNG-fake-image-bytes").decode("ascii")


def _nano_agent(tmp_path):
    from researchclaw.agents.figure_agent.nano_banana import NanoBananaAgent

    return NanoBananaAgent(llm=None, gemini_api_key="k", use_sdk=False,
                           output_dir=tmp_path)


def _gemini_payload(**extra):
    payload = {
        "candidates": [{"content": {"parts": [
            {"inlineData": {"mimeType": "image/png", "data": _GEMINI_IMAGE}},
        ]}}],
    }
    payload.update(extra)
    return payload


def _nano_figures():
    return [{"figure_id": "concept_1", "description": "overview",
             "figure_type": "architecture_diagram", "section": "Method"}]


def test_nano_banana_success_is_ledgered_with_image_bytes(monkeypatch, tmp_path):
    agent = _nano_agent(tmp_path)
    seen = _patch_urlopen(monkeypatch, [
        _gemini_payload(usageMetadata={"promptTokenCount": 7,
                                       "totalTokenCount": 9})])
    result = agent.execute({"image_figures": _nano_figures(), "topic": "t",
                            "output_dir": str(tmp_path)})
    assert result.success
    (record,) = collect_image_call_records()
    assert record["provider"] == "gemini"
    assert record["mode"] == "prompt"
    assert record["status"] == "succeeded"
    assert record["image_bytes"] == len(base64.b64decode(_GEMINI_IMAGE))
    assert record["figure_id"] == "concept_1"
    assert record["usage"] == {"promptTokenCount": 7, "totalTokenCount": 9}
    assert record["endpoint"].endswith(":generateContent")
    assert len(seen) == 1
    build_image_call_ledger(collect_image_call_records())


def test_nano_banana_failure_is_ledgered(monkeypatch, tmp_path):
    agent = _nano_agent(tmp_path)
    _patch_urlopen(monkeypatch, [
        urllib.error.HTTPError("url", 429, "rate limited", {},
                               io.BytesIO(b"rate limited"))])
    result = agent.execute({"image_figures": _nano_figures(), "topic": "t",
                            "output_dir": str(tmp_path)})
    assert not result.success
    (record,) = collect_image_call_records()
    assert record["status"] == "failed"
    assert record["error_type"] == "HTTPError"
    assert record["image_bytes"] is None
    assert record["usage"] is None


def test_nano_banana_no_image_response_is_a_failed_attempt(monkeypatch, tmp_path):
    agent = _nano_agent(tmp_path)
    _patch_urlopen(monkeypatch, [{"candidates": []}])
    result = agent.execute({"image_figures": _nano_figures(), "topic": "t",
                            "output_dir": str(tmp_path)})
    assert not result.success
    (record,) = collect_image_call_records()
    assert record["status"] == "failed"
    assert record["error_type"] == "RuntimeError"


# ---------------------------------------------------------------------------
# Acceptance cross-check for Nano Banana results
# ---------------------------------------------------------------------------


def _freeze_image_ledger(tmp_path):
    frozen = build_image_call_ledger(collect_image_call_records())
    (tmp_path / "image_call_ledger.json").write_text(
        json.dumps(frozen, indent=2), encoding="utf-8")


def _nano_results(tmp_path, entries):
    stage = tmp_path / "stage-22-export_publish"
    stage.mkdir(parents=True)
    (stage / "nano_banana_results.json").write_text(
        json.dumps({"generated": entries}, indent=2), encoding="utf-8")


def test_acceptance_accepts_fully_ledgered_nano_banana_results(monkeypatch, tmp_path):
    from researchclaw.pipeline.final_acceptance import _nano_banana_ledger_problem

    agent = _nano_agent(tmp_path)
    _patch_urlopen(monkeypatch, [_gemini_payload()])
    agent.execute({"image_figures": _nano_figures(), "topic": "t",
                   "output_dir": str(tmp_path)})
    _freeze_image_ledger(tmp_path)
    _nano_results(tmp_path, [{"figure_id": "concept_1", "success": True}])
    assert _nano_banana_ledger_problem(tmp_path) is None


def test_acceptance_requires_the_run_ledger_for_nano_banana_results(tmp_path):
    from researchclaw.pipeline.final_acceptance import _nano_banana_ledger_problem

    _nano_results(tmp_path, [{"figure_id": "concept_1", "success": True}])
    problem = _nano_banana_ledger_problem(tmp_path)
    assert problem == ("nano_banana_calls_missing_run_ledger",
                       "image_call_ledger.json")


def test_acceptance_rejects_unledgered_nano_banana_figures(monkeypatch, tmp_path):
    from researchclaw.pipeline.final_acceptance import _nano_banana_ledger_problem

    agent = _nano_agent(tmp_path)
    _patch_urlopen(monkeypatch, [_gemini_payload()])
    agent.execute({"image_figures": _nano_figures(), "topic": "t",
                   "output_dir": str(tmp_path)})
    _freeze_image_ledger(tmp_path)
    _nano_results(tmp_path, [
        {"figure_id": "concept_1", "success": True},
        {"figure_id": "concept_2", "success": False},
    ])
    problem = _nano_banana_ledger_problem(tmp_path)
    assert problem == ("nano_banana_calls_not_fully_ledgered",
                       "image_call_ledger.json")


# ---------------------------------------------------------------------------
# Resource ledger: subprocess declaration + stage wall clock
# ---------------------------------------------------------------------------


def _write_stage_file(tmp_path, relpath, payload):
    path = tmp_path / relpath
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_resource_ledger_declares_beast_mode_subprocess_invocations(tmp_path):
    _write_stage_file(tmp_path, "stage-10-code_generation/beast_mode_log.json",
                      {"success": True, "elapsed_sec": 12.5,
                       "model": "openai/m"})
    _write_stage_file(tmp_path, "stage-13-iterative_refine/beast_mode_log.json",
                      {"success": False, "elapsed_sec": 3.0, "model": "openai/m"})
    ledger = build_resource_ledger(tmp_path)
    rows = [row for row in ledger["rows"]
            if row["kind"] == "code_agent_subprocess"]
    assert [(row["source"], row["calls"], row["failures"], row["limit"])
            for row in rows] == [
        ("stage-10-code_generation/beast_mode_log.json", 1, 0, None),
        ("stage-13-iterative_refine/beast_mode_log.json", 1, 1, None),
    ]
    assert ledger_issues(ledger) == []
    validate_resource_ledger(tmp_path, ledger)


def test_resource_ledger_aggregates_stage_wall_clock(tmp_path):
    _write_stage_file(tmp_path, "stage-05-literature_screen/stage_health.json",
                      {"stage_id": "05-literature_screen", "duration_sec": 1.25,
                       "status": "done"})
    _write_stage_file(tmp_path, "stage-06-knowledge_extract/stage_health.json",
                      {"stage_id": "06-knowledge_extract", "duration_sec": 0.75,
                       "status": "failed"})
    ledger = build_resource_ledger(tmp_path)
    wall = ledger["stage_wall_clock"]
    assert [entry["source"] for entry in wall["entries"]] == [
        "stage-05-literature_screen/stage_health.json",
        "stage-06-knowledge_extract/stage_health.json",
    ]
    assert wall["entries"][1]["status"] == "failed"
    assert wall["total_duration_sec"] == 2.0
    assert ledger_issues(ledger) == []
    validate_resource_ledger(tmp_path, ledger)


def test_stage_wall_clock_without_stages_is_null_not_zero(tmp_path):
    ledger = build_resource_ledger(tmp_path)
    assert ledger["stage_wall_clock"] == {"entries": [],
                                          "total_duration_sec": None}


def test_malformed_stage_health_fails_into_errors_not_the_wall_clock(tmp_path):
    _write_stage_file(tmp_path, "stage-07-synthesis/stage_health.json",
                      {"duration_sec": "fast", "status": "done"})
    ledger = build_resource_ledger(tmp_path)
    assert ledger["stage_wall_clock"]["entries"] == []
    assert ledger["stage_wall_clock"]["total_duration_sec"] is None
    assert ledger["errors"] == [{
        "source": "stage-07-synthesis/stage_health.json",
        "reason": "stage_health duration_sec is not a number"}]


def test_tampered_stage_health_breaks_revalidation(tmp_path):
    _write_stage_file(tmp_path, "stage-08-hypothesis_gen/stage_health.json",
                      {"duration_sec": 2.0, "status": "done"})
    ledger = build_resource_ledger(tmp_path)
    (tmp_path / "stage-08-hypothesis_gen/stage_health.json").write_text(
        json.dumps({"duration_sec": 9.0, "status": "done"}), encoding="utf-8")
    with pytest.raises(Exception):
        validate_resource_ledger(tmp_path, ledger)


# ---------------------------------------------------------------------------
# Generated medical-audit experiment
# ---------------------------------------------------------------------------


_MEDICAL_CONTRACT = {
    "executor_template": "binary_risk_audit",
    "source_paths": ["patients.csv"],
    "runtime": {
        "outcome_field": "mortality",
        "identifier_fields": ["patient_id"],
        "allowed_fields": ["age", "sex"],
        "provider": "stub",
        "model": "stub-model",
        "api_key_env": "ARC_TEST_MEDICAL_KEY",
        "conditions": ["DirectStructured", "FullAuditedAgent"],
        "base_url": "http://stub",
        "timeout_sec": 5,
    },
}


def _medical_response(req):
    body = req.data.decode("utf-8")
    if "force_abstain" in body:
        text = '{"accept": true, "force_abstain": false, "critique": "ok"}'
    else:
        text = '{"risk_probability": 0.25, "abstain": false, "rationale": "mild"}'
    return _FakeResponse(json.dumps({
        "content": [{"text": text}],
        "usage": {"input_tokens": 5, "output_tokens": 1},
    }).encode("utf-8"))


def _patients_csv(tmp_path):
    (tmp_path / "patients.csv").write_text(
        "patient_id,age,sex,mortality\n"
        "p1,71,M,1\np2,54,F,0\n",
        encoding="utf-8")


def _run_medical_experiment(monkeypatch, exp_dir, urlopen):
    import runpy

    monkeypatch.setenv("ARC_TEST_MEDICAL_KEY", "stub-key")
    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    monkeypatch.chdir(exp_dir)
    monkeypatch.syspath_prepend(str(exp_dir))
    runpy.run_path(str(exp_dir / "main.py"), run_name="__main__")


def test_medical_experiment_ledgers_every_api_attempt(monkeypatch, tmp_path):
    from researchclaw.experiment.medical_llm_audit_templates import write_template

    write_template(tmp_path, _MEDICAL_CONTRACT)
    exp_dir = tmp_path / "experiment"
    _patients_csv(exp_dir)
    seen: list[object] = []

    def fake_urlopen(req, timeout=None):
        seen.append(req)
        return _medical_response(req)

    _run_medical_experiment(monkeypatch, exp_dir, fake_urlopen)
    calls = json.loads(
        (exp_dir / "model_call_ledger.json").read_text(encoding="utf-8"))
    # 2 conditions x 2 rows scored + 2 critic calls (FullAuditedAgent rows)
    assert len(calls) == 6
    assert all(entry["status"] == "succeeded" for entry in calls)
    assert all(entry["usage"] == {"input_tokens": 5, "output_tokens": 1}
               for entry in calls)
    assert sum(1 for entry in calls if entry["role"] == "scorer") == 4
    assert sum(1 for entry in calls if entry["role"] == "critic") == 2
    assert all(entry["response_sha"] for entry in calls)
    assert all(entry["condition"] in ("DirectStructured", "FullAuditedAgent")
               for entry in calls)
    results = json.loads(
        (exp_dir / "results.json").read_text(encoding="utf-8"))
    assert results["model_calls"] == {"attempts": 6, "failures": 0}


def test_medical_experiment_ledgers_every_retry_after_persistent_failure(
        monkeypatch, tmp_path):
    from researchclaw.experiment.medical_llm_audit_templates import write_template

    write_template(tmp_path, _MEDICAL_CONTRACT)
    exp_dir = tmp_path / "experiment"
    _patients_csv(exp_dir)

    def failing_urlopen(req, timeout=None):
        raise urllib.error.URLError("down")

    monkeypatch.setattr(time, "sleep", lambda seconds: None)
    _run_medical_experiment(monkeypatch, exp_dir, failing_urlopen)
    calls = json.loads(
        (exp_dir / "model_call_ledger.json").read_text(encoding="utf-8"))
    # 4 events x 4 retry attempts; the critic is never reached on failure
    assert len(calls) == 16
    assert all(entry["status"] == "failed" for entry in calls)
    assert all(entry["error_type"] == "URLError" for entry in calls)
    assert [entry["attempt"] for entry in calls[:4]] == [1, 2, 3, 4]
    results = json.loads(
        (exp_dir / "results.json").read_text(encoding="utf-8"))
    assert results["model_calls"] == {"attempts": 16, "failures": 16}
