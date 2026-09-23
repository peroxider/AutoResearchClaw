"""Run-level LLM call ledger: what the endpoints actually served, frozen and validated."""
import json
import urllib.request
from types import SimpleNamespace
from typing import Any

import pytest

from researchclaw.llm.acp_client import ACPClient, ACPConfig
from researchclaw.llm.call_ledger import (
    build_call_ledger, call_ledger_totals, collect_call_records, reset_call_ledger,
    validate_call_ledger, write_call_ledger,
)
from researchclaw.llm.client import LLMClient, LLMConfig
from researchclaw.pipeline.evidence_store import content_hash
from researchclaw.pipeline.resource_ledger import (
    build_resource_ledger, ledger_issues, validate_resource_ledger,
)


@pytest.fixture(autouse=True)
def clean_registry():
    reset_call_ledger()
    yield
    reset_call_ledger()


class _DummyHTTPResponse:
    def __init__(self, payload):
        self._payload = payload

    def read(self):
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None


def make_client(primary: str, fallbacks: list[str]) -> LLMClient:
    config = LLMConfig(base_url="https://api.example.com/v1", api_key="test-key",
                       primary_model=primary, fallback_models=fallbacks,
                       max_retries=1, retry_base_delay=0)
    return LLMClient(config)


def serve(monkeypatch, script):
    """Each chat attempt consumes the next script entry (payload or exception)."""
    attempts = iter(script)

    def fake_urlopen(req: urllib.request.Request, timeout: int) -> _DummyHTTPResponse:
        payload: Any = next(attempts)
        if isinstance(payload, Exception):
            raise payload
        return _DummyHTTPResponse(payload)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)


def response(model: str = "served-alias", content: str = "ok",
             usage: dict | None = None) -> dict:
    payload = {"model": model, "choices": [{"message": {"content": content},
                                            "finish_reason": "stop"}]}
    if usage is not None:
        payload["usage"] = usage
    return payload


def test_served_model_endpoint_and_usage_are_recorded(monkeypatch):
    serve(monkeypatch, [response(model="vendor/actual-2026-09",
                                 usage={"prompt_tokens": 10, "completion_tokens": 5,
                                        "total_tokens": 15})])
    client = make_client("alias-a", [])
    resp = client.chat([{"role": "user", "content": "hello"}])
    assert resp.model == "vendor/actual-2026-09"
    (record,) = collect_call_records()
    assert record["status"] == "succeeded"
    assert record["requested_chain"] == ["alias-a"]
    # The recorded served model is what the endpoint reported, not the
    # requested display name.
    assert record["served_model"] == "vendor/actual-2026-09"
    assert record["endpoint"] == "https://api.example.com/v1/chat/completions"
    assert record["adapter"] == "openai"
    assert record["max_tokens"] == 4096 and record["temperature"] == 0.7
    assert record["prompt_tokens"] == 10 and record["completion_tokens"] == 5
    assert record["total_tokens"] == 15
    assert record["fallback_failures"] == [] and record["truncated"] is False
    assert record["duration_seconds"] >= 0


def test_fallback_failure_chain_is_recorded(monkeypatch):
    serve(monkeypatch, [RuntimeError("boom"), response(usage={"total_tokens": 7})])
    client = make_client("primary-a", ["backup-b"])
    client.chat([{"role": "user", "content": "hi"}])
    (record,) = collect_call_records()
    assert record["requested_chain"] == ["primary-a", "backup-b"]
    assert record["fallback_failures"] == [{"model": "primary-a",
                                            "error": "RuntimeError: boom"}]
    assert record["served_model"] == "served-alias"
    assert record["total_tokens"] == 7


def test_exhausted_chain_records_a_failed_call(monkeypatch):
    serve(monkeypatch, [RuntimeError("first"), RuntimeError("second")])
    client = make_client("primary-a", ["backup-b"])
    with pytest.raises(RuntimeError):
        client.chat([{"role": "user", "content": "hi"}])
    (record,) = collect_call_records()
    assert record["status"] == "failed" and record["served_model"] is None
    assert record["prompt_tokens"] is None and record["total_tokens"] is None
    assert [failure["model"] for failure in record["fallback_failures"]] == \
        ["primary-a", "backup-b"]
    assert call_ledger_totals([record])["failures"] == 1


def test_write_freezes_totals_and_rejects_tampering(monkeypatch, tmp_path):
    serve(monkeypatch, [response(usage={"prompt_tokens": 10, "completion_tokens": 5,
                                        "total_tokens": 15})])
    make_client("alias-a", []).chat([{"role": "user", "content": "hi"}])
    ledger = write_call_ledger(tmp_path)
    assert ledger is not None and (tmp_path / "llm_call_ledger.json").is_file()
    validate_call_ledger(ledger)
    assert ledger["totals"] == {"calls": 1, "failures": 0, "prompt_tokens": 10,
                                "completion_tokens": 5, "total_tokens": 15}
    # Successful write drains the clients; a second write records nothing new.
    assert collect_call_records() == []
    assert write_call_ledger(tmp_path) is None
    stored = json.loads((tmp_path / "llm_call_ledger.json").read_text())
    validate_call_ledger(stored)
    tampered = dict(stored, calls=[dict(stored["calls"][0], total_tokens=999)])
    with pytest.raises(ValueError, match="version changed"):
        validate_call_ledger(tampered)
    # With the version hash recomputed over the tampered payload, the
    # totals check is reached on its own and still rejects.
    def rehashed(document: dict) -> dict:
        return dict(document, version=content_hash(
            {key: value for key, value in document.items() if key != "version"}))

    with pytest.raises(ValueError, match="totals differ"):
        validate_call_ledger(rehashed(tampered))
    with pytest.raises(ValueError, match="totals differ"):
        validate_call_ledger(rehashed({"schema_version": 1, "calls": [], "totals": {}}))
    with pytest.raises(ValueError, match="malformed"):
        validate_call_ledger(rehashed(dict(stored, calls=[dict(stored["calls"][0], status="other")])))
    with pytest.raises(ValueError, match="malformed"):
        validate_call_ledger(rehashed(dict(stored, calls=[dict(stored["calls"][0], total_tokens=True)])))


def test_write_without_records_writes_nothing(tmp_path):
    assert write_call_ledger(tmp_path) is None
    assert not (tmp_path / "llm_call_ledger.json").exists()


def test_registry_spans_clients_and_never_double_counts(monkeypatch, tmp_path):
    serve(monkeypatch, [response(), response()])
    make_client("a", []).chat([{"role": "user", "content": "1"}])
    make_client("b", []).chat([{"role": "user", "content": "2"}])
    ledger = write_call_ledger(tmp_path)
    assert ledger["totals"]["calls"] == 2
    assert collect_call_records() == []


def test_resource_ledger_derives_the_llm_chat_row(monkeypatch, tmp_path):
    serve(monkeypatch, [response(usage={"prompt_tokens": 10, "completion_tokens": 5,
                                        "total_tokens": 15}),
                        RuntimeError("down")])
    make_client("a", []).chat([{"role": "user", "content": "1"}])
    with pytest.raises(RuntimeError):
        make_client("c", []).chat([{"role": "user", "content": "2"}])
    frozen = write_call_ledger(tmp_path)
    assert frozen is not None
    ledger = build_resource_ledger(tmp_path)
    (row,) = [row for row in ledger["rows"] if row["source"] == "llm_call_ledger.json"]
    assert row == {"source": "llm_call_ledger.json", "kind": "llm_chat",
                   "calls": 2, "limit": None, "failures": 1, "tokens": 15,
                   "version": frozen["version"]}
    assert ledger_issues(ledger) == []
    validate_resource_ledger(tmp_path, ledger)
    # Tampering with the frozen file breaks the stored ledger's match with
    # the artifacts; revalidation of the pre-tamper ledger fails closed.
    document = json.loads((tmp_path / "llm_call_ledger.json").read_text())
    document["calls"][0]["total_tokens"] = 999
    (tmp_path / "llm_call_ledger.json").write_text(json.dumps(document))
    with pytest.raises(Exception):
        validate_resource_ledger(tmp_path, ledger)


def test_malformed_llm_ledger_fails_closed_in_resource_ledger(tmp_path):
    (tmp_path / "llm_call_ledger.json").write_text('{"calls": "not-a-list"}')
    ledger = build_resource_ledger(tmp_path)
    assert ledger["rows"] == []
    assert ledger_issues(ledger) == ["unreadable_resource_source:llm_call_ledger.json"]


def test_acp_calls_record_agent_identity_without_invented_usage(monkeypatch):
    client = ACPClient(ACPConfig(agent="codex", acpx_command=""))
    monkeypatch.setattr(client, "_send_prompt", lambda prompt: "answer")
    client.chat([{"role": "user", "content": "hi"}])
    (record,) = collect_call_records()
    assert record["status"] == "succeeded"
    assert record["served_model"] == "acp:codex" and record["adapter"] == "acp"
    assert record["endpoint"] == "acpx:auto"
    assert record["prompt_tokens"] is None and record["total_tokens"] is None
    monkeypatch.setattr(client, "_send_prompt",
                        lambda prompt: (_ for _ in ()).throw(RuntimeError("acp down")))
    with pytest.raises(RuntimeError):
        client.chat([{"role": "user", "content": "hi"}])
    failure = collect_call_records()[-1]
    assert failure["status"] == "failed" and failure["served_model"] is None


def test_anthropic_clients_record_the_real_messages_endpoint():
    client = make_client("display-a", [])
    client._anthropic = SimpleNamespace(
        base_url="https://api.anthropic.com",
        chat_completion=lambda *args, **kwargs: {
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
            "model": "claude-served"})
    client.chat([{"role": "user", "content": "hi"}])
    (record,) = collect_call_records()
    assert record["endpoint"] == "https://api.anthropic.com/v1/messages"
    assert record["adapter"] == "anthropic"
    assert record["served_model"] == "claude-served"
    assert record["total_tokens"] == 3


def test_absent_usage_block_stays_unmeasured_not_zero(monkeypatch):
    serve(monkeypatch, [response(),
                        response(usage={"prompt_tokens": 2, "completion_tokens": 3,
                                        "total_tokens": 5})])
    client = make_client("alias-a", [])
    client.chat([{"role": "user", "content": "1"}])
    client.chat([{"role": "user", "content": "2"}])
    first, second = collect_call_records()
    assert first["prompt_tokens"] is None and first["total_tokens"] is None
    assert second["total_tokens"] == 5
    ledger = build_call_ledger(collect_call_records())
    # The unmeasured call contributes nothing instead of a fabricated 0.
    assert ledger["totals"] == {"calls": 2, "failures": 0, "prompt_tokens": 2,
                                "completion_tokens": 3, "total_tokens": 5}
    validate_call_ledger(ledger)


def test_null_usage_response_is_not_recorded_as_a_failure(monkeypatch):
    serve(monkeypatch, [{"model": "served-alias",
                         "choices": [{"message": {"content": "ok"},
                                      "finish_reason": "stop"}],
                         "usage": None}])
    make_client("alias-a", []).chat([{"role": "user", "content": "hi"}])
    (record,) = collect_call_records()
    assert record["status"] == "succeeded"
    assert record["fallback_failures"] == []
    assert record["total_tokens"] is None


def test_build_call_ledger_totals_tolerate_unmeasured_tokens():
    records = [{"status": "succeeded", "fallback_failures": [],
                "total_tokens": None, "prompt_tokens": None,
                "completion_tokens": None},
               {"status": "succeeded", "fallback_failures": [],
                "total_tokens": 4, "prompt_tokens": 3,
                "completion_tokens": 1}]
    ledger = build_call_ledger(records)
    assert ledger["totals"] == {"calls": 2, "failures": 0, "prompt_tokens": 3,
                                "completion_tokens": 1, "total_tokens": 4}
    validate_call_ledger(ledger)
