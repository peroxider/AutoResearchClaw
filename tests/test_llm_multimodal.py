"""Wire-shape and boundary tests for independently reviewed image calls."""
from __future__ import annotations

import base64

import pytest

from researchclaw.llm.client import LLMClient, LLMConfig, LLMResponse


def client(*, wire_api="chat_completions") -> LLMClient:
    return LLMClient(LLMConfig(base_url="https://example.invalid/v1", api_key="test",
                               wire_api=wire_api, primary_model="vision-reviewer",
                               fallback_models=[], max_retries=1))


def test_chat_image_builds_bounded_chat_completions_payload_and_is_ledgered(monkeypatch):
    reviewer = client()
    captured = {}

    def call(model, messages, max_tokens, temperature, json_mode):
        captured.update(model=model, messages=messages, max_tokens=max_tokens,
                        temperature=temperature, json_mode=json_mode)
        return LLMResponse(content='{"status":"passed"}', model="served-vision",
                           prompt_tokens=3, completion_tokens=2, total_tokens=5,
                           raw={"usage": {"prompt_tokens": 3, "completion_tokens": 2,
                                          "total_tokens": 5}})

    monkeypatch.setattr(reviewer, "_call_with_retry", call)
    response = reviewer.chat_image("Inspect semantics", b"\x89PNG fixture", max_tokens=321)
    parts = captured["messages"][0]["content"]
    assert response.model == "served-vision"
    assert parts[0] == {"type": "text", "text": "Inspect semantics"}
    assert parts[1]["image_url"]["url"] == (
        "data:image/png;base64," + base64.b64encode(b"\x89PNG fixture").decode("ascii"))
    assert captured["max_tokens"] == 321 and captured["temperature"] == 0
    assert reviewer._call_records[-1]["served_model"] == "served-vision"
    assert reviewer._call_records[-1]["total_tokens"] == 5


def test_responses_wire_converts_image_parts_without_stringifying_them():
    reviewer = client(wire_api="responses")
    data_url = "data:image/png;base64,AAAA"
    converted = reviewer._messages_to_responses_input([{"role": "user", "content": [
        {"type": "text", "text": "Review"},
        {"type": "image_url", "image_url": {"url": data_url}},
    ]}])
    assert converted == [{"role": "user", "content": [
        {"type": "input_text", "text": "Review"},
        {"type": "input_image", "image_url": data_url},
    ]}]


@pytest.mark.parametrize("prompt,payload,mime", [
    pytest.param("", b"x", "image/png", id="empty-prompt"),
    pytest.param("review", b"", "image/png", id="empty-image"),
    pytest.param("review", b"x", "image/gif", id="unsupported-mime"),
])
def test_chat_image_rejects_invalid_or_unbounded_inputs_before_network(prompt, payload, mime):
    with pytest.raises(ValueError):
        client().chat_image(prompt, payload, mime_type=mime)


def test_chat_image_rejects_oversized_input_before_base64_expansion():
    with pytest.raises(ValueError, match="10 MB"):
        client().chat_image("review", b"x" * 10_000_001)


def test_chat_image_rejects_unaudited_anthropic_wire_adapter():
    reviewer = client()
    reviewer._anthropic = object()
    with pytest.raises(ValueError, match="Anthropic"):
        reviewer.chat_image("review", b"image")
