# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for context_management compaction: sent on cloud OpenAI only, a no-op on other base URLs."""

import asyncio
import json

import httpx

from core.inference import external_provider as ep_mod
from core.inference.external_provider import ExternalProviderClient


def _drive(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


def _capture(monkeypatch, *, base_url: str, threshold) -> dict:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            content = (
                b"event: response.completed\n"
                b'data: {"type":"response.completed",'
                b'"response":{"output":[],"usage":{"input_tokens":0,'
                b'"output_tokens":0}}}\n\n'
            ),
            headers = {"content-type": "text/event-stream"},
        )

    monkeypatch.setattr(
        ep_mod,
        "_http_client",
        httpx.AsyncClient(transport = httpx.MockTransport(handler)),
    )

    async def run():
        client = ExternalProviderClient(
            provider_type = "openai",
            base_url = base_url,
            api_key = "sk-test",
        )
        async for _ in client.stream_chat_completion(
            messages = [{"role": "user", "content": "hi"}],
            model = "gpt-5.5",
            temperature = 0.7,
            top_p = 0.95,
            max_tokens = 32,
            reasoning_effort = "medium",
            compaction_threshold = threshold,
        ):
            pass
        await client.close()

    _drive(run())
    return captured


def _capture_at_threshold(
    *args,
    threshold = 200_000,
    **kwargs,
):
    """_capture at the shared 200k compaction threshold."""
    return _capture(*args, threshold = threshold, **kwargs)


def test_cloud_openai_sets_compaction_block(monkeypatch):
    captured = _capture_at_threshold(monkeypatch, base_url = "https://api.openai.com/v1")
    assert captured["body"].get("context_management") == [
        {"type": "compaction", "compact_threshold": 200_000}
    ]


def test_cloud_openai_below_default_threshold_passes_through(monkeypatch):
    # OpenAI accepts thresholds below the default without clamping.
    captured = _capture(
        monkeypatch,
        base_url = "https://api.openai.com/v1",
        threshold = 60_000,
    )
    assert captured["body"]["context_management"] == [
        {"type": "compaction", "compact_threshold": 60_000}
    ]


def test_cloud_openai_threshold_capped_at_200k(monkeypatch):
    captured = _capture(monkeypatch, base_url = "https://api.openai.com/v1", threshold = 750_000)
    assert captured["body"]["context_management"] == [
        {"type": "compaction", "compact_threshold": 200_000}
    ]


def test_non_cloud_base_silently_drops_compaction(monkeypatch):
    # OpenAI-compatible local providers reject context_management.
    captured = _capture_at_threshold(monkeypatch, base_url = "http://127.0.0.1:11434/v1")
    assert "context_management" not in captured["body"]


def test_azure_openai_base_url_carries_compaction_block(monkeypatch):
    # Azure OpenAI exposes the same /v1/responses extensions, so treat it as cloud.
    captured = _capture_at_threshold(
        monkeypatch,
        base_url = "https://my-resource.openai.azure.com/openai/v1",
    )
    assert captured["body"].get("context_management") == [
        {"type": "compaction", "compact_threshold": 200_000}
    ]
    assert captured["body"].get("prompt_cache_retention") == "24h"


def test_azure_openai_mixed_case_base_url_matches(monkeypatch):
    captured = _capture(
        monkeypatch,
        base_url = "https://My-Resource.OpenAI.Azure.Com/openai/v1",
        threshold = 50_000,
    )
    assert captured["body"].get("context_management") == [
        {"type": "compaction", "compact_threshold": 50_000}
    ]


def test_cloud_gate_uses_hostname_not_substring(monkeypatch):
    # Hostname-anchored check: cloud hostnames embedded in a path or subdomain must not get cloud fields.
    for evil in [
        "https://evil.com/api.openai.com/v1",
        "https://api.openai.com.attacker.com/v1",
        "https://attacker.com/.openai.azure.com/v1",
        "https://my-resource.openai.azure.com.attacker.com/openai/v1",
        "https://attacker.com/.services.ai.azure.com/openai/v1",
        "https://my-resource.services.ai.azure.com.attacker.com/openai/v1",
        "https://my-resource.services.ai.azure.com@attacker.com/openai/v1",
    ]:
        captured = _capture_at_threshold(monkeypatch, base_url = evil)
        assert "context_management" not in captured["body"], evil
        assert "prompt_cache_retention" not in captured["body"], evil


def test_omitted_threshold_no_body_field(monkeypatch):
    captured = _capture(
        monkeypatch,
        base_url = "https://api.openai.com/v1",
        threshold = None,
    )
    assert "context_management" not in captured["body"]


def test_chat_completion_request_accepts_any_positive_compaction_threshold():
    # Schema floor stays ge=1: the field is a no-op elsewhere; per-provider helpers enforce real floors.
    import pytest as _pytest

    from models.inference import ChatCompletionRequest

    with _pytest.raises(Exception):
        ChatCompletionRequest.model_validate(
            {
                "model": "default",
                "messages": [{"role": "user", "content": "hi"}],
                "compaction_threshold": 0,
            }
        )

    # positive thresholds stay schema-valid because providers enforce support and minimums.
    for v in (1, 5_000, 9_999, 10_000, 200_000):
        req = ChatCompletionRequest.model_validate(
            {
                "model": "default",
                "messages": [{"role": "user", "content": "hi"}],
                "compaction_threshold": v,
            }
        )
        assert req.compaction_threshold == v


def _stream(
    monkeypatch,
    handler,
    *,
    messages,
    threshold = None,
    base_url = "https://api.openai.com/v1",
    enabled_tools = None,
    compaction_fallback = None,
):
    monkeypatch.setattr(
        ep_mod,
        "_http_client",
        httpx.AsyncClient(transport = httpx.MockTransport(handler)),
    )

    async def run():
        client = ExternalProviderClient(
            provider_type = "openai",
            base_url = base_url,
            api_key = "sk-test",
        )
        extra = {}
        if enabled_tools is not None:
            extra["enabled_tools"] = enabled_tools
        if compaction_fallback is not None:
            extra["compaction_fallback"] = compaction_fallback
        lines = [
            line
            async for line in client.stream_chat_completion(
                messages = messages,
                model = "gpt-5.5",
                temperature = 0.7,
                top_p = 0.95,
                max_tokens = 32,
                compaction_threshold = threshold,
                **extra,
            )
        ]
        await client.close()
        return lines

    return _drive(run())


_EMPTY_COMPLETED = (
    b"event: response.completed\n"
    b'data: {"type":"response.completed",'
    b'"response":{"output":[],"usage":{"input_tokens":0,"output_tokens":0}}}\n\n'
)


def test_a_compaction_output_item_is_handed_to_the_client(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content = (
                b"event: response.output_item.done\n"
                b'data: {"type":"response.output_item.done","output_index":0,'
                b'"item":{"type":"compaction","id":"cmp_1","encrypted_content":"gAAAA-opaque"}}\n\n'
                + _EMPTY_COMPLETED
            ),
            headers = {"content-type": "text/event-stream"},
        )

    lines = _stream(
        monkeypatch, handler, messages = [{"role": "user", "content": "hi"}], threshold = 200_000
    )
    events = [
        json.loads(line[len("data:") :])["_toolEvent"]
        for line in lines
        if line.startswith("data:") and "compaction_block" in line
    ]
    assert [event["encrypted_content"] for event in events] == ["gAAAA-opaque"]


def test_a_replayed_compaction_item_replaces_the_history_it_covers(monkeypatch):
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200, content = _EMPTY_COMPLETED, headers = {"content-type": "text/event-stream"}
        )

    _stream(
        monkeypatch,
        handler,
        messages = [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "turn 1"},
            {
                "role": "assistant",
                "content": [
                    {"type": "compaction", "encrypted_content": "gAAAA-opaque"},
                    {"type": "text", "text": "answer 1"},
                ],
            },
            {"role": "user", "content": "turn 2"},
        ],
    )
    items = captured["body"]["input"]
    assert items[0] == {"type": "compaction", "encrypted_content": "gAAAA-opaque"}
    assert "turn 1" not in json.dumps(items)
    assert "turn 2" in json.dumps(items)
    assert captured["body"]["instructions"] == "Be brief."


def test_a_replayed_compaction_item_clears_an_earlier_response_reference(monkeypatch):
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200, content = _EMPTY_COMPLETED, headers = {"content-type": "text/event-stream"}
        )

    _stream(
        monkeypatch,
        handler,
        enabled_tools = ["image_generation"],
        messages = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "image_generation_call",
                        "id": "image_earlier",
                        "response_id": "resp_earlier",
                    }
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "compaction", "encrypted_content": "gAAAA-current"}],
            },
            {"role": "user", "content": "edit the current result"},
        ],
    )
    assert "previous_response_id" not in captured["body"]
    assert "image_earlier" not in json.dumps(captured["body"]["input"])


def test_a_replayed_compaction_item_clears_earlier_manual_image_replay(monkeypatch):
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200, content = _EMPTY_COMPLETED, headers = {"content-type": "text/event-stream"}
        )

    _stream(
        monkeypatch,
        handler,
        enabled_tools = ["image_generation"],
        messages = [
            {
                "role": "assistant",
                "content": [
                    {"type": "reasoning", "id": "rs_earlier", "summary": []},
                    {"type": "image_generation_call", "id": "image_earlier"},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "compaction", "encrypted_content": "gAAAA-current"}],
            },
            {"role": "user", "content": "continue"},
        ],
    )
    serialized = json.dumps(captured["body"]["input"])
    assert "rs_earlier" not in serialized
    assert "image_earlier" not in serialized


def test_a_deployment_without_compaction_is_retried_without_it(monkeypatch):
    bodies: list = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        bodies.append(body)
        if "context_management" in body:
            return httpx.Response(
                400,
                json = {
                    "error": {
                        "message": "compact_threshold is not enabled.",
                        "type": "invalid_request_error",
                        "param": "compact_threshold",
                        "code": "unsupported_parameter",
                    }
                },
            )
        return httpx.Response(
            200, content = _EMPTY_COMPLETED, headers = {"content-type": "text/event-stream"}
        )

    lines = _stream(
        monkeypatch,
        handler,
        messages = [{"role": "user", "content": "hi"}],
        threshold = 96_000,
        base_url = "https://myres.openai.azure.com/openai/v1",
    )
    assert len(bodies) == 2
    assert "context_management" not in bodies[1]
    assert not any('"error"' in line for line in lines)


def test_a_compaction_rejection_refits_history_before_retry(monkeypatch):
    bodies: list[dict] = []
    fallback_calls: list[list[dict]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        bodies.append(body)
        if "context_management" in body:
            return httpx.Response(
                400,
                json = {
                    "error": {
                        "message": "context_management is unavailable for this deployment",
                        "type": "invalid_request_error",
                    }
                },
            )
        return httpx.Response(
            200, content = _EMPTY_COMPLETED, headers = {"content-type": "text/event-stream"}
        )

    async def fallback(messages):
        fallback_calls.append(messages)
        fitted = [messages[0], messages[-1]]
        notice = "data: " + json.dumps(
            {
                "id": "chatcmpl-fallback",
                "object": "chat.completion.chunk",
                "choices": [],
                "context_truncated": {"dropped_messages": len(messages) - len(fitted)},
            }
        )
        return fitted, 32, notice

    messages = [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "old turn"},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "latest turn"},
    ]
    lines = _stream(
        monkeypatch,
        handler,
        messages = messages,
        threshold = 96_000,
        base_url = "https://myres.openai.azure.com/openai/v1",
        compaction_fallback = fallback,
    )
    assert fallback_calls == [messages]
    assert len(bodies) == 2
    assert "context_management" not in bodies[1]
    assert "old turn" not in json.dumps(bodies[1]["input"])
    assert "latest turn" in json.dumps(bodies[1]["input"])
    assert any("context_truncated" in line for line in lines)


def test_a_compaction_rejection_is_remembered_across_tool_loop_turns(monkeypatch):
    bodies: list[dict] = []
    fallback_calls: list[list[dict]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        bodies.append(body)
        if "context_management" in body:
            return httpx.Response(
                400,
                json = {
                    "error": {
                        "message": "context_management is unavailable for this deployment",
                        "type": "invalid_request_error",
                    }
                },
            )
        return httpx.Response(
            200, content = _EMPTY_COMPLETED, headers = {"content-type": "text/event-stream"}
        )

    async def fallback(messages):
        fallback_calls.append(messages)
        return messages, 32, None

    turns = [
        [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "old turn"},
            {"role": "user", "content": "first provider turn"},
        ],
        [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "old turn"},
            {
                "role": "assistant",
                "content": "calling a Studio tool",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "terminal", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "tool result"},
        ],
    ]
    monkeypatch.setattr(
        ep_mod,
        "_http_client",
        httpx.AsyncClient(transport = httpx.MockTransport(handler)),
    )

    async def run():
        client = ExternalProviderClient(
            provider_type = "openai",
            base_url = "https://myres.openai.azure.com/openai/v1",
            api_key = "sk-test",
        )
        for messages in turns:
            async for _ in client.stream_chat_completion(
                messages = messages,
                model = "gpt-5.5",
                max_tokens = 32,
                compaction_threshold = 96_000,
                compaction_fallback = fallback,
            ):
                pass
        await client.close()

    _drive(run())
    assert fallback_calls == turns
    assert len(bodies) == 3
    assert "context_management" in bodies[0]
    assert all("context_management" not in body for body in bodies[1:])


def test_build_external_messages_passes_the_compaction_item_to_openai():
    from models.inference import ChatMessage
    from routes.inference import _build_external_messages

    msgs = [
        ChatMessage.model_validate(
            {
                "role": "assistant",
                "content": [
                    {"type": "compaction", "encrypted_content": "gAAAA-opaque"},
                    {"type": "text", "text": "answer"},
                ],
            }
        )
    ]
    out = _build_external_messages(msgs, supports_vision = True, provider_type = "openai")
    assert out[0]["content"][0] == {"type": "compaction", "encrypted_content": "gAAAA-opaque"}
    anthropic = _build_external_messages(msgs, supports_vision = True, provider_type = "anthropic")
    deepseek = _build_external_messages(msgs, supports_vision = True, provider_type = "deepseek")
    for out in (anthropic, deepseek):
        assert "gAAAA-opaque" not in json.dumps(out)


def test_a_tool_round_compaction_item_replaces_the_history_it_covers(monkeypatch):
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200, content = _EMPTY_COMPLETED, headers = {"content-type": "text/event-stream"}
        )

    _stream(
        monkeypatch,
        handler,
        messages = [
            {"role": "user", "content": "turn 1"},
            {
                "role": "assistant",
                "content": "",
                "extra_content": {"openai_responses_compaction": "gAAAA-opaque"},
            },
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "found"},
        ],
    )
    items = captured["body"]["input"]
    assert items[0] == {"type": "compaction", "encrypted_content": "gAAAA-opaque"}
    assert "turn 1" not in json.dumps(items)
    assert [item["type"] for item in items[1:]] == ["function_call", "function_call_output"]


def test_a_replayed_compaction_item_drops_the_image_chain_it_covers(monkeypatch):
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200, content = _EMPTY_COMPLETED, headers = {"content-type": "text/event-stream"}
        )

    _stream(
        monkeypatch,
        handler,
        enabled_tools = ["image_generation"],
        messages = [
            {"role": "user", "content": "draw a cat"},
            {
                "role": "assistant",
                "content": [
                    {"type": "image_generation_call", "id": "ig_1", "response_id": "resp_1"}
                ],
            },
            {"role": "user", "content": "turn 2"},
            {
                "role": "assistant",
                "content": [
                    {"type": "compaction", "encrypted_content": "gAAAA-opaque"},
                    {"type": "text", "text": "answer 2"},
                ],
            },
            {"role": "user", "content": "turn 3"},
        ],
    )
    body = captured["body"]
    assert "previous_response_id" not in body
    assert body["input"][0] == {"type": "compaction", "encrypted_content": "gAAAA-opaque"}
    assert "ig_1" not in json.dumps(body["input"])
