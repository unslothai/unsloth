# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The compaction trigger is clamped up to the 50k-token minimum so the request does not 400."""

import asyncio
import json

import httpx
import pytest

from core.inference import external_provider as ep_mod
from core.inference.external_provider import (
    ExternalProviderClient,
    _anthropic_supports_compaction,
)


def _drive(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


def _make_client() -> ExternalProviderClient:
    return ExternalProviderClient(
        provider_type = "anthropic",
        base_url = "https://api.anthropic.com/v1",
        api_key = "sk-ant-test",
    )


def _capture(
    monkeypatch,
    model: str,
    threshold,
    tools = None,
) -> dict:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content.decode("utf-8"))
        captured["headers"] = dict(request.headers)
        return httpx.Response(
            200,
            content = b'event: message_stop\ndata: {"type": "message_stop"}\n\n',
            headers = {"content-type": "text/event-stream"},
        )

    monkeypatch.setattr(
        ep_mod,
        "_http_client",
        httpx.AsyncClient(transport = httpx.MockTransport(handler)),
    )

    async def run():
        client = _make_client()
        async for _ in client.stream_chat_completion(
            messages = [{"role": "user", "content": "hi"}],
            model = model,
            temperature = 0.7,
            top_p = 0.95,
            max_tokens = 32,
            enabled_tools = tools,
            compaction_threshold = threshold,
        ):
            pass
        await client.close()

    _drive(run())
    return captured


@pytest.mark.parametrize(
    "model, supported",
    [
        ("claude-opus-4-7", True),
        ("claude-opus-4-6", True),
        ("claude-sonnet-4-6", True),
        ("claude-mythos-preview", True),
        ("claude-opus-4-5-20251101", False),
        ("claude-sonnet-4-5-20250929", False),
        ("claude-haiku-4-5-20251001", False),
        ("claude-opus-4-1-20250805", False),
        ("claude-opus-4-20250514", False),
        ("claude-sonnet-4-20250514", False),
        ("claude-3-5-sonnet-20241022", False),
    ],
)
def test_supports_compaction_gate(model, supported):
    assert _anthropic_supports_compaction(model) is supported


def test_supported_model_attaches_compaction_block_and_beta(monkeypatch):
    captured = _capture(monkeypatch, "claude-opus-4-7", 150_000)
    cm = captured["body"].get("context_management")
    assert cm == {
        "edits": [
            {
                "type": "compact_20260112",
                "trigger": {"type": "input_tokens", "value": 150_000},
            }
        ]
    }, cm
    assert "compact-2026-01-12" in captured["headers"].get("anthropic-beta", "")


def test_threshold_clamped_to_50k_minimum(monkeypatch):
    captured = _capture(monkeypatch, "claude-opus-4-7", 60_000)
    assert captured["body"]["context_management"]["edits"][0]["trigger"]["value"] == 60_000
    captured = _capture(monkeypatch, "claude-opus-4-7", 1)
    assert captured["body"]["context_management"]["edits"][0]["trigger"]["value"] == 50_000


def test_threshold_capped_at_200k(monkeypatch):
    captured = _capture(monkeypatch, "claude-opus-4-7", 750_000)
    assert captured["body"]["context_management"]["edits"][0]["trigger"]["value"] == 200_000


# ── beta header merge with code execution ────────────────────────────


def test_compaction_beta_merges_with_code_execution_beta(monkeypatch):
    captured = _capture(
        monkeypatch,
        "claude-opus-4-7",
        150_000,
        tools = ["code_execution"],
    )
    beta = captured["headers"].get("anthropic-beta", "")
    assert "code-execution-2025-08-25" in beta
    assert "compact-2026-01-12" in beta


def test_unsupported_model_silently_drops_compaction(monkeypatch):
    captured = _capture(monkeypatch, "claude-haiku-4-5-20251001", 150_000)
    assert "context_management" not in captured["body"]
    assert "compact-2026-01-12" not in captured["headers"].get("anthropic-beta", "")


def test_omitted_threshold_no_body_field(monkeypatch):
    captured = _capture(monkeypatch, "claude-opus-4-7", None)
    assert "context_management" not in captured["body"]
    assert "compact-2026-01-12" not in captured["headers"].get("anthropic-beta", "")


def test_chat_completion_request_accepts_sub_50k_compaction_threshold():
    # ge=50_000 would 422 before the in-helper clamp fires.
    from models.inference import ChatCompletionRequest

    req = ChatCompletionRequest.model_validate(
        {
            "model": "default",
            "messages": [{"role": "user", "content": "hi"}],
            "compaction_threshold": 1,
        }
    )
    assert req.compaction_threshold == 1

    req = ChatCompletionRequest.model_validate(
        {
            "model": "default",
            "messages": [{"role": "user", "content": "hi"}],
            "compaction_threshold": 49_999,
        }
    )
    assert req.compaction_threshold == 49_999

    with pytest.raises(Exception):
        ChatCompletionRequest.model_validate(
            {
                "model": "default",
                "messages": [{"role": "user", "content": "hi"}],
                "compaction_threshold": 0,
            }
        )


def test_message_delta_iterations_array_aggregates_compaction_tokens(monkeypatch, capsys):
    # Compaction tokens live in usage.iterations and must be folded into the summary.

    def http_handler(request: httpx.Request) -> httpx.Response:
        body = (
            b"event: message_start\n"
            b'data: {"type":"message_start","message":{"usage":{"input_tokens":23000,"output_tokens":0}}}\n\n'
            b"event: message_delta\n"
            b'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},'
            b'"usage":{"input_tokens":23000,"output_tokens":1000,'
            b'"iterations":['
            b'{"type":"compaction","input_tokens":180000,"output_tokens":3500},'
            b'{"type":"message","input_tokens":23000,"output_tokens":1000}'
            b"]}}\n\n"
            b"event: message_stop\n"
            b'data: {"type":"message_stop"}\n\n'
        )
        return httpx.Response(
            200,
            content = body,
            headers = {"content-type": "text/event-stream"},
        )

    monkeypatch.setattr(
        ep_mod,
        "_http_client",
        httpx.AsyncClient(transport = httpx.MockTransport(http_handler)),
    )

    async def run():
        client = _make_client()
        async for _ in client.stream_chat_completion(
            messages = [{"role": "user", "content": "hi"}],
            model = "claude-opus-4-7",
            temperature = 0.7,
            top_p = 0.95,
            max_tokens = 32,
            compaction_threshold = 150_000,
        ):
            pass
        await client.close()

    _drive(run())

    out = capsys.readouterr().out
    summary = next(
        (line for line in out.splitlines() if "Anthropic stream complete" in line),
        "",
    )
    assert "compaction_input_tokens=180000" in summary, summary
    assert "compaction_output_tokens=3500" in summary, summary


def test_message_delta_no_iterations_leaves_compaction_keys_unset(monkeypatch, capsys):
    # No fresh iterations array: must not invent compaction keys (double-bill).
    def http_handler(request: httpx.Request) -> httpx.Response:
        body = (
            b"event: message_delta\n"
            b'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},'
            b'"usage":{"input_tokens":1234,"output_tokens":5}}\n\n'
            b"event: message_stop\n"
            b'data: {"type":"message_stop"}\n\n'
        )
        return httpx.Response(
            200,
            content = body,
            headers = {"content-type": "text/event-stream"},
        )

    monkeypatch.setattr(
        ep_mod,
        "_http_client",
        httpx.AsyncClient(transport = httpx.MockTransport(http_handler)),
    )

    async def run():
        client = _make_client()
        async for _ in client.stream_chat_completion(
            messages = [{"role": "user", "content": "hi"}],
            model = "claude-opus-4-7",
            temperature = 0.7,
            top_p = 0.95,
            max_tokens = 32,
            compaction_threshold = 150_000,
        ):
            pass
        await client.close()

    _drive(run())

    out = capsys.readouterr().out
    summary = next(
        (line for line in out.splitlines() if "Anthropic stream complete" in line),
        "",
    )
    assert "compaction_input_tokens=None" in summary, summary
    assert "compaction_output_tokens=None" in summary, summary


def _async_collect(agen):
    async def run():
        out = []
        async for line in agen:
            out.append(line)
        return out

    return _drive(run())


def test_compaction_block_emitted_as_tool_event(monkeypatch):
    # The compaction block must surface so the next turn keeps state.

    def http_handler(request: httpx.Request) -> httpx.Response:
        body = (
            b"event: message_start\n"
            b'data: {"type":"message_start","message":{"usage":{}}}\n\n'
            b"event: content_block_start\n"
            b'data: {"type":"content_block_start","index":0,'
            b'"content_block":{"type":"compaction","content":""}}\n\n'
            b"event: content_block_delta\n"
            b'data: {"type":"content_block_delta","index":0,'
            b'"delta":{"type":"text_delta","text":"Summary so far: "}}\n\n'
            b"event: content_block_delta\n"
            b'data: {"type":"content_block_delta","index":0,'
            b'"delta":{"type":"text_delta","text":"user asked about caching."}}\n\n'
            b"event: content_block_stop\n"
            b'data: {"type":"content_block_stop","index":0}\n\n'
            b"event: content_block_start\n"
            b'data: {"type":"content_block_start","index":1,'
            b'"content_block":{"type":"text","text":""}}\n\n'
            b"event: content_block_delta\n"
            b'data: {"type":"content_block_delta","index":1,'
            b'"delta":{"type":"text_delta","text":"Here is my answer."}}\n\n'
            b"event: content_block_stop\n"
            b'data: {"type":"content_block_stop","index":1}\n\n'
            b"event: message_delta\n"
            b'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},'
            b'"usage":{"input_tokens":100,"output_tokens":10}}\n\n'
            b"event: message_stop\n"
            b'data: {"type":"message_stop"}\n\n'
        )
        return httpx.Response(
            200,
            content = body,
            headers = {"content-type": "text/event-stream"},
        )

    monkeypatch.setattr(
        ep_mod,
        "_http_client",
        httpx.AsyncClient(transport = httpx.MockTransport(http_handler)),
    )

    client = _make_client()
    lines = _async_collect(
        client._stream_anthropic(
            messages = [{"role": "user", "content": "hi"}],
            model = "claude-opus-4-7",
            temperature = 0.7,
            top_p = 0.95,
            max_tokens = 1024,
            compaction_threshold = 150_000,
        )
    )
    _drive(client.close())

    events = []
    for line in lines:
        if not line.startswith("data:"):
            continue
        raw = line[len("data:") :].strip()
        if not raw or raw == "[DONE]":
            continue
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if "compaction_block" in raw:
            events.append(raw)
    assert events, f"no compaction_block tool event found in {lines}"
    payload = events[0]
    assert "Summary so far: user asked about caching." in payload, payload

    content_text = ""
    for line in lines:
        if not line.startswith("data:"):
            continue
        raw = line[len("data:") :].strip()
        if not raw or raw == "[DONE]":
            continue
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if parsed.get("object") != "chat.completion.chunk":
            continue
        for choice in parsed.get("choices") or []:
            delta = choice.get("delta") or {}
            chunk = delta.get("content")
            if isinstance(chunk, str):
                content_text += chunk
    assert "Summary so far" not in content_text, content_text
    assert "Here is my answer." in content_text, content_text


def test_failed_compaction_block_is_not_replayed_or_persisted(monkeypatch):
    def http_handler(request: httpx.Request) -> httpx.Response:
        body = (
            b"event: message_start\n"
            b'data: {"type":"message_start","message":{"usage":{}}}\n\n'
            b"event: content_block_start\n"
            b'data: {"type":"content_block_start","index":0,'
            b'"content_block":{"type":"compaction","content":null,'
            b'"encrypted_content":"opaque-failed"}}\n\n'
            b"event: content_block_stop\n"
            b'data: {"type":"content_block_stop","index":0}\n\n'
            b"event: message_delta\n"
            b'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},'
            b'"usage":{"input_tokens":100,"output_tokens":1}}\n\n'
            b"event: message_stop\n"
            b'data: {"type":"message_stop"}\n\n'
        )
        return httpx.Response(200, content = body, headers = {"content-type": "text/event-stream"})

    monkeypatch.setattr(
        ep_mod,
        "_http_client",
        httpx.AsyncClient(transport = httpx.MockTransport(http_handler)),
    )
    client = _make_client()
    lines = _async_collect(
        client._stream_anthropic(
            messages = [{"role": "user", "content": "hi"}],
            model = "claude-opus-4-7",
            temperature = 0.7,
            top_p = 0.95,
            max_tokens = 1024,
            compaction_threshold = 150_000,
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "terminal",
                        "description": "Run a command",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
        )
    )
    _drive(client.close())

    payloads = [
        json.loads(line[len("data:") :])
        for line in lines
        if line.startswith("data:") and line[len("data:") :].strip() != "[DONE]"
    ]
    assert not any(
        (payload.get("_toolEvent") or {}).get("type") == "compaction_block" for payload in payloads
    )
    native_blocks = [
        block
        for payload in payloads
        for choice in payload.get("choices") or []
        for block in (
            ((choice.get("delta") or {}).get("extra_content") or {})
            .get("anthropic", {})
            .get("content", [])
        )
    ]
    assert not any(block.get("type") == "compaction" for block in native_blocks)


def test_compaction_block_round_trips_through_outbound_messages(monkeypatch):
    captured: dict = {}

    def http_handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            content = b'event: message_stop\ndata: {"type": "message_stop"}\n\n',
            headers = {"content-type": "text/event-stream"},
        )

    monkeypatch.setattr(
        ep_mod,
        "_http_client",
        httpx.AsyncClient(transport = httpx.MockTransport(http_handler)),
    )

    client = _make_client()

    async def run():
        async for _ in client.stream_chat_completion(
            messages = [
                {"role": "user", "content": "turn 1 question"},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "compaction",
                            "content": "PRIOR SUMMARY: user asked about caching.",
                        },
                        {"type": "text", "text": "Sure, here's an answer."},
                    ],
                },
                {"role": "user", "content": "turn 2 follow-up"},
            ],
            model = "claude-opus-4-7",
            temperature = 0.7,
            top_p = 0.95,
            max_tokens = 32,
            compaction_threshold = 150_000,
        ):
            pass

    _drive(run())
    _drive(client.close())

    msgs = captured["body"]["messages"]
    assistant = next((m for m in msgs if m["role"] == "assistant"), None)
    assert assistant is not None, msgs
    parts = assistant["content"]
    types = [p.get("type") for p in parts if isinstance(p, dict)]
    assert "compaction" in types, parts
    compaction_part = next(p for p in parts if p.get("type") == "compaction")
    assert compaction_part["content"] == "PRIOR SUMMARY: user asked about caching."


def test_compaction_content_part_accepted_by_chat_message_schema():
    # Without the Pydantic Tag the discriminated Union would 422.
    from models.inference import ChatMessage

    msg = ChatMessage.model_validate(
        {
            "role": "assistant",
            "content": [
                {"type": "compaction", "content": "summary text"},
                {"type": "text", "text": "answer prose"},
            ],
        }
    )
    assert isinstance(msg.content, list)
    assert msg.content[0].type == "compaction"
    assert msg.content[0].content == "summary text"
    assert msg.content[1].type == "text"


def test_build_external_messages_passes_compaction_for_anthropic_only():
    from models.inference import ChatMessage
    from routes.inference import _build_external_messages

    msgs = [
        ChatMessage.model_validate(
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "compaction",
                        "content": "prior summary",
                        "encrypted_content": "opaque-compaction",
                    },
                    {"type": "text", "text": "answer"},
                ],
            }
        )
    ]
    out = _build_external_messages(msgs, supports_vision = True, provider_type = "anthropic")
    assert len(out) == 1
    parts = out[0]["content"]
    assert parts[0] == {
        "type": "compaction",
        "content": "prior summary",
        "encrypted_content": "opaque-compaction",
    }
    assert parts[1] == {"type": "text", "text": "answer"}


def test_build_external_messages_strips_compaction_for_non_anthropic_providers():
    # Non-Anthropic validators reject compaction parts, so strip them.
    from models.inference import ChatMessage
    from routes.inference import _build_external_messages

    msgs = [
        ChatMessage.model_validate(
            {
                "role": "assistant",
                "content": [
                    {"type": "compaction", "content": "prior summary"},
                    {"type": "text", "text": "answer"},
                ],
            }
        )
    ]
    for provider in ("openai", "deepseek", "mistral", "gemini", "kimi", "openrouter"):
        out = _build_external_messages(msgs, supports_vision = True, provider_type = provider)
        assert len(out) == 1, (provider, out)
        parts = out[0]["content"]
        types = [p.get("type") for p in parts if isinstance(p, dict)]
        assert "compaction" not in types, (provider, parts)
        assert {"type": "text", "text": "answer"} in parts, (provider, parts)


def test_build_external_messages_strips_compaction_when_provider_type_unknown():
    from models.inference import ChatMessage
    from routes.inference import _build_external_messages

    msgs = [
        ChatMessage.model_validate(
            {
                "role": "assistant",
                "content": [
                    {"type": "compaction", "content": "prior summary"},
                    {"type": "text", "text": "answer"},
                ],
            }
        )
    ]
    out = _build_external_messages(msgs, supports_vision = True)
    parts = out[0]["content"]
    types = [p.get("type") for p in parts if isinstance(p, dict)]
    assert "compaction" not in types, parts


def test_build_external_messages_non_vision_anthropic_keeps_compaction():
    from models.inference import ChatMessage
    from routes.inference import _build_external_messages

    msgs = [
        ChatMessage.model_validate(
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "compaction",
                        "content": "prior summary",
                        "encrypted_content": "opaque-compaction",
                    },
                    {"type": "text", "text": "answer"},
                ],
            }
        )
    ]
    out = _build_external_messages(msgs, supports_vision = False, provider_type = "anthropic")
    parts = out[0]["content"]
    assert {
        "type": "compaction",
        "content": "prior summary",
        "encrypted_content": "opaque-compaction",
    } in parts
    # Non-anthropic + non-vision -> compaction stripped, text collapsed
    # back to a string.
    out2 = _build_external_messages(msgs, supports_vision = False, provider_type = "deepseek")
    assert out2[0]["content"] == "answer", out2


def test_compaction_delta_summary_reaches_the_tool_event(monkeypatch):
    def http_handler(request: httpx.Request) -> httpx.Response:
        body = (
            b"event: content_block_start\n"
            b'data: {"type":"content_block_start","index":0,'
            b'"content_block":{"type":"compaction","content":null}}\n\n'
            b"event: content_block_delta\n"
            b'data: {"type":"content_block_delta","index":0,'
            b'"delta":{"type":"compaction_delta","content":"User is planning a trip.",'
            b'"encrypted_content":"opaque-compaction"}}\n\n'
            b"event: content_block_stop\n"
            b'data: {"type":"content_block_stop","index":0}\n\n'
            b"event: message_stop\n"
            b'data: {"type":"message_stop"}\n\n'
        )
        return httpx.Response(200, content = body, headers = {"content-type": "text/event-stream"})

    monkeypatch.setattr(
        ep_mod,
        "_http_client",
        httpx.AsyncClient(transport = httpx.MockTransport(http_handler)),
    )
    client = _make_client()
    lines = _async_collect(
        client._stream_anthropic(
            messages = [{"role": "user", "content": "hi"}],
            model = "claude-sonnet-4-6",
            temperature = 0.7,
            top_p = 0.95,
            max_tokens = 1024,
            compaction_threshold = 150_000,
        )
    )
    _drive(client.close())

    events = [
        json.loads(line[len("data:") :])["_toolEvent"]
        for line in lines
        if line.startswith("data:") and "compaction_block" in line
    ]
    assert events == [
        {
            "type": "compaction_block",
            "content": "User is planning a trip.",
            "encrypted_content": "opaque-compaction",
        }
    ]


def _replay(monkeypatch, model: str, threshold) -> dict:
    captured: dict = {}

    def http_handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content.decode("utf-8"))
        captured["headers"] = dict(request.headers)
        return httpx.Response(
            200,
            content = b'event: message_stop\ndata: {"type": "message_stop"}\n\n',
            headers = {"content-type": "text/event-stream"},
        )

    monkeypatch.setattr(
        ep_mod,
        "_http_client",
        httpx.AsyncClient(transport = httpx.MockTransport(http_handler)),
    )
    client = _make_client()

    async def run():
        async for _ in client.stream_chat_completion(
            messages = [
                {"role": "user", "content": "turn 1"},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "compaction",
                            "content": "PRIOR SUMMARY",
                            "encrypted_content": "opaque-compaction",
                        },
                        {"type": "text", "text": "answer"},
                    ],
                },
                {"role": "user", "content": "turn 2"},
            ],
            model = model,
            temperature = 0.7,
            top_p = 0.95,
            max_tokens = 32,
            compaction_threshold = threshold,
        ):
            pass

    _drive(run())
    _drive(client.close())
    return captured


def test_a_replayed_compaction_block_keeps_its_beta_with_auto_compact_off(monkeypatch):
    captured = _replay(monkeypatch, "claude-sonnet-4-6", None)
    assistant = captured["body"]["messages"][1]
    assert assistant["content"][0] == {
        "type": "compaction",
        "content": "PRIOR SUMMARY",
        "encrypted_content": "opaque-compaction",
    }
    assert "compact-2026-01-12" in captured["headers"].get("anthropic-beta", "")
    assert "context_management" not in captured["body"]


def test_a_model_without_compaction_is_never_sent_a_compaction_block(monkeypatch):
    captured = _replay(monkeypatch, "claude-haiku-4-5", 150_000)
    assistant = captured["body"]["messages"][1]
    assert [part["type"] for part in assistant["content"]] == ["text"]
    assert "compact-2026-01-12" not in captured["headers"].get("anthropic-beta", "")


def _native_replay(monkeypatch, model: str, threshold) -> dict:
    captured: dict = {}

    def http_handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content.decode("utf-8"))
        captured["headers"] = dict(request.headers)
        return httpx.Response(
            200,
            content = b'event: message_stop\ndata: {"type": "message_stop"}\n\n',
            headers = {"content-type": "text/event-stream"},
        )

    monkeypatch.setattr(
        ep_mod,
        "_http_client",
        httpx.AsyncClient(transport = httpx.MockTransport(http_handler)),
    )
    client = _make_client()
    native = [
        {"type": "compaction", "content": "PRIOR SUMMARY"},
        {"type": "text", "text": "answer"},
    ]

    async def run():
        async for _ in client.stream_chat_completion(
            messages = [
                {"role": "user", "content": "turn 1"},
                {
                    "role": "assistant",
                    "content": "answer",
                    "extra_content": {"anthropic": {"content": native}},
                },
                {"role": "user", "content": "turn 2"},
            ],
            model = model,
            temperature = 0.7,
            top_p = 0.95,
            max_tokens = 32,
            compaction_threshold = threshold,
        ):
            pass

    _drive(run())
    _drive(client.close())
    return captured


def test_a_replayed_native_compaction_block_keeps_its_beta_with_auto_compact_off(monkeypatch):
    captured = _native_replay(monkeypatch, "claude-sonnet-4-6", None)
    assistant = captured["body"]["messages"][1]
    assert assistant["content"][0] == {"type": "compaction", "content": "PRIOR SUMMARY"}
    assert "compact-2026-01-12" in captured["headers"].get("anthropic-beta", "")


def test_a_model_without_compaction_is_never_sent_a_native_compaction_block(monkeypatch):
    captured = _native_replay(monkeypatch, "claude-haiku-4-5", 150_000)
    assistant = captured["body"]["messages"][1]
    assert [part["type"] for part in assistant["content"]] == ["text"]
    assert "compact-2026-01-12" not in captured["headers"].get("anthropic-beta", "")
