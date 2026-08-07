# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unit tests for Anthropic server-side context compaction wiring.

Compaction is a beta (header ``compact-2026-01-12``) gated to Opus 4.6/4.7,
Sonnet 4.6, and Mythos preview. When enabled, Unsloth attaches
``context_management.edits[{type:"compact_20260112", trigger:{type:"input_tokens",
value:N}}]``; the 50k-token minimum is clamped up so the request doesn't 400.

Pins body shape per model, beta header merge with code-execution, threshold
clamping, and silent no-op on unsupported models.
"""

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


# ── support gate matches the doc table ───────────────────────────────


@pytest.mark.parametrize(
    "model, supported",
    [
        ("claude-opus-4-7", True),
        ("claude-opus-4-6", True),
        ("claude-sonnet-4-6", True),
        ("claude-mythos-preview", True),
        # NOT supported per the docs.
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


# ── outbound shape on supported model ────────────────────────────────


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
    # Below-min values get clamped UP so we don't 400 upstream.
    captured = _capture(monkeypatch, "claude-opus-4-7", 60_000)
    assert captured["body"]["context_management"]["edits"][0]["trigger"]["value"] == 60_000
    captured = _capture(monkeypatch, "claude-opus-4-7", 1)
    assert captured["body"]["context_management"]["edits"][0]["trigger"]["value"] == 50_000


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


# ── silent no-op on unsupported model ────────────────────────────────


def test_unsupported_model_silently_drops_compaction(monkeypatch):
    captured = _capture(monkeypatch, "claude-haiku-4-5-20251001", 150_000)
    assert "context_management" not in captured["body"]
    # The beta header must not carry compact-2026-01-12 either.
    assert "compact-2026-01-12" not in captured["headers"].get("anthropic-beta", "")


# ── omitted threshold leaves body untouched ─────────────────────────


def test_omitted_threshold_no_body_field(monkeypatch):
    captured = _capture(monkeypatch, "claude-opus-4-7", None)
    assert "context_management" not in captured["body"]
    assert "compact-2026-01-12" not in captured["headers"].get("anthropic-beta", "")


# ── ChatCompletionRequest schema accepts sub-50k threshold ──────────


def test_chat_completion_request_accepts_sub_50k_compaction_threshold():
    # ge=50_000 on the field would 422 before the in-helper clamp fires; the
    # schema must accept any positive int and let _stream_anthropic clamp up.
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

    # Non-positive values are still rejected so blank-string posts
    # don't sneak through.
    with pytest.raises(Exception):
        ChatCompletionRequest.model_validate(
            {
                "model": "default",
                "messages": [{"role": "user", "content": "hi"}],
                "compaction_threshold": 0,
            }
        )


# ── usage.iterations[] surfaces compaction tokens ──────────────────


def test_message_delta_iterations_array_aggregates_compaction_tokens(monkeypatch, capsys):
    # On mid-stream compaction the message_delta usage carries
    # `iterations: [{type:"compaction", ...}, ...]`. Top-level tokens only cover
    # the `message` iteration, so the helper folds compaction totals into
    # last_usage and surfaces them in the closing summary log.

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

    # structlog renders the closing summary onto stdout; check the rendered line.
    out = capsys.readouterr().out
    summary = next(
        (line for line in out.splitlines() if "Anthropic stream complete" in line),
        "",
    )
    assert "compaction_input_tokens=180000" in summary, summary
    assert "compaction_output_tokens=3500" in summary, summary


def test_message_delta_no_iterations_leaves_compaction_keys_unset(monkeypatch, capsys):
    # Re-applying a prior compaction block emits no fresh iterations array;
    # the helper must not invent compaction keys (would double-bill).
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


# ── compaction block round-trip (Codex P1) ──────────────────────────


def _async_collect(agen):
    async def run():
        out = []
        async for line in agen:
            out.append(line)
        return out

    return _drive(run())


def test_compaction_block_emitted_as_tool_event(monkeypatch):
    # When Anthropic compacts during a turn, the response carries a
    # `{type:"compaction", content:"<summary>"}` block. The translator must surface
    # it so the chat-adapter persists it; else the next turn loses state and re-compacts.

    def http_handler(request: httpx.Request) -> httpx.Response:
        # Compaction blocks arrive as a content_block_start with type:"compaction",
        # with the summary on the start event AND/OR streamed via text_delta on the
        # same index. Test the streamed-delta path (harder case).
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

    # Pull tool_events out of the SSE stream and check for the
    # compaction_block payload.
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
        # tool_event payloads ride inside the chunk as JSON; just match the marker.
        if "compaction_block" in raw:
            events.append(raw)
    assert events, f"no compaction_block tool event found in {lines}"
    # The summary text must come through intact.
    payload = events[0]
    assert "Summary so far: user asked about caching." in payload, payload

    # The user-visible content stream must NOT carry the compaction
    # summary -- only the assistant prose ("Here is my answer.").
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


def test_compaction_block_round_trips_through_outbound_messages(monkeypatch):
    # The next turn's outbound body must forward a persisted
    # {type:"compaction", content:"..."} block verbatim so the API recognises the state.
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
    # The assistant turn must include the compaction block on the wire.
    assistant = next((m for m in msgs if m["role"] == "assistant"), None)
    assert assistant is not None, msgs
    parts = assistant["content"]
    types = [p.get("type") for p in parts if isinstance(p, dict)]
    assert "compaction" in types, parts
    compaction_part = next(p for p in parts if p.get("type") == "compaction")
    assert compaction_part["content"] == "PRIOR SUMMARY: user asked about caching."


def test_compaction_content_part_accepted_by_chat_message_schema():
    # Without the Pydantic Tag the discriminated Union would 422 at parse time.
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
    # Compaction is Anthropic-only; the builder must gate it on
    # provider_type=="anthropic" since others 400 on the unknown content type.
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
    out = _build_external_messages(msgs, supports_vision = True, provider_type = "anthropic")
    assert len(out) == 1
    parts = out[0]["content"]
    assert parts[0] == {"type": "compaction", "content": "prior summary"}
    assert parts[1] == {"type": "text", "text": "answer"}


def test_build_external_messages_strips_compaction_for_non_anthropic_providers():
    # A provider switch or reused history can hand compaction blocks to a
    # non-Anthropic provider whose validator rejects the unknown type, so the
    # builder must strip the part for every non-anthropic provider.
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
        # Text part survives.
        assert {"type": "text", "text": "answer"} in parts, (provider, parts)


def test_build_external_messages_strips_compaction_when_provider_type_unknown():
    # Defensive: provider_type=None (legacy path) must also strip the part.
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
    # Defensive: gate the non-vision branch by provider_type too, so future
    # config changes don't drop compaction for Anthropic.
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
    out = _build_external_messages(msgs, supports_vision = False, provider_type = "anthropic")
    parts = out[0]["content"]
    assert {"type": "compaction", "content": "prior summary"} in parts
    # Non-anthropic + non-vision -> compaction stripped, text collapsed
    # back to a string.
    out2 = _build_external_messages(msgs, supports_vision = False, provider_type = "deepseek")
    assert out2[0]["content"] == "answer", out2
