# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for Anthropic fast-mode wiring and streaming refusal handling.

fast_mode=True on Opus 4.6/4.7 attaches the ``fast-mode-2026-02-01`` beta
header and sets ``speed: "fast"``; unsupported models drop both. Streaming
``stop_reason: "refusal"`` surfaces a user notice before the
``content_filter`` finish chunk.
https://platform.claude.com/docs/en/test-and-evaluate/strengthen-guardrails/handle-streaming-refusals
"""

import asyncio
import json

import httpx

from core.inference import external_provider as ep_mod
from core.inference.external_provider import ExternalProviderClient


def _drive(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


def _make_client() -> ExternalProviderClient:
    return ExternalProviderClient(
        provider_type = "anthropic",
        base_url = "https://api.anthropic.com/v1",
        api_key = "sk-ant-test",
    )


def _empty_message_sse() -> bytes:
    return (
        b'event: message_start\ndata: {"type":"message_start","message":'
        b'{"id":"m1","content":[],"model":"claude-opus-4-7","role":"assistant",'
        b'"stop_reason":null,"usage":{"input_tokens":1,"output_tokens":1}}}\n\n'
        b'event: message_delta\ndata: {"type":"message_delta",'
        b'"delta":{"stop_reason":"end_turn"}}\n\n'
        b'event: message_stop\ndata: {"type":"message_stop"}\n\n'
    )


def _refusal_sse() -> bytes:
    return (
        b'event: message_start\ndata: {"type":"message_start","message":'
        b'{"id":"m1","content":[],"model":"claude-opus-4-7","role":"assistant",'
        b'"stop_reason":null,"usage":{"input_tokens":1,"output_tokens":1}}}\n\n'
        b'event: content_block_start\ndata: {"type":"content_block_start",'
        b'"index":0,"content_block":{"type":"text","text":""}}\n\n'
        b'event: content_block_delta\ndata: {"type":"content_block_delta",'
        b'"index":0,"delta":{"type":"text_delta","text":"Hello."}}\n\n'
        b'event: content_block_stop\ndata: {"type":"content_block_stop","index":0}\n\n'
        b'event: message_delta\ndata: {"type":"message_delta",'
        b'"delta":{"stop_reason":"refusal"}}\n\n'
        b'event: message_stop\ndata: {"type":"message_stop"}\n\n'
    )


def _capture(
    monkeypatch,
    sse: bytes = b"",
    **kwargs,
) -> tuple[dict, list[str]]:
    """Install a MockTransport, drive one streamed call; return body+lines."""
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content.decode("utf-8"))
        captured["headers"] = dict(request.headers)
        return httpx.Response(
            200,
            content = sse or _empty_message_sse(),
            headers = {"content-type": "text/event-stream"},
        )

    monkeypatch.setattr(
        ep_mod,
        "_http_client",
        httpx.AsyncClient(transport = httpx.MockTransport(handler)),
    )

    out_lines: list[str] = []

    async def run():
        client = _make_client()
        try:
            async for line in client.stream_chat_completion(
                messages = [{"role": "user", "content": "hi"}],
                model = kwargs.get("model", "claude-opus-4-7"),
                temperature = 0.7,
                top_p = 0.95,
                max_tokens = 32,
                fast_mode = kwargs.get("fast_mode"),
            ):
                out_lines.append(line)
        finally:
            await client.close()

    _drive(run())
    return captured, out_lines


def test_fast_mode_attaches_beta_header_and_speed_on_opus_4_7(monkeypatch):
    cap, _ = _capture(monkeypatch, fast_mode = True, model = "claude-opus-4-7")
    assert cap["body"].get("speed") == "fast", cap["body"]
    beta = cap["headers"].get("anthropic-beta", "")
    assert "fast-mode-2026-02-01" in beta, beta


def test_fast_mode_attaches_beta_header_and_speed_on_opus_4_6(monkeypatch):
    cap, _ = _capture(monkeypatch, fast_mode = True, model = "claude-opus-4-6")
    assert cap["body"].get("speed") == "fast", cap["body"]
    assert "fast-mode-2026-02-01" in cap["headers"].get("anthropic-beta", "")


def test_fast_mode_dropped_on_sonnet(monkeypatch):
    cap, _ = _capture(monkeypatch, fast_mode = True, model = "claude-sonnet-4-6")
    assert "speed" not in cap["body"], cap["body"]
    assert "fast-mode-2026-02-01" not in cap["headers"].get("anthropic-beta", "")


def test_fast_mode_dropped_on_haiku(monkeypatch):
    cap, _ = _capture(monkeypatch, fast_mode = True, model = "claude-haiku-4-5")
    assert "speed" not in cap["body"], cap["body"]
    assert "fast-mode-2026-02-01" not in cap["headers"].get("anthropic-beta", "")


def test_fast_mode_dropped_on_older_opus(monkeypatch):
    cap, _ = _capture(monkeypatch, fast_mode = True, model = "claude-opus-4-5")
    assert "speed" not in cap["body"], cap["body"]


def test_fast_mode_false_does_not_attach_header_or_field(monkeypatch):
    cap, _ = _capture(monkeypatch, fast_mode = False)
    assert "speed" not in cap["body"], cap["body"]
    assert "fast-mode-2026-02-01" not in cap["headers"].get("anthropic-beta", "")


def test_fast_mode_none_does_not_attach_header_or_field(monkeypatch):
    cap, _ = _capture(monkeypatch, fast_mode = None)
    assert "speed" not in cap["body"], cap["body"]
    assert "fast-mode-2026-02-01" not in cap["headers"].get("anthropic-beta", "")


def test_refusal_emits_user_facing_notice_and_content_filter_finish(monkeypatch):
    _, lines = _capture(monkeypatch, sse = _refusal_sse())
    body = "\n".join(lines)
    # User-visible refusal notice.
    assert "stopped by Anthropic's safety classifier" in body, body
    # OpenAI-spec finish_reason mapping.
    assert '"finish_reason": "content_filter"' in body, body
    # Original deltas preserved before the refusal supplement.
    assert "Hello." in body, body


def test_refusal_emits_tool_event_for_chat_adapter_drop(monkeypatch):
    """Refused turns emit an out-of-band `_toolEvent` that the chat-adapter
    latches into assistant `metadata.custom.anthropicRefusal`, driving the
    next-request prune. Tool event (not text) prevents spoofing.
    """
    _, lines = _capture(monkeypatch, sse = _refusal_sse())
    body = "\n".join(lines)
    assert '"_toolEvent": {"type": "anthropic_refusal"}' in body, body
    # Visible refusal text must not embed a sentinel that could spoof a
    # context reset if echoed by another assistant message.
    assert "studio:anthropic-refusal" not in body, body
