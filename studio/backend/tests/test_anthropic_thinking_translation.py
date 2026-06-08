# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Unit tests for the Anthropic extended-thinking translation in
external_provider.

Covers:
- Adaptive-mode request body nests effort under
  ``output_config: {effort: "<level>"}`` per the Messages API
  reference (a top-level ``effort`` field 400s with
  "effort: Extra inputs are not permitted").
- Streaming SSE: ``content_block_delta`` with
  ``delta.type == "thinking_delta"`` is translated into inline
  ``<think>...</think>`` chat-completion chunks so the frontend's
  reasoning-panel pipeline lifts it correctly.
- The ``<think>`` tag closes when the first ``text_delta`` arrives,
  on ``content_block_stop``, on ``message_delta``, or on
  ``message_stop``.
- Thinking is paired with ``temperature=1`` and no ``top_p`` /
  ``top_k`` on the wire (Anthropic extended-thinking contract).
"""

import asyncio
import json

import httpx

from core.inference import external_provider as ep_mod
from core.inference.external_provider import ExternalProviderClient


def _drive(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


async def _collect(agen):
    out = []
    async for line in agen:
        out.append(line)
    return out


def _mock_http_client(monkeypatch, handler):
    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(ep_mod, "_http_client", httpx.AsyncClient(transport = transport))


def _make_client() -> ExternalProviderClient:
    return ExternalProviderClient(
        provider_type = "anthropic",
        base_url = "https://api.anthropic.com/v1",
        api_key = "sk-ant-test",
    )


def _anthropic_sse(events: list[dict]) -> bytes:
    """Serialize a list of Messages-API event dicts as an SSE byte stream."""
    chunks: list[str] = []
    for event in events:
        chunks.append(f"event: {event['type']}")
        chunks.append(f"data: {json.dumps(event)}")
        chunks.append("")
    return ("\n".join(chunks) + "\n").encode("utf-8")


def _payloads_from_lines(lines: list[str]) -> list:
    out = []
    for line in lines:
        if not line.startswith("data:"):
            continue
        raw = line[len("data:") :].strip()
        if not raw:
            continue
        if raw == "[DONE]":
            out.append("[DONE]")
        else:
            out.append(json.loads(raw))
    return out


def test_adaptive_thinking_body_uses_output_config_effort_shape(monkeypatch):
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            content = _anthropic_sse([{"type": "message_stop"}]),
            headers = {"content-type": "text/event-stream"},
        )

    _mock_http_client(monkeypatch, handler)

    async def run():
        client = _make_client()
        async for _ in client._stream_anthropic(
            messages = [{"role": "user", "content": "hi"}],
            model = "claude-opus-4-6",
            temperature = 0.7,
            top_p = 0.95,
            max_tokens = 4096,
            top_k = None,
            enable_thinking = None,
            reasoning_effort = "medium",
        ):
            pass
        await client.close()

    _drive(run())

    body = captured["body"]
    # display=summarized is set explicitly so Opus 4.7 (which defaults to
    # "omitted") still emits thinking_delta events for the reasoning panel.
    assert body["thinking"] == {"type": "adaptive", "display": "summarized"}
    # Documented shape: effort is nested under output_config.
    # A top-level `effort` field produces a 400:
    #   "effort: Extra inputs are not permitted".
    assert body["output_config"] == {"effort": "medium"}
    assert "effort" not in body
    # Extended-thinking contract: temperature=1, no top_p / top_k.
    assert body["temperature"] == 1
    assert "top_p" not in body
    assert "top_k" not in body


def test_adaptive_thinking_maps_xhigh_to_max_on_claude_4_6(monkeypatch):
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            content = _anthropic_sse([{"type": "message_stop"}]),
            headers = {"content-type": "text/event-stream"},
        )

    _mock_http_client(monkeypatch, handler)

    async def run():
        client = _make_client()
        async for _ in client._stream_anthropic(
            messages = [{"role": "user", "content": "hi"}],
            model = "claude-sonnet-4-6",
            temperature = 0.7,
            top_p = 0.95,
            max_tokens = 4096,
            top_k = None,
            enable_thinking = None,
            reasoning_effort = "xhigh",
        ):
            pass
        await client.close()

    _drive(run())

    assert captured["body"]["output_config"] == {"effort": "max"}


def test_adaptive_thinking_keeps_max_on_claude_4_6(monkeypatch):
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            content = _anthropic_sse([{"type": "message_stop"}]),
            headers = {"content-type": "text/event-stream"},
        )

    _mock_http_client(monkeypatch, handler)

    async def run():
        client = _make_client()
        async for _ in client._stream_anthropic(
            messages = [{"role": "user", "content": "hi"}],
            model = "claude-opus-4-6",
            temperature = 0.7,
            top_p = 0.95,
            max_tokens = 4096,
            top_k = None,
            enable_thinking = None,
            reasoning_effort = "max",
        ):
            pass
        await client.close()

    _drive(run())

    assert captured["body"]["output_config"] == {"effort": "max"}


def test_adaptive_thinking_keeps_xhigh_on_claude_4_7(monkeypatch):
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            content = _anthropic_sse([{"type": "message_stop"}]),
            headers = {"content-type": "text/event-stream"},
        )

    _mock_http_client(monkeypatch, handler)

    async def run():
        client = _make_client()
        async for _ in client._stream_anthropic(
            messages = [{"role": "user", "content": "hi"}],
            model = "claude-opus-4-7",
            temperature = 0.7,
            top_p = 0.95,
            max_tokens = 4096,
            top_k = None,
            enable_thinking = None,
            reasoning_effort = "xhigh",
        ):
            pass
        await client.close()

    _drive(run())

    body = captured["body"]
    assert body["output_config"] == {"effort": "xhigh"}
    assert "effort" not in body


def test_manual_thinking_body_uses_budget_tokens_on_4_5(monkeypatch):
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            content = _anthropic_sse([{"type": "message_stop"}]),
            headers = {"content-type": "text/event-stream"},
        )

    _mock_http_client(monkeypatch, handler)

    async def run():
        client = _make_client()
        async for _ in client._stream_anthropic(
            messages = [{"role": "user", "content": "hi"}],
            model = "claude-opus-4-5",
            temperature = 0.7,
            top_p = 0.95,
            max_tokens = 1024,
            top_k = None,
            enable_thinking = None,
            reasoning_effort = "high",
        ):
            pass
        await client.close()

    _drive(run())

    body = captured["body"]
    assert body["thinking"] == {"type": "enabled", "budget_tokens": 4096}
    # max_tokens must be strictly greater than budget_tokens; we shipped 1024
    # and budget is 4096, so the wrapper should bump max_tokens.
    assert body["max_tokens"] > body["thinking"]["budget_tokens"]
    # Manual-thinking path does not use output_config / effort — those are
    # the adaptive-mode controls (Claude 4.6 / 4.7).
    assert "effort" not in body
    assert "output_config" not in body


def test_thinking_delta_wrapped_in_think_tags(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        events = [
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "thinking", "thinking": "", "signature": ""},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "thinking_delta", "thinking": "First "},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "thinking_delta", "thinking": "I plan."},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "signature_delta", "signature": "abc123"},
            },
            {"type": "content_block_stop", "index": 0},
            {
                "type": "content_block_start",
                "index": 1,
                "content_block": {"type": "text", "text": ""},
            },
            {
                "type": "content_block_delta",
                "index": 1,
                "delta": {"type": "text_delta", "text": "Answer."},
            },
            {"type": "content_block_stop", "index": 1},
            {"type": "message_delta", "delta": {"stop_reason": "end_turn"}},
            {"type": "message_stop"},
        ]
        return httpx.Response(
            200,
            content = _anthropic_sse(events),
            headers = {"content-type": "text/event-stream"},
        )

    _mock_http_client(monkeypatch, handler)

    async def run():
        client = _make_client()
        lines = await _collect(
            client._stream_anthropic(
                messages = [{"role": "user", "content": "hi"}],
                model = "claude-opus-4-6",
                temperature = 0.7,
                top_p = 0.95,
                max_tokens = 4096,
                top_k = None,
                enable_thinking = True,
                reasoning_effort = None,
            )
        )
        await client.close()
        return lines

    lines = _drive(run())
    payloads = _payloads_from_lines(lines)

    combined = "".join(
        p["choices"][0]["delta"].get("content", "")
        for p in payloads
        if isinstance(p, dict) and p["choices"][0]["delta"]
    )

    # Reasoning text should be wrapped in <think>...</think>, followed by the
    # answer text, and the stream should terminate with [DONE].
    assert "<think>First I plan.</think>" in combined
    assert combined.endswith("Answer.")
    # signature_delta is intentionally dropped — no leaked signature text.
    assert "abc123" not in combined
    assert "[DONE]" in payloads


def test_thinking_only_turn_closes_tag_without_text_delta(monkeypatch):
    """display=omitted on Claude 4.7 emits a signature_delta and no text.

    The <think> open is still triggered by the (synthetic) thinking_delta;
    we want content_block_stop to close it cleanly so the tag never leaks
    into the next chunk."""

    def handler(request: httpx.Request) -> httpx.Response:
        events = [
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "thinking", "thinking": "", "signature": ""},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "thinking_delta", "thinking": "internal"},
            },
            {"type": "content_block_stop", "index": 0},
            {"type": "message_delta", "delta": {"stop_reason": "end_turn"}},
            {"type": "message_stop"},
        ]
        return httpx.Response(
            200,
            content = _anthropic_sse(events),
            headers = {"content-type": "text/event-stream"},
        )

    _mock_http_client(monkeypatch, handler)

    async def run():
        client = _make_client()
        lines = await _collect(
            client._stream_anthropic(
                messages = [{"role": "user", "content": "hi"}],
                model = "claude-opus-4-7",
                temperature = 0.7,
                top_p = 0.95,
                max_tokens = 4096,
                top_k = None,
                enable_thinking = True,
                reasoning_effort = None,
            )
        )
        await client.close()
        return lines

    payloads = _payloads_from_lines(_drive(run()))
    combined = "".join(
        p["choices"][0]["delta"].get("content", "")
        for p in payloads
        if isinstance(p, dict) and p["choices"][0]["delta"]
    )
    assert combined == "<think>internal</think>"
