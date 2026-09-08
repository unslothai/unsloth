# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for the Anthropic ``model_context_window_exceeded`` stop reason.

The reply filled the model's context window, so the turn is truncated: it maps
to OpenAI's ``finish_reason: "length"`` and carries an out-of-band
``_toolEvent`` telling the frontend this particular cut cannot be resumed.
Sonnet 4.5 and newer return the stop reason without a beta header.
https://platform.claude.com/docs/en/api/handling-stop-reasons
"""

import asyncio
import json

import httpx

from core.inference import external_provider as ep_mod
from core.inference.external_provider import ExternalProviderClient


def _drive(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


def _sse(stop_reason: str) -> bytes:
    return (
        b'event: message_start\ndata: {"type":"message_start","message":'
        b'{"id":"m1","content":[],"model":"claude-sonnet-4-5","role":"assistant",'
        b'"stop_reason":null,"usage":{"input_tokens":1,"output_tokens":1}}}\n\n'
        b'event: content_block_start\ndata: {"type":"content_block_start",'
        b'"index":0,"content_block":{"type":"text","text":""}}\n\n'
        b'event: content_block_delta\ndata: {"type":"content_block_delta",'
        b'"index":0,"delta":{"type":"text_delta","text":"Half an ans"}}\n\n'
        b'event: content_block_stop\ndata: {"type":"content_block_stop","index":0}\n\n'
        b'event: message_delta\ndata: {"type":"message_delta",'
        b'"delta":{"stop_reason":"' + stop_reason.encode() + b'"}}\n\n'
        b'event: message_stop\ndata: {"type":"message_stop"}\n\n'
    )


def _lines(monkeypatch, sse: bytes) -> list[str]:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content = sse,
            headers = {"content-type": "text/event-stream"},
        )

    monkeypatch.setattr(
        ep_mod,
        "_http_client",
        httpx.AsyncClient(transport = httpx.MockTransport(handler)),
    )

    out_lines: list[str] = []

    async def run():
        client = ExternalProviderClient(
            provider_type = "anthropic",
            base_url = "https://api.anthropic.com/v1",
            api_key = "sk-ant-test",
        )
        try:
            async for line in client.stream_chat_completion(
                messages = [{"role": "user", "content": "hi"}],
                model = "claude-sonnet-4-5",
                temperature = 0.7,
                top_p = 0.95,
                max_tokens = 32,
            ):
                out_lines.append(line)
        finally:
            await client.close()

    _drive(run())
    return out_lines


def _finish_reasons(lines: list[str]) -> list[str]:
    seen: list[str] = []
    for line in lines:
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        payload = json.loads(line[len("data: ") :])
        for choice in payload.get("choices") or []:
            reason = choice.get("finish_reason")
            if reason:
                seen.append(reason)
    return seen


def test_context_window_exceeded_finishes_as_length(monkeypatch):
    """The window filled up mid-answer, so the turn is truncated. Mapping it to
    "stop" tells every OpenAI-compatible client the answer is complete."""
    lines = _lines(monkeypatch, _sse("model_context_window_exceeded"))
    assert _finish_reasons(lines) == ["length"], lines
    assert "Half an ans" in "\n".join(lines), lines


def test_context_window_exceeded_emits_tool_event(monkeypatch):
    """`length` alone arms the automatic continuation, which would replay the
    partial as more prompt against the window that just overflowed. The event
    is what tells the frontend to show the notice without resuming."""
    body = "\n".join(_lines(monkeypatch, _sse("model_context_window_exceeded")))
    assert '"_toolEvent": {"type": "context_window_exceeded"}' in body, body


def test_max_tokens_still_finishes_as_length_without_the_event(monkeypatch):
    """The resumable cut is unchanged: hitting the caller's cap leaves room to
    continue into."""
    body = "\n".join(_lines(monkeypatch, _sse("max_tokens")))
    assert '"finish_reason": "length"' in body, body
    assert "context_window_exceeded" not in body, body


def test_end_turn_is_still_a_completed_answer(monkeypatch):
    body = "\n".join(_lines(monkeypatch, _sse("end_turn")))
    assert '"finish_reason": "stop"' in body, body
    assert "context_window_exceeded" not in body, body
