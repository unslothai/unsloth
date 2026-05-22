# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unit tests for Anthropic server-side context compaction wiring.

Compaction is a beta feature (header ``compact-2026-01-12``) gated to
Opus 4.6, Opus 4.7, Sonnet 4.6, and Mythos preview. When enabled,
Studio attaches ``context_management.edits[{type:"compact_20260112",
trigger:{type:"input_tokens", value:N}}]`` to the outbound body. The
minimum upstream-accepted threshold is 50k tokens; lower values are
clamped to 50k so the request doesn't 400.

These tests pin: the body shape per model, the beta header merge with
the existing code-execution beta, threshold clamping, and silent no-op
on unsupported models.
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


def _capture(monkeypatch, model: str, threshold, tools = None) -> dict:
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
    assert (
        captured["body"]["context_management"]["edits"][0]["trigger"]["value"] == 60_000
    )
    captured = _capture(monkeypatch, "claude-opus-4-7", 1)
    assert (
        captured["body"]["context_management"]["edits"][0]["trigger"]["value"] == 50_000
    )


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
    assert "compact-2026-01-12" not in captured["headers"].get(
        "anthropic-beta",
        "",
    )


# ── omitted threshold leaves body untouched ─────────────────────────


def test_omitted_threshold_no_body_field(monkeypatch):
    captured = _capture(monkeypatch, "claude-opus-4-7", None)
    assert "context_management" not in captured["body"]
    assert "compact-2026-01-12" not in captured["headers"].get(
        "anthropic-beta",
        "",
    )


# ── ChatCompletionRequest schema accepts sub-50k threshold ──────────


def test_chat_completion_request_accepts_sub_50k_compaction_threshold():
    # Codex P1 caught that ge=50_000 on the field caused FastAPI to
    # 422 the request before the in-helper clamp could fire. The
    # schema must accept any positive int and let _stream_anthropic
    # clamp upward.
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


def test_message_delta_iterations_array_aggregates_compaction_tokens(
    monkeypatch, capsys
):
    # When Anthropic compacts mid-stream, the SSE message_delta usage
    # payload carries `iterations: [{type:"compaction", ...}, ...]`.
    # The top-level input_tokens / output_tokens only account for the
    # `message` iteration, so the cost surface needs the compaction
    # totals exposed separately. The stream helper folds them into
    # last_usage as `compaction_input_tokens` / `compaction_output_tokens`
    # and surfaces them in the closing summary log so an operator can
    # eyeball "did compaction cost us 180k tokens this turn?".

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

    # structlog renders the closing summary through the stdlib bridge,
    # which lands on stdout. Capture and check the rendered line.
    out = capsys.readouterr().out
    summary = next(
        (line for line in out.splitlines() if "Anthropic stream complete" in line),
        "",
    )
    assert "compaction_input_tokens=180000" in summary, summary
    assert "compaction_output_tokens=3500" in summary, summary


def test_message_delta_no_iterations_leaves_compaction_keys_unset(monkeypatch, capsys):
    # Re-applying a previous compaction block does NOT emit a fresh
    # iterations array. The helper must not invent compaction keys
    # in that case (would otherwise double-bill).
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
