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
        provider_type="anthropic",
        base_url="https://api.anthropic.com/v1",
        api_key="sk-ant-test",
    )


def _capture(monkeypatch, model: str, threshold, tools=None) -> dict:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content.decode("utf-8"))
        captured["headers"] = dict(request.headers)
        return httpx.Response(
            200,
            content=b"event: message_stop\ndata: {\"type\": \"message_stop\"}\n\n",
            headers={"content-type": "text/event-stream"},
        )

    monkeypatch.setattr(
        ep_mod,
        "_http_client",
        httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )

    async def run():
        client = _make_client()
        async for _ in client.stream_chat_completion(
            messages=[{"role": "user", "content": "hi"}],
            model=model,
            temperature=0.7,
            top_p=0.95,
            max_tokens=32,
            enabled_tools=tools,
            compaction_threshold=threshold,
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
    assert captured["body"]["context_management"]["edits"][0][
        "trigger"
    ]["value"] == 60_000
    captured = _capture(monkeypatch, "claude-opus-4-7", 1)
    assert captured["body"]["context_management"]["edits"][0][
        "trigger"
    ]["value"] == 50_000


# ── beta header merge with code execution ────────────────────────────


def test_compaction_beta_merges_with_code_execution_beta(monkeypatch):
    captured = _capture(
        monkeypatch,
        "claude-opus-4-7",
        150_000,
        tools=["code_execution"],
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
        "anthropic-beta", "",
    )


# ── omitted threshold leaves body untouched ─────────────────────────


def test_omitted_threshold_no_body_field(monkeypatch):
    captured = _capture(monkeypatch, "claude-opus-4-7", None)
    assert "context_management" not in captured["body"]
    assert "compact-2026-01-12" not in captured["headers"].get(
        "anthropic-beta", "",
    )
