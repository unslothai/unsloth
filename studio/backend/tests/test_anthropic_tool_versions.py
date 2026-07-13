# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for the per-model Anthropic tool-version dispatch helpers in
``core.inference.external_provider``.

Anthropic ships date-pinned tool versions per model family; sending the
wrong-dated variant to a model 400s upstream. Pins the dispatch matrix for
web_search/web_fetch/code_execution helpers, the ``_stream_anthropic`` body
integration, and the unchanged code-execution beta header."""

import asyncio
import json

import httpx
import pytest

from core.inference import external_provider as ep_mod
from core.inference.external_provider import (
    ExternalProviderClient,
    _anthropic_code_execution_version,
    _anthropic_web_fetch_version,
    _anthropic_web_search_version,
)


# ── helper-level dispatch matrix ────────────────────────────────────


@pytest.mark.parametrize(
    "model,expected",
    [
        ("claude-opus-4-7", "web_search_20260209"),
        ("claude-opus-4-6", "web_search_20260209"),
        ("claude-sonnet-4-6", "web_search_20260209"),
        ("claude-opus-4-5-20251101", "web_search_20250305"),
        ("claude-sonnet-4-5-20250929", "web_search_20250305"),
        ("claude-haiku-4-5-20251001", "web_search_20250305"),
        ("claude-opus-4-1-20250805", "web_search_20250305"),
        ("claude-opus-4-20250514", "web_search_20250305"),
        ("claude-sonnet-4-20250514", "web_search_20250305"),
        ("claude-3-5-sonnet-20241022", "web_search_20250305"),
    ],
)
def test_web_search_version_dispatch(model, expected):
    assert _anthropic_web_search_version(model) == expected


@pytest.mark.parametrize(
    "model,expected",
    [
        ("claude-opus-4-7", "web_fetch_20260209"),
        ("claude-opus-4-6", "web_fetch_20260209"),
        ("claude-sonnet-4-6", "web_fetch_20260209"),
        ("claude-opus-4-5-20251101", "web_fetch_20250910"),
        ("claude-sonnet-4-5-20250929", "web_fetch_20250910"),
        ("claude-haiku-4-5-20251001", "web_fetch_20250910"),
        ("claude-opus-4-1-20250805", "web_fetch_20250910"),
    ],
)
def test_web_fetch_version_dispatch(model, expected):
    assert _anthropic_web_fetch_version(model) == expected


@pytest.mark.parametrize(
    "model,expected",
    [
        ("claude-opus-4-7", "code_execution_20260120"),
        ("claude-opus-4-6", "code_execution_20260120"),
        ("claude-sonnet-4-6", "code_execution_20260120"),
        ("claude-opus-4-5-20251101", "code_execution_20260120"),
        ("claude-sonnet-4-5-20250929", "code_execution_20260120"),
        # Haiku 4.5 only lists the legacy version in the model table.
        ("claude-haiku-4-5-20251001", "code_execution_20250825"),
        ("claude-opus-4-1-20250805", "code_execution_20250825"),
        # Deprecated 4.0 lineage still works on the legacy version.
        ("claude-opus-4-20250514", "code_execution_20250825"),
        ("claude-sonnet-4-20250514", "code_execution_20250825"),
    ],
)
def test_code_execution_version_dispatch(model, expected):
    assert _anthropic_code_execution_version(model) == expected


# ── streaming integration: outbound body carries the right versions ──


def _drive(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


def _mock_http_client(monkeypatch, handler):
    monkeypatch.setattr(
        ep_mod,
        "_http_client",
        httpx.AsyncClient(transport = httpx.MockTransport(handler)),
    )


def _make_client() -> ExternalProviderClient:
    return ExternalProviderClient(
        provider_type = "anthropic",
        base_url = "https://api.anthropic.com/v1",
        api_key = "sk-ant-test",
    )


def _capture_outbound(monkeypatch, model: str) -> dict:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content.decode("utf-8"))
        captured["headers"] = dict(request.headers)
        return httpx.Response(
            200,
            content = b'event: message_stop\ndata: {"type": "message_stop"}\n\n',
            headers = {"content-type": "text/event-stream"},
        )

    _mock_http_client(monkeypatch, handler)

    async def run():
        client = _make_client()
        async for _ in client._stream_anthropic(
            messages = [{"role": "user", "content": "hi"}],
            model = model,
            temperature = 0.7,
            top_p = 0.95,
            max_tokens = 64,
            enabled_tools = ["web_search", "code_execution"],
        ):
            pass
        await client.close()

    _drive(run())
    return captured


def test_outbound_body_merges_system_messages(monkeypatch):
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content.decode("utf-8")))
        return httpx.Response(
            200,
            content = b'event: message_stop\ndata: {"type": "message_stop"}\n\n',
            headers = {"content-type": "text/event-stream"},
        )

    _mock_http_client(monkeypatch, handler)

    async def run():
        client = _make_client()
        async for _ in client._stream_anthropic(
            messages = [
                {"role": "system", "content": "project instructions"},
                {"role": "system", "content": "saved memory context"},
                {"role": "user", "content": "hi"},
            ],
            model = "claude-opus-4-7",
            temperature = 0.7,
            top_p = 0.95,
            max_tokens = 64,
        ):
            pass
        await client.close()

    _drive(run())
    system = captured["system"]
    system_text = system if isinstance(system, str) else system[0]["text"]
    assert system_text == "project instructions\n\nsaved memory context"
    [user_message] = captured["messages"]
    user_content = user_message["content"]
    user_text = user_content if isinstance(user_content, str) else user_content[0]["text"]
    assert user_message["role"] == "user"
    assert user_text == "hi"


def test_outbound_body_uses_new_versions_on_opus_4_7(monkeypatch):
    captured = _capture_outbound(monkeypatch, "claude-opus-4-7")
    tool_types = {t.get("type") for t in (captured["body"].get("tools") or [])}
    assert "web_search_20260209" in tool_types
    assert "code_execution_20260120" in tool_types
    assert "web_search_20250305" not in tool_types
    assert "code_execution_20250825" not in tool_types
    # One beta header gates both _20250825 and _20260120.
    assert "code-execution-2025-08-25" in captured["headers"].get("anthropic-beta", "")


def test_outbound_body_falls_back_on_haiku_4_5(monkeypatch):
    captured = _capture_outbound(monkeypatch, "claude-haiku-4-5-20251001")
    tool_types = {t.get("type") for t in (captured["body"].get("tools") or [])}
    # Haiku 4.5 only accepts the legacy versions.
    assert "web_search_20250305" in tool_types
    assert "code_execution_20250825" in tool_types
    assert "web_search_20260209" not in tool_types
    assert "code_execution_20260120" not in tool_types


def test_outbound_body_mixes_versions_on_sonnet_4_5(monkeypatch):
    # Sonnet 4.5 gets the new code_execution but the old web_search.
    captured = _capture_outbound(monkeypatch, "claude-sonnet-4-5-20250929")
    tool_types = {t.get("type") for t in (captured["body"].get("tools") or [])}
    assert "web_search_20250305" in tool_types
    assert "code_execution_20260120" in tool_types
    assert "web_search_20260209" not in tool_types
    assert "code_execution_20250825" not in tool_types
