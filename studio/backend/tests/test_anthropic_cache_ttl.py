# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Pins the cache_control ttl body shape: 1h on both markers, omitted by default, garbage dropped."""

import asyncio
import json

import httpx
import pytest

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


def _capture(monkeypatch, ttl = None) -> dict:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content.decode("utf-8"))
        captured["headers"] = dict(request.headers)
        return httpx.Response(
            200,
            content = (b"event: message_stop\n" b'data: {"type": "message_stop"}\n\n'),
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
            messages = [
                {"role": "system", "content": "Be brief."},
                {"role": "user", "content": "hi"},
            ],
            model = "claude-opus-4-7",
            temperature = 0.7,
            top_p = 0.95,
            max_tokens = 32,
            enable_prompt_caching = True,
            prompt_cache_ttl = ttl,
        ):
            pass
        await client.close()

    _drive(run())
    return captured


def _cache_controls(body: dict) -> list[dict]:
    """Pull every cache_control marker from the system block + tail message."""
    out = []
    sys_blocks = body.get("system") or []
    if isinstance(sys_blocks, list):
        for b in sys_blocks:
            if isinstance(b, dict) and "cache_control" in b:
                out.append(b["cache_control"])
    msgs = body.get("messages") or []
    if msgs:
        tail = msgs[-1].get("content")
        if isinstance(tail, list):
            for b in tail:
                if isinstance(b, dict) and "cache_control" in b:
                    out.append(b["cache_control"])
    return out


def test_omitted_ttl_uses_default_5m_pool(monkeypatch):
    captured = _capture(monkeypatch, ttl = None)
    ccs = _cache_controls(captured["body"])
    assert len(ccs) == 2, ccs
    for cc in ccs:
        assert cc == {"type": "ephemeral"}, cc


def test_explicit_5m_ttl_round_trips(monkeypatch):
    captured = _capture(monkeypatch, ttl = "5m")
    ccs = _cache_controls(captured["body"])
    assert len(ccs) == 2, ccs
    for cc in ccs:
        assert cc == {"type": "ephemeral", "ttl": "5m"}, cc


def test_1h_ttl_writes_into_1h_pool(monkeypatch):
    captured = _capture(monkeypatch, ttl = "1h")
    ccs = _cache_controls(captured["body"])
    assert len(ccs) == 2, ccs
    for cc in ccs:
        assert cc == {"type": "ephemeral", "ttl": "1h"}, cc


def test_1h_ttl_does_not_send_extended_cache_ttl_beta_header(monkeypatch):
    # The 1h TTL beta header is GA; fail if it is re-added.
    captured = _capture(monkeypatch, ttl = "1h")
    beta = captured["headers"].get("anthropic-beta", "")
    assert "extended-cache-ttl-2025-04-11" not in beta, beta


def test_5m_ttl_does_not_send_extended_cache_ttl_beta_header(monkeypatch):
    captured = _capture(monkeypatch, ttl = "5m")
    beta = captured["headers"].get("anthropic-beta", "")
    assert "extended-cache-ttl-2025-04-11" not in beta, beta


@pytest.mark.parametrize("bogus", ["6m", "2h", "", "forever", "1d", "0", "1"])
def test_unknown_ttl_silently_dropped(monkeypatch, bogus):
    captured = _capture(monkeypatch, ttl = bogus)
    ccs = _cache_controls(captured["body"])
    assert len(ccs) == 2, ccs
    for cc in ccs:
        assert cc == {"type": "ephemeral"}, cc


def test_opt_out_skips_cache_control(monkeypatch):
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content.decode("utf-8"))
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
            messages = [
                {"role": "system", "content": "Be brief."},
                {"role": "user", "content": "hi"},
            ],
            model = "claude-opus-4-7",
            temperature = 0.7,
            top_p = 0.95,
            max_tokens = 32,
            enable_prompt_caching = False,
            prompt_cache_ttl = "1h",
        ):
            pass
        await client.close()

    _drive(run())
    assert _cache_controls(captured["body"]) == []


def _oai_compat_body(
    monkeypatch,
    model,
    provider_type = "openrouter",
    **kwargs,
) -> dict:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            content = b"data: [DONE]\n\n",
            headers = {"content-type": "text/event-stream"},
        )

    monkeypatch.setattr(
        ep_mod,
        "_http_client",
        httpx.AsyncClient(transport = httpx.MockTransport(handler)),
    )

    async def run():
        client = ExternalProviderClient(
            provider_type = provider_type,
            base_url = "https://example.test/v1",
            api_key = "sk-test",
        )
        async for _ in client.stream_chat_completion(
            messages = [{"role": "user", "content": "hi"}], model = model, **kwargs
        ):
            pass
        await client.close()

    _drive(run())
    return captured["body"]


@pytest.mark.parametrize(
    "kwargs,expected",
    [
        ({}, {"type": "ephemeral"}),
        (
            {"enable_prompt_caching": True, "prompt_cache_ttl": "1h"},
            {"type": "ephemeral", "ttl": "1h"},
        ),
        ({"prompt_cache_ttl": "6m"}, {"type": "ephemeral"}),
    ],
)
@pytest.mark.parametrize("model", ["anthropic/claude-sonnet-4.6", "~anthropic/claude-opus-latest"])
def test_openrouter_claude_gets_top_level_cache_control(monkeypatch, model, kwargs, expected):
    assert _oai_compat_body(monkeypatch, model, **kwargs)["cache_control"] == expected


@pytest.mark.parametrize(
    "model,kwargs",
    [
        ("anthropic/claude-sonnet-4.6", {"enable_prompt_caching": False, "prompt_cache_ttl": "1h"}),
        ("deepseek/deepseek-v3.2", {"enable_prompt_caching": True}),
        ("openrouter/auto", {}),
    ],
)
def test_openrouter_cache_control_skipped_off_claude_or_when_disabled(monkeypatch, model, kwargs):
    assert "cache_control" not in _oai_compat_body(monkeypatch, model, **kwargs)


@pytest.mark.parametrize("caching", [True, False])
def test_openrouter_session_id_follows_the_thread(monkeypatch, caching):
    body = _oai_compat_body(
        monkeypatch, "deepseek/deepseek-v3.2", thread_id = "t" * 300, enable_prompt_caching = caching
    )
    assert body["session_id"] == "t" * 256
    assert "session_id" not in _oai_compat_body(monkeypatch, "deepseek/deepseek-v3.2")
    # Strict OpenAI-compatible endpoints 400 on unknown body fields.
    assert "session_id" not in _oai_compat_body(
        monkeypatch, "deepseek-chat", "deepseek", thread_id = "t"
    )
