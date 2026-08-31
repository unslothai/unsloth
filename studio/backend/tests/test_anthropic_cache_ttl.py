# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unit tests for prompt_cache_ttl threading on the Anthropic path.

Anthropic's ``cache_control`` marker takes an optional ``ttl``: default 5m
pool, ``ttl:"1h"`` the 1h pool. These tests pin the outbound body shape:
"1h" puts ``ttl:"1h"`` on both markers; default omits the field; garbage
values are silently dropped.
"""

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


# ── default (omitted) writes into the 5m pool ──────────────────────


def test_omitted_ttl_uses_default_5m_pool(monkeypatch):
    captured = _capture(monkeypatch, ttl = None)
    ccs = _cache_controls(captured["body"])
    assert len(ccs) == 2, ccs
    for cc in ccs:
        assert cc == {"type": "ephemeral"}, cc


# ── explicit 5m round-trips as-is ─────────────────────────────────


def test_explicit_5m_ttl_round_trips(monkeypatch):
    captured = _capture(monkeypatch, ttl = "5m")
    ccs = _cache_controls(captured["body"])
    assert len(ccs) == 2, ccs
    for cc in ccs:
        assert cc == {"type": "ephemeral", "ttl": "5m"}, cc


# ── 1h writes the new pool field on every marker ───────────────────


def test_1h_ttl_writes_into_1h_pool(monkeypatch):
    captured = _capture(monkeypatch, ttl = "1h")
    ccs = _cache_controls(captured["body"])
    assert len(ccs) == 2, ccs
    for cc in ccs:
        assert cc == {"type": "ephemeral", "ttl": "1h"}, cc


def test_1h_ttl_does_not_send_extended_cache_ttl_beta_header(monkeypatch):
    # The extended-cache-ttl-2025-04-11 beta header is now GA (verified live
    # 2026-05-22); 1h TTL works with no beta header. Pin so a regression that
    # re-adds the header surfaces here.
    captured = _capture(monkeypatch, ttl = "1h")
    beta = captured["headers"].get("anthropic-beta", "")
    assert "extended-cache-ttl-2025-04-11" not in beta, beta


def test_5m_ttl_does_not_send_extended_cache_ttl_beta_header(monkeypatch):
    captured = _capture(monkeypatch, ttl = "5m")
    beta = captured["headers"].get("anthropic-beta", "")
    assert "extended-cache-ttl-2025-04-11" not in beta, beta


# ── unknown values are dropped, not forwarded ──────────────────────


@pytest.mark.parametrize("bogus", ["6m", "2h", "", "forever", "1d", "0", "1"])
def test_unknown_ttl_silently_dropped(monkeypatch, bogus):
    captured = _capture(monkeypatch, ttl = bogus)
    ccs = _cache_controls(captured["body"])
    assert len(ccs) == 2, ccs
    for cc in ccs:
        # Bogus TTLs must not round-trip; marker stays at default (no ttl = 5m).
        assert cc == {"type": "ephemeral"}, cc


# ── opt-out still skips cache_control entirely ─────────────────────


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
            prompt_cache_ttl = "1h",  # ignored when caching is off
        ):
            pass
        await client.close()

    _drive(run())
    assert _cache_controls(captured["body"]) == []
