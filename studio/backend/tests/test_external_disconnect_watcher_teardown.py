# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The ChatGPT subscription and NPU routes must not hang on a disconnect watcher that survives cancel().

Same failure as #7617 (see test_disconnect_watcher_teardown.py): Request.is_disconnected() can
swallow a cancel(), so awaiting the watcher outright holds a finished reply open until the
client itself disconnects.
"""

import asyncio
from types import SimpleNamespace

import routes.inference as inf
from core.inference import openai_codex_auth as codex_auth
from core.inference import openai_codex_client as codex_client
from core.inference.openai_codex_auth import OPENAI_CODEX_API_BASE
from core.inference.providers import get_provider_info
from models.inference import ChatCompletionRequest
from .test_npu_chat_route import _request, flm  # noqa: F401  (flm is a fixture)


def _swallowing_is_disconnected(release):
    """is_disconnected() at its worst under h11: the background watcher is parked in it whenever
    cancel() lands, and absorbs it. Inline polls from the stream itself answer at once."""

    async def is_disconnected():
        if not asyncio.current_task().get_coro().__qualname__.endswith("_watch_disconnect"):
            return False
        try:
            await release.wait()
        except asyncio.CancelledError:
            if not release.is_set():
                return False
            raise
        return False

    return is_disconnected


def _short_bounded_stop(monkeypatch):
    real_stop = inf._stop_local_disconnect_cancel_watcher
    monkeypatch.setattr(
        inf,
        "_stop_local_disconnect_cancel_watcher",
        lambda task, timeout_s = 0.2: real_stop(task, timeout_s),
    )


async def _ends_within(
    coro,
    release,
    timeout_s = 5.0,
):
    task = asyncio.create_task(coro)
    try:
        done, _pending = await asyncio.wait({task}, timeout = timeout_s)
        return bool(done)
    finally:
        release.set()
        await asyncio.wait({task}, timeout = 5.0)
        # Let abandoned watchers see the release and exit before the loop closes.
        await asyncio.sleep(0.05)


def _codex_provider(monkeypatch):
    model = get_provider_info("openai_codex")["default_models"][0]
    monkeypatch.setattr(
        inf.providers_db,
        "get_provider",
        lambda _pid: {
            "id": _pid,
            "provider_type": "openai_codex",
            "base_url": OPENAI_CODEX_API_BASE,
            "display_name": "ChatGPT subscription",
            "is_enabled": True,
            "models": [model],
        },
    )
    monkeypatch.setattr(codex_auth, "load_oauth_bundle", lambda _pid: {"account_id": "acct-1"})
    monkeypatch.setattr(codex_client, "subscription_catalog_matches_account", lambda *_a: True)
    monkeypatch.setattr(codex_client, "subscription_catalog_known", lambda _pid: False)
    monkeypatch.setattr(codex_client, "subscription_catalog_stale", lambda _pid: False)
    monkeypatch.setattr(codex_client, "saved_models_proven_for", lambda *_a: True)

    async def _resolve_access(_provider_id, **_kwargs):
        return "token", "acct-1"

    monkeypatch.setattr(codex_auth, "resolve_access", _resolve_access)

    class FakeCodexClient:
        def __init__(self, *_args, **_kwargs):
            pass

        def stream(self, **_kwargs):
            async def _stream():
                # Upstream takes a moment, so the watcher is polling by the time the reply ends.
                await asyncio.sleep(0.2)
                yield 'data: {"type":"response.output_text.delta","delta":"hi"}'
                yield 'data: {"type":"response.completed"}'

            return _stream()

        async def close(self):
            pass

    monkeypatch.setattr(codex_client, "OpenAICodexClient", FakeCodexClient)
    return model


def test_codex_stream_ends_when_its_disconnect_watcher_swallows_cancel(monkeypatch):
    model = _codex_provider(monkeypatch)
    _short_bounded_stop(monkeypatch)

    async def _run():
        release = asyncio.Event()
        request = SimpleNamespace(
            headers = {},
            state = SimpleNamespace(skip_api_monitor = True),
            is_disconnected = _swallowing_is_disconnected(release),
        )
        payload = ChatCompletionRequest(
            messages = [{"role": "user", "content": "hello"}],
            provider_id = "codex-1",
            external_model = model,
            stream = True,
        )
        chunks = []

        async def _consume():
            response = await inf._proxy_to_external_provider(payload, request, current_subject = "t")
            async for chunk in response.body_iterator:
                chunks.append(chunk)

        ended = await _ends_within(_consume(), release)
        return ended, chunks

    ended, chunks = asyncio.run(_run())
    assert any("[DONE]" in str(chunk) for chunk in chunks), chunks
    assert ended, "the reply was fully sent but the stream never ended"


def test_npu_reply_returns_when_its_disconnect_watcher_swallows_cancel(flm, monkeypatch):  # noqa: F811
    import httpx

    from core.inference import external_provider as ep_mod

    flm()
    _short_bounded_stop(monkeypatch)

    async def _run():
        release = asyncio.Event()
        request = _request()
        request.is_disconnected = _swallowing_is_disconnected(release)
        payload = ChatCompletionRequest(
            messages = [{"role": "user", "content": "count"}], stream = False
        )
        out = {}
        client = httpx.AsyncClient()
        loopback = httpx.AsyncClient(trust_env = False)
        previous = ep_mod._http_client, ep_mod._loopback_http_client
        ep_mod._http_client, ep_mod._loopback_http_client = client, loopback

        async def _reply():
            out["response"] = await inf._npu_chat_completions(payload, request, "tester")

        try:
            ended = await _ends_within(_reply(), release)
        finally:
            ep_mod._http_client, ep_mod._loopback_http_client = previous
            await client.aclose()
            await loopback.aclose()
        return ended, out

    ended, out = asyncio.run(_run())
    assert ended, "the NPU reply was collected but the handler never returned"
    assert out["response"].status_code == 200
