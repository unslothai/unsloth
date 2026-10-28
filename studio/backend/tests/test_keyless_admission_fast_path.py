# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""KeylessToolPolicyMiddleware runs on every request. With the setting off (the default) or a fresh settings
cache it has nothing to read, so it must not borrow an AnyIO worker thread to find that out: a saturated pool
would otherwise stall even the desktop app's liveness poll."""

from __future__ import annotations

import asyncio
import threading
import time
from types import SimpleNamespace

import anyio.to_thread
import pytest

import utils.keyless_api_access as keyless
from auth import storage
from utils.keyless_api_access import (
    KEYLESS_ADMISSION_STATE_KEY,
    KEYLESS_SCOPE_OFF,
    KeylessToolPolicyMiddleware,
    _reset_scope_cache,
    set_keyless_api_access,
)


@pytest.fixture(autouse = True)
def isolated_auth_db(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "DB_PATH", tmp_path / "auth.db")
    monkeypatch.setattr(storage, "_BOOTSTRAP_PW_PATH", tmp_path / ".bootstrap_password")
    monkeypatch.setattr(storage, "_bootstrap_password", None)
    _reset_scope_cache()
    yield
    _reset_scope_cache()


@pytest.fixture
def hops(monkeypatch):
    counted = [0]
    real = anyio.to_thread.run_sync

    async def run_sync(*args, **kwargs):
        counted[0] += 1
        return await real(*args, **kwargs)

    monkeypatch.setattr(anyio.to_thread, "run_sync", run_sync)
    return counted


def _scope(path = "/v1/chat/completions", method = "POST"):
    return {
        "type": "http",
        "method": method,
        "path": path,
        "root_path": "",
        "query_string": b"",
        "headers": [],
        "client": ("127.0.0.1", 50000),
        "server": ("127.0.0.1", 8000),
        "scheme": "http",
        "app": SimpleNamespace(
            state = SimpleNamespace(
                bind_host = "127.0.0.1",
                secure = False,
                remote_access_is_colab = False,
                lan_access_is_colab = False,
                lan_access_secure_launch = False,
                cloudflare_url = None,
            )
        ),
    }


def _run_middleware(scope):
    seen = []

    async def downstream(asgi_scope, _receive, _send):
        seen.append(asgi_scope["state"][KEYLESS_ADMISSION_STATE_KEY])

    async def go():
        await KeylessToolPolicyMiddleware(downstream)(scope, None, None)

    asyncio.run(go())
    return seen


def test_setting_off_admits_nobody_without_a_worker_thread(hops):
    set_keyless_api_access(KEYLESS_SCOPE_OFF)
    hops[0] = 0
    assert _run_middleware(_scope()) == [False]
    assert hops[0] == 0


def test_a_fresh_cache_is_answered_on_the_loop(hops):
    set_keyless_api_access("inference", tools = False)
    hops[0] = 0
    assert asyncio.run(keyless._settings_async())[:2] == ("inference", False)
    assert hops[0] == 0


def test_a_stale_cache_still_reads_the_database(hops):
    set_keyless_api_access("inference", tools = False)
    keyless._cached_settings = (
        time.monotonic() - keyless._SETTINGS_CACHE_TTL_S - 1,
        KEYLESS_SCOPE_OFF,
        False,
    )
    hops[0] = 0
    # The stale cached value says off; the database says inference, so the read happened.
    assert asyncio.run(keyless._settings_async())[:2] == ("inference", False)
    assert hops[0] == 1


def test_a_write_in_flight_reads_as_off():
    set_keyless_api_access("full", tools = True)
    keyless._settings_write_inflight = object()
    try:
        scope, tools, _generation = asyncio.run(keyless._settings_async())
    finally:
        keyless._settings_write_inflight = None
    assert (scope, tools) == (KEYLESS_SCOPE_OFF, False)


def test_an_admitting_scope_still_classifies_the_request(hops):
    """Only the off answer is short-circuited; a scope that can admit still runs the full check."""
    set_keyless_api_access("inference", tools = False)
    hops[0] = 0
    assert _run_middleware(_scope()) == [True]
    assert hops[0] == 1
    assert _run_middleware(_scope("/api/train/start")) == [False]


def test_liveness_answers_while_the_worker_pool_is_exhausted(monkeypatch):
    import httpx
    from starlette.concurrency import run_in_threadpool

    from main import app

    # Keep the cache fresh for the whole test, whatever the host's speed.
    monkeypatch.setattr(keyless, "_SETTINGS_CACHE_TTL_S", 3600.0)
    set_keyless_api_access(KEYLESS_SCOPE_OFF)

    async def go():
        transport = httpx.ASGITransport(app = app, client = ("127.0.0.1", 50000))
        async with httpx.AsyncClient(
            transport = transport, base_url = "http://127.0.0.1:8888"
        ) as client:
            assert (await client.get("/api/liveness")).status_code == 200
            limiter = anyio.to_thread.current_default_thread_limiter()
            original_tokens = limiter.total_tokens
            release = threading.Event()
            limiter.total_tokens = 1
            holder = asyncio.ensure_future(run_in_threadpool(release.wait, 30))
            try:
                await asyncio.sleep(0.05)
                response = await asyncio.wait_for(client.get("/api/liveness"), timeout = 5)
            finally:
                release.set()
                await holder
                limiter.total_tokens = original_tokens
            return response

    response = asyncio.run(go())
    assert response.status_code == 200
    assert response.json()["status"] == "alive"
