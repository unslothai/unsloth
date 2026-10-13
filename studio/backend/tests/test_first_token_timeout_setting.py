# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
import math
import threading
import time
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from core.inference.llama_cpp import LlamaCppBackend
from routes import inference, settings
from storage import studio_db
from utils import first_token_timeout as limit
from utils.account_context import OWNER, AccountContext, bind_account, reset_account


@pytest.fixture(autouse = True)
def isolated(monkeypatch, tmp_path):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(studio_db, "_schema_ready", set())
    monkeypatch.delenv(inference._OPENAI_COMPAT_FIRST_TOKEN_TIMEOUT_ENV, raising = False)


def client(account = OWNER):
    app = FastAPI()
    app.include_router(settings.router)

    async def subject():
        token = bind_account(account)
        try:
            yield account.username
        finally:
            reset_account(token)

    app.dependency_overrides[settings.get_current_subject] = subject
    return TestClient(app)


def test_settings_persist_and_both_inference_paths_use_them():
    with client() as api:
        assert api.get("/first-token-timeout").json()["minutes"] == 20
        for minutes, seconds in [(75, 4500.0), (0, math.inf), (20, 1200.0)]:
            assert api.put("/first-token-timeout", json = {"minutes": minutes}).status_code == 200
            assert limit.timeout_seconds() == seconds
            assert inference._first_token_timeout_s() == seconds
            assert inference._llama_streaming_generation_timeout().read == limit.socket_timeout(
                seconds
            )
        assert api.get("/first-token-timeout").json()["minutes"] == 20


@pytest.mark.parametrize("value", [-1, 10081, True, "60", 1.5, None])
def test_invalid_settings_are_rejected(value):
    with client() as api:
        assert api.put("/first-token-timeout", json = {"minutes": value}).status_code == 422
    assert limit.get_minutes() == 20


def test_managed_account_reads_owner_setting_but_cannot_change_it():
    limit.set_minutes(90)
    with client(AccountContext("a" * 32, "alice")) as api:
        assert api.get("/first-token-timeout").json()["minutes"] == 90
        assert api.put("/first-token-timeout", json = {"minutes": 0}).status_code == 403
    assert limit.get_minutes() == 90


@pytest.mark.parametrize("stored", [-1, 10081, True, "60", 1.5, None, {"minutes": 60}])
def test_unusable_stored_value_falls_back_to_default(stored):
    studio_db.upsert_app_settings({limit.SETTING_KEY: stored})
    assert limit.get_minutes() == limit.DEFAULT_MINUTES


def test_explicit_api_env_override_is_preserved(monkeypatch):
    limit.set_minutes(0)
    monkeypatch.setenv(inference._OPENAI_COMPAT_FIRST_TOKEN_TIMEOUT_ENV, "42")
    assert inference._first_token_timeout_s() == 42
    assert limit.timeout_seconds() == math.inf
    assert inference._OPENAI_COMPAT_FIRST_TOKEN_TIMEOUT_ENV in inference._friendly_error(
        httpx.ReadTimeout("first token")
    )


def test_unlimited_socket_wait_keeps_connect_timeout():
    limit.set_minutes(0)
    timeout = inference._llama_streaming_generation_timeout()
    assert timeout.read is None
    assert timeout.connect == inference._DEFAULT_FIRST_TOKEN_TIMEOUT_S
    assert timeout.write == inference._DEFAULT_FIRST_TOKEN_TIMEOUT_S


@pytest.mark.parametrize("minutes,read", [(1, 1200.0), (90, 5400.0), (0, None)])
def test_shortened_limit_does_not_cut_non_streamed_replies(minutes, read):
    from core.inference.engine_transport import engine_request_timeout

    limit.set_minutes(minutes)
    assert inference._llama_non_streaming_generation_timeout().read == read
    assert engine_request_timeout().read == 1200.0  # Managed engines ignore the GGUF limit.
    assert inference._llama_streaming_generation_timeout().read == (minutes * 60 or None)


def test_sync_custom_wait_survives_original_limit_and_then_expires(monkeypatch):
    limit.set_minutes(60)
    clock = [0.0]
    monkeypatch.setattr(limit.time, "monotonic", lambda: clock[0])

    class Response:
        request = SimpleNamespace(extensions = {"timeout": {"read": 1200}})

        def iter_text(self):
            for moment in (1300, 3601):
                clock[0] = moment
                yield 'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n'

    iterator = LlamaCppBackend._iter_text_cancellable(Response())
    with limit.track_first_token_wait() as wait:
        next(iterator)
        assert wait.active()  # Past the original 20-minute limit.
        next(iterator)
        with pytest.raises(httpx.ReadTimeout):
            next(iterator)
        assert not wait.active()


def test_sync_unlimited_prefill_still_switches_to_decode_stall(monkeypatch):
    limit.set_minutes(0)
    clock = [0.0]
    monkeypatch.setattr(limit.time, "monotonic", lambda: clock[0])

    class Response:
        request = SimpleNamespace(extensions = {"timeout": {"read": 1200}})

        def iter_text(self):
            clock[0] = 86400.0  # A day of simulated prefill.
            yield 'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n'
            clock[0] += 121
            raise httpx.ReadTimeout("silent decode")

    response = Response()
    with limit.track_first_token_wait() as wait:
        iterator = LlamaCppBackend._iter_text_cancellable(response)
        assert "hello" in next(iterator)
        assert not wait.active()
        assert response.request.extensions["timeout"]["read"] == 120
        with pytest.raises(httpx.ReadTimeout, match = "stopped producing tokens"):
            next(iterator)


@pytest.mark.asyncio
async def test_async_pump_arms_saved_limit_until_output_and_stays_cancellable():
    limit.set_minutes(60)
    cancelled = threading.Event()

    async def upstream():
        yield b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n'
        yield b'data: {"prompt_progress":{"processed":8}}\n\n'
        yield b'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n'
        await asyncio.Future()

    with limit.track_first_token_wait() as wait:
        pump = inference._aiter_llama_stream_items(
            upstream(),
            cancel_event = cancelled,
            track_prefill_progress = True,
            keepalive_interval_s = 0.01,
        )
        await anext(pump)
        assert 3500 < wait.deadline - time.monotonic() <= 3600
        limit.set_minutes(90)  # Advancing prefill re-arms with the limit then saved.
        await anext(pump)
        assert 5300 < wait.deadline - time.monotonic() <= 5400
        assert b"hello" in await anext(pump)
        assert not wait.active()
        assert await anext(pump) is inference._LLAMA_STREAM_KEEPALIVE
        cancelled.set()
        with pytest.raises(StopAsyncIteration):
            await asyncio.wait_for(anext(pump), 1)
