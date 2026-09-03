# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Regression tests for keeping auth.db reads off the event loop."""

from __future__ import annotations

import asyncio
import threading

import jwt
import pytest
from fastapi import FastAPI
from fastapi.security import HTTPAuthorizationCredentials
from fastapi.testclient import TestClient

import auth.authentication as authentication
import routes.auth as auth_routes
from auth.storage import API_KEY_PREFIX

SECRET = "test-secret-long-enough-for-hs256-hmac-keys"
SUBJECT = "test-user"


def _credentials(token: str) -> HTTPAuthorizationCredentials:
    return HTTPAuthorizationCredentials(scheme = "Bearer", credentials = token)


def _record_thread(threads: list[int], result):
    def _stub(*_args, **_kwargs):
        threads.append(threading.get_ident())
        return result

    return _stub


def _jwt_case(monkeypatch, threads):
    monkeypatch.setattr(
        authentication,
        "get_user_and_secret",
        _record_thread(threads, ("salt", "hash", SECRET, False)),
    )
    token = jwt.encode({"sub": SUBJECT}, SECRET, algorithm = authentication.ALGORITHM)
    return authentication.get_current_subject(_credentials(token))


def _api_key_case(monkeypatch, threads):
    monkeypatch.setattr(
        authentication,
        "validate_api_key_with_credential",
        _record_thread(threads, (SUBJECT, SECRET)),
    )
    return authentication.get_current_subject(_credentials(f"{API_KEY_PREFIX}key"))


def _desktop_case(monkeypatch, threads):
    monkeypatch.setattr(authentication, "is_desktop_access_token", _record_thread(threads, True))
    return authentication.authenticated_via_desktop_jwt(_credentials("token"))


async def _keyless_case(monkeypatch, threads):
    monkeypatch.setattr(
        authentication, "get_user_and_secret", _record_thread(threads, ("s", "h", SECRET, False))
    )
    await authentication.get_current_subject(authentication._KEYLESS_CREDENTIALS)
    from utils import keyless_api_access as keyless

    monkeypatch.setattr(keyless, "keyless_request_allowed", _record_thread(threads, True))
    await authentication.credentials_for_token(object(), None)
    monkeypatch.setattr(keyless, "get_keyless_api_tools_enabled", _record_thread(threads, True))
    await keyless.KeylessToolPolicyMiddleware(lambda *_args: asyncio.sleep(0))(
        {"type": "http"}, None, None
    )


@pytest.mark.parametrize(
    "build_call",
    [_jwt_case, _api_key_case, _desktop_case, _keyless_case],
)
def test_the_dependency_reads_off_the_event_loop_thread(monkeypatch, build_call):
    threads: list[int] = []

    async def _drive():
        await build_call(monkeypatch, threads)
        return threading.get_ident()

    loop_thread = asyncio.run(_drive())

    assert threads, "the credential read never ran"
    assert all(thread != loop_thread for thread in threads), "credential read ran on event loop"


def test_the_status_route_reads_off_the_event_loop_thread(monkeypatch):
    """Verify FastAPI dispatches the sync status handler to its threadpool."""
    threads: list[int] = []
    monkeypatch.setattr(auth_routes.storage, "is_initialized", _record_thread(threads, True))
    monkeypatch.setattr(
        auth_routes.storage, "requires_password_change", _record_thread(threads, False)
    )

    app = FastAPI()
    app.include_router(auth_routes.router, prefix = "/api/auth")
    loop_threads: list[int] = []

    @app.get("/loop-thread")
    async def _loop_thread():
        loop_threads.append(threading.get_ident())
        return {}

    with TestClient(app) as client:
        assert client.get("/loop-thread").status_code == 200
        assert client.get("/api/auth/status").status_code == 200

    assert threads, "the status route never read the auth store"
    assert loop_threads, "the reference route never ran"
    assert threads[0] != loop_threads[0], "auth_status read auth.db on the event loop thread"
