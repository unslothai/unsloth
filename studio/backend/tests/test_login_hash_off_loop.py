# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unauthenticated login guesses must not run PBKDF2 on the event loop, and moving it off the
loop must not let a concurrent burst slip past the per-account limit."""

import asyncio
import time

import pytest
from fastapi import HTTPException
from starlette.requests import Request

import routes.auth as auth_routes
from models.auth import AuthLoginRequest


def _request(ip: str) -> Request:
    return Request({"type": "http", "client": (ip, 40000), "headers": []})


@pytest.fixture
def slow_owner_hash(monkeypatch):
    def patch(seconds: float):
        def slow_verify(*_args, **_kwargs):
            time.sleep(seconds)
            return False

        monkeypatch.setattr(auth_routes.hashing, "verify_password", slow_verify)
        monkeypatch.setattr(auth_routes.policy, "installation_is_multi_user", lambda: False)
        monkeypatch.setattr(
            auth_routes.storage,
            "get_user_and_secret",
            lambda _name: ("salt", "hash", "jwt-secret", False),
        )

    return patch


def _attempt(ip: str):
    payload = AuthLoginRequest(
        username = auth_routes.storage.DEFAULT_ADMIN_USERNAME, password = "wrong-password"
    )
    return auth_routes.login(payload, _request(ip))


def test_a_password_check_does_not_freeze_the_event_loop(slow_owner_hash):
    slow_owner_hash(1.0)

    async def scenario():
        ticks = 0

        async def ticker():
            nonlocal ticks
            while True:
                await asyncio.sleep(0.01)
                ticks += 1

        tick_task = asyncio.create_task(ticker())
        await asyncio.sleep(0)
        with pytest.raises(HTTPException) as raised:
            await _attempt("203.0.113.71")
        tick_task.cancel()
        return ticks, raised.value.status_code

    ticks, status = asyncio.run(scenario())
    assert status == 401
    assert ticks >= 20, ticks


def test_a_concurrent_burst_is_still_held_to_the_account_limit(slow_owner_hash):
    slow_owner_hash(0.05)

    async def scenario():
        results = await asyncio.gather(
            *(_attempt("203.0.113.72") for _ in range(40)), return_exceptions = True
        )
        return sorted(r.status_code for r in results)

    codes = asyncio.run(scenario())
    assert codes.count(401) == auth_routes._LOGIN_MAX_FAILS, codes
    assert codes.count(429) == 40 - auth_routes._LOGIN_MAX_FAILS, codes
