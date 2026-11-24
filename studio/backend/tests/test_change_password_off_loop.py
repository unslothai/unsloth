# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Repeated change-password checks by any signed-in account must not run PBKDF2 on the event loop."""

import asyncio
import time

import pytest
from fastapi import HTTPException

import routes.auth as auth_routes
from models.auth import ChangePasswordRequest


def test_a_change_password_check_does_not_freeze_the_event_loop(monkeypatch):
    def slow_verify(*_args, **_kwargs):
        time.sleep(1.0)
        return False

    monkeypatch.setattr(auth_routes.hashing, "verify_password", slow_verify)
    monkeypatch.setattr(
        auth_routes.storage,
        "get_user_and_secret",
        lambda _name: ("salt", "hash", "jwt-secret", False),
    )
    payload = ChangePasswordRequest(
        current_password = "wrong-password", new_password = "another-password"
    )

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
            await auth_routes.change_password(payload, None, "alice", False)
        tick_task.cancel()
        return ticks, raised.value.status_code, raised.value.detail

    ticks, status, detail = asyncio.run(scenario())
    assert (status, detail) == (401, "Current password is incorrect")
    assert ticks >= 20, ticks
