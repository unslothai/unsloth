# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A research run's workflow key must be pinned to the account that claimed it, not to its username."""

from __future__ import annotations

import asyncio
import secrets
import sqlite3
from types import SimpleNamespace

import pytest

from auth import policy
from auth import storage as auth_storage
from core.research_runs import ResearchSupervisor
from utils.account_context import AccountContext, arun_as


@pytest.fixture
def auth_db(tmp_path, monkeypatch):
    monkeypatch.setattr(auth_storage, "DB_PATH", tmp_path / "auth.db")
    monkeypatch.setattr(auth_storage, "_BOOTSTRAP_PW_PATH", tmp_path / ".bootstrap_password")
    monkeypatch.setattr(auth_storage, "_bootstrap_password", None)
    monkeypatch.setattr(auth_storage, "_api_key_pbkdf2_salt_cache", None)
    auth_storage._reset_api_key_hash_cache()
    policy.invalidate_account_cache()
    auth_storage.create_initial_user("unsloth", "owner-password", secrets.token_urlsafe(32))
    yield auth_storage
    auth_storage._reset_api_key_hash_cache()
    policy.invalidate_account_cache()


def _alice() -> str:
    return auth_storage.issue_account_setup_code(username = "alice")["account"]["account_id"]


def _workflow_key_rows():
    conn = sqlite3.connect(auth_storage.DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        return [
            dict(row)
            for row in conn.execute(
                "SELECT username, account_id FROM api_keys WHERE name = ?",
                (auth_storage.DEEP_RESEARCH_WORKFLOW_KEY_NAME,),
            ).fetchall()
        ]
    finally:
        conn.close()


def test_workflow_key_is_pinned_to_the_claimed_account(auth_db, monkeypatch):
    old_id = _alice()
    old_account = AccountContext(old_id, "alice")
    auth_storage.delete_account(old_id, lambda account: None)
    new_id = _alice()
    assert new_id != old_id

    supervisor = ResearchSupervisor(SimpleNamespace(state = SimpleNamespace(server_port = 1)))

    async def stop_after_mint(*args, **kwargs):
        raise RuntimeError("stop after the key is minted")

    monkeypatch.setattr(supervisor, "_note_phase", stop_after_mint)
    run = {
        "id": "run-1",
        "ownerSubject": "alice",
        "config": {"model": "local", "budgets": {"modelTimeoutSeconds": 5}},
    }

    async def drive():
        with pytest.raises(RuntimeError):
            await arun_as(old_account, supervisor._stream_completion(run, [], phase = "plan"))

    asyncio.run(drive())

    rows = _workflow_key_rows()
    assert rows, "the workflow key was never minted"
    assert [row["account_id"] for row in rows] == [
        old_id
    ], f"workflow key bound to {rows} instead of the claiming account {old_id}"
