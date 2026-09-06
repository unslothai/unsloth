# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A managed account's API keys are addressed by its immutable id.

A request that authenticated before its account was deleted and the name
created again still carries the old account. Listing or revoking under that
request must not reach the namesake's keys. The owner's keys keep the plain
username query.
"""

import secrets
import sqlite3

import pytest

from auth import policy, storage


@pytest.fixture
def auth_db(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "DB_PATH", tmp_path / "auth.db")
    monkeypatch.setattr(storage, "_BOOTSTRAP_PW_PATH", tmp_path / ".bootstrap_password")
    monkeypatch.setattr(storage, "_bootstrap_password", None)
    policy.invalidate_account_cache()
    storage.create_initial_user("unsloth", "owner-password", secrets.token_urlsafe(32))
    yield storage
    policy.invalidate_account_cache()


def _alice() -> str:
    return storage.issue_account_setup_code(username = "alice")["account"]["account_id"]


def test_namesake_keys_stay_apart(auth_db):
    old_id = _alice()
    _raw, old_row = storage.create_api_key("alice", name = "old", account_id = old_id)
    assert old_row["account_id"] == old_id
    storage.delete_account(old_id, lambda account: None)
    new_id = _alice()
    assert new_id != old_id
    _raw, new_row = storage.create_api_key("alice", name = "NEW-PRIVATE", account_id = new_id)

    # The old request sees nothing of the namesake and cannot revoke its key.
    assert storage.list_api_keys("alice", account_id = old_id) == []
    assert not storage.revoke_api_key("alice", new_row["id"], account_id = old_id)
    assert [row["name"] for row in storage.list_api_keys("alice", account_id = new_id)] == [
        "NEW-PRIVATE"
    ]
    assert storage.revoke_api_key("alice", new_row["id"], account_id = new_id)


def test_owner_keys_keep_the_username_query(auth_db):
    _raw, row = storage.create_api_key("unsloth", name = "cli")
    assert row["account_id"] is None
    assert [r["name"] for r in storage.list_api_keys("unsloth")] == ["cli"]
    assert storage.revoke_api_key("unsloth", row["id"])


def test_existing_managed_rows_are_pinned_on_upgrade(auth_db):
    alice = _alice()
    conn = sqlite3.connect(storage.DB_PATH)
    try:
        conn.execute(
            "INSERT INTO api_keys (username, key_prefix, key_hash, name, created_at) VALUES ('alice', 'p', 'h1', 'k', 'now')"
        )
        conn.execute(
            "INSERT INTO api_keys (username, key_prefix, key_hash, name, created_at) VALUES ('unsloth', 'p', 'h2', 'k', 'now')"
        )
        conn.commit()
        storage._ensure_account_api_keys(conn, set())
        rows = dict(conn.execute("SELECT key_hash, account_id FROM api_keys").fetchall())
    finally:
        conn.close()
    assert rows["h1"] == alice and rows["h2"] is None


def test_managed_keys_survive_an_older_builds_bulk_delete(auth_db):
    """A build without account support empties the key table on an owner reset."""
    alice = _alice()
    raw, row = storage.create_api_key("alice", name = "cli", account_id = alice)
    owner_raw, _owner_row = storage.create_api_key("unsloth", name = "owner-cli")
    assert storage.validate_api_key_account(raw) is not None
    conn = sqlite3.connect(storage.DB_PATH)
    try:
        conn.execute("DELETE FROM api_keys")
        conn.commit()
        assert conn.execute("SELECT COUNT(*) FROM account_api_keys").fetchone()[0] == 1
        conn.row_factory = sqlite3.Row
        storage._account_keys_synced.clear()
        storage._ensure_account_api_keys(conn, {"account_id"})
    finally:
        conn.close()
    verified = storage.validate_api_key_account(raw)
    assert verified is not None and verified[0]["account_id"] == alice
    assert storage.validate_api_key_account(owner_raw) is None
    assert [r["name"] for r in storage.list_api_keys("alice", account_id = alice)] == ["cli"]
    # A revoked key stays revoked through the same round trip.
    restored_id = storage.list_api_keys("alice", account_id = alice)[0]["id"]
    assert storage.revoke_api_key("alice", restored_id, account_id = alice)
    conn = sqlite3.connect(storage.DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("DELETE FROM api_keys")
        conn.commit()
        storage._account_keys_synced.clear()
        storage._ensure_account_api_keys(conn, {"account_id"})
    finally:
        conn.close()
    assert storage.validate_api_key_account(raw) is None


def test_deleting_an_account_removes_its_key_copy(auth_db):
    alice = _alice()
    storage.create_api_key("alice", name = "cli", account_id = alice)
    storage.delete_account(alice, lambda account: None)
    conn = sqlite3.connect(storage.DB_PATH)
    try:
        assert conn.execute("SELECT COUNT(*) FROM account_api_keys").fetchone()[0] == 0
    finally:
        conn.close()


def test_a_managed_key_minted_without_an_explicit_scope_is_still_pinned(auth_db):
    alice = _alice()
    _raw, row = storage.create_api_key("alice", name = "data-recipe workflow", internal = True)
    assert row["account_id"] == alice
