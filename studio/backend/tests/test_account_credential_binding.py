# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""An API key binds the account it was validated for, and a deactivated
account keeps the host closed until it is deleted."""

import asyncio
import secrets

import pytest
from fastapi import HTTPException

from auth import authentication, policy, storage
from utils.account_context import OWNER, current_account, reset_account


@pytest.fixture
def auth_db(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "DB_PATH", tmp_path / "auth.db")
    monkeypatch.setattr(storage, "_BOOTSTRAP_PW_PATH", tmp_path / ".bootstrap_password")
    monkeypatch.setattr(storage, "_bootstrap_password", None)
    policy.invalidate_account_cache()
    storage.create_initial_user("unsloth", "owner-password", secrets.token_urlsafe(32))
    yield storage
    policy.invalidate_account_cache()


def _managed(username: str) -> dict:
    return storage.issue_account_setup_code(username = username)["account"]


def _credentials(token):
    return authentication.HTTPAuthorizationCredentials(scheme = "Bearer", credentials = token)


def _authenticate(token: str):
    """Run the auth dependency and return the account it bound."""
    from utils.account_context import _current  # noqa: PLC0415

    before = _current.get()

    async def go():
        result = await authentication._get_current_credential(
            _credentials(token), allow_password_change = False
        )
        return result, current_account()

    try:
        return asyncio.run(go())
    finally:
        _current.set(before)


def test_api_key_is_bound_from_the_row_it_was_validated_against(auth_db):
    alice = _managed("alice")
    raw = storage.create_api_key(alice["username"], name = "cli")[0]
    verified = storage.validate_api_key_account(raw)
    assert verified is not None
    record, secret = verified
    assert record["account_id"] == alice["account_id"]
    assert record["role"] == "user" and record["username"] == "alice"
    assert storage.validate_api_key_with_credential(raw) == ("alice", secret)


def test_api_key_of_a_replaced_username_never_binds_the_replacement(auth_db, monkeypatch):
    """The auth dependency resolves identity from the validated row alone.

    The old key was revoked with the account, so it no longer validates. To show
    that no second lookup by username exists, the validation is pinned to the
    old account's row while the name already belongs to the new account.
    """
    old = _managed("alice")
    raw = storage.create_api_key("alice", name = "cli")[0]
    old_record, secret = storage.validate_api_key_account(raw)
    storage.delete_account(old["account_id"], lambda account: None)
    new = _managed("alice")
    assert new["account_id"] != old["account_id"]

    with pytest.raises(HTTPException) as exc:
        _authenticate(raw)
    assert exc.value.status_code == 401

    monkeypatch.setattr(
        authentication, "validate_api_key_account", lambda key: (old_record, secret)
    )
    (username, _generation), bound = _authenticate(raw)
    assert username == "alice"
    assert bound.account_id == old["account_id"]
    assert bound.account_id != new["account_id"]


def test_api_key_of_a_deactivated_account_does_not_validate(auth_db):
    alice = _managed("alice")
    raw = storage.create_api_key("alice", name = "cli")[0]
    assert storage.validate_api_key_account(raw) is not None
    storage.set_account_active(alice["account_id"], False)
    assert storage.validate_api_key_account(raw) is None
    with pytest.raises(HTTPException) as exc:
        _authenticate(raw)
    assert exc.value.status_code == 401


def test_owner_api_key_binds_the_owner(auth_db):
    raw = storage.create_api_key("unsloth", name = "cli")[0]
    (username, _generation), bound = _authenticate(raw)
    assert username == "unsloth"
    assert bound == OWNER


def test_deactivated_account_keeps_full_access_closed_until_deleted(auth_db):
    assert policy.full_access_permitted() is True
    alice = _managed("alice")
    assert policy.login_mode() == "multi"
    assert policy.full_access_permitted() is False

    storage.set_account_active(alice["account_id"], False)
    assert policy.login_mode() == "single"
    assert policy.installation_is_multi_user() is False
    assert policy.managed_account_count() == 1
    assert policy.full_access_permitted() is False

    storage.set_account_active(alice["account_id"], True)
    assert policy.full_access_permitted() is False

    storage.delete_account(alice["account_id"], lambda account: None)
    assert policy.login_mode() == "single"
    assert policy.managed_account_count() == 0
    assert policy.full_access_permitted() is True


def test_status_reports_full_access_with_a_deactivated_account(auth_db):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from routes import auth as auth_routes

    app = FastAPI()
    app.include_router(auth_routes.router, prefix = "/api/auth")
    client = TestClient(app)
    assert client.get("/api/auth/status").json()["full_access"] is True
    alice = _managed("alice")
    storage.set_account_active(alice["account_id"], False)
    body = client.get("/api/auth/status").json()
    assert body["login_mode"] == "single"
    assert body["full_access"] is False
    assert "alice" not in client.get("/api/auth/status").text


def test_unreadable_auth_db_keeps_the_host_closed(auth_db, monkeypatch):
    def boom():
        raise OSError("auth.db unreadable")

    monkeypatch.setattr(storage, "account_counts", boom)
    policy.invalidate_account_cache()
    assert policy.login_mode() == "single"
    assert policy.full_access_permitted() is False
