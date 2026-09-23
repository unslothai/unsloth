# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""No policy read may answer from the pre-write state once an account mutation has committed.

The observation point is the account write's own ``conn.close()``: the INSERT/UPDATE is durable by
then, and it sits inside the window that a concurrent request would land in.
"""

import importlib.util
import secrets
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from auth import policy, storage


@pytest.fixture
def auth_db(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "DB_PATH", tmp_path / "auth.db")
    monkeypatch.setattr(storage, "_BOOTSTRAP_PW_PATH", tmp_path / ".bootstrap_password")
    monkeypatch.setattr(storage, "_bootstrap_password", None)
    policy.invalidate_account_cache()
    storage.create_initial_user(
        storage.DEFAULT_ADMIN_USERNAME, "owner-password", secrets.token_urlsafe(32)
    )
    yield storage
    policy.invalidate_account_cache()


def _auth_client():
    route_path = Path(storage.__file__).resolve().parents[1] / "routes" / "auth.py"
    spec = importlib.util.spec_from_file_location("_policy_window_auth_route", route_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    app = FastAPI()
    app.include_router(module.router, prefix = "/api/auth")
    return TestClient(app)


class _ObservingConnection:
    """Delegates to a real connection and runs ``observe`` once, as the write closes it."""

    def __init__(self, conn, seen, observe):
        self._conn, self._seen, self._observe = conn, seen, observe

    def __getattr__(self, name):
        return getattr(self._conn, name)

    def __enter__(self):
        self._conn.__enter__()
        return self

    def __exit__(self, *exc):
        return self._conn.__exit__(*exc)

    def close(self):
        if self._observe is not None:
            observe, self._observe = self._observe, None
            self._seen["counts"] = storage.account_counts()
            self._seen["observed"] = observe()
        self._conn.close()


def _observe_when_the_write_commits(monkeypatch, observe):
    seen = {}
    real = storage.get_connection
    armed = [True]

    def get_connection():
        conn = real()
        if not armed[0]:
            return conn
        armed[0] = False
        return _ObservingConnection(conn, seen, observe)

    monkeypatch.setattr(storage, "get_connection", get_connection)
    return seen


def test_desktop_login_while_the_first_account_is_created_is_not_granted(auth_db, monkeypatch):
    raw = storage.create_desktop_secret()
    client = _auth_client()

    # A single-user install answers desktop-login from the cached owner-only count.
    granted = client.post("/api/auth/desktop-login", json = {"secret": raw})
    assert granted.status_code == 200 and granted.json().get("access_token")

    seen = _observe_when_the_write_commits(
        monkeypatch,
        lambda: client.post("/api/auth/desktop-login", json = {"secret": raw}).json(),
    )
    storage.issue_account_setup_code(username = "alice")

    assert seen["counts"] == (2, 1), "the managed account must already be durable"
    assert seen["observed"] == {"login_required": True, "login_mode": "multi"}


def test_the_policy_verdict_follows_the_committed_row_while_creating(auth_db, monkeypatch):
    assert policy.installation_is_multi_user() is False

    seen = _observe_when_the_write_commits(monkeypatch, policy.installation_is_multi_user)
    storage.issue_account_setup_code(username = "alice")

    assert seen["counts"] == (2, 1)
    assert seen["observed"] is True
    assert policy.installation_is_multi_user() is True


def test_the_policy_verdict_follows_the_committed_row_while_reactivating(auth_db, monkeypatch):
    account = storage.issue_account_setup_code(username = "alice")["account"]
    storage.set_account_active(account["account_id"], False)
    assert policy.installation_is_multi_user() is False

    seen = _observe_when_the_write_commits(monkeypatch, policy.installation_is_multi_user)
    storage.set_account_active(account["account_id"], True)

    assert seen["counts"] == (2, 1)
    assert seen["observed"] is True
    assert policy.installation_is_multi_user() is True
