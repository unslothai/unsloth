# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing

import pytest

from auth import policy


def status(client, *, count: int) -> None:
    response = client.get("/api/auth/status")
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {
        "initialized", "default_username", "requires_password_change",
        "bootstrap_deadline_seconds", "login_mode",
    }
    assert body["initialized"] == (count > 0)
    assert body["default_username"] == "unsloth"
    assert body["login_mode"] == ("multi" if count > 1 else "single")
    assert "alice" not in response.text and "bob" not in response.text
    assert policy.full_access_permitted() == (count <= 1)


def test_zero_one_two_back_to_one_and_zero(isolated_auth, account_client):
    status(account_client, count = 0)
    isolated_auth.create_initial_user("unsloth", "owner-password", "owner-secret" * 4)
    status(account_client, count = 1)
    isolated_auth.create_initial_user("alice", "alice-password", "alice-secret" * 4)
    status(account_client, count = 2)
    isolated_auth.delete_user("alice")
    status(account_client, count = 1)
    isolated_auth.delete_user("unsloth")
    status(account_client, count = 0)


def test_activation_changes_mode_when_mutator_invalidates_cache(isolated_auth, account_client):
    for username in ("unsloth", "alice"):
        isolated_auth.create_initial_user(username, "account-password", username * 12)
    status(account_client, count = 2)
    for active, count in ((0, 1), (1, 2)):
        with closing(sqlite3.connect(isolated_auth.DB_PATH)) as conn:
            conn.execute("UPDATE auth_user SET is_active=? WHERE username='alice'", (active,))
            conn.commit()
        policy.invalidate_account_cache()
        status(account_client, count = count)


def test_concurrent_creation_refreshes_a_warm_single_mode(isolated_auth, account_client):
    isolated_auth.create_initial_user("unsloth", "owner-password", "owner-secret" * 4)
    status(account_client, count = 1)
    barrier = threading.Barrier(3)

    def create(username):
        barrier.wait(timeout = 10)
        isolated_auth.create_initial_user(username, "account-password", username * 12)

    with ThreadPoolExecutor(max_workers = 2) as pool:
        futures = [pool.submit(create, username) for username in ("alice", "bob")]
        barrier.wait(timeout = 10)
        for future in futures:
            future.result(timeout = 30)
    assert isolated_auth.count_active_accounts() == 3
    assert len({isolated_auth.get_account(name).account_id for name in ("unsloth", "alice", "bob")}) == 3
    status(account_client, count = 3)


def test_duplicate_concurrent_creation_has_one_winner(isolated_auth):
    isolated_auth.create_initial_user("unsloth", "owner-password", "owner-secret" * 4)
    assert policy.login_mode() == "single"
    barrier = threading.Barrier(2)

    def create():
        barrier.wait(timeout = 10)
        try:
            isolated_auth.create_initial_user("alice", "alice-password", "alice-secret" * 4)
            return "created"
        except sqlite3.IntegrityError:
            return "duplicate"

    with ThreadPoolExecutor(max_workers = 2) as pool:
        results = list(pool.map(lambda _: create(), range(2)))
    assert sorted(results) == ["created", "duplicate"]
    assert isolated_auth.count_active_accounts() == 2
    assert policy.login_mode() == "multi"


def test_invalidation_racing_a_stale_count_does_not_poison_cache(isolated_auth, monkeypatch):
    isolated_auth.create_initial_user("unsloth", "owner-password", "owner-secret" * 4)
    entered, release = threading.Event(), threading.Event()
    real_count = isolated_auth.count_active_accounts

    def delayed_count():
        count = real_count()
        entered.set()
        assert release.wait(10)
        return count

    monkeypatch.setattr(isolated_auth, "count_active_accounts", delayed_count)
    with ThreadPoolExecutor(max_workers = 1) as pool:
        pending = pool.submit(policy.login_mode)
        try:
            assert entered.wait(10)
            isolated_auth.create_initial_user("alice", "alice-password", "alice-secret" * 4)
        finally:
            release.set()
        assert pending.result(timeout = 10) == "single"
    monkeypatch.setattr(isolated_auth, "count_active_accounts", real_count)
    assert policy.login_mode() == "multi"


@pytest.mark.xfail(strict = True, reason = "worker 01")
def test_deactivated_account_cannot_keep_using_a_previously_issued_api_key(isolated_auth, accounts, account_client):
    raw_key, _ = isolated_auth.create_api_key("alice", name = "before-deactivation")
    headers = {"Authorization": f"Bearer {raw_key}"}
    assert account_client.get("/account-probe", headers = headers).status_code == 200
    with closing(sqlite3.connect(isolated_auth.DB_PATH)) as conn:
        conn.execute("UPDATE auth_user SET is_active=0 WHERE username='alice'")
        conn.commit()
    policy.invalidate_account_cache()
    response = account_client.get("/account-probe", headers = headers)
    assert response.status_code == 401, response.text
