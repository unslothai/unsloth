# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Per-request I/O never depends on how many accounts exist, and the owner's routing is fixed."""

import os
import secrets
import sqlite3
from collections import Counter
from unittest.mock import patch

from auth import policy, storage
from core.inference.llama_admission import get_llama_admission_queue
from hub.services.models import account_access as access
from state import active_generations
from utils.account_context import OWNER, AccountContext, run_as

from .support import bearer

PATHS = ("/api/auth/status", "/account-probe", "/api/chat/threads")


def _create(names):
    for name in names:
        storage.create_initial_user(name, "account-password", secrets.token_urlsafe(32))
    policy.invalidate_account_cache()


def _vector(client, headers, path) -> dict:
    """sqlite connections, statements, mkdirs and auth.db count queries for one warm request."""
    counters = Counter()
    connect, mkdir, count = sqlite3.connect, os.mkdir, storage.account_counts

    def open_connection(*args, **kwargs):
        counters["connections"] += 1
        conn = connect(*args, **kwargs)
        conn.set_trace_callback(
            lambda _: counters.__setitem__("statements", counters["statements"] + 1)
        )
        return conn

    def create_directory(*args, **kwargs):
        counters["mkdir"] += 1
        return mkdir(*args, **kwargs)

    def counted(*args, **kwargs):
        counters["account_counts"] += 1
        return count(*args, **kwargs)

    for _ in range(2):
        assert client.get(path, headers = headers).status_code == 200
    with (
        patch.object(sqlite3, "connect", open_connection),
        patch.object(os, "mkdir", create_directory),
        patch.object(storage, "account_counts", counted),
    ):
        assert client.get(path, headers = headers).status_code == 200
    return dict(counters)


def _vectors(client, username):
    headers = bearer(username)
    return {path: _vector(client, headers, path) for path in PATHS}


def test_owner_requests_cost_the_same_with_zero_and_many_accounts(isolated_auth, account_client):
    _create(["unsloth"])
    alone = _vectors(account_client, "unsloth")
    _create([f"user{index:02d}" for index in range(12)])
    crowded = _vectors(account_client, "unsloth")
    assert alone["/account-probe"]["connections"] >= 1, "the counters must observe the auth lookup"
    assert crowded == alone
    assert all(vector.get("account_counts", 0) == 0 for vector in alone.values())


def test_managed_requests_do_not_grow_with_account_count(isolated_auth, account_client):
    _create(["unsloth", "alice"])
    few = _vectors(account_client, "alice")
    _create([f"user{index:02d}" for index in range(12)])
    assert _vectors(account_client, "alice") == few
    assert all(vector.get("account_counts", 0) == 0 for vector in few.values())


def test_owner_routing_facts_and_shared_queue(isolated_auth, account_client):
    _create(["unsloth"])
    headers = bearer("unsloth")
    for path in PATHS:
        assert account_client.get(path, headers = headers).status_code == 200
    assert run_as(OWNER, access.account_scope) is None
    assert run_as(OWNER, access.resident_hidden, "chat") is False
    assert access._resident_sharers == {}
    assert active_generations._FENCED == set()

    _create(["alice"])
    alice = storage.get_account("alice")
    managed = AccountContext(alice.account_id, alice.username, alice.role)
    assert run_as(OWNER, access.account_scope) == "owner"
    assert run_as(managed, access.account_scope) == alice.account_id
    assert run_as(OWNER, access.resident_hidden, "chat") is False
    # Admission queues are keyed by the resident server, never by the calling account.
    key = "http://127.0.0.1:1"
    assert run_as(OWNER, get_llama_admission_queue, key) is run_as(
        managed, get_llama_admission_queue, key
    )
