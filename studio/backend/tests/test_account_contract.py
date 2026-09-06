# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The account contract every isolation change builds on.

Two properties matter above all others and both are pinned here first: an
install with one account resolves EXACTLY as it did before accounts existed,
and a second account can never reach the first one's storage or generations.
"""

from __future__ import annotations

import asyncio
import secrets
import sqlite3
import threading
from pathlib import Path

import pytest

from utils.account_context import (
    OWNER,
    OWNER_ACCOUNT_ID,
    AccountContext,
    account_thread,
    arun_as,
    bind_account,
    current_account,
    current_account_id,
    is_owner_context,
    reset_account,
    run_as,
)

ALICE = AccountContext("a1b2c3d4e5f6a7b8", "alice", "user")
BOB = AccountContext("0f0e0d0c0b0a0908", "bob", "user")


@pytest.fixture
def studio_home(tmp_path, monkeypatch):
    home = tmp_path / "studio"
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(home))
    monkeypatch.delenv("UNSLOTH_STUDIO_PROJECTS_HOME", raising = False)
    monkeypatch.setenv("UNSLOTH_STUDIO_DOCUMENTS_HOME", str(tmp_path / "Documents"))
    return home


@pytest.fixture
def auth_db(tmp_path, monkeypatch):
    from auth import policy, storage

    monkeypatch.setattr(storage, "DB_PATH", tmp_path / "auth" / "auth.db")
    monkeypatch.setattr(storage, "_BOOTSTRAP_PW_PATH", tmp_path / "auth" / ".bootstrap_password")
    policy.invalidate_account_cache()
    yield storage
    policy.invalidate_account_cache()


# ----------------------------------------------------------------- context


def test_the_default_context_is_the_owner():
    assert current_account() == OWNER
    assert current_account_id() == OWNER_ACCOUNT_ID
    assert is_owner_context() is True


def test_bind_and_reset_restore_the_previous_account():
    token = bind_account(ALICE)
    try:
        assert current_account() == ALICE
        assert is_owner_context() is False
    finally:
        reset_account(token)
    assert current_account() == OWNER


def test_run_as_binds_for_the_call_only():
    seen = run_as(ALICE, current_account_id)
    assert seen == ALICE.account_id
    assert current_account() == OWNER


def test_run_as_refuses_a_coroutine():
    async def later():
        return current_account_id()

    with pytest.raises(TypeError):
        run_as(ALICE, later)


def test_arun_as_keeps_the_binding_across_the_await():
    async def later():
        await asyncio.sleep(0)
        return current_account_id()

    assert asyncio.run(arun_as(ALICE, later())) == ALICE.account_id
    assert current_account() == OWNER


def test_a_plain_thread_does_not_inherit_the_binding():
    seen: list[str] = []
    token = bind_account(ALICE)
    try:
        t = threading.Thread(target = lambda: seen.append(current_account_id()))
        t.start()
        t.join(5)
    finally:
        reset_account(token)
    assert seen == [OWNER_ACCOUNT_ID]


def test_account_thread_captures_the_account_at_creation():
    seen: list[str] = []
    token = bind_account(ALICE)
    try:
        t = account_thread(target = lambda: seen.append(current_account_id()))
    finally:
        reset_account(token)
    t.start()
    t.join(5)
    assert seen == [ALICE.account_id]


# ----------------------------------------------------------------- roots


def test_the_owner_keeps_every_historical_path(studio_home):
    from utils.paths import storage_roots as r

    expected = {
        r.workspace_root: studio_home,
        r.studio_db_path: studio_home / "studio.db",
        r.assets_root: studio_home / "assets",
        r.outputs_root: studio_home / "outputs",
        r.exports_root: studio_home / "exports",
        r.rag_root: studio_home / "rag",
        r.tensorboard_root: studio_home / "runs",
        r.cache_root: studio_home / "cache",
        r.auth_db_path: studio_home / "auth" / "auth.db",
    }
    for fn, want in expected.items():
        assert fn() == want, fn.__name__
    assert "accounts" not in str(r.tmp_root())
    assert r.project_workspaces_root().parts[-2:] == ("Unsloth Studio", "Projects")


def test_a_managed_account_is_rooted_under_its_id(studio_home):
    from utils.paths import storage_roots as r

    base = studio_home / "accounts" / ALICE.account_id
    assert run_as(ALICE, r.workspace_root) == base
    assert run_as(ALICE, r.studio_db_path) == base / "studio.db"
    assert run_as(ALICE, r.assets_root) == base / "assets"
    assert run_as(ALICE, r.rag_root) == base / "rag"
    assert run_as(ALICE, r.tensorboard_root) == base / "runs"
    assert run_as(ALICE, r.tmp_root).parts[-2:] == ("accounts", ALICE.account_id)
    assert run_as(ALICE, r.project_workspaces_root).parts[-3:] == ("Accounts", ALICE.account_id, "Projects")
    # Shared on purpose.
    assert run_as(ALICE, r.cache_root) == studio_home / "cache"
    assert run_as(ALICE, r.auth_db_path) == studio_home / "auth" / "auth.db"


def test_two_accounts_never_share_a_private_root(studio_home):
    from utils.paths import storage_roots as r
    for fn in (r.workspace_root, r.studio_db_path, r.rag_db_path, r.outputs_root, r.tmp_root):
        assert run_as(ALICE, fn) != run_as(BOB, fn), fn.__name__
        assert run_as(ALICE, fn) != fn(), fn.__name__


def test_root_resolution_creates_nothing(studio_home):
    from utils.paths import storage_roots as r
    for fn in (r.workspace_root, r.studio_db_path, r.assets_root, r.tmp_root):
        fn()
        run_as(ALICE, fn)
    assert not studio_home.exists()


# ----------------------------------------------------------------- identity


def test_an_old_auth_db_gains_the_identity_columns_with_the_owner_pinned(auth_db, tmp_path):
    db = tmp_path / "auth" / "auth.db"
    db.parent.mkdir(parents = True)
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE auth_user (id INTEGER PRIMARY KEY, username TEXT UNIQUE NOT NULL, "
        "password_salt TEXT NOT NULL, password_hash TEXT NOT NULL, jwt_secret TEXT NOT NULL)"
    )
    conn.execute(
        "INSERT INTO auth_user (username, password_salt, password_hash, jwt_secret) VALUES ('unsloth','s','h','j')"
    )
    conn.commit()
    conn.close()

    record = auth_db.get_user_record("unsloth")
    assert record["account_id"] == OWNER_ACCOUNT_ID
    assert record["role"] == "owner"
    assert record["is_active"] == 1
    assert auth_db.get_account("unsloth") == OWNER
    assert auth_db.count_active_accounts() == 1


def test_the_seeded_owner_is_the_owner_and_a_second_account_is_not(auth_db):
    auth_db.create_initial_user("unsloth", "owner-password", secrets.token_urlsafe(32))
    auth_db.create_initial_user("alice", "alice-password", secrets.token_urlsafe(32))
    owner = auth_db.get_account("unsloth")
    alice = auth_db.get_account("alice")
    assert owner == OWNER
    assert alice.role == "user"
    assert alice.account_id not in ("", OWNER_ACCOUNT_ID, "alice")
    assert len(alice.account_id) == 32


def test_login_mode_follows_the_account_count(auth_db):
    from auth import policy

    auth_db.create_initial_user("unsloth", "owner-password", secrets.token_urlsafe(32))
    assert policy.login_mode() == "single"
    assert policy.installation_is_multi_user() is False
    assert policy.full_access_permitted() is True
    auth_db.create_initial_user("alice", "alice-password", secrets.token_urlsafe(32))
    assert policy.login_mode() == "multi"
    assert policy.full_access_permitted() is False
    auth_db.delete_user("alice")
    assert policy.login_mode() == "single"


def test_the_policy_cache_costs_nothing_on_the_hot_path(auth_db, monkeypatch):
    from auth import policy

    auth_db.create_initial_user("unsloth", "owner-password", secrets.token_urlsafe(32))
    assert policy.installation_is_multi_user() is False
    calls = []
    monkeypatch.setattr(auth_db, "count_active_accounts", lambda: calls.append(1) or 1)
    for _ in range(50):
        policy.installation_is_multi_user()
    assert calls == []


def test_auth_status_reports_login_mode(auth_db):
    from routes import auth as auth_routes

    auth_db.create_initial_user("unsloth", "owner-password", secrets.token_urlsafe(32))
    assert auth_routes.auth_status().login_mode == "single"
    auth_db.create_initial_user("alice", "alice-password", secrets.token_urlsafe(32))
    status = auth_routes.auth_status()
    assert status.login_mode == "multi"
    assert "alice" not in status.model_dump_json()


def test_a_request_is_bound_to_the_account_its_token_names(auth_db):
    from auth import authentication
    from fastapi.security import HTTPAuthorizationCredentials

    auth_db.create_initial_user("unsloth", "owner-password", secrets.token_urlsafe(32))
    auth_db.create_initial_user("alice", "alice-password", secrets.token_urlsafe(32))
    alice = auth_db.get_account("alice")

    async def resolve(token):
        creds = HTTPAuthorizationCredentials(scheme = "Bearer", credentials = token)
        subject, _gen = await authentication._get_current_credential(creds, allow_password_change = False)
        return subject, current_account()

    subject, bound = asyncio.run(resolve(authentication.create_access_token(subject = "alice")))
    assert subject == "alice"
    assert bound == alice
    subject, bound = asyncio.run(resolve(authentication.create_access_token(subject = "unsloth")))
    assert (subject, bound) == ("unsloth", OWNER)


# ----------------------------------------------------------------- ownership


def test_generations_are_scoped_to_their_account():
    from state import active_generations as ag

    ag.reset_for_tests()
    ev_a, ev_b = threading.Event(), threading.Event()
    with run_as(ALICE, ag.ActiveGeneration, ev_a, thread_id = "t1"), run_as(BOB, ag.ActiveGeneration, ev_b, thread_id = "t1"):
        assert ag.count() == 2
        assert ag.count(ALICE.account_id) == 1
        assert ag.foreign_count(ALICE.account_id) == 1
        # Same client-chosen thread id in two accounts: only Alice's stops.
        assert run_as(ALICE, ag.cancel_thread, "t1") == 1
        assert ev_a.is_set() and not ev_b.is_set()
        assert ag.cancel_all(ALICE.account_id) == 1
        assert not ev_b.is_set()
        assert ag.cancel_all() == 2
        assert ev_b.is_set()
    ag.reset_for_tests()


def test_a_load_never_evicts_another_accounts_active_generation(monkeypatch):
    from core.inference import gpu_arbiter as arb
    from state import active_generations as ag

    ag.reset_for_tests()
    evicted = []
    monkeypatch.setitem(arb._EVICTORS, arb.CHAT, lambda: evicted.append("chat"))
    monkeypatch.setitem(arb._EVICTORS, arb.DIFFUSION, lambda: evicted.append("diffusion"))
    arb.release(arb.CHAT)
    arb.release(arb.DIFFUSION)

    run_as(ALICE, arb.acquire_for, arb.CHAT)
    assert arb.owner_account() == ALICE.account_id
    with run_as(ALICE, ag.ActiveGeneration, threading.Event(), thread_id = "t1"):
        with pytest.raises(arb.GpuBusyForAnotherAccountError):
            run_as(BOB, arb.acquire_for, arb.DIFFUSION)
        assert evicted == []
        # Alice may still evict herself, as before.
        run_as(ALICE, arb.acquire_for, arb.DIFFUSION)
        assert evicted == ["chat"]
    # Idle: Bob swaps as before.
    run_as(BOB, arb.acquire_for, arb.CHAT)
    assert evicted == ["chat", "diffusion"]
    assert arb.owner_account() == BOB.account_id
    arb.release(arb.CHAT)
    ag.reset_for_tests()


def test_the_owner_alone_behaves_exactly_as_before(monkeypatch):
    """Single-user install: every generation and every lease is the owner's, so
    nothing here can ever refuse."""
    from core.inference import gpu_arbiter as arb
    from state import active_generations as ag

    ag.reset_for_tests()
    evicted = []
    monkeypatch.setitem(arb._EVICTORS, arb.CHAT, lambda: evicted.append("chat"))
    monkeypatch.setitem(arb._EVICTORS, arb.DIFFUSION, lambda: evicted.append("diffusion"))
    arb.release(arb.CHAT)
    arb.acquire_for(arb.CHAT)
    with ag.ActiveGeneration(threading.Event(), thread_id = "t1"):
        arb.acquire_for(arb.DIFFUSION)
    assert evicted == ["chat"]
    assert ag.cancel_all() == 0
    arb.release(arb.DIFFUSION)
