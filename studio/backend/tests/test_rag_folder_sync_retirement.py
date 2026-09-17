# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Deleting an account must not kill the one linked-folder worker that serves every account."""

from __future__ import annotations

import importlib
import threading
import time
import weakref

import pytest

from auth import policy
from core.rag import folder_sync
from core.training import account_jobs as jobs
from storage import rag_db
from utils.account_context import AccountContext, run_as

ALICE = AccountContext("a" * 32, "alice")
BOB = AccountContext("b" * 32, "bob")

requires_sqlite_vec = pytest.mark.skipif(
    not rag_db.RAG_AVAILABLE, reason = "sqlite-vec is not installed"
)


@pytest.fixture
def two_accounts(tmp_path, monkeypatch, stub_embeddings):
    from utils.paths import storage_roots as roots

    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("UNSLOTH_STUDIO_PROJECTS_HOME", str(tmp_path / "projects"))
    monkeypatch.setenv("UNSLOTH_STUDIO_DOCUMENTS_HOME", str(tmp_path / "docs"))
    monkeypatch.setattr(rag_db, "_schema_ready", set())
    monkeypatch.setattr(policy, "installation_is_multi_user", lambda: True)
    monkeypatch.setattr(policy, "installation_has_managed_accounts", lambda: True)
    monkeypatch.setattr(jobs, "_services", weakref.WeakSet())
    monkeypatch.setattr(jobs, "_retired", set())
    for module, name in (
        ("hub.services.datasets.downloads", "retire_account_downloads"),
        ("hub.services.models.downloads", "retire_account_downloads"),
        ("core.rag.ingestion", "retire_account_ingestions"),
        ("routes.inference", "retire_stt_downloads"),
    ):
        monkeypatch.setattr(importlib.import_module(module), name, lambda: None)
    monkeypatch.setattr("core.research_runs.retire_account_research", lambda account: None)
    monkeypatch.setattr(
        folder_sync,
        "job_accounts",
        lambda: [a for a in (ALICE, BOB) if a.account_id not in jobs._retired],
    )
    homes = {}
    for account in (ALICE, BOB):
        workspace = run_as(account, roots.workspace_root)
        workspace.mkdir(parents = True, exist_ok = True)
        homes[account.account_id] = workspace
    return homes


def _link(account, workspace, name: str):
    source = workspace / f"linked-{name}"
    source.mkdir(parents = True, exist_ok = True)
    (source / "notes.txt").write_text(f"{name} text", encoding = "utf-8")

    def create():
        folder = folder_sync.create_folder(
            scope_type = "knowledge_base", scope_id = f"kb-{name}", path = str(source), name = name
        )
        return folder, folder_sync.request_sync(folder["id"])

    return run_as(account, create)


@requires_sqlite_vec
def test_retiring_an_account_mid_sync_keeps_the_shared_folder_worker_alive(
    two_accounts, monkeypatch
):
    from routes.accounts import retire_account_roots

    blocked, release = threading.Event(), threading.Event()
    real_snapshot = folder_sync._snapshot

    def slow_snapshot(root, metadata):
        blocked.set()
        assert release.wait(60)
        return real_snapshot(root, metadata)

    monkeypatch.setattr(folder_sync, "_snapshot", slow_snapshot)
    _link(ALICE, two_accounts[ALICE.account_id], "alice")

    stop = threading.Event()
    worker = threading.Thread(target = folder_sync._worker, args = (stop,), daemon = True)
    worker.start()
    try:
        assert blocked.wait(30), "the worker never reached the snapshot"
        retire_account_roots(ALICE)
        release.set()

        _, bob_job = _link(BOB, two_accounts[BOB.account_id], "bob")
        folder_sync._wake.set()
        deadline = time.time() + 30
        status = None
        while time.time() < deadline:
            status = run_as(BOB, folder_sync.get_job, bob_job)["status"]
            if status in ("completed", "failed"):
                break
            time.sleep(0.2)
        assert worker.is_alive(), "retiring one account killed the shared linked-folder worker"
        assert not two_accounts[
            ALICE.account_id
        ].exists(), "a late linked-folder snapshot recreated the deleted workspace"
        assert status == "completed", f"the surviving account never synced: {status}"
    finally:
        stop.set()
        release.set()
        folder_sync._wake.set()
        worker.join(30)


def _corrupt_rag_db(account) -> None:
    """Replace the account's rag.db with bytes SQLite refuses to open."""
    from utils.paths import storage_roots as roots

    path = run_as(account, roots.rag_db_path)
    path.parent.mkdir(parents = True, exist_ok = True)
    path.write_bytes(b"not a sqlite database" * 64)


@requires_sqlite_vec
def test_a_corrupt_account_database_does_not_hide_the_next_accounts_jobs(two_accounts):
    _link(BOB, two_accounts[BOB.account_id], "bob")
    _corrupt_rag_db(ALICE)

    selected = folder_sync._next_account_job()

    assert selected is not None, "a corrupt account database hid every job behind it"
    assert selected[0].account_id == BOB.account_id


@requires_sqlite_vec
def test_the_shared_worker_still_syncs_accounts_behind_a_corrupt_database(two_accounts):
    _, bob_job = _link(BOB, two_accounts[BOB.account_id], "bob")
    _corrupt_rag_db(ALICE)

    stop = threading.Event()
    worker = threading.Thread(target = folder_sync._worker, args = (stop,), daemon = True)
    worker.start()
    try:
        deadline = time.time() + 60
        status = None
        while time.time() < deadline:
            status = run_as(BOB, folder_sync.get_job, bob_job)["status"]
            if status in ("completed", "failed"):
                break
            folder_sync._wake.set()
            time.sleep(0.2)
        assert worker.is_alive(), "a corrupt account database killed the shared worker"
        assert status == "completed", f"the healthy account never synced: {status}"
    finally:
        stop.set()
        folder_sync._wake.set()
        worker.join(30)
