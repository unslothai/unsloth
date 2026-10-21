# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Deleting an account must reap its RAG ingestion workers, not just drop their leases.

A worker parked in a long parse notices retirement only at its next progress checkpoint, so
without a join the delete renames the roots, returns 204, and the thread then lands its
cleanup writes in the renamed-aside database."""

from __future__ import annotations

import importlib
import sqlite3
import threading
import weakref

import pytest

from auth import policy
from core.rag import ingestion, parsers, store
from core.training import account_jobs as jobs
from utils.account_context import AccountContext, run_as

ALICE = AccountContext("a" * 32, "alice")


@pytest.fixture
def alice_home(tmp_path, monkeypatch, stub_embeddings):
    from storage import rag_db
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
        ("core.rag.folder_sync", "retire_account_sync"),
    ):
        monkeypatch.setattr(importlib.import_module(module), name, lambda: None)
    monkeypatch.setattr("core.research_runs.retire_account_research", lambda account: None)
    workspace = run_as(ALICE, roots.workspace_root)
    workspace.mkdir(parents = True, exist_ok = True)
    return workspace


def test_deleting_an_account_reaps_its_blocked_ingestion_worker(alice_home, monkeypatch):
    from routes.accounts import retire_account_roots

    monkeypatch.setattr(ingestion, "_RETIRE_JOIN_SECONDS", 1.0, raising = False)
    path = alice_home / "doc.txt"
    path.write_text("alpha bravo charlie " * 200, encoding = "utf-8")

    blocked, release = threading.Event(), threading.Event()
    real_parse = parsers.parse

    def slow_parse(stored_path, *args, **kwargs):
        blocked.set()
        assert release.wait(60)
        return real_parse(stored_path, *args, **kwargs)

    monkeypatch.setattr(parsers, "parse", slow_parse)

    scope = store.kb_scope("K1")
    _doc_id, job_id = run_as(
        ALICE, ingestion.start_ingestion, scope, "K1", None, "doc.txt", str(path)
    )
    worker = ingestion._workers[(ALICE.account_id, job_id)]
    assert blocked.wait(30), "the worker never reached parse"

    try:
        with pytest.raises(jobs.AccountRetirementError):
            retire_account_roots(ALICE)
        assert alice_home.exists(), "the roots moved out from under a live ingestion worker"
        assert not [p for p in alice_home.parent.iterdir() if "-deleted-" in p.name]
    finally:
        release.set()
    worker.join(60)
    assert not worker.is_alive()

    # With the worker gone, retirement completes and the roots move with nothing left writing.
    retire_account_roots(ALICE)
    tombstones = [p for p in alice_home.parent.iterdir() if "-deleted-" in p.name]
    assert tombstones and not alice_home.exists()
    db = tombstones[0] / "rag" / "rag.db"
    before = db.stat().st_mtime_ns
    conn = sqlite3.connect(str(db))
    try:
        assert conn.execute("SELECT 1 FROM ingestion_jobs WHERE id=?", (job_id,)).fetchone()
    finally:
        conn.close()
    assert db.stat().st_mtime_ns == before
