# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import os
import platform
import shutil
import sqlite3
import threading
import uuid
from pathlib import Path

import pytest

from storage import studio_db
from utils.paths import studio_db_path


def _reset_studio_db(
    tmp_path,
    monkeypatch,
    projects_home = None,
):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setenv(
        "UNSLOTH_STUDIO_PROJECTS_HOME",
        str(projects_home if projects_home is not None else tmp_path / "Projects"),
    )
    monkeypatch.setattr(studio_db, "_schema_ready", False)


@pytest.fixture
def workspace_projects_home(tmp_path):
    """Projects root outside the platform delete denylist.

    macOS tmp_path resolves under /private/tmp, which the delete guard refuses;
    only the denied case falls back to a home subdir.
    """
    candidate = tmp_path / "Projects"
    resolved = str(candidate.resolve())
    check = os.path.normcase(resolved) if platform.system() == "Windows" else resolved
    denied = studio_db._denied_path_prefixes()
    if any(check == p or check.startswith(p + os.sep) for p in denied):
        candidate = Path.home() / ".unsloth-studio-tests" / uuid.uuid4().hex
    candidate.mkdir(parents = True, exist_ok = True)
    try:
        yield candidate
    finally:
        if ".unsloth-studio-tests" in candidate.parts:
            shutil.rmtree(candidate, ignore_errors = True)


def _thread(thread_id: str = "thread-1") -> dict:
    return {
        "id": thread_id,
        "title": "Test Chat",
        "modelType": "base",
        "modelId": "test-model",
        "pairId": None,
        "archived": False,
        "createdAt": 1_700_000_000_000,
    }


def _message(
    message_id: str,
    created_at: int,
    content: str,
    thread_id: str = "thread-1",
) -> dict:
    return {
        "id": message_id,
        "threadId": thread_id,
        "parentId": None,
        "role": "user",
        "content": [{"type": "text", "text": content}],
        "createdAt": created_at,
    }


def _project(project_id: str = "project-1") -> dict:
    return {
        "id": project_id,
        "name": "Research",
        "instructions": "Use terse answers.",
        "archived": False,
        "createdAt": 1_700_000_000_000,
        "updatedAt": 1_700_000_000_000,
    }


def test_sync_chat_messages_upserts_without_pruning(tmp_path, monkeypatch):
    _reset_studio_db(tmp_path, monkeypatch)
    studio_db.upsert_chat_thread(_thread())
    studio_db.sync_chat_messages(
        "thread-1",
        [
            _message("msg-1", 1, "keep me"),
            _message("msg-2", 2, "old text"),
        ],
        prune_missing = True,
    )

    messages = studio_db.sync_chat_messages(
        "thread-1",
        [_message("msg-2", 2, "updated text")],
    )

    by_id = {message["id"]: message for message in messages}
    assert set(by_id) == {"msg-1", "msg-2"}
    assert by_id["msg-2"]["content"] == [{"type": "text", "text": "updated text"}]


def test_chat_thread_updated_at_bumps_on_message_writes(tmp_path, monkeypatch):
    _reset_studio_db(tmp_path, monkeypatch)
    thread = studio_db.upsert_chat_thread(_thread())
    assert thread["updatedAt"] == thread["createdAt"]

    studio_db.upsert_chat_message(_message("msg-1", 1_700_000_000_500, "hi"))
    assert studio_db.get_chat_thread("thread-1")["updatedAt"] == 1_700_000_000_500

    studio_db.upsert_chat_message(_message("msg-0", 1_600_000_000_000, "old"))
    assert studio_db.get_chat_thread("thread-1")["updatedAt"] == 1_700_000_000_500

    studio_db.sync_chat_messages(
        "thread-1",
        [_message("msg-2", 1_700_000_001_000, "newer")],
    )
    assert studio_db.get_chat_thread("thread-1")["updatedAt"] == 1_700_000_001_000


def test_chat_thread_updated_at_recomputed_when_pruning(tmp_path, monkeypatch):
    _reset_studio_db(tmp_path, monkeypatch)
    thread = studio_db.upsert_chat_thread(_thread())
    studio_db.sync_chat_messages(
        "thread-1",
        [
            _message("msg-1", 1_700_000_000_500, "older"),
            _message("msg-2", 1_700_000_001_000, "newest"),
        ],
        prune_missing = True,
    )
    assert studio_db.get_chat_thread("thread-1")["updatedAt"] == 1_700_000_001_000

    # Pruning the newest message must lower updated_at to the remaining one.
    studio_db.sync_chat_messages(
        "thread-1",
        [_message("msg-1", 1_700_000_000_500, "older")],
        prune_missing = True,
    )
    assert studio_db.get_chat_thread("thread-1")["updatedAt"] == 1_700_000_000_500

    # Pruning every message falls back to created_at.
    studio_db.sync_chat_messages("thread-1", [], prune_missing = True)
    assert studio_db.get_chat_thread("thread-1")["updatedAt"] == thread["createdAt"]


def test_chat_thread_updated_at_survives_thread_resave(tmp_path, monkeypatch):
    _reset_studio_db(tmp_path, monkeypatch)
    studio_db.upsert_chat_thread(_thread())
    studio_db.upsert_chat_message(_message("msg-1", 1_700_000_000_500, "hi"))

    studio_db.upsert_chat_thread(_thread())
    assert studio_db.get_chat_thread("thread-1")["updatedAt"] == 1_700_000_000_500


def test_list_chat_threads_orders_by_last_activity(tmp_path, monkeypatch):
    _reset_studio_db(tmp_path, monkeypatch)
    older = _thread("thread-old")
    older["createdAt"] = 1_700_000_000_000
    newer = _thread("thread-new")
    newer["createdAt"] = 1_700_000_100_000
    studio_db.upsert_chat_thread(older)
    studio_db.upsert_chat_thread(newer)
    assert [t["id"] for t in studio_db.list_chat_threads()] == ["thread-new", "thread-old"]

    studio_db.upsert_chat_message(
        _message("msg-1", 1_700_000_200_000, "hi", thread_id = "thread-old")
    )
    assert [t["id"] for t in studio_db.list_chat_threads()] == ["thread-old", "thread-new"]


def test_chat_threads_updated_at_migration_backfills_from_messages(tmp_path, monkeypatch):
    _reset_studio_db(tmp_path, monkeypatch)
    db_path = studio_db_path()
    db_path.parent.mkdir(parents = True, exist_ok = True)
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            """
            CREATE TABLE chat_threads (
                id TEXT NOT NULL PRIMARY KEY,
                title TEXT NOT NULL,
                model_type TEXT NOT NULL,
                model_id TEXT,
                pair_id TEXT,
                archived INTEGER NOT NULL DEFAULT 0,
                created_at INTEGER NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE chat_messages (
                id TEXT NOT NULL PRIMARY KEY,
                thread_id TEXT NOT NULL,
                parent_id TEXT,
                role TEXT NOT NULL,
                content_json TEXT NOT NULL,
                attachments_json TEXT,
                metadata_json TEXT,
                created_at INTEGER NOT NULL
            )
            """
        )
        conn.execute(
            "INSERT INTO chat_threads (id, title, model_type, created_at) VALUES (?, ?, ?, ?)",
            ("thread-with-msgs", "Old", "base", 1_700_000_000_000),
        )
        conn.execute(
            "INSERT INTO chat_threads (id, title, model_type, created_at) VALUES (?, ?, ?, ?)",
            ("thread-empty", "Empty", "base", 1_700_000_050_000),
        )
        # Fork-like thread: copied ancestor messages predate the thread itself.
        conn.execute(
            "INSERT INTO chat_threads (id, title, model_type, created_at) VALUES (?, ?, ?, ?)",
            ("thread-fork", "Fork", "base", 1_700_000_100_000),
        )
        conn.executemany(
            "INSERT INTO chat_messages (id, thread_id, role, content_json, created_at) VALUES (?, ?, ?, ?, ?)",
            [
                ("m1", "thread-with-msgs", "user", "[]", 1_700_000_001_000),
                ("m2", "thread-with-msgs", "assistant", "[]", 1_700_000_002_000),
                ("m3", "thread-fork", "user", "[]", 1_700_000_001_000),
            ],
        )
        conn.commit()
    finally:
        conn.close()

    assert studio_db.get_chat_thread("thread-with-msgs")["updatedAt"] == 1_700_000_002_000
    assert studio_db.get_chat_thread("thread-empty")["updatedAt"] == 1_700_000_050_000
    assert studio_db.get_chat_thread("thread-fork")["updatedAt"] == 1_700_000_100_000


def test_chat_projects_delete_cascades_threads_and_messages(tmp_path, monkeypatch):
    _reset_studio_db(tmp_path, monkeypatch)
    project = studio_db.upsert_chat_project(_project())
    assert project["rootPath"].startswith(str(tmp_path / "Projects"))
    assert (tmp_path / "Projects" / "Research-project").exists()
    assert (tmp_path / "Projects" / "Research-project" / "sandbox").is_dir()
    assert not (tmp_path / "Projects" / "Research-project" / "chats").exists()
    assert not (tmp_path / "Projects" / "Research-project" / "files").exists()
    assert not (tmp_path / "Projects" / "Research-project" / "exports").exists()
    studio_db.upsert_chat_thread({**_thread(), "projectId": "project-1"})
    studio_db.upsert_chat_message(_message("msg-1", 1, "delete with project"))

    [thread] = studio_db.list_chat_threads(project_id = "project-1")
    assert thread["projectId"] == "project-1"

    deleted = studio_db.delete_chat_project("project-1")

    assert deleted is not None
    assert deleted["id"] == "project-1"
    assert studio_db.get_chat_project("project-1") is None
    assert studio_db.list_chat_threads(project_id = "project-1") == []
    assert studio_db.get_chat_thread("thread-1") is None
    assert studio_db.list_chat_messages("thread-1") == []
    assert (tmp_path / "Projects" / "Research-project").exists()


def test_chat_project_delete_files_removes_workspace(
    tmp_path, monkeypatch, workspace_projects_home
):
    _reset_studio_db(tmp_path, monkeypatch, projects_home = workspace_projects_home)
    project = studio_db.upsert_chat_project(_project())
    # Derive root from the created project so it tracks the projects home.
    root = Path(project["rootPath"])
    marker = root / "sandbox" / "marker.txt"
    marker.write_text("created by code execution", encoding = "utf-8")

    deleted = studio_db.delete_chat_project(project["id"], delete_files = True)

    assert deleted is not None
    assert deleted["rootPath"] == project["rootPath"]
    assert not root.exists()
    assert studio_db.get_chat_project(project["id"]) is None


def test_sync_chat_messages_prunes_when_requested(tmp_path, monkeypatch):
    _reset_studio_db(tmp_path, monkeypatch)
    studio_db.upsert_chat_thread(_thread())
    studio_db.sync_chat_messages(
        "thread-1",
        [
            _message("msg-1", 1, "delete me"),
            _message("msg-2", 2, "keep me"),
        ],
    )

    messages = studio_db.sync_chat_messages(
        "thread-1",
        [_message("msg-2", 2, "keep me")],
        prune_missing = True,
    )

    assert [message["id"] for message in messages] == ["msg-2"]


def test_upsert_chat_message_rejects_cross_thread_id_conflict(tmp_path, monkeypatch):
    _reset_studio_db(tmp_path, monkeypatch)
    studio_db.upsert_chat_thread(_thread("thread-1"))
    studio_db.upsert_chat_thread(_thread("thread-2"))
    studio_db.upsert_chat_message(_message("msg-1", 1, "original", "thread-1"))

    with pytest.raises(studio_db.ChatMessageConflictError):
        studio_db.upsert_chat_message(_message("msg-1", 2, "moved", "thread-2"))

    assert [m["id"] for m in studio_db.list_chat_messages("thread-1")] == ["msg-1"]
    assert studio_db.list_chat_messages("thread-2") == []


def test_sync_chat_messages_detects_conflict_before_prune(tmp_path, monkeypatch):
    _reset_studio_db(tmp_path, monkeypatch)
    studio_db.upsert_chat_thread(_thread("thread-1"))
    studio_db.upsert_chat_thread(_thread("thread-2"))
    studio_db.sync_chat_messages(
        "thread-1",
        [_message("keep-me", 1, "keep", "thread-1")],
    )
    studio_db.upsert_chat_message(_message("conflict", 2, "other", "thread-2"))

    with pytest.raises(studio_db.ChatMessageConflictError):
        studio_db.sync_chat_messages(
            "thread-1",
            [_message("conflict", 3, "bad", "thread-1")],
            prune_missing = True,
        )

    assert [m["id"] for m in studio_db.list_chat_messages("thread-1")] == ["keep-me"]
    assert [m["id"] for m in studio_db.list_chat_messages("thread-2")] == ["conflict"]


def test_settings_merge_atomic_under_concurrency(tmp_path, monkeypatch):
    """Two threads writing distinct keys must not drop each other's update."""
    _reset_studio_db(tmp_path, monkeypatch)
    studio_db.upsert_chat_settings_merge({"inferenceParams": {}})

    barrier = threading.Barrier(2)

    def writer(key: str, value: float) -> None:
        barrier.wait()
        studio_db.upsert_chat_settings_merge({"inferenceParams": {key: value}})

    t1 = threading.Thread(target = writer, args = ("temperature", 0.7))
    t2 = threading.Thread(target = writer, args = ("topP", 0.9))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    merged = studio_db.list_chat_settings()["inferenceParams"]
    assert merged.get("temperature") == 0.7
    assert merged.get("topP") == 0.9


def test_settings_merge_preserves_nested_keys(tmp_path, monkeypatch):
    _reset_studio_db(tmp_path, monkeypatch)
    studio_db.upsert_chat_settings_merge({"inferenceParams": {"temperature": 0.5, "topP": 0.8}})
    studio_db.upsert_chat_settings_merge({"inferenceParams": {"temperature": 0.9}})

    params = studio_db.list_chat_settings()["inferenceParams"]
    assert params == {"temperature": 0.9, "topP": 0.8}


def test_settings_merge_quarantines_corrupt_json_and_rejects_partial_patch(tmp_path, monkeypatch):
    _reset_studio_db(tmp_path, monkeypatch)
    studio_db.upsert_chat_settings_merge({"inferenceParams": {"temperature": 0.5, "topP": 0.8}})
    conn = studio_db.get_connection()
    try:
        conn.execute(
            "UPDATE chat_settings SET value_json = ? WHERE key = ?",
            ('{"temperature": 0.5', "inferenceParams"),
        )
        conn.commit()
    finally:
        conn.close()

    with pytest.raises(studio_db.CorruptSettingsError):
        studio_db.upsert_chat_settings_merge({"inferenceParams": {"temperature": 0.9}})

    conn = studio_db.get_connection()
    try:
        quarantined = conn.execute(
            "SELECT key, value_json, reason FROM chat_settings_quarantine"
        ).fetchall()
        remaining = conn.execute(
            "SELECT key FROM chat_settings WHERE key = ?",
            ("inferenceParams",),
        ).fetchall()
    finally:
        conn.close()
    assert [row["key"] for row in quarantined] == ["inferenceParams"]
    assert quarantined[0]["reason"] == "json_decode_error"
    assert remaining == []


def test_settings_merge_replaces_corrupt_scalar_after_quarantine(tmp_path, monkeypatch):
    _reset_studio_db(tmp_path, monkeypatch)
    studio_db.upsert_chat_settings_merge({"autoTitle": False})
    conn = studio_db.get_connection()
    try:
        conn.execute(
            "UPDATE chat_settings SET value_json = ? WHERE key = ?",
            ("not-json", "autoTitle"),
        )
        conn.commit()
    finally:
        conn.close()

    settings = studio_db.upsert_chat_settings_merge({"autoTitle": True})

    assert settings["autoTitle"] is True
    conn = studio_db.get_connection()
    try:
        quarantined = conn.execute("SELECT key, reason FROM chat_settings_quarantine").fetchall()
    finally:
        conn.close()
    assert [(row["key"], row["reason"]) for row in quarantined] == [
        ("autoTitle", "json_decode_error")
    ]


def test_list_chat_messages_for_threads_chunks_over_900_ids(tmp_path, monkeypatch):
    """SQLite host-parameter limit is 999 on older builds; chunk at 900."""
    _reset_studio_db(tmp_path, monkeypatch)
    n = 901
    for i in range(n):
        studio_db.upsert_chat_thread(
            {
                "id": f"t-{i}",
                "title": "T",
                "modelType": "base",
                "modelId": "m",
                "pairId": None,
                "archived": False,
                "createdAt": 1_700_000_000_000 + i,
            }
        )
        studio_db.upsert_chat_message(
            {
                "id": f"m-{i}",
                "threadId": f"t-{i}",
                "parentId": None,
                "role": "user",
                "content": [{"type": "text", "text": "hi"}],
                "createdAt": 1_700_000_000_000 + i,
            }
        )
    out = studio_db.list_chat_messages_for_threads([f"t-{i}" for i in range(n)])
    assert len(out) == n
    assert {m["threadId"] for m in out} == {f"t-{i}" for i in range(n)}


# ---------------------------------------------------------------------------
# Legacy Dexie import ledger
# ---------------------------------------------------------------------------


def test_legacy_imports_empty_by_default(tmp_path, monkeypatch):
    _reset_studio_db(tmp_path, monkeypatch)
    assert studio_db.list_chat_legacy_imports() == []


def test_legacy_imports_records_and_lists(tmp_path, monkeypatch):
    _reset_studio_db(tmp_path, monkeypatch)
    accepted, inserted = studio_db.upsert_chat_legacy_imports(
        ["legacy-a", "legacy-b", "legacy-c"],
    )
    assert accepted == 3
    assert inserted == 3
    assert set(studio_db.list_chat_legacy_imports()) == {"legacy-a", "legacy-b", "legacy-c"}


def test_legacy_imports_is_idempotent(tmp_path, monkeypatch):
    _reset_studio_db(tmp_path, monkeypatch)
    accepted1, inserted1 = studio_db.upsert_chat_legacy_imports(
        ["legacy-a", "legacy-b"],
    )
    accepted2, inserted2 = studio_db.upsert_chat_legacy_imports(
        ["legacy-b", "legacy-c"],
    )
    assert (accepted1, inserted1) == (2, 2)
    # legacy-b is already in the ledger, only legacy-c is genuinely new.
    assert (accepted2, inserted2) == (2, 1)
    assert set(studio_db.list_chat_legacy_imports()) == {"legacy-a", "legacy-b", "legacy-c"}


def test_legacy_imports_dedups_input(tmp_path, monkeypatch):
    _reset_studio_db(tmp_path, monkeypatch)
    accepted, inserted = studio_db.upsert_chat_legacy_imports(
        ["x", "x", "y", "x"],
    )
    # accepted is the deduped non-empty input size; inserted is the rows newly
    # added to the ledger after ON CONFLICT DO NOTHING.
    assert accepted == 2
    assert inserted == 2
    assert set(studio_db.list_chat_legacy_imports()) == {"x", "y"}


def test_legacy_imports_ignores_empty(tmp_path, monkeypatch):
    _reset_studio_db(tmp_path, monkeypatch)
    assert studio_db.upsert_chat_legacy_imports([]) == (0, 0)
    assert studio_db.upsert_chat_legacy_imports(["", None]) == (0, 0)  # type: ignore[list-item]
    assert studio_db.list_chat_legacy_imports() == []


# ---------------------------------------------------------------------------
# fork_chat_thread
# ---------------------------------------------------------------------------


def _msg(mid: str, parent: str | None, t: int) -> dict:
    return {
        "id": mid,
        "threadId": "src",
        "parentId": parent,
        "role": "user",
        "content": [{"type": "text", "text": mid}],
        "createdAt": t,
    }


def test_fork_chat_thread_copies_ancestry_with_fresh_ids(tmp_path, monkeypatch):
    _reset_studio_db(tmp_path, monkeypatch)
    studio_db.upsert_chat_thread(
        {**_thread("src"), "title": "Original", "openaiCodeExecContainerId": "cnt-x"}
    )
    # Linear chain: m1 -> m2 -> m3. Plus a sibling m4 off m2 (should NOT
    # be copied since we fork at m3).
    studio_db.sync_chat_messages(
        "src",
        [
            _msg("m1", None, 1),
            _msg("m2", "m1", 2),
            _msg("m3", "m2", 3),
            _msg("m4", "m2", 4),  # sibling — must be excluded
        ],
    )

    counter = {"i": 0}

    def id_factory():
        counter["i"] += 1
        return f"new-{counter['i']}"

    forked = studio_db.fork_chat_thread(
        source_thread_id = "src",
        branch_message_id = "m3",
        new_thread_id = "fork-1",
        new_title = "fork · Original",
        created_at = 99,
        id_factory = id_factory,
    )
    assert forked is not None
    assert forked["id"] == "fork-1"
    assert forked["forkedFromThreadId"] == "src"
    assert forked["forkedFromMessageId"] == "m3"
    # Container ids reset on fork.
    assert forked["openaiCodeExecContainerId"] is None

    copied = studio_db.list_chat_messages("fork-1")
    # 3 ancestors (m1, m2, m3); m4 excluded.
    assert len(copied) == 3
    # parent_id rewritten using new ids; root has parentId None.
    assert copied[0]["parentId"] is None
    assert copied[1]["parentId"] == copied[0]["id"]
    assert copied[2]["parentId"] == copied[1]["id"]
    # All new ids regenerated.
    assert {m["id"] for m in copied}.isdisjoint({"m1", "m2", "m3"})


def test_fork_chat_thread_preserves_project_id(tmp_path, monkeypatch):
    _reset_studio_db(tmp_path, monkeypatch)
    studio_db.upsert_chat_project(_project("project-1"))
    studio_db.upsert_chat_thread({**_thread("src"), "projectId": "project-1"})
    studio_db.upsert_chat_message(_msg("m1", None, 1))

    forked = studio_db.fork_chat_thread(
        source_thread_id = "src",
        branch_message_id = "m1",
        new_thread_id = "fork-1",
        new_title = "fork · Original",
        created_at = 99,
        id_factory = lambda: "new-1",
    )

    assert forked is not None
    assert forked["projectId"] == "project-1"
    assert {thread["id"] for thread in studio_db.list_chat_threads(project_id = "project-1")} == {
        "fork-1",
        "src",
    }


def test_fork_chat_thread_returns_none_for_missing_source(tmp_path, monkeypatch):
    _reset_studio_db(tmp_path, monkeypatch)
    result = studio_db.fork_chat_thread(
        source_thread_id = "nope",
        branch_message_id = "m1",
        new_thread_id = "fork",
        new_title = "f",
        created_at = 1,
        id_factory = lambda: "x",
    )
    assert result is None


def test_count_forks_for_message(tmp_path, monkeypatch):
    _reset_studio_db(tmp_path, monkeypatch)
    studio_db.upsert_chat_thread(_thread("src"))
    studio_db.sync_chat_messages("src", [_msg("m1", None, 1)])
    assert studio_db.count_forks_for_message("src", "m1") == 0

    counter = {"i": 0}

    def id_factory():
        counter["i"] += 1
        return f"id-{counter['i']}"

    studio_db.fork_chat_thread(
        source_thread_id = "src",
        branch_message_id = "m1",
        new_thread_id = "f1",
        new_title = "f1",
        created_at = 2,
        id_factory = id_factory,
    )
    studio_db.fork_chat_thread(
        source_thread_id = "src",
        branch_message_id = "m1",
        new_thread_id = "f2",
        new_title = "f2",
        created_at = 3,
        id_factory = id_factory,
    )
    assert studio_db.count_forks_for_message("src", "m1") == 2
