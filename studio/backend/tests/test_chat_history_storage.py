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


def test_chat_export_reads_one_snapshot_during_concurrent_delete(tmp_path, monkeypatch):
    _reset_studio_db(tmp_path, monkeypatch)
    project = studio_db.upsert_chat_project(_project())
    thread = {**_thread(), "projectId": project["id"]}
    studio_db.upsert_chat_thread(thread)
    studio_db.upsert_chat_message(_message("msg-1", 1, "hello"))

    original_get_connection = studio_db.get_connection
    reader = original_get_connection()
    deleted = False

    class InterleavingConnection:
        def execute(
            self,
            sql,
            parameters = (),
        ):
            nonlocal deleted
            cursor = reader.execute(sql, parameters)
            if not deleted and "SELECT * FROM chat_threads" in sql:
                deleted = True
                writer = original_get_connection()
                try:
                    writer.execute("DELETE FROM chat_threads WHERE id = ?", (thread["id"],))
                    writer.execute("DELETE FROM chat_projects WHERE id = ?", (project["id"],))
                    writer.commit()
                finally:
                    writer.close()
            return cursor

        def __getattr__(self, name):
            return getattr(reader, name)

    monkeypatch.setattr(studio_db, "get_connection", lambda: InterleavingConnection())
    projects, threads, messages = studio_db.build_chat_history_export()

    assert deleted
    assert [item["id"] for item in projects] == [project["id"]]
    assert [item["id"] for item in threads] == [thread["id"]]
    assert [item["id"] for item in messages] == ["msg-1"]


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


def test_chat_thread_preserves_gguf_variant(tmp_path, monkeypatch):
    _reset_studio_db(tmp_path, monkeypatch)
    thread = {**_thread(), "modelGgufVariant": "Q6_K"}

    assert studio_db.upsert_chat_thread(thread)["modelGgufVariant"] == "Q6_K"
    studio_db.upsert_chat_thread(_thread())
    assert studio_db.get_chat_thread("thread-1")["modelGgufVariant"] == "Q6_K"

    updated = studio_db.update_chat_thread("thread-1", {"modelGgufVariant": "Q8_0"})
    assert updated is not None
    assert updated["modelGgufVariant"] == "Q8_0"

    replacement = {**_thread(), "modelId": "other-model"}
    assert studio_db.upsert_chat_thread(replacement)["modelGgufVariant"] is None


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
    assert studio_db.get_chat_thread("thread-with-msgs")["modelGgufVariant"] is None


def test_chat_projects_delete_cascades_threads_and_messages(tmp_path, monkeypatch):
    _reset_studio_db(tmp_path, monkeypatch)
    project = studio_db.upsert_chat_project(_project())
    project_root = Path(project["rootPath"])
    assert project_root.parent == tmp_path / "Projects"
    assert project_root.name.startswith("Research-project")
    assert project_root.is_dir()
    assert (project_root / "sandbox").is_dir()
    assert not (project_root / "chats").exists()
    assert not (project_root / "files").exists()
    assert not (project_root / "exports").exists()
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
    assert project_root.exists()
    with pytest.raises(studio_db.ChatThreadDeletedError):
        studio_db.upsert_chat_thread({**_thread(), "projectId": "project-1"})


def test_thread_delete_blocks_a_late_create_with_the_same_id(tmp_path, monkeypatch):
    _reset_studio_db(tmp_path, monkeypatch)

    studio_db.delete_chat_threads(["late-thread"])

    with pytest.raises(studio_db.ChatThreadDeletedError):
        studio_db.upsert_chat_thread(_thread("late-thread"))
    assert studio_db.get_chat_thread("late-thread") is None


def test_clear_blocks_a_stale_recreate_of_a_deleted_thread(tmp_path, monkeypatch):
    _reset_studio_db(tmp_path, monkeypatch)
    studio_db.upsert_chat_thread(_thread("thread-1"))

    studio_db.clear_chat_history_with_active_research_runs(["pending-thread"])

    with pytest.raises(studio_db.ChatThreadDeletedError):
        studio_db.upsert_chat_thread(_thread("thread-1"))
    with pytest.raises(studio_db.ChatThreadDeletedError):
        studio_db.upsert_chat_thread(_thread("pending-thread"))
    assert studio_db.get_chat_thread("thread-1") is None
    assert studio_db.get_chat_thread("pending-thread") is None


def test_clear_reports_only_threads_that_had_a_row(tmp_path, monkeypatch):
    _reset_studio_db(tmp_path, monkeypatch)
    studio_db.upsert_chat_thread(_thread("thread-1"))

    _, deleted_ids = studio_db.clear_chat_history_with_active_research_runs(["never-committed"])

    # the fenced id is still tombstoned, but reporting it would inflate the cleared count
    assert deleted_ids == ["thread-1"]
    with pytest.raises(studio_db.ChatThreadDeletedError):
        studio_db.upsert_chat_thread(_thread("never-committed"))


def test_repeated_clear_operation_does_not_delete_later_threads(tmp_path, monkeypatch):
    _reset_studio_db(tmp_path, monkeypatch)
    studio_db.upsert_chat_thread(_thread("before-clear"))

    _, first_deleted_ids = studio_db.clear_chat_history_with_active_research_runs(
        operation_id = "clear-operation-1"
    )
    studio_db.upsert_chat_thread(_thread("after-clear"))

    _, repeated_deleted_ids = studio_db.clear_chat_history_with_active_research_runs(
        operation_id = "clear-operation-1"
    )

    assert first_deleted_ids == ["before-clear"]
    assert repeated_deleted_ids == first_deleted_ids
    assert studio_db.get_chat_thread("before-clear") is None
    assert studio_db.get_chat_thread("after-clear") is not None


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


def test_settings_merge_keeps_each_model_s_remembered_params(tmp_path, monkeypatch):
    """Per-model memory patches one model at a time, so the merge has to keep the
    others. Without this, tuning a second model would wipe the first one's settings
    and the switch back would land on whatever the last edit happened to be."""
    _reset_studio_db(tmp_path, monkeypatch)
    studio_db.upsert_chat_settings_merge(
        {"inferenceParamsByModel": {"qwen": {"temperature": 0.2, "topP": 0.8}}}
    )
    studio_db.upsert_chat_settings_merge(
        {"inferenceParamsByModel": {"llama": {"temperature": 0.9}}}
    )
    # A second edit to the first model merges into its own entry.
    studio_db.upsert_chat_settings_merge({"inferenceParamsByModel": {"qwen": {"temperature": 0.4}}})

    by_model = studio_db.list_chat_settings()["inferenceParamsByModel"]
    assert by_model == {"qwen": {"temperature": 0.4, "topP": 0.8}, "llama": {"temperature": 0.9}}


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
        {
            **_thread("src"),
            "title": "Original",
            "modelGgufVariant": "Q6_K",
            "openaiCodeExecContainerId": "cnt-x",
        }
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
    assert forked["modelGgufVariant"] == "Q6_K"
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


def test_fork_chat_thread_detaches_research_run_metadata(tmp_path, monkeypatch):
    _reset_studio_db(tmp_path, monkeypatch)
    studio_db.upsert_chat_thread(_thread("src"))
    studio_db.upsert_chat_message(_msg("user", None, 1))
    studio_db.upsert_chat_message(
        {
            "id": "research-report",
            "threadId": "src",
            "parentId": "user",
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "# Copied report",
                    "researchRunId": "run-source",
                },
                {
                    "type": "source",
                    "url": "https://example.com",
                    "title": "Example",
                    "researchStatus": "completed",
                },
            ],
            "metadata": {
                "researchRunId": "run-source",
                "researchStatus": "completed",
                "researchPlanRevision": 1,
                "serverManaged": True,
                "model": "local-model",
            },
            "createdAt": 2,
        }
    )

    studio_db.fork_chat_thread(
        source_thread_id = "src",
        branch_message_id = "research-report",
        new_thread_id = "fork-1",
        new_title = "fork",
        created_at = 3,
        id_factory = iter(("fork-user", "fork-report")).__next__,
    )

    report = next(
        message
        for message in studio_db.list_chat_messages("fork-1")
        if message["role"] == "assistant"
    )
    assert report["content"][0]["text"] == "# Copied report"
    assert report["content"][1]["url"] == "https://example.com"
    assert all(
        not ({"researchRunId", "researchStatus", "serverManaged"} & set(part))
        for part in report["content"]
    )
    assert report["metadata"] == {"model": "local-model"}


def test_fork_detachment_detects_non_id_research_content_keys():
    content_json, metadata_json = studio_db._detach_research_message_json(
        '[{"type":"text","text":"Report","serverManaged":true}]',
        '{"model":"local-model","generationRunId":"run-1","generationSeq":3}',
    )

    assert "serverManaged" not in content_json
    assert "generationRunId" not in metadata_json
    assert metadata_json == '{"model": "local-model"}'


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


def test_fork_chat_thread_rejects_a_deleted_target_id(tmp_path, monkeypatch):
    _reset_studio_db(tmp_path, monkeypatch)
    studio_db.upsert_chat_thread(_thread("src"))
    studio_db.upsert_chat_message(_msg("m1", None, 1))
    studio_db.delete_chat_threads(["fork"])

    with pytest.raises(studio_db.ChatThreadDeletedError):
        studio_db.fork_chat_thread(
            source_thread_id = "src",
            branch_message_id = "m1",
            new_thread_id = "fork",
            new_title = "f",
            created_at = 2,
            id_factory = lambda: "new-1",
        )
    assert studio_db.get_chat_thread("fork") is None


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


def test_fork_counts_for_thread(tmp_path, monkeypatch):
    """One read for the whole thread, so a rendered thread costs one request, not one per message."""
    _reset_studio_db(tmp_path, monkeypatch)
    studio_db.upsert_chat_thread(_thread("src"))
    studio_db.sync_chat_messages(
        "src", [_msg("m1", None, 1), _msg("m2", "m1", 2), _msg("m3", "m2", 3)]
    )
    assert studio_db.fork_counts_for_thread("src") == {}

    counter = {"i": 0}

    def id_factory():
        counter["i"] += 1
        return f"id-{counter['i']}"

    for index, (branch, new_id) in enumerate([("m1", "f1"), ("m1", "f2"), ("m2", "f3")]):
        studio_db.fork_chat_thread(
            source_thread_id = "src",
            branch_message_id = branch,
            new_thread_id = new_id,
            new_title = new_id,
            created_at = 10 + index,
            id_factory = id_factory,
        )

    counts = studio_db.fork_counts_for_thread("src")
    assert counts == {"m1": 2, "m2": 1}
    # Same answer as the per-message read it replaces, message for message.
    for message_id in ["m1", "m2", "m3"]:
        assert counts.get(message_id, 0) == studio_db.count_forks_for_message("src", message_id)
    # Another thread's forks never leak in.
    assert studio_db.fork_counts_for_thread("f1") == {}


def _research_thread(
    tmp_path,
    monkeypatch,
    *,
    extra_ancestors: int = 1,
):
    """A thread shaped `a0 -> ... -> prompt -> report`, with the pair claimed by a research run.

    Returns the ancestor ids in order. `prompt` and `report` are the server-managed pair.
    """
    from storage import research_runs_db

    _reset_studio_db(tmp_path, monkeypatch)
    studio_db.upsert_chat_thread(_thread("src"))

    ancestors = [f"a{index}" for index in range(extra_ancestors)]
    chain = [*ancestors, "prompt", "report"]
    messages = []
    for position, message_id in enumerate(chain):
        messages.append(
            {
                "id": message_id,
                "threadId": "src",
                "parentId": chain[position - 1] if position else None,
                "role": "assistant" if message_id == "report" else "user",
                "content": [{"type": "text", "text": message_id}],
                "metadata": (
                    {"researchRunId": "run-1", "serverManaged": True}
                    if message_id == "report"
                    else None
                ),
                "createdAt": position + 1,
            }
        )
    studio_db.sync_chat_messages("src", messages)
    research_runs_db.create_run(
        run_id = "run-1",
        owner_subject = "owner",
        thread_id = "src",
        user_message_id = "prompt",
        assistant_message_id = "report",
        config = {},
        created_at = 1,
    )
    # create_run may rewrite the pair, so the baseline has to be what the server now holds.
    return ancestors, studio_db.list_chat_messages("src")


def _without(messages: list[dict], *drop: str) -> list[dict]:
    return [dict(m) for m in messages if m["id"] not in drop]


def test_deleting_an_ancestor_relinks_the_research_prompt(tmp_path, monkeypatch):
    # The headline case: assistant-ui relinks the protected prompt to the deleted node's parent,
    # and the guard must read that as the repair it is rather than an edit.
    _, messages = _research_thread(tmp_path, monkeypatch)
    payload = _without(messages, "a0")
    payload[0]["parentId"] = None

    synced = studio_db.sync_chat_messages("src", payload, prune_missing = True)

    assert [m["id"] for m in synced] == ["prompt", "report"]
    assert synced[0]["parentId"] is None


def test_deleting_a_mid_chain_ancestor_relinks_to_the_surviving_grandparent(tmp_path, monkeypatch):
    _, messages = _research_thread(tmp_path, monkeypatch, extra_ancestors = 3)
    payload = _without(messages, "a1", "a2")
    next(m for m in payload if m["id"] == "prompt")["parentId"] = "a0"

    synced = studio_db.sync_chat_messages("src", payload, prune_missing = True)

    assert next(m for m in synced if m["id"] == "prompt")["parentId"] == "a0"


def test_a_relink_to_a_surviving_message_that_is_not_the_ancestor_is_ignored(tmp_path, monkeypatch):
    # The bulk sync keeps the server copy instead of rejecting the batch, so the protection is
    # that the claim is dropped: the reseat is walked from the stored chain, and a client cannot
    # use a pruned parent as cover for pointing a protected message anywhere it likes.
    _, messages = _research_thread(tmp_path, monkeypatch, extra_ancestors = 3)
    payload = _without(messages, "a1", "a2")
    next(m for m in payload if m["id"] == "prompt")["parentId"] = "report"

    synced = studio_db.sync_chat_messages("src", payload, prune_missing = True)

    assert next(m for m in synced if m["id"] == "prompt")["parentId"] == "a0"


def test_a_relink_with_nothing_pruned_leaves_the_stored_parent_alone(tmp_path, monkeypatch):
    _, messages = _research_thread(tmp_path, monkeypatch)
    payload = [dict(m) for m in messages]
    next(m for m in payload if m["id"] == "prompt")["parentId"] = None

    synced = studio_db.sync_chat_messages("src", payload, prune_missing = True)

    # Nothing was deleted, so there is no repair to make and the claim is simply dropped.
    assert next(m for m in synced if m["id"] == "prompt")["parentId"] == "a0"


def test_a_relink_is_ignored_when_pruning_is_off(tmp_path, monkeypatch):
    _, messages = _research_thread(tmp_path, monkeypatch)
    payload = _without(messages, "a0")
    payload[0]["parentId"] = None

    synced = studio_db.sync_chat_messages("src", payload, prune_missing = False)

    # With pruning off the omitted ancestor survives, so the stored parent still resolves.
    assert {m["id"] for m in synced} >= {"a0", "prompt"}
    assert next(m for m in synced if m["id"] == "prompt")["parentId"] == "a0"


@pytest.mark.parametrize(
    "field, value",
    [
        ("content", [{"type": "text", "text": "edited"}]),
        ("role", "assistant"),
        ("metadata", {"tampered": True}),
        ("createdAt", 999),
    ],
)
def test_the_reseat_does_not_carry_any_other_edit(tmp_path, monkeypatch, field, value):
    # Structure is repaired, content is not adopted: the reseat must not become a hole through
    # which a drifted autosave rewrites the protected row.
    _, messages = _research_thread(tmp_path, monkeypatch)
    stored = next(m for m in messages if m["id"] == "prompt")
    payload = _without(messages, "a0")
    payload[0]["parentId"] = None
    payload[0][field] = value

    synced = studio_db.sync_chat_messages("src", payload, prune_missing = True)

    prompt = next(m for m in synced if m["id"] == "prompt")
    # The deleted ancestor was the root, so the repair is a reseat to the root.
    assert prompt["parentId"] is None
    # .get on both sides: an absent key is how a None metadata comes back, and the point is
    # that the client's value is not there either way.
    assert prompt.get(field) == stored.get(field)
    assert prompt.get(field) != value


def test_deleting_a_protected_message_itself_is_still_refused(tmp_path, monkeypatch):
    # Update permission is not delete permission. Omitting a protected message no longer 409s
    # the batch, but it must not delete it either.
    _, messages = _research_thread(tmp_path, monkeypatch)

    for dropped in ("prompt", "report"):
        synced = studio_db.sync_chat_messages(
            "src", _without(messages, dropped), prune_missing = True
        )
        assert dropped in {m["id"] for m in synced}


def test_a_research_prompt_already_at_the_root_is_unaffected(tmp_path, monkeypatch):
    _, messages = _research_thread(tmp_path, monkeypatch, extra_ancestors = 0)

    synced = studio_db.sync_chat_messages("src", messages, prune_missing = True)

    assert [m["id"] for m in synced] == ["prompt", "report"]


def test_an_unrelated_sibling_can_still_be_deleted(tmp_path, monkeypatch):
    _, messages = _research_thread(tmp_path, monkeypatch)
    sibling = {
        "id": "sibling",
        "threadId": "src",
        "parentId": "a0",
        "role": "user",
        "content": [{"type": "text", "text": "sibling"}],
        "createdAt": 9,
    }
    studio_db.sync_chat_messages("src", [*messages, sibling])

    synced = studio_db.sync_chat_messages("src", messages, prune_missing = True)

    assert "sibling" not in {m["id"] for m in synced}


def test_a_plain_message_whose_parent_is_pruned_is_never_guarded(tmp_path, monkeypatch):
    # Only protected ids reach the guard at all; an ordinary relink must stay untouched by any of
    # this, including when its own parent is the pruned node.
    _, messages = _research_thread(tmp_path, monkeypatch)
    plain_parent = {
        "id": "plain-parent",
        "threadId": "src",
        "parentId": "report",
        "role": "user",
        "content": [{"type": "text", "text": "plain-parent"}],
        "createdAt": 8,
    }
    plain_child = {
        "id": "plain-child",
        "threadId": "src",
        "parentId": "plain-parent",
        "role": "assistant",
        "content": [{"type": "text", "text": "plain-child"}],
        "createdAt": 9,
    }
    studio_db.sync_chat_messages("src", [*messages, plain_parent, plain_child])

    relinked = {**plain_child, "parentId": "report"}
    synced = studio_db.sync_chat_messages("src", [*messages, relinked], prune_missing = True)

    assert next(m for m in synced if m["id"] == "plain-child")["parentId"] == "report"


def test_a_corrupt_self_link_resolves_to_the_root_rather_than_itself(tmp_path, monkeypatch):
    # A thread can only reach this shape by storing a cycle among its own unprotected rows, but
    # the walk must still hand back a link the tree can hold rather than a message's own id.
    _, messages = _research_thread(tmp_path, monkeypatch)
    cyclic = [dict(m) for m in messages]
    next(m for m in cyclic if m["id"] == "a0")["parentId"] = "prompt"
    studio_db.sync_chat_messages("src", cyclic)

    conn = studio_db.get_connection()
    try:
        assert studio_db._surviving_parent_id(conn, "src", "prompt", {"a0"}) is None
    finally:
        conn.close()


def test_an_empty_stored_parent_reads_as_the_root(tmp_path, monkeypatch):
    # parent_id is nullable, so '' is only reachable through a direct writer, but the helper and
    # the caller's `or None` normalization must agree about it either way.
    _, messages = _research_thread(tmp_path, monkeypatch)
    conn = studio_db.get_connection()
    try:
        conn.execute("UPDATE chat_messages SET parent_id = '' WHERE id = 'a0'")
        conn.commit()
        assert studio_db._surviving_parent_id(conn, "src", "a0", set()) is None
    finally:
        conn.close()


def test_deleting_a_thread_signals_only_research_runs_a_worker_owns(tmp_path, monkeypatch):
    # A fresh run sits in 'planning' with no lease. Deleting its thread cascades the row away, so
    # no worker can ever claim it; signalling it would leave a cancellation event in the
    # supervisor that nothing is left to consume.
    _research_thread(tmp_path, monkeypatch)

    assert studio_db.delete_chat_threads_with_active_research_runs(["src"]) == []


def test_deleting_a_thread_signals_a_leased_research_run(tmp_path, monkeypatch):
    # The same run once a worker owns it: that worker is still running and has to be told.
    _research_thread(tmp_path, monkeypatch)
    conn = studio_db.get_connection()
    try:
        conn.execute(
            "UPDATE research_runs SET lease_owner = 'worker-1', status = 'running' WHERE id = ?",
            ("run-1",),
        )
        conn.commit()
    finally:
        conn.close()

    assert studio_db.delete_chat_threads_with_active_research_runs(["src"]) == ["run-1"]


def test_replaying_a_clear_does_not_signal_its_research_runs_again(tmp_path, monkeypatch):
    # The request that recorded the operation already signalled these runs on its way out. Its
    # worker may have exited since, so a second signal would leave a cancellation event in the
    # supervisor that nothing is left to consume.
    _research_thread(tmp_path, monkeypatch)
    conn = studio_db.get_connection()
    try:
        conn.execute(
            "UPDATE research_runs SET lease_owner = 'worker-1', status = 'running' WHERE id = ?",
            ("run-1",),
        )
        conn.commit()
    finally:
        conn.close()

    first = studio_db.clear_chat_history_with_active_research_runs(operation_id = "op-1")
    assert first == (["run-1"], ["src"])

    replay = studio_db.clear_chat_history_with_active_research_runs(operation_id = "op-1")
    assert replay == ([], ["src"])
