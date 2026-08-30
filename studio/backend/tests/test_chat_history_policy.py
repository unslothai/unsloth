# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
import os
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from core.inference import tools
from core.rag import conversation_archive, history_cleanup, store, tool
from models.inference import ChatCompletionRequest
from routes import chat_generation_runs, chat_history, inference, rag, research_runs
from storage import studio_db
from utils import chat_history_policy


@pytest.mark.parametrize("value", (None, "", "0", "false", "NO", "off"))
def test_policy_false_values(value):
    assert chat_history_policy._parse(value) is False


@pytest.mark.parametrize("value", ("1", "true", "YES", " on ", "invalid"))
def test_policy_true_and_invalid_values_fail_closed(value):
    assert chat_history_policy._parse(value) is True


@pytest.fixture
def no_chat_history(monkeypatch):
    monkeypatch.setattr(chat_history_policy, "NO_CHAT_HISTORY", True)


def test_history_reads_are_empty_and_writes_are_rejected(no_chat_history, monkeypatch):
    def unexpected(*_args, **_kwargs):
        raise AssertionError("history storage must not be read or written")

    monkeypatch.setattr(chat_history, "list_chat_threads", unexpected)
    monkeypatch.setattr(chat_history, "upsert_chat_thread", unexpected)
    monkeypatch.setattr(chat_history, "build_chat_history_export", unexpected)

    assert chat_history.list_threads(current_subject = "alice").threads == []
    assert chat_history.count_threads(current_subject = "alice").count == 0
    assert chat_history.export_history(current_subject = "alice").threads == []

    payload = chat_history.ChatThread(
        id = "thread-1",
        modelType = "base",
        modelId = "local",
        createdAt = 1,
    )
    with pytest.raises(HTTPException) as exc_info:
        chat_history.save_thread(payload, current_subject = "alice")
    assert exc_info.value.status_code == 403
    assert exc_info.value.detail == chat_history_policy.DISABLED_DETAIL


def test_destructive_cleanup_remains_available(no_chat_history, monkeypatch):
    deleted = []

    def delete(message_id, attachment_id):
        deleted.append((message_id, attachment_id))
        return True

    monkeypatch.setattr(chat_history, "delete_chat_attachment", delete)
    assert chat_history.delete_attachment("message-1", "attachment-1", "alice") == {"ok": True}
    assert deleted == [("message-1", "attachment-1")]


def test_project_cleanup_response_redacts_stored_content(no_chat_history, monkeypatch):
    project = {
        "id": "project-1",
        "name": "Secret project",
        "instructions": "Secret instructions",
        "rootPath": "/secret/root",
        "sandboxPath": None,
        "archived": True,
        "createdAt": 1,
        "updatedAt": 2,
        "memberIds": [],
        "activeResearchRunIds": [],
        "activeChatGenerationRunIds": [],
    }
    monkeypatch.setattr(chat_history, "delete_chat_project", lambda *_args, **_kwargs: project)
    monkeypatch.setattr(chat_history, "_archive_cutoff", lambda: None)
    monkeypatch.setattr(chat_history, "_cancel_research_runs", lambda *_args: None)
    monkeypatch.setattr(chat_history, "_cancel_chat_generation_runs", lambda *_args: None)
    monkeypatch.setattr(chat_history, "_cancel_active_generations", lambda *_args: None)
    monkeypatch.setattr(chat_history, "_delete_project_rag_sources", lambda *_args: None)
    monkeypatch.setattr(
        chat_history, "_remove_conversation_archives", lambda *_args, **_kwargs: None
    )

    async def remove_sandboxes(*_args, **_kwargs):
        return [], []

    monkeypatch.setattr(chat_history, "_remove_sandboxes", remove_sandboxes)
    response = asyncio.run(
        chat_history.delete_project(
            "project-1",
            SimpleNamespace(),
            delete_files = False,
            current_subject = "alice",
        )
    )
    assert response.id == "project-1"
    assert response.name == ""
    assert response.instructions == ""
    assert response.rootPath is None
    assert response.sandboxPath is None


def test_durable_run_creation_is_rejected_and_lists_are_hidden(no_chat_history):
    generation = chat_generation_runs.CreateChatGenerationRun(
        runId = "run-1",
        threadId = "thread-1",
        userMessageId = "user-1",
        assistantMessageId = "assistant-1",
        requestPayload = {},
    )
    with pytest.raises(HTTPException) as generation_error:
        asyncio.run(
            chat_generation_runs.create_chat_generation_run(
                generation,
                SimpleNamespace(),
                current_subject = "alice",
            )
        )
    assert generation_error.value.status_code == 403
    assert chat_generation_runs.active_chat_generation_runs("thread-1", "alice") == {"runs": []}

    research = research_runs.CreateResearchRun(threadId = "thread-1", userMessageId = "user-1")
    with pytest.raises(HTTPException) as research_error:
        research_runs.create_research_run(research, SimpleNamespace(), current_subject = "alice")
    assert research_error.value.status_code == 403
    assert research_runs.active_research_runs("thread-1", "alice") == {
        "runs": [],
        "hasRun": False,
    }


def test_durable_run_cancellation_returns_no_stored_content(no_chat_history, monkeypatch):
    request = SimpleNamespace(app = SimpleNamespace(state = SimpleNamespace()))

    monkeypatch.setattr(
        chat_generation_runs,
        "_require_run",
        lambda _run_id: {"requestPayload": {"messages": ["secret"]}},
    )
    monkeypatch.setattr(
        chat_generation_runs.db,
        "request_cancel",
        lambda _run_id: {
            "id": "generation-1",
            "status": "cancelled",
            "requestPayload": {"messages": ["secret"]},
        },
    )
    assert chat_generation_runs.cancel_chat_generation_run(
        "generation-1", request, current_subject = "alice"
    ) == {"id": "generation-1", "status": "cancelled"}

    monkeypatch.setattr(
        research_runs,
        "_require_run",
        lambda _run_id: {"config": {"question": "secret"}},
    )
    monkeypatch.setattr(research_runs.db, "request_cancel", lambda _run_id: "cancelled")
    monkeypatch.setattr(
        research_runs,
        "_sync_assistant",
        lambda _run: (_ for _ in ()).throw(
            AssertionError("cancellation must not rewrite chat history")
        ),
    )
    assert research_runs.cancel_research_run("research-1", request, current_subject = "alice") == {
        "id": "research-1",
        "status": "cancelled",
    }


def test_only_explicit_knowledge_base_rag_scopes_remain(no_chat_history):
    assert tool._resolve_scope("kb-1", "thread-1", "project-1") == "kb_kb-1"
    assert tool._resolve_scope(None, "thread-1", None) is None
    assert tool._resolve_scope(None, None, "project-1") is None
    assert tool.whole_document_context(scope_thread_id = "thread-1", max_tokens = 100) is None
    assert conversation_archive.enabled() is False


def test_project_session_cannot_resolve_a_stored_workspace(no_chat_history, monkeypatch):
    monkeypatch.setattr(
        tools,
        "_get_project_workdir",
        lambda _session_id: (_ for _ in ()).throw(
            AssertionError("project storage must not be resolved")
        ),
    )
    assert tools._project_workdir_for(tools.project_session_id("project-1")) is None


def test_file_tools_and_output_spills_are_disabled(no_chat_history, monkeypatch, tmp_path):
    available = tools.apply_chat_history_tool_policy(tools.ALL_TOOLS)
    names = {entry["function"]["name"] for entry in available}
    assert {"python", "terminal", "edit_file"}.isdisjoint(names)
    assert "web_search" in names

    monkeypatch.setattr(
        tools,
        "_get_workdir",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("disabled tools must not create a conversation sandbox")
        ),
    )
    for name, arguments in (
        ("python", {"code": "print('secret')"}),
        ("terminal", {"command": "printf secret > output.txt"}),
        (
            "edit_file",
            {"path": "output.txt", "edits": [{"old_string": "", "new_string": "secret"}]},
        ),
    ):
        result = tools.execute_tool(name, arguments, session_id = "thread-1")
        assert "unavailable while chat history is disabled" in result

    scope = tools._spill_scope("thread-1", "thread-1")
    spill, complete = tools._spill_full_output("secret" * 10_000, str(tmp_path), scope)
    assert (spill, complete) == (None, True)
    assert list(tmp_path.iterdir()) == []


def test_request_tool_catalog_keeps_only_non_file_tools(no_chat_history, monkeypatch):
    monkeypatch.setattr(inference, "_search_images_enabled", lambda: False)
    payload = ChatCompletionRequest(
        messages = [{"role": "user", "content": "search"}],
        enabled_tools = ["python", "terminal", "edit_file", "web_search"],
        deep_research_armed = True,
    )

    selected = asyncio.run(
        inference._select_request_tools(payload, tools_on = True, mcp_allowed = False)
    )
    assert [entry["function"]["name"] for entry in selected] == ["web_search"]
    hosted_payload = ChatCompletionRequest(
        messages = [{"role": "user", "content": "search"}],
        enabled_tools = ["web_search"],
        deep_research_armed = True,
    )
    assert inference._selects_only_provider_hosted_tools(hosted_payload, "openai") is True

    anthropic = inference._select_anthropic_server_tools(
        tools.ALL_TOOLS,
        {"python", "terminal", "web_search"},
        None,
    )
    assert [entry["function"]["name"] for entry in anthropic] == ["web_search"]


def test_sandbox_reads_are_hidden_without_resolving_stored_paths(no_chat_history, monkeypatch):
    async def authenticate(*_args, **_kwargs):
        return None

    monkeypatch.setattr(inference, "_authenticate_header_or_query", authenticate)
    monkeypatch.setattr(
        inference,
        "_sandbox_dir_for",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("hidden sandbox paths must not be resolved")
        ),
    )

    listed = asyncio.run(
        inference.list_sandbox_files("thread-1", request = None, token = None, session = None)
    )
    assert listed == {"path": "", "files": []}

    with pytest.raises(HTTPException) as reveal_error:
        asyncio.run(
            inference.reveal_sandbox_dir("thread-1", request = None, token = None, session = None)
        )
    assert reveal_error.value.status_code == 404

    with pytest.raises(HTTPException) as file_error:
        asyncio.run(
            inference.serve_sandbox_file(
                "thread-1",
                "private.txt",
                request = None,
                token = None,
                session = None,
            )
        )
    assert file_error.value.status_code == 404


def test_clear_history_does_not_return_stored_thread_or_sandbox_ids(no_chat_history, monkeypatch):
    from core.inference import search_images

    deleted_projects = []
    monkeypatch.setattr(chat_history, "_archive_cutoff", lambda: None)
    monkeypatch.setattr(chat_history, "list_chat_threads", lambda: [{"id": "thread-secret"}])
    monkeypatch.setattr(
        chat_history,
        "list_chat_projects",
        lambda **_kwargs: [{"id": "project-secret"}],
    )
    monkeypatch.setattr(
        chat_history,
        "delete_chat_project",
        lambda project_id, **_kwargs: deleted_projects.append(project_id),
    )
    monkeypatch.setattr(
        chat_history,
        "clear_chat_history",
        lambda **_kwargs: (["thread-secret"], [], []),
    )
    monkeypatch.setattr(chat_history, "_cancel_active_generations", lambda *_args: None)
    monkeypatch.setattr(chat_history, "_cancel_research_runs", lambda *_args: None)
    monkeypatch.setattr(chat_history, "_cancel_chat_generation_runs", lambda *_args: None)
    monkeypatch.setattr(chat_history, "_remove_conversation_archives", lambda *_a, **_k: None)
    rag_cleanups = []
    monkeypatch.setattr(
        history_cleanup,
        "clear_non_knowledge_base_data",
        lambda: rag_cleanups.append(True),
    )

    async def remove_sandboxes(*_args, **_kwargs):
        return 0, ["thread-secret"]

    monkeypatch.setattr(chat_history, "_remove_sandboxes", remove_sandboxes)
    monkeypatch.setattr(search_images, "snapshot_and_fence_registrations", lambda: set())
    monkeypatch.setattr(search_images, "clear_cache", lambda *_args: None)
    monkeypatch.setattr(tools, "collect_orphaned_project_workspaces", lambda: None)
    monkeypatch.setattr(tools, "preserve_orphaned_project", lambda *_args: True)

    result = asyncio.run(
        chat_history.clear_history(request = SimpleNamespace(), current_subject = "alice")
    )
    assert result["deletedThreadIds"] == []
    assert result["sandboxes_kept"] == []
    assert deleted_projects == ["project-secret"]
    assert rag_cleanups == [True]


@pytest.mark.parametrize("same_workspace", (True, False))
@pytest.mark.parametrize("handoff_failure", (True, False))
def test_no_history_clear_keeps_a_recreated_project_workspace(
    no_chat_history, monkeypatch, tmp_path, same_workspace, handoff_failure
):
    from core.inference import search_images

    monkeypatch.setenv("UNSLOTH_STUDIO_PROJECTS_HOME", str(tmp_path / "projects"))
    monkeypatch.setattr(studio_db, "_schema_ready", False)
    project = studio_db.upsert_chat_project(
        {
            "id": "project-recreated",
            "name": "Recreated",
            "instructions": "",
            "archived": False,
            "createdAt": 1,
            "updatedAt": 1,
        }
    )
    workspace = Path(project["rootPath"])
    (workspace / "fresh.txt").write_text("keep", encoding = "utf-8")
    old_workspace = workspace if same_workspace else tmp_path / "Old-project"
    (old_workspace / "sandbox").mkdir(parents = True, exist_ok = True)
    tools.record_orphaned_project(
        project["id"],
        str(old_workspace / "sandbox"),
        True,
        str(old_workspace),
    )

    monkeypatch.setattr(chat_history, "_archive_cutoff", lambda: None)
    monkeypatch.setattr(chat_history, "_cancel_active_generations", lambda *_args: None)
    monkeypatch.setattr(chat_history, "_cancel_research_runs", lambda *_args: None)
    monkeypatch.setattr(chat_history, "_cancel_chat_generation_runs", lambda *_args: None)
    monkeypatch.setattr(chat_history, "_remove_conversation_archives", lambda *_a, **_k: None)
    monkeypatch.setattr(history_cleanup, "clear_non_knowledge_base_data", lambda: 0)
    monkeypatch.setattr(search_images, "snapshot_and_fence_registrations", lambda: set())
    monkeypatch.setattr(search_images, "clear_cache", lambda *_args: None)

    request = SimpleNamespace(app = SimpleNamespace(state = SimpleNamespace()))
    real_write_record = tools._write_orphan_record
    real_delete_workspace = studio_db.delete_project_workspace
    if handoff_failure:
        if same_workspace:
            monkeypatch.setattr(tools, "_write_orphan_record", lambda *_args: False)
        else:
            monkeypatch.setattr(studio_db, "delete_project_workspace", lambda *_args: None)
    if handoff_failure:
        with pytest.raises(HTTPException) as incomplete:
            asyncio.run(chat_history.clear_history(request = request, current_subject = "alice"))
        assert incomplete.value.status_code == 503
        assert incomplete.value.detail == "Chat history cleanup is incomplete. Retry the request."
    else:
        asyncio.run(chat_history.clear_history(request = request, current_subject = "alice"))

    if handoff_failure:
        assert studio_db.get_chat_project(project["id"]) is not None
        assert [record[3] for record in tools.list_orphaned_projects()] == [True]
        monkeypatch.setattr(tools, "_write_orphan_record", real_write_record)
        monkeypatch.setattr(studio_db, "delete_project_workspace", real_delete_workspace)
    asyncio.run(chat_history.clear_history(request = request, current_subject = "alice"))

    assert studio_db.get_chat_project(project["id"]) is None
    assert (workspace / "fresh.txt").read_text(encoding = "utf-8") == "keep"
    assert [record[3] for record in tools.list_orphaned_projects()] == [False]
    if not same_workspace:
        assert not old_workspace.exists()


def test_no_history_rag_cleanup_preserves_global_knowledge_bases(
    no_chat_history, monkeypatch, tmp_path
):
    db_path = tmp_path / "rag.db"
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("external", encoding = "utf-8")

    paths = {
        name: uploads / f"{name}.txt" for name in ("kb", "legacy-kb", "thread", "project", "shared")
    }
    for name, path in paths.items():
        path.write_text(name, encoding = "utf-8")

    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE knowledge_bases(id TEXT PRIMARY KEY);
        CREATE TABLE documents(
            id TEXT PRIMARY KEY, scope TEXT NOT NULL, kb_id TEXT, stored_path TEXT
        );
        CREATE TABLE chunks(id TEXT PRIMARY KEY, document_id TEXT NOT NULL);
        CREATE TABLE chunks_fts(text TEXT, chunk_id TEXT, scope TEXT);
        CREATE TABLE ingestion_jobs(id TEXT PRIMARY KEY, document_id TEXT, scope TEXT);
        CREATE TABLE rag_job_leases(kind TEXT, job_id TEXT);
        CREATE TABLE linked_folders(
            id TEXT PRIMARY KEY, scope_type TEXT, scope_id TEXT, scope TEXT
        );
        CREATE TABLE linked_folder_files(folder_id TEXT, document_id TEXT);
        CREATE TABLE linked_folder_sync_jobs(id TEXT PRIMARY KEY, folder_id TEXT);
        CREATE TABLE linked_folder_retired_scopes(
            scope TEXT PRIMARY KEY, retired_at TEXT NOT NULL, purged_at TEXT
        );
        """
    )
    conn.execute("INSERT INTO knowledge_bases VALUES('kb-1')")
    documents = (
        ("kb", "unusual-global-scope", "kb-1", str(paths["kb"])),
        ("legacy-kb", "kb_kb-1", None, str(paths["legacy-kb"])),
        ("thread", "thread-secret", None, str(paths["thread"])),
        ("project", "project-secret", None, str(paths["project"])),
        ("outside", "thread-outside", None, str(outside)),
        ("shared-chat", "thread-shared", None, str(paths["shared"])),
        ("shared-kb", "kb_kb-1", "kb-1", str(paths["shared"])),
        ("research", "research_scrape_active", None, None),
    )
    conn.executemany("INSERT INTO documents VALUES(?,?,?,?)", documents)
    conn.executemany(
        "INSERT INTO chunks VALUES(?,?)",
        [(f"chunk-{doc[0]}", doc[0]) for doc in documents],
    )
    conn.executemany(
        "INSERT INTO chunks_fts VALUES(?,?,?)",
        [("secret", f"chunk-{doc[0]}", doc[1]) for doc in documents],
    )
    conn.executemany(
        "INSERT INTO ingestion_jobs VALUES(?,?,?)",
        [(f"job-{doc[0]}", doc[0], doc[1]) for doc in documents],
    )
    conn.executemany(
        "INSERT INTO rag_job_leases VALUES('ingestion', ?)",
        [(f"job-{doc[0]}",) for doc in documents],
    )
    conn.executemany(
        "INSERT INTO linked_folders VALUES(?,?,?,?)",
        [
            ("project-folder", "project", "project-1", "project-secret"),
            ("kb-folder", "knowledge_base", "kb-1", "kb_kb-1"),
            ("orphan-kb-folder", "knowledge_base", "missing", "kb_missing"),
        ],
    )
    conn.executemany(
        "INSERT INTO linked_folder_files VALUES(?,?)",
        [
            ("project-folder", "project"),
            ("kb-folder", "kb"),
            ("orphan-kb-folder", "project"),
        ],
    )
    conn.executemany(
        "INSERT INTO linked_folder_sync_jobs VALUES(?,?)",
        [
            ("sync-project", "project-folder"),
            ("sync-kb", "kb-folder"),
            ("sync-orphan-kb", "orphan-kb-folder"),
        ],
    )
    conn.executemany(
        "INSERT INTO rag_job_leases VALUES('folder_sync', ?)",
        [("sync-project",), ("sync-kb",), ("sync-orphan-kb",)],
    )
    conn.commit()
    conn.close()

    def metadata_connection():
        connection = sqlite3.connect(db_path)
        connection.row_factory = sqlite3.Row
        return connection

    monkeypatch.setattr(
        history_cleanup.rag_db,
        "get_connection",
        lambda: (_ for _ in ()).throw(
            history_cleanup.rag_db.RagExtensionUnavailable("sqlite-vec missing")
        ),
    )
    monkeypatch.setattr(history_cleanup.rag_db, "get_metadata_connection", metadata_connection)
    monkeypatch.setattr(history_cleanup, "rag_uploads_root", lambda: uploads)

    assert history_cleanup.clear_non_knowledge_base_data() == 5

    with metadata_connection() as checked:
        assert {row["id"] for row in checked.execute("SELECT id FROM documents")} == {
            "kb",
            "legacy-kb",
            "shared-kb",
        }
        assert {row["id"] for row in checked.execute("SELECT id FROM linked_folders")} == {
            "kb-folder"
        }
        assert {
            row["scope"]
            for row in checked.execute("SELECT scope FROM linked_folder_retired_scopes")
        } == {
            "thread-secret",
            "project-secret",
            "thread-outside",
            "thread-shared",
            "research_scrape_active",
            "kb_missing",
        }
        assert {
            row["document_id"] for row in checked.execute("SELECT document_id FROM chunks")
        } == {"kb", "legacy-kb", "shared-kb"}
        assert {
            row["document_id"] for row in checked.execute("SELECT document_id FROM ingestion_jobs")
        } == {"kb", "legacy-kb", "shared-kb"}

    assert not paths["thread"].exists()
    assert not paths["project"].exists()
    assert paths["kb"].exists()
    assert paths["legacy-kb"].exists()
    assert paths["shared"].exists()
    assert outside.exists()


def test_managed_rag_paths_use_their_canonical_identity(monkeypatch, tmp_path):
    uploads = tmp_path / "uploads"
    alias = tmp_path / "uploads-alias"
    uploads.mkdir()
    alias.mkdir()
    candidate = alias / "shared.txt"
    candidate.write_text("shared", encoding = "utf-8")
    canonical = uploads / candidate.name
    realpath = os.path.realpath

    def resolve(path):
        path = os.fspath(path)
        if path == str(alias):
            return str(uploads)
        if path == str(candidate):
            return str(canonical)
        return realpath(path)

    monkeypatch.setattr(history_cleanup, "rag_uploads_root", lambda: uploads)
    monkeypatch.setattr(history_cleanup.os.path, "realpath", resolve)

    assert history_cleanup._managed_path(str(candidate)) == os.path.normcase(str(canonical))


def test_no_history_rag_visibility_accepts_legacy_kb_scope(no_chat_history):
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE knowledge_bases(id TEXT PRIMARY KEY)")
    conn.execute("INSERT INTO knowledge_bases VALUES('kb-1')")
    legacy = {"id": "legacy", "scope": "kb_kb-1", "kb_id": None}
    orphan = {"id": "orphan", "scope": "kb_missing", "kb_id": None}

    assert store.document_knowledge_base_id(conn, legacy) == "kb-1"
    rag._require_visible_document(conn, legacy)
    with pytest.raises(HTTPException) as exc_info:
        rag._require_visible_document(conn, orphan)
    assert exc_info.value.status_code == 404


def test_no_history_kb_listing_requires_an_existing_owner(no_chat_history, monkeypatch, tmp_path):
    db_path = tmp_path / "rag-list.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE knowledge_bases(id TEXT PRIMARY KEY)")
        conn.execute("INSERT INTO knowledge_bases VALUES('kb-1')")

    def connection():
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        return conn

    monkeypatch.setattr(rag, "_require_rag", lambda: None)
    monkeypatch.setattr(rag, "_rag_connection", connection)
    monkeypatch.setattr(
        rag.store,
        "list_documents",
        lambda _conn, scope: [
            {
                "id": "legacy",
                "scope": scope,
                "kb_id": None,
                "filename": "legacy.txt",
                "status": "completed",
            }
        ],
    )

    listed = rag.list_kb_documents("kb-1", subject = "alice")
    assert listed["documents"][0]["kbId"] == "kb-1"
    with pytest.raises(HTTPException) as missing:
        rag.list_kb_documents("missing", subject = "alice")
    assert missing.value.status_code == 404


def test_no_history_ownerless_kb_folders_cannot_be_seen_or_synced(
    no_chat_history, monkeypatch, tmp_path
):
    db_path = tmp_path / "rag-folders.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE knowledge_bases(id TEXT PRIMARY KEY)")
        conn.execute("INSERT INTO knowledge_bases VALUES('kb-1')")

    def connection():
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        return conn

    folders = {
        "orphan": {
            "id": "orphan",
            "scope_type": "knowledge_base",
            "scope_id": "missing",
            "scope": "kb_missing",
        },
        "legacy": {
            "id": "legacy",
            "scope_type": "knowledge_base",
            "scope_id": "stale",
            "scope": "kb_kb-1",
        },
    }
    enqueued = []
    monkeypatch.setattr(rag, "_require_rag", lambda: None)
    monkeypatch.setattr(rag, "_rag_connection", connection)
    monkeypatch.setattr(rag.folder_sync, "get_folder", folders.get)
    monkeypatch.setattr(
        rag.folder_sync,
        "request_sync",
        lambda folder_id, **_kwargs: enqueued.append(folder_id),
    )

    rag._require_visible_folder("legacy")
    with pytest.raises(HTTPException) as hidden:
        rag.sync_folder("orphan", subject = "alice")
    assert hidden.value.status_code == 404
    assert enqueued == []

    with pytest.raises(HTTPException) as missing_scope:
        rag.list_linked_folders(scope_type = "knowledge_base", scope_id = "missing", subject = "alice")
    assert missing_scope.value.status_code == 404


def test_policy_off_kb_listing_keeps_orphan_cleanup_contract(monkeypatch, tmp_path):
    db_path = tmp_path / "rag-list-policy-off.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE knowledge_bases(id TEXT PRIMARY KEY)")

    def connection():
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        return conn

    monkeypatch.setattr(chat_history_policy, "NO_CHAT_HISTORY", False)
    monkeypatch.setattr(rag, "_require_rag", lambda: None)
    monkeypatch.setattr(rag, "_rag_connection", connection)
    monkeypatch.setattr(rag.store, "list_documents", lambda _conn, _scope: [])

    assert rag.list_kb_documents("deleted-kb", subject = "alice") == {"documents": []}


def test_conversation_rag_search_is_rejected_before_storage(no_chat_history, monkeypatch):
    def unexpected():
        raise AssertionError("conversation RAG must not initialize storage")

    monkeypatch.setattr(rag, "_require_rag", unexpected)
    with pytest.raises(HTTPException) as exc_info:
        rag.search(
            rag.SearchRequest(query = "question", thread_id = "thread-1"),
            subject = "alice",
        )
    assert exc_info.value.status_code == 403


def test_rag_job_visibility_fails_closed_without_a_kb_document(
    no_chat_history, monkeypatch, tmp_path
):
    db_path = tmp_path / "rag-jobs.db"
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE knowledge_bases(id TEXT PRIMARY KEY);
            CREATE TABLE documents(id TEXT PRIMARY KEY, scope TEXT, kb_id TEXT);
            INSERT INTO knowledge_bases VALUES('kb-1');
            INSERT INTO documents VALUES('conversation', 'thread-1', NULL);
            INSERT INTO documents VALUES('explicit-kb', 'unusual', 'kb-1');
            INSERT INTO documents VALUES('legacy-kb', 'kb_kb-1', NULL);
            """
        )

    def connection():
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        return conn

    monkeypatch.setattr(rag, "_rag_connection", connection)
    with pytest.raises(HTTPException) as missing:
        rag._require_visible_document_id("missing")
    assert missing.value.status_code == 404

    with pytest.raises(HTTPException) as conversation:
        rag._require_visible_document_id("conversation")
    assert conversation.value.status_code == 404

    rag._require_visible_document_id("explicit-kb")
    rag._require_visible_document_id("legacy-kb")
