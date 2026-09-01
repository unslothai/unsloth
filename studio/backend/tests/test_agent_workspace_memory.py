# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import json
import threading
import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from auth.authentication import authenticated_via_api_key, get_current_subject
from core.agent_workspace.background import BackgroundTaskManager
from core.agent_workspace.common import AgentWorkspaceError
from core.agent_workspace.memory import (
    get_memory_entry,
    memory_context,
    run_dream_task,
    search_memory,
    write_memory_entry,
)
from core.agent_workspace.project_automation import install_project_skill, skill_digest
from core.agent_workspace.project_context import (
    resolve_project_context,
    strip_server_project_context,
)
from core.inference.tools import execute_tool
from storage import studio_db
from routes import agent_workspace as agent_workspace_routes


def _project(root, project_id = "project"):
    metadata = root.stat()
    return studio_db.upsert_chat_project(
        {
            "id": project_id,
            "name": "Memory project",
            "instructions": "",
            "rootPath": str(root),
            "workspaceKind": "folder",
            "workspaceDeviceId": str(metadata.st_dev),
            "workspaceFileId": str(metadata.st_ino),
            "archived": False,
            "createdAt": 1,
            "updatedAt": 1,
        }
    )


def _thread(thread_id, project_id = "project"):
    return studio_db.upsert_chat_thread(
        {
            "id": thread_id,
            "title": thread_id,
            "modelType": "base",
            "modelId": "test-model",
            "projectId": project_id,
            "archived": False,
            "createdAt": 1,
        }
    )


def _message(
    message_id,
    thread_id,
    content,
    *,
    role = "user",
    metadata = None,
):
    return studio_db.upsert_chat_message(
        {
            "id": message_id,
            "threadId": thread_id,
            "parentId": None,
            "role": role,
            "content": [{"type": "text", "text": content}],
            "metadata": metadata,
            "createdAt": int(message_id[-1]) + 1,
        }
    )


def _wait(
    manager,
    task_id,
    timeout = 5,
):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        from core.agent_workspace.state import get_background_task

        task = get_background_task(task_id)
        if task and task["status"] in {"completed", "failed", "cancelled"}:
            return task
        time.sleep(0.01)
    raise AssertionError("dream task did not stop")


def _client():
    app = FastAPI()
    app.include_router(agent_workspace_routes.router, prefix = "/api/agent-workspace")
    app.dependency_overrides[get_current_subject] = lambda: "test-subject"
    app.dependency_overrides[authenticated_via_api_key] = lambda: False
    return TestClient(app)


def test_memory_versions_hashes_permissions_and_context(tmp_path):
    _project(tmp_path)
    organization = write_memory_entry(
        "project",
        "organization/team.md",
        "Use the release checklist.\n",
        actor = "user",
    )
    project = write_memory_entry(
        "project",
        "project/preferences.md",
        "Prefer focused tests.\n",
        actor = "user",
    )
    with pytest.raises(AgentWorkspaceError, match = "organization memory"):
        write_memory_entry("project", "organization/team.md", "changed", actor = "agent")

    updated = write_memory_entry(
        "project",
        "project/preferences.md",
        "Prefer focused backend tests.\n",
        expected_hash = project["hash"],
        actor = "agent",
    )
    assert organization["version"] == 1
    assert updated["version"] == 2
    assert updated["hash"] != project["hash"]
    with pytest.raises(AgentWorkspaceError, match = "changed since"):
        write_memory_entry(
            "project",
            "project/preferences.md",
            "stale overwrite",
            expected_hash = project["hash"],
            actor = "agent",
        )
    assert len(search_memory("project", "focused", actor = "agent")) == 1
    rendered = memory_context("project", "release checklist")
    assert "team.md" in rendered
    assert "preferences.md" not in rendered
    assert "not instructions" in rendered
    assert get_memory_entry("project", "project/preferences.md")["version"] == 2


def test_project_context_and_memory_tools_are_project_scoped(tmp_path):
    _project(tmp_path)
    write_memory_entry("project", "project/runtime.md", "Use the local runtime.")
    context = resolve_project_context("project-project", query = "local runtime")
    assert context is not None
    assert '<unsloth_memory version="1">' in context.addition
    assert "runtime.md" in context.memory
    stripped = strip_server_project_context("caller\n" + context.addition)
    assert stripped == "caller"
    written = execute_tool(
        "memory_write",
        {"path": "agent/scratch.md", "content": "temporary note"},
        session_id = "project-project",
    )
    assert '"path": "agent/scratch.md"' in written
    denied = execute_tool(
        "memory_write",
        {"path": "organization/blocked.md", "content": "no"},
        session_id = "project-project",
    )
    assert "organization memory" in denied
    outside = execute_tool(
        "memory_search",
        {"query": "runtime"},
        session_id = "ordinary-chat",
    )
    assert "project session" in outside


def test_agents_cannot_read_or_modify_other_sessions_private_memory(tmp_path):
    _project(tmp_path)
    private = write_memory_entry(
        "project",
        "agent/scratch.md",
        "Only session one may read this.",
        actor = "agent",
        source_session_id = "project-project",
    )
    assert (
        get_memory_entry(
            "project", "agent/scratch.md", actor = "agent", session_id = "project-project"
        )["hash"]
        == private["hash"]
    )
    with pytest.raises(AgentWorkspaceError, match = "private"):
        get_memory_entry("project", "agent/scratch.md", actor = "agent", session_id = "project-other")
    assert search_memory("project", "session one", actor = "agent", session_id = "project-other") == []
    with pytest.raises(AgentWorkspaceError, match = "private"):
        write_memory_entry(
            "project",
            "agent/scratch.md",
            "Session two overwrite.",
            expected_hash = private["hash"],
            actor = "agent",
            source_session_id = "project-other",
        )


def test_memory_tools_bind_private_entries_to_the_conversation_not_project_sandbox(tmp_path):
    _project(tmp_path)
    created = json.loads(
        execute_tool(
            "memory_write",
            {"path": "agent/scratch.md", "content": "conversation one"},
            session_id = "project-project",
            thread_id = "thread-one",
        )
    )
    read = execute_tool(
        "memory_read",
        {"path": "agent/scratch.md"},
        session_id = "project-project",
        thread_id = "thread-one",
    )
    assert json.loads(read)["hash"] == created["hash"]
    denied = execute_tool(
        "memory_read",
        {"path": "agent/scratch.md"},
        session_id = "project-project",
        thread_id = "thread-two",
    )
    assert "private" in denied


def test_project_skill_guidance_is_read_only_on_demand(tmp_path):
    _project(tmp_path)
    guidance = "Read the repository map before changing multiple subsystems."
    skill = install_project_skill(
        "project",
        name = "Repository mapper",
        description = "Maps repository truth sources.",
        source = "project:skills/repository-mapper/SKILL.md",
        guidance = guidance,
        content_digest = skill_digest(guidance),
        enabled = True,
    )
    context = resolve_project_context("project-project")
    assert context is not None
    assert skill["id"] in context.addition
    assert guidance not in context.addition
    assert "project_skill_read" in context.addition
    rendered = execute_tool(
        "project_skill_read", {"skill_id": skill["id"]}, session_id = "project-project"
    )
    assert json.loads(rendered)["guidance"] == guidance
    disabled = execute_tool(
        "project_skill_read", {"skill_id": skill["id"]}, session_id = "ordinary-chat"
    )
    assert "project session" in disabled


def test_dream_returns_reviewable_proposals_without_mutating_memory(tmp_path):
    _project(tmp_path)
    _thread("thread-1")
    _thread("thread-2")
    _message("message-1", "thread-1", "I prefer focused tests.")
    _message("message-2", "thread-2", "I prefer focused tests.")
    result = run_dream_task(
        "project",
        {"threadIds": ["thread-1", "thread-2"], "instructions": "Keep it concise."},
        threading.Event(),
    )
    assert result["analyzerCount"] == 2
    assert result["subAgentCount"] == 0
    assert result["proposals"]
    proposal = result["proposals"][0]
    assert proposal["prevalence"] == {"transcripts": 2, "selected": 2, "ratio": 1.0}
    assert proposal["decision"] == "pending"
    with pytest.raises(AgentWorkspaceError, match = "not found"):
        get_memory_entry("project", proposal["path"])


def test_dream_uses_tool_failures_and_honors_focus_instructions(tmp_path):
    _project(tmp_path)
    _thread("thread-1")
    _thread("thread-2")
    failure = {"toolCalls": [{"name": "web_search", "status": "failed", "error": "Timeout"}]}
    _message("message-1", "thread-1", "I prefer focused tests.", metadata = failure)
    _message("message-2", "thread-2", "I prefer focused tests.", metadata = failure)
    result = run_dream_task(
        "project",
        {
            "threadIds": ["thread-1", "thread-2"],
            "instructions": "Focus only on web_search failures.",
        },
        threading.Event(),
    )
    assert result["steering"]["focus"] == ("web_search failures",)
    assert len(result["proposals"]) == 1
    assert "Tool web_search failed: Timeout" in result["proposals"][0]["content"]


def test_dream_uses_durable_background_lifecycle(tmp_path):
    _project(tmp_path)
    _thread("thread-1")
    _thread("thread-2")
    _message("message-1", "thread-1", "I prefer focused tests.")
    _message("message-2", "thread-2", "I prefer focused tests.")
    manager = BackgroundTaskManager(max_workers = 1)
    try:
        queued = manager.enqueue_dream("project", thread_ids = ["thread-1", "thread-2"], start = True)
        completed = _wait(manager, queued["id"])
    finally:
        manager._executor.shutdown(wait = True)
    assert completed["kind"] == "dream"
    assert completed["status"] == "completed"
    assert completed["result"]["proposals"]


def test_dream_routes_hold_proposals_until_user_acceptance(tmp_path):
    _project(tmp_path)
    _thread("thread-1")
    _thread("thread-2")
    _message("message-1", "thread-1", "I prefer focused tests.")
    _message("message-2", "thread-2", "I prefer focused tests.")
    client = _client()
    response = client.post(
        "/api/agent-workspace/projects/project/memory/dreams",
        json = {"threadIds": ["thread-1", "thread-2"], "start": True},
    )
    assert response.status_code == 202
    dream_id = response.json()["id"]
    deadline = time.monotonic() + 5
    dream = None
    while time.monotonic() < deadline:
        dream = client.get(f"/api/agent-workspace/projects/project/memory/dreams/{dream_id}").json()
        if dream["status"] == "completed":
            break
        time.sleep(0.01)
    assert dream is not None and dream["status"] == "completed"
    proposal = dream["result"]["proposals"][0]
    decision = client.post(
        f"/api/agent-workspace/projects/project/memory/dreams/{dream_id}/proposals/{proposal['id']}",
        json = {"decision": "accept"},
    )
    assert decision.status_code == 200
    assert decision.json()["proposal"]["decision"] == "accepted"
    entry = client.get(
        "/api/agent-workspace/projects/project/memory/entry",
        params = {"path": proposal["path"]},
    )
    assert entry.status_code == 200
    assert entry.json()["dreamId"] == dream_id


def test_dream_cleanup_deletion_requires_and_records_user_acceptance(tmp_path):
    _project(tmp_path)
    _thread("thread-1")
    _thread("thread-2")
    _message("message-1", "thread-1", "A routine status update.")
    _message("message-2", "thread-2", "Another routine status update.")
    stale = write_memory_entry(
        "project",
        "project/dreams/obsolete-observation.md",
        "# Dreamed observation\n\nObsolete.\n",
        actor = "user",
        dream_id = "previous-dream",
    )
    declined_cleanup = run_dream_task(
        "project",
        {
            "threadIds": ["thread-1", "thread-2"],
            "instructions": "Never delete stale memory.",
        },
        threading.Event(),
    )
    assert declined_cleanup["steering"]["staleCleanup"] is False
    assert declined_cleanup["proposals"] == []
    client = _client()
    response = client.post(
        "/api/agent-workspace/projects/project/memory/dreams",
        json = {
            "threadIds": ["thread-1", "thread-2"],
            "instructions": "Clean up stale memory.",
            "start": True,
        },
    )
    assert response.status_code == 202
    dream_id = response.json()["id"]
    deadline = time.monotonic() + 5
    dream = None
    while time.monotonic() < deadline:
        dream = client.get(f"/api/agent-workspace/projects/project/memory/dreams/{dream_id}").json()
        if dream["status"] == "completed":
            break
        time.sleep(0.01)
    assert dream is not None and dream["status"] == "completed"
    proposal = next(item for item in dream["result"]["proposals"] if item["path"] == stale["path"])
    assert proposal["operation"] == "delete"
    decision = client.post(
        f"/api/agent-workspace/projects/project/memory/dreams/{dream_id}/proposals/{proposal['id']}",
        json = {"decision": "accept", "expectedHash": stale["hash"]},
    )
    assert decision.status_code == 200
    assert decision.json()["proposal"]["deletedEntry"]["path"] == stale["path"]
    with pytest.raises(AgentWorkspaceError, match = "not found"):
        get_memory_entry("project", stale["path"])
