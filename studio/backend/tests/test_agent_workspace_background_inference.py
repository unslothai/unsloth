# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
import json
import subprocess
import threading
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from auth.authentication import authenticated_via_api_key, get_current_subject
from core.agent_workspace.background import BackgroundTaskManager
from core.agent_workspace.common import AgentWorkspaceError
from core.agent_workspace import inference_executor as background_inference
from core.agent_workspace.inference_executor import (
    capture_runtime_snapshot,
    execute_background_agent,
)
from core.agent_workspace.state import (
    get_background_task,
    get_worktree,
    transition_worktree_status,
)
from core.agent_workspace.worktrees import cleanup_worktree, create_worktree
from core.inference.tools import (
    background_task_session_id,
    execute_tool,
    resolve_sandbox_workdir,
)
from routes import agent_workspace as agent_workspace_routes
from storage import providers_db, studio_db


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd = root,
        check = True,
        stdout = subprocess.PIPE,
        stderr = subprocess.PIPE,
        text = True,
    ).stdout.strip()


def _repository(root: Path) -> None:
    _git(root, "init", "-q")
    _git(root, "config", "user.name", "Test")
    _git(root, "config", "user.email", "test@example.invalid")
    (root / "shared.txt").write_text("base\n", encoding = "utf-8")
    _git(root, "add", "shared.txt")
    _git(root, "commit", "-qm", "base")


def _folder_project(root: Path, project_id: str = "project") -> dict:
    metadata = root.stat()
    return studio_db.upsert_chat_project(
        {
            "id": project_id,
            "name": "Project",
            "instructions": "",
            "rootPath": str(root),
            "workspaceKind": "folder",
            "workspaceDeviceId": str(metadata.st_dev),
            "workspaceFileId": str(metadata.st_ino),
            "goal": "Ship the task",
            "goalStatus": "active",
            "goalUpdatedAt": 7,
            "archived": False,
            "createdAt": 1,
            "updatedAt": 1,
        }
    )


def _wait_task(task_id: str, timeout: float = 5) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        task = get_background_task(task_id)
        if task and task["status"] in {
            "cancelled",
            "completed",
            "failed",
            "interrupted",
        }:
            return task
        time.sleep(0.01)
    raise AssertionError("background task did not stop")


def test_background_relevance_isolates_sibling_instruction_scopes(tmp_path):
    workspace = tmp_path / "workspace"
    (workspace / "src").mkdir(parents = True)
    (workspace / "docs").mkdir()
    (workspace / "AGENTS.md").write_text("root background rule", encoding = "utf-8")
    (workspace / "src" / "AGENTS.md").write_text("src background rule", encoding = "utf-8")
    (workspace / "docs" / "AGENTS.md").write_text("docs background rule", encoding = "utf-8")
    (workspace / "src" / "worker.py").write_text("pass\n", encoding = "utf-8")
    (workspace / "docs" / "guide.md").write_text("guide\n", encoding = "utf-8")
    _folder_project(workspace)
    identity = (workspace.stat().st_dev, workspace.stat().st_ino)

    def context(instruction: str):
        return SimpleNamespace(
            project_id = "project",
            cwd = workspace,
            instruction = instruction,
            expected_root_identity = identity,
            goal_snapshot = "Ship the task",
            goal_status_snapshot = "active",
            plan_snapshot = None,
        )

    targeted = background_inference._agent_messages(context("Update src/worker.py"))[0]["content"]
    assert "root background rule" in targeted
    assert "src background rule" in targeted
    assert "docs background rule" not in targeted
    assert 'path value="src/worker.py"' in targeted
    assert 'path value="docs/guide.md"' not in targeted

    generic = background_inference._agent_messages(context("Tell me about this project"))[0][
        "content"
    ]
    assert "root background rule" in generic
    assert "src background rule" not in generic
    assert "docs background rule" not in generic
    assert "<unsloth_repository_selection" not in generic


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(agent_workspace_routes.router, prefix = "/api/agent-workspace")
    app.dependency_overrides[get_current_subject] = lambda: "test-subject"
    app.dependency_overrides[authenticated_via_api_key] = lambda: False
    return TestClient(app)


def test_runtime_snapshot_rejects_secrets_and_persists_only_saved_provider_identity(monkeypatch):
    config = {
        "id": "provider-1",
        "provider_type": "openai",
        "base_url": "https://api.openai.com/v1",
        "is_enabled": 1,
        "models": ["gpt-5.4"],
        "updated_at": "2026-08-25T12:00:00Z",
    }
    monkeypatch.setattr(providers_db, "get_provider", lambda _provider_id: dict(config))
    monkeypatch.setattr(
        background_inference.credential_secrets,
        "get_provider_api_key_binding",
        lambda _provider_id: "a" * 64,
    )

    snapshot = capture_runtime_snapshot(
        {
            "kind": "provider",
            "providerId": "provider-1",
            "model": "gpt-5.4",
            "permissionMode": "off",
            "reasoningEffort": "high",
            "maxOutputTokens": 4096,
        }
    )

    assert snapshot == {
        "kind": "provider",
        "model": "gpt-5.4",
        "providerId": "provider-1",
        "providerType": "openai",
        "permissionMode": "off",
        "reasoningEffort": "high",
        "maxOutputTokens": 4096,
        "routingDigest": snapshot["routingDigest"],
        "credentialBindingDigest": "a" * 64,
    }
    assert "apiKey" not in snapshot
    assert "baseUrl" not in snapshot
    with pytest.raises(AgentWorkspaceError, match = "selection is invalid"):
        capture_runtime_snapshot(
            {
                "kind": "provider",
                "providerId": "provider-1",
                "model": "gpt-5.4",
                "permissionMode": "off",
                "apiKey": "must-not-persist",
            }
        )


def test_provider_routing_and_codex_account_bindings_reject_drift(monkeypatch):
    config = {
        "id": "provider-1",
        "provider_type": "openai",
        "base_url": "https://api.openai.com/v1",
        "is_enabled": 1,
        "models": ["gpt-5.4"],
        "updated_at": "first",
    }
    monkeypatch.setattr(providers_db, "get_provider", lambda _provider_id: dict(config))
    snapshot = capture_runtime_snapshot(
        {
            "kind": "provider",
            "providerId": "provider-1",
            "model": "gpt-5.4",
            "permissionMode": "off",
        }
    )
    config["updated_at"] = "changed-after-enqueue"
    with pytest.raises(AgentWorkspaceError, match = "changed after this task was queued"):
        background_inference._current_provider(snapshot)

    from core.inference import openai_codex_auth

    codex_config = {
        "id": "codex-1",
        "provider_type": "openai_codex",
        "base_url": "https://chatgpt.com/backend-api",
        "is_enabled": 1,
        "models": ["gpt-5.4"],
        "updated_at": "first",
    }
    monkeypatch.setattr(providers_db, "get_provider", lambda _provider_id: dict(codex_config))
    monkeypatch.setattr(
        openai_codex_auth,
        "load_oauth_bundle",
        lambda _provider_id: {"account_id": "account-at-enqueue"},
    )
    codex_snapshot = capture_runtime_snapshot(
        {
            "kind": "provider",
            "providerId": "codex-1",
            "model": "gpt-5.4",
            "permissionMode": "off",
        }
    )

    async def changed_access(_provider_id, **_kwargs):
        return "runtime-token", "different-account"

    monkeypatch.setattr(openai_codex_auth, "resolve_access", changed_access)
    with pytest.raises(AgentWorkspaceError, match = "changed accounts"):
        asyncio.run(
            background_inference._run_codex(
                codex_snapshot,
                [{"role": "user", "content": "test"}],
                [],
                "agent-task-test",
                threading.Event(),
            )
        )


def test_external_provider_task_rejects_credential_replacement_before_egress(monkeypatch):
    config = {
        "id": "provider-1",
        "provider_type": "openai",
        "base_url": "https://api.openai.com/v1",
        "is_enabled": 1,
        "models": ["gpt-5.4"],
        "updated_at": "unchanged",
    }
    monkeypatch.setattr(providers_db, "get_provider", lambda _provider_id: dict(config))
    monkeypatch.setattr(
        background_inference.credential_secrets,
        "get_provider_api_key_binding",
        lambda _provider_id: "a" * 64,
    )
    snapshot = capture_runtime_snapshot(
        {
            "kind": "provider",
            "providerId": "provider-1",
            "model": "gpt-5.4",
            "permissionMode": "off",
        }
    )
    monkeypatch.setattr(
        background_inference.credential_secrets,
        "get_provider_api_key_with_binding",
        lambda _provider_id: ("replacement-secret", "b" * 64),
    )

    with pytest.raises(AgentWorkspaceError, match = "credential changed"):
        asyncio.run(
            background_inference._run_external(
                snapshot,
                [{"role": "user", "content": "test"}],
                [],
                "agent-task-test",
                threading.Event(),
            )
        )


@pytest.mark.parametrize("permission_mode", ["ask", "auto"])
def test_background_executor_rejects_interactive_permission_modes(permission_mode):
    snapshot = capture_runtime_snapshot(
        {
            "kind": "local",
            "model": "local-model",
            "permissionMode": permission_mode,
        }
    )
    with pytest.raises(AgentWorkspaceError, match = "cannot pause"):
        background_inference._validate_snapshot(snapshot)


def test_background_route_persists_credential_free_local_runtime(tmp_path, monkeypatch):
    _folder_project(tmp_path)
    monkeypatch.setattr(agent_workspace_routes, "_require_execution_boundary", lambda: None)
    agent_workspace_routes.background_manager.register_agent_executor(None)
    try:
        response = _client().post(
            "/api/agent-workspace/projects/project/background/agent",
            json = {
                "instruction": "Queue this",
                "runtime": {
                    "kind": "local",
                    "model": "local/model.gguf",
                    "permissionMode": "full",
                    "maxOutputTokens": 2048,
                },
                "start": False,
            },
        )
        rejected = _client().post(
            "/api/agent-workspace/projects/project/background/agent",
            json = {
                "instruction": "Do not queue this",
                "runtime": {
                    "kind": "local",
                    "model": "local/model.gguf",
                    "permissionMode": "off",
                    "apiKey": "must-not-enter-the-task-row",
                },
                "start": False,
            },
        )
        missing = _client().post(
            "/api/agent-workspace/projects/project/background/agent",
            json = {"instruction": "No runtime is not startable", "start": False},
        )
        interactive = _client().post(
            "/api/agent-workspace/projects/project/background/agent",
            json = {
                "instruction": "No background approval channel",
                "runtime": {
                    "kind": "local",
                    "model": "local/model.gguf",
                    "permissionMode": "auto",
                },
                "start": False,
            },
        )
    finally:
        agent_workspace_routes.background_manager.register_agent_executor(None)

    assert response.status_code == 200
    runtime = response.json()["payload"]["runtime"]
    assert runtime["model"] == "local/model.gguf"
    assert runtime["permissionMode"] == "full"
    assert runtime["providerId"] is None
    assert "apiKey" not in runtime
    assert rejected.status_code == 422
    assert missing.status_code == 422
    assert interactive.status_code == 422


def test_primary_task_session_is_immutable_and_revalidated_after_stop(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _folder_project(workspace)
    selection = {
        "kind": "local",
        "model": "queued-model",
        "permissionMode": "off",
    }
    observed = {}

    def executor(context, _cancel_event):
        session_id = background_task_session_id(context.task_id)
        observed["runtime"] = context.runtime_snapshot
        observed["cwd"] = Path(resolve_sandbox_workdir(session_id))
        observed["sessionId"] = session_id
        observed["insideEdit"] = execute_tool(
            "edit_file",
            {"path": "inside.txt", "old_string": "", "new_string": "inside\n"},
            session_id = session_id,
            disable_sandbox = True,
        )
        observed["outsideEdit"] = execute_tool(
            "edit_file",
            {"path": "../outside.txt", "old_string": "", "new_string": "outside\n"},
            session_id = session_id,
            disable_sandbox = True,
        )
        return {"output": "done"}

    manager = BackgroundTaskManager(max_workers = 1)
    manager.register_agent_executor(executor)
    try:
        queued = manager.enqueue_agent(
            "project",
            "Use the bound workspace",
            runtime_selection = selection,
            start = False,
        )
        selection["model"] = "mutated-after-enqueue"
        selection["permissionMode"] = "full"
        manager.start(queued["id"])
        finished = _wait_task(queued["id"])
    finally:
        manager._executor.shutdown(wait = True)

    assert finished["status"] == "completed"
    assert observed["runtime"]["model"] == "queued-model"
    assert observed["runtime"]["permissionMode"] == "off"
    assert observed["cwd"] == workspace.resolve()
    assert observed["insideEdit"].startswith("Created")
    assert observed["outsideEdit"].startswith("Error:")
    assert (workspace / "inside.txt").read_text(encoding = "utf-8") == "inside\n"
    assert not (workspace.parent / "outside.txt").exists()
    with pytest.raises(RuntimeError, match = "not active"):
        resolve_sandbox_workdir(observed["sessionId"])
    with pytest.raises(RuntimeError, match = "not active"):
        resolve_sandbox_workdir(background_task_session_id(str(uuid.uuid4())))


def test_task_session_isolated_to_owned_worktree_and_rejects_tampering(tmp_path, monkeypatch):
    repository = tmp_path / "repo"
    repository.mkdir()
    _repository(repository)
    _folder_project(repository)
    monkeypatch.setenv("UNSLOTH_STUDIO_PROJECTS_HOME", str(tmp_path / "studio-projects"))
    worktree = create_worktree("project")
    ready = threading.Event()
    release = threading.Event()
    observed = {}

    def executor(context, _cancel_event):
        session_id = background_task_session_id(context.task_id)
        observed["sessionId"] = session_id
        observed["cwd"] = Path(resolve_sandbox_workdir(session_id))
        ready.set()
        release.wait(timeout = 5)
        return {"output": "done"}

    manager = BackgroundTaskManager(max_workers = 1)
    manager.register_agent_executor(executor)
    task = None
    original_marker = None
    marker = Path(worktree["markerPath"])
    try:
        task = manager.enqueue_agent(
            "project",
            "Use only the owned worktree",
            runtime_selection = {
                "kind": "local",
                "model": "queued-model",
                "permissionMode": "off",
            },
            worktree_id = worktree["id"],
            start = True,
        )
        assert ready.wait(timeout = 5)
        assert observed["cwd"] == Path(worktree["path"])
        assert observed["cwd"] != repository.resolve()

        original_marker = marker.read_bytes()
        payload = json.loads(original_marker)
        payload["backgroundTaskId"] = str(uuid.uuid4())
        marker.write_text(json.dumps(payload), encoding = "utf-8")
        with pytest.raises(RuntimeError, match = "marker|task|unavailable|owned"):
            resolve_sandbox_workdir(observed["sessionId"])
        marker.write_bytes(original_marker)

        transition_worktree_status(worktree["id"], {"active"}, "removed")
        with pytest.raises(RuntimeError, match = "not active|unavailable|owned"):
            resolve_sandbox_workdir(observed["sessionId"])
        transition_worktree_status(worktree["id"], {"removed"}, "active")
        release.set()
        assert _wait_task(task["id"])["status"] == "completed"
    finally:
        if original_marker is not None and marker.exists():
            marker.write_bytes(original_marker)
        record = get_worktree(worktree["id"])
        if record is not None and record["status"] == "removed" and Path(record["path"]).exists():
            transition_worktree_status(worktree["id"], {"removed"}, "active")
        release.set()
        manager._executor.shutdown(wait = True)

    cleanup_worktree("project", worktree["id"])


def test_production_executor_fails_honestly_when_selected_local_model_is_unavailable(
    tmp_path, monkeypatch
):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _folder_project(workspace)
    from core.inference import orchestrator
    from core.inference import runtime_registry

    monkeypatch.setattr(runtime_registry, "peek_llama_cpp_backend", lambda: None)
    monkeypatch.setattr(orchestrator, "peek_inference_backend", lambda: None)
    manager = BackgroundTaskManager(max_workers = 1)
    manager.register_agent_executor(execute_background_agent)
    try:
        task = manager.enqueue_agent(
            "project",
            "Run with a missing model",
            runtime_selection = {
                "kind": "local",
                "model": "not-loaded-model",
                "permissionMode": "off",
            },
            start = True,
        )
        finished = _wait_task(task["id"])
    finally:
        manager._executor.shutdown(wait = True)

    assert finished["status"] == "failed"
    assert "selected local model is not loaded" in finished["error"].lower()
    assert finished["result"] is None


def test_production_executor_dispatches_loaded_llama_backend(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _folder_project(workspace)
    from core.inference import orchestrator
    from core.inference import runtime_registry

    observed = {}

    class FakeLlamaBackend:
        base_url = f"test-llama-{uuid.uuid4()}"
        effective_parallel_slots = 1
        hf_repo = "org/llama-model"
        is_loaded = True
        model_identifier = "org/llama-model"
        _openai_advertised_id = "org/llama-model"

        def generate_chat_completion_with_tools(self, **kwargs):
            observed.update(kwargs)
            yield {"type": "content", "text": "llama completed"}

    llama = FakeLlamaBackend()
    monkeypatch.setattr(runtime_registry, "peek_llama_cpp_backend", lambda: llama)
    monkeypatch.setattr(
        orchestrator,
        "peek_inference_backend",
        lambda: (_ for _ in ()).throw(AssertionError("fallback backend was used")),
    )
    manager = BackgroundTaskManager(max_workers = 1)
    manager.register_agent_executor(execute_background_agent)
    try:
        task = manager.enqueue_agent(
            "project",
            "Use the loaded GGUF runtime",
            runtime_selection = {
                "kind": "local",
                "model": "org/llama-model",
                "permissionMode": "full",
            },
            start = True,
        )
        finished = _wait_task(task["id"])
    finally:
        manager._executor.shutdown(wait = True)

    assert finished["status"] == "completed"
    assert finished["result"]["engine"] == "llama_cpp"
    assert finished["result"]["output"] == "llama completed"
    assert observed["session_id"] == background_task_session_id(task["id"])
    assert observed["cancel_event"] is not None
    assert observed["permission_mode"] == "full"
    assert observed["bypass_permissions"] is True


def test_production_executor_dispatches_external_provider_tool_loop(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _folder_project(workspace)
    from core.inference import external_provider
    from core.inference import external_tool_transport
    from core.inference import studio_tool_loop
    from storage import credential_secrets

    config = {
        "id": "provider-1",
        "provider_type": "openai",
        "base_url": "https://api.openai.com/v1",
        "is_enabled": 1,
        "models": ["gpt-5.4"],
        "updated_at": "unchanged",
    }
    observed = {}
    monkeypatch.setattr(providers_db, "get_provider", lambda _provider_id: dict(config))
    monkeypatch.setattr(
        credential_secrets,
        "get_provider_api_key_binding",
        lambda _provider_id: "a" * 64,
    )
    monkeypatch.setattr(
        credential_secrets,
        "get_provider_api_key_with_binding",
        lambda _provider_id: ("runtime-only-secret", "a" * 64),
    )

    class FakeExternalClient:
        def __init__(self, provider_type, base_url, api_key):
            observed["client"] = (provider_type, base_url, api_key)

        async def close(self):
            observed["closed"] = True

    class FakeTransport:
        def __init__(self, client, **kwargs):
            observed["transport"] = (client, kwargs)

    async def fake_stream(_transport, *, run, policy, cancel_event):
        observed["run"] = run
        observed["policy"] = policy
        observed["cancelEvent"] = cancel_event
        yield 'data: {"choices":[{"delta":{"content":"external completed"}}]}'
        yield "data: [DONE]"

    monkeypatch.setattr(external_provider, "ExternalProviderClient", FakeExternalClient)
    monkeypatch.setattr(external_tool_transport, "OAICompatTransport", FakeTransport)
    monkeypatch.setattr(studio_tool_loop, "stream_with_studio_tools", fake_stream)
    manager = BackgroundTaskManager(max_workers = 1)
    manager.register_agent_executor(execute_background_agent)
    try:
        task = manager.enqueue_agent(
            "project",
            "Use the saved provider",
            runtime_selection = {
                "kind": "provider",
                "providerId": "provider-1",
                "model": "gpt-5.4",
                "permissionMode": "off",
            },
            start = True,
        )
        finished = _wait_task(task["id"])
    finally:
        manager._executor.shutdown(wait = True)

    assert finished["status"] == "completed"
    assert finished["result"]["engine"] == "openai"
    assert finished["result"]["output"] == "external completed"
    assert observed["client"] == ("openai", "https://api.openai.com/v1", "runtime-only-secret")
    assert observed["run"].session_id == background_task_session_id(task["id"])
    assert observed["policy"].permission_mode == "off"
    assert observed["cancelEvent"] is not None
    assert observed["closed"] is True
    assert "runtime-only-secret" not in json.dumps(get_background_task(task["id"])["payload"])


@pytest.mark.parametrize("mutation", ["remove", "replace"])
def test_provider_task_revalidates_repository_identity_before_completion(
    tmp_path, monkeypatch, mutation
):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    original_identity = (workspace.stat().st_dev, workspace.stat().st_ino)
    _folder_project(workspace)
    from core.inference import external_provider
    from core.inference import external_tool_transport
    from core.inference import studio_tool_loop
    from storage import credential_secrets

    config = {
        "id": "provider-1",
        "provider_type": "openai",
        "base_url": "https://api.openai.com/v1",
        "is_enabled": 1,
        "models": ["gpt-5.4"],
        "updated_at": "unchanged",
    }
    entered_stream = threading.Event()
    release_stream = threading.Event()
    monkeypatch.setattr(providers_db, "get_provider", lambda _provider_id: dict(config))
    monkeypatch.setattr(
        credential_secrets,
        "get_provider_api_key_binding",
        lambda _provider_id: "a" * 64,
    )
    monkeypatch.setattr(
        credential_secrets,
        "get_provider_api_key_with_binding",
        lambda _provider_id: ("runtime-only-secret", "a" * 64),
    )

    class FakeExternalClient:
        def __init__(self, provider_type, base_url, api_key):
            del provider_type, base_url, api_key

        async def close(self):
            pass

    class FakeTransport:
        def __init__(self, client, **kwargs):
            del client, kwargs

    async def blocked_stream(_transport, *, run, policy, cancel_event):
        del run, policy, cancel_event
        entered_stream.set()
        released = await asyncio.to_thread(release_stream.wait, 5)
        if not released:
            raise RuntimeError("test stream barrier timed out")
        yield 'data: {"choices":[{"delta":{"content":"provider completed"}}]}'
        yield "data: [DONE]"

    monkeypatch.setattr(external_provider, "ExternalProviderClient", FakeExternalClient)
    monkeypatch.setattr(external_tool_transport, "OAICompatTransport", FakeTransport)
    monkeypatch.setattr(studio_tool_loop, "stream_with_studio_tools", blocked_stream)

    manager = BackgroundTaskManager(max_workers = 1)
    manager.register_agent_executor(execute_background_agent)
    displaced = tmp_path / "displaced-workspace"
    try:
        task = manager.enqueue_agent(
            "project",
            "Finish only in the bound repository",
            runtime_selection = {
                "kind": "provider",
                "providerId": "provider-1",
                "model": "gpt-5.4",
                "permissionMode": "off",
            },
            start = True,
        )
        assert entered_stream.wait(timeout = 5), json.dumps(
            get_background_task(task["id"]), sort_keys = True
        )
        if mutation == "remove":
            workspace.rmdir()
        else:
            workspace.rename(displaced)
            workspace.mkdir()
            assert (workspace.stat().st_dev, workspace.stat().st_ino) != original_identity
        release_stream.set()
        finished = _wait_task(task["id"])
    finally:
        release_stream.set()
        manager._executor.shutdown(wait = True)

    assert finished["status"] == "failed"
    assert finished["result"] is None
    assert "workspace" in finished["error"].lower()
    if mutation == "remove":
        assert not workspace.exists()
    else:
        assert workspace.is_dir()
        assert displaced.is_dir()
        assert (workspace.stat().st_dev, workspace.stat().st_ino) != original_identity


def test_production_executor_dispatches_codex_tool_loop(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _folder_project(workspace)
    from core.inference import openai_codex_auth
    from core.inference import openai_codex_client
    from core.inference import openai_codex_tool_loop

    config = {
        "id": "codex-1",
        "provider_type": "openai_codex",
        "base_url": "https://chatgpt.com/backend-api",
        "is_enabled": 1,
        "models": ["gpt-5.4"],
        "updated_at": "unchanged",
    }
    observed = {}
    monkeypatch.setattr(providers_db, "get_provider", lambda _provider_id: dict(config))
    monkeypatch.setattr(
        openai_codex_auth,
        "load_oauth_bundle",
        lambda _provider_id: {"account_id": "account-1"},
    )

    async def resolve_access(_provider_id, **_kwargs):
        return "runtime-only-token", "account-1"

    class FakeCodexClient:
        def __init__(self, access_token, account_id, refresh_access):
            observed["client"] = (access_token, account_id)
            observed["refresh"] = refresh_access

        async def close(self):
            observed["closed"] = True

    async def fake_stream(_client, *, run, policy, cancel_event):
        observed["run"] = run
        observed["policy"] = policy
        observed["cancelEvent"] = cancel_event
        yield 'data: {"choices":[{"delta":{"content":"codex completed"}}]}'
        yield "data: [DONE]"

    monkeypatch.setattr(openai_codex_auth, "resolve_access", resolve_access)
    monkeypatch.setattr(openai_codex_client, "OpenAICodexClient", FakeCodexClient)
    monkeypatch.setattr(openai_codex_tool_loop, "stream_codex_with_studio_tools", fake_stream)
    manager = BackgroundTaskManager(max_workers = 1)
    manager.register_agent_executor(execute_background_agent)
    try:
        task = manager.enqueue_agent(
            "project",
            "Use the connected subscription",
            runtime_selection = {
                "kind": "provider",
                "providerId": "codex-1",
                "model": "gpt-5.4",
                "permissionMode": "off",
            },
            start = True,
        )
        finished = _wait_task(task["id"])
    finally:
        manager._executor.shutdown(wait = True)

    assert finished["status"] == "completed"
    assert finished["result"]["engine"] == "openai_codex"
    assert finished["result"]["output"] == "codex completed"
    assert observed["client"] == ("runtime-only-token", "account-1")
    assert observed["run"].session_id == background_task_session_id(task["id"])
    assert observed["policy"].permission_mode == "off"
    assert observed["cancelEvent"] is not None
    assert observed["closed"] is True
    assert "runtime-only-token" not in json.dumps(get_background_task(task["id"])["payload"])
    assert "account-1" not in json.dumps(get_background_task(task["id"])["payload"])


def test_production_executor_propagates_cancel_and_bounds_output(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _folder_project(workspace)
    from core.inference import orchestrator
    from core.inference import runtime_registry

    entered = threading.Event()
    observed = {}

    class FakeBackend:
        active_model_name = "loaded-model"

        def generate_chat_completion_with_tools(self, **kwargs):
            observed["cancelEvent"] = kwargs["cancel_event"]

            def stream():
                entered.set()
                while not kwargs["cancel_event"].wait(timeout = 0.01):
                    yield {"type": "status", "text": "running"}

            return stream()

    monkeypatch.setattr(runtime_registry, "peek_llama_cpp_backend", lambda: None)
    monkeypatch.setattr(orchestrator, "peek_inference_backend", lambda: FakeBackend())
    manager = BackgroundTaskManager(max_workers = 1)
    manager.register_agent_executor(execute_background_agent)
    try:
        task = manager.enqueue_agent(
            "project",
            "Wait for cancellation",
            runtime_selection = {
                "kind": "local",
                "model": "loaded-model",
                "permissionMode": "off",
            },
            start = True,
        )
        assert entered.wait(timeout = 3)
        manager.cancel(task["id"])
        finished = _wait_task(task["id"])
    finally:
        manager._executor.shutdown(wait = True)

    assert finished["status"] == "cancelled"
    assert finished["cancelRequested"] is True
    assert observed["cancelEvent"].is_set()

    class LargeOutputBackend:
        active_model_name = "loaded-model"

        def generate_chat_completion_with_tools(self, **_kwargs):
            yield {"type": "content", "text": "x" * (1024 * 1024)}

    monkeypatch.setattr(orchestrator, "peek_inference_backend", lambda: LargeOutputBackend())
    manager = BackgroundTaskManager(max_workers = 1)
    manager.register_agent_executor(execute_background_agent)
    try:
        task = manager.enqueue_agent(
            "project",
            "Return a bounded result",
            runtime_selection = {
                "kind": "local",
                "model": "loaded-model",
                "permissionMode": "off",
            },
            start = True,
        )
        bounded = _wait_task(task["id"])
    finally:
        manager._executor.shutdown(wait = True)

    assert bounded["status"] == "completed"
    assert bounded["result"]["engine"] == "local"
    assert bounded["result"]["outputBytes"] == 1024 * 1024
    assert bounded["result"]["outputTruncated"] is True
    assert len(bounded["result"]["output"].encode("utf-8")) == 900 * 1024
