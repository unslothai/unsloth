# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import json
import sqlite3
import sys
import threading
import time
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from auth.authentication import authenticated_via_api_key, get_current_subject
from core.agent_workspace.common import AgentWorkspaceError
from core.agent_workspace.graphs import (
    GraphLoopAdapter,
    GraphRunManager,
    create_graph,
    create_graph_run,
    delete_graph,
    decide_graph_approval,
    get_graph_approval,
    get_graph_run,
    list_graph_events,
    list_graph_revisions,
    list_graph_runs,
    list_node_executions,
    recover_graph_runs,
    update_graph,
    validate_graph_spec,
)
from core.inference import mcp_client
from storage import mcp_servers_db, studio_db
from routes import agent_workspace as agent_workspace_routes


def _folder_project(root, project_id: str = "project") -> dict:
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
            "goal": "Graph goal",
            "goalStatus": "active",
            "goalUpdatedAt": 1,
            "archived": False,
            "createdAt": 1,
            "updatedAt": 1,
        }
    )


def _spec(
    *nodes,
    edges = None,
    name = "Graph",
):
    return {
        "name": name,
        "nodes": list(nodes),
        "edges": edges or [],
    }


def _node(
    node_id,
    node_type,
    config = None,
    retry_policy = None,
):
    normalized_config = dict(config or {})
    if node_type in {"loop", "model"}:
        normalized_config.setdefault(
            "runtime",
            {
                "kind": "local",
                "model": "test-model",
                "permissionMode": "off",
                "maxOutputTokens": 32,
            },
        )
    result = {"id": node_id, "type": node_type, "config": normalized_config}
    if retry_policy is not None:
        result["retryPolicy"] = retry_policy
    return result


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(agent_workspace_routes.router, prefix = "/api/agent-workspace")
    app.dependency_overrides[get_current_subject] = lambda: "test-subject"
    app.dependency_overrides[authenticated_via_api_key] = lambda: False
    return TestClient(app)


def _wait(run_id, timeout = 5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        run = get_graph_run("project", run_id)
        if run and run["status"] in {"paused", "cancelled", "completed", "failed", "interrupted"}:
            return run
        time.sleep(0.01)
    raise AssertionError("graph run did not stop")


def _wait_for_approval(run_id, timeout = 5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        approvals = [
            event
            for event in list_graph_events("project", run_id)
            if event["type"] == "approval.required"
        ]
        if approvals:
            return get_graph_approval(
                "project",
                run_id,
                approvals[-1]["payload"]["approvalId"],
            )
        time.sleep(0.01)
    raise AssertionError("graph approval did not appear")


class _EchoAdapter(GraphLoopAdapter):
    def __init__(self):
        self.instructions = []

    def run(self, project_id, instruction, runtime, cancel_event):
        self.instructions.append((project_id, instruction, runtime))
        return {"output": instruction}


def test_graph_validation_rejects_dangling_cycles_duplicates_and_unreachable():
    root = _node("input", "input")
    output = _node("output", "output")
    with pytest.raises(AgentWorkspaceError, match = "existing nodes"):
        validate_graph_spec(_spec(root, output, edges = [{"from": "input", "to": "missing"}]))
    with pytest.raises(AgentWorkspaceError, match = "root input"):
        validate_graph_spec(
            _spec(
                root,
                output,
                edges = [
                    {"from": "input", "to": "output"},
                    {"from": "output", "to": "input"},
                ],
            )
        )
    with pytest.raises(AgentWorkspaceError, match = "unique"):
        validate_graph_spec(_spec(root, root, output, edges = [{"from": "input", "to": "output"}]))
    with pytest.raises(AgentWorkspaceError, match = "unreachable"):
        validate_graph_spec(
            _spec(root, output, _node("dead", "output"), edges = [{"from": "input", "to": "output"}])
        )
    with pytest.raises(AgentWorkspaceError, match = "single condition edge"):
        validate_graph_spec(
            _spec(
                root,
                _node("check", "condition", {"path": "input.ok"}),
                output,
                edges = [
                    {"from": "input", "to": "check"},
                    {"from": "check", "to": "output", "when": "true"},
                ],
            )
        )
    with pytest.raises(AgentWorkspaceError, match = "Every graph path"):
        validate_graph_spec(
            _spec(
                root,
                _node("check", "condition", {"path": "input.ok"}),
                output,
                _node(
                    "dead-end",
                    "tool",
                    {
                        "serverId": "server",
                        "toolName": "read",
                        "arguments": {},
                        "sideEffecting": False,
                    },
                ),
                edges = [
                    {"from": "input", "to": "check"},
                    {"from": "check", "to": "output", "when": "true"},
                    {"from": "check", "to": "dead-end", "when": "false"},
                ],
            )
        )
    with pytest.raises(AgentWorkspaceError, match = "Output nodes must be terminal"):
        validate_graph_spec(
            _spec(
                root,
                output,
                _node("last", "output"),
                edges = [
                    {"from": "input", "to": "output"},
                    {"from": "output", "to": "last"},
                ],
            )
        )


@pytest.mark.parametrize(("node_type", "text_key"), [("loop", "instruction"), ("model", "prompt")])
def test_loop_and_model_nodes_require_a_pinned_runtime(node_type, text_key):
    with pytest.raises(AgentWorkspaceError, match = f"{node_type} runtime is required"):
        validate_graph_spec(
            _spec(
                _node("input", "input"),
                {"id": "work", "type": node_type, "config": {text_key: "run"}},
                _node("output", "output"),
                edges = [
                    {"from": "input", "to": "work"},
                    {"from": "work", "to": "output"},
                ],
            )
        )


def test_condition_node_selects_a_validated_branch(tmp_path):
    _folder_project(tmp_path)
    manager = GraphRunManager(max_workers = 1)
    try:
        graph = create_graph(
            "project",
            _spec(
                _node("input", "input"),
                _node("check", "condition", {"path": "input.ok"}),
                _node("yes", "output", {"path": "input"}),
                _node("no", "output", {"path": "input"}),
                edges = [
                    {"from": "input", "to": "check"},
                    {"from": "check", "to": "yes", "when": "true"},
                    {"from": "check", "to": "no", "when": "false"},
                ],
            ),
        )
        true_run = manager.enqueue("project", graph["id"], {"ok": True})
        false_run = manager.enqueue("project", graph["id"], {"ok": False})
        assert _wait(true_run["id"])["output"] == {"ok": True}
        assert _wait(false_run["id"])["output"] == {"ok": False}
        assert [item["nodeId"] for item in list_node_executions("project", true_run["id"])] == [
            "input",
            "check",
            "yes",
        ]
        assert [item["nodeId"] for item in list_node_executions("project", false_run["id"])] == [
            "input",
            "check",
            "no",
        ]
    finally:
        manager._executor.shutdown(wait = True)


@pytest.mark.parametrize(
    ("value", "schema_type"),
    [(False, "boolean"), (0, "integer"), ("", "string"), (None, "null")],
)
def test_graph_preserves_falsy_typed_output(tmp_path, value, schema_type):
    _folder_project(tmp_path)
    manager = GraphRunManager(max_workers = 1)
    try:
        graph = create_graph(
            "project",
            {
                **_spec(
                    _node("input", "input"),
                    _node("output", "output", {"path": "input.flag"}),
                    edges = [{"from": "input", "to": "output"}],
                ),
                "outputSchema": {"type": schema_type},
            },
        )
        run = manager.enqueue("project", graph["id"], {"flag": value})
        finished = _wait(run["id"])
        assert finished["status"] == "completed"
        assert finished["output"] == value
    finally:
        manager._executor.shutdown(wait = True)


def test_graph_enforces_revision_output_budget(tmp_path):
    _folder_project(tmp_path)
    manager = GraphRunManager(max_workers = 1)
    try:
        graph = create_graph(
            "project",
            {
                **_spec(
                    _node("input", "input"),
                    _node("output", "output", {"path": "input"}),
                    edges = [{"from": "input", "to": "output"}],
                ),
                "limits": {"maxNodes": 2, "maxRunSeconds": 60, "maxOutputBytes": 1024},
            },
        )
        run = manager.enqueue("project", graph["id"], {"value": "x" * 2000})
        finished = _wait(run["id"])
        assert finished["status"] == "failed"
        assert "too large" in (finished["error"] or "")
    finally:
        manager._executor.shutdown(wait = True)


def test_graph_delete_preserves_active_runs_until_stopped(tmp_path):
    _folder_project(tmp_path)
    graph = create_graph(
        "project",
        _spec(
            _node("input", "input"),
            _node("output", "output"),
            edges = [{"from": "input", "to": "output"}],
        ),
    )
    run = create_graph_run("project", graph["id"], {})
    with pytest.raises(AgentWorkspaceError, match = "active graph runs"):
        delete_graph("project", graph["id"])
    manager = GraphRunManager(max_workers = 1)
    try:
        manager.start(run["id"])
        assert _wait(run["id"])["status"] == "completed"
        delete_graph("project", graph["id"])
        assert get_graph_run("project", run["id"]) is None
    finally:
        manager._executor.shutdown(wait = True)


def test_graph_revisions_are_immutable_and_runs_pin_revision(tmp_path):
    _folder_project(tmp_path)
    graph = create_graph(
        "project",
        _spec(
            _node("input", "input"),
            _node("output", "output"),
            edges = [{"from": "input", "to": "output"}],
        ),
    )
    run = GraphRunManager(max_workers = 1).enqueue(
        "project", graph["id"], {"value": "before"}, start = False
    )
    updated = update_graph(
        "project",
        graph["id"],
        _spec(
            _node("input", "input"),
            _node("output", "output", {"path": "input.value"}),
            edges = [{"from": "input", "to": "output"}],
            name = "Graph v2",
        ),
        expected_revision = 1,
    )
    assert updated["currentRevision"] == 2
    assert run["revision"] == 1


def test_legacy_revision_gets_runtime_defaults_without_mutation(tmp_path):
    _folder_project(tmp_path)
    graph = create_graph(
        "project",
        _spec(
            _node("input", "input"),
            _node("output", "output"),
            edges = [{"from": "input", "to": "output"}],
        ),
    )
    from core.agent_workspace.state import connection

    conn = connection()
    row = conn.execute(
        "SELECT document_json FROM agent_graph_revisions WHERE graph_id = ? AND revision = 1",
        (graph["id"],),
    ).fetchone()
    stored = json.loads(row["document_json"])
    for node in stored["nodes"]:
        node.pop("retryPolicy", None)
    stored["limits"].pop("maxIterations", None)
    stored["limits"].pop("maxOutputTokens", None)
    conn.execute(
        "UPDATE agent_graph_revisions SET document_json = ? WHERE graph_id = ? AND revision = 1",
        (json.dumps(stored), graph["id"]),
    )
    conn.commit()
    conn.close()

    manager = GraphRunManager(max_workers = 1)
    try:
        run = manager.enqueue("project", graph["id"], {"value": "legacy"})
        finished = _wait(run["id"])
        assert finished["status"] == "completed"
        assert finished["iterationCount"] == 2
    finally:
        manager._executor.shutdown(wait = True)


def test_graph_run_enforces_revision_input_schema(tmp_path):
    _folder_project(tmp_path)
    graph = create_graph(
        "project",
        {
            **_spec(
                _node("input", "input"),
                _node("output", "output"),
                edges = [{"from": "input", "to": "output"}],
            ),
            "inputSchema": {
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
                "additionalProperties": False,
            },
        },
    )
    with pytest.raises(AgentWorkspaceError, match = "missing required"):
        create_graph_run("project", graph["id"], {})
    run = create_graph_run("project", graph["id"], {"value": "ok"})
    assert run["revision"] == 1


def test_graph_schema_rejects_unenforced_keywords_and_nonfinite_json():
    base = _spec(
        _node("input", "input"),
        _node("output", "output"),
        edges = [{"from": "input", "to": "output"}],
    )
    with pytest.raises(AgentWorkspaceError, match = "unsupported fields"):
        validate_graph_spec({**base, "inputSchema": {"type": "string", "minLength": 1}})
    with pytest.raises(AgentWorkspaceError, match = "additionalProperties"):
        validate_graph_spec(
            {
                **base,
                "inputSchema": {"type": "object", "additionalProperties": "false"},
            }
        )
    with pytest.raises(AgentWorkspaceError, match = "JSON serializable"):
        validate_graph_spec({**base, "metadata": {"score": float("nan")}})
    with pytest.raises(AgentWorkspaceError, match = "input schema must describe an object"):
        validate_graph_spec({**base, "inputSchema": {"type": "string"}})
    with pytest.raises(AgentWorkspaceError, match = "input schema must describe an object"):
        validate_graph_spec({**base, "inputSchema": {}})


def test_sequential_graph_uses_one_existing_loop_adapter_and_records_events(tmp_path):
    _folder_project(tmp_path)
    adapter = _EchoAdapter()
    manager = GraphRunManager(max_workers = 1, loop_adapter = adapter)
    try:
        graph = create_graph(
            "project",
            _spec(
                _node("input", "input"),
                _node("loop", "loop", {"instruction": "echo {input}"}),
                _node("output", "output"),
                edges = [
                    {"from": "input", "to": "loop"},
                    {"from": "loop", "to": "output"},
                ],
            ),
        )
        run = manager.enqueue("project", graph["id"], {"value": "x"})
        finished = _wait(run["id"])
        assert finished["status"] == "completed"
        assert finished["revision"] == 1
        assert finished["output"] == {"output": 'echo {"value": "x"}'}
        assert adapter.instructions == [
            (
                "project",
                'echo {"value": "x"}',
                {
                    "kind": "local",
                    "model": "test-model",
                    "permissionMode": "off",
                    "maxOutputTokens": 32,
                },
            )
        ]
        executions = list_node_executions("project", run["id"])
        assert [item["status"] for item in executions] == ["completed"] * 3
        events = list_graph_events("project", run["id"])
        assert [event["type"] for event in events].count("node.completed") == 3
        assert events[-1]["type"] == "run.completed"
    finally:
        manager._executor.shutdown(wait = True)


def test_graph_pause_resume_retries_current_node(tmp_path):
    _folder_project(tmp_path)
    entered = threading.Event()

    class _PausingAdapter(GraphLoopAdapter):
        def __init__(self):
            self.calls = 0

        def run(self, project_id, instruction, runtime, cancel_event):
            self.calls += 1
            entered.set()
            if self.calls == 1:
                cancel_event.wait(timeout = 2)
            return {"output": f"call-{self.calls}"}

    adapter = _PausingAdapter()
    manager = GraphRunManager(max_workers = 1, loop_adapter = adapter)
    try:
        graph = create_graph(
            "project",
            _spec(
                _node("input", "input"),
                _node("loop", "loop", {"instruction": "run"}),
                _node("output", "output"),
                edges = [{"from": "input", "to": "loop"}, {"from": "loop", "to": "output"}],
            ),
        )
        run = manager.enqueue("project", graph["id"], {})
        assert entered.wait(timeout = 2)
        paused = manager.pause(run["id"])
        assert paused["status"] in {"pausing", "paused"}
        stopped = _wait(run["id"])
        assert stopped["status"] == "paused"
        resumed = manager.resume(run["id"])
        assert resumed["status"] == "running"
        assert _wait(run["id"])["status"] == "completed"
        assert adapter.calls == 2
    finally:
        manager._executor.shutdown(wait = True)


def test_graph_approval_blocks_until_decision(tmp_path):
    _folder_project(tmp_path)
    manager = GraphRunManager(max_workers = 1)
    try:
        graph = create_graph(
            "project",
            _spec(
                _node("input", "input"),
                _node("approval", "approval", {"title": "Ship it"}),
                _node("output", "output"),
                edges = [
                    {"from": "input", "to": "approval"},
                    {"from": "approval", "to": "output"},
                ],
            ),
        )
        run = manager.enqueue("project", graph["id"], {})
        deadline = time.monotonic() + 3
        approval = None
        while time.monotonic() < deadline:
            approval_events = [
                event
                for event in list_graph_events("project", run["id"])
                if event["type"] == "approval.required"
            ]
            if approval_events:
                approval = get_graph_approval(
                    "project", run["id"], approval_events[0]["payload"]["approvalId"]
                )
                break
            time.sleep(0.01)
        assert approval and approval["status"] == "pending"
        decided = decide_graph_approval("project", run["id"], approval["id"], "approved")
        assert decided["status"] == "approved"
        assert _wait(run["id"])["status"] == "completed"
        assert any(
            event["type"] == "approval.decided" for event in list_graph_events("project", run["id"])
        )
    finally:
        manager._executor.shutdown(wait = True)


def test_terminal_run_rejects_a_late_approval_decision(tmp_path):
    _folder_project(tmp_path)
    graph = create_graph(
        "project",
        _spec(
            _node("input", "input"),
            _node("approval", "approval", {"title": "Continue"}),
            _node("output", "output"),
            edges = [
                {"from": "input", "to": "approval"},
                {"from": "approval", "to": "output"},
            ],
        ),
    )
    manager = GraphRunManager(max_workers = 1)
    try:
        run = manager.enqueue("project", graph["id"], {})
        approval = _wait_for_approval(run["id"])
        manager.cancel(run["id"])
        assert _wait(run["id"])["status"] == "cancelled"
        with pytest.raises(AgentWorkspaceError, match = "no longer awaiting approval"):
            decide_graph_approval("project", run["id"], approval["id"], "approved")
        assert get_graph_approval("project", run["id"], approval["id"])["status"] == "pending"
    finally:
        manager._executor.shutdown(wait = True)


def test_approval_releases_the_runtime_snapshot_captured_at_run_creation(tmp_path, monkeypatch):
    _folder_project(tmp_path)
    current_credential = ["a" * 64]

    def capture_runtime(selection):
        return {
            "kind": "provider",
            "model": selection["model"],
            "providerId": selection["providerId"],
            "providerType": "openai",
            "permissionMode": selection["permissionMode"],
            "reasoningEffort": None,
            "maxOutputTokens": selection.get("maxOutputTokens", 32),
            "routingDigest": "c" * 64,
            "credentialBindingDigest": current_credential[0],
        }

    monkeypatch.setattr(
        "core.agent_workspace.inference_executor.capture_runtime_snapshot",
        capture_runtime,
    )
    observed = []

    class _SnapshotAdapter(GraphLoopAdapter):
        def run(
            self,
            project_id,
            instruction,
            runtime,
            cancel_event,
            *,
            runtime_snapshot = None,
            checkpoint = None,
            checkpoint_callback = None,
            before_start = None,
        ):
            observed.append(runtime_snapshot)
            before_start("provider-binding-test")
            return {"output": "done"}

    graph = create_graph(
        "project",
        _spec(
            _node("input", "input"),
            _node("approval", "approval", {"title": "Use provider"}),
            _node(
                "loop",
                "loop",
                {
                    "instruction": "run",
                    "runtime": {
                        "kind": "provider",
                        "model": "provider-model",
                        "providerId": "provider",
                        "permissionMode": "off",
                        "maxOutputTokens": 32,
                    },
                },
            ),
            _node("output", "output"),
            edges = [
                {"from": "input", "to": "approval"},
                {"from": "approval", "to": "loop"},
                {"from": "loop", "to": "output"},
            ],
        ),
    )
    manager = GraphRunManager(max_workers = 1, loop_adapter = _SnapshotAdapter())
    try:
        run = manager.enqueue("project", graph["id"], {})
        approval = _wait_for_approval(run["id"])
        current_credential[0] = "b" * 64
        decide_graph_approval("project", run["id"], approval["id"], "approved")
        assert _wait(run["id"])["status"] == "completed"
        assert observed[0]["credentialBindingDigest"] == "a" * 64
    finally:
        manager._executor.shutdown(wait = True)


def test_api_key_cannot_release_a_saved_tool_after_approval(tmp_path, monkeypatch):
    _folder_project(tmp_path)
    monkeypatch.setattr(
        "core.agent_workspace.graphs.mcp_servers_db.get_server",
        lambda server_id: {
            "id": server_id,
            "url": "http://example.test/mcp",
            "is_enabled": True,
            "headers_json": None,
            "use_oauth": False,
        },
    )
    graph = create_graph(
        "project",
        {
            **_spec(
                _node("input", "input"),
                _node("approval", "approval", {"title": "Allow saved tool"}),
                _node(
                    "tool",
                    "tool",
                    {
                        "serverId": "saved-server",
                        "toolName": "write",
                        "arguments": {},
                        "sideEffecting": True,
                        "idempotencyKey": "approval-boundary",
                    },
                ),
                _node("output", "output"),
                edges = [
                    {"from": "input", "to": "approval"},
                    {"from": "approval", "to": "tool"},
                    {"from": "tool", "to": "output"},
                ],
            ),
            "permissions": {"allowedToolServerIds": ["saved-server"]},
        },
    )
    run = create_graph_run("project", graph["id"], {})
    from core.agent_workspace.graphs import get_or_create_approval

    approval = get_or_create_approval(
        "project",
        run["id"],
        _node(
            "approval",
            "approval",
            {"title": "Allow saved tool", "description": ""},
        ),
    )
    app = FastAPI()
    app.include_router(agent_workspace_routes.router, prefix = "/api/agent-workspace")
    app.dependency_overrides[get_current_subject] = lambda: "api-key-subject"
    app.dependency_overrides[authenticated_via_api_key] = lambda: True
    client = TestClient(app)

    response = client.post(
        f"/api/agent-workspace/projects/project/graph-runs/{run['id']}/approvals/{approval['id']}",
        json = {"decision": "approved"},
    )

    assert response.status_code == 403
    assert response.json()["detail"] == "Remote access requires a UI session."
    assert get_graph_approval("project", run["id"], approval["id"])["status"] == "pending"


def test_approval_decision_and_event_commit_together(tmp_path, monkeypatch):
    _folder_project(tmp_path)
    graph = create_graph(
        "project",
        _spec(
            _node("input", "input"),
            _node("approval", "approval", {"title": "Atomic approval"}),
            _node("output", "output"),
            edges = [
                {"from": "input", "to": "approval"},
                {"from": "approval", "to": "output"},
            ],
        ),
    )
    run = create_graph_run("project", graph["id"], {})
    from core.agent_workspace import graphs as graph_module

    approval = graph_module.get_or_create_approval(
        "project",
        run["id"],
        _node(
            "approval",
            "approval",
            {"title": "Atomic approval", "description": ""},
        ),
    )
    real_insert = graph_module._insert_graph_event

    def reject_decision_event(conn, run_id, event_type, **kwargs):
        if event_type == "approval.decided":
            raise sqlite3.OperationalError("approval event rejected")
        return real_insert(conn, run_id, event_type, **kwargs)

    monkeypatch.setattr(graph_module, "_insert_graph_event", reject_decision_event)
    with pytest.raises(sqlite3.OperationalError, match = "approval event rejected"):
        decide_graph_approval("project", run["id"], approval["id"], "approved")

    assert get_graph_approval("project", run["id"], approval["id"])["status"] == "pending"
    assert not any(
        event["type"] == "approval.decided" for event in list_graph_events("project", run["id"])
    )


def test_graph_recovery_marks_active_runs_interrupted(tmp_path):
    _folder_project(tmp_path)
    graph = create_graph(
        "project",
        _spec(
            _node("input", "input"),
            _node("output", "output"),
            edges = [{"from": "input", "to": "output"}],
        ),
    )
    manager = GraphRunManager(max_workers = 1)
    run = manager.enqueue("project", graph["id"], {}, start = False)
    from core.agent_workspace.graphs import claim_graph_run

    claim_graph_run(run["id"])
    manager._executor.shutdown(wait = True)
    assert recover_graph_runs() == 1
    assert get_graph_run("project", run["id"])["status"] == "interrupted"
    assert list_graph_events("project", run["id"])[-1]["type"] == "run.interrupted"


def test_queued_pause_and_cancel_have_durable_lifecycle_events(tmp_path):
    _folder_project(tmp_path)
    graph = create_graph(
        "project",
        _spec(
            _node("input", "input"),
            _node("output", "output"),
            edges = [{"from": "input", "to": "output"}],
        ),
    )
    manager = GraphRunManager(max_workers = 1)
    try:
        run = manager.enqueue("project", graph["id"], {}, start = False)
        assert manager.pause(run["id"])["status"] == "paused"
        cancelled = manager.cancel(run["id"])
        assert cancelled["status"] == "cancelled"
        assert cancelled["pauseRequested"] is False
        assert cancelled["cancelRequested"] is True
        assert [event["type"] for event in list_graph_events("project", run["id"])] == [
            "run.created",
            "run.paused",
            "run.cancelled",
        ]
    finally:
        manager._executor.shutdown(wait = True)


def test_cancelling_a_paused_run_marks_its_paused_node_cancelled(tmp_path):
    _folder_project(tmp_path)
    entered = threading.Event()

    class _PauseAdapter(GraphLoopAdapter):
        def run(self, project_id, instruction, runtime, cancel_event):
            entered.set()
            cancel_event.wait(timeout = 2)
            return {"output": "paused"}

    graph = create_graph(
        "project",
        _spec(
            _node("input", "input"),
            _node("loop", "loop", {"instruction": "pause"}),
            _node("output", "output"),
            edges = [
                {"from": "input", "to": "loop"},
                {"from": "loop", "to": "output"},
            ],
        ),
    )
    manager = GraphRunManager(max_workers = 1, loop_adapter = _PauseAdapter())
    try:
        run = manager.enqueue("project", graph["id"], {})
        assert entered.wait(timeout = 2)
        manager.pause(run["id"])
        assert _wait(run["id"])["status"] == "paused"

        assert manager.cancel(run["id"])["status"] == "cancelled"
        loop_execution = next(
            execution
            for execution in list_node_executions("project", run["id"])
            if execution["nodeId"] == "loop"
        )
        assert loop_execution["status"] == "cancelled"
        assert loop_execution["error"] == "Graph run cancelled."
    finally:
        manager._executor.shutdown(wait = True)


def test_hard_run_control_reasons_cannot_be_downgraded_to_pause():
    from core.agent_workspace.graphs import _RunControl
    for hard_reason in ("cancel", "budget", "shutdown"):
        control = _RunControl()
        control.request(hard_reason)
        control.request("pause")
        assert control.should_cancel_work() is True


def test_graph_app_exit_interrupts_active_node(tmp_path):
    _folder_project(tmp_path)
    entered = threading.Event()

    class _BlockingAdapter(GraphLoopAdapter):
        def run(self, project_id, instruction, runtime, cancel_event):
            entered.set()
            assert cancel_event.wait(timeout = 2)
            raise AgentWorkspaceError("Studio is stopping.")

    manager = GraphRunManager(max_workers = 1, loop_adapter = _BlockingAdapter())
    try:
        graph = create_graph(
            "project",
            _spec(
                _node("input", "input"),
                _node("loop", "loop", {"instruction": "run"}),
                _node("output", "output"),
                edges = [
                    {"from": "input", "to": "loop"},
                    {"from": "loop", "to": "output"},
                ],
            ),
        )
        run = manager.enqueue("project", graph["id"], {})
        assert entered.wait(timeout = 2)

        manager.prepare_for_app_exit(timeout_seconds = 2)

        stopped = get_graph_run("project", run["id"])
        assert stopped["status"] == "interrupted"
        assert stopped["currentNodeId"] == "loop"
        loop_execution = [
            item for item in list_node_executions("project", run["id"]) if item["nodeId"] == "loop"
        ][0]
        assert loop_execution["status"] == "interrupted"
        assert any(
            event["type"] == "run.interrupted" for event in list_graph_events("project", run["id"])
        )
    finally:
        manager._executor.shutdown(wait = True)


def test_graph_api_is_project_scoped_and_pins_revision(tmp_path):
    _folder_project(tmp_path)
    client = _client()
    payload = _spec(
        _node("input", "input"),
        _node("output", "output"),
        edges = [{"from": "input", "to": "output"}],
    )
    response = client.post("/api/agent-workspace/projects/project/graphs", json = payload)
    assert response.status_code == 200
    graph = response.json()
    assert graph["currentRevision"] == 1

    run_response = client.post(
        f"/api/agent-workspace/projects/project/graphs/{graph['id']}/runs",
        json = {"input": {"value": "ok"}, "idempotencyKey": "request-1"},
    )
    assert run_response.status_code == 200
    run = run_response.json()
    assert run["revision"] == 1
    repeated = client.post(
        f"/api/agent-workspace/projects/project/graphs/{graph['id']}/runs",
        json = {"input": {"value": "ok"}, "idempotencyKey": "request-1"},
    )
    assert repeated.status_code == 200
    assert repeated.json()["id"] == run["id"]
    conflicting = client.post(
        f"/api/agent-workspace/projects/project/graphs/{graph['id']}/runs",
        json = {"input": {"value": "different"}, "idempotencyKey": "request-1"},
    )
    assert conflicting.status_code == 409
    assert "different input or revision" in conflicting.json()["detail"]
    second_graph = create_graph("project", {**payload, "name": "Graph 2"})
    second_run = create_graph_run(
        "project",
        second_graph["id"],
        {"value": "other"},
        idempotency_key = "request-1",
    )
    assert second_run["id"] != run["id"]
    assert (
        client.get(f"/api/agent-workspace/projects/other/graphs/{graph['id']}").status_code == 404
    )
    assert _wait(run["id"])["status"] == "completed"
    assert client.delete(f"/api/agent-workspace/projects/project/graphs/{graph['id']}").json() == {
        "deleted": True
    }
    assert (
        client.get(f"/api/agent-workspace/projects/project/graphs/{graph['id']}").status_code == 404
    )


def test_graph_idempotency_key_rejects_a_different_revision(tmp_path):
    _folder_project(tmp_path)
    document = _spec(
        _node("input", "input"),
        _node("output", "output"),
        edges = [{"from": "input", "to": "output"}],
    )
    graph = create_graph("project", document)
    first = create_graph_run(
        "project",
        graph["id"],
        {"value": "same"},
        idempotency_key = "stable-request",
    )
    update_graph(
        "project",
        graph["id"],
        {**document, "description": "revision two"},
        expected_revision = 1,
    )

    with pytest.raises(AgentWorkspaceError, match = "different input or revision"):
        create_graph_run(
            "project",
            graph["id"],
            {"value": "same"},
            idempotency_key = "stable-request",
        )
    assert get_graph_run("project", first["id"])["revision"] == 1


def test_graph_idempotency_key_canonicalizes_object_key_order(tmp_path):
    _folder_project(tmp_path)
    graph = create_graph(
        "project",
        _spec(
            _node("input", "input"),
            _node("output", "output"),
            edges = [{"from": "input", "to": "output"}],
        ),
    )
    first = create_graph_run(
        "project",
        graph["id"],
        {"outer": {"a": 1, "b": 2}, "value": 3},
        idempotency_key = "canonical-request",
    )
    repeated = create_graph_run(
        "project",
        graph["id"],
        {"value": 3, "outer": {"b": 2, "a": 1}},
        idempotency_key = "canonical-request",
    )

    assert repeated["id"] == first["id"]


def test_retry_policy_is_safe_and_persisted_per_attempt(tmp_path):
    _folder_project(tmp_path)

    class _FlakyAdapter(GraphLoopAdapter):
        def __init__(self):
            self.calls = 0

        def run(self, project_id, instruction, runtime, cancel_event):
            self.calls += 1
            if self.calls == 1:
                raise AgentWorkspaceError("temporary failure")
            return {"output": instruction}

    runtime = {
        "kind": "local",
        "model": "test-model",
        "permissionMode": "off",
        "maxOutputTokens": 10,
    }
    with pytest.raises(AgentWorkspaceError, match = "permissionMode"):
        validate_graph_spec(
            _spec(
                _node("input", "input"),
                _node(
                    "loop",
                    "loop",
                    {
                        "instruction": "run",
                        "runtime": {
                            "kind": "local",
                            "model": "test-model",
                            "permissionMode": "full",
                        },
                    },
                    {"maxAttempts": 2},
                ),
                _node("output", "output"),
                edges = [
                    {"from": "input", "to": "loop"},
                    {"from": "loop", "to": "output"},
                ],
            )
        )
    adapter = _FlakyAdapter()
    manager = GraphRunManager(max_workers = 1, loop_adapter = adapter)
    try:
        graph = create_graph(
            "project",
            {
                **_spec(
                    _node("input", "input"),
                    _node(
                        "loop",
                        "loop",
                        {"instruction": "run", "runtime": runtime},
                        {"maxAttempts": 2, "backoffMs": 1, "retryOn": ["error"]},
                    ),
                    _node("output", "output"),
                    edges = [
                        {"from": "input", "to": "loop"},
                        {"from": "loop", "to": "output"},
                    ],
                ),
                "limits": {
                    "maxNodes": 3,
                    "maxRunSeconds": 60,
                    "maxOutputBytes": 1024,
                    "maxIterations": 4,
                    "maxOutputTokens": 20,
                },
            },
        )
        run = manager.enqueue("project", graph["id"], {})
        finished = _wait(run["id"])
        assert finished["status"] == "completed"
        assert finished["iterationCount"] == 4
        assert finished["reservedOutputTokens"] == 20
        loop_attempts = [
            item for item in list_node_executions("project", run["id"]) if item["nodeId"] == "loop"
        ]
        assert [(item["attempt"], item["status"]) for item in loop_attempts] == [
            (1, "failed"),
            (2, "completed"),
        ]
        assert any(
            event["type"] == "node.retrying" for event in list_graph_events("project", run["id"])
        )
    finally:
        manager._executor.shutdown(wait = True)


def test_iteration_and_output_token_budgets_fail_closed(tmp_path):
    _folder_project(tmp_path)

    class _AlwaysFailsOnce(GraphLoopAdapter):
        def __init__(self):
            self.calls = 0

        def run(self, project_id, instruction, runtime, cancel_event):
            self.calls += 1
            if self.calls == 1:
                raise AgentWorkspaceError("retry me")
            return {"output": "ok"}

    runtime = {
        "kind": "local",
        "model": "test-model",
        "permissionMode": "off",
        "maxOutputTokens": 10,
    }
    adapter = _AlwaysFailsOnce()
    manager = GraphRunManager(max_workers = 1, loop_adapter = adapter)
    try:
        graph = create_graph(
            "project",
            {
                **_spec(
                    _node("input", "input"),
                    _node(
                        "loop",
                        "loop",
                        {"instruction": "run", "runtime": runtime},
                        {"maxAttempts": 2},
                    ),
                    _node("output", "output"),
                    edges = [
                        {"from": "input", "to": "loop"},
                        {"from": "loop", "to": "output"},
                    ],
                ),
                "limits": {
                    "maxNodes": 3,
                    "maxRunSeconds": 60,
                    "maxOutputBytes": 1024,
                    "maxIterations": 3,
                    "maxOutputTokens": 20,
                },
            },
        )
        iteration_run = manager.enqueue("project", graph["id"], {})
        iteration_finished = _wait(iteration_run["id"])
        assert iteration_finished["status"] == "failed"
        assert "iteration budget" in (iteration_finished["error"] or "")

        token_graph = create_graph(
            "project",
            {
                **_spec(
                    _node("input", "input"),
                    _node("loop", "loop", {"instruction": "run", "runtime": runtime}),
                    _node("output", "output"),
                    edges = [
                        {"from": "input", "to": "loop"},
                        {"from": "loop", "to": "output"},
                    ],
                    name = "Token graph",
                ),
                "limits": {
                    "maxNodes": 3,
                    "maxRunSeconds": 60,
                    "maxOutputBytes": 1024,
                    "maxIterations": 3,
                    "maxOutputTokens": 9,
                },
            },
        )
        token_run = manager.enqueue("project", token_graph["id"], {})
        token_finished = _wait(token_run["id"])
        assert token_finished["status"] == "failed"
        assert "output token budget" in (token_finished["error"] or "")
    finally:
        manager._executor.shutdown(wait = True)


def test_loop_checkpoint_reuses_completed_work_after_pause(tmp_path):
    _folder_project(tmp_path)
    entered = threading.Event()

    class _CheckpointAdapter(GraphLoopAdapter):
        def __init__(self):
            self.calls = 0
            self.reuses = 0

        def run(
            self,
            project_id,
            instruction,
            runtime,
            cancel_event,
            *,
            checkpoint = None,
            checkpoint_callback = None,
            before_start = None,
        ):
            if checkpoint and checkpoint.get("status") == "completed":
                self.reuses += 1
                return {"output": "checkpointed"}
            if before_start:
                before_start("durable-task")
            self.calls += 1
            if checkpoint_callback:
                checkpoint_callback(
                    {
                        "backgroundTaskId": "durable-task",
                        "status": "completed",
                        "toolIterations": 2,
                    }
                )
            entered.set()
            cancel_event.wait(timeout = 2)
            return {"output": "checkpointed"}

    adapter = _CheckpointAdapter()
    manager = GraphRunManager(max_workers = 1, loop_adapter = adapter)
    try:
        graph = create_graph(
            "project",
            _spec(
                _node("input", "input"),
                _node("loop", "loop", {"instruction": "run"}),
                _node("output", "output"),
                edges = [
                    {"from": "input", "to": "loop"},
                    {"from": "loop", "to": "output"},
                ],
            ),
        )
        run = manager.enqueue("project", graph["id"], {})
        assert entered.wait(timeout = 2)
        manager.pause(run["id"])
        assert _wait(run["id"])["status"] == "paused"
        checkpoint = [
            item for item in list_node_executions("project", run["id"]) if item["nodeId"] == "loop"
        ][0]["checkpoint"]
        assert checkpoint == {
            "backgroundTaskId": "durable-task",
            "status": "completed",
            "toolIterations": 2,
            "outputTokenReservationId": "durable-task",
        }
        manager.resume(run["id"])
        assert _wait(run["id"])["status"] == "completed"
        assert adapter.calls == 1
        assert adapter.reuses == 1
    finally:
        manager._executor.shutdown(wait = True)


def test_loop_adapter_persists_task_id_before_durable_enqueue(monkeypatch):
    order = []
    runtime_snapshot = {
        "kind": "local",
        "model": "test-model",
        "providerId": None,
        "providerType": "local",
        "permissionMode": "off",
        "reasoningEffort": None,
        "maxOutputTokens": 32,
    }
    task = {
        "id": "00000000-0000-4000-8000-000000000001",
        "projectId": "project",
        "status": "queued",
        "result": None,
    }

    def enqueue(project_id, instruction, **kwargs):
        order.append(("enqueue", kwargs["task_id"]))
        assert kwargs["start"] is False
        assert kwargs["runtime_selection"] is None
        assert kwargs["runtime_snapshot"] == runtime_snapshot
        task["id"] = kwargs["task_id"]
        return dict(task)

    def start(task_id):
        order.append(("start", task_id))
        task["status"] = "completed"
        task["result"] = {"output": "done", "toolEvents": 4}
        return dict(task)

    monkeypatch.setattr("core.agent_workspace.background.manager.enqueue_agent", enqueue)
    monkeypatch.setattr("core.agent_workspace.background.manager.start", start)
    monkeypatch.setattr(
        "core.agent_workspace.graphs.background_manager_task", lambda task_id: dict(task)
    )

    adapter = GraphLoopAdapter()
    result = adapter.run(
        "project",
        "run",
        {"kind": "local", "model": "test-model", "permissionMode": "off"},
        threading.Event(),
        runtime_snapshot = runtime_snapshot,
        checkpoint_callback = lambda checkpoint: order.append(
            ("checkpoint", checkpoint["status"], checkpoint["backgroundTaskId"])
        ),
        before_start = lambda task_id: order.append(("reserve", task_id)),
    )
    assert result["output"] == "done"
    assert [item[0] for item in order] == [
        "checkpoint",
        "reserve",
        "enqueue",
        "checkpoint",
        "start",
        "checkpoint",
    ]
    assert order[0][2] == order[2][1]
    assert order[1][1] == order[2][1]
    assert order[-1][1:] == ("completed", task["id"])


def test_loop_adapter_rotates_terminal_checkpoint_task_id(monkeypatch):
    old_task_id = "00000000-0000-4000-8000-000000000001"
    tasks = {
        old_task_id: {
            "id": old_task_id,
            "projectId": "project",
            "status": "cancelled",
            "result": None,
        }
    }
    enqueued_ids = []

    def enqueue(project_id, instruction, **kwargs):
        task_id = kwargs["task_id"]
        assert task_id != old_task_id
        enqueued_ids.append(task_id)
        tasks[task_id] = {
            "id": task_id,
            "projectId": project_id,
            "status": "queued",
            "result": None,
        }
        return dict(tasks[task_id])

    def start(task_id):
        tasks[task_id]["status"] = "completed"
        tasks[task_id]["result"] = {"output": "resumed", "toolEvents": 0}
        return dict(tasks[task_id])

    monkeypatch.setattr("core.agent_workspace.background.manager.enqueue_agent", enqueue)
    monkeypatch.setattr("core.agent_workspace.background.manager.start", start)
    monkeypatch.setattr(
        "core.agent_workspace.graphs.background_manager_task",
        lambda task_id: dict(tasks[task_id]) if task_id in tasks else None,
    )

    checkpoints = []
    result = GraphLoopAdapter().run(
        "project",
        "resume",
        {"permissionMode": "off"},
        threading.Event(),
        checkpoint = {"backgroundTaskId": old_task_id, "status": "cancelled"},
        checkpoint_callback = checkpoints.append,
    )

    assert result["output"] == "resumed"
    assert len(enqueued_ids) == 1
    assert checkpoints[0]["backgroundTaskId"] == enqueued_ids[0]
    assert checkpoints[-1]["status"] == "completed"


def test_loop_enqueue_failure_reuses_atomic_token_reservation(tmp_path, monkeypatch):
    _folder_project(tmp_path)
    tasks = {}
    enqueue_calls = 0

    def enqueue(project_id, instruction, **kwargs):
        nonlocal enqueue_calls
        enqueue_calls += 1
        if enqueue_calls == 1:
            raise RuntimeError("crashed before durable enqueue")
        task = {
            "id": kwargs["task_id"],
            "projectId": project_id,
            "status": "queued",
            "result": None,
        }
        tasks[task["id"]] = task
        return dict(task)

    def start(task_id):
        tasks[task_id]["status"] = "completed"
        tasks[task_id]["result"] = {"output": "done", "toolEvents": 0}
        return dict(tasks[task_id])

    monkeypatch.setattr("core.agent_workspace.background.manager.enqueue_agent", enqueue)
    monkeypatch.setattr("core.agent_workspace.background.manager.start", start)
    monkeypatch.setattr(
        "core.agent_workspace.graphs.background_manager_task",
        lambda task_id: dict(tasks[task_id]) if task_id in tasks else None,
    )
    runtime = {
        "kind": "local",
        "model": "test-model",
        "permissionMode": "off",
        "maxOutputTokens": 3,
    }
    graph = create_graph(
        "project",
        {
            **_spec(
                _node("input", "input"),
                _node(
                    "loop",
                    "loop",
                    {"instruction": "run", "runtime": runtime},
                    {"maxAttempts": 2, "backoffMs": 0, "retryOn": ["error"]},
                ),
                _node("output", "output"),
                edges = [
                    {"from": "input", "to": "loop"},
                    {"from": "loop", "to": "output"},
                ],
            ),
            "limits": {"maxOutputTokens": 3},
        },
    )
    manager = GraphRunManager(max_workers = 1)
    try:
        run = manager.enqueue("project", graph["id"], {})
        finished = _wait(run["id"])
        assert finished["status"] == "completed"
        assert finished["reservedOutputTokens"] == 3
        assert enqueue_calls == 2
        attempts = [
            item for item in list_node_executions("project", run["id"]) if item["nodeId"] == "loop"
        ]
        assert (
            attempts[0]["checkpoint"]["outputTokenReservationId"]
            == attempts[1]["checkpoint"]["outputTokenReservationId"]
        )
    finally:
        manager._executor.shutdown(wait = True)


def test_side_effecting_tools_require_keys_and_are_not_replayed(tmp_path, monkeypatch):
    _folder_project(tmp_path)
    with pytest.raises(AgentWorkspaceError, match = "idempotencyKey"):
        validate_graph_spec(
            {
                **_spec(
                    _node("input", "input"),
                    _node(
                        "tool",
                        "tool",
                        {"serverId": "server", "toolName": "write", "arguments": {}},
                    ),
                    _node("output", "output"),
                    edges = [
                        {"from": "input", "to": "tool"},
                        {"from": "tool", "to": "output"},
                    ],
                ),
                "permissions": {"allowedToolServerIds": ["server"]},
            }
        )

    monkeypatch.setattr(
        "core.agent_workspace.graphs.mcp_servers_db.get_server",
        lambda server_id: {
            "id": server_id,
            "url": "http://example.test/mcp",
            "is_enabled": True,
            "headers": {},
            "use_oauth": False,
        },
    )
    calls = []

    def call_tool(*args, **kwargs):
        calls.append((args, kwargs))
        return {"written": True}

    monkeypatch.setattr("core.agent_workspace.graphs.call_tool_sync", call_tool)
    graph = create_graph(
        "project",
        {
            **_spec(
                _node("input", "input"),
                _node(
                    "tool",
                    "tool",
                    {
                        "serverId": "server",
                        "toolName": "write",
                        "arguments": {"value": "{input.value}"},
                        "sideEffecting": True,
                        "idempotencyKey": "write-value-1",
                    },
                ),
                _node("output", "output"),
                edges = [
                    {"from": "input", "to": "tool"},
                    {"from": "tool", "to": "output"},
                ],
            ),
            "permissions": {"allowedToolServerIds": ["server"]},
        },
    )
    manager = GraphRunManager(max_workers = 1)
    try:
        first = manager.enqueue("project", graph["id"], {"value": "same"})
        second = manager.enqueue("project", graph["id"], {"value": "same"})
        assert _wait(first["id"])["status"] == "completed"
        assert _wait(second["id"])["status"] == "completed"
        assert len(calls) == 1
        assert calls[0][0][3] == {"value": "same"}
        assert any(
            event["type"] == "tool.effect.reused"
            for event in list_graph_events("project", second["id"])
        )
    finally:
        manager._executor.shutdown(wait = True)


def test_uncertain_tool_effect_fails_closed_across_runs(tmp_path, monkeypatch):
    _folder_project(tmp_path)
    monkeypatch.setattr(
        "core.agent_workspace.graphs.mcp_servers_db.get_server",
        lambda server_id: {
            "id": server_id,
            "url": "http://example.test/mcp",
            "is_enabled": True,
            "headers": {},
            "use_oauth": False,
        },
    )
    calls = 0

    def fail_tool(*args, **kwargs):
        nonlocal calls
        calls += 1
        return "Error: connection ended after dispatch"

    monkeypatch.setattr("core.agent_workspace.graphs.call_tool_sync", fail_tool)
    graph = create_graph(
        "project",
        {
            **_spec(
                _node("input", "input"),
                _node(
                    "tool",
                    "tool",
                    {
                        "serverId": "server",
                        "toolName": "write",
                        "arguments": {"value": 1},
                        "sideEffecting": True,
                        "idempotencyKey": "uncertain-write",
                    },
                ),
                _node("output", "output"),
                edges = [
                    {"from": "input", "to": "tool"},
                    {"from": "tool", "to": "output"},
                ],
            ),
            "permissions": {"allowedToolServerIds": ["server"]},
        },
    )
    manager = GraphRunManager(max_workers = 1)
    try:
        first = manager.enqueue("project", graph["id"], {})
        assert _wait(first["id"])["status"] == "failed"
        assert "may have occurred" in (get_graph_run("project", first["id"])["error"] or "")
        second = manager.enqueue("project", graph["id"], {})
        assert _wait(second["id"])["status"] == "failed"
        assert "state is uncertain" in (get_graph_run("project", second["id"])["error"] or "")
        assert calls == 1
    finally:
        manager._executor.shutdown(wait = True)


@pytest.mark.timeout(60)
def test_real_local_stdio_mcp_graph_smoke(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setenv("UNSLOTH_STUDIO_ALLOW_STDIO_MCP", "1")
    monkeypatch.setattr(mcp_servers_db, "_schema_ready", False)
    _folder_project(tmp_path)
    fixture = Path(__file__).parent / "fixtures" / "mcp_argument_echo_server.py"
    server_url = mcp_client.join_stdio_command([sys.executable, str(fixture), "graph-smoke"])
    mcp_servers_db.create_server(
        "real-echo",
        "Real graph echo",
        server_url,
        headers_json = json.dumps({"UNSLOTH_MCP_ARGUMENT_MARKER": "sloth-graph"}),
    )
    graph = create_graph(
        "project",
        {
            **_spec(
                _node("input", "input"),
                _node(
                    "tool",
                    "tool",
                    {
                        "serverId": "real-echo",
                        "toolName": "launch_state",
                        "arguments": {},
                        "sideEffecting": False,
                        "timeoutSeconds": 20,
                    },
                ),
                _node("output", "output"),
                edges = [
                    {"from": "input", "to": "tool"},
                    {"from": "tool", "to": "output"},
                ],
                name = "Real MCP smoke",
            ),
            "permissions": {"allowedToolServerIds": ["real-echo"]},
            "outputSchema": {"type": "string"},
        },
    )
    manager = GraphRunManager(max_workers = 1)
    try:
        run = manager.enqueue("project", graph["id"], {})
        finished = _wait(run["id"], timeout = 30)
        assert finished["status"] == "completed"
        assert json.loads(finished["output"]) == {
            "arguments": ["graph-smoke"],
            "marker": "sloth-graph",
        }
    finally:
        manager._executor.shutdown(wait = True)
        mcp_client.close_stdio_sessions(server_url)


def test_validate_and_revision_history_apis_are_project_scoped(tmp_path):
    _folder_project(tmp_path)
    client = _client()
    payload = _spec(
        _node("input", "input"),
        _node("output", "output"),
        edges = [{"from": "input", "to": "output"}],
    )
    validated = client.post("/api/agent-workspace/projects/project/graphs/validate", json = payload)
    assert validated.status_code == 200
    assert validated.json()["document"]["limits"]["maxIterations"] == 100
    graph = client.post("/api/agent-workspace/projects/project/graphs", json = payload).json()
    update = client.put(
        f"/api/agent-workspace/projects/project/graphs/{graph['id']}",
        json = {**payload, "name": "Graph revision 2", "expectedRevision": 1},
    )
    assert update.status_code == 200
    revisions = client.get(f"/api/agent-workspace/projects/project/graphs/{graph['id']}/revisions")
    assert revisions.status_code == 200
    assert [item["revision"] for item in revisions.json()["revisions"]] == [2, 1]
    assert [item["revision"] for item in list_graph_revisions("project", graph["id"])] == [2, 1]
    assert (
        client.get(
            f"/api/agent-workspace/projects/other/graphs/{graph['id']}/revisions"
        ).status_code
        == 404
    )


def test_graph_source_round_trips_without_output_redaction(tmp_path):
    _folder_project(tmp_path)
    client = _client()
    instruction = "  Inspect /Users/example/project/.env and report token=literal-value\n"
    payload = _spec(
        _node("input", "input"),
        _node("loop", "loop", {"instruction": instruction}),
        _node("output", "output"),
        edges = [
            {"from": "input", "to": "loop"},
            {"from": "loop", "to": "output"},
        ],
        name = "Exact source",
    )

    validated = client.post("/api/agent-workspace/projects/project/graphs/validate", json = payload)
    assert validated.status_code == 200
    assert validated.json()["document"]["nodes"][1]["config"]["instruction"] == instruction

    graph = client.post("/api/agent-workspace/projects/project/graphs", json = payload).json()
    fetched = client.get(f"/api/agent-workspace/projects/project/graphs/{graph['id']}")
    assert fetched.status_code == 200
    assert fetched.json()["revision"]["nodes"][1]["config"]["instruction"] == instruction


def test_graph_rejects_malformed_and_oversized_run_input(tmp_path):
    _folder_project(tmp_path)
    graph = create_graph(
        "project",
        _spec(
            _node("input", "input"),
            _node("output", "output"),
            edges = [{"from": "input", "to": "output"}],
        ),
    )
    with pytest.raises(AgentWorkspaceError, match = "object"):
        create_graph_run("project", graph["id"], ["not", "an", "object"])
    with pytest.raises(AgentWorkspaceError, match = "too large"):
        create_graph_run("project", graph["id"], {"value": "x" * (300 * 1024)})


def test_node_admission_rolls_back_budget_execution_and_event_together(tmp_path, monkeypatch):
    _folder_project(tmp_path)
    graph = create_graph(
        "project",
        _spec(
            _node("input", "input"),
            _node("output", "output"),
            edges = [{"from": "input", "to": "output"}],
        ),
    )
    run = create_graph_run("project", graph["id"], {})
    from core.agent_workspace.graphs import admit_node_execution, claim_graph_run

    claim_graph_run(run["id"])

    def fail_event(*args, **kwargs):
        raise RuntimeError("crash during admission")

    monkeypatch.setattr("core.agent_workspace.graphs._insert_graph_event", fail_event)
    with pytest.raises(RuntimeError, match = "crash during admission"):
        admit_node_execution(
            run["id"],
            {"id": "input", "type": "input"},
            None,
            1,
            max_iterations = 2,
        )

    assert get_graph_run("project", run["id"])["iterationCount"] == 0
    assert list_node_executions("project", run["id"]) == []
    assert not any(
        event["type"] == "node.started" for event in list_graph_events("project", run["id"])
    )


def test_restart_completes_pending_cancellation_and_refuses_resume(tmp_path):
    _folder_project(tmp_path)
    graph = create_graph(
        "project",
        _spec(
            _node("input", "input"),
            _node("output", "output"),
            edges = [{"from": "input", "to": "output"}],
        ),
    )
    run = create_graph_run("project", graph["id"], {})
    from core.agent_workspace.graphs import (
        claim_graph_run,
        create_node_execution,
        request_graph_cancel,
        resume_graph_run,
    )

    claim_graph_run(run["id"])
    create_node_execution(run["id"], {"id": "input", "type": "input"}, None, 1)
    assert request_graph_cancel(run["id"])["status"] == "cancelling"
    assert recover_graph_runs() == 1

    recovered = get_graph_run("project", run["id"])
    assert recovered["status"] == "cancelled"
    assert recovered["cancelRequested"] is True
    assert list_node_executions("project", run["id"])[0]["status"] == "cancelled"
    assert list_graph_events("project", run["id"])[-1]["type"] == "run.cancelled"
    with pytest.raises(AgentWorkspaceError, match = "cannot be resumed"):
        resume_graph_run(run["id"])


def test_persisted_cancellation_dominates_a_concurrent_failure_finish(tmp_path):
    _folder_project(tmp_path)
    graph = create_graph(
        "project",
        _spec(
            _node("input", "input"),
            _node("output", "output"),
            edges = [{"from": "input", "to": "output"}],
        ),
    )
    run = create_graph_run("project", graph["id"], {})
    from core.agent_workspace.graphs import claim_graph_run, finish_graph_run, request_graph_cancel

    claim_graph_run(run["id"])
    request_graph_cancel(run["id"])
    finished = finish_graph_run(
        run["id"],
        "failed",
        "run.failed",
        error = "worker failed after the cancellation commit",
    )

    assert finished["status"] == "cancelled"
    assert finished["error"] == "Graph run cancelled."
    assert list_graph_events("project", run["id"])[-1]["type"] == "run.cancelled"


def test_recovery_advances_past_a_durably_completed_cursor_node(tmp_path):
    _folder_project(tmp_path)
    graph = create_graph(
        "project",
        _spec(
            _node("input", "input"),
            _node("output", "output"),
            edges = [{"from": "input", "to": "output"}],
        ),
    )
    run = create_graph_run("project", graph["id"], {"value": "once"})
    from core.agent_workspace.graphs import (
        claim_graph_run,
        consume_graph_budget,
        create_node_execution,
        finish_node_execution,
        update_graph_run,
    )

    claim_graph_run(run["id"])
    update_graph_run(run["id"], current_node_id = "input")
    consume_graph_budget(run["id"], max_iterations = 100, max_output_tokens = 262_144, iterations = 1)
    execution = create_node_execution(run["id"], {"id": "input", "type": "input"}, None, 1)
    finish_node_execution(execution["id"], "completed", output = {"value": "once"})
    assert recover_graph_runs() == 1

    manager = GraphRunManager(max_workers = 1)
    try:
        manager.resume(run["id"])
        finished = _wait(run["id"])
        assert finished["status"] == "completed"
        assert finished["iterationCount"] == 2
        assert [item["nodeId"] for item in list_node_executions("project", run["id"])] == [
            "input",
            "output",
        ]
    finally:
        manager._executor.shutdown(wait = True)


def test_recovery_reconstructs_completed_path_independent_of_row_order(tmp_path, monkeypatch):
    _folder_project(tmp_path)
    graph = create_graph(
        "project",
        {
            **_spec(
                _node("input", "input"),
                _node("condition", "condition", {"path": "input.ok"}),
                _node("yes", "output"),
                _node("no", "output"),
                edges = [
                    {"from": "input", "to": "condition"},
                    {"from": "condition", "to": "yes", "when": "true"},
                    {"from": "condition", "to": "no", "when": "false"},
                ],
            ),
            "outputSchema": {"type": "boolean"},
        },
    )
    run = create_graph_run("project", graph["id"], {"ok": True})
    from core.agent_workspace.graphs import (
        claim_graph_run,
        create_node_execution,
        finish_node_execution,
    )

    claim_graph_run(run["id"])
    input_execution = create_node_execution(run["id"], {"id": "input", "type": "input"}, None, 1)
    finish_node_execution(input_execution["id"], "completed", output = {"ok": True})
    condition_execution = create_node_execution(
        run["id"], {"id": "condition", "type": "condition"}, {"ok": True}, 1
    )
    finish_node_execution(condition_execution["id"], "completed", output = True)
    assert recover_graph_runs() == 1

    real_list = list_node_executions

    def reversed_executions(project_id, run_id):
        return list(reversed(real_list(project_id, run_id)))

    monkeypatch.setattr(
        "core.agent_workspace.graphs.list_node_executions",
        reversed_executions,
    )
    manager = GraphRunManager(max_workers = 1)
    try:
        manager.resume(run["id"])
        finished = _wait(run["id"])
        assert finished["status"] == "completed"
        assert finished["output"] is True
    finally:
        manager._executor.shutdown(wait = True)


def test_recovery_does_not_exceed_the_pinned_retry_policy(tmp_path):
    _folder_project(tmp_path)

    class _MustNotRun(GraphLoopAdapter):
        def __init__(self):
            self.calls = 0

        def run(self, project_id, instruction, runtime, cancel_event):
            self.calls += 1
            return {"output": "unexpected"}

    adapter = _MustNotRun()
    graph = create_graph(
        "project",
        _spec(
            _node("input", "input"),
            _node(
                "loop",
                "loop",
                {"instruction": "run"},
                {"maxAttempts": 1, "backoffMs": 0, "retryOn": ["error"]},
            ),
            _node("output", "output"),
            edges = [
                {"from": "input", "to": "loop"},
                {"from": "loop", "to": "output"},
            ],
        ),
    )
    run = create_graph_run("project", graph["id"], {})
    from core.agent_workspace.graphs import (
        claim_graph_run,
        create_node_execution,
        finish_node_execution,
        update_graph_run,
    )

    claim_graph_run(run["id"])
    input_execution = create_node_execution(run["id"], {"id": "input", "type": "input"}, None, 1)
    finish_node_execution(input_execution["id"], "completed", output = {})
    failed_execution = create_node_execution(run["id"], {"id": "loop", "type": "loop"}, {}, 1)
    finish_node_execution(failed_execution["id"], "failed", error = "first failure")
    update_graph_run(run["id"], current_node_id = "loop")
    assert recover_graph_runs() == 1

    manager = GraphRunManager(max_workers = 1, loop_adapter = adapter)
    try:
        manager.resume(run["id"])
        finished = _wait(run["id"])
        assert finished["status"] == "failed"
        assert finished["error"] == "first failure"
        assert adapter.calls == 0
        assert len(list_node_executions("project", run["id"])) == 2
    finally:
        manager._executor.shutdown(wait = True)


def test_committed_node_completion_survives_a_lost_commit_ack(tmp_path, monkeypatch):
    _folder_project(tmp_path)
    graph = create_graph(
        "project",
        _spec(
            _node("input", "input"),
            _node("output", "output"),
            edges = [{"from": "input", "to": "output"}],
        ),
    )
    from core.agent_workspace import graphs as graph_module

    real_complete = graph_module.complete_node_execution
    raised = False

    def complete_then_lose_ack(*args, **kwargs):
        nonlocal raised
        real_complete(*args, **kwargs)
        if not raised:
            raised = True
            raise sqlite3.OperationalError("commit acknowledgement lost")

    monkeypatch.setattr(graph_module, "complete_node_execution", complete_then_lose_ack)
    manager = GraphRunManager(max_workers = 1)
    try:
        run = manager.enqueue("project", graph["id"], {"value": "kept"})
        finished = _wait(run["id"])
        assert finished["status"] == "completed"
        assert finished["output"] == {"value": "kept"}
        input_execution = list_node_executions("project", run["id"])[0]
        assert input_execution["status"] == "completed"
        assert input_execution["output"] == {"value": "kept"}
    finally:
        manager._executor.shutdown(wait = True)


def test_terminal_cancel_request_wins_over_graph_completion(tmp_path, monkeypatch):
    _folder_project(tmp_path)
    graph = create_graph(
        "project",
        _spec(
            _node("input", "input"),
            _node("output", "output"),
            edges = [{"from": "input", "to": "output"}],
        ),
    )
    from core.agent_workspace import graphs as graph_module

    real_complete = graph_module.complete_node_execution

    def complete_then_cancel(run_id, execution_id, node_id, attempt, output, next_node_id):
        real_complete(run_id, execution_id, node_id, attempt, output, next_node_id)
        if node_id == "output":
            graph_module.request_graph_cancel(run_id)

    monkeypatch.setattr(graph_module, "complete_node_execution", complete_then_cancel)
    manager = GraphRunManager(max_workers = 1)
    try:
        run = manager.enqueue("project", graph["id"], {"value": "stop"})
        finished = _wait(run["id"])
        assert finished["status"] == "cancelled"
        assert finished["cancelRequested"] is True
        event_types = [event["type"] for event in list_graph_events("project", run["id"])]
        assert event_types[-1] == "run.cancelled"
        assert "run.completed" not in event_types
    finally:
        manager._executor.shutdown(wait = True)


def test_run_completion_event_is_not_a_separate_append(tmp_path, monkeypatch):
    _folder_project(tmp_path)
    graph = create_graph(
        "project",
        _spec(
            _node("input", "input"),
            _node("output", "output"),
            edges = [{"from": "input", "to": "output"}],
        ),
    )
    from core.agent_workspace import graphs as graph_module

    real_append = graph_module.append_graph_event

    def reject_separate_completion(run_id, event_type, **kwargs):
        if event_type == "run.completed":
            raise sqlite3.OperationalError("separate completion append rejected")
        return real_append(run_id, event_type, **kwargs)

    monkeypatch.setattr(graph_module, "append_graph_event", reject_separate_completion)
    manager = GraphRunManager(max_workers = 1)
    try:
        run = manager.enqueue("project", graph["id"], {})
        finished = _wait(run["id"])
        assert finished["status"] == "completed"
        assert [event["type"] for event in list_graph_events("project", run["id"])][
            -1
        ] == "run.completed"
    finally:
        manager._executor.shutdown(wait = True)


def test_queued_runs_recover_as_interrupted_and_can_start_through_api(tmp_path):
    _folder_project(tmp_path)
    client = _client()
    payload = _spec(
        _node("input", "input"),
        _node("output", "output"),
        edges = [{"from": "input", "to": "output"}],
    )
    graph = client.post("/api/agent-workspace/projects/project/graphs", json = payload).json()
    queued = client.post(
        f"/api/agent-workspace/projects/project/graphs/{graph['id']}/runs",
        json = {"input": {"value": 1}, "start": False},
    ).json()
    assert queued["status"] == "queued"
    assert recover_graph_runs() == 1
    assert get_graph_run("project", queued["id"])["status"] == "interrupted"

    resumed = client.post(f"/api/agent-workspace/projects/project/graph-runs/{queued['id']}/resume")
    assert resumed.status_code == 200
    assert _wait(queued["id"])["status"] == "completed"

    second = client.post(
        f"/api/agent-workspace/projects/project/graphs/{graph['id']}/runs",
        json = {"input": {"value": 2}, "start": False},
    ).json()
    started = client.post(f"/api/agent-workspace/projects/project/graph-runs/{second['id']}/start")
    assert started.status_code == 200
    assert _wait(second["id"])["status"] == "completed"


def test_project_deletion_fences_resume_and_scans_all_active_history(tmp_path):
    _folder_project(tmp_path)
    graph = create_graph(
        "project",
        _spec(
            _node("input", "input"),
            _node("output", "output"),
            edges = [{"from": "input", "to": "output"}],
        ),
    )
    paused = create_graph_run("project", graph["id"], {})
    from core.agent_workspace.graphs import request_graph_pause
    from core.agent_workspace.state import connection

    request_graph_pause(paused["id"])
    conn = connection()
    rows = [
        (
            f"history-{index}",
            "project",
            graph["id"],
            1,
            "{}",
            "completed",
            index + 10,
            index + 10,
            index + 10,
        )
        for index in range(501)
    ]
    conn.executemany(
        "INSERT INTO agent_graph_runs(id, project_id, graph_id, revision, input_json, status, created_at, updated_at, completed_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.execute(
        "UPDATE agent_graph_runs SET created_at = 1, updated_at = 1 WHERE id = ?",
        (paused["id"],),
    )
    conn.commit()
    conn.close()

    manager = GraphRunManager(max_workers = 1)
    manager.begin_project_deletion("project")
    try:
        with pytest.raises(AgentWorkspaceError, match = "deletion"):
            manager.resume(paused["id"])
        stopped = manager.cancel_project_runs_and_wait("project")
        assert [item["id"] for item in stopped] == [paused["id"]]
        assert get_graph_run("project", paused["id"])["status"] == "cancelled"
    finally:
        manager.finish_project_deletion("project")
        manager._executor.shutdown(wait = True)


def test_run_runtime_budget_uses_the_persisted_start_time(tmp_path):
    _folder_project(tmp_path)
    graph = create_graph(
        "project",
        {
            **_spec(
                _node("input", "input"),
                _node("output", "output"),
                edges = [{"from": "input", "to": "output"}],
            ),
            "limits": {
                "maxNodes": 2,
                "maxRunSeconds": 1,
                "maxOutputBytes": 1024,
                "maxIterations": 2,
                "maxOutputTokens": 1,
            },
        },
    )
    run = create_graph_run("project", graph["id"], {})
    from core.agent_workspace.state import connection

    conn = connection()
    conn.execute(
        "UPDATE agent_graph_runs SET started_at = ? WHERE id = ?",
        (int(time.time() * 1000) - 5_000, run["id"]),
    )
    conn.commit()
    conn.close()

    manager = GraphRunManager(max_workers = 1)
    try:
        manager.start(run["id"])
        finished = _wait(run["id"])
        assert finished["status"] == "failed"
        assert "budget exhausted" in (finished["error"] or "")
        assert finished["iterationCount"] == 0
    finally:
        manager._executor.shutdown(wait = True)


def test_shutdown_during_retry_backoff_is_interrupted(tmp_path):
    _folder_project(tmp_path)

    class _Fails(GraphLoopAdapter):
        def run(self, project_id, instruction, runtime, cancel_event):
            raise AgentWorkspaceError("retry")

    runtime = {
        "kind": "local",
        "model": "test-model",
        "permissionMode": "off",
        "maxOutputTokens": 1,
    }
    graph = create_graph(
        "project",
        _spec(
            _node("input", "input"),
            _node(
                "loop",
                "loop",
                {"instruction": "run", "runtime": runtime},
                {"maxAttempts": 2, "backoffMs": 5_000, "retryOn": ["error"]},
            ),
            _node("output", "output"),
            edges = [
                {"from": "input", "to": "loop"},
                {"from": "loop", "to": "output"},
            ],
        ),
    )
    manager = GraphRunManager(max_workers = 1, loop_adapter = _Fails())
    try:
        run = manager.enqueue("project", graph["id"], {})
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            if any(
                event["type"] == "node.retrying"
                for event in list_graph_events("project", run["id"])
            ):
                break
            time.sleep(0.01)
        else:
            raise AssertionError("run did not enter retry backoff")
        manager.prepare_for_app_exit(timeout_seconds = 2)
        assert get_graph_run("project", run["id"])["status"] == "interrupted"
    finally:
        manager._executor.shutdown(wait = True)


def test_retry_requests_are_idempotent(tmp_path):
    _folder_project(tmp_path)
    graph = create_graph(
        "project",
        _spec(
            _node("input", "input"),
            _node("output", "output"),
            edges = [{"from": "input", "to": "output"}],
        ),
    )
    previous = create_graph_run("project", graph["id"], {})
    from core.agent_workspace.graphs import update_graph_run

    update_graph_run(previous["id"], status = "failed", error = "retry")
    manager = GraphRunManager(max_workers = 1)
    try:
        first = manager.retry("project", previous["id"], start = False)
        second = manager.retry("project", previous["id"], start = False)
        assert first["id"] == second["id"]
        assert first["retryOfRunId"] == previous["id"]
        assert len(list_graph_runs("project", graph["id"])) == 2
        manager.start(first["id"])
        assert _wait(first["id"])["status"] == "completed"
        assert manager.retry("project", previous["id"])["id"] == first["id"]
    finally:
        manager._executor.shutdown(wait = True)


def test_retry_route_rechecks_the_execution_boundary(tmp_path, monkeypatch):
    _folder_project(tmp_path)
    runtime = {
        "kind": "local",
        "model": "test-model",
        "permissionMode": "off",
        "maxOutputTokens": 1,
    }
    graph = create_graph(
        "project",
        _spec(
            _node("input", "input"),
            _node("loop", "loop", {"instruction": "run", "runtime": runtime}),
            _node("output", "output"),
            edges = [
                {"from": "input", "to": "loop"},
                {"from": "loop", "to": "output"},
            ],
        ),
    )
    previous = create_graph_run("project", graph["id"], {})
    from core.agent_workspace.graphs import update_graph_run

    update_graph_run(previous["id"], status = "failed", error = "retry")

    def blocked():
        raise AgentWorkspaceError("Execution boundary unavailable.")

    monkeypatch.setattr(agent_workspace_routes, "_require_execution_boundary", blocked)
    response = _client().post(
        f"/api/agent-workspace/projects/project/graph-runs/{previous['id']}/retry"
    )
    assert response.status_code == 409
    assert len(list_graph_runs("project", graph["id"])) == 1


def test_stopped_side_effect_capable_loop_fails_closed(tmp_path):
    _folder_project(tmp_path)

    class _Fails(GraphLoopAdapter):
        def run(self, project_id, instruction, runtime, cancel_event):
            raise AgentWorkspaceError("task stopped after workspace edits")

    runtime = {
        "kind": "local",
        "model": "test-model",
        "permissionMode": "full",
        "maxOutputTokens": 1,
    }
    graph = create_graph(
        "project",
        _spec(
            _node("input", "input"),
            _node("loop", "loop", {"instruction": "edit", "runtime": runtime}),
            _node("output", "output"),
            edges = [
                {"from": "input", "to": "loop"},
                {"from": "loop", "to": "output"},
            ],
        ),
    )
    manager = GraphRunManager(max_workers = 1, loop_adapter = _Fails())
    try:
        run = manager.enqueue("project", graph["id"], {})
        assert _wait(run["id"])["status"] == "failed"
        with pytest.raises(AgentWorkspaceError, match = "Inspect the project"):
            manager.retry("project", run["id"], start = False)
    finally:
        manager._executor.shutdown(wait = True)


def test_tool_dispatch_revalidates_server_configuration(tmp_path, monkeypatch):
    _folder_project(tmp_path)
    server = {
        "id": "server",
        "url": "http://example.test/mcp",
        "is_enabled": True,
        "headers": {},
        "use_oauth": False,
    }
    monkeypatch.setattr(
        "core.agent_workspace.graphs.mcp_servers_db.get_server",
        lambda server_id: dict(server),
    )

    def call_tool(*args, **kwargs):
        server["is_enabled"] = False
        assert kwargs["config_check"]() is False
        return "Error: MCP server or approved operation changed before dispatch"

    monkeypatch.setattr("core.agent_workspace.graphs.call_tool_sync", call_tool)
    graph = create_graph(
        "project",
        {
            **_spec(
                _node("input", "input"),
                _node(
                    "tool",
                    "tool",
                    {
                        "serverId": "server",
                        "toolName": "read",
                        "arguments": {},
                        "sideEffecting": False,
                    },
                ),
                _node("output", "output"),
                edges = [
                    {"from": "input", "to": "tool"},
                    {"from": "tool", "to": "output"},
                ],
            ),
            "permissions": {"allowedToolServerIds": ["server"]},
        },
    )
    manager = GraphRunManager(max_workers = 1)
    try:
        run = manager.enqueue("project", graph["id"], {})
        assert _wait(run["id"])["status"] == "failed"
        assert "changed before dispatch" in (get_graph_run("project", run["id"])["error"] or "")
    finally:
        manager._executor.shutdown(wait = True)


def test_approval_cannot_release_a_changed_mcp_endpoint(tmp_path, monkeypatch):
    _folder_project(tmp_path)
    server = {
        "id": "server",
        "url": "http://example.test/first",
        "is_enabled": True,
        "headers_json": json.dumps({"Authorization": "Bearer first"}),
        "use_oauth": False,
    }
    monkeypatch.setattr(
        "core.agent_workspace.graphs.mcp_servers_db.get_server",
        lambda server_id: dict(server),
    )
    calls = []
    monkeypatch.setattr(
        "core.agent_workspace.graphs.call_tool_sync",
        lambda *args, **kwargs: calls.append((args, kwargs)) or {"ok": True},
    )
    graph = create_graph(
        "project",
        {
            **_spec(
                _node("input", "input"),
                _node("approval", "approval", {"title": "Use connector"}),
                _node(
                    "tool",
                    "tool",
                    {
                        "serverId": "server",
                        "toolName": "read",
                        "arguments": {},
                        "sideEffecting": False,
                    },
                ),
                _node("output", "output"),
                edges = [
                    {"from": "input", "to": "approval"},
                    {"from": "approval", "to": "tool"},
                    {"from": "tool", "to": "output"},
                ],
            ),
            "permissions": {"allowedToolServerIds": ["server"]},
        },
    )
    manager = GraphRunManager(max_workers = 1)
    try:
        run = manager.enqueue("project", graph["id"], {})
        approval = _wait_for_approval(run["id"])
        server["url"] = "http://example.test/second"
        decide_graph_approval("project", run["id"], approval["id"], "approved")
        finished = _wait(run["id"])
        assert finished["status"] == "failed"
        assert "changed after this run was created" in (finished["error"] or "")
        assert calls == []
    finally:
        manager._executor.shutdown(wait = True)


def test_approval_cannot_release_a_changed_mcp_oauth_account(tmp_path, monkeypatch):
    _folder_project(tmp_path)
    server = {
        "id": "server",
        "url": "https://example.test/mcp",
        "is_enabled": True,
        "headers_json": None,
        "use_oauth": True,
    }
    oauth_binding = ["a" * 64]
    monkeypatch.setattr(
        "core.agent_workspace.graphs.mcp_servers_db.get_server",
        lambda server_id: dict(server),
    )
    monkeypatch.setattr(
        "core.agent_workspace.graphs.oauth_credential_binding",
        lambda url: oauth_binding[0],
    )
    calls = []
    monkeypatch.setattr(
        "core.agent_workspace.graphs.call_tool_sync",
        lambda *args, **kwargs: calls.append((args, kwargs)) or {"ok": True},
    )
    graph = create_graph(
        "project",
        {
            **_spec(
                _node("input", "input"),
                _node("approval", "approval", {"title": "Use OAuth connector"}),
                _node(
                    "tool",
                    "tool",
                    {
                        "serverId": "server",
                        "toolName": "read",
                        "arguments": {},
                        "sideEffecting": False,
                    },
                ),
                _node("output", "output"),
                edges = [
                    {"from": "input", "to": "approval"},
                    {"from": "approval", "to": "tool"},
                    {"from": "tool", "to": "output"},
                ],
            ),
            "permissions": {"allowedToolServerIds": ["server"]},
        },
    )
    manager = GraphRunManager(max_workers = 1)
    try:
        run = manager.enqueue("project", graph["id"], {})
        approval = _wait_for_approval(run["id"])
        oauth_binding[0] = "b" * 64
        decide_graph_approval("project", run["id"], approval["id"], "approved")
        finished = _wait(run["id"])
        assert finished["status"] == "failed"
        assert "changed after this run was created" in (finished["error"] or "")
        assert calls == []
    finally:
        manager._executor.shutdown(wait = True)


def test_mcp_timeout_uses_timeout_retry_policy(tmp_path, monkeypatch):
    _folder_project(tmp_path)
    monkeypatch.setattr(
        "core.agent_workspace.graphs.mcp_servers_db.get_server",
        lambda server_id: {
            "id": server_id,
            "url": "http://example.test/mcp",
            "is_enabled": True,
            "headers": {},
            "use_oauth": False,
        },
    )
    calls = 0

    def call_tool(*args, **kwargs):
        nonlocal calls
        calls += 1
        return "Error: MCP tool 'read' timed out after 1s" if calls == 1 else {"ok": True}

    monkeypatch.setattr("core.agent_workspace.graphs.call_tool_sync", call_tool)
    graph = create_graph(
        "project",
        {
            **_spec(
                _node("input", "input"),
                _node(
                    "tool",
                    "tool",
                    {
                        "serverId": "server",
                        "toolName": "read",
                        "arguments": {},
                        "sideEffecting": False,
                    },
                    {"maxAttempts": 2, "backoffMs": 0, "retryOn": ["timeout"]},
                ),
                _node("output", "output"),
                edges = [
                    {"from": "input", "to": "tool"},
                    {"from": "tool", "to": "output"},
                ],
            ),
            "permissions": {"allowedToolServerIds": ["server"]},
        },
    )
    manager = GraphRunManager(max_workers = 1)
    try:
        run = manager.enqueue("project", graph["id"], {})
        assert _wait(run["id"])["status"] == "completed"
        assert calls == 2
    finally:
        manager._executor.shutdown(wait = True)


def test_tool_effect_receipt_survives_graph_history_deletion(tmp_path, monkeypatch):
    _folder_project(tmp_path)
    monkeypatch.setattr(
        "core.agent_workspace.graphs.mcp_servers_db.get_server",
        lambda server_id: {
            "id": server_id,
            "url": "http://example.test/mcp",
            "is_enabled": True,
            "headers": {},
            "use_oauth": False,
        },
    )
    calls = 0

    def call_tool(*args, **kwargs):
        nonlocal calls
        calls += 1
        return {"written": True}

    monkeypatch.setattr("core.agent_workspace.graphs.call_tool_sync", call_tool)
    document = {
        **_spec(
            _node("input", "input"),
            _node(
                "tool",
                "tool",
                {
                    "serverId": "server",
                    "toolName": "write",
                    "arguments": {"value": 1},
                    "sideEffecting": True,
                    "idempotencyKey": "durable-effect",
                },
            ),
            _node("output", "output"),
            edges = [
                {"from": "input", "to": "tool"},
                {"from": "tool", "to": "output"},
            ],
            name = "Durable effect",
        ),
        "permissions": {"allowedToolServerIds": ["server"]},
    }
    manager = GraphRunManager(max_workers = 1)
    try:
        graph = create_graph("project", document)
        first = manager.enqueue("project", graph["id"], {})
        assert _wait(first["id"])["status"] == "completed"
        delete_graph("project", graph["id"])

        replacement = create_graph("project", document)
        second = manager.enqueue("project", replacement["id"], {})
        assert _wait(second["id"])["status"] == "completed"
        assert calls == 1
    finally:
        manager._executor.shutdown(wait = True)


def test_graph_schema_upgrade_preserves_old_runs_and_adds_durable_columns(tmp_path):
    _folder_project(tmp_path)
    from core.agent_workspace.state import connection

    conn = connection()
    conn.executescript(
        """
        CREATE TABLE agent_graph_runs (
            id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL,
            graph_id TEXT NOT NULL,
            revision INTEGER NOT NULL,
            input_json TEXT NOT NULL,
            output_json TEXT,
            error TEXT,
            current_node_id TEXT,
            status TEXT NOT NULL,
            attempt INTEGER NOT NULL DEFAULT 1,
            retry_of_run_id TEXT,
            idempotency_key TEXT,
            pause_requested INTEGER NOT NULL DEFAULT 0,
            cancel_requested INTEGER NOT NULL DEFAULT 0,
            created_at INTEGER NOT NULL,
            updated_at INTEGER NOT NULL,
            started_at INTEGER,
            completed_at INTEGER
        );
        CREATE TABLE agent_graph_node_executions (
            id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            node_id TEXT NOT NULL,
            node_type TEXT NOT NULL,
            attempt INTEGER NOT NULL,
            status TEXT NOT NULL,
            input_json TEXT,
            output_json TEXT,
            error TEXT,
            created_at INTEGER NOT NULL,
            started_at INTEGER,
            completed_at INTEGER,
            UNIQUE(run_id, node_id, attempt)
        );
        INSERT INTO agent_graph_runs(
            id, project_id, graph_id, revision, input_json, status, created_at, updated_at
        ) VALUES ('old-run', 'project', 'old-graph', 1, '{}', 'completed', 1, 1);
        INSERT INTO agent_graph_node_executions(
            id, run_id, node_id, node_type, attempt, status, created_at
        ) VALUES ('old-node', 'old-run', 'input', 'input', 1, 'completed', 1);
        """
    )
    conn.commit()
    conn.close()

    run = get_graph_run("project", "old-run")
    assert run is not None
    assert run["iterationCount"] == 0
    assert run["reservedOutputTokens"] == 0
    execution = list_node_executions("project", "old-run")[0]
    assert execution["checkpoint"] is None


def test_tool_effect_schema_upgrade_removes_graph_and_run_cascades(tmp_path):
    _folder_project(tmp_path)
    graph = create_graph(
        "project",
        _spec(
            _node("input", "input"),
            _node("output", "output"),
            edges = [{"from": "input", "to": "output"}],
        ),
    )
    run = create_graph_run("project", graph["id"], {})
    from core.agent_workspace.graphs import update_graph_run
    from core.agent_workspace.state import connection

    update_graph_run(run["id"], status = "completed", output = {})
    conn = connection()
    conn.executescript(
        """
        DROP INDEX IF EXISTS idx_agent_graph_tool_effects_run;
        DROP TABLE agent_graph_tool_effects;
        CREATE TABLE agent_graph_tool_effects (
            id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL REFERENCES chat_projects(id) ON DELETE CASCADE,
            graph_id TEXT NOT NULL REFERENCES agent_graphs(id) ON DELETE CASCADE,
            run_id TEXT NOT NULL REFERENCES agent_graph_runs(id) ON DELETE CASCADE,
            node_id TEXT NOT NULL,
            server_id TEXT NOT NULL,
            tool_name TEXT NOT NULL,
            idempotency_key TEXT NOT NULL,
            arguments_hash TEXT NOT NULL,
            status TEXT NOT NULL,
            output_json TEXT,
            error TEXT,
            created_at INTEGER NOT NULL,
            updated_at INTEGER NOT NULL,
            completed_at INTEGER,
            UNIQUE(project_id, server_id, tool_name, idempotency_key)
        );
        """
    )
    conn.execute(
        "INSERT INTO agent_graph_tool_effects(id, project_id, graph_id, run_id, node_id, server_id, tool_name, "
        "idempotency_key, arguments_hash, status, output_json, created_at, updated_at, completed_at) "
        "VALUES ('effect', 'project', ?, ?, 'tool', 'server', 'write', 'key', 'hash', 'completed', '{}', 1, 1, 1)",
        (graph["id"], run["id"]),
    )
    conn.commit()
    conn.close()

    assert get_graph_run("project", run["id"]) is not None
    conn = connection()
    foreign_tables = {
        row[2]
        for row in conn.execute("PRAGMA foreign_key_list(agent_graph_tool_effects)").fetchall()
    }
    conn.close()
    assert foreign_tables == {"chat_projects"}

    delete_graph("project", graph["id"])
    conn = connection()
    receipt = conn.execute(
        "SELECT status FROM agent_graph_tool_effects WHERE id = 'effect'"
    ).fetchone()
    conn.close()
    assert receipt["status"] == "completed"


def test_tool_effect_migration_recovers_a_legacy_table_left_by_interruption(tmp_path):
    _folder_project(tmp_path)
    graph = create_graph(
        "project",
        _spec(
            _node("input", "input"),
            _node("output", "output"),
            edges = [{"from": "input", "to": "output"}],
        ),
    )
    run = create_graph_run("project", graph["id"], {})
    from core.agent_workspace.graphs import _create_tool_effects_table
    from core.agent_workspace.state import connection

    conn = connection()
    conn.execute(
        "INSERT INTO agent_graph_tool_effects(id, project_id, graph_id, run_id, node_id, server_id, tool_name, "
        "idempotency_key, arguments_hash, status, output_json, created_at, updated_at, completed_at) "
        "VALUES ('recover-effect', 'project', ?, ?, 'tool', 'server', 'write', 'recover-key', "
        "'recover-hash', 'completed', '{\"ok\":true}', 1, 1, 1)",
        (graph["id"], run["id"]),
    )
    conn.execute("DROP INDEX IF EXISTS idx_agent_graph_tool_effects_run")
    conn.execute("ALTER TABLE agent_graph_tool_effects RENAME TO agent_graph_tool_effects_legacy")
    _create_tool_effects_table(conn)
    conn.commit()
    conn.close()

    assert get_graph_run("project", run["id"]) is not None
    conn = connection()
    receipt = conn.execute(
        "SELECT status, output_json FROM agent_graph_tool_effects WHERE id = 'recover-effect'"
    ).fetchone()
    legacy = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' "
        "AND name = 'agent_graph_tool_effects_legacy'"
    ).fetchone()
    conn.close()
    assert receipt["status"] == "completed"
    assert json.loads(receipt["output_json"]) == {"ok": True}
    assert legacy is None


def test_native_loop_pause_waits_for_a_safe_completed_checkpoint(tmp_path, monkeypatch):
    _folder_project(tmp_path)
    entered = threading.Event()
    task = {
        "id": "00000000-0000-4000-8000-000000000001",
        "projectId": "project",
        "status": "queued",
        "result": None,
    }
    cancellations = []

    def enqueue(project_id, instruction, **kwargs):
        task.update(
            {
                "id": kwargs["task_id"],
                "projectId": project_id,
                "status": "queued",
                "result": None,
                "payload": {"runtime": kwargs["runtime_snapshot"]},
            }
        )
        return dict(task)

    def start(task_id):
        task["status"] = "running"
        entered.set()
        return dict(task)

    def cancel(task_id):
        cancellations.append(task_id)
        task["status"] = "cancelled"
        return dict(task)

    monkeypatch.setattr("core.agent_workspace.background.manager.enqueue_agent", enqueue)
    monkeypatch.setattr("core.agent_workspace.background.manager.start", start)
    monkeypatch.setattr("core.agent_workspace.background.manager.cancel", cancel)
    monkeypatch.setattr(
        "core.agent_workspace.graphs.background_manager_task", lambda task_id: dict(task)
    )
    runtime = {
        "kind": "local",
        "model": "test-model",
        "permissionMode": "full",
        "maxOutputTokens": 1,
    }
    graph = create_graph(
        "project",
        _spec(
            _node("input", "input"),
            _node("loop", "loop", {"instruction": "edit", "runtime": runtime}),
            _node("output", "output"),
            edges = [
                {"from": "input", "to": "loop"},
                {"from": "loop", "to": "output"},
            ],
        ),
    )
    manager = GraphRunManager(max_workers = 1)
    try:
        run = manager.enqueue("project", graph["id"], {})
        assert entered.wait(timeout = 2)
        manager.pause(run["id"])
        time.sleep(0.05)
        assert cancellations == []
        task["status"] = "completed"
        task["result"] = {"output": "done", "toolEvents": 0}
        assert _wait(run["id"])["status"] == "paused"

        manager.resume(run["id"])
        assert _wait(run["id"])["status"] == "completed"
        assert cancellations == []
        assert (
            len(
                [
                    item
                    for item in list_node_executions("project", run["id"])
                    if item["nodeId"] == "loop"
                ]
            )
            == 2
        )
    finally:
        manager._executor.shutdown(wait = True)


def test_runtime_budget_expiry_during_retry_backoff_is_failed(tmp_path):
    _folder_project(tmp_path)

    class _Fails(GraphLoopAdapter):
        def run(self, project_id, instruction, runtime, cancel_event):
            raise AgentWorkspaceError("retry")

    runtime = {
        "kind": "local",
        "model": "test-model",
        "permissionMode": "off",
        "maxOutputTokens": 1,
    }
    graph = create_graph(
        "project",
        {
            **_spec(
                _node("input", "input"),
                _node(
                    "loop",
                    "loop",
                    {"instruction": "run", "runtime": runtime},
                    {"maxAttempts": 2, "backoffMs": 5_000, "retryOn": ["error"]},
                ),
                _node("output", "output"),
                edges = [
                    {"from": "input", "to": "loop"},
                    {"from": "loop", "to": "output"},
                ],
            ),
            "limits": {
                "maxNodes": 3,
                "maxRunSeconds": 1,
                "maxOutputBytes": 1024,
                "maxIterations": 4,
                "maxOutputTokens": 2,
            },
        },
    )
    manager = GraphRunManager(max_workers = 1, loop_adapter = _Fails())
    try:
        run = manager.enqueue("project", graph["id"], {})
        finished = _wait(run["id"], timeout = 3)
        assert finished["status"] == "failed"
        assert finished["error"] == "Graph run budget exhausted."
        assert any(
            event["type"] == "run.failed" for event in list_graph_events("project", run["id"])
        )
    finally:
        manager._executor.shutdown(wait = True)
