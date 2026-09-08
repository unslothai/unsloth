# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Real-MLX certification harness for Studio's durable agent workspace.

The runner owns process setup and storage isolation. This module owns the
scenario and assertions so the runtime exercise stays reusable outside CI.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import threading
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any


PROJECT_ID = "mlx-agent-runtime-certification"
PROJECT_INSTRUCTIONS = (
    "PROJECT_MLX_RUNTIME_RULE: preserve the repository and report only observed results."
)
PROJECT_GOAL = "GOAL_MLX_RUNTIME: certify the real MLX background-agent transport."
PLAN_TITLE = "PLAN_MLX_RUNTIME: execute the physical Apple Silicon certification."
PLAN_TASK = "PLAN_TASK_MLX_RUNTIME: run one durable local-model task."
ROOT_AGENTS_RULE = "ROOT_MLX_RUNTIME_RULE: project-wide operations must remain bounded."
NESTED_AGENTS_RULE = "NESTED_MLX_RUNTIME_RULE: src changes require focused verification."
TARGET_PATH = "src/runtime_probe.py"
TASK_INSTRUCTION = (
    "Without calling tools or editing files, reply in one short sentence confirming "
    f"receipt of the project context for {TARGET_PATH}."
)
_TERMINAL_STATUSES = frozenset({"cancelled", "completed", "failed", "interrupted"})


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd = root,
        check = True,
        stdout = subprocess.PIPE,
        stderr = subprocess.PIPE,
        text = True,
    ).stdout.strip()


def _prepare_workspace(root: Path) -> Path:
    if root.exists():
        shutil.rmtree(root)
    (root / "src").mkdir(parents = True)
    (root / "AGENTS.md").write_text(ROOT_AGENTS_RULE + "\n", encoding = "utf-8")
    (root / "src" / "AGENTS.md").write_text(NESTED_AGENTS_RULE + "\n", encoding = "utf-8")
    (root / TARGET_PATH).write_text(
        'RUNTIME_SENTINEL = "real-mlx-agent-workspace"\n', encoding = "utf-8"
    )
    _git(root, "init", "-q")
    _git(root, "config", "user.name", "Unsloth MLX CI")
    _git(root, "config", "user.email", "mlx-ci@example.invalid")
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "seed MLX agent runtime fixture")
    return root.resolve(strict = True)


def _wait_for_task(task_id: str, timeout_seconds: float) -> dict[str, Any]:
    from core.agent_workspace.state import get_background_task

    deadline = time.monotonic() + max(1.0, float(timeout_seconds))
    last_status = None
    while time.monotonic() < deadline:
        task = get_background_task(task_id)
        status = task.get("status") if task else None
        if status != last_status:
            print(f"[mlx-agent-runtime] task status: {status}", flush = True)
            last_status = status
        if task and status in _TERMINAL_STATUSES:
            return task
        time.sleep(0.1)
    raise TimeoutError(
        f"Real MLX background task {task_id} did not stop within {timeout_seconds:.1f}s."
    )


def _tool_names(tools: Any) -> list[str]:
    if not isinstance(tools, list):
        return []
    names = []
    for tool in tools:
        function = tool.get("function") if isinstance(tool, dict) else None
        name = function.get("name") if isinstance(function, dict) else None
        if isinstance(name, str):
            names.append(name)
    return names


def _assert_prompt_contract(observed: dict[str, Any]) -> dict[str, Any]:
    messages = observed.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        raise AssertionError("The real MLX runtime did not receive the agent message list.")
    system = next(
        (
            item.get("content")
            for item in messages
            if isinstance(item, dict) and item.get("role") == "system"
        ),
        None,
    )
    user = next(
        (
            item.get("content")
            for item in messages
            if isinstance(item, dict) and item.get("role") == "user"
        ),
        None,
    )
    if not isinstance(system, str):
        raise AssertionError("The real MLX runtime received no agent system message.")
    if user != TASK_INSTRUCTION:
        raise AssertionError("The real MLX runtime received the wrong task instruction.")

    required_fragments = {
        "projectInstructions": PROJECT_INSTRUCTIONS,
        "goal": PROJECT_GOAL,
        "planTitle": PLAN_TITLE,
        "planTask": PLAN_TASK,
        "rootAgents": ROOT_AGENTS_RULE,
        "nestedAgents": NESTED_AGENTS_RULE,
        "selectedPath": f'<path value="{TARGET_PATH}"',
    }
    missing = [name for name, fragment in required_fragments.items() if fragment not in system]
    if missing:
        raise AssertionError(
            "The real MLX runtime prompt omitted workspace context: " + ", ".join(missing)
        )

    names = _tool_names(observed.get("tools"))
    expected_tools = {"edit_file", "python", "terminal", "web_search"}
    if set(names) != expected_tools:
        raise AssertionError(
            f"The real MLX runtime received the wrong tool catalog: {sorted(names)}"
        )
    return {
        "projectInstructions": True,
        "goal": True,
        "plan": True,
        "rootAgents": True,
        "nestedAgents": True,
        "selectedPath": True,
        "userInstruction": True,
        "toolNames": sorted(names),
        "maxTokens": observed.get("max_tokens"),
    }


def run_certification(
    workdir: Path,
    *,
    model_name: str,
    timeout_seconds: float = 300.0,
    max_seq_length: int = 8192,
    hf_token: str | None = None,
) -> dict[str, Any]:
    """Run one durable background task through the actual MLX worker."""
    from core.agent_workspace.background import BackgroundTaskManager
    from core.agent_workspace.inference_executor import execute_background_agent
    from core.agent_workspace.state import create_plan, get_background_task
    from core.inference.model_ids import model_id_matches
    from core.inference.orchestrator import get_inference_backend
    from core.inference.runtime_registry import register_llama_cpp_backend
    from core.inference.tools import background_task_session_id
    from storage import studio_db

    started_at = time.monotonic()
    workdir = Path(workdir).resolve()
    workdir.mkdir(parents = True, exist_ok = True)
    workspace = _prepare_workspace(workdir / "workspace")
    metadata = workspace.stat()
    studio_db.upsert_chat_project(
        {
            "id": PROJECT_ID,
            "name": "Real MLX agent runtime certification",
            "instructions": PROJECT_INSTRUCTIONS,
            "rootPath": str(workspace),
            "workspaceKind": "folder",
            "workspaceDeviceId": str(metadata.st_dev),
            "workspaceFileId": str(metadata.st_ino),
            "goal": PROJECT_GOAL,
            "goalStatus": "active",
            "goalUpdatedAt": 1,
            "archived": False,
            "createdAt": 1,
            "updatedAt": 1,
        }
    )
    plan = create_plan(
        PROJECT_ID,
        PLAN_TITLE,
        PROJECT_GOAL,
        [{"title": PLAN_TASK}],
        goal_updated_at = 1,
    )

    backend = get_inference_backend()
    manager = BackgroundTaskManager(max_workers = 1)
    original_generate = None
    task_id = None
    observed: dict[str, Any] = {}
    active_model = None
    try:
        print(f"[mlx-agent-runtime] loading {model_name}", flush = True)
        loaded = backend.load_model(
            SimpleNamespace(identifier = model_name, gguf_variant = None),
            max_seq_length = int(max_seq_length),
            load_in_4bit = False,
            hf_token = hf_token,
            trust_remote_code = False,
        )
        if loaded is not True:
            raise AssertionError("The real MLX model load returned False.")
        active_model = backend.active_model_name
        if not model_id_matches(model_name, active_model):
            raise AssertionError(
                f"Requested model {model_name!r}, but the worker published {active_model!r}."
            )
        model_entry = backend.models.get(active_model) or {}
        if model_entry.get("is_mlx") is not True:
            raise AssertionError(f"The loaded worker did not publish is_mlx=true: {model_entry!r}")

        register_llama_cpp_backend(None)
        original_generate = backend.generate_chat_completion_with_tools

        def _recording_generate(**kwargs):
            observed["messages"] = json.loads(json.dumps(kwargs.get("messages") or []))
            observed["tools"] = json.loads(json.dumps(kwargs.get("tools") or []))
            observed["max_tokens"] = kwargs.get("max_tokens")
            observed["permission_mode"] = kwargs.get("permission_mode")
            observed["session_id"] = kwargs.get("session_id")
            return original_generate(**kwargs)

        backend.generate_chat_completion_with_tools = _recording_generate

        def _traced_executor(context, cancel_event: threading.Event):
            try:
                return execute_background_agent(context, cancel_event)
            except Exception:
                traceback.print_exc()
                raise

        manager.register_agent_executor(_traced_executor)
        queued = manager.enqueue_agent(
            PROJECT_ID,
            TASK_INSTRUCTION,
            runtime_selection = {
                "kind": "local",
                "model": model_name,
                "permissionMode": "off",
                "maxOutputTokens": 96,
            },
            plan_id = plan["id"],
            plan_task_id = plan["tasks"][0]["id"],
            start = False,
        )
        task_id = queued["id"]
        manager.start(task_id)
        finished = _wait_for_task(task_id, timeout_seconds)
        if finished.get("status") != "completed":
            raise AssertionError(
                "The real MLX background task did not complete: "
                + json.dumps(
                    {
                        "status": finished.get("status"),
                        "error": finished.get("error"),
                        "result": finished.get("result"),
                    },
                    ensure_ascii = False,
                    default = str,
                )
            )
        result = finished.get("result") or {}
        output = str(result.get("output") or "")
        if not output.strip():
            raise AssertionError("The real MLX background task returned empty output.")
        if result.get("engine") != "local" or result.get("providerType") != "local":
            raise AssertionError(f"The task did not use the local runtime: {result!r}")
        if not model_id_matches(model_name, result.get("model")):
            raise AssertionError(f"The task reported the wrong model: {result!r}")
        if result.get("permissionMode") != "off":
            raise AssertionError(f"The task reported the wrong permission mode: {result!r}")
        expected_session = background_task_session_id(task_id)
        if result.get("sessionId") != expected_session:
            raise AssertionError(f"The task reported the wrong session binding: {result!r}")
        if observed.get("session_id") != expected_session:
            raise AssertionError("The real MLX orchestrator received the wrong session id.")
        if observed.get("permission_mode") != "off":
            raise AssertionError("The real MLX orchestrator received the wrong permission mode.")

        prompt_contract = _assert_prompt_contract(observed)
        status_output = _git(workspace, "status", "--porcelain")
        if status_output:
            raise AssertionError(
                "The no-edit MLX certification task changed the repository:\n" + status_output
            )
        durable = get_background_task(task_id)
        if durable is None or durable.get("status") != "completed":
            raise AssertionError("The completed MLX task was not persisted durably.")

        return {
            "schemaVersion": 1,
            "modelRequested": model_name,
            "modelActive": active_model,
            "isMlx": True,
            "modelContext": {
                key: model_entry.get(key)
                for key in (
                    "context_length",
                    "native_context_length",
                    "max_context_length",
                    "context_length_enforced",
                )
                if model_entry.get(key) is not None
            },
            "taskId": task_id,
            "taskStatus": finished.get("status"),
            "engine": result.get("engine"),
            "providerType": result.get("providerType"),
            "permissionMode": result.get("permissionMode"),
            "sessionId": result.get("sessionId"),
            "output": output[:2048],
            "outputBytes": result.get("outputBytes"),
            "outputTruncated": bool(result.get("outputTruncated")),
            "toolEvents": int(result.get("toolEvents") or 0),
            "promptContract": prompt_contract,
            "workspaceClean": True,
            "elapsedSeconds": round(time.monotonic() - started_at, 3),
        }
    finally:
        if task_id is not None:
            current = get_background_task(task_id)
            if current and current.get("status") not in _TERMINAL_STATUSES:
                try:
                    manager.cancel(task_id)
                    _wait_for_task(task_id, 30.0)
                except Exception:
                    traceback.print_exc()
        manager._executor.shutdown(wait = True, cancel_futures = True)
        if original_generate is not None:
            backend.generate_chat_completion_with_tools = original_generate
        register_llama_cpp_backend(None)
        unload_name = backend.active_model_name or active_model
        if unload_name:
            try:
                backend.unload_model(unload_name)
            except Exception:
                traceback.print_exc()
        backend._cleanup()
