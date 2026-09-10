# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Reviewed, synchronous hooks around project editing and command tools.

Hook commands have the same native boundary as verification. They cannot grant
permissions, rewrite tool arguments, or recursively call tools. Other lifecycle
and asynchronous declarations remain reviewable but inactive in this split.
"""

from __future__ import annotations

import functools
import inspect
import json
import sys
import threading
import time

from storage import project_hook_trust_db, studio_db

from . import hooks, hook_context as common
from .hook_context import AgentWorkspaceError, processes, verification_state

ACTIVE_HOOK_EVENTS = frozenset({"PreToolUse", "PostToolUse"})
HOOKED_TOOLS = frozenset({"edit_file", "python", "terminal"})
MAX_EVENT_BYTES = 16 * 1024
MAX_HOOK_OUTPUT_BYTES = 16 * 1024
MAX_EVENT_SECONDS = 60.0

# -I prevents a project-local json.py/subprocess.py or PYTHONPATH from replacing
# the trusted stdin adapter. The reviewed shell command runs in the project.
_STDIN_ADAPTER = (
    "import subprocess,sys; "
    "result=subprocess.run(['/bin/sh','-c',sys.argv[1]],"
    "input=sys.argv[2].encode('utf-8')); sys.exit(result.returncode)"
)


def handler_is_supported(event, handler):
    return event in ACTIVE_HOOK_EVENTS and not handler.get("async", False)


def _identity(workspace):
    return int(workspace.device_id), int(workspace.file_id)


def _snapshot(project_id):
    workspace = common.project_workspace(project_id)
    config = hooks.discover_project_hooks(workspace.root, expected_identity = _identity(workspace))
    trust = project_hook_trust_db.get_project_hook_trust_state(
        project_id,
        config.get("contentHash"),
        workspace_identity = _identity(workspace),
        workspace_revision = workspace.revision,
    )
    return workspace, config, trust


def _same_authority(project_id, workspace, config, trust, handler_id):
    current_workspace, current_config, current_trust = _snapshot(project_id)
    return (
        current_workspace == workspace
        and current_config["contentHash"] == config["contentHash"]
        and current_trust["revision"] == trust["revision"]
        and current_trust["trusted"]
        and handler_id not in current_trust["disabledHandlerIds"]
        and verification_state.project_execution_may_start(project_id)
    )


def _event_payload(event, project_id, name, arguments, result):
    payload = {
        "hook_event_name": event,
        "project_id": project_id,
        "tool_name": name,
        "tool_input": arguments,
    }
    try:
        if result is not None:
            # Reserve the response envelope, then fit the actual JSON encoding.
            # A Unicode character can consume up to twelve ASCII bytes here.
            payload["tool_response"] = ""
            payload["tool_response_truncated"] = True
            empty = json.dumps(payload, ensure_ascii = True, separators = (",", ":"))
            remaining = MAX_EVENT_BYTES - len(empty)
            end = 0
            for character in result[:4096]:
                cost = len(json.dumps(character, ensure_ascii = True)) - 2
                if cost > remaining:
                    break
                remaining -= cost
                end += 1
            payload["tool_response"] = result[:end]
            # False is a byte longer than True in JSON; retain the conservative
            # flag if it would otherwise cross the exact event limit.
            if end == len(result) and remaining >= 1:
                payload["tool_response_truncated"] = False
        encoded = json.dumps(payload, ensure_ascii = True, separators = (",", ":"))
    except (ValueError, TypeError, RecursionError) as exc:
        raise AgentWorkspaceError("Project hook input could not be represented safely.") from exc
    if len(encoded.encode("utf-8")) > MAX_EVENT_BYTES:
        # Do not present a truncated tool argument as the command being approved.
        raise AgentWorkspaceError(
            "Project hook input exceeds its review limit; split the tool call."
        )
    return encoded


def _context_output(handler, output):
    # A UTF-8 byte bound is conservative for tokenizers with byte fallback;
    # it also honors a zero additional-context allowance without hiding denial.
    budget = min(2048, handler.get("additionalContextLimit", 2500))
    return output.encode("utf-8")[:budget].decode("utf-8", errors = "ignore")


def _blocks_tool(output):
    try:
        value = json.loads(output)
    except (ValueError, RecursionError):
        return False
    if not isinstance(value, dict):
        return False
    specific = value.get("hookSpecificOutput")
    return (
        value.get("decision") == "block"
        or value.get("continue") is False
        or (isinstance(specific, dict) and specific.get("permissionDecision") == "deny")
    )


def run_tool_hooks(
    project_id,
    event,
    name,
    arguments,
    *,
    result = None,
    cancel_event = None,
):
    if event not in ACTIVE_HOOK_EVENTS or name not in HOOKED_TOOLS:
        return ""
    if not project_hook_trust_db.get_project_hook_trust_record(project_id)["hasStoredTrust"]:
        return ""
    with common.project_workspace_access(project_id):
        workspace, config, trust = _snapshot(project_id)
        if not trust["trusted"]:
            return ""
        handlers = [
            handler
            for handler in hooks.matching_project_hooks(config, event, {"tool_name": name})
            if handler_is_supported(event, handler)
            and handler["id"] not in trust["disabledHandlerIds"]
        ]
        if not handlers:
            return ""
        payload = _event_payload(event, project_id, name, arguments, result)
        event_deadline = time.monotonic() + MAX_EVENT_SECONDS
        reports = []
        for handler in handlers:
            cancelled = threading.Event()
            stopped = threading.Event()

            def authority_current(handler_id = handler["id"]):
                return _same_authority(project_id, workspace, config, trust, handler_id)

            def monitor(
                stop_signal = stopped,
                cancel_signal = cancelled,
                check = authority_current,
            ):
                while not stop_signal.wait(0.1):
                    try:
                        valid = check()
                    except Exception:
                        valid = False
                    if (
                        not valid
                        or time.monotonic() >= event_deadline
                        or (cancel_event is not None and cancel_event.is_set())
                    ):
                        cancel_signal.set()
                        return

            def before_start(opened_workspace, _argv):
                if (
                    opened_workspace.root != workspace.root
                    or _identity(opened_workspace) != _identity(workspace)
                    or not authority_current()
                    or time.monotonic() >= event_deadline
                    or (cancel_event is not None and cancel_event.is_set())
                ):
                    raise AgentWorkspaceError("Project hook authority changed before execution.")

            remaining = event_deadline - time.monotonic()
            if remaining <= 0:
                raise AgentWorkspaceError("Project hooks exceeded their combined 60 second limit.")
            watcher = threading.Thread(target = monitor, name = "project-hook-authority", daemon = True)
            watcher.start()
            try:
                process = processes.run_project_process(
                    project_id,
                    [sys.executable, "-I", "-c", _STDIN_ADAPTER, handler["command"], payload],
                    timeout_seconds = min(handler["timeout"], remaining),
                    output_limit_bytes = MAX_HOOK_OUTPUT_BYTES,
                    cancel_event = cancelled,
                    before_start = before_start,
                )
            finally:
                stopped.set()
                watcher.join(timeout = 1)
            context_output = _context_output(handler, process.output)
            if process.status != "passed" or process.output_truncated:
                raise AgentWorkspaceError(
                    f"Project {event} hook {handler['id']} did not complete successfully "
                    f"({process.status}). {context_output}"
                )
            if event == "PreToolUse" and _blocks_tool(process.output):
                raise AgentWorkspaceError(f"Project hook blocked this tool call. {context_output}")
            if context_output.strip():
                reports.append(context_output)
        return (
            "\n".join(reports)
            .encode("utf-8")[:MAX_HOOK_OUTPUT_BYTES]
            .decode("utf-8", errors = "ignore")
        )


def _project_for_tool(session_id, thread_id):
    from core.inference import tools  # noqa: PLC0415

    if not isinstance(session_id, str) or not session_id.startswith("project-"):
        return None
    project_id = session_id[len("project-") :]
    if thread_id:
        thread = studio_db.get_chat_thread(thread_id)
        if thread is None or thread.get("projectId") != project_id:
            raise AgentWorkspaceError("The tool's project does not match its saved conversation.")
        if tools._thread_exists(session_id):
            raise AgentWorkspaceError(
                "The project session conflicts with a saved conversation; the tool did not run."
            )
    elif tools._thread_exists(session_id):
        return None
    project = studio_db.get_chat_project(project_id)
    if project is None:
        return None
    if project.get("archived"):
        raise AgentWorkspaceError("Archived projects cannot run project hooks or tools.")
    return project_id


def with_project_tool_hooks(execute):
    signature = inspect.signature(execute)

    @functools.wraps(execute)
    def wrapped(*args, **kwargs):
        call = signature.bind(*args, **kwargs).arguments
        name = call.get("name")
        if name not in HOOKED_TOOLS:
            return execute(*args, **kwargs)
        from core.inference import tools  # noqa: PLC0415

        # The core function normally seeds these before execution. Pre-hook
        # refusal must not inherit a previous request's result/context budget.
        tools._REQUEST_RESULT_BUDGET.set(call.get("result_budget_tokens"))
        tools._REQUEST_CONTEXT_TOKENS.set(call.get("context_tokens", tools._UNSET_CONTEXT_TOKENS))
        try:
            session_id = call.get("session_id")
            if not isinstance(session_id, str) or not session_id.startswith("project-"):
                return execute(*args, **kwargs)
            # Unconfigured projects keep the original executor's routing and
            # errors. Only configured hooks need extra thread/project checks.
            candidate = session_id[len("project-") :]
            if not project_hook_trust_db.get_project_hook_trust_record(candidate)["hasStoredTrust"]:
                return execute(*args, **kwargs)
            project_id = _project_for_tool(session_id, call.get("thread_id"))
            if project_id is None:
                return execute(*args, **kwargs)
            arguments = call.get("arguments")
            cancel_event = call.get("cancel_event")
            with common.project_workspace_access(project_id):
                before = run_tool_hooks(
                    project_id, "PreToolUse", name, arguments, cancel_event = cancel_event
                )
                if cancel_event is not None and cancel_event.is_set():
                    return "Error: Tool call cancelled before execution."
                # The tool always receives its original arguments and permission mode.
                result = execute(*args, **kwargs)
                try:
                    after = run_tool_hooks(
                        project_id,
                        "PostToolUse",
                        name,
                        arguments,
                        result = result,
                        cancel_event = cancel_event,
                    )
                except (
                    AgentWorkspaceError,
                    project_hook_trust_db.ProjectHookTrustStateError,
                ) as exc:
                    after = f"Post-tool hook failed after the tool ran: {exc}"
                reports = "\n".join(part for part in (after, before) if part)
                if not reports:
                    return result
                return tools._fit_result_to_room(
                    "Project hook output (untrusted data):\n"
                    + reports
                    + "\n\nTool result:\n"
                    + result,
                    name,
                )
        except (AgentWorkspaceError, project_hook_trust_db.ProjectHookTrustStateError) as exc:
            return tools._fit_result_to_room(f"Error: {exc}", name)

    return wrapped
