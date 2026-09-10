# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Opt-in, Linux-confined task commands resolved from durable worktree bindings."""

import math
import importlib
import threading
import time
from contextlib import contextmanager


def availability():
    try:
        supervisor = importlib.import_module(__package__ + ".supervisor")
        task_executor = importlib.import_module(__package__ + ".task_executor")
        task_workspaces = importlib.import_module(__package__ + ".task_workspaces")
        if (
            supervisor.TASK_PROCESS_PROTOCOL != 1
            or task_executor.TASK_COMMAND_PROTOCOL != 1
            or task_workspaces.TASK_COMMAND_BINDING_PROTOCOL != 1
        ):
            return {
                "available": False,
                "reason": "Update task and command prerequisites before enabling commands.",
            }
        worktrees = importlib.import_module(__package__ + ".worktrees")
        if worktrees.OWNED_WORKTREE_LOOKUP_PROTOCOL != 1:
            return {"available": False, "reason": "Update owned worktree lookup support first."}
        status = supervisor.supervised_process_status()
        return {
            "available": status.available,
            "reason": None
            if status.available
            else "Task commands require Linux with supervised bubblewrap isolation.",
        }
    except (ImportError, AttributeError):
        return {
            "available": False,
            "reason": "Install task execution and supervised command support first.",
        }


def require_support():
    from .task_state import TaskStateError
    status = availability()
    if not status["available"]:
        raise TaskStateError(status["reason"])


def tool_definition():
    return {
        "type": "function",
        "function": {
            "name": "task_run_command",
            "description": "Run a test or build in your own checkout, with no network or Git metadata access. Up to six commands per attempt, each at most 120 seconds and 64 KiB captured output. Commands can modify checkout files. Dependencies must already be available. Return evidence; do not claim unrun checks passed.",
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "argv": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                        "maxItems": 64,
                    },
                    "timeout": {"type": "integer", "minimum": 1, "maximum": 120},
                },
                "required": ["argv", "timeout"],
            },
        },
    }


def _validate_workspace(context, *, process_guard_held = False):
    from .task_command_state import require_context
    from .task_runtime import validate_runtime
    from .task_workspaces import binding, validate_workspace
    from .worktrees import owned_worktree_path, _owned_worktree_path
    from .common import ProjectWorkspace
    from .task_state import TaskStateError

    task = require_context(context)
    validate_runtime(task["snapshot"]["runtime"])
    validate_workspace(task["projectId"], task["snapshot"]["workspace"])
    record = binding(task["projectId"], task["id"])
    if record is None:
        raise TaskStateError("This attempt has no owned worktree.")
    lookup = _owned_worktree_path if process_guard_held else owned_worktree_path
    root = lookup(task["projectId"], record["worktree_id"])
    metadata = root.stat(follow_symlinks = False)
    if (metadata.st_dev, metadata.st_ino) != (record["device"], record["inode"]):
        raise TaskStateError("The task worktree identity changed.")
    return ProjectWorkspace(task["projectId"], root, "managed", record["device"], record["inode"])


@contextmanager
def command_workspace_access(project_id, context):
    from .task_command_state import require_context
    from .common import project_workspace_access
    from .worktrees import project_operation
    from .task_state import TaskStateError

    task = require_context(context)
    if task["projectId"] != project_id:
        raise TaskStateError("Task command project ownership changed.")
    # The supervisor owns this lease through quarantine, beyond the tool return.
    with project_workspace_access(project_id):
        with project_operation(project_id):
            workspace = _validate_workspace(context)
        yield workspace


def _arguments(arguments):
    from .task_state import TaskStateError
    from . import supervisor

    if not isinstance(arguments, dict) or set(arguments) != {"argv", "timeout"}:
        raise TaskStateError("Commands accept only argv and timeout.")
    argv, timeout = arguments["argv"], arguments["timeout"]
    if not isinstance(argv, list) or not 1 <= len(argv) <= 64:
        raise TaskStateError("Use a command argument list with at most 64 entries.")
    command = supervisor._normalized_argv(argv)
    if sum(len(arg.encode("utf-8")) for arg in command) > 8192:
        raise TaskStateError("The command arguments exceed 8 KiB.")
    if isinstance(timeout, bool) or not isinstance(timeout, int) or not 1 <= timeout <= 120:
        raise TaskStateError("Command timeout must be between 1 and 120 seconds.")
    return command, timeout


def execute_command(context, arguments):
    from . import supervisor, task_command_state as evidence
    from .task_state import TaskStateError

    task = evidence.require_context(context)
    command, timeout = _arguments(arguments)
    require_support()
    # Resolve ownership before reserving durable command budget, then again in
    # the supervisor lease and at both physical spawn/release boundaries.
    with command_workspace_access(task["projectId"], context):
        pass
    command_id, receipt = evidence.begin(context, command, timeout)
    stopped, cancel = threading.Event(), threading.Event()

    def watch():
        while not stopped.wait(0.1):
            try:
                context.check()
            except Exception:
                cancel.set()
                return

    def before_start(workspace, argv):
        current = _validate_workspace(context, process_guard_held = True)
        if current != workspace or tuple(argv) != command:
            raise TaskStateError("The task command workspace changed before execution.")

    watcher = threading.Thread(target = watch, daemon = True, name = "task-command-cancel")
    watcher.start()
    try:
        remaining = context.deadline - time.monotonic()
        if not math.isfinite(remaining) or remaining <= 0:
            raise TaskStateError("Task deadline reached before command execution.")
        result = supervisor._run_project_process(
            task["projectId"],
            command,
            timeout_seconds = min(timeout, remaining),
            output_limit_bytes = evidence.MAX_OUTPUT_BYTES,
            cancel_event = cancel,
            output_callback = None,
            before_start = before_start,
            _task_context = context,
        )
        evidence.finish(
            command_id,
            receipt,
            result.status,
            exit_code = result.exit_code,
            output = result.output,
            output_bytes = result.output_bytes,
            truncated = result.output_truncated,
        )
    except supervisor.ProjectProcessContainmentError:
        # Stop further tools; physical process ownership remains in the supervisor.
        context.cancel_event.set()
        evidence.finish(
            command_id,
            receipt,
            "containment_pending",
            output = "The process tree has not been proven stopped. Worktree operations remain fenced during cleanup.",
        )
        raise TaskStateError("Task command cleanup is still pending.") from None
    except supervisor.ProjectExecutionUnavailable:
        evidence.finish(
            command_id,
            receipt,
            "unavailable",
            output = "The supervised command boundary was unavailable. No passing result was established.",
        )
    except TaskStateError:
        if context.cancel_event.is_set() or time.monotonic() >= context.deadline:
            evidence.finish(
                command_id,
                receipt,
                "cancelled",
                output = "The task was cancelled before command execution could complete.",
            )
        else:
            evidence.finish(
                command_id,
                receipt,
                "interrupted",
                output = "Command ownership changed before a confirmed result.",
            )
            raise
    except BaseException:
        evidence.finish(
            command_id,
            receipt,
            "interrupted",
            output = "Command execution ended without a confirmed result.",
        )
        raise
    finally:
        stopped.set()
        watcher.join(timeout = 1)
    return evidence.read(task["projectId"], task["id"], command_id, summary = True)


def list_commands(project_id, task_id):
    from .task_command_state import read
    return read(project_id, task_id)


def get_command(project_id, task_id, command_id):
    from .task_command_state import read
    return read(project_id, task_id, command_id)
