# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Small stdio MCP bridge from cloud Claude Code to a local Claude Code child."""

from __future__ import annotations

import json
import os
import signal
import shutil
import subprocess
import sys
import threading
from typing import Any, Callable

from unsloth_cli.commands.start import (
    _CLAUDE_ENV_UNSET,
    _SUBAGENT_DESCRIPTION,
    _SUBAGENT_INSTRUCTIONS,
    _claude_flags,
    _claude_local_env,
    _wsl_shim_env,
)

_MAX_RESULT_CHARACTERS = 100_000
_CANCEL_POLL_SECONDS = 0.1
_CANCEL_GRACE_SECONDS = 2.0


def _required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing {name}.")
    return value


def _bounded(text: str) -> str:
    if len(text) <= _MAX_RESULT_CHARACTERS:
        return text
    return text[:_MAX_RESULT_CHARACTERS] + "\n\n[Local agent output truncated]"


def _result_text(stdout: str) -> str:
    lines = [line for line in stdout.splitlines() if line.strip()]
    candidates = [stdout.strip(), *reversed(lines)]
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except ValueError:
            continue
        if not isinstance(payload, dict):
            continue
        result = payload.get("result")
        if payload.get("is_error"):
            raise RuntimeError(str(result or "The local Claude agent failed."))
        if isinstance(result, str) and result.strip():
            return _bounded(result.strip())
    raise RuntimeError("The local Claude agent returned no readable result.")


def _stop_child(process: subprocess.Popen) -> None:
    """Stop the Claude child and any tool processes it started."""
    if process.poll() is not None:
        return
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                capture_output = True,
                timeout = 15,
                check = False,
            )
        except Exception:
            process.terminate()
    else:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except OSError:
            process.terminate()
    try:
        process.wait(timeout = _CANCEL_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            process.kill()
        else:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except OSError:
                process.kill()
        process.wait()


def run_local_agent(task: str, cancel_event: threading.Event | None = None) -> str:
    base = _required_env("UNSLOTH_CLAUDE_SUBAGENT_BASE_URL")
    key = _required_env("UNSLOTH_CLAUDE_SUBAGENT_API_KEY")
    model = _required_env("UNSLOTH_CLAUDE_SUBAGENT_MODEL")
    window = int(os.environ.get("UNSLOTH_CLAUDE_SUBAGENT_CONTEXT_WINDOW", "0") or 0)
    entry = {"id": model, "context_length": window}
    local_env = _claude_local_env(base, key, entry)
    child_env = dict(os.environ)

    executable = shutil.which("claude")
    if executable is None:
        raise RuntimeError("`claude` is not installed or is not on PATH.")
    cancel_event = cancel_event or threading.Event()
    if cancel_event.is_set():
        raise RuntimeError("The local Claude agent was cancelled.")
    command = [
        "claude",
        "--model",
        model,
        *_claude_flags(model),
        "--permission-mode",
        (
            "bypassPermissions"
            if os.environ.get("UNSLOTH_CLAUDE_SUBAGENT_BYPASS_PERMISSIONS") == "1"
            else "acceptEdits"
        ),
        "--print",
        "--output-format",
        "json",
        "--no-session-persistence",
        "--append-system-prompt",
        _SUBAGENT_INSTRUCTIONS,
        f"Task: {task}",
    ]
    bridged, wsl_names = _wsl_shim_env(command, local_env, _CLAUDE_ENV_UNSET)
    if wsl_names:
        from unsloth_cli.commands.start import _merge_wslenv

        bridged = {**bridged, "PWD": os.getcwd()}
        child_env["WSLENV"] = _merge_wslenv(child_env.get("WSLENV", ""), wsl_names)
        for name in _CLAUDE_ENV_UNSET:
            child_env[name] = ""
    else:
        for name in _CLAUDE_ENV_UNSET:
            child_env.pop(name, None)
    child_env.update(bridged)
    popen_kwargs: dict[str, Any] = {
        "cwd": os.environ.get("CLAUDE_PROJECT_DIR") or os.getcwd(),
        "env": child_env,
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "text": True,
    }
    if os.name == "nt":
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        popen_kwargs["start_new_session"] = True
    process = subprocess.Popen(
        [executable, *command[1:]],
        **popen_kwargs,
    )
    try:
        while True:
            try:
                stdout, stderr = process.communicate(timeout = _CANCEL_POLL_SECONDS)
                break
            except subprocess.TimeoutExpired:
                if cancel_event.is_set():
                    _stop_child(process)
                    raise RuntimeError("The local Claude agent was cancelled.")
    except BaseException:
        if process.poll() is None:
            _stop_child(process)
        raise
    if process.returncode != 0:
        detail = stderr.strip() or stdout.strip()
        raise RuntimeError(
            _bounded(detail) or f"Local Claude exited with code {process.returncode}."
        )
    return _result_text(stdout)


def _response(request: dict, run_agent: Callable[[str], str] = run_local_agent) -> dict | None:
    request_id = request.get("id")
    method = request.get("method")
    if request_id is None:
        return None
    if method == "initialize":
        protocol = (request.get("params") or {}).get("protocolVersion") or "2025-06-18"
        result = {
            "protocolVersion": protocol,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": "unsloth-local-agent", "version": "1.0.0"},
        }
    elif method == "ping":
        result = {}
    elif method == "tools/list":
        result = {
            "tools": [
                {
                    "name": "unsloth_agent",
                    "title": "Unsloth local agent",
                    "description": _SUBAGENT_DESCRIPTION,
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "task": {
                                "type": "string",
                                "description": "The complete task for the local Unsloth agent.",
                            }
                        },
                        "required": ["task"],
                        "additionalProperties": False,
                    },
                    "annotations": {
                        "readOnlyHint": False,
                        "destructiveHint": True,
                        "idempotentHint": False,
                        "openWorldHint": True,
                    },
                    "_meta": {"anthropic/maxResultSizeChars": _MAX_RESULT_CHARACTERS},
                }
            ]
        }
    elif method == "tools/call":
        params = request.get("params") or {}
        arguments = params.get("arguments") or {}
        task = arguments.get("task") if params.get("name") == "unsloth_agent" else None
        if not isinstance(task, str) or not task.strip():
            result = {
                "content": [{"type": "text", "text": "A non-empty task is required."}],
                "isError": True,
            }
        else:
            try:
                text = run_agent(task.strip())
                result = {"content": [{"type": "text", "text": text}], "isError": False}
            except Exception as exc:
                result = {
                    "content": [{"type": "text", "text": str(exc)}],
                    "isError": True,
                }
    else:
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": -32601, "message": f"Method not found: {method}"},
        }
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def serve(
    stdin: Any = sys.stdin,
    stdout: Any = sys.stdout,
    run_agent: Callable[[str, threading.Event], str] = run_local_agent,
) -> None:
    active: dict[object, threading.Event] = {}
    workers: list[threading.Thread] = []
    state_lock = threading.RLock()
    output_lock = threading.Lock()
    shutdown_started = threading.Event()

    def cancel_active() -> None:
        with state_lock:
            pending = list(active.values())
        for cancel_event in pending:
            cancel_event.set()

    def handle_shutdown(_signum: int, _frame: Any) -> None:
        # Claude Code stops stdio MCP servers with SIGINT when a tool call is
        # cancelled, and may send it more than once. Only the first signal should
        # unwind stdin; later signals must not interrupt process-tree cleanup.
        first_signal = not shutdown_started.is_set()
        shutdown_started.set()
        cancel_active()
        if first_signal:
            raise KeyboardInterrupt

    previous_handlers: dict[int, Any] = {}
    if threading.current_thread() is threading.main_thread():
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[signum] = signal.signal(signum, handle_shutdown)

    def send(response: dict | None) -> None:
        if response is None:
            return
        with output_lock:
            stdout.write(json.dumps(response, separators = (",", ":")) + "\n")
            stdout.flush()

    def call_tool(request: dict, request_id: object, cancel_event: threading.Event) -> None:
        try:
            response = _response(
                request,
                run_agent = lambda task: run_agent(task, cancel_event),
            )
            if not cancel_event.is_set():
                send(response)
        finally:
            with state_lock:
                if active.get(request_id) is cancel_event:
                    active.pop(request_id, None)

    try:
        for line in stdin:
            try:
                request = json.loads(line)
                if not isinstance(request, dict):
                    response = None
                elif request.get("method") == "notifications/cancelled":
                    request_id = (request.get("params") or {}).get("requestId")
                    with state_lock:
                        cancel_event = active.get(request_id)
                    if cancel_event is not None:
                        cancel_event.set()
                    response = None
                elif request.get("method") == "tools/call" and request.get("id") is not None:
                    request_id = request["id"]
                    cancel_event = threading.Event()
                    with state_lock:
                        active[request_id] = cancel_event
                    worker = threading.Thread(
                        target = call_tool,
                        args = (request, request_id, cancel_event),
                        name = f"unsloth-agent-{request_id}",
                    )
                    workers.append(worker)
                    worker.start()
                    response = None
                else:
                    response = _response(request)
            except Exception as exc:
                response = {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32603, "message": str(exc)},
                }
            send(response)
    except KeyboardInterrupt:
        pass
    finally:
        cancel_active()
        for worker in workers:
            if worker.ident is not None:
                worker.join()
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


if __name__ == "__main__":
    serve()
