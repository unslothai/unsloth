# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Small stdio MCP bridge from cloud Claude Code to a local Claude Code child."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
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


def run_local_agent(task: str) -> str:
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
    completed = subprocess.run(
        [executable, *command[1:]],
        cwd = os.environ.get("CLAUDE_PROJECT_DIR") or os.getcwd(),
        env = child_env,
        capture_output = True,
        text = True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(_bounded(detail) or f"Local Claude exited with code {completed.returncode}.")
    return _result_text(completed.stdout)


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


def serve(stdin: Any = sys.stdin, stdout: Any = sys.stdout) -> None:
    for line in stdin:
        try:
            request = json.loads(line)
            response = _response(request) if isinstance(request, dict) else None
        except Exception as exc:
            response = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32603, "message": str(exc)},
            }
        if response is not None:
            stdout.write(json.dumps(response, separators = (",", ":")) + "\n")
            stdout.flush()


if __name__ == "__main__":
    serve()
