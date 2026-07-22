# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Small stdio MCP bridge from cloud Codex to an explicit local Codex child."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

from unsloth_cli.claude_subagent_mcp import _bounded, _stop_child, serve
from unsloth_cli.commands.start import (
    _CODEX_ENV_KEY,
    _CODEX_ENV_UNSET,
    _CODEX_PROFILE,
    _CODEX_SUBAGENT_CONFIG_ENV,
    _CODEX_SUBAGENT_MCP_TOOL,
    _CODEX_SUBAGENT_TOOL_DESCRIPTION,
    _SUBAGENT_INSTRUCTIONS,
    _merge_wslenv,
    _wsl_shim_env,
)

_CANCEL_POLL_SECONDS = 0.1
_SERVER_INSTRUCTIONS = (
    "When the user asks to spawn an Unsloth agent or a local agent, call "
    "spawn_local_agent with the complete task. Do not use Codex's built-in "
    "spawn_agent for those requests. Use built-in agents for other subagent requests."
)


def _config() -> dict:
    path = os.environ.get(_CODEX_SUBAGENT_CONFIG_ENV, "").strip()
    if not path:
        raise RuntimeError(f"Missing {_CODEX_SUBAGENT_CONFIG_ENV}.")
    try:
        config = json.loads(Path(path).read_text(encoding = "utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError("Could not read the local Codex agent configuration.") from exc
    if not isinstance(config, dict):
        raise RuntimeError("The local Codex agent configuration must be an object.")
    for name in ("api_key", "codex_home"):
        if not isinstance(config.get(name), str) or not config[name].strip():
            raise RuntimeError(f"The local Codex agent configuration is missing {name}.")
    return config


def _result_text(stdout: str) -> str:
    messages = []
    errors = []
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if not isinstance(event, dict):
            continue
        item = event.get("item")
        if (
            event.get("type") == "item.completed"
            and isinstance(item, dict)
            and item.get("type") == "agent_message"
            and isinstance(item.get("text"), str)
            and item["text"].strip()
        ):
            messages.append(item["text"].strip())
        if event.get("type") in ("error", "turn.failed"):
            detail = event.get("message") or event.get("error")
            if isinstance(detail, dict):
                detail = detail.get("message") or json.dumps(detail)
            if detail:
                errors.append(str(detail))
    if errors:
        raise RuntimeError(_bounded(errors[-1]))
    if messages:
        return _bounded(messages[-1])
    raise RuntimeError("The local Codex agent returned no readable result.")


def run_local_agent(task: str, cancel_event: threading.Event | None = None) -> str:
    config = _config()
    executable = shutil.which("codex")
    if executable is None:
        raise RuntimeError("`codex` is not installed or is not on PATH.")
    cancel_event = cancel_event or threading.Event()
    if cancel_event.is_set():
        raise RuntimeError("The local Codex agent was cancelled.")

    permissions = (
        ["--dangerously-bypass-approvals-and-sandbox"]
        if config.get("bypass_permissions") is True
        else ["--sandbox", "workspace-write", "--ask-for-approval", "never"]
    )
    command = [
        "codex",
        "--oss",
        "--profile",
        _CODEX_PROFILE,
        *permissions,
        "exec",
        "--ephemeral",
        "--json",
        "--skip-git-repo-check",
        f"{_SUBAGENT_INSTRUCTIONS}\n\nTask: {task}",
    ]
    local_env = {
        _CODEX_ENV_KEY: config["api_key"],
        "CODEX_HOME": config["codex_home"],
        "CODEX_SQLITE_HOME": config["codex_home"],
    }
    bridged, wsl_names = _wsl_shim_env(command, local_env, _CODEX_ENV_UNSET)
    child_env = dict(os.environ)
    if wsl_names:
        bridged = {**bridged, "PWD": os.getcwd()}
        child_env["WSLENV"] = _merge_wslenv(child_env.get("WSLENV", ""), wsl_names)
        for name in _CODEX_ENV_UNSET:
            child_env[name] = ""
    else:
        for name in _CODEX_ENV_UNSET:
            child_env.pop(name, None)
    child_env.update(bridged)
    popen_kwargs: dict[str, Any] = {
        "cwd": os.getcwd(),
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
    process = subprocess.Popen([executable, *command[1:]], **popen_kwargs)
    try:
        while True:
            try:
                stdout, stderr = process.communicate(timeout = _CANCEL_POLL_SECONDS)
                break
            except subprocess.TimeoutExpired:
                if cancel_event.is_set():
                    _stop_child(process)
                    raise RuntimeError("The local Codex agent was cancelled.")
    except BaseException:
        if process.poll() is None:
            _stop_child(process)
        raise
    if process.returncode != 0:
        detail = stderr.strip() or stdout.strip()
        raise RuntimeError(
            _bounded(detail) or f"Local Codex exited with code {process.returncode}."
        )
    return _result_text(stdout)


def main() -> None:
    if len(sys.argv) > 1:
        os.environ[_CODEX_SUBAGENT_CONFIG_ENV] = sys.argv[1]
    serve(
        run_agent = run_local_agent,
        tool_name = _CODEX_SUBAGENT_MCP_TOOL,
        tool_description = _CODEX_SUBAGENT_TOOL_DESCRIPTION,
        instructions = _SERVER_INSTRUCTIONS,
    )


if __name__ == "__main__":
    main()
