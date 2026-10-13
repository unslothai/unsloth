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
    _CODEX_AUTO_ROUTE_TOOL_DESCRIPTION,
    _CODEX_ENV_KEY,
    _CODEX_ENV_UNSET,
    _CODEX_PROFILE,
    _CODEX_SUBAGENT_CONFIG_ENV,
    _CODEX_SUBAGENT_MCP_TOOL,
    _CODEX_SUBAGENT_TOOL_DESCRIPTION,
    _CODEX_SUBAGENT_ROUTING_INSTRUCTIONS,
    _SUBAGENT_INSTRUCTIONS,
    _merge_wslenv,
    _prefer_windows_cmd_sibling,
    _resolved_launch_command,
    _wsl_shim_env,
)

_CANCEL_POLL_SECONDS = 0.1
_SERVER_INSTRUCTIONS = _CODEX_SUBAGENT_ROUTING_INSTRUCTIONS


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


def _events(stdout: str) -> tuple[list[str], list[str], list[dict]]:
    messages = []
    errors = []
    commands = []
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
        if (
            event.get("type") == "item.completed"
            and isinstance(item, dict)
            and item.get("type") == "command_execution"
        ):
            commands.append(item)
        if event.get("type") in ("error", "turn.failed"):
            detail = event.get("message") or event.get("error")
            if isinstance(detail, dict):
                detail = detail.get("message") or json.dumps(detail)
            if detail:
                errors.append(str(detail))
    return messages, errors, commands


def _result_text(stdout: str, show_last_command: bool = False) -> str:
    messages, errors, commands = _events(stdout)
    if errors:
        raise RuntimeError(_bounded(errors[-1]))
    if not messages:
        raise RuntimeError("The local Codex agent returned no readable result.")
    if show_last_command and commands:
        # A small local model can report an edit that its own command never made; the raw output lets the parent check.
        last = commands[-1]
        output = str(last.get("aggregated_output") or "")[-2000:]
        return _bounded(
            f"{messages[-1]}\n\nLast command it ran: {last.get('command')}\n"
            f"Exit code: {last.get('exit_code')}\nOutput:\n{output}"
        )
    return _bounded(messages[-1])


def run_local_agent(
    task: str,
    cancel_event: threading.Event | None = None,
    show_last_command: bool = False,
) -> str:
    config = _config()
    executable = _prefer_windows_cmd_sibling(shutil.which("codex"))
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
        # Codex otherwise waits forever for a refused connection, so a stopped server hung the parent's call.
        "-c",
        "features.unbounded_connection_retries=false",
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
        "encoding": "utf-8",
        "errors": "replace",
    }
    if os.name == "nt":
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        popen_kwargs["start_new_session"] = True
    # cmd.exe splits CR/LF inside `%*`, so resolve npm shims rather than spawning the .cmd raw.
    launch_command = _resolved_launch_command(executable, command[1:], child_env)
    process = subprocess.Popen(launch_command, **popen_kwargs)
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
        # stderr is codex's own banner; the last error event says why the run stopped.
        errors = _events(stdout)[1]
        detail = (errors[-1] if errors else "") or stderr.strip() or stdout.strip()
        raise RuntimeError(
            _bounded(detail) or f"Local Codex exited with code {process.returncode}."
        )
    return _result_text(stdout, show_last_command)


def main() -> None:
    if len(sys.argv) > 1:
        os.environ[_CODEX_SUBAGENT_CONFIG_ENV] = sys.argv[1]
    try:
        auto_route = _config().get("auto_route") is True
    except RuntimeError:
        auto_route = False
    run_agent = run_local_agent
    if auto_route:

        def run_agent(task: str, cancel_event: threading.Event) -> str:
            try:
                return run_local_agent(task, cancel_event, show_last_command = True)
            except Exception as exc:
                if cancel_event.is_set():
                    raise
                # The parent's shell is off with --auto-route, so it must not pretend the step ran.
                raise RuntimeError(
                    f"{exc}\n\nThe local agent failed and your own shell is disabled in this "
                    "session, so this step was not done. Tell the user so plainly."
                ) from exc

    serve(
        run_agent = run_agent,
        tool_name = _CODEX_SUBAGENT_MCP_TOOL,
        tool_description = (
            _CODEX_AUTO_ROUTE_TOOL_DESCRIPTION if auto_route else _CODEX_SUBAGENT_TOOL_DESCRIPTION
        ),
        instructions = _SERVER_INSTRUCTIONS,
    )


if __name__ == "__main__":
    main()
