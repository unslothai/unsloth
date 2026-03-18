# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Tool definitions and executors for LLM tool calling.

Supports web search (DuckDuckGo), Python code execution, and terminal commands.
"""

import os

os.environ["UNSLOTH_IS_PRESENT"] = "1"

import subprocess
import sys
import tempfile
import threading

from unsloth_zoo.rl_environments import check_signal_escape_patterns

_EXEC_TIMEOUT = 300  # 5 minutes
_MAX_OUTPUT_CHARS = 8000  # truncate long output
_BASH_BLOCKED_WORDS = {"rm", "sudo", "dd", "chmod", "mkfs", "shutdown", "reboot"}

# Persistent working directory shared across tool calls within the server
# lifetime so files written by one call are visible to the next.
# Uses ~/studio_sandbox/ so files are accessible to the user.
_persistent_workdir: str | None = None


def _get_workdir() -> str:
    """Return (and lazily create) a persistent working directory for tool execution."""
    global _persistent_workdir
    if _persistent_workdir is None or not os.path.isdir(_persistent_workdir):
        home = os.path.expanduser("~")
        workdir = os.path.join(home, "studio_sandbox")
        os.makedirs(workdir, exist_ok = True)
        _persistent_workdir = workdir
    return _persistent_workdir


WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web for current information, recent events, or facts you are uncertain about.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query",
                }
            },
            "required": ["query"],
        },
    },
}

PYTHON_TOOL = {
    "type": "function",
    "function": {
        "name": "python",
        "description": "Execute Python code in a sandbox and return stdout/stderr.",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "The Python code to run",
                }
            },
            "required": ["code"],
        },
    },
}

TERMINAL_TOOL = {
    "type": "function",
    "function": {
        "name": "terminal",
        "description": "Execute a terminal command and return stdout/stderr.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The command to run",
                }
            },
            "required": ["command"],
        },
    },
}

ALL_TOOLS = [WEB_SEARCH_TOOL, PYTHON_TOOL, TERMINAL_TOOL]


def execute_tool(name: str, arguments: dict, cancel_event = None) -> str:
    """Execute a tool by name with the given arguments. Returns result as a string."""
    if name == "web_search":
        return _web_search(arguments.get("query", ""))
    if name == "python":
        return _python_exec(arguments.get("code", ""), cancel_event)
    if name == "terminal":
        return _bash_exec(arguments.get("command", ""), cancel_event)
    return f"Unknown tool: {name}"


def _web_search(query: str, max_results: int = 5) -> str:
    """Search the web using DuckDuckGo and return formatted results."""
    if not query.strip():
        return "No query provided."
    try:
        from ddgs import DDGS

        results = DDGS().text(query, max_results = max_results)
        if not results:
            return "No results found."
        parts = []
        for r in results:
            parts.append(
                f"Title: {r.get('title', '')}\n"
                f"URL: {r.get('href', '')}\n"
                f"Snippet: {r.get('body', '')}"
            )
        return "\n\n---\n\n".join(parts)
    except Exception as e:
        return f"Search failed: {e}"


def _check_code_safety(code: str) -> str | None:
    """Validate code safety using unsloth_zoo.

    Returns an error message string if the code is unsafe, or None if OK.
    """
    # Check for signal/timeout escape patterns
    safe, info = check_signal_escape_patterns(code)
    if not safe:
        reasons = [
            item.get("description", "") for item in info.get("signal_tampering", [])
        ]
        return (
            f"Error: unsafe code detected ({'; '.join(reasons)}). "
            f"Please remove signal manipulation from your code."
        )

    return None


def _cancel_watcher(proc, cancel_event, poll_interval = 0.2):
    """Daemon thread that kills a process when cancel_event is set."""
    while proc.poll() is None:
        if cancel_event is not None and cancel_event.is_set():
            proc.kill()
            return
        cancel_event.wait(poll_interval) if cancel_event else None


def _truncate(text: str, limit: int = _MAX_OUTPUT_CHARS) -> str:
    if len(text) > limit:
        return text[:limit] + f"\n\n... (truncated, {len(text)} chars total)"
    return text


def _python_exec(code: str, cancel_event = None) -> str:
    """Execute Python code in a subprocess sandbox."""
    if not code or not code.strip():
        return "No code provided."

    # Validate imports and code safety
    error = _check_code_safety(code)
    if error:
        return error

    tmp_path = None
    try:
        fd, tmp_path = tempfile.mkstemp(suffix = ".py", prefix = "studio_exec_")
        with os.fdopen(fd, "w") as f:
            f.write(code)

        proc = subprocess.Popen(
            [sys.executable, tmp_path],
            stdout = subprocess.PIPE,
            stderr = subprocess.STDOUT,
            text = True,
            cwd = _get_workdir(),
        )

        # Spawn cancel watcher if we have a cancel event
        if cancel_event is not None:
            watcher = threading.Thread(
                target = _cancel_watcher, args = (proc, cancel_event), daemon = True
            )
            watcher.start()

        try:
            output, _ = proc.communicate(timeout = _EXEC_TIMEOUT)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()
            return _truncate("Execution timed out after 5 minutes.")

        if cancel_event is not None and cancel_event.is_set():
            return "Execution cancelled."

        result = output or ""
        if proc.returncode != 0:
            result = f"Exit code {proc.returncode}:\n{result}"
        return _truncate(result) if result.strip() else "(no output)"

    except Exception as e:
        return f"Execution error: {e}"
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def _bash_exec(command: str, cancel_event = None) -> str:
    """Execute a bash command in a subprocess sandbox."""
    if not command or not command.strip():
        return "No command provided."

    # Block dangerous commands
    tokens = set(command.lower().split())
    blocked = tokens & _BASH_BLOCKED_WORDS
    if blocked:
        return f"Blocked command(s) for safety: {', '.join(sorted(blocked))}"

    try:
        workdir = _get_workdir()
        proc = subprocess.Popen(
            ["bash", "-c", command],
            stdout = subprocess.PIPE,
            stderr = subprocess.STDOUT,
            text = True,
            cwd = workdir,
        )

        if cancel_event is not None:
            watcher = threading.Thread(
                target = _cancel_watcher, args = (proc, cancel_event), daemon = True
            )
            watcher.start()

        try:
            output, _ = proc.communicate(timeout = _EXEC_TIMEOUT)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()
            return _truncate("Execution timed out after 5 minutes.")

        if cancel_event is not None and cancel_event.is_set():
            return "Execution cancelled."

        result = output or ""
        if proc.returncode != 0:
            result = f"Exit code {proc.returncode}:\n{result}"
        return _truncate(result) if result.strip() else "(no output)"

    except Exception as e:
        return f"Execution error: {e}"
