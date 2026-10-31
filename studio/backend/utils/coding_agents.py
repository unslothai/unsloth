# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Detects which unsloth start <agent> CLIs are on PATH so the UI can default to a runnable one"""

import os
import shutil
import subprocess
from typing import Optional


_DEEPSEEK_HARNESS_HELP_MARKER = "DeepSeek Harness"
_DEEPSEEK_HARNESS_FILE_MARKERS = (
    b"deepseek harness",
    b"@deepseek-ai/dsh",
    b"@deepseek-ai+dsh",
)

# Keep in sync with the `unsloth start <agent>` subcommands defined in
# unsloth_cli/commands/start.py, in the order the Settings pickers list them. Each
# entry is the executable that subcommand needs on PATH (see _EXECUTABLES).
CODING_AGENTS: tuple[str, ...] = (
    "claude",
    "codex",
    "vscode",
    "dsh",
    "opencode",
    "hermes",
    "openclaw",
    "pi",
    "vibe",
)

# `unsloth start vscode` edits VS Code's settings; `code` is the launcher an install puts on PATH.
_EXECUTABLES = {"vscode": "code"}


def is_deepseek_harness_executable(executable: str, *, allow_execution: bool = True) -> bool:
    """Return whether ``executable`` identifies itself as DeepSeek Harness."""
    try:
        with open(executable, "rb") as launcher:
            contents = launcher.read(256 * 1024).lower()
    except OSError:
        contents = b""
    if any(marker in contents for marker in _DEEPSEEK_HARNESS_FILE_MARKERS):
        return True
    if not allow_execution:
        return False
    try:
        result = subprocess.run(
            [executable, "--help"],
            check = False,
            capture_output = True,
            text = True,
            encoding = "utf-8",
            errors = "replace",
            timeout = 5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    output = (result.stdout or "") + (result.stderr or "")
    return result.returncode == 0 and _DEEPSEEK_HARNESS_HELP_MARKER in output


def deepseek_harness_executables_on_path(path: Optional[str] = None) -> list[str]:
    """Return every distinct ``dsh`` executable in PATH order."""
    if path is None:
        path = os.environ.get("PATH")
    if path is None:
        path = os.defpath
    executables = []
    seen = set()
    for directory in path.split(os.pathsep):
        try:
            executable = shutil.which("dsh", path = directory)
        except OSError:
            continue
        if executable is None:
            continue
        key = os.path.normcase(os.path.abspath(executable))
        if key in seen:
            continue
        seen.add(key)
        executables.append(executable)
    return executables


def _is_on_path(agent: str) -> bool:
    # PATH lookups can still raise; this is advisory, so a failure reads as not installed, not an error.
    try:
        executable = shutil.which(_EXECUTABLES.get(agent, agent))
        if executable is None:
            return False
        if agent == "dsh":
            # Must not execute an arbitrary same-named program when the panel opens.
            if is_deepseek_harness_executable(executable, allow_execution = False):
                return True
            return any(
                is_deepseek_harness_executable(candidate, allow_execution = False)
                for candidate in deepseek_harness_executables_on_path()
                if os.path.normcase(os.path.abspath(candidate))
                != os.path.normcase(os.path.abspath(executable))
            )
        return True
    except OSError:
        return False


def detect_installed_coding_agents() -> list[str]:
    """Ordered as CODING_AGENTS, so the first installed entry is the preferred default."""
    return [agent for agent in CODING_AGENTS if _is_on_path(agent)]
