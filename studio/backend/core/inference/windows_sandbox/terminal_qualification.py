# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Per-selected-shell qualification outcome; compatibility never grants isolation.

A shell that cannot run its required batch/child controls remains unavailable.
Successful compatibility is necessary, but still requires the separate complete
filesystem, network, IPC, inherited-handle and descendant enforcement matrix.
"""

from dataclasses import dataclass
import os

from .profiles import WindowsRuntimeError
from .terminal_probe import run_terminal_probe

TERMINAL_PROFILE_ID = "windows-lpac-terminal-v2"


@dataclass(frozen = True)
class TerminalQualification:
    selected_executable: str
    qualified: bool
    profile_id: str
    reason: str
    failure_code: str
    checks: tuple[str, ...] = ()
    runtime_digest: str = ""
    content_digest: str = ""
    transient: bool = False
    limitations: tuple[str, ...] = ()


def qualify_terminal_runtime(
    selected_executable,
    *,
    store_root,
    timeout = 30,
    cancel = None,
):
    """Measure the selected shell without retrying a failed payload elsewhere."""
    selected = os.fspath(selected_executable)
    try:
        measured = run_terminal_probe(
            selected,
            store_root = store_root,
            timeout = timeout,
            cancel = cancel,
        )
    except WindowsRuntimeError as error:
        code = error.code
        return TerminalQualification(
            selected,
            False,
            TERMINAL_PROFILE_ID,
            str(error),
            code,
            transient = code
            in {
                "WINDOWS_SANDBOX_CANCELLED",
                "WINDOWS_SANDBOX_STARTUP_TIMEOUT",
                "WINDOWS_SANDBOX_SCAN_LIMIT",
                "WINDOWS_SANDBOX_CLEANUP_FAILED",
            },
            limitations = ("terminal_runtime_unqualified",),
        )
    return TerminalQualification(
        measured.selected_executable,
        False,
        TERMINAL_PROFILE_ID,
        "Terminal compatibility passed, but complete native enforcement qualification is required.",
        "WINDOWS_SANDBOX_QUALIFICATION_INCOMPLETE",
        checks = measured.checks,
        runtime_digest = measured.runtime_digest,
        content_digest = measured.content_digest,
        limitations = ("terminal_enforcement_unqualified",),
    )
