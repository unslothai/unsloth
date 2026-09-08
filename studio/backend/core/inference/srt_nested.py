# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""Trusted eligibility controls and separate consent custody for nested SRT."""

from dataclasses import dataclass
import os
import subprocess
import sys

from .srt_diagnostics import ProbeReason, environment_context
from .tool_isolation import LimitedGrantError, LimitedGrantStore

NESTED_PROFILE = "srt-0.0.75-nested-cap-drop-v1"
NESTED_DISCLOSURE = (
    "Applies only to Python and Terminal tool calls. Keeps file and network restrictions, "
    "but shares the container's process information and relies partly on its isolation."
)


def container_context():
    return sys.platform == "linux" and (
        environment_context() == "colab"
        or os.path.isfile("/.dockerenv")
        or os.path.isfile("/run/.containerenv")
    )


def _proc_control(*, nested):
    # Only this proc mount choice differs. Both controls drop capabilities and
    # isolate network and PID namespaces; neither executes a tool command.
    return subprocess.run(
        [
            "/usr/bin/bwrap",
            "--die-with-parent",
            "--ro-bind",
            "/",
            "/",
            "--dev",
            "/dev",
            "--unshare-user",
            "--unshare-pid",
            "--unshare-net",
            "--cap-drop",
            "ALL",
            *(["--bind", "/proc", "/proc"] if nested else ["--proc", "/proc"]),
            "--",
            "/bin/true",
        ],
        stdin = subprocess.DEVNULL,
        stdout = subprocess.DEVNULL,
        stderr = subprocess.DEVNULL,
        env = {"PATH": os.defpath},
        timeout = 3,
        close_fds = True,
    ).returncode


def eligibility(standard_result):
    """Offer only after a typed standard failure and an exact differential control.

    This determines whether trying the separately consented variant is relevant.
    It is never execution admission or native qualification.
    """
    available, reason = standard_result
    if available or not isinstance(reason, ProbeReason) or reason.code != "operation_unsupported":
        return False
    if not container_context() or not os.access("/usr/bin/bwrap", os.X_OK):
        return False
    try:
        # A recovered normal control, missing prerequisite or timeout offers no
        # alternate mode. Stderr, browser labels and client fields are unused.
        return _proc_control(nested = False) == 1 and _proc_control(nested = True) == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


@dataclass(frozen = True)
class NestedGrant:
    token: str
    expires_at: float
    probe_generation: str
    mode: str = "container_isolation"


class NestedGrantStore:
    """Separate token namespace; a Limited grant cannot authorize this variant."""

    def __init__(
        self,
        *,
        ttl_seconds = 300,
        max_entries = 1024,
    ):
        self._store = LimitedGrantStore(ttl_seconds = ttl_seconds, max_entries = max_entries)

    @staticmethod
    def _generation(generation):
        if not isinstance(generation, str) or not generation:
            raise LimitedGrantError(
                "INVALID_PROBE_GENERATION", "A current nested probe generation is required."
            )
        return NESTED_PROFILE + ":" + generation

    def issue(self, *, current_subject, tool_ui_session_id, probe_generation):
        issued = self._store.issue(
            current_subject = current_subject,
            tool_ui_session_id = tool_ui_session_id,
            probe_generation = self._generation(probe_generation),
        )
        return NestedGrant(issued.token, issued.expires_at, probe_generation)

    def validate(self, token, *, current_subject, tool_ui_session_id, probe_generation):
        self._store.validate(
            token,
            current_subject = current_subject,
            tool_ui_session_id = tool_ui_session_id,
            probe_generation = self._generation(probe_generation),
            requested_mode = "limited",
        )


NESTED_GRANTS = NestedGrantStore()
