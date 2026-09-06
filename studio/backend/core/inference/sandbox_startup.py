# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Startup-time reporting and warm-up for the OS sandbox.

Two operator-facing pieces the predecessor sandbox PRs carried:

* A console notice when OS isolation is unavailable or unqualified. Tools default
  to ``os_isolation_required`` and fail closed, so on such a host every Python and
  Terminal tool call refuses. Without this, the first the operator hears of it is a
  refused tool call mid-conversation, so say it once at startup with the
  capability's own reason and remediation.
* A background warm probe. The first ``capability_snapshot()`` on a cold host
  scans the system roots and can take up to two minutes; running it at startup
  keeps that cost off the first tool call. The result is cached inside
  ``os_sandbox``, so this thread runs at most once per process.

Stdlib only, and every import of ``os_sandbox`` is deferred to call time: this
module is imported from the startup path, which must not pull the sandbox
machinery in eagerly.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from core.inference.os_sandbox import SandboxCapability

__all__ = [
    "format_sandbox_startup_notice",
    "print_sandbox_startup_notice",
    "start_sandbox_capability_warmup",
    "reset_sandbox_warmup_state",
]

# Guards the state below and makes the notice a single write, so a warm probe
# finishing mid-banner can never interleave lines into it.
_PRINT_LOCK = threading.Lock()
_WARMUP_LOCK = threading.Lock()
_warmup_thread: Optional[threading.Thread] = None
_warmup_completed = False

# How long the notice waits for an in-flight warm probe before giving up. The
# slow path is the system-root scan, which only a host where Bubblewrap already
# works pays; the unavailable cases this notice exists for (no bwrap binary,
# AppArmor refusing the user namespace) fail in well under a second.
_DEFAULT_NOTICE_WAIT_SECONDS = 20.0


def _capability_snapshot() -> "SandboxCapability":
    from core.inference.os_sandbox import capability_snapshot
    return capability_snapshot()


def format_sandbox_startup_notice(capability: "SandboxCapability") -> str:
    """The startup notice for ``capability``, or "" when OS isolation is healthy.

    ``remediation`` is emitted verbatim and unindented: it carries a copy-pasteable
    AppArmor profile whose own indentation is load bearing.
    """
    if capability.available and capability.qualified:
        return ""
    lines = [
        "",
        "  OS isolation for tool calls is unavailable on this machine.",
        "  Python and Terminal tool calls refuse by default until it is fixed; pick",
        "  Limited or Full in the Unsloth tool settings to run them anyway.",
        f"  Detected: {capability.backend} backend, {capability.environment} environment.",
        f"  Reason: {capability.reason}",
    ]
    lines.extend(capability.remediation.splitlines())
    return "\n".join(lines)


def print_sandbox_startup_notice(
    capability: "Optional[SandboxCapability]" = None,
    *,
    wait: "Optional[float]" = _DEFAULT_NOTICE_WAIT_SECONDS,
) -> None:
    """Print the notice for the current capability. Silent when the sandbox is fine.

    When a warm probe is still in flight this waits ``wait`` seconds for it rather
    than racing it, then gives up: the banner must not stall behind a system-root
    scan, and a tool call on such a host returns the same reason and remediation.
    Pass ``wait = None`` to probe synchronously.
    """
    if capability is None:
        thread = _warmup_thread
        if wait is not None and thread is not None and thread.is_alive():
            thread.join(wait)
            if thread.is_alive():
                return
        try:
            capability = _capability_snapshot()
        except Exception:  # noqa: BLE001 -- a startup notice never breaks startup
            return
    notice = format_sandbox_startup_notice(capability)
    if not notice:
        return
    with _PRINT_LOCK:
        print(notice, flush = True)


def _warm_sandbox_capability() -> None:
    global _warmup_completed
    try:
        _capability_snapshot()
    except Exception:  # noqa: BLE001 -- warming is best effort, never fatal
        pass
    finally:
        _warmup_completed = True


def start_sandbox_capability_warmup() -> Optional[threading.Thread]:
    """Probe the sandbox capability in a daemon thread. Returns immediately.

    At most one thread per process: the snapshot is cached inside ``os_sandbox``,
    so a second call is cheap and a repeated app startup (every test that builds
    the app) must not leak another thread.
    """
    global _warmup_thread
    with _WARMUP_LOCK:
        if _warmup_completed:
            return None
        if _warmup_thread is not None and _warmup_thread.is_alive():
            return _warmup_thread
        thread = threading.Thread(
            target = _warm_sandbox_capability,
            name = "unsloth-sandbox-capability-warmup",
            daemon = True,
        )
        _warmup_thread = thread
        thread.start()
        return thread


def reset_sandbox_warmup_state() -> None:
    """Forget the warm-up bookkeeping. For tests only."""
    global _warmup_thread, _warmup_completed
    with _WARMUP_LOCK:
        _warmup_thread = None
        _warmup_completed = False
