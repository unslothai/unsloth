# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Startup-time reporting and warm-up for the OS sandbox.

Two operator-facing pieces:

* A console notice when OS isolation is unavailable. Tools default to ``auto``,
  which runs them exactly as they ran before this landed, so nothing is broken
  and nothing is refused -- but the operator is entitled to know that the OS
  boundary is not there, and to be told once at startup rather than discovering
  it in a tool result badge mid-conversation. ``required`` is the mode that does
  refuse on such a host, so the notice names it.
* A background warm probe. The first ``capability_snapshot()`` launches a real
  sandbox and waits for it; doing that at startup keeps the cost off the first
  tool call. The result is cached inside ``os_sandbox``, so this thread runs at
  most once per process.

Stdlib only, and every import of ``os_sandbox`` is deferred to call time: this
module is imported from the startup path, which must not pull the sandbox
machinery -- or the probe it triggers -- in eagerly.
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
# probe is hard-bounded at ``sandbox_probe.PROBE_TIMEOUT_SECONDS``; the cases
# this notice exists for (no bwrap binary, AppArmor refusing the user namespace)
# fail in well under a second, so this only ever bites on a wedged helper, which
# is exactly when the banner must not block startup.
_DEFAULT_NOTICE_WAIT_SECONDS = 20.0


def _capability_snapshot() -> "SandboxCapability":
    from core.inference.os_sandbox import capability_snapshot
    return capability_snapshot()


def format_sandbox_startup_notice(capability: "SandboxCapability") -> str:
    """The startup notice for ``capability``, or "" when OS isolation is healthy.

    ``remediation`` is emitted verbatim and unindented: it carries a
    copy-pasteable AppArmor profile path whose own spelling is load bearing, and
    an operator pasting a re-wrapped line into a shell gets nothing.
    """
    if capability.available:
        return ""
    lines = [
        "",
        "  OS isolation for tool calls is unavailable on this machine.",
        "  Python and Terminal still run, with Studio's software safeguards only;",
        "  pick Required in the Unsloth tool settings to refuse them instead.",
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
    than racing it, then gives up: the banner must not stall behind a wedged
    sandbox helper, and a tool call on such a host reports the same reason and
    remediation in its own result. Pass ``wait = None`` to probe synchronously.
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
