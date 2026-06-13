# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Bind-host trust policy for the Studio backend.

Stdlib only -- safe to import without the rest of the backend.

`is_external_host` mirrors the CLI's `unsloth_cli/_tool_policy.py`: a loopback
bind is the user's own machine, any other address is network-reachable. The
logic is duplicated rather than shared because the backend is self-contained
(see run.py: "can be moved to any directory") and runs from a venv that may not
have `unsloth_cli` on sys.path. Keep the two in sync.
"""

from __future__ import annotations

import ipaddress
import os

# Hostname aliases for loopback; IP literals are classified by ipaddress below.
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})

# Whether a loopback launch in THIS process auto-enabled the gate. run_server
# normally runs once per process, but if it is reused with a different host
# (embedders, tests) a stale loopback default must not carry into a later
# public bind, so we only ever take back a value we set ourselves.
_auto_enabled = False


def is_external_host(host: str) -> bool:
    """True when `host` is reachable from beyond loopback.

    Covers the whole 127.0.0.0/8 range and ::1 via ipaddress, plus the
    `localhost` hostname which is not an IP literal.
    """
    host = host.lower()
    if host in _LOOPBACK_HOSTS:
        return False
    try:
        return not ipaddress.ip_address(host).is_loopback
    except ValueError:
        # Not an IP literal (an unresolved hostname); treat as network-reachable.
        return True


def apply_stdio_mcp_loopback_default(host: str) -> None:
    """Default stdio MCP servers on when bound to loopback.

    A loopback bind is the user's own machine -- the same trust boundary the
    Tauri desktop app relies on (see main.py, which also binds 127.0.0.1 and
    setdefaults this var). An explicit operator value wins: a pre-set
    `UNSLOTH_STUDIO_ALLOW_STDIO_MCP=0` force-disables and `=1` opts in, including
    on a network bind. We only ever set or clear a default we applied ourselves,
    so reusing run_server with a public host after a loopback one does not leave
    the gate on.
    """
    global _auto_enabled
    current = os.environ.get("UNSLOTH_STUDIO_ALLOW_STDIO_MCP")
    # An explicit operator value is one we did not set; never touch it.
    if current is not None and not _auto_enabled:
        return
    if is_external_host(host):
        if _auto_enabled:
            os.environ.pop("UNSLOTH_STUDIO_ALLOW_STDIO_MCP", None)
            _auto_enabled = False
    else:
        os.environ["UNSLOTH_STUDIO_ALLOW_STDIO_MCP"] = "1"
        _auto_enabled = True


def loopback_default_active() -> bool:
    """True when stdio MCP is on only because a loopback bind auto-enabled it,
    rather than an explicit operator opt-in. Lets the gate tell the two apart."""
    return _auto_enabled


def _reset_loopback_default_state() -> None:
    """Test hook: forget any auto-enable applied earlier in this process."""
    global _auto_enabled
    _auto_enabled = False
