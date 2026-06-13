# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Bind-host trust policy for the Studio backend.

Stdlib only -- safe to import without the rest of the backend.

`_LOOPBACK_HOSTS` / `is_external_host` mirror the CLI's
`unsloth_cli/_tool_policy.py`: a loopback bind is the user's own machine, any
other address is network-reachable. The logic is duplicated rather than shared
because the backend is self-contained (see run.py: "can be moved to any
directory") and runs from a venv that may not have `unsloth_cli` on sys.path.
Keep the two in sync.
"""

from __future__ import annotations

import os

# Loopback aliases; any other bind address is treated as network-reachable.
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


def is_external_host(host: str) -> bool:
    """True when `host` is reachable from beyond loopback."""
    return host.lower() not in _LOOPBACK_HOSTS


def apply_stdio_mcp_loopback_default(host: str) -> None:
    """Default stdio MCP servers on when bound to loopback.

    A loopback bind is the user's own machine -- the same trust boundary the
    Tauri desktop app relies on (see main.py, which also binds 127.0.0.1 and
    setdefaults this var). `setdefault` preserves an explicit
    `UNSLOTH_STUDIO_ALLOW_STDIO_MCP=0` opt-out as well as the desktop app's own
    `=1`. Network and Colab (0.0.0.0) binds are left untouched, so the gate
    stays off there unless an operator opts in out-of-band.
    """
    if not is_external_host(host):
        os.environ.setdefault("UNSLOTH_STUDIO_ALLOW_STDIO_MCP", "1")
