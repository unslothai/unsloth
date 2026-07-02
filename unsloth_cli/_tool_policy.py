# Copyright 2025-present the Unsloth AI Inc. team. All rights reserved.

"""Pure resolver for `unsloth studio [run] --enable-tools/--disable-tools`.

Kept as a standalone module so the truth table can be unit-tested
without spinning up Typer or the studio venv.
"""

from typing import Callable, Optional

import typer

# Loopback aliases; any other bind address is treated as network-reachable.
# Mirrored in studio/backend/utils/host_policy.py (kept separate because the
# backend is self-contained); keep the two in sync.
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


def is_external_host(host: str) -> bool:
    """True when `host` is reachable from beyond loopback."""
    return host.lower() not in _LOOPBACK_HOSTS


def resolve_tool_policy(
    host: str,
    flag: Optional[bool],
    yes: bool,
    silent: bool,
    prompt: Callable[[str], bool] = typer.confirm,
) -> bool:
    """Resolve the server-side tool policy. Tools default on for every bind;
    an explicit --enable-tools/--disable-tools (`flag`) forces on/off. `host`,
    `yes`, `silent`, `prompt` are kept for signature compatibility and no longer
    affect the result (network binds no longer prompt)."""
    return True if flag is None else flag
