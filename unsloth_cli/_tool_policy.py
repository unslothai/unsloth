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
) -> Optional[bool]:
    """Resolve the process-wide server-side tool OVERRIDE.

    An explicit --enable-tools/--disable-tools (`flag`) forces tools on/off for
    every request. With no flag the result is None: tools still default on for
    every bind (the backend installs that default in `_apply_cli_tool_policy`),
    but as a default rather than an override, so a request's own
    `enable_tools: false` is honored -- which is what the Unsloth UI sends with
    its tool pills off. `host`, `yes`, `silent`, `prompt` are kept for signature
    compatibility; no bind prompts."""
    return flag
