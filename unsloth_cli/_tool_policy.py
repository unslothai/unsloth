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
    """Return the resolved server-side tool policy.

    Tools default ON for every bind: loopback, the `--secure` authenticated
    Cloudflare HTTPS tunnel, and a raw network bind alike. `--secure` is a
    loopback bind fronted by an authenticated tunnel, not a raw public port, so
    it must not strip tools; and for a raw bind the operator owns network
    security. An explicit `--enable-tools/--disable-tools` flag forces the
    policy on or off for every request.

    There is no longer a network-exposure confirmation prompt -- a default-on
    policy that prompted on every network launch would be pure friction -- so
    `yes`/`silent`/`prompt` are accepted for backward compatibility but no
    longer affect the result.

    Args:
        host: The bind address (retained for signature compatibility).
        flag: Tri-state from `--enable-tools/--disable-tools` (None if neither passed).
        yes: True if `--yes/-y` was passed (no longer consulted).
        silent: True if `--silent/-q` was passed (no longer consulted).
        prompt: Confirmation callable (no longer consulted).
    """
    return True if flag is None else flag
