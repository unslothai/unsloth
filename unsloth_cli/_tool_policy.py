# Copyright 2025-present the Unsloth AI Inc. team. All rights reserved.

"""Pure resolver for `unsloth studio [run] --enable-tools/--disable-tools`.

Kept as a standalone module so the truth table can be unit-tested
without spinning up Typer or the studio venv.
"""

import ipaddress
import socket
from typing import Callable, Optional

import typer

# loopback aliases mirror the self-contained backend bind policy
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


def is_external_host(host: str) -> bool:
    """True when `host` is reachable from beyond loopback."""
    return host.lower() not in _LOOPBACK_HOSTS


def is_wildcard_host(host: str) -> bool:
    """True when the host resolves to an unspecified bind address."""
    if not isinstance(host, str):
        return False
    if host == "":
        return True
    try:
        literal = ipaddress.ip_address(host)
    except ValueError:
        literal = None
    if literal is not None:
        return literal.is_unspecified
    try:
        addresses = socket.getaddrinfo(host, 0, socket.AF_UNSPEC, socket.SOCK_STREAM)
    except OSError:
        return False
    for _family, _kind, _protocol, _name, sockaddr in addresses:
        try:
            if ipaddress.ip_address(sockaddr[0]).is_unspecified:
                return True
        except (IndexError, ValueError):
            continue
    return False


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
