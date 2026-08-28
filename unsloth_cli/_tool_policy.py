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


def _normalized_ip(address: str):
    try:
        parsed = ipaddress.ip_address(address)
    except ValueError:
        return None
    if isinstance(parsed, ipaddress.IPv6Address) and parsed.ipv4_mapped is not None:
        parsed = parsed.ipv4_mapped
    return parsed


def _unspecified_address(host: str):
    if not isinstance(host, str) or not host:
        return None
    literal = _normalized_ip(host)
    if literal is not None:
        return literal if literal.is_unspecified else None
    try:
        legacy_ipv4 = ipaddress.IPv4Address(socket.inet_aton(host))
    except OSError:
        pass
    else:
        return legacy_ipv4 if legacy_ipv4.is_unspecified else None
    try:
        addresses = socket.getaddrinfo(host, 0, socket.AF_UNSPEC, socket.SOCK_STREAM)
    except OSError:
        return None
    ipv6_unspecified = None
    for _family, _kind, _protocol, _name, sockaddr in addresses:
        try:
            parsed = _normalized_ip(sockaddr[0])
        except IndexError:
            continue
        if parsed is not None and not parsed.is_unspecified:
            continue
        if isinstance(parsed, ipaddress.IPv4Address):
            return parsed
        if parsed is not None:
            ipv6_unspecified = parsed
    return ipv6_unspecified


def is_wildcard_host(host: str) -> bool:
    """True when the host resolves to an unspecified bind address."""
    return _unspecified_address(host) is not None


def wildcard_loopback_host(host: str) -> Optional[str]:
    """The loopback address reachable through a wildcard bind."""
    address = _unspecified_address(host)
    if isinstance(address, ipaddress.IPv6Address):
        return "::1"
    return "127.0.0.1" if address is not None else None


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
