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


def _literal_ip_address(host: str):
    if not isinstance(host, str) or not host:
        return None
    literal = _normalized_ip(host)
    if literal is not None:
        return literal
    try:
        return ipaddress.IPv4Address(socket.inet_aton(host))
    except OSError:
        return None


def _resolved_host_ip_addresses(host: str):
    if not isinstance(host, str) or not host:
        return ()
    try:
        addresses = socket.getaddrinfo(host, 0, socket.AF_UNSPEC, socket.SOCK_STREAM)
    except OSError:
        return ()
    resolved = []
    for _family, _kind, _protocol, _name, sockaddr in addresses:
        try:
            parsed = ipaddress.ip_address(sockaddr[0])
        except (IndexError, ValueError):
            continue
        if parsed not in resolved:
            resolved.append(parsed)
    return tuple(resolved)


def _resolved_ip_addresses(host: str):
    literal = _literal_ip_address(host)
    if literal is not None:
        return (literal,)
    resolved = []
    for parsed in _resolved_host_ip_addresses(host):
        if isinstance(parsed, ipaddress.IPv6Address) and parsed.ipv4_mapped is not None:
            parsed = parsed.ipv4_mapped
        if parsed not in resolved:
            resolved.append(parsed)
    return tuple(resolved)


def wildcard_ip_versions(host: str) -> tuple[int, ...]:
    """IP versions for every unspecified address this host resolves to."""
    versions = {
        address.version for address in _resolved_ip_addresses(host) if address.is_unspecified
    }
    return tuple(version for version in (4, 6) if version in versions)


def resolved_bind_address_count(host: str) -> int:
    """Number of distinct socket addresses this host resolves to."""
    if _literal_ip_address(host) is not None:
        return 1
    if not isinstance(host, str) or not host:
        return 0
    try:
        addresses = socket.getaddrinfo(host, 0, socket.AF_UNSPEC, socket.SOCK_STREAM)
    except OSError:
        return 0
    endpoints = {
        (family, tuple(sockaddr))
        for family, _kind, _protocol, _name, sockaddr in addresses
        if sockaddr
    }
    return len(endpoints)


def is_wildcard_host(host: str) -> bool:
    """True when the host resolves to an unspecified bind address."""
    return bool(wildcard_ip_versions(host))


def normalize_wildcard_bind_host(host: str) -> str:
    """Return a safe canonical bind for an effective wildcard host."""
    if isinstance(host, str):
        try:
            parsed_literal = ipaddress.ip_address(host)
        except ValueError:
            pass
        else:
            if (
                isinstance(parsed_literal, ipaddress.IPv6Address)
                and parsed_literal.ipv4_mapped is not None
            ):
                return str(parsed_literal.ipv4_mapped)
    literal = _literal_ip_address(host)
    if literal is not None:
        if not literal.is_unspecified:
            return host
        return "::" if literal.version == 6 else "0.0.0.0"

    raw_addresses = _resolved_host_ip_addresses(host)
    addresses = []
    has_mapped_address = False
    for address in raw_addresses:
        if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
            address = address.ipv4_mapped
            has_mapped_address = True
        if address not in addresses:
            addresses.append(address)
    if has_mapped_address:
        if len(addresses) == 1:
            return str(addresses[0])
        raise ValueError(
            f"--host {host!r} resolves to ambiguous IPv4-mapped addresses; "
            "use an explicit bind address."
        )
    wildcard_versions = {address.version for address in addresses if address.is_unspecified}
    if not wildcard_versions:
        return host
    specific_versions = {address.version for address in addresses if not address.is_unspecified}
    if len(wildcard_versions) == 2 and not specific_versions:
        return host
    if specific_versions - wildcard_versions or (len(wildcard_versions) == 2 and specific_versions):
        raise ValueError(
            f"--host {host!r} mixes wildcard and specific address families; "
            "use an explicit bind address."
        )
    return "::" if 6 in wildcard_versions else "0.0.0.0"


def wildcard_loopback_host(host: str) -> Optional[str]:
    """The loopback address reachable through a wildcard bind."""
    versions = wildcard_ip_versions(host)
    if 4 in versions:
        return "127.0.0.1"
    return "::1" if 6 in versions else None


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
