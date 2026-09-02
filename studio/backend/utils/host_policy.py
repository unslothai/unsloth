# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Bind-host trust policy for the Unsloth backend.

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
import socket

# Loopback aliases; any other bind address is treated as network-reachable. Only
# the exact aliases the rest of the stack assumes for loopback (health checks,
# banner URLs, run.py all hard-code 127.0.0.1), so other 127.0.0.0/8 addresses
# are deliberately left out -- they are not supported launch hosts.
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})

# Whether a loopback launch in THIS process auto-enabled the gate. run_server
# normally runs once per process, but if it is reused with a different host
# (embedders, tests) a stale loopback default must not carry into a later
# public bind, so we only ever take back a value we set ourselves.
_auto_enabled = False
_remote_connector_active = False
_lan_connector_active = False


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


def wildcard_loopback_host(host: str) -> "str | None":
    """The loopback address reachable through a wildcard bind."""
    versions = wildcard_ip_versions(host)
    if 4 in versions:
        return "127.0.0.1"
    return "::1" if 6 in versions else None


# Tauri desktop webview origins. api-only serving (the desktop app calling a
# local backend) locks CORS to these.
_TAURI_CORS_ORIGINS = (
    "tauri://localhost",  # Linux/macOS Tauri webview
    "http://tauri.localhost",  # Windows Tauri webview
    "http://localhost",  # dev fallback
    "http://localhost:5173",  # Tauri dev/Vite
    "http://127.0.0.1:5173",  # Tauri dev/Vite fallback
)


def cors_origins_for_mode(*, api_only: bool, secure: bool) -> list[str]:
    """Allowed CORS origins. Default is any-origin (["*"]); api-only locks down
    to the Tauri desktop app, except in secure mode where the API is published
    over Cloudflare and must stay reachable from remote browser origins."""
    if api_only and not secure:
        return list(_TAURI_CORS_ORIGINS)
    return ["*"]


def apply_stdio_mcp_loopback_default(host: str, *, is_colab: bool = False) -> None:
    """Default stdio MCP servers on when bound to loopback.

    A loopback bind is the user's own machine -- the same trust boundary the
    Tauri desktop app relies on (see main.py, which uses this same helper).
    Colab is excluded: even its loopback is a hosted VM
    reachable through Colab's proxy, so it stays off unless opted in. An explicit
    operator value wins: a pre-set `UNSLOTH_STUDIO_ALLOW_STDIO_MCP=0`
    force-disables and `=1` opts in, including on a network bind. We only ever
    set or clear a default we applied ourselves, so reusing run_server with a
    public host after a loopback one does not leave the gate on.
    """
    global _auto_enabled
    current = os.environ.get("UNSLOTH_STUDIO_ALLOW_STDIO_MCP")
    # If our prior auto-default was changed out from under us (in-process reuse),
    # relinquish ownership: an explicit =0 is then honored below as a sticky
    # force-disable, while a cleared var falls back to the host default like a
    # fresh process.
    if _auto_enabled and current != "1":
        _auto_enabled = False
    # An explicit operator value is one we did not set; never touch it.
    if current is not None and not _auto_enabled:
        return
    if is_colab or is_external_host(host):
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


def set_remote_connector_active(active: bool) -> None:
    """Publish whether a connector may carry requests from beyond loopback."""
    global _remote_connector_active
    _remote_connector_active = bool(active)


def set_lan_connector_active(active: bool) -> None:
    """Publish whether a runtime LAN listener is serving beyond loopback."""
    global _lan_connector_active
    _lan_connector_active = bool(active)


def tunnel_connector_active() -> bool:
    """True while a tunnel is publishing this server past the local network."""
    return _remote_connector_active


def lan_connector_active() -> bool:
    """True while a runtime LAN listener is serving the local network."""
    return _lan_connector_active


def remote_connector_active() -> bool:
    """True while any connector can carry a request from beyond loopback."""
    return _remote_connector_active or _lan_connector_active


def _reset_loopback_default_state() -> None:
    """Test hook: forget runtime trust state applied earlier in this process."""
    global _auto_enabled, _remote_connector_active, _lan_connector_active
    _auto_enabled = False
    _remote_connector_active = False
    _lan_connector_active = False
