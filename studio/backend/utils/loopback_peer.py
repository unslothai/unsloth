# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Which OS user opened a loopback TCP connection, read from /proc/net/tcp{,6} on Linux.

A loopback port is not an identity: on a shared host any OS user can connect to it. Used only to
keep the first-run password off pages fetched by a different local account. Unknown (not Linux,
no /proc, entry not found) returns None and callers keep today's behaviour."""

from __future__ import annotations

import ipaddress
import os
import sys
from typing import Optional

_PROC_TABLES = ("/proc/net/tcp", "/proc/net/tcp6")
# TIME_WAIT, CLOSE, LISTEN: orphaned (UID 0) or not a connection; no other state can hide a socket
# that can still read the response, so a half-closed client is still found under its own UID.
_TCP_NOT_A_LIVE_CLIENT = frozenset(("06", "07", "0A"))


def _ip(value) -> Optional[ipaddress._BaseAddress]:
    try:
        addr = ipaddress.ip_address(value)
    except ValueError:
        return None
    mapped = getattr(addr, "ipv4_mapped", None)
    return mapped or addr


def _endpoint(field: str):
    """`0100007F:1F90` (tcp) or 32-hex-digit (tcp6) address:port, kernel byte order, as (ip, port)."""
    try:
        hexaddr, hexport = field.split(":")
        raw = bytes.fromhex(hexaddr)
        if len(raw) not in (4, 16):
            return None
        raw = b"".join(raw[i : i + 4][::-1] for i in range(0, len(raw), 4))
        addr = ipaddress.IPv4Address(raw) if len(raw) == 4 else ipaddress.IPv6Address(raw)
        return (getattr(addr, "ipv4_mapped", None) or addr), int(hexport, 16)
    except (ValueError, TypeError):
        return None


def loopback_peer_uid(client, server) -> Optional[int]:
    """UID owning the client end (local=client, remote=server) of a socket that can still read, else None.

    `client` and `server` are (host, port); the whole tuple must match so a stale entry that only
    shares the ports (another loopback address) cannot stand in for this connection."""
    if not sys.platform.startswith("linux") or not client or not server:
        return None
    want = ((_ip(client[0]), client[1]), (_ip(server[0]), server[1]))
    if None in (want[0][0], want[1][0]) or not want[0][1] or not want[1][1]:
        return None
    for table in _PROC_TABLES:
        try:
            with open(table, "r", encoding = "ascii", errors = "replace") as fh:
                next(fh, None)  # header
                for line in fh:
                    parts = line.split()
                    if len(parts) < 8 or parts[3].upper() in _TCP_NOT_A_LIVE_CLIENT:
                        continue
                    if (_endpoint(parts[1]), _endpoint(parts[2])) != want:
                        continue
                    try:
                        return int(parts[7])
                    except ValueError:
                        return None
        except OSError:
            continue
    return None


def peer_is_another_local_user(request) -> bool:
    """True only when the peer is provably a different non-root OS user than this process."""
    try:
        client = getattr(request, "client", None)
        scope = getattr(request, "scope", None) or {}
        server = scope.get("server") if isinstance(scope, dict) else None
        client_ep = (client.host, client.port) if client is not None else None
        server_ep = (server[0], server[1]) if server and len(server) > 1 else None
        uid = loopback_peer_uid(client_ep, server_ep)
        if uid is None:
            return False
        # root can already read the 0600 password file, so it gains nothing here.
        return uid not in (os.geteuid(), 0)
    except Exception:
        return False
