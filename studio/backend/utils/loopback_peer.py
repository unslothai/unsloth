# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Which OS user opened a loopback TCP connection, read from /proc/net/tcp{,6} on Linux.

A loopback port is not an identity: on a shared host any OS user can connect to it. Used only to
keep the first-run password off pages fetched by a different local account. Unknown (not Linux,
no /proc, entry not found) returns None and callers keep today's behaviour."""

from __future__ import annotations

import os
import sys
from typing import Optional

_PROC_TABLES = ("/proc/net/tcp", "/proc/net/tcp6")


def _port_of(field: str) -> Optional[int]:
    try:
        return int(field.rsplit(":", 1)[1], 16)
    except (IndexError, ValueError):
        return None


def loopback_peer_uid(client_port: Optional[int], server_port: Optional[int]) -> Optional[int]:
    """UID owning the client end (local=client_port, remote=server_port), else None."""
    if not sys.platform.startswith("linux") or not client_port or not server_port:
        return None
    for table in _PROC_TABLES:
        try:
            with open(table, "r", encoding = "ascii", errors = "replace") as fh:
                next(fh, None)  # header
                for line in fh:
                    parts = line.split()
                    if len(parts) < 8:
                        continue
                    if _port_of(parts[1]) != client_port or _port_of(parts[2]) != server_port:
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
        client_port = getattr(client, "port", None) if client is not None else None
        server_port = server[1] if server and len(server) > 1 else None
        uid = loopback_peer_uid(client_port, server_port)
        if uid is None:
            return False
        # root can already read the 0600 password file, so it gains nothing here.
        return uid not in (os.geteuid(), 0)
    except Exception:
        return False
