# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Per-session approved SSH deployment targets.

When a user confirms a tool call that reaches an SSH server, the target host is
recorded here so later terminal and python invocations can connect consistently.
"""

import threading
from typing import Iterable, Optional

_lock = threading.Lock()
# session_id -> normalized host literals approved for that sandbox session
_approved: dict[str, set[str]] = {}


def normalize_host(host: str) -> str:
    """Normalize a host literal for set membership."""
    h = (host or "").strip().lower().rstrip(".")
    if h.startswith("[") and "]" in h:
        h = h[1 : h.index("]")]
    return h


def approve_hosts(session_id: Optional[str], hosts: Iterable[str]) -> None:
    """Record one or more approved SSH targets for a sandbox session."""
    if not session_id:
        return
    normalized = {normalize_host(h) for h in hosts if h and normalize_host(h)}
    if not normalized:
        return
    with _lock:
        bucket = _approved.setdefault(session_id, set())
        bucket.update(normalized)


def is_host_approved(session_id: Optional[str], host: str) -> bool:
    if not session_id:
        return False
    key = normalize_host(host)
    if not key:
        return False
    with _lock:
        return key in _approved.get(session_id, set())


def approved_hosts(session_id: Optional[str]) -> frozenset[str]:
    if not session_id:
        return frozenset()
    with _lock:
        return frozenset(_approved.get(session_id, set()))


def clear_session(session_id: Optional[str]) -> None:
    if not session_id:
        return
    with _lock:
        _approved.pop(session_id, None)


def reset_ssh_approvals() -> None:
    """Test helper: drop every recorded approval."""
    with _lock:
        _approved.clear()
