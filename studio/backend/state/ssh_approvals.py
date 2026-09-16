# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Per-session approved SSH deployment targets.

When a user confirms a tool call that reaches an SSH server, the target host is
recorded here so later terminal and python invocations can connect consistently.
"""

import ipaddress
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterable, Optional

from utils.account_context import current_account_id

_lock = threading.Lock()
# account and session -> approved host literals
_approved: dict[tuple[str, str], set[str]] = {}
_call_approved: ContextVar[tuple[str, frozenset[str]] | None] = ContextVar(
    "ssh_call_approved", default = None
)


def normalize_host(host: str) -> str:
    """Normalize a host literal for set membership."""
    h = (host or "").strip().lower().rstrip(".")
    if h.startswith("[") and "]" in h:
        h = h[1 : h.index("]")]
    try:
        return str(ipaddress.ip_address(h))
    except ValueError:
        pass
    return h


def approve_hosts(session_id: Optional[str], hosts: Iterable[str]) -> None:
    """Record one or more approved SSH targets for a sandbox session."""
    if not session_id:
        return
    normalized = {normalize_host(h) for h in hosts if h and normalize_host(h)}
    if not normalized:
        return
    with _lock:
        bucket = _approved.setdefault((current_account_id(), session_id), set())
        bucket.update(normalized)


def is_host_approved(session_id: Optional[str], host: str) -> bool:
    key = normalize_host(host)
    if not key:
        return False
    if not session_id:
        approval = _call_approved.get()
        return bool(approval and approval[0] == current_account_id() and key in approval[1])
    with _lock:
        return key in _approved.get((current_account_id(), session_id), set())


@contextmanager
def temporary_ssh_approval(hosts: Iterable[str]):
    """Approve hosts only within the executing tool's context."""
    token = _call_approved.set(
        (current_account_id(), frozenset(normalize_host(host) for host in hosts))
    )
    try:
        yield
    finally:
        _call_approved.reset(token)


def approved_hosts(session_id: Optional[str]) -> frozenset[str]:
    if not session_id:
        return frozenset()
    with _lock:
        return frozenset(_approved.get((current_account_id(), session_id), set()))


def clear_session(session_id: Optional[str]) -> None:
    if not session_id:
        return
    with _lock:
        _approved.pop((current_account_id(), session_id), None)


def reset_ssh_approvals() -> None:
    """Test helper: drop every recorded approval."""
    with _lock:
        _approved.clear()
