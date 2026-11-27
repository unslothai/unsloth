# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Connections Python and Terminal asked for: the pending questions, each chat's answers, and a recent log."""

import secrets
import threading
import time
from collections import OrderedDict, deque
from typing import Optional

_MAX_CHATS = 256
_MAX_LOG = 300
_REMIND_SECONDS = 5.0

_lock = threading.Lock()
# request_id -> {"event", "decision", "session", "account_id", "host"}
_pending: dict[str, dict] = {}
# (account_id, chat) -> what that chat allowed and how much it sent: {"hosts", "uploads", "sent"}
_chats: "OrderedDict[tuple[str, str], dict]" = OrderedDict()
_log: "deque[dict]" = deque(maxlen = _MAX_LOG)


def begin_request(session_id: Optional[str], account_id: str, host: str) -> tuple[str, dict]:
    request_id = secrets.token_urlsafe(16)
    slot = {
        "event": threading.Event(),
        "decision": None,
        "session": session_id or "",
        "account_id": account_id,
        "host": host,
    }
    with _lock:
        _pending[request_id] = slot
    return request_id, slot


def wait_request(
    request_id: str,
    slot: dict,
    stopped,
    timeout: float,
    remind = None,
) -> str:
    """remind() runs every few seconds while the question is open, for a chat tab that joined late."""
    try:
        started = time.monotonic()
        reminded = started
        while not slot["event"].wait(timeout = 0.25):
            now = time.monotonic()
            if stopped() or now - started >= timeout:
                break
            if remind is not None and now - reminded >= _REMIND_SECONDS:
                reminded = now
                remind()
        with _lock:
            return slot["decision"] or "deny"
    finally:
        with _lock:
            _pending.pop(request_id, None)


def pending_host(request_id: str, session_id: Optional[str], account_id: str) -> Optional[str]:
    with _lock:
        slot = _pending.get(request_id)
        if (
            not slot
            or slot["account_id"] != account_id
            or slot["session"] != (session_id or "")
            or slot["event"].is_set()
        ):
            return None
        return slot["host"]


def resolve_request(
    request_id: str, decision: str, session_id: Optional[str], account_id: str
) -> bool:
    with _lock:
        slot = _pending.get(request_id)
        if (
            not slot
            or slot["account_id"] != account_id
            or slot["session"] != (session_id or "")
            or slot["event"].is_set()
        ):
            return False
        slot["decision"] = decision
        slot["event"].set()
    return True


def _chat(account_id: str, chat: str) -> dict:
    state = _chats.setdefault((account_id, chat), {"hosts": set(), "uploads": set(), "sent": {}})
    _chats.move_to_end((account_id, chat))
    while len(_chats) > _MAX_CHATS:
        _chats.popitem(last = False)
    return state


def grant(
    account_id: str,
    chat: str,
    host: str,
    kind: str = "hosts",
) -> None:
    """kind "hosts" lets the chat reach the website; "uploads" lets it send past the budget."""
    with _lock:
        _chat(account_id, chat)[kind].add(host)


def granted(
    account_id: str,
    chat: str,
    host: str,
    kind: str = "hosts",
) -> bool:
    with _lock:
        state = _chats.get((account_id, chat))
        return bool(state) and host in state[kind]


def add_sent(account_id: str, chat: str, host: str, sent: int) -> int:
    with _lock:
        totals = _chat(account_id, chat)["sent"]
        totals[host] = totals.get(host, 0) + sent
        return totals[host]


def record(
    tool: str,
    host: str,
    port: int,
    verdict: str,
    sent: int = 0,
    received: int = 0,
) -> None:
    with _lock:
        _log.append(
            {
                "at": time.time(),
                "tool": tool,
                "host": host,
                "port": port,
                "verdict": verdict,
                "sent": sent,
                "received": received,
            }
        )


def recent_connections() -> list[dict]:
    with _lock:
        entries = list(_log)
    rows: "OrderedDict[str, dict]" = OrderedDict()
    for entry in reversed(entries):
        row = rows.get(entry["host"])
        if row is None:
            rows[entry["host"]] = {**entry, "count": 1}
        else:
            row["count"] += 1
            row["sent"] += entry["sent"]
            row["received"] += entry["received"]
    return list(rows.values())
