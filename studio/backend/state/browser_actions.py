# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Browser tool slots answered by POST /api/inference/browser-action-result; first answer wins;
registered before announced; scoped to session and account"""

from __future__ import annotations

import secrets
import threading
from typing import Any, Optional

_lock = threading.Lock()
_pending: dict[str, dict] = {}


def new_action_id() -> str:
    return secrets.token_urlsafe(16)


def begin_browser_action(action_id: str, session_id: Optional[str], account_id: str) -> dict:
    slot = {
        "event": threading.Event(),
        "result": None,
        "session": session_id or "",
        "account": account_id or "",
    }
    with _lock:
        _pending[action_id] = slot
    return slot


def discard_browser_action(action_id: str, slot: dict) -> None:
    """Drop a slot that will never be waited on (the stream closed before the tab was asked)."""
    with _lock:
        if _pending.get(action_id) is slot:
            _pending.pop(action_id, None)


def wait_browser_action(
    slot: dict,
    action_id: str,
    cancel_event = None,
    timeout: Optional[float] = None,
) -> tuple[str, Optional[dict]]:
    """Block until the tab answers. Returns ``("answered", result)``, ``("cancelled", None)`` or
    ``("expired", None)``; always removes its own slot."""
    waited = 0.0
    try:
        while not slot["event"].wait(timeout = 0.25):
            if cancel_event is not None and cancel_event.is_set():
                return "cancelled", None
            waited += 0.25
            if timeout is not None and waited >= timeout:
                return "expired", None
        return "answered", slot["result"]
    finally:
        with _lock:
            if _pending.get(action_id) is slot:
                _pending.pop(action_id, None)


def resolve_browser_action(
    action_id: str, result: Any, session_id: Optional[str], account_id: str
) -> bool:
    if not action_id or not isinstance(result, dict):
        return False
    with _lock:
        slot = _pending.get(action_id)
        if not slot or slot["event"].is_set():
            return False
        if slot["session"] != (session_id or "") or slot["account"] != (account_id or ""):
            return False
        slot["result"] = result
        slot["event"].set()
    return True
