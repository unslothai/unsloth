# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""tool calls handed to the client to run; the loop parks here until the client answers."""

import secrets
import threading
from typing import Optional

# how long an announced request may sit before a client says it is running it.
PICKUP_TIMEOUT_S = 20.0
# long because the client may await the user's approval; cancel_event still ends the wait early.
RESULT_TIMEOUT_S = 1800.0

_POLL_S = 0.25

CLIENT_DONE = "done"
CLIENT_CANCELLED = "cancelled"
CLIENT_UNCLAIMED = "unclaimed"
CLIENT_EXPIRED = "expired"

_lock = threading.Lock()
# request_id -> {"claimed": Event, "done": Event, "session": str, "result": str|None, "images": list}
_pending: dict[str, dict] = {}


def begin_client_tool(session_id) -> "tuple[str, dict]":
    """register a slot before announcing, so a fast client cannot answer a request with no slot."""
    request_id = secrets.token_urlsafe(16)
    slot = {
        "claimed": threading.Event(),
        "done": threading.Event(),
        "session": session_id or "",
        "result": None,
        "images": [],
    }
    with _lock:
        _pending[request_id] = slot
    return request_id, slot


def wait_client_tool(
    slot,
    request_id,
    cancel_event = None,
    pickup_timeout = None,
    result_timeout = None,
) -> "tuple[str, Optional[str], list]":
    """block until the client answers; result and images only mean something for CLIENT_DONE."""
    pickup_timeout = PICKUP_TIMEOUT_S if pickup_timeout is None else pickup_timeout
    result_timeout = RESULT_TIMEOUT_S if result_timeout is None else result_timeout

    def _give_up(outcome):
        # under the lock, or a late claim would run an action the model is told never happened.
        with _lock:
            if slot["done"].is_set():
                return CLIENT_DONE, slot["result"], list(slot["images"])
            if outcome == CLIENT_UNCLAIMED and slot["claimed"].is_set():
                return None
            if _pending.get(request_id) is slot:
                _pending.pop(request_id, None)
            return outcome, None, []

    try:
        waited = 0.0
        while not slot["done"].wait(timeout = _POLL_S):
            if cancel_event is not None and cancel_event.is_set():
                return _give_up(CLIENT_CANCELLED)
            waited += _POLL_S
            if not slot["claimed"].is_set():
                if waited >= pickup_timeout:
                    settled = _give_up(CLIENT_UNCLAIMED)
                    if settled is not None:
                        return settled
            elif waited >= result_timeout:
                return _give_up(CLIENT_EXPIRED)
        return CLIENT_DONE, slot["result"], list(slot["images"])
    finally:
        with _lock:
            if _pending.get(request_id) is slot:
                _pending.pop(request_id, None)


def abort_client_tool(slot, request_id) -> None:
    """drop a slot that was registered but never waited on."""
    with _lock:
        if _pending.get(request_id) is slot:
            _pending.pop(request_id, None)


def _slot_for(request_id, session_id):
    if not request_id:
        return None
    slot = _pending.get(request_id)
    if not slot:
        return None
    if session_id is not None and slot["session"] != (session_id or ""):
        return None
    return slot


def claim_client_tool(request_id, session_id = None) -> bool:
    """mark a request as running; false if not pending or claimed, so a replay cannot rerun it."""
    with _lock:
        slot = _slot_for(request_id, session_id)
        if slot is None or slot["done"].is_set() or slot["claimed"].is_set():
            return False
        slot["claimed"].set()
    return True


def resolve_client_tool(
    request_id,
    result,
    images = None,
    session_id = None,
) -> bool:
    """deliver the result and unblock the loop; the first answer wins, like a tool decision."""
    with _lock:
        slot = _slot_for(request_id, session_id)
        if slot is None or slot["done"].is_set():
            return False
        slot["result"] = result
        slot["images"] = list(images or [])
        slot["claimed"].set()
        slot["done"].set()
    return True
