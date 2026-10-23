# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Client load tokens for the image and video slots, so an unload can name the load it means.

A load request may carry ``load_token`` (an opaque id the client makes up). The load route
records it when the engine accepts the load, under ``account_access.media_load_lock``, together
with the engine, its load epoch (``_load_token``, bumped by every begin_load and unload) and the
state that was resident before. ``POST .../unload`` with ``expected_load_id`` then unloads only
while that record still describes the engine: no begin_load or unload ran since (same epoch),
and either a load is in flight (it can only be that one) or a state that load published is
resident. Anything else answers without touching the slot:

- ``superseded``: the token was admitted, but its load ended or another load replaced it;
- ``cancelled``: the token was never admitted (its POST is still on its way, or was lost), so it
  is remembered and that load is refused when it arrives, for the account that sent the unload only.

A load sent without a token gets one from Studio, and the status routes report the token of the
load in the slot as ``load_id``, so a client can unload exactly the model it saw.

The check and the unload run under the same lock as the admission, so no load can be admitted
between them. Loads without a token, and unloads without ``expected_load_id``, behave as before.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Optional

# A cancellation is kept until its load arrives and is refused (which consumes it) or this long
# has passed. No hard bound on a load POST's lifetime exists to derive it from (the MCP forwards
# in-process with no client timeout, studio_mcp/forward.py, and the route's waits before
# admission, the chat context save and the arbiter's eviction, have none either), so it is set far
# beyond any real one: a load still on its way after a day is not one an unload is racing.
_CANCELLED_TTL_S = 24 * 3600.0
# Memory guard per (slot, account): only an account's own newer tokens can push one out, never
# another account's.
_PER_ACCOUNT = 4096
_now = time.monotonic
MAX_TOKEN_LEN = 128


class LoadCancelledError(RuntimeError):
    """A load whose token an unload already cancelled; the load routes answer it as a 409."""

    def __init__(self) -> None:
        super().__init__("This load was cancelled by an unload before it started.")


@dataclass
class _Record:
    token: Optional[str]
    engine: Any
    epoch: Any
    prior: Any


_lock = threading.Lock()
# The last load admitted per slot.
_current: dict[str, _Record] = {}
# The load that published the resident state before the last admission, as (token, engine, state):
# a newer load that failed leaves it resident.
_published: dict[str, tuple[str, Any, Any]] = {}
# Per (slot, account) whose load was admitted with the token. Losing an old one only turns a
# superseded answer into cancelled, which touches nothing either.
_seen: dict[tuple[str, str], "OrderedDict[str, None]"] = {}
# Per (slot, account) that sent the unload: token -> expiry (time.monotonic()). An unload cancels
# a token for its own account only.
_cancelled: dict[tuple[str, str], "OrderedDict[str, float]"] = {}


def _account() -> str:
    from utils.account_context import current_account_id
    return current_account_id()


def _remember(
    table: dict,
    modality: str,
    token: str,
    value: Any = None,
) -> None:
    tokens = table.setdefault((modality, _account()), OrderedDict())
    tokens[token] = value
    tokens.move_to_end(token)
    while len(tokens) > _PER_ACCOUNT:
        tokens.popitem(last = False)


def _cancel(modality: str, token: str) -> None:
    now = _now()
    tokens = _cancelled.get((modality, _account()))
    # Expiries are in insertion order (one TTL), so the expired ones are at the front.
    while tokens and next(iter(tokens.values())) <= now:
        tokens.popitem(last = False)
    _remember(_cancelled, modality, token, now + _CANCELLED_TTL_S)


def _media_lock():
    from hub.services.models.account_access import media_load_lock
    return media_load_lock


def refuse_cancelled(modality: str, load_token: Optional[str]) -> None:
    """Raise LoadCancelledError when an unload already cancelled ``load_token``."""
    if load_token:
        with _lock:
            tokens = _cancelled.get((modality, _account()), {})
            expiry = tokens.pop(load_token, None)
            # Consumed: its one load has now arrived, and is refused.
            if expiry is not None and expiry > _now():
                raise LoadCancelledError()


def admitting(modality: str, engine: Any, load_token: Optional[str]) -> Any:
    """Before the engine's begin_load: refuse a cancelled token, and snapshot what is resident.

    Called inside ``account_access.admit_media_load``, so under ``media_load_lock``, as is
    ``admitted`` once begin_load returned. Every image and video load goes through both, with or
    without a token, so a stale record never outlives a newer load.
    """
    refuse_cancelled(modality, load_token)
    with _lock:
        record = _current.get(modality)
        state = getattr(engine, "_state", None)
        loading = getattr(engine, "loading_repo_ids", None)
        in_flight = callable(loading) and bool(loading())
        if (
            record is not None
            and state is not None
            and not in_flight
            and _describes(record, engine)
        ):
            # The resident model's load: still named while a newer load fails to replace it.
            _published[modality] = (record.token, engine, state)
    # Before the worker starts: a cached load can publish its state before begin_load returns.
    return state


def admitted(modality: str, engine: Any, load_token: Optional[str], prior: Any) -> None:
    """The engine accepted the load: record it as the one in the slot (a load without a token gets one)."""
    with _lock:
        token = load_token or f"studio-{uuid.uuid4().hex}"
        _current[modality] = _Record(token, engine, getattr(engine, "_load_token", None), prior)
        # Per account like _cancelled: another account's earlier use of a token never answers
        # superseded for this one, which then records its own cancellation.
        _remember(_seen, modality, token)


def _describes(record: _Record, engine: Any) -> bool:
    # The engine still holds the load this record names: no begin_load or unload since, and
    # either a load in flight or the state that load published.
    if record.engine is not engine or getattr(engine, "_load_token", None) != record.epoch:
        return False
    loading = getattr(engine, "loading_repo_ids", None)
    if callable(loading) and loading():
        return True
    state = getattr(engine, "_state", None)
    return state is not None and state is not record.prior


def _slot_token(modality: str, engine: Any) -> Optional[str]:
    # The token of the load in flight or resident in ``engine``, if Studio can name it (under _lock).
    record = _current.get(modality)
    if record is not None and _describes(record, engine):
        return record.token
    loading = getattr(engine, "loading_repo_ids", None)
    if callable(loading) and loading():
        return None
    published = _published.get(modality)
    state = getattr(engine, "_state", None)
    if published is not None and published[1] is engine and state is not None:
        if published[2] is state:
            return published[0]
    return None


def snapshot(
    modality: str, engine: Any, read_status: Callable[[], Any]
) -> tuple[Any, Optional[str]]:
    """``read_status()`` and the ``load_id`` of the same state: both under the admission lock, so no
    load is admitted between them and the id never names a load the status does not show."""
    with _media_lock():
        status = read_status()
        with _lock:
            return status, _slot_token(modality, engine)


def current_id(modality: str, engine: Any) -> Optional[str]:
    """The ``load_id`` the status routes report: the token of the load in the slot, or None."""
    with _lock:
        return _slot_token(modality, engine)


def unload_if_current(
    modality: str, engine: Any, load_token: str, unload: Callable[[], Any]
) -> tuple[str, Any]:
    """Unload only the load ``load_token`` names: ("unloaded", unload()), ("superseded", None) or ("cancelled", None)."""
    with _media_lock():
        with _lock:
            record = _current.get(modality)
            if _slot_token(modality, engine) != load_token:
                if load_token in _seen.get((modality, _account()), ()):
                    return "superseded", None
                # Not admitted yet: refuse it when it comes, which a lost POST response cannot stop.
                # Only for this account, so no account can cancel another's load in advance.
                _cancel(modality, load_token)
                return "cancelled", None
        # Still under media_load_lock, so no load is admitted before this unload has run
        # (as retire_media_load does).
        status = unload()
        with _lock:
            if _current.get(modality) is record:
                del _current[modality]
            _published.pop(modality, None)
        return "unloaded", status


def _reset_for_tests() -> None:
    with _lock:
        _current.clear()
        _published.clear()
        _seen.clear()
        _cancelled.clear()
