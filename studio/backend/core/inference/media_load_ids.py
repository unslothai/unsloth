# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""load_token and expected_load_id: unload only while that load still owns the slot, else answer
superseded or cancelled without touching it; check and unload share the admission lock; loads
without a token and unloads without expected_load_id behave as before."""

from __future__ import annotations

import threading
import time
import uuid
import weakref
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
# Bumped under _lock on every admission and token unload, so a status read can tell whether its
# state changed while it ran.
_generation = 0
# Status reads snapshot() tries before answering with no load id (an admission landed in each).
_SNAPSHOT_TRIES = 3
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
    # A weak reference to the state resident before this load (see _weak), never the model itself.
    prior: Callable[[], Any]


def _weak(state: Any) -> Callable[[], Any]:
    """``state`` without keeping it alive: a torn-down model (pipeline, weights) must be freed as soon
    as its engine drops it, not when this registry is next cleared. Call it to get the state back,
    or None once it is gone. The engines' states are plain dataclasses, so weakly referable; anything
    that is not is held as is."""
    if state is None:
        return lambda: None
    try:
        return weakref.ref(state)
    except TypeError:
        return lambda: state


_lock = threading.Lock()
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
            _published[modality] = (record.token, engine, _weak(state))
    # Before the worker starts: a cached load can publish its state before begin_load returns.
    return state


def admitted(modality: str, engine: Any, load_token: Optional[str], prior: Any) -> None:
    """The engine accepted the load: record it as the one in the slot (a load without a token gets one)."""
    global _generation
    with _lock:
        _generation += 1
        token = load_token or f"studio-{uuid.uuid4().hex}"
        _current[modality] = _Record(
            token, engine, getattr(engine, "_load_token", None), _weak(prior)
        )
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
    return state is not None and state is not record.prior()


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
        if published[2]() is state:
            return published[0]
    return None


def _resident_token(modality: str, engine: Any) -> Optional[str]:
    # The token of the state resident in ``engine`` (under _lock). Unlike _slot_token, a load in
    # flight over it does not hide it: a replacement that fails leaves this model loaded.
    state = getattr(engine, "_state", None)
    if state is None:
        return None
    record = _current.get(modality)
    loading = getattr(engine, "loading_repo_ids", None)
    in_flight = callable(loading) and bool(loading())
    if record is not None and not in_flight and _describes(record, engine):
        return record.token
    published = _published.get(modality)
    if published is not None and published[1] is engine and published[2]() is state:
        return published[0]
    return None


def snapshot(
    modality: str, engine: Any, read_status: Callable[[], Any]
) -> tuple[Any, Optional[str]]:
    """``read_status()`` and the ``load_id`` of the same state, so the id never names a load the status
    does not show.

    The read runs outside every lock (it can block, and must not hold up admissions or unloads); the
    id is taken only if no admission or token unload landed while it ran, else the read is retried.
    After ``_SNAPSHOT_TRIES`` the answer carries no id. A load still starting when the read begins
    records its token only at its admission, which moves the generation, and an engine that began a
    load since its record was taken is not named by it, so neither pairing can mix two states.
    """
    status, load_id, _ = snapshot_ids(modality, engine, read_status)
    return status, load_id


def snapshot_ids(
    modality: str, engine: Any, read_status: Callable[[], Any]
) -> tuple[Any, Optional[str], Optional[str]]:
    """``snapshot`` plus ``resident_load_id``: the token of the model resident under a load still in
    flight (the same as ``load_id`` when none is), taken under the same generation check."""
    status = None
    for _ in range(_SNAPSHOT_TRIES):
        with _lock:
            generation = _generation
        status = read_status()
        with _lock:
            if _generation == generation:
                return status, _slot_token(modality, engine), _resident_token(modality, engine)
    return status, None, None


def current_id(modality: str, engine: Any) -> Optional[str]:
    """The ``load_id`` the status routes report: the token of the load in the slot, or None."""
    with _lock:
        return _slot_token(modality, engine)


def _in_flight_unlocked(engine: Any) -> bool:
    # Read inside the engine's own lock (from its unload), so never through loading_repo_ids(),
    # which takes that lock: its in-flight marker, as loading_repo_ids() reads it.
    if not hasattr(engine, "_loading"):
        loading = getattr(engine, "loading_repo_ids", None)
        return bool(callable(loading) and loading())
    loading = engine._loading
    return loading is not None and getattr(loading, "error", None) is None


def _still_named(record: Optional[_Record], published: Any, engine: Any) -> Callable[[], bool]:
    """What the engine's unload rechecks under the lock its load worker settles under: the load the
    token named is still the one in flight or resident (no load ended or began since the check)."""
    if record is not None:

        def guard() -> bool:
            if getattr(engine, "_load_token", None) != record.epoch:
                return False
            if _in_flight_unlocked(engine):
                return True
            state = getattr(engine, "_state", None)
            return state is not None and state is not record.prior()

        return guard
    # The weak reference, so this recheck does not keep alive the model the unload tears down.
    resident = published[2]

    def guard_resident() -> bool:
        state = getattr(engine, "_state", None)
        return state is not None and state is resident() and not _in_flight_unlocked(engine)

    return guard_resident


def unload_if_current(
    modality: str, engine: Any, load_token: str, unload: Callable[[Callable[[], bool]], Any]
) -> tuple[str, Any]:
    """Unload only the load ``load_token`` names: ("unloaded", status), ("superseded", None) or ("cancelled", None).

    ``unload(only_if)`` runs the engine's unload with that recheck: it answers None, and tears
    nothing down, if the named load ended in between (a replacement whose worker failed leaves the
    previous model resident, which is not this token's to unload).
    """
    global _generation
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
            named = record if record is not None and record.token == load_token else None
            only_if = _still_named(named, _published.get(modality), engine)
        # Still under media_load_lock, so no load is admitted before this unload has run
        # (as retire_media_load does).
        status = unload(only_if)
        if status is None:
            return "superseded", None
        with _lock:
            _generation += 1
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
