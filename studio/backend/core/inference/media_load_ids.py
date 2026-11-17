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

# A cancellation waits for its load or a day; no hard bound on a load POST exists to derive it.
_CANCELLED_TTL_S = 24 * 3600.0
# Per (slot, account) guard: a newer token evicts only its own account's entries.
_PER_ACCOUNT = 4096
_now = time.monotonic
# Bumped under _lock on each admission and token unload, so a status read can spot a change.
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
    """Holds state weakly so a torn-down model is freed once its engine drops it; states that cannot
    be weakly referenced are held strongly; call it for the state, or None once gone"""
    if state is None:
        return lambda: None
    try:
        return weakref.ref(state)
    except TypeError:
        return lambda: state


_lock = threading.Lock()
_current: dict[str, _Record] = {}
# Previous publisher (token, engine, state) of the resident state; a failed load keeps it.
_published: dict[str, tuple[str, Any, Any]] = {}
# Per (slot, account) admitted tokens; losing one turns a superseded answer into cancelled.
_seen: dict[tuple[str, str], "OrderedDict[str, None]"] = {}
# Per (slot, account) that unloaded: token to expiry; cancels only that account's tokens.
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
    """Refuses a cancelled token and snapshots resident state before begin_load, under
    media_load_lock; every image and video load goes through it, with or without a token"""
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
        # Per account, like _cancelled: other accounts' tokens never supersede this one.
        _remember(_seen, modality, token)


def _describes(record: _Record, engine: Any) -> bool:
    # Engine still holds this record's load, in flight or published, with no unload since.
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
    """Status and load_id pair only if no admission or token unload landed during the read; the read
    runs outside all locks, must not block admissions, is retried, and after _SNAPSHOT_TRIES
    gives no id"""
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
    """Unload only the load load_token names; returns (unloaded, status), (superseded, None) or
    (cancelled, None); a failed replacement leaves the previous model resident"""
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
        # Under media_load_lock, so no load is admitted before this unload runs.
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
