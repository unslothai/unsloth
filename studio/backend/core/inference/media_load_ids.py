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
  is remembered and that load is refused when it arrives.

The check and the unload run under the same lock as the admission, so no load can be admitted
between them. Loads without a token, and unloads without ``expected_load_id``, behave as before.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Optional

# Tokens remembered per slot, for the superseded / cancelled answers (bounded: oldest go first).
_REMEMBERED = 512
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
_current: dict[str, _Record] = {}
_seen: dict[str, "OrderedDict[str, None]"] = {}
_cancelled: dict[str, "OrderedDict[str, None]"] = {}


def _remember(table: dict[str, "OrderedDict[str, None]"], modality: str, token: str) -> None:
    tokens = table.setdefault(modality, OrderedDict())
    tokens[token] = None
    tokens.move_to_end(token)
    while len(tokens) > _REMEMBERED:
        tokens.popitem(last = False)


def _media_lock():
    from hub.services.models.account_access import media_load_lock
    return media_load_lock


def refuse_cancelled(modality: str, load_token: Optional[str]) -> None:
    """Raise LoadCancelledError when an unload already cancelled ``load_token``."""
    if load_token:
        with _lock:
            if load_token in _cancelled.get(modality, ()):
                raise LoadCancelledError()


def admitting(modality: str, engine: Any, load_token: Optional[str]) -> Any:
    """Before the engine's begin_load: refuse a cancelled token, and snapshot what is resident.

    Called inside ``account_access.admit_media_load``, so under ``media_load_lock``, as is
    ``admitted`` once begin_load returned. Every image and video load goes through both, with or
    without a token, so a stale record never outlives a newer load.
    """
    refuse_cancelled(modality, load_token)
    # Before the worker starts: a cached load can publish its state before begin_load returns.
    return getattr(engine, "_state", None)


def admitted(modality: str, engine: Any, load_token: Optional[str], prior: Any) -> None:
    """The engine accepted the load: record it as the one in the slot."""
    with _lock:
        _current[modality] = _Record(
            load_token, engine, getattr(engine, "_load_token", None), prior
        )
        if load_token:
            _remember(_seen, modality, load_token)


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


def unload_if_current(
    modality: str, engine: Any, load_token: str, unload: Callable[[], Any]
) -> tuple[str, Any]:
    """Unload only the load ``load_token`` names: ("unloaded", unload()), ("superseded", None) or ("cancelled", None)."""
    with _media_lock():
        with _lock:
            record = _current.get(modality)
            matched = (
                record is not None and record.token == load_token and _describes(record, engine)
            )
            if not matched:
                if load_token in _seen.get(modality, ()):
                    return "superseded", None
                # Not admitted yet: refuse it when it comes, which a lost POST response cannot stop.
                _remember(_cancelled, modality, load_token)
                return "cancelled", None
        # Still under media_load_lock, so no load is admitted before this unload has run
        # (as retire_media_load does).
        status = unload()
        with _lock:
            if _current.get(modality) is record:
                del _current[modality]
        return "unloaded", status


def _reset_for_tests() -> None:
    with _lock:
        _current.clear()
        _seen.clear()
        _cancelled.clear()
