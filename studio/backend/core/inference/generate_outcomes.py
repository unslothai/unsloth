# SPDX-License-Identifier: AGPL-3.0-only Copyright 2026-present the Unsloth AI Inc.

"""Terminal outcomes of recent image generations, keyed by the attempt that ran them.

Both engines retain a failed generation's reason so idle progress can still say WHY once
``_gen`` is cleared. One slot is not enough: a queued client can start its own run before
the client whose POST was lost reaches its next poll, and clearing the slot then discards
the reason that client is waiting for, leaving it to read the newcomer going idle as its
own success. Keyed by attempt and bounded, since only a settling caller reads one.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Optional

# Bounded: only a settling caller reads an outcome, and it reads it within seconds.
_RETAINED_GENERATE_FAILURES = 16


def attempt_scope_key(attempt_id) -> Optional[str]:
    """*attempt_id* qualified by the acting account, or None when there is no id."""
    if not attempt_id:
        return None
    from utils.account_context import current_account_id

    return f"{current_account_id()}\x00{attempt_id}"


_OUTCOMES: "OrderedDict[str, tuple[str, bool]]" = OrderedDict()
_OUTCOMES_LOCK = threading.Lock()


def _retain_generate_failure(
    attempt_id,
    reason: str,
    logged: bool = True,
) -> None:
    """Record *reason* against *attempt_id*, oldest entries first out."""
    attempt_id = attempt_scope_key(attempt_id)
    if not attempt_id:
        return
    with _OUTCOMES_LOCK:
        _OUTCOMES.pop(attempt_id, None)
        _OUTCOMES[attempt_id] = (reason, logged)
        while len(_OUTCOMES) > _RETAINED_GENERATE_FAILURES:
            _OUTCOMES.popitem(last = False)


def mark_generate_failure_unlogged(attempt_id) -> None:
    """Note that this attempt's retained reason never reached a log."""
    attempt_id = attempt_scope_key(attempt_id)
    if not attempt_id:
        return
    with _OUTCOMES_LOCK:
        record = _OUTCOMES.get(attempt_id)
        if record is not None:
            _OUTCOMES[attempt_id] = (record[0], False)


def clear_generate_failure(attempt_id) -> None:
    """Forget any retained outcome for *attempt_id*: a new execution owns that id now."""
    key = attempt_scope_key(attempt_id)
    if not key:
        return
    with _OUTCOMES_LOCK:
        _OUTCOMES.pop(key, None)


def generate_failure_was_logged(attempt_id) -> Optional[bool]:
    """Whether this attempt's retained reason reached a log, or None if nothing is held."""
    attempt_id = attempt_scope_key(attempt_id)
    if not attempt_id:
        return None
    with _OUTCOMES_LOCK:
        record = _OUTCOMES.get(attempt_id)
    return None if record is None else record[1]


def generate_failure_for_attempt(attempt_id) -> Optional[str]:
    """The raw reason the generation *attempt_id* failed with, if it did and is remembered."""
    attempt_id = attempt_scope_key(attempt_id)
    if not attempt_id:
        return None
    with _OUTCOMES_LOCK:
        record = _OUTCOMES.get(attempt_id)
    return None if record is None else record[0]
