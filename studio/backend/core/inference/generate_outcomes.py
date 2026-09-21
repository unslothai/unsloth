# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

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
    """*attempt_id* qualified by the acting account, or None when there is no id.

    The engines are shared across accounts, so an attempt is only identified by the pair.
    It also makes a named lookup safe to answer before the guards that hide ANOTHER
    account's generation: a caller can only ever match its own account's attempt.
    """
    if not attempt_id:
        return None
    from utils.account_context import current_account_id

    return f"{current_account_id()}\x00{attempt_id}"


# One store per process, not per engine instance: a request can switch the active engine,
# and an outcome kept on the instance that ran the generation goes with it.
_OUTCOMES: "OrderedDict[str, str]" = OrderedDict()
_OUTCOMES_LOCK = threading.Lock()


def _retain_generate_failure(attempt_id, reason: str) -> None:
    """Record *reason* against *attempt_id*, oldest entries first out."""
    attempt_id = attempt_scope_key(attempt_id)
    if not attempt_id:
        return
    with _OUTCOMES_LOCK:
        _OUTCOMES.pop(attempt_id, None)
        _OUTCOMES[attempt_id] = reason
        while len(_OUTCOMES) > _RETAINED_GENERATE_FAILURES:
            _OUTCOMES.popitem(last = False)


def generate_failure_for_attempt(attempt_id) -> Optional[str]:
    """The raw reason the generation *attempt_id* failed with, if it did and is remembered.

    Per attempt and independent of the active engine, so a caller settling a lost POST is
    answered about ITS OWN generation however many have run since. None for an attempt that
    succeeded, never ran, or has aged out.
    """
    attempt_id = attempt_scope_key(attempt_id)
    if not attempt_id:
        return None
    with _OUTCOMES_LOCK:
        return _OUTCOMES.get(attempt_id)
