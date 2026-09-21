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

from collections import OrderedDict
from typing import Optional

# A failed generation's reason has to outlive the run AND the runs after it: a queued client
# can take the slot before the client whose POST was lost gets its next one-second poll, and
# one retained slot would hand that client either nothing or someone else's outcome. Small
# and bounded, since only a settling caller reads one and it reads it within seconds.
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


def _retain_generate_failure(engine, attempt_id, reason: str) -> None:
    """Record *reason* against *attempt_id* on *engine*, oldest entries first out."""
    attempt_id = attempt_scope_key(attempt_id)
    if not attempt_id:
        return
    outcomes = getattr(engine, "_generate_outcomes", None)
    if outcomes is None:
        outcomes = OrderedDict()
        engine._generate_outcomes = outcomes
    outcomes.pop(attempt_id, None)
    outcomes[attempt_id] = reason
    while len(outcomes) > _RETAINED_GENERATE_FAILURES:
        outcomes.popitem(last = False)


def generate_failure_for_attempt(engine, attempt_id) -> Optional[str]:
    """The raw reason the generation *attempt_id* failed with, if it did and is remembered.

    Per attempt, so a caller settling a lost POST is answered about ITS OWN generation
    however many have run since. None for an attempt that succeeded, never ran, or has
    aged out.
    """
    attempt_id = attempt_scope_key(attempt_id)
    if not attempt_id:
        return None
    return (getattr(engine, "_generate_outcomes", None) or {}).get(attempt_id)
