# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""One Hub listing per repo for a batch of download plans.

Sizing a picker's expander plans every candidate quant, and each plan independently lists the
checkpoint repo and the base repo. On unsloth/LTX-2.3-GGUF that is 63 plans and ~126 identical
``model_info`` calls, enough to exhaust the Hub rate limit and fail the downloads that follow.

Scoped through a context variable rather than a parameter so the planners keep their signatures
and a caller that does not opt in keeps today's behaviour exactly: outside ``shared_plan_metadata``
every lookup goes to the Hub.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Optional

_cache: ContextVar[Optional[dict]] = ContextVar("plan_metadata_cache", default = None)
# Guards the slot table only, never a Hub call: candidates plan on worker threads, so two of them
# racing the same key would otherwise both fetch, and the loser's error could overwrite the
# winner's listing and omit every row that needed it.
_slots_guard = threading.Lock()


class _Slot:
    """One key's listing, produced once and awaited by everyone else."""

    __slots__ = ("done", "value", "error")

    def __init__(self) -> None:
        self.done = threading.Event()
        self.value: Any = None
        self.error: Optional[BaseException] = None


@contextmanager
def shared_plan_metadata():
    """Reuse each repo's listing for the plans built inside this block."""
    token = _cache.set({})
    try:
        yield
    finally:
        _cache.reset(token)


def plan_model_info(
    api: Any,
    repo_id: str,
    *,
    files_metadata: bool = True,
    token = None,
) -> Any:
    """``api.model_info(repo_id, ...)``, served from the batch's listings when one is active.

    Failures are remembered and re-raised, not retried: an offline or rate-limiting Hub would
    otherwise have all 63 LTX candidates repeat the same slow failing call, turning one outage
    into a request storm. Each candidate still reports no size, which is the same answer a retry
    would have reached."""
    cache = _cache.get()
    if cache is None:
        return api.model_info(repo_id, files_metadata = files_metadata, token = token)
    key = (repo_id, files_metadata, token)
    with _slots_guard:
        slot = cache.get(key)
        produce = slot is None
        if produce:
            slot = cache[key] = _Slot()
    if produce:
        try:
            slot.value = api.model_info(repo_id, files_metadata = files_metadata, token = token)
        except Exception as exc:  # noqa: BLE001 -- kept to re-raise, never swallowed
            slot.error = exc
        finally:
            # In finally, so a producer that dies unexpectedly cannot leave the others waiting.
            slot.done.set()
    else:
        slot.done.wait()
    if slot.error is not None:
        raise slot.error
    return slot.value
