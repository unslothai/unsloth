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

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Optional

_cache: ContextVar[Optional[dict]] = ContextVar("plan_metadata_cache", default = None)


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

    Raises exactly what the underlying call raises: a failure is not cached, so one repo that was
    unreachable for a single candidate does not decide the rest of the batch."""
    cache = _cache.get()
    if cache is None:
        return api.model_info(repo_id, files_metadata = files_metadata, token = token)
    key = (repo_id, files_metadata, token)
    if key not in cache:
        cache[key] = api.model_info(repo_id, files_metadata = files_metadata, token = token)
    return cache[key]
