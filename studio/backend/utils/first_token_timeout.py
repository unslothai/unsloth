# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Installation-wide local inference wait limit and request-scoped prefill lease.

Infinity represents Unlimited in deadline arithmetic only; socket APIs receive None.
The mutable wait state crosses asyncio.to_thread's copied context so durable runs can
renew their lease during an explicitly permitted wait, without disabling orphan cleanup.
"""

from __future__ import annotations

import math
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

DEFAULT_MINUTES = 20
MAX_MINUTES = 10080  # One week; 0 explicitly means Unlimited.
SETTING_KEY = "first_token_timeout_minutes"


def get_minutes() -> int:
    try:
        from storage.studio_db import get_app_setting
        from utils.account_context import OWNER, run_as

        value = run_as(OWNER, get_app_setting, SETTING_KEY, DEFAULT_MINUTES)
        if type(value) is int and 0 <= value <= MAX_MINUTES:
            return value
    except Exception:
        pass  # An unavailable settings store must not break inference.
    return DEFAULT_MINUTES


def set_minutes(value: int) -> None:
    if type(value) is not int or not 0 <= value <= MAX_MINUTES:
        raise ValueError(f"Use 0 for Unlimited or 1–{MAX_MINUTES} whole minutes.")
    from storage.studio_db import upsert_app_settings
    from utils.account_context import OWNER, run_as

    run_as(OWNER, upsert_app_settings, {SETTING_KEY: value})


def timeout_seconds() -> float:
    minutes = get_minutes()
    return minutes * 60.0 if minutes else math.inf


def socket_timeout(seconds: float) -> float | None:
    return seconds if math.isfinite(seconds) else None


@dataclass
class FirstTokenWait:
    deadline: float | None = None

    def active(self) -> bool:
        return self.deadline is not None and time.monotonic() < self.deadline


_current_wait: ContextVar[FirstTokenWait | None] = ContextVar("first_token_wait", default = None)


@contextmanager
def track_first_token_wait():
    state = FirstTokenWait()
    token = _current_wait.set(state)
    try:
        yield state
    finally:
        state.deadline = None
        _current_wait.reset(token)


def start_first_token_wait(seconds: float | None = None) -> float:
    deadline = time.monotonic() + (timeout_seconds() if seconds is None else seconds)
    state = _current_wait.get()
    if state is not None:
        state.deadline = deadline
    return deadline


def finish_first_token_wait() -> None:
    state = _current_wait.get()
    if state is not None:
        state.deadline = None
