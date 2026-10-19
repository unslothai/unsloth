# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""One-time, short-lived handoff of internal credentials to the Studio frontend."""

from __future__ import annotations

import secrets
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional


@dataclass(frozen = True, slots = True)
class OIDCSessionHandoff:
    access_token: str
    refresh_token: str
    account_id: str
    created_at: float


class OIDCSessionHandoffManager:
    def __init__(
        self,
        *,
        max_age_seconds: float = 60.0,
        max_handoffs: int = 1024,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_age_seconds <= 0 or max_handoffs <= 0:
            raise ValueError("OIDC handoff limits must be positive")
        self._max_age = max_age_seconds
        self._max_handoffs = max_handoffs
        self._clock = clock
        self._handoffs: dict[str, OIDCSessionHandoff] = {}
        self._lock = threading.Lock()

    def _prune(self, now: float) -> None:
        for code, handoff in tuple(self._handoffs.items()):
            if now - handoff.created_at > self._max_age:
                self._handoffs.pop(code, None)

    def create(self, *, access_token: str, refresh_token: str, account_id: str) -> str:
        now = self._clock()
        code = secrets.token_urlsafe(32)
        with self._lock:
            self._prune(now)
            if len(self._handoffs) >= self._max_handoffs:
                oldest = min(self._handoffs, key = lambda key: self._handoffs[key].created_at)
                self._handoffs.pop(oldest, None)
            self._handoffs[code] = OIDCSessionHandoff(
                access_token = access_token,
                refresh_token = refresh_token,
                account_id = account_id,
                created_at = now,
            )
        return code

    def consume(self, code: str) -> Optional[OIDCSessionHandoff]:
        now = self._clock()
        with self._lock:
            handoff = self._handoffs.pop(code, None)
            self._prune(now)
        if handoff is None or now - handoff.created_at > self._max_age:
            return None
        return handoff


oidc_session_handoffs = OIDCSessionHandoffManager()
