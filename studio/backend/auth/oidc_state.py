# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Single-use OIDC authorization state, nonce, and PKCE material."""

from __future__ import annotations

import base64
import hashlib
import secrets
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional


@dataclass(frozen = True, slots = True)
class OIDCLoginAttempt:
    nonce: str
    code_verifier: str
    created_at: float


@dataclass(frozen = True, slots = True)
class OIDCLoginParameters:
    state: str
    nonce: str
    code_verifier: str
    code_challenge: str


def pkce_challenge(code_verifier: str) -> str:
    digest = hashlib.sha256(code_verifier.encode("ascii")).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


class OIDCStateManager:
    """Thread-safe, bounded state store for short-lived authorization attempts."""

    def __init__(
        self,
        *,
        max_age_seconds: float = 300.0,
        max_attempts: int = 4096,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_age_seconds <= 0 or max_attempts <= 0:
            raise ValueError("OIDC state limits must be positive")
        self._max_age = max_age_seconds
        self._max_attempts = max_attempts
        self._clock = clock
        self._attempts: dict[str, OIDCLoginAttempt] = {}
        self._lock = threading.Lock()

    def _prune(self, now: float) -> None:
        expired = [
            state
            for state, attempt in self._attempts.items()
            if now - attempt.created_at > self._max_age
        ]
        for state in expired:
            self._attempts.pop(state, None)

    def create(self) -> OIDCLoginParameters:
        now = self._clock()
        state = secrets.token_urlsafe(32)
        nonce = secrets.token_urlsafe(32)
        # RFC 7636 permits 43-128 unreserved characters. token_urlsafe(64) is 86.
        code_verifier = secrets.token_urlsafe(64)
        with self._lock:
            self._prune(now)
            if len(self._attempts) >= self._max_attempts:
                oldest = min(self._attempts, key = lambda key: self._attempts[key].created_at)
                self._attempts.pop(oldest, None)
            self._attempts[state] = OIDCLoginAttempt(
                nonce = nonce,
                code_verifier = code_verifier,
                created_at = now,
            )
        return OIDCLoginParameters(
            state = state,
            nonce = nonce,
            code_verifier = code_verifier,
            code_challenge = pkce_challenge(code_verifier),
        )

    def consume(self, state: str) -> Optional[OIDCLoginAttempt]:
        """Return and delete a valid attempt; invalid and replayed states return None."""

        if not isinstance(state, str) or not state:
            return None
        now = self._clock()
        with self._lock:
            attempt = self._attempts.pop(state, None)
            self._prune(now)
        if attempt is None or now - attempt.created_at > self._max_age:
            return None
        return attempt

    def clear(self) -> None:
        with self._lock:
            self._attempts.clear()

    def __len__(self) -> int:
        with self._lock:
            self._prune(self._clock())
            return len(self._attempts)


oidc_state_manager = OIDCStateManager()
