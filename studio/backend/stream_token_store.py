# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Thread-safe one-time token store for Option B (separate streaming server).

Tokens are short-lived (10 seconds) and consumed on first use.
"""

import threading
import time
import uuid
from typing import Optional


class StreamTokenStore:
    """Thread-safe store for short-lived, one-time-use streaming tokens."""

    def __init__(self, ttl_seconds: float = 10.0) -> None:
        self._ttl = ttl_seconds
        self._lock = threading.Lock()
        # token -> {"username": str, "expires": float}
        self._tokens: dict[str, dict] = {}

    def create_token(self, username: str) -> str:
        """Create a new one-time token for the given user. Returns the token string."""
        token = uuid.uuid4().hex
        expires = time.monotonic() + self._ttl
        with self._lock:
            self._purge_expired()
            self._tokens[token] = {"username": username, "expires": expires}
        return token

    def consume_token(self, token: str) -> Optional[str]:
        """
        Validate and consume a token. Returns the username if valid, None otherwise.
        The token is deleted after consumption (one-time use).
        """
        with self._lock:
            self._purge_expired()
            entry = self._tokens.pop(token, None)
        if entry is None:
            return None
        if time.monotonic() > entry["expires"]:
            return None
        return entry["username"]

    def _purge_expired(self) -> None:
        """Remove expired tokens. Must be called while holding _lock."""
        now = time.monotonic()
        expired = [k for k, v in self._tokens.items() if now > v["expires"]]
        for k in expired:
            del self._tokens[k]


# Module-level singleton
_store = StreamTokenStore()


def create_stream_token(username: str) -> str:
    return _store.create_token(username)


def consume_stream_token(token: str) -> Optional[str]:
    return _store.consume_token(token)
