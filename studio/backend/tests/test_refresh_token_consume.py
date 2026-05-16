# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Tests for consume_refresh_token's atomic-rotation contract.

PR 5375 introduced ``DELETE ... RETURNING`` for single-use refresh-token
rotation. ``RETURNING`` is SQLite 3.35+. This module exercises both the
modern path and the SELECT+DELETE fallback so older system SQLite
(e.g. Ubuntu 20.04, some Windows builds) keeps refresh working.
"""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

import pytest

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))


@pytest.fixture()
def isolated_storage(tmp_path, monkeypatch):
    """Point auth.storage at a fresh DB under tmp_path for every test.

    auth.storage.DB_PATH is computed at module load from utils.paths.
    Rather than re-importing the module, we point ``DB_PATH`` at a
    tmp_path SQLite file and rely on get_connection's CREATE TABLE IF
    NOT EXISTS to lazily build the schema on first use.
    """
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    import importlib

    if "auth.storage" in sys.modules:
        importlib.reload(sys.modules["auth.storage"])
    storage = importlib.import_module("auth.storage")
    # Force DB_PATH under tmp_path so every test gets a clean DB.
    monkeypatch.setattr(storage, "DB_PATH", Path(tmp_path) / "auth.db")
    # Reset the RETURNING-feature cache so each test re-probes.
    monkeypatch.setattr(storage, "_RETURNING_SUPPORTED", None)
    yield storage


def _make_token(storage, *, is_desktop = False):
    """Insert a fresh, far-future refresh token and return its raw form."""
    import secrets as _secrets
    from datetime import datetime, timedelta, timezone

    raw = _secrets.token_urlsafe(32)
    expires = (datetime.now(timezone.utc) + timedelta(days = 14)).isoformat()
    storage.save_refresh_token(raw, "unsloth", expires, is_desktop = is_desktop)
    return raw


class TestSingleUseRotation:
    def test_consume_returns_username_on_first_use(self, isolated_storage):
        storage = isolated_storage
        token = _make_token(storage)
        result = storage.consume_refresh_token(token)
        assert result == ("unsloth", False)

    def test_consume_returns_none_on_replay(self, isolated_storage):
        storage = isolated_storage
        token = _make_token(storage)
        first = storage.consume_refresh_token(token)
        second = storage.consume_refresh_token(token)
        assert first == ("unsloth", False)
        assert second is None

    def test_consume_returns_none_for_unknown_token(self, isolated_storage):
        storage = isolated_storage
        assert storage.consume_refresh_token("not-a-real-token") is None

    def test_desktop_flag_round_trips(self, isolated_storage):
        storage = isolated_storage
        token = _make_token(storage, is_desktop = True)
        assert storage.consume_refresh_token(token) == ("unsloth", True)


class TestReturningFallback:
    """Pin the SELECT+DELETE fallback so non-RETURNING SQLite still works."""

    def test_fallback_path_consumes_atomically(self, isolated_storage, monkeypatch):
        storage = isolated_storage
        # Force the fallback branch regardless of the underlying sqlite
        # version. ``_supports_returning`` caches the result on first
        # probe so setting it directly is enough.
        monkeypatch.setattr(storage, "_RETURNING_SUPPORTED", False)
        token = _make_token(storage)
        first = storage.consume_refresh_token(token)
        second = storage.consume_refresh_token(token)
        assert first == ("unsloth", False)
        assert second is None

    def test_fallback_unknown_token_returns_none(self, isolated_storage, monkeypatch):
        storage = isolated_storage
        monkeypatch.setattr(storage, "_RETURNING_SUPPORTED", False)
        assert storage.consume_refresh_token("unknown") is None

    def test_fallback_race_only_one_wins(self, isolated_storage, monkeypatch):
        storage = isolated_storage
        monkeypatch.setattr(storage, "_RETURNING_SUPPORTED", False)
        token = _make_token(storage)
        # Hammer with N threads; exactly one should observe the token.
        # We are testing the rowcount-on-DELETE guarantee in the
        # fallback path -- not raw throughput.
        winners: list = []
        losers: list = []
        barrier = threading.Barrier(8)

        def attempt():
            barrier.wait()
            r = storage.consume_refresh_token(token)
            (winners if r else losers).append(r)

        threads = [threading.Thread(target = attempt) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert (
            len(winners) == 1
        ), f"expected exactly 1 winner; winners={winners} losers={losers}"
        assert winners[0] == ("unsloth", False)
        assert all(r is None for r in losers)


class TestReturningSupportedProbe:
    def test_returns_bool_on_any_sqlite(self, isolated_storage):
        storage = isolated_storage
        # Force re-probe.
        storage._RETURNING_SUPPORTED = None
        result = storage._supports_returning()
        assert isinstance(result, bool)
