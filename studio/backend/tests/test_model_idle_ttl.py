# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Idle-TTL settings + backend activity tracking that drive auto-eviction."""

import sys
import time
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import utils.model_ttl_settings as ttl  # noqa: E402


def test_coerce_bounds_and_types():
    assert ttl._coerce_ttl_seconds(60) == 60
    assert ttl._coerce_ttl_seconds("120") == 120
    assert ttl._coerce_ttl_seconds(0) == 0
    assert ttl._coerce_ttl_seconds(-5) is None
    assert ttl._coerce_ttl_seconds(ttl.MAX_MODEL_IDLE_TTL_SECONDS + 1) is None
    assert ttl._coerce_ttl_seconds(True) is None
    assert ttl._coerce_ttl_seconds("abc") is None


def test_validate_raises_on_invalid():
    with pytest.raises(ValueError):
        ttl.validate_model_idle_ttl_seconds(-1)
    assert ttl.validate_model_idle_ttl_seconds(300) == 300


def test_default_from_env(monkeypatch):
    monkeypatch.delenv("UNSLOTH_MODEL_IDLE_TTL", raising = False)
    assert ttl.default_model_idle_ttl_seconds() == 0  # disabled by default
    monkeypatch.setenv("UNSLOTH_MODEL_IDLE_TTL", "900")
    assert ttl.default_model_idle_ttl_seconds() == 900
    monkeypatch.setenv("UNSLOTH_MODEL_IDLE_TTL", "bad")
    assert ttl.default_model_idle_ttl_seconds() == 0


def test_get_set_roundtrip(monkeypatch):
    store: dict = {}
    monkeypatch.setattr("storage.studio_db.get_app_setting", lambda k, d = None: store.get(k, d))
    monkeypatch.setattr("storage.studio_db.upsert_app_settings", lambda d: store.update(d))
    monkeypatch.delenv("UNSLOTH_MODEL_IDLE_TTL", raising = False)

    assert ttl.get_model_idle_ttl_seconds() == 0  # default disabled
    assert ttl.set_model_idle_ttl_seconds(600) == 600
    assert store[ttl.MODEL_IDLE_TTL_SETTING_KEY] == 600
    assert ttl.get_model_idle_ttl_seconds() == 600


def test_backend_idle_seconds_and_activity(monkeypatch):
    from routes.inference import get_llama_cpp_backend

    backend = get_llama_cpp_backend()

    # No model loaded -> idle is undefined (None), so the evictor never fires.
    monkeypatch.setattr(type(backend), "is_loaded", property(lambda self: False))
    assert backend.idle_seconds is None

    # Loaded -> idle measured from the last activity timestamp.
    monkeypatch.setattr(type(backend), "is_loaded", property(lambda self: True))
    backend._last_activity = time.monotonic() - 5.0
    assert backend.idle_seconds >= 4.0
    backend.note_activity()
    assert backend.idle_seconds < 1.0


def _loaded_backend(monkeypatch):
    from routes.inference import get_llama_cpp_backend

    backend = get_llama_cpp_backend()
    monkeypatch.setattr(type(backend), "is_loaded", property(lambda self: True))
    # Spy unload so eviction does not touch a real subprocess.
    calls = {"unload": 0}

    def _fake_unload():
        calls["unload"] += 1
        return True

    monkeypatch.setattr(backend, "unload_model", _fake_unload)
    backend._inflight_requests = 0
    return backend, calls


def test_evict_if_idle_disabled_and_not_idle(monkeypatch):
    backend, calls = _loaded_backend(monkeypatch)
    backend._last_activity = time.monotonic() - 100.0
    # ttl 0/None disables eviction entirely.
    assert backend.evict_if_idle(0) is None
    assert backend.evict_if_idle(None) is None
    # Idle below the TTL: nothing happens.
    backend.note_activity()
    assert backend.evict_if_idle(60) is None
    assert calls["unload"] == 0


def test_evict_if_idle_unloads_when_idle(monkeypatch):
    backend, calls = _loaded_backend(monkeypatch)
    backend._last_activity = time.monotonic() - 120.0
    evicted = backend.evict_if_idle(60)
    assert evicted is not None and evicted >= 60
    assert calls["unload"] == 1


def test_evict_if_idle_skips_when_request_in_flight(monkeypatch):
    backend, calls = _loaded_backend(monkeypatch)
    backend._last_activity = time.monotonic() - 120.0
    with backend.request_in_flight():
        # A request is active (even if it has not streamed a token yet): the
        # model must not be evicted out from under it.
        assert backend.evict_if_idle(60) is None
        assert calls["unload"] == 0
    # Once it finishes, activity was just refreshed, so still no eviction.
    assert backend.idle_seconds < 1.0


def test_request_in_flight_balances_counter(monkeypatch):
    backend, _ = _loaded_backend(monkeypatch)
    assert backend._inflight_requests == 0
    with backend.request_in_flight():
        assert backend._inflight_requests == 1
        with backend.request_in_flight():
            assert backend._inflight_requests == 2
        assert backend._inflight_requests == 1
    assert backend._inflight_requests == 0
