# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""routes/llama.py: the source_build field is exposed and the handlers run the
(now subprocess-touching) detection off the event loop via a worker thread.

The route file is loaded standalone with a stubbed auth dependency so the test
does not pull the whole routes package (matplotlib-heavy training router) and
works in a minimal env.
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
import threading
import types
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

pytest.importorskip("fastapi")


def _load_route():
    # Prefer the real auth module; stub it only in minimal envs where its
    # deps are absent. Stubs are popped after the load so they never leak
    # into sys.modules for the rest of the suite.
    stubbed = []
    try:
        import auth.authentication  # noqa: F401
    except Exception:
        auth_pkg = types.ModuleType("auth")
        auth_pkg.__path__ = []
        auth_mod = types.ModuleType("auth.authentication")
        auth_mod.get_current_subject = lambda: "test"
        for name, stub in (("auth", auth_pkg), ("auth.authentication", auth_mod)):
            if name not in sys.modules:
                sys.modules[name] = stub
                stubbed.append(name)
    try:
        spec = importlib.util.spec_from_file_location(
            "llama_route_under_test", str(_BACKEND / "routes" / "llama.py")
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules["llama_route_under_test"] = mod  # so pydantic resolves forward refs
        spec.loader.exec_module(mod)
        return mod
    finally:
        for name in stubbed:
            sys.modules.pop(name, None)


rl = _load_route()


def test_status_response_exposes_source_build():
    payload = {
        "supported": True,
        "update_available": True,
        "stale": False,
        "installed_tag": None,
        "latest_tag": "b9585",
        "published_repo": "unslothai/llama.cpp",
        "installed_at_utc": None,
        "age_days": None,
        "source_build": True,
        "job": {"state": "idle", "reload_required": False},
    }
    model = rl.LlamaUpdateStatusResponse(**payload)
    assert model.model_dump()["source_build"] is True
    assert model.model_dump()["job"]["reload_required"] is False
    # Extra/unknown keys must not crash the response model.
    rl.LlamaUpdateStatusResponse(**{**payload, "unexpected": 1})


def test_status_response_exposes_update_size_bytes():
    payload = {
        "supported": True,
        "update_available": True,
        "stale": False,
        "installed_tag": "b9493",
        "latest_tag": "b9518",
        "published_repo": "unslothai/llama.cpp",
        "installed_at_utc": None,
        "age_days": None,
        "source_build": False,
        "update_size_bytes": 123_456_789,
        "job": {"state": "idle"},
    }
    model = rl.LlamaUpdateStatusResponse(**payload)
    assert model.model_dump()["update_size_bytes"] == 123_456_789
    # Omitted -> defaults to None (the offline / no-matching-asset case).
    without = {k: v for k, v in payload.items() if k != "update_size_bytes"}
    assert rl.LlamaUpdateStatusResponse(**without).model_dump()["update_size_bytes"] is None


def test_status_handler_runs_off_event_loop(monkeypatch):
    seen = {}

    def fake_status(force_refresh = False):
        seen["thread"] = threading.current_thread()
        return {
            "supported": True,
            "update_available": True,
            "source_build": True,
            "latest_tag": "b9585",
            "job": {"state": "idle"},
        }

    monkeypatch.setattr(rl, "get_update_status", fake_status)
    out = asyncio.run(rl.llama_update_status(force_refresh = False, current_subject = "t"))
    assert out.source_build is True
    # Detection ran in a worker thread, not the event-loop thread.
    assert seen["thread"] is not threading.main_thread()


def test_update_handler_runs_off_event_loop(monkeypatch):
    seen = {}

    def fake_start():
        seen["thread"] = threading.current_thread()
        return {"started": True, "reason": None, "job": {"state": "running"}}

    monkeypatch.setattr(rl, "start_update", fake_start)
    out = asyncio.run(rl.llama_update(current_subject = "t"))
    assert out.started is True
    assert seen["thread"] is not threading.main_thread()
