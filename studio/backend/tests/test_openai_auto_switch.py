# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Opt-in OpenAI /v1 model auto-switch: resolver, hook, and settings coercion.

No GPU or llama-server: the backend and the load route are mocked, mirroring
tests/test_gguf_completion_usage.py.
"""

import asyncio

import pytest

import routes.inference as inference_route
from models.inference import LoadRequest
from core.inference import local_model_resolver as resolver
from utils import openai_auto_switch_settings as settings


class _FakeBackend:
    def __init__(self, loaded_id=None):
        self.model_identifier = loaded_id
        self.is_loaded = loaded_id is not None


class _LoadRecorder:
    """Stand-in for the load route: records calls and simulates a load."""

    def __init__(self, backend, fail=False):
        self.backend = backend
        self.calls = []
        self.fail = fail

    async def __call__(self, request, fastapi_request, current_subject=None):
        self.calls.append(request)
        if self.fail:
            from fastapi import HTTPException
            raise HTTPException(status_code=503, detail="load failed")
        self.backend.model_identifier = request.model_path
        self.backend.is_loaded = True
        return None


def _wire(monkeypatch, *, enabled, resolves_to, backend, recorder):
    monkeypatch.setattr(settings, "get_openai_auto_switch_enabled", lambda: enabled)
    monkeypatch.setattr(resolver, "resolve_local_gguf", lambda _m: resolves_to)
    monkeypatch.setattr(inference_route, "get_llama_cpp_backend", lambda: backend)
    monkeypatch.setattr(inference_route, "load_model", recorder)


def _run_hook(model="some/model"):
    asyncio.run(inference_route._maybe_auto_switch_model(model, object(), "tester"))


def test_flag_off_never_loads(monkeypatch):
    backend = _FakeBackend("unsloth/A-GGUF")
    rec = _LoadRecorder(backend)
    _wire(monkeypatch, enabled=False, resolves_to=("unsloth/B-GGUF", None), backend=backend, recorder=rec)
    _run_hook("unsloth/B-GGUF")
    assert rec.calls == []


def test_unknown_model_falls_through(monkeypatch):
    backend = _FakeBackend("unsloth/A-GGUF")
    rec = _LoadRecorder(backend)
    _wire(monkeypatch, enabled=True, resolves_to=None, backend=backend, recorder=rec)
    _run_hook("gpt-4o-mini")
    assert rec.calls == []


def test_already_loaded_does_not_reload(monkeypatch):
    backend = _FakeBackend("unsloth/A-GGUF")
    rec = _LoadRecorder(backend)
    # Case-insensitive match against the loaded identifier.
    _wire(monkeypatch, enabled=True, resolves_to=("unsloth/a-gguf", None), backend=backend, recorder=rec)
    _run_hook("unsloth/A-GGUF")
    assert rec.calls == []


def test_known_unloaded_model_switches_once(monkeypatch):
    backend = _FakeBackend("unsloth/A-GGUF")
    rec = _LoadRecorder(backend)
    _wire(monkeypatch, enabled=True, resolves_to=("unsloth/B-GGUF", "Q4_K_M"), backend=backend, recorder=rec)
    _run_hook("unsloth/B-GGUF:Q4_K_M")
    assert len(rec.calls) == 1
    req = rec.calls[0]
    assert isinstance(req, LoadRequest)
    assert req.model_path == "unsloth/B-GGUF"
    assert req.gguf_variant == "Q4_K_M"
    assert backend.model_identifier == "unsloth/B-GGUF"


def test_concurrent_same_target_loads_once(monkeypatch):
    backend = _FakeBackend(None)
    rec = _LoadRecorder(backend)
    _wire(monkeypatch, enabled=True, resolves_to=("unsloth/B-GGUF", None), backend=backend, recorder=rec)

    async def _race():
        await asyncio.gather(
            inference_route._maybe_auto_switch_model("unsloth/B-GGUF", object(), "t"),
            inference_route._maybe_auto_switch_model("unsloth/B-GGUF", object(), "t"),
        )

    asyncio.run(_race())
    assert len(rec.calls) == 1


def test_load_failure_propagates(monkeypatch):
    from fastapi import HTTPException
    backend = _FakeBackend("unsloth/A-GGUF")
    rec = _LoadRecorder(backend, fail=True)
    _wire(monkeypatch, enabled=True, resolves_to=("unsloth/B-GGUF", None), backend=backend, recorder=rec)
    with pytest.raises(HTTPException):
        _run_hook("unsloth/B-GGUF")


# ── resolver ────────────────────────────────────────────────────────

def test_resolver_matches_and_splits_variant(monkeypatch):
    monkeypatch.setattr(resolver, "_build_index", lambda: {"unsloth/b-gguf": "unsloth/B-GGUF"})
    resolver._scan = (0.0, {})  # force a rescan
    assert resolver.resolve_local_gguf("unsloth/B-GGUF:UD-Q5_K_XL") == ("unsloth/B-GGUF", "UD-Q5_K_XL")
    assert resolver.resolve_local_gguf("unsloth/B-GGUF") == ("unsloth/B-GGUF", None)
    assert resolver.resolve_local_gguf("totally/unknown") is None
    assert resolver.resolve_local_gguf("") is None


# ── settings coercion ───────────────────────────────────────────────

def test_setting_coercion():
    assert settings._coerce_bool("on") is True
    assert settings._coerce_bool("off") is False
    assert settings._coerce_bool("garbage") is None
    assert settings._coerce_int("5") == 5
    assert settings._coerce_int(-3) == 0
    assert settings._coerce_int("nope") is None
