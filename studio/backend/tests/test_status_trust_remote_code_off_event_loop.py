# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""/api/inference/status is polled: its auto_map fallback can reach the Hub, so it must not
run on the event loop thread (an unreachable Hub parked the loop and /api/liveness)."""

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace

import routes.inference as inference_routes


def _backend(active = "unsloth/Qwen3-8B", info = None):
    return SimpleNamespace(
        active_model_name = active,
        models = {active: dict(info or {})} if active else {},
        loading_models = set(),
    )


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


def _status():
    return _run(inference_routes._slot_status(current_subject = "t"))


def test_the_auto_map_fallback_runs_off_the_event_loop_thread(monkeypatch):
    threads: list[tuple[int, str]] = []

    def _auto_map(*_args, **_kwargs):
        threads.append((threading.get_ident(), threading.current_thread().name))
        return False

    monkeypatch.setattr(inference_routes, "_peek_inference_backend", lambda *a, **k: _backend())
    monkeypatch.setattr(inference_routes, "_auto_map_trust_remote_code", _auto_map)

    loop_thread = threading.get_ident()
    _status()

    assert threads, "status never ran the auto_map fallback"
    assert all(ident != loop_thread for ident, _ in threads)
    # Not the default executor: it drives local token streaming.
    assert all(name.startswith("inference-status") for _, name in threads), threads


def test_the_fallback_runs_inside_the_offline_guard(monkeypatch):
    guarded: list[tuple] = []

    def _offline_guarded(targets, fn, /, *args, **kwargs):
        guarded.append(tuple(targets))
        return fn(*args, **kwargs)

    monkeypatch.setattr(inference_routes, "_peek_inference_backend", lambda *a, **k: _backend())
    monkeypatch.setattr(inference_routes, "_offline_guarded", _offline_guarded)
    monkeypatch.setattr(inference_routes, "_auto_map_trust_remote_code", lambda *a, **k: False)

    _status()

    assert guarded == [("unsloth/Qwen3-8B",)]


def test_a_trust_decision_stored_at_load_skips_the_guard_and_the_thread(monkeypatch):
    entered: list[str] = []

    monkeypatch.setattr(
        inference_routes,
        "_peek_inference_backend",
        lambda *a, **k: _backend(info = {"requires_trust_remote_code": True}),
    )
    monkeypatch.setattr(
        inference_routes,
        "_offline_guarded",
        lambda *a, **k: entered.append("guard") or False,
    )
    monkeypatch.setattr(
        inference_routes,
        "_auto_map_trust_remote_code",
        lambda *a, **k: entered.append("auto_map") or False,
    )

    response = _status()

    assert response.requires_trust_remote_code is True
    assert entered == []


def test_a_load_completing_mid_resolve_does_not_mix_two_models(monkeypatch):
    backend = _backend("unsloth/Qwen3-8B")

    def _auto_map(*_args, **_kwargs):
        backend.active_model_name = "unsloth/Llama-3.1-8B"
        backend.models = {"unsloth/Llama-3.1-8B": {}}
        return True

    monkeypatch.setattr(inference_routes, "_peek_inference_backend", lambda *a, **k: backend)
    monkeypatch.setattr(inference_routes, "_auto_map_trust_remote_code", _auto_map)

    response = _status()

    assert response.active_model == "unsloth/Qwen3-8B"
    assert response.requires_trust_remote_code is True
    assert response.loaded == ["unsloth/Qwen3-8B"]


def test_no_loaded_model_reads_nothing(monkeypatch):
    called: list[int] = []

    monkeypatch.setattr(inference_routes, "_peek_inference_backend", lambda *a, **k: _backend(None))
    monkeypatch.setattr(
        inference_routes,
        "_auto_map_trust_remote_code",
        lambda *a, **k: called.append(1) or False,
    )

    response = _status()

    assert not called
    assert response.requires_trust_remote_code is False
