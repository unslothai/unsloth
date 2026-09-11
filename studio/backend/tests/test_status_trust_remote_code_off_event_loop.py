# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Reporting trust_remote_code on /api/inference/status must not read the Hub from the
event loop thread.

The frontend polls this route continuously, and the auto_map fallback reads raw config
JSON from the Hub, so on an unreachable Hub that read parks the loop and the server stops
answering anything, /api/liveness included.

Asserts which thread the read ran on rather than timing it.
"""

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace

import routes.inference as inference_routes


def _backend(active = "unsloth/Qwen3-8B"):
    return SimpleNamespace(
        active_model_name = active,
        models = {active: {}} if active else {},
        loading_models = set(),
    )


def test_the_auto_map_fallback_runs_off_the_event_loop_thread(monkeypatch):
    threads: list[int] = []

    def _resolve(*_args, **_kwargs):
        threads.append(threading.get_ident())
        return False

    monkeypatch.setattr(inference_routes, "_peek_inference_backend", lambda *a, **k: _backend())
    monkeypatch.setattr(inference_routes, "_resolve_loaded_trust_remote_code", _resolve)

    # run_until_complete drives the loop on this thread.
    loop_thread = threading.get_ident()
    asyncio.new_event_loop().run_until_complete(
        inference_routes.get_status(current_subject = "t")
    )

    assert threads, "status never resolved trust_remote_code"
    assert loop_thread not in threads, "trust_remote_code was resolved on the event loop thread"


def test_the_fallback_runs_inside_the_offline_guard(monkeypatch):
    """Off-loop is not enough: the guard is what bounds the read to the memoised
    reachability verdict instead of the connect timeout."""
    guarded: list[tuple] = []

    def _offline_guarded(targets, fn, /, *args, **kwargs):
        guarded.append(tuple(targets))
        return fn(*args, **kwargs)

    monkeypatch.setattr(inference_routes, "_peek_inference_backend", lambda *a, **k: _backend())
    monkeypatch.setattr(inference_routes, "_offline_guarded", _offline_guarded)
    monkeypatch.setattr(
        inference_routes, "_resolve_loaded_trust_remote_code", lambda *a, **k: False
    )

    asyncio.new_event_loop().run_until_complete(
        inference_routes.get_status(current_subject = "t")
    )

    assert guarded == [("unsloth/Qwen3-8B",)], (
        "the trust_remote_code read did not go through _offline_guarded"
    )


def test_no_loaded_model_reads_nothing(monkeypatch):
    """Nothing is loaded, so there is no repo to ask about."""
    called: list[int] = []

    monkeypatch.setattr(inference_routes, "_peek_inference_backend", lambda *a, **k: _backend(None))
    monkeypatch.setattr(
        inference_routes,
        "_resolve_loaded_trust_remote_code",
        lambda *a, **k: called.append(1) or False,
    )

    response = asyncio.new_event_loop().run_until_complete(
        inference_routes.get_status(current_subject = "t")
    )

    assert not called, "status resolved trust_remote_code with no model loaded"
    assert response.requires_trust_remote_code is False
