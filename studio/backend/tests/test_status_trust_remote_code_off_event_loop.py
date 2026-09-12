# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Reporting trust_remote_code on /api/inference/status must not read the Hub from the
event loop thread.

The frontend polls this route continuously, and the auto_map fallback reads raw config
JSON from the Hub, so on an unreachable Hub that read parks the loop and the server stops
answering anything, /api/liveness included.

Only that fallback leaves the loop. The value stored at load and the YAML default are
dict reads, and the offline guard probes reachability on entry, so sending those through
it would charge a polled route a network probe for nothing.

Asserts which thread the read ran on rather than timing it.
"""

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


def test_the_auto_map_fallback_runs_off_the_event_loop_thread(monkeypatch):
    threads: list[tuple[int, str]] = []

    def _auto_map(*_args, **_kwargs):
        threads.append((threading.get_ident(), threading.current_thread().name))
        return False

    monkeypatch.setattr(inference_routes, "_peek_inference_backend", lambda *a, **k: _backend())
    monkeypatch.setattr(inference_routes, "_auto_map_trust_remote_code", _auto_map)

    # run_until_complete drives the loop on this thread.
    loop_thread = threading.get_ident()
    _run(inference_routes.get_status(current_subject = "t"))

    assert threads, "status never ran the auto_map fallback"
    assert all(
        ident != loop_thread for ident, _ in threads
    ), "trust_remote_code was resolved on the event loop thread"
    # The bounded status executor, not the default one: that pool drives local token
    # streaming, and enough overlapping polls on a slow Hub would starve it.
    assert all(
        name.startswith("inference-status") for _, name in threads
    ), f"the fallback ran on {[n for _, n in threads]}, not on _STATUS_PROBE_EXECUTOR"


def test_the_fallback_runs_inside_the_offline_guard(monkeypatch):
    """Off-loop is not enough: the guard is what bounds the read to the memoised
    reachability verdict instead of the connect timeout."""
    guarded: list[tuple] = []

    def _offline_guarded(targets, fn, /, *args, **kwargs):
        guarded.append(tuple(targets))
        return fn(*args, **kwargs)

    monkeypatch.setattr(inference_routes, "_peek_inference_backend", lambda *a, **k: _backend())
    monkeypatch.setattr(inference_routes, "_offline_guarded", _offline_guarded)
    monkeypatch.setattr(inference_routes, "_auto_map_trust_remote_code", lambda *a, **k: False)

    _run(inference_routes.get_status(current_subject = "t"))

    assert guarded == [
        ("unsloth/Qwen3-8B",)
    ], "the auto_map read did not go through _offline_guarded"


def test_a_trust_decision_stored_at_load_skips_the_guard_and_the_thread(monkeypatch):
    """The load path stores requires_trust_remote_code on the model. Reporting it back is
    a dict read, and the guard probes the Hub on entry (memoised 5s, up to 3s cold), so a
    polled route must answer from the stored value without entering either."""
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

    response = _run(inference_routes.get_status(current_subject = "t"))

    assert response.requires_trust_remote_code is True
    assert entered == [], f"a stored trust decision still went through {entered}"


def test_a_load_completing_mid_resolve_does_not_mix_two_models(monkeypatch):
    """Going off-loop puts a suspension point between the snapshot and the response. A
    load landing in that window must not pair the new model's identity with the trust
    requirement, capabilities, or resident list read for the one it replaced."""
    backend = _backend("unsloth/Qwen3-8B")

    def _auto_map(*_args, **_kwargs):
        backend.active_model_name = "unsloth/Llama-3.1-8B"
        backend.models = {"unsloth/Llama-3.1-8B": {}}
        return True

    monkeypatch.setattr(inference_routes, "_peek_inference_backend", lambda *a, **k: backend)
    monkeypatch.setattr(inference_routes, "_auto_map_trust_remote_code", _auto_map)

    response = _run(inference_routes.get_status(current_subject = "t"))

    assert response.active_model == "unsloth/Qwen3-8B", (
        "reported the model that landed mid-resolve, while carrying the trust_remote_code "
        "and capabilities read for the one it replaced"
    )
    assert response.requires_trust_remote_code is True
    # Same snapshot as active_model: a response whose active model is missing from its own
    # resident list reads as "the active model is an unloaded cache entry".
    assert response.loaded == [
        "unsloth/Qwen3-8B"
    ], f"loaded={response.loaded} was read after the await, against active_model from before it"


def test_no_loaded_model_reads_nothing(monkeypatch):
    """Nothing is loaded, so there is no repo to ask about."""
    called: list[int] = []

    monkeypatch.setattr(inference_routes, "_peek_inference_backend", lambda *a, **k: _backend(None))
    monkeypatch.setattr(
        inference_routes,
        "_auto_map_trust_remote_code",
        lambda *a, **k: called.append(1) or False,
    )

    response = _run(inference_routes.get_status(current_subject = "t"))

    assert not called, "status ran the auto_map fallback with no model loaded"
    assert response.requires_trust_remote_code is False
