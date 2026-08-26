# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Verify that slow inference status probes run off the event loop."""

from __future__ import annotations

import asyncio
import sys
import threading
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import routes.inference as inference_route  # noqa: E402

# Multiple turns make scheduler progress unambiguous.
_CONTROL_TURNS = 5
# Prevent a regression from hanging the suite.
_GUARD_SECONDS = 10.0


class _FakeLlamaBackend:
    is_loaded = False


class _FakeInferenceBackend:
    active_model_name = None
    models: dict = {}
    loading_models: set = set()


def _patch_status_dependencies(monkeypatch):
    """Stub everything the route touches other than the two slow probes."""
    monkeypatch.setattr(inference_route, "get_llama_cpp_backend", _FakeLlamaBackend)
    monkeypatch.setattr(inference_route, "get_inference_backend", _FakeInferenceBackend)
    monkeypatch.setattr(
        inference_route,
        "_detect_safetensors_features",
        lambda *_args: {
            "supports_reasoning": False,
            "reasoning_style": "enable_thinking",
            "reasoning_effort_levels": [],
            "reasoning_always_on": False,
            "supports_preserve_thinking": False,
            "supports_tools": False,
        },
    )


def _patch_slow_probes(monkeypatch, *, entered, release):
    """Block the capability probe and stub the GitHub request."""
    from utils import llama_cpp_freshness

    def _find_binary(_cls):
        return "/nonexistent/llama-server"

    def _probe_capabilities(_cls, _binary):
        entered.set()
        release.wait(timeout = _GUARD_SECONDS)
        return {"found": True, "supports_mtp": True}

    def _check_freshness(_binary):
        return {"stale": True, "installed_tag": "b1", "latest_tag": "b2"}

    monkeypatch.setattr(
        _FakeLlamaBackend,
        "_find_llama_server_binary",
        classmethod(_find_binary),
        raising = False,
    )
    monkeypatch.setattr(
        _FakeLlamaBackend,
        "probe_server_capabilities",
        classmethod(_probe_capabilities),
        raising = False,
    )
    monkeypatch.setattr(llama_cpp_freshness, "check_prebuilt_freshness", _check_freshness)


def test_status_probe_runs_off_the_event_loop(monkeypatch):
    """The blocked probe must not stall the shared streaming loop."""
    _patch_status_dependencies(monkeypatch)
    entered = threading.Event()
    release = threading.Event()
    _patch_slow_probes(monkeypatch, entered = entered, release = release)

    async def _run():
        turns = 0

        async def _control():
            nonlocal turns
            for _ in range(_CONTROL_TURNS):
                await asyncio.sleep(0)
                turns += 1

        status = asyncio.create_task(inference_route.get_status(current_subject = "test"))
        control = asyncio.create_task(_control())
        # Wait without blocking the event loop.
        started = await asyncio.to_thread(entered.wait, _GUARD_SECONDS)
        await control
        # The control task finished while the probe remained blocked.
        probe_in_flight = not status.done()
        release.set()
        response = await asyncio.wait_for(status, timeout = _GUARD_SECONDS)
        return response, started, turns, probe_in_flight

    response, started, turns, probe_in_flight = asyncio.run(_run())

    assert started, "the probe never ran"
    assert turns == _CONTROL_TURNS
    assert probe_in_flight, "the status request finished its probe on the event loop"
    assert response.llama_cpp_supports_mtp is True
    assert response.llama_cpp_prebuilt_stale is True
    assert response.llama_cpp_installed_tag == "b1"
    assert response.llama_cpp_latest_tag == "b2"


def test_overlapping_status_probes_leave_default_executor_for_streaming(monkeypatch):
    """Slow polls cannot starve the workers that advance local token streams."""
    _patch_status_dependencies(monkeypatch)
    entered = threading.Event()
    release = threading.Event()
    _patch_slow_probes(monkeypatch, entered = entered, release = release)

    async def _wait_for_probe():
        deadline = asyncio.get_running_loop().time() + _GUARD_SECONDS
        while not entered.is_set() and asyncio.get_running_loop().time() < deadline:
            await asyncio.sleep(0.001)
        return entered.is_set()

    async def _run():
        loop = asyncio.get_running_loop()
        # One worker makes default-executor starvation deterministic.  The status
        # executor remains separate, so two overlapping polls still leave it free.
        loop.set_default_executor(ThreadPoolExecutor(max_workers = 1))
        statuses = [
            asyncio.create_task(inference_route.get_status(current_subject = "test"))
            for _ in range(2)
        ]
        try:
            started = await _wait_for_probe()
            token = await asyncio.wait_for(
                asyncio.to_thread(lambda: "token"), timeout = _GUARD_SECONDS
            )
        finally:
            release.set()
        responses = await asyncio.gather(*statuses)
        return started, token, responses

    started, token, responses = asyncio.run(_run())

    assert started, "the status probe never ran"
    assert token == "token"
    assert len(responses) == 2


def test_status_reports_tracked_load_before_backend_lock(monkeypatch):
    """Model resolution is visible before a GGUF backend acquires its load lock."""
    _patch_status_dependencies(monkeypatch)
    monkeypatch.setattr(
        inference_route,
        "_probe_llama_cpp_status",
        lambda _backend: (False, {}),
    )
    attempt = inference_route._ScopedLoadAttempt(
        token = "attempt",
        request_id = None,
        model_path = "org/slow-model-GGUF",
        subject = "test",
        cancel_event = threading.Event(),
        cancel_complete = threading.Event(),
    )
    monkeypatch.setattr(inference_route, "_running_load_attempt", attempt)

    response = asyncio.run(inference_route.get_status(current_subject = "test"))

    assert response.active_model is None
    assert response.loading == ["org/slow-model-GGUF"]


def test_status_keeps_resident_model_visible_during_tracked_preflight(monkeypatch):
    """A new load must not hide the model that is still serving."""
    _patch_status_dependencies(monkeypatch)
    monkeypatch.setattr(
        inference_route,
        "_probe_llama_cpp_status",
        lambda _backend: (False, {}),
    )
    backend = _FakeInferenceBackend()
    backend.active_model_name = "org/resident-model"
    backend.models = {"org/resident-model": {}}
    backend.loading_models = set()
    monkeypatch.setattr(inference_route, "get_inference_backend", lambda: backend)
    monkeypatch.setattr(inference_route, "load_inference_config", lambda _model: None)
    attempt = inference_route._ScopedLoadAttempt(
        token = "attempt",
        request_id = None,
        model_path = "org/incoming-model",
        subject = "test",
        cancel_event = threading.Event(),
        cancel_complete = threading.Event(),
    )
    monkeypatch.setattr(inference_route, "_running_load_attempt", attempt)

    response = asyncio.run(inference_route.get_status(current_subject = "test"))

    assert response.active_model == "org/resident-model"
    assert response.model_identifier == "org/resident-model"
    assert response.loaded == ["org/resident-model"]
    assert response.loading == ["org/incoming-model"]


def test_status_reports_gguf_load_while_serial_lock_is_held(monkeypatch):
    """The RLock-compatible probe exposes direct GGUF loads after preflight."""
    lock = threading.RLock()
    entered = threading.Event()
    release = threading.Event()

    def _hold_lock():
        with lock:
            entered.set()
            release.wait(_GUARD_SECONDS)

    holder = threading.Thread(target = _hold_lock)
    holder.start()
    assert entered.wait(_GUARD_SECONDS)
    backend = _FakeLlamaBackend()
    backend._serial_load_lock = lock
    backend._model_identifier = "org/direct-GGUF"
    monkeypatch.setattr(inference_route, "get_llama_cpp_backend", lambda: backend)
    monkeypatch.setattr(
        inference_route,
        "_probe_llama_cpp_status",
        lambda _backend: (False, {}),
    )
    monkeypatch.setattr(inference_route, "_running_load_attempt", None)
    try:
        response = asyncio.run(inference_route.get_status(current_subject = "test"))
    finally:
        release.set()
        holder.join(_GUARD_SECONDS)

    assert not holder.is_alive()
    assert response.active_model is None
    assert response.is_gguf is True
    assert response.loading == ["org/direct-GGUF"]


def _queued_attempt(model_path: str, token: str = "t") -> "inference_route._ScopedLoadAttempt":
    return inference_route._ScopedLoadAttempt(
        token = token,
        request_id = None,
        model_path = model_path,
        subject = "test",
        cancel_event = threading.Event(),
        cancel_complete = threading.Event(),
    )


def _load_request(model_path: str):
    return types.SimpleNamespace(
        model_path = model_path,
        load_request_id = None,
        force_cancel_active = False,
    )


def test_status_sees_a_load_still_queued_on_the_lifecycle_gate(monkeypatch):
    """/load builds its attempt before queueing on inference_lifecycle_gate, which an
    unload, a media auto-switch or a transformers install can hold for minutes.
    _running_load_attempt is assigned only inside that gate, so reading it alone
    reported the wait as idle and the chat UI auto-loaded a default over the queued
    request or adopted the model that was on its way out."""
    from core.inference.llama_keepwarm import inference_lifecycle_gate

    _patch_status_dependencies(monkeypatch)
    monkeypatch.setattr(inference_route, "_probe_llama_cpp_status", lambda _backend: (False, {}))
    monkeypatch.setattr(inference_route, "_raise_if_sidecar_swap_in_progress", lambda: None)
    monkeypatch.setattr(inference_route, "_running_load_attempt", None)

    async def _never_runs(*_args, **_kwargs):
        raise AssertionError("the impl must not run while the gate is held")

    monkeypatch.setattr(inference_route, "_run_tracked_load_model_impl", _never_runs)

    # Track progress through the real _begin_load_attempt, not through the registry
    # this fix adds, so the test fails on the old code as a wrong status rather than
    # as a missing attribute.
    _real_begin = inference_route._begin_load_attempt
    begun: list = []

    def _spy(request, subject):
        attempt = _real_begin(request, subject)
        begun.append(attempt)
        return attempt

    monkeypatch.setattr(inference_route, "_begin_load_attempt", _spy)

    async def _drive():
        async with inference_lifecycle_gate():
            task = asyncio.create_task(
                inference_route.load_model_gated(
                    _load_request("org/queued-model-GGUF"), object(), "test"
                )
            )
            # Everything before the gate is synchronous, so once the attempt exists the
            # next turn parks the task on the gate. Yield rather than guess a sleep.
            for _ in range(_CONTROL_TURNS):
                await asyncio.sleep(0)
                if begun:
                    break
            for _ in range(_CONTROL_TURNS):
                await asyncio.sleep(0)
            status = await inference_route.get_status(current_subject = "test")
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, AssertionError):
            pass
        return status

    response = asyncio.run(asyncio.wait_for(_drive(), _GUARD_SECONDS))

    assert begun, "the load never started"
    assert response.active_model is None
    assert response.loading == ["org/queued-model-GGUF"]


def test_status_prefers_the_running_load_over_one_still_queued(monkeypatch):
    """The attempt holding the gate is the one actually loading; a queued sibling must
    not relabel it, or the UI hydrates capabilities for the wrong model."""
    _patch_status_dependencies(monkeypatch)
    monkeypatch.setattr(inference_route, "_probe_llama_cpp_status", lambda _backend: (False, {}))
    monkeypatch.setattr(
        inference_route, "_running_load_attempt", _queued_attempt("org/running-model", "a")
    )
    monkeypatch.setattr(
        inference_route,
        "_pending_load_attempts",
        {"b": _queued_attempt("org/queued-model", "b")},
    )

    response = asyncio.run(inference_route.get_status(current_subject = "test"))

    assert response.loading == ["org/running-model"]


def test_a_failed_load_clears_its_pending_attempt(monkeypatch):
    """A leak here pins a phantom loading row for the life of the process, which the
    frontend reads as a load that never finishes."""
    monkeypatch.setattr(inference_route, "_raise_if_sidecar_swap_in_progress", lambda: None)
    seen: list[list[str]] = []

    async def _blow_up(*_args, **_kwargs):
        seen.append([a.model_path for a in inference_route._pending_load_attempts.values()])
        raise RuntimeError("load blew up")

    monkeypatch.setattr(inference_route, "_run_tracked_load_model_impl", _blow_up)

    with pytest.raises(RuntimeError):
        asyncio.run(
            inference_route.load_model_gated(
                _load_request("org/model-GGUF"), object(), "test", user_initiated = True
            )
        )

    assert seen == [["org/model-GGUF"]], "the attempt must be visible while the load runs"
    assert inference_route._pending_load_attempts == {}, "a failed load must not leak"
