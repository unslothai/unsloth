# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Verify that slow inference status probes run off the event loop."""

from __future__ import annotations

import asyncio
import sys
import threading
from contextlib import asynccontextmanager
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


def _attempt(model_path: str):
    return inference_route._ScopedLoadAttempt(
        token = "attempt",
        request_id = None,
        model_path = model_path,
        subject = "test",
        cancel_event = threading.Event(),
        cancel_complete = threading.Event(),
    )


def _patch_fast_status(monkeypatch, backend = None):
    _patch_status_dependencies(monkeypatch)
    monkeypatch.setattr(
        inference_route,
        "_probe_llama_cpp_status",
        lambda _backend: (False, {}),
    )
    monkeypatch.setattr(inference_route, "_peek_inference_backend", lambda: backend)
    monkeypatch.setattr(inference_route, "_running_load_attempt", None)
    monkeypatch.setattr(inference_route, "_pending_load_attempts", {})


def test_status_reports_a_load_queued_on_the_lifecycle_gate(monkeypatch):
    _patch_fast_status(monkeypatch)
    monkeypatch.setattr(
        inference_route,
        "_pending_load_attempts",
        {"attempt": _attempt("org/slow-model-GGUF")},
    )

    response = asyncio.run(inference_route.get_status(current_subject = "test"))

    assert response.active_model is None
    assert response.loading == ["org/slow-model-GGUF"]


def test_status_keeps_the_resident_model_visible_during_a_load(monkeypatch):
    backend = _FakeInferenceBackend()
    backend.active_model_name = "org/resident-model"
    backend.models = {"org/resident-model": {}}
    backend.loading_models = set()
    _patch_fast_status(monkeypatch, backend)
    monkeypatch.setattr(inference_route, "load_inference_config", lambda _model: None)
    monkeypatch.setattr(
        inference_route,
        "_running_load_attempt",
        _attempt("org/incoming-model"),
    )

    response = asyncio.run(inference_route.get_status(current_subject = "test"))

    assert response.active_model == "org/resident-model"
    assert response.model_identifier == "org/resident-model"
    assert response.loaded == ["org/resident-model"]
    assert response.loading == ["org/incoming-model"]


def test_load_is_registered_before_the_lifecycle_gate_and_always_cleared(monkeypatch):
    from core.inference import llama_keepwarm

    monkeypatch.setattr(inference_route, "_raise_if_sidecar_swap_in_progress", lambda: None)
    monkeypatch.setattr(inference_route, "_pending_load_attempts", {})
    seen = []

    @asynccontextmanager
    async def _gate():
        seen.append([a.model_path for a in inference_route._pending_load_attempts.values()])
        yield

    async def _fail(*_args, **_kwargs):
        raise RuntimeError("load failed")

    monkeypatch.setattr(llama_keepwarm, "inference_lifecycle_gate", _gate)
    monkeypatch.setattr(inference_route, "_run_tracked_load_model_impl", _fail)

    request = inference_route.LoadRequest(model_path = "org/queued-model-GGUF")
    with pytest.raises(RuntimeError):
        asyncio.run(inference_route.load_model_gated(request, object(), "test"))

    assert seen == [["org/queued-model-GGUF"]]
    assert inference_route._pending_load_attempts == {}


def test_status_reports_an_on_device_load_by_its_public_id(monkeypatch):
    """A load still on its way to the backend must not publish the on-disk path the
    completed load is careful to keep out of the same response."""
    _patch_fast_status(monkeypatch)
    monkeypatch.setattr(
        inference_route,
        "_running_load_attempt",
        _attempt("/home/alice/models/Qwen3-30B-A3B-Q4_K_M.gguf"),
    )

    response = asyncio.run(inference_route.get_status(current_subject = "test"))

    assert response.loading == ["Qwen3-30B-A3B-Q4_K_M"]


def test_status_reports_a_leased_native_load_by_its_registered_label(monkeypatch):
    """Once the grant is redeemed the label is what every other field reports."""
    from utils import native_path_leases

    path = "/home/alice/Downloads/private-model.gguf"
    native_path_leases._remember_native_path_for_redaction(path, "private-model")
    try:
        _patch_fast_status(monkeypatch)
        monkeypatch.setattr(inference_route, "_running_load_attempt", _attempt(path))

        response = asyncio.run(inference_route.get_status(current_subject = "test"))
    finally:
        with native_path_leases._REDACTION_LOCK:
            native_path_leases._NATIVE_PATH_LABELS.pop(path, None)
            if path in native_path_leases._NATIVE_PATH_REDACTIONS:
                native_path_leases._NATIVE_PATH_REDACTIONS.remove(path)

    assert response.loading == ["private-model"]


def test_status_leaves_a_hub_repo_id_alone(monkeypatch):
    """The redaction only has to reach paths; a repo id is already public."""
    _patch_fast_status(monkeypatch)
    monkeypatch.setattr(
        inference_route,
        "_running_load_attempt",
        _attempt("unsloth/gemma-4-E2B-it-GGUF"),
    )

    response = asyncio.run(inference_route.get_status(current_subject = "test"))

    assert response.loading == ["unsloth/gemma-4-E2B-it-GGUF"]


def test_status_does_not_list_one_transformers_load_twice(monkeypatch):
    """The backend names the load it is running and the registry names the one the
    route accepted. Reporting the attempt by its public id must not un-merge them."""
    backend = _FakeInferenceBackend()
    backend.active_model_name = None
    backend.models = {}
    backend.loading_models = {"/home/alice/models/local-llama"}
    _patch_fast_status(monkeypatch, backend)
    monkeypatch.setattr(inference_route, "load_inference_config", lambda _model: None)
    monkeypatch.setattr(
        inference_route,
        "_running_load_attempt",
        _attempt("/home/alice/models/local-llama"),
    )

    response = asyncio.run(inference_route.get_status(current_subject = "test"))

    assert response.loading == ["/home/alice/models/local-llama"]
