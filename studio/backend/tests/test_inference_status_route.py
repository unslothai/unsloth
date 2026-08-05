# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Verify that slow inference status probes run off the event loop."""

from __future__ import annotations

import asyncio
import sys
import threading
from pathlib import Path

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
