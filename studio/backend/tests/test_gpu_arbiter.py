# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unit tests for the single-GPU arbiter.

The real evictors (which tear down live backends) are replaced with recorders, so
these verify only the ownership/eviction sequencing — no torch, GPU, or subprocess.
"""

from __future__ import annotations

import pytest

import core.inference.gpu_arbiter as arb


@pytest.fixture
def calls(monkeypatch):
    recorded: list[str] = []
    monkeypatch.setattr(arb, "_owner", None)
    monkeypatch.setitem(arb._EVICTORS, arb.CHAT, lambda: recorded.append("evict-chat"))
    monkeypatch.setitem(arb._EVICTORS, arb.DIFFUSION, lambda: recorded.append("evict-diffusion"))
    return recorded


def test_first_acquire_evicts_nothing(calls):
    arb.acquire_for(arb.CHAT)
    assert calls == []
    assert arb.current_owner() == arb.CHAT


def test_diffusion_load_evicts_chat(calls):
    arb.acquire_for(arb.CHAT)
    arb.acquire_for(arb.DIFFUSION)
    assert calls == ["evict-chat"]
    assert arb.current_owner() == arb.DIFFUSION


def test_chat_load_evicts_diffusion(calls):
    arb.acquire_for(arb.DIFFUSION)
    arb.acquire_for(arb.CHAT)
    assert calls == ["evict-diffusion"]
    assert arb.current_owner() == arb.CHAT


def test_reacquiring_same_owner_does_not_evict(calls):
    arb.acquire_for(arb.CHAT)
    arb.acquire_for(arb.CHAT)
    assert calls == []
    assert arb.current_owner() == arb.CHAT


def test_release_clears_owner(calls):
    arb.acquire_for(arb.DIFFUSION)
    arb.release(arb.DIFFUSION)
    assert arb.current_owner() is None
    # A subsequent chat load then has nothing to evict.
    arb.acquire_for(arb.CHAT)
    assert calls == []


def test_release_by_non_owner_is_noop(calls):
    arb.acquire_for(arb.CHAT)
    arb.release(arb.DIFFUSION)
    assert arb.current_owner() == arb.CHAT


def test_unknown_owner_raises(calls):
    with pytest.raises(ValueError):
        arb.acquire_for("gpu")


def test_evict_chat_unloads_a_still_loading_chat_backend(monkeypatch):
    # A chat model still starting up is is_active (process exists) but not yet
    # is_loaded (healthy). Eviction must still unload it, or the load would keep
    # allocating VRAM after the GPU was handed to diffusion.
    import core.inference as core_inference
    import routes.inference as routes_inference

    unloaded: list[bool] = []

    class _FakeLlama:
        is_active = True
        is_loaded = False  # still loading: skipped if eviction gates on is_loaded

        def unload_model(self):
            unloaded.append(True)

        def _wait_for_vram_settle(self, *, since_kill):
            pass

    class _FakeOrchestrator:
        active_model_name = None

        def unload_model(self, name):
            pass

        def _shutdown_subprocess(self, timeout = 5.0):
            pass

    monkeypatch.setattr(routes_inference, "get_llama_cpp_backend", lambda: _FakeLlama())
    monkeypatch.setattr(core_inference, "get_inference_backend", lambda: _FakeOrchestrator())

    arb._evict_chat()

    assert unloaded == [True]  # still-loading chat backend was unloaded, not skipped
