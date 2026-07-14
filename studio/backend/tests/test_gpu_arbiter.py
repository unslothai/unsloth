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


def test_release_if_drops_only_when_predicate_true(calls):
    arb.acquire_for(arb.DIFFUSION)
    # Predicate false -> ownership kept.
    assert arb.release_if(arb.DIFFUSION, lambda: False) is False
    assert arb.current_owner() == arb.DIFFUSION
    # Predicate true -> ownership dropped.
    assert arb.release_if(arb.DIFFUSION, lambda: True) is True
    assert arb.current_owner() is None


def test_release_if_by_non_owner_is_noop(calls):
    arb.acquire_for(arb.CHAT)
    # Predicate is never even consulted for a non-owner; ownership is untouched.
    consulted: list[bool] = []
    assert arb.release_if(arb.DIFFUSION, lambda: consulted.append(True) or True) is False
    assert consulted == []
    assert arb.current_owner() == arb.CHAT


def test_release_if_predicate_sees_a_reregistered_same_owner_load(calls):
    # The race release_if closes: a slow unload's predicate reports a load now in flight (a
    # re-registered same-owner load), so ownership must stay with DIFFUSION.
    arb.acquire_for(arb.DIFFUSION)
    loading = {"in_flight": True}
    assert arb.release_if(arb.DIFFUSION, lambda: not loading["in_flight"]) is False
    assert arb.current_owner() == arb.DIFFUSION


def test_register_runs_under_ownership_and_returns_result(calls):
    # A register callback runs after ownership transfers (owner already set) and its
    # return value is forwarded -- the route uses this to register the in-flight load.
    seen_owner: list = []

    def register():
        seen_owner.append(arb.current_owner())
        return "status-dict"

    result = arb.acquire_for(arb.DIFFUSION, register)
    assert result == "status-dict"
    assert seen_owner == [arb.DIFFUSION]
    assert arb.current_owner() == arb.DIFFUSION


def test_register_failure_leaves_ownership_in_place(calls):
    # A failing register (e.g. begin_load reporting a load already in progress) propagates
    # but must not drop ownership -- the prior handoff (chat already evicted) stands.
    arb.acquire_for(arb.CHAT)

    def register():
        raise RuntimeError("A diffusion load is already in progress.")

    with pytest.raises(RuntimeError):
        arb.acquire_for(arb.DIFFUSION, register)
    assert calls == ["evict-chat"]
    assert arb.current_owner() == arb.DIFFUSION


def test_competing_acquire_blocks_until_register_completes(monkeypatch):
    # While DIFFUSION registers its load, a competing VIDEO acquire must block (not evict) until
    # the load is in-flight; holding the lock across register makes eviction never race it.
    import threading
    import time

    monkeypatch.setattr(arb, "_owner", None)
    evicted: list = []
    monkeypatch.setitem(arb._EVICTORS, arb.DIFFUSION, lambda: evicted.append("evict-diffusion"))
    monkeypatch.setitem(arb._EVICTORS, arb.VIDEO, lambda: evicted.append("evict-video"))

    in_register = threading.Event()
    release_register = threading.Event()

    def register():
        in_register.set()
        # Hold the arbiter lock here; a competing acquire_for(VIDEO) must block until we return.
        assert release_register.wait(2.0)
        return "loading"

    loader = threading.Thread(target = lambda: arb.acquire_for(arb.DIFFUSION, register))
    loader.start()
    assert in_register.wait(2.0)

    competitor_done = threading.Event()
    threading.Thread(
        target = lambda: (arb.acquire_for(arb.VIDEO), competitor_done.set()),
    ).start()

    # The competitor cannot evict DIFFUSION while register still holds the lock.
    time.sleep(0.1)
    assert evicted == []
    assert not competitor_done.is_set()

    # Let register finish; ownership is now safely registered, so the competitor proceeds.
    release_register.set()
    loader.join(2.0)
    assert competitor_done.wait(2.0)
    assert evicted == ["evict-diffusion"]
    assert arb.current_owner() == arb.VIDEO
