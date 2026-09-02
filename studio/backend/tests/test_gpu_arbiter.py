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


def test_chat_load_can_refuse_to_evict_diffusion(calls):
    arb.acquire_for(arb.DIFFUSION)
    with pytest.raises(arb.GpuOwnerBusyError) as excinfo:
        arb.acquire_for(arb.CHAT, allow_evict = False)
    assert excinfo.value.owner == arb.DIFFUSION
    assert calls == []
    assert arb.current_owner() == arb.DIFFUSION


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
    # A chat model still starting up is is_active but not yet is_loaded, and eviction must still unload it or the load keeps allocating VRAM after the handover.
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
    # The predicate is never consulted for a non-owner; ownership is untouched.
    consulted: list[bool] = []
    assert arb.release_if(arb.DIFFUSION, lambda: consulted.append(True) or True) is False
    assert consulted == []
    assert arb.current_owner() == arb.CHAT


def test_release_if_predicate_sees_a_reregistered_same_owner_load(calls):
    # The race release_if closes: a slow unload's predicate reports a load now in flight, so ownership stays with DIFFUSION.
    arb.acquire_for(arb.DIFFUSION)
    loading = {"in_flight": True}
    assert arb.release_if(arb.DIFFUSION, lambda: not loading["in_flight"]) is False
    assert arb.current_owner() == arb.DIFFUSION


def test_register_runs_under_ownership_and_returns_result(calls):
    # A register callback runs after ownership transfers and its return value is forwarded; the route registers the in-flight load with it.
    seen_owner: list = []

    def register():
        seen_owner.append(arb.current_owner())
        return "status-dict"

    result = arb.acquire_for(arb.DIFFUSION, register)
    assert result == "status-dict"
    assert seen_owner == [arb.DIFFUSION]
    assert arb.current_owner() == arb.DIFFUSION


def test_register_failure_leaves_ownership_in_place(calls):
    # A failing register (e.g. begin_load reporting a load in progress) propagates but must not drop ownership: the prior handoff stands.
    arb.acquire_for(arb.CHAT)

    def register():
        raise RuntimeError("A diffusion load is already in progress.")

    with pytest.raises(RuntimeError):
        arb.acquire_for(arb.DIFFUSION, register)
    assert calls == ["evict-chat"]
    assert arb.current_owner() == arb.DIFFUSION


def test_competing_acquire_blocks_until_register_completes(monkeypatch):
    # While DIFFUSION registers its load, a competing VIDEO acquire must block (not evict) until the load is in-flight; holding the lock across register stops eviction racing it.
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


def test_evict_chat_cancels_a_chat_load_that_has_not_spawned_yet(monkeypatch):
    # An HF chat load has no llama-server process until its GGUF finished downloading, which takes minutes. Gating only on
    # is_active let the evictor find nothing to cancel and the chat load spawn onto the same device; the in-flight marker makes it cancellable.
    import core.inference as core_inference
    import routes.inference as routes_inference
    from core.inference.llama_cpp import chat_load_in_flight

    unloaded: list[bool] = []

    class _FakeLlama:
        is_active = False  # nothing spawned yet: the download is still running
        is_loaded = False

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

    # No load in flight: nothing to cancel.
    arb._evict_chat()
    assert unloaded == []

    with chat_load_in_flight():
        arb._evict_chat()
    assert unloaded == [True]

    # The marker is released with the load, so a later eviction is a no-op again.
    arb._evict_chat()
    assert unloaded == [True]


def test_evict_chat_cancels_an_in_flight_safetensors_load(monkeypatch):
    # The orchestrator publishes active_model_name only once its worker reports success, so an in-flight safetensors load is
    # visible ONLY in loading_models; gating the cancellation on active_model_name let that worker allocate alongside the new pipeline.
    import core.inference as core_inference
    import routes.inference as routes_inference

    cancelled: list[str] = []

    class _FakeLlama:
        is_active = False
        is_loaded = False

        def unload_model(self):
            pass

        def _wait_for_vram_settle(self, *, since_kill):
            pass

    class _FakeOrchestrator:
        active_model_name = None  # not published yet: the load is still running
        loading_models = {"unsloth/Qwen3-4B"}

        def unload_model(self, name):
            raise AssertionError("unload_model must not run for an unpublished load")

        def cancel_load(self, name):
            cancelled.append(name)
            return True

        def _shutdown_subprocess(self, timeout = 5.0):
            pass

    monkeypatch.setattr(routes_inference, "get_llama_cpp_backend", lambda: _FakeLlama())
    monkeypatch.setattr(core_inference, "get_inference_backend", lambda: _FakeOrchestrator())

    arb._evict_chat()

    assert cancelled == ["unsloth/Qwen3-4B"]


def test_evict_chat_cancels_every_pending_load_over_a_live_snapshot(monkeypatch):
    # cancel_load discards the marker it cancels, so iterate a snapshot (mutating the live set during iteration raises).
    import core.inference as core_inference
    import routes.inference as routes_inference

    cancelled: list[str] = []

    class _FakeLlama:
        is_active = False
        is_loaded = False

        def unload_model(self):
            pass

        def _wait_for_vram_settle(self, *, since_kill):
            pass

    class _FakeOrchestrator:
        active_model_name = None

        def __init__(self):
            self.loading_models = {"a/one", "b/two"}

        def cancel_load(self, name):
            self.loading_models.discard(name)
            cancelled.append(name)
            return True

        def _shutdown_subprocess(self, timeout = 5.0):
            pass

    orchestrator = _FakeOrchestrator()
    monkeypatch.setattr(routes_inference, "get_llama_cpp_backend", lambda: _FakeLlama())
    monkeypatch.setattr(core_inference, "get_inference_backend", lambda: orchestrator)

    arb._evict_chat()

    assert sorted(cancelled) == ["a/one", "b/two"]
    assert orchestrator.loading_models == set()


def test_the_safetensors_load_yields_a_gpu_it_lost_while_loading():
    # Mirror of the GGUF branch's guard: an Images/Video acquire can land between the eviction and the load's publish, so the load must undo itself, not leave two models resident.
    from pathlib import Path

    route_src = (Path(__file__).resolve().parent.parent / "routes" / "inference.py").read_text(
        encoding = "utf-8"
    )
    load_impl = route_src[route_src.index("async def _load_model_impl") :]
    unsloth_load = load_impl.index("success = await asyncio.to_thread(")
    tail = load_impl[unsloth_load:]
    guard = tail.index("if current_owner() != CHAT:")
    assert "await asyncio.to_thread(backend.unload_model, config.identifier)" in tail[guard:]
    assert tail.index("status_code = 409", guard) > guard
