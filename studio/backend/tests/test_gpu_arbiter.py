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


# Keep multiple models loaded set to When they fit / Always (gpu_residency.coreside_locked) next to Off.


@pytest.fixture
def cores(calls, monkeypatch):
    import sys
    import types

    import core.inference.gpu_residency as res
    import utils.hardware.hardware  # noqa: F401
    from core.inference import model_slots, video_minimax_h3
    from core.inference.api_monitor import api_monitor
    from state import active_generations

    env = types.SimpleNamespace(calls = calls, lifecycle = [], h3 = [], probes = [])
    for name, value in {"_residents": {}, "_media_needs": {}, "_accounts": {}, "_owner_account": None,
                        "_prior_account": None}.items():
        monkeypatch.setattr(arb, name, value)
    monkeypatch.setitem(arb._EVICTORS, arb.VIDEO, lambda: calls.append("evict-video"))
    for owner, kind in ((arb.CHAT, "chat"), (arb.DIFFUSION, "image"), (arb.VIDEO, "video")):
        monkeypatch.setitem(arb._SLOT_REPORTERS, owner, lambda kind = kind: [{"kind": kind, "model": f"org/{kind}"}])
    monkeypatch.setattr(arb, "_release_idle_video_servers", env.h3.append)
    monkeypatch.setattr(sys.modules["utils.hardware.hardware"], "get_parent_visible_gpu_ids", lambda: [])
    monkeypatch.setattr(res, "chat_load_in_flight", lambda: False)
    monkeypatch.setattr(res, "_media_busy", lambda owner: (False, False))
    rows = {gpu: {"free_mib": 8192, "total_mib": 24576} for gpu in (0, 1)}
    monkeypatch.setattr(res, "_free_mib_by_gpu", lambda: (rows, None))
    monkeypatch.setattr(model_slots, "eviction_victims", lambda *args, **kwargs: [])
    monkeypatch.setattr(video_minimax_h3, "release_h3_native_servers", lambda reason: 0)
    monkeypatch.setattr(api_monitor, "record_lifecycle", lambda **event: env.lifecycle.append(event))

    def fits(*blockers):
        # Short while any of ``blockers`` is still resident, so a re-check after an eviction passes.
        def probe(owner, need_mib):
            env.probes.append(owner)
            short = any(name in arb._view() for name in blockers)
            return not short, {"need_mib": {0: 6144}, "free_mib": {0: 2048},
                               "short_mib": {0: 5120} if short else {}, "reason": "short" if short else "fits"}
        monkeypatch.setattr(res, "can_load_alongside", probe)

    def no_probe(owner, need_mib):
        raise AssertionError("can_load_alongside must not run here")

    env.fits = fits
    env.no_probe = lambda: monkeypatch.setattr(res, "can_load_alongside", no_probe)
    active_generations.reset_for_tests()
    yield env
    active_generations.reset_for_tests()


CHAT_SLOT, IMAGE_SLOT = {"kind": "chat", "model": "org/chat"}, {"kind": "image", "model": "org/image"}


@pytest.mark.parametrize(
    "policy, need, short, evicts",
    [
        ("off", {0: 4096}, False, True),  # today's swap
        ("fit", {0: 4096}, False, False),  # fits: both stay
        ("fit", {0: 4096}, True, True),  # short: chat freed, cheapest first
        ("fit", None, False, True),  # unpriced: evict first, as Off does
        ("always", {0: 4096}, False, False),  # fits: joins
    ],
)
def test_image_load_beside_a_resident_chat_model_policy_matrix(cores, policy, need, short, evicts):
    cores.fits(*([arb.CHAT] if short else []))
    arb.acquire_for(arb.CHAT, gpu_ids = [0])
    evicted: list = []
    arb.acquire_for(arb.DIFFUSION, lambda: None, policy = policy, need_mib = need, evicted = evicted)
    assert cores.calls == (["evict-chat"] if evicts else [])
    assert set(arb.residents()) == ({arb.DIFFUSION} if evicts else {arb.CHAT, arb.DIFFUSION})
    assert evicted == ([CHAT_SLOT] if evicts else [])
    # The fit path records each eviction in the API monitor; Off keeps today's silent swap.
    assert [event["reason"] for event in cores.lifecycle] == (["evicted"] if evicts and policy != "off" else [])
    assert arb._media_needs.get(arb.DIFFUSION) == (need if policy != "off" else None)


@pytest.mark.parametrize("managed", [False, True])
def test_always_refuses_an_image_load_that_does_not_fit_without_evicting(cores, managed):
    from utils.account_context import AccountContext, run_as

    cores.fits(arb.CHAT)
    arb.acquire_for(arb.CHAT, gpu_ids = [0])

    def load():
        arb.acquire_for(arb.DIFFUSION, lambda: None, policy = "always", need_mib = {0: 6144})

    with pytest.raises(arb.MultiModelAlwaysRefusal) as refused:
        run_as(AccountContext("b" * 32, "bob"), load) if managed else load()
    message = str(refused.value)
    if managed:
        assert "is set to keep loaded" in message and "GB" not in message
    else:
        assert message.startswith("This model needs about 6.0 GB of GPU memory, and 2.0 GB is free")
        assert "is set to Always" in message
    assert cores.calls == [] and cores.lifecycle == []
    assert set(arb.residents()) == {arb.CHAT}


@pytest.mark.parametrize("policy", ["off", "fit", "always"])
def test_image_load_on_another_gpu_never_evicts_chat_unless_off(cores, policy):
    cores.no_probe()
    arb.acquire_for(arb.CHAT, gpu_ids = [0])
    arb.acquire_for(arb.DIFFUSION, lambda: None, policy = policy, need_mib = {1: 4096})
    if policy == "off":
        assert cores.calls == ["evict-chat"]
        assert arb.residents() == {arb.DIFFUSION: frozenset({1})}
    else:
        assert cores.calls == []
        assert arb.residents() == {arb.CHAT: frozenset({0}), arb.DIFFUSION: frozenset({1})}


@pytest.mark.parametrize(
    "policy, priced, kwargs, short, outcome",
    [
        ("fit", True, {}, False, "join"),  # the launch prices the idle image model
        ("always", True, {}, False, "join"),
        ("fit", False, {}, False, "evict"),  # unpriced media can't be counted
        ("always", False, {}, False, "refuse"),
        ("fit", True, {"evict_first": True}, False, "evict"),  # no estimate before launch
        ("fit", True, {"need_mib": {0: 6144}}, False, "join"),  # audio.cpp, priced from native_required_gb
        ("fit", True, {"need_mib": {0: 6144}}, True, "evict"),
    ],
)
def test_chat_load_beside_an_idle_image_model(cores, policy, priced, kwargs, short, outcome):
    cores.fits(*([arb.DIFFUSION] if short else []))
    arb.acquire_for(arb.DIFFUSION, lambda: None, policy = "fit" if priced else "off",
                    need_mib = {0: 4096} if priced else None, gpu_ids = [0])
    assert (arb.DIFFUSION in arb._media_needs) is priced
    evicted: list = []

    def load():
        arb.acquire_for(arb.CHAT, gpu_ids = [0], policy = policy, evicted = evicted, **kwargs)

    if outcome == "refuse":
        with pytest.raises(arb.MultiModelAlwaysRefusal, match = "is set to Always"):
            load()
    else:
        load()
    assert cores.calls == (["evict-diffusion"] if outcome == "evict" else [])
    assert evicted == ([IMAGE_SLOT] if outcome == "evict" else [])
    expected = {"join": {arb.CHAT, arb.DIFFUSION}, "evict": {arb.CHAT}, "refuse": {arb.DIFFUSION}}[outcome]
    assert set(arb.residents()) == expected


@pytest.mark.parametrize(
    "policy, need, refused",
    [
        ("fit", {1: 4096}, False),  # Bob's chat is on GPU 0: nothing of his is evicted
        ("fit", {0: 4096}, True),  # short on Bob's GPU: his stream blocks the eviction
        ("off", {1: 4096}, True),  # Off keeps the global guard
    ],
)
def test_another_accounts_stream_only_blocks_evicting_its_own_resident(cores, policy, need, refused):
    import threading

    from state.active_generations import ActiveGeneration
    from utils.account_context import AccountContext, run_as

    alice, bob = AccountContext("a" * 32, "alice"), AccountContext("b" * 32, "bob")
    cores.fits(arb.CHAT)
    run_as(bob, arb.acquire_for, arb.CHAT, lambda: None, gpu_ids = [0])
    with ActiveGeneration(threading.Event(), run_id = "bob-chat", account_id = bob.account_id):
        def load():
            run_as(alice, arb.acquire_for, arb.DIFFUSION, lambda: None, policy = policy, need_mib = need)

        if refused:
            with pytest.raises(arb.GpuBusyForAnotherAccountError):
                load()
        else:
            load()
    assert cores.calls == []
    assert arb.holds(arb.CHAT) and arb.holds(arb.DIFFUSION) is not refused
    assert arb.owner_account(arb.CHAT) == bob.account_id


def test_free_for_collects_what_it_evicts_and_spares_other_gpus(cores):
    import core.inference.gpu_residency as res

    cores.fits(arb.CHAT)
    arb.acquire_for(arb.CHAT, gpu_ids = [0])
    arb.acquire_for(arb.VIDEO, lambda: None, policy = "fit", need_mib = {1: 4096})
    evicted: list = []
    freed = res.free_for(arb.DIFFUSION, [0], "test", evicted, need_mib = {0: 6144})
    assert freed == ["org/chat"] and evicted == [CHAT_SLOT]
    assert cores.calls == ["evict-chat"]
    assert set(arb.residents()) == {arb.VIDEO}
    assert cores.lifecycle == [{"event": "unload", "model": "org/chat", "reason": "evicted"}]


@pytest.mark.parametrize("video_resident", [True, False])
def test_a_fit_join_keeps_the_video_server_while_video_is_resident(cores, video_resident):
    cores.fits()
    arb.acquire_for(arb.CHAT, gpu_ids = [0])
    if video_resident:
        arb.acquire_for(arb.VIDEO, lambda: None, policy = "fit", need_mib = {1: 4096})
    cores.h3.clear()
    arb.acquire_for(arb.DIFFUSION, lambda: None, policy = "fit", need_mib = {0: 4096})
    assert cores.calls == []
    assert cores.h3 == ([] if video_resident else [f"GPU acquired for {arb.DIFFUSION}"])


@pytest.mark.parametrize("policy", ["always", "fit"])
def test_image_load_behind_an_in_flight_chat_load(cores, monkeypatch, policy):
    import core.inference.gpu_residency as res
    from fastapi import HTTPException

    cores.fits(arb.CHAT)
    monkeypatch.setattr(res, "chat_load_in_flight", lambda: True)
    arb.acquire_for(arb.CHAT, gpu_ids = [0])

    def load():
        arb.acquire_for_request(arb.DIFFUSION, lambda: None, policy = policy, need_mib = {0: 4096})

    if policy == "always":
        # Its size isn't known yet, so Always answers the retryable K3 409 instead of the Always refusal.
        with pytest.raises(HTTPException) as refused:
            load()
        assert refused.value.status_code == 409
        detail = refused.value.detail
        assert detail["error"] == "gpu_busy" and detail["retry_after"] == 15
        assert detail["message"].startswith("Unsloth Studio is loading a chat model")
        assert refused.value.headers == {"Retry-After": "15"}
        assert cores.calls == [] and set(arb.residents()) == {arb.CHAT}
    else:
        load()  # today's handoff
        assert cores.calls == ["evict-chat"] and set(arb.residents()) == {arb.DIFFUSION}


@pytest.mark.parametrize("owner", [arb.DIFFUSION, arb.VIDEO])
def test_a_lone_media_load_under_fit_never_probes(cores, owner):
    cores.no_probe()
    for _ in range(2):  # a reload of the same owner has no other resident either
        arb.acquire_for(owner, lambda: None, policy = "fit", need_mib = {0: 4096})
    assert cores.calls == []
    assert arb.residents() == {owner: frozenset({0})}
    assert arb._media_needs[owner] == {0: 4096}


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
    # Gated on chat_load_needs_gpu like the GGUF branch: a load that never took the
    # arbiter has no ownership to lose, and checking against a None owner would 409
    # every CPU-placed audio load.
    guard = tail.index("if chat_load_needs_gpu and current_owner() != CHAT:")
    assert "await asyncio.to_thread(backend.unload_model, config.identifier)" in tail[guard:]
    assert tail.index("status_code = 409", guard) > guard


def test_restore_owner_account_returns_residency_to_the_displaced_account(calls, monkeypatch):
    from utils.account_context import AccountContext, run_as

    alice, bob = AccountContext("a" * 32, "alice"), AccountContext("b" * 32, "bob")
    monkeypatch.setattr(arb, "_owner_account", None)
    monkeypatch.setattr(arb, "_prior_account", None)
    run_as(alice, arb.acquire_for, arb.VIDEO, lambda: None)
    run_as(bob, arb.acquire_for, arb.VIDEO, lambda: None)
    assert arb.owner_account() == bob.account_id
    assert run_as(bob, arb.restore_owner_account, arb.VIDEO) is True
    assert arb.owner_account() == alice.account_id
    assert run_as(alice, arb.restore_owner_account, arb.VIDEO) is False


def test_restore_owner_account_is_a_noop_once_someone_else_holds_residency(calls, monkeypatch):
    from utils.account_context import AccountContext, run_as

    alice, bob, carol = (
        AccountContext("a" * 32, "alice"),
        AccountContext("b" * 32, "bob"),
        AccountContext("c" * 32, "carol"),
    )
    monkeypatch.setattr(arb, "_owner_account", None)
    monkeypatch.setattr(arb, "_prior_account", None)
    run_as(alice, arb.acquire_for, arb.VIDEO, lambda: None)
    run_as(bob, arb.acquire_for, arb.VIDEO, lambda: None)
    run_as(carol, arb.acquire_for, arb.VIDEO, lambda: None)
    assert run_as(bob, arb.restore_owner_account, arb.VIDEO) is False
    assert arb.owner_account() == carol.account_id
