# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unit tests for the single-GPU arbiter.

The real evictors (which tear down live backends) are replaced with recorders, so
these verify only the ownership/eviction sequencing — no torch, GPU, or subprocess.
"""

from __future__ import annotations

import types

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
    # Gated on chat_load_needs_gpu like the GGUF branch: a load that never took the
    # arbiter has no ownership to lose, and checking against a None owner would 409
    # every CPU-placed audio load.
    guard = tail.index("if chat_load_needs_gpu and not holds(CHAT):")
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


# Multi-resident arbiter: Keep multiple models loaded policies, residents, eviction reports.

from utils.account_context import AccountContext, run_as

ALICE = AccountContext("a" * 32, "alice")
BOB = AccountContext("b" * 32, "bob")


@pytest.fixture
def arb_log(monkeypatch):
    log: list[str] = []
    for name, value in (
        ("_owner", None),
        ("_residents", {}),
        ("_accounts", {}),
        ("_owner_account", None),
        ("_prior_account", None),
    ):
        monkeypatch.setattr(arb, name, value)
    for owner in (arb.CHAT, arb.DIFFUSION, arb.VIDEO, arb.DECISIONS):
        monkeypatch.setitem(arb._EVICTORS, owner, lambda owner = owner: log.append(f"evict-{owner}"))
    for owner in (arb.CHAT, arb.DIFFUSION, arb.VIDEO):
        monkeypatch.setitem(arb._SLOT_REPORTERS, owner, lambda: [])
    monkeypatch.setattr(arb, "_release_idle_video_servers", lambda reason: log.append("release-h3"))
    return log


def _report(monkeypatch, log, owner, *slots):
    monkeypatch.setitem(
        arb._SLOT_REPORTERS, owner, lambda: (log.append(f"report-{owner}"), list(slots))[1]
    )


@pytest.mark.parametrize(
    "resident, owner, kind, model",
    [
        (arb.CHAT, arb.DIFFUSION, "chat", "unsloth/Qwen3-4B-GGUF"),
        (arb.DIFFUSION, arb.CHAT, "image", "black-forest-labs/FLUX.1-dev"),
        (arb.DIFFUSION, arb.VIDEO, "image", "black-forest-labs/FLUX.1-dev"),
    ],
)
def test_always_refuses_instead_of_evicting(arb_log, monkeypatch, resident, owner, kind, model):
    arb.acquire_for(resident)
    _report(monkeypatch, arb_log, resident, {"kind": kind, "model": model, "account": "owner"})
    arb_log.clear()
    registered: list[bool] = []
    with pytest.raises(arb.MultiModelAlwaysRefusal) as excinfo:
        arb.acquire_for(owner, lambda: registered.append(True), policy = "always")
    assert not [entry for entry in arb_log if entry.startswith("evict-")] and registered == []
    message = str(excinfo.value)
    assert f"the {kind} model {model}" in message and "is set to Always" in message
    assert (excinfo.value.kind, excinfo.value.model) == (kind, model)
    assert arb.residents() == {resident: None}

    http = excinfo.value.as_http_exception("/api/inference/load")
    assert http.status_code == 409 and http.headers == {"X-Unsloth-Refusal": "multi-model-always"}
    assert isinstance(http.detail, str) and "is set to Always" in http.detail
    v1 = excinfo.value.as_http_exception("/v1/chat/completions")
    assert v1.status_code == 409 and v1.detail["error"]["code"] == "multi_model_always"
    assert model in v1.detail["error"]["message"]


def test_always_refusal_for_a_managed_account_hides_another_accounts_model(arb_log, monkeypatch):
    arb.acquire_for(arb.CHAT)
    _report(
        monkeypatch,
        arb_log,
        arb.CHAT,
        {"kind": "chat", "model": "bob/private-GGUF", "account": BOB.account_id},
    )
    with pytest.raises(arb.MultiModelAlwaysRefusal) as excinfo:
        run_as(ALICE, arb.acquire_for, arb.DIFFUSION, policy = "always")
    message = str(excinfo.value)
    assert "bob/private-GGUF" not in message and "the chat model," in message
    assert "is set to keep loaded" in message and "is set to Always" not in message
    # Alice's own model stays named.
    _report(
        monkeypatch,
        arb_log,
        arb.CHAT,
        {"kind": "chat", "model": "alice/mine-GGUF", "account": ALICE.account_id},
    )
    with pytest.raises(arb.MultiModelAlwaysRefusal, match = "alice/mine-GGUF"):
        run_as(ALICE, arb.acquire_for, arb.DIFFUSION, policy = "always")


@pytest.mark.parametrize("policy", ["off", "fit"])
def test_off_and_fit_evict_then_release_idle_video_then_register(arb_log, policy):
    arb.acquire_for(arb.DIFFUSION)
    arb_log.clear()
    arb.acquire_for(arb.CHAT, lambda: arb_log.append("register"), **arb.policy_kwargs(policy))
    assert arb_log == ["evict-diffusion", "release-h3", "register"]
    assert arb.residents() == {arb.CHAT: None}
    assert arb.policy_kwargs("off") == {} and arb.policy_kwargs("always") == {"policy": "always"}


def test_always_still_evicts_decisions_and_allow_evict_false_still_refuses(arb_log):
    arb.acquire_for(arb.DECISIONS)
    arb.acquire_for(arb.CHAT, policy = "always")
    assert "evict-decisions" in arb_log and arb.residents() == {arb.CHAT: None}
    arb.acquire_for(arb.DECISIONS, policy = "always")
    assert "evict-chat" in arb_log
    arb.acquire_for(arb.CHAT)
    with pytest.raises(arb.GpuOwnerBusyError) as excinfo:
        arb.acquire_for(arb.DECISIONS, allow_evict = False, policy = "always")
    assert excinfo.value.owner == arb.CHAT


def test_residents_use_physical_gpu_ids_under_a_masked_device_list(arb_log, monkeypatch):
    import utils.hardware.hardware as hardware

    monkeypatch.setattr(hardware, "get_parent_visible_gpu_ids", lambda: [2, 3])
    arb.acquire_for(arb.CHAT, gpu_ids = [0])
    assert arb.residents() == {"chat": frozenset({2})}
    assert arb.physical_gpu_ids([1, 5]) == frozenset({3, 5}) and arb.physical_gpu_ids(None) is None
    monkeypatch.setattr(hardware, "get_parent_visible_gpu_ids", lambda: [])
    assert arb.physical_gpu_ids([0]) == frozenset({0})


def test_a_directly_set_owner_wins_over_stale_residents(arb_log, monkeypatch):
    monkeypatch.setattr(arb, "_owner", arb.DIFFUSION)
    assert arb.holds(arb.DIFFUSION) and arb.residents() == {arb.DIFFUSION: None}
    monkeypatch.setattr(arb, "_residents", {arb.CHAT: None})
    monkeypatch.setattr(arb, "_owner", None)
    assert arb.residents() == {} and not arb.holds(arb.CHAT)
    monkeypatch.setattr(arb, "current_owner", lambda: arb.CHAT)
    assert arb.holds(arb.CHAT) and arb.residents() == {arb.CHAT: None}


def test_release_and_release_if_keep_owner_and_residents_in_sync(arb_log, monkeypatch):
    monkeypatch.setattr(arb, "_residents", {arb.CHAT: frozenset({0}), arb.DIFFUSION: None})
    monkeypatch.setattr(arb, "_owner", arb.DIFFUSION)
    arb.release(arb.DIFFUSION)
    assert arb.current_owner() == arb.CHAT and arb.residents() == {arb.CHAT: frozenset({0})}
    assert arb.release_if(arb.CHAT, lambda: True) is True
    assert arb.current_owner() is None and arb.residents() == {} and arb._residents == {}


def test_evicted_collects_chat_primary_and_kept_slots_image_and_video(arb_log, monkeypatch):
    chat = [
        {"kind": "chat", "model": "org/A-GGUF", "account": "owner"},
        {"kind": "chat", "model": "org/B-GGUF", "account": "owner"},
    ]
    _report(monkeypatch, arb_log, arb.CHAT, *chat)
    _report(
        monkeypatch,
        arb_log,
        arb.DIFFUSION,
        {"kind": "image", "model": "org/Img", "account": "owner"},
    )
    _report(
        monkeypatch, arb_log, arb.VIDEO, {"kind": "video", "model": "org/Vid", "account": "owner"}
    )
    arb.acquire_for(arb.CHAT)
    evicted: list = []
    arb.acquire_for(arb.DIFFUSION, evicted = evicted)
    assert evicted == [
        {"kind": "chat", "model": "org/A-GGUF"},
        {"kind": "chat", "model": "org/B-GGUF"},
    ]
    assert arb_log.index("report-chat") < arb_log.index("evict-chat")
    # Through the sink: the idle H3 release and non-resident video report nothing.
    sink: list = []
    token = arb.eviction_sink.set(sink)
    try:
        arb.acquire_for(arb.VIDEO)
        arb.acquire_for(arb.CHAT)
    finally:
        arb.eviction_sink.reset(token)
    assert sink == [{"kind": "image", "model": "org/Img"}, {"kind": "video", "model": "org/Vid"}]
    assert "release-h3" in arb_log
    arb.note_evictions([{"kind": "chat", "model": "x", "account": None}], evicted)
    assert evicted[-1] == {"kind": "chat", "model": "x"}


def test_owner_account_and_restore_are_tracked_per_owner(arb_log):
    run_as(ALICE, arb.acquire_for, arb.VIDEO, lambda: None)
    run_as(BOB, arb.acquire_for, arb.VIDEO, lambda: None)
    # Two residents at once (as a "fit" acquire leaves them): CHAT claimed by alice beside bob's VIDEO.
    arb._residents[arb.CHAT] = None
    arb._accounts[arb.CHAT] = (ALICE.account_id, None)
    assert arb.owner_account(arb.VIDEO) == BOB.account_id
    assert arb.owner_account(arb.CHAT) == ALICE.account_id
    assert run_as(ALICE, arb.restore_owner_account, arb.CHAT) is False  # no prior account
    assert run_as(BOB, arb.restore_owner_account, arb.VIDEO) is True
    assert arb.owner_account(arb.VIDEO) == ALICE.account_id
    assert arb.owner_account(arb.CHAT) == ALICE.account_id
    arb._accounts[arb.CHAT] = (BOB.account_id, ALICE.account_id)
    assert run_as(BOB, arb.restore_owner_account, arb.CHAT) is True
    assert arb.owner_account(arb.CHAT) == ALICE.account_id


def test_always_evicts_a_media_claim_that_holds_nothing_unless_it_is_loading(arb_log, monkeypatch):
    # An image load that failed in the background keeps its claim yet holds no model.
    probe = {"loading": False, "broken": False}

    def loading(owner):
        if probe["broken"]:
            raise RuntimeError("status unreadable")
        return probe["loading"]

    monkeypatch.setattr(arb, "_media_status", lambda owner: None)
    monkeypatch.setattr(arb, "_media_loading", loading)
    arb.acquire_for(arb.DIFFUSION)
    # Loading, or unreadable: Always keeps it.
    for key in ("loading", "broken"):
        probe[key] = True
        with pytest.raises(arb.MultiModelAlwaysRefusal):
            arb.acquire_for(arb.CHAT, policy = "always")
        probe[key] = False
    arb_log.clear()
    arb.acquire_for(arb.CHAT, policy = "always")
    assert "evict-diffusion" in arb_log and arb.residents() == {arb.CHAT: None}


@pytest.mark.parametrize(
    "policy, requested, refused",
    [
        ("always", ("org/Other",), True),
        ("always", ("org/Flux",), False),
        ("always", ("/models/flux", "org/Flux"), False),
        ("fit", ("org/Other",), False),
    ],
)
def test_always_refuses_replacing_a_loaded_media_model_with_another_repo(
    arb_log, monkeypatch, policy, requested, refused
):
    monkeypatch.setattr(arb, "_media_status", lambda owner: {"loaded": True, "repo_id": "org/Flux"})
    _report(monkeypatch, arb_log, arb.DIFFUSION, {"kind": "image", "model": "org/Flux"})
    if refused:
        with pytest.raises(arb.MultiModelAlwaysRefusal) as excinfo:
            arb.refuse_replacing_media(arb.DIFFUSION, policy, requested)
        assert (excinfo.value.kind, excinfo.value.model) == ("image", "org/Flux")
    else:
        # Off and fit replace it, and the load reports it.
        replaced = arb.refuse_replacing_media(arb.DIFFUSION, policy, requested)
        assert replaced == ([{"kind": "image", "model": "org/Flux"}] if policy == "fit" else [])
    monkeypatch.setattr(arb, "_media_status", lambda owner: None)
    monkeypatch.setattr(arb, "_media_loading", lambda owner: False)
    assert arb.refuse_replacing_media(arb.VIDEO, "always", ("org/Other",)) == []


def test_a_media_load_registered_meanwhile_is_not_replaced_under_always(arb_log, monkeypatch):
    # Two loads checked before either registered: the second re-checks under the arbiter lock.
    monkeypatch.setattr(arb, "_media_status", lambda owner: None)
    monkeypatch.setattr(arb, "_media_loading", lambda owner: True)
    monkeypatch.setattr(
        arb,
        "_media_backend",
        lambda owner: types.SimpleNamespace(loading_repo_ids = lambda: ("org/First",)),
    )
    begun = []

    def begin():
        begun.append(True)
        return "started"

    second = arb.guarded_media_begin(arb.DIFFUSION, "always", ("org/Second",), begin)
    with pytest.raises(arb.MultiModelAlwaysRefusal):
        arb.acquire_for(arb.DIFFUSION, second, policy = "always")
    with pytest.raises(arb.MultiModelAlwaysRefusal):
        arb.begin_media_load(arb.DIFFUSION, "always", ("org/Second",), begin)
    assert begun == []
    assert arb.begin_media_load(arb.DIFFUSION, "always", ("org/First",), begin) == "started"
    assert arb.guarded_media_begin(arb.DIFFUSION, "fit", ("org/Second",), begin) is begin

    def unreadable(owner):
        raise RuntimeError("status unreadable")

    monkeypatch.setattr(arb, "_media_status", unreadable)
    with pytest.raises(arb.MultiModelAlwaysRefusal):
        arb.refuse_replacing_media(arb.DIFFUSION, "always", ("org/First",))
    assert arb.refuse_replacing_media(arb.DIFFUSION, "off", ("org/First",)) == []


def test_always_evicts_a_chat_claim_that_holds_nothing_unless_a_chat_load_is_in_flight(
    arb_log, monkeypatch
):
    busy = {"now": True}
    monkeypatch.setattr(arb, "_chat_occupied", lambda: busy["now"])
    arb.acquire_for(arb.CHAT)
    with pytest.raises(arb.MultiModelAlwaysRefusal):
        arb.acquire_for(arb.DIFFUSION, policy = "always")
    busy["now"] = False
    arb_log.clear()
    arb.acquire_for(arb.DIFFUSION, policy = "always")
    assert "evict-chat" in arb_log and arb.residents() == {arb.DIFFUSION: None}
