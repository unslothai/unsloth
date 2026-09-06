# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Deterministic account/GPU simulations and real seam tests, without a GPU.

Backends are recorders; the arbiter, generation registry, admission queue and
route guards are real. Streams advance only at explicit test checkpoints.
"""

from __future__ import annotations

import asyncio
import itertools
import threading
import time
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from auth import policy
from core.inference import gpu_arbiter as arb
from core.inference import llama_admission as admission
from core.inference import llama_keepwarm as keepwarm
from core.inference import media_auto_switch as media
from core.inference import media_keepwarm
from core.inference import media_switch_backends as media_backends
from core.inference import local_model_resolver as resolver
from core.inference.media_model_index import MediaModelPick
from state import active_generations as generations
from utils import openai_auto_switch_settings as settings
from utils.account_context import OWNER, AccountContext, arun_as, current_account_id, run_as

ALICE = AccountContext("a" * 32, "alice")
BOB = AccountContext("b" * 32, "bob")
CAROL = AccountContext("c" * 32, "carol")
ACCOUNTS = (ALICE, BOB, CAROL)
BACKENDS = (arb.CHAT, arb.DIFFUSION, arb.VIDEO)


@pytest.fixture(autouse = True)
def isolated(monkeypatch):
    monkeypatch.setattr(policy, "installation_is_multi_user", lambda: True)
    monkeypatch.setattr(arb, "_owner", None)
    monkeypatch.setattr(arb, "_owner_account", None)
    monkeypatch.setattr(arb, "_owner_epoch", 0)
    monkeypatch.setattr(keepwarm, "_inflight", 0)
    monkeypatch.setattr(keepwarm, "_pending", 0)
    monkeypatch.setattr(keepwarm, "_admitted_inference", 0)
    generations.reset_for_tests()
    admission.reset_llama_admission_queues()
    yield
    generations.reset_for_tests()
    admission.reset_llama_admission_queues()


@pytest.fixture
def route():
    from routes import inference
    return inference


class Simulator:
    def __init__(self, monkeypatch):
        self.resident = None
        self.evictions = []
        self.runs = {}
        for owner in BACKENDS:
            monkeypatch.setitem(arb._EVICTORS, owner, self.evict)

    def evict(self):
        self.evictions.append(self.resident)
        self.resident = None

    def load(self, account, owner, model):
        if self.resident == (owner, model):
            run_as(account, arb.acquire_for_request, owner)
            return "reused"

        def register():
            if self.resident is not None:
                self.evict()
            self.resident = (owner, model)

        run_as(account, arb.acquire_for_request, owner, register)
        return "loaded"

    def start(self, account, name):
        event = threading.Event()
        tracker = run_as(
            account, generations.ActiveGeneration, event,
            thread_id = "same-client-id", run_id = name, model = self.resident[1],
        )
        tracker.__enter__()
        self.runs[name] = (tracker, event)
        return event

    def finish(self, name):
        tracker, _event = self.runs.pop(name)
        tracker.__exit__(None, None, None)

    def delete(self, account):
        # Account deletion must signal the immutable id, then let each run unwind.
        return generations.cancel_all(account.account_id)

    def training(self, account):
        def reserve():
            # Required integration: run this guard BEFORE the training route's
            # destructive cleanup, while its lifecycle/admission gate is held.
            arb.raise_if_other_accounts_active()
            arb.release(arb.DIFFUSION)
            arb.release(arb.VIDEO)
            self.evict()  # stand-in for the later chat teardown
            arb.release(arb.CHAT)
        run_as(account, reserve)


@pytest.fixture
def sim(monkeypatch):
    return Simulator(monkeypatch)


@pytest.mark.parametrize("first,second", tuple(itertools.permutations(ACCOUNTS, 2)))
@pytest.mark.parametrize("outgoing,incoming", tuple(itertools.product(BACKENDS, repeat = 2)))
def test_sim_different_models_busy_then_idle_swap(sim, first, second, outgoing, incoming):
    assert sim.load(first, outgoing, "one") == "loaded"
    event = sim.start(first, "stream")
    before = arb.owner_snapshot()
    with pytest.raises(HTTPException) as refused:
        sim.load(second, incoming, "two")
    assert refused.value.status_code == 409
    assert refused.value.detail["error"] == "gpu_busy"
    assert int(refused.value.headers["Retry-After"]) >= 1
    assert arb.owner_snapshot() == before
    assert arb.owner_account() == first.account_id
    assert sim.evictions == [] and not event.is_set()
    sim.finish("stream")
    assert sim.load(second, incoming, "two") == "loaded"
    assert sim.evictions == [(outgoing, "one")]
    assert sim.resident == (incoming, "two")


def test_sim_third_account_must_wait_for_both_streams(sim):
    sim.load(ALICE, arb.CHAT, "shared")
    a = sim.start(ALICE, "a")
    assert sim.load(BOB, arb.CHAT, "shared") == "reused"
    b = sim.start(BOB, "b")
    for finished in (None, "a"):
        if finished:
            sim.finish(finished)
        with pytest.raises(HTTPException):
            sim.load(CAROL, arb.CHAT, "different")
        assert not a.is_set() and not b.is_set()
        assert not sim.evictions
    sim.finish("b")
    assert sim.load(CAROL, arb.CHAT, "different") == "loaded"


@pytest.mark.parametrize("deleting", [False, True])
def test_sim_stop_and_delete_cancel_only_the_target(sim, route, deleting):
    sim.load(ALICE, arb.CHAT, "shared")
    a = sim.start(ALICE, "a")
    b = sim.start(BOB, "b")
    if deleting:
        assert sim.delete(ALICE) == 1
    else:
        with pytest.raises(HTTPException) as refused:
            run_as(ALICE, route._raise_or_cancel_active_generations, force = True, action = "Load")
        assert refused.value.detail["error"] == "gpu_busy"
    assert a.is_set() and not b.is_set()
    assert generations.foreign_count(BOB.account_id) == 1  # cancellation is not completion
    sim.finish("a")
    assert generations.count(BOB.account_id) == 1
    assert sim.load(CAROL, arb.CHAT, "shared") == "reused"
    sim.finish("b")


def test_sim_training_preflight_and_release_do_not_terminate_foreign_chat(sim):
    sim.load(ALICE, arb.CHAT, "shared")
    event = sim.start(ALICE, "a")
    # These are the two existing release calls in training.py. They neither
    # revoke CHAT nor cancel any generation, even when called by another user.
    run_as(BOB, arb.release, arb.DIFFUSION)
    run_as(BOB, arb.release, arb.VIDEO)
    assert arb.current_owner() == arb.CHAT
    with pytest.raises(arb.GpuBusyForAnotherAccountError):
        sim.training(BOB)
    assert not event.is_set() and sim.evictions == []
    sim.finish("a")
    sim.training(BOB)
    assert sim.evictions == [(arb.CHAT, "shared")]
    assert arb.current_owner() is None


@pytest.mark.parametrize("force,cancel", tuple(itertools.product([False, True], repeat = 2)))
def test_route_preflight_and_force_respect_accounts(route, sim, force, cancel):
    sim.load(ALICE, arb.CHAT, "shared")
    mine = sim.start(ALICE, "a")
    theirs = sim.start(BOB, "b")
    with pytest.raises(HTTPException) as refused:
        run_as(
            ALICE, route._raise_or_cancel_active_generations,
            force = force, cancel = cancel, action = "Replacing the model",
        )
    assert refused.value.status_code == 409
    assert refused.value.detail["error"] == "gpu_busy"
    assert mine.is_set() == (force and cancel)
    assert not theirs.is_set()
    body = str(refused.value.detail)
    for private in (BOB.account_id, "same-client-id", "shared"):
        assert private not in body
    sim.finish("a")
    sim.finish("b")


def test_route_own_stream_keeps_existing_conflict_and_force_shape(route, sim):
    sim.load(ALICE, arb.CHAT, "shared")
    event = sim.start(ALICE, "a")
    with pytest.raises(HTTPException) as refused:
        run_as(ALICE, route._raise_or_cancel_active_generations, force = False, action = "Load")
    assert refused.value.detail["error"] == "active_generations"
    assert refused.value.detail["thread_ids"] == ["same-client-id"]
    assert run_as(ALICE, route._raise_or_cancel_active_generations, force = True, action = "Load") == 1
    assert event.is_set()
    sim.finish("a")


@pytest.mark.parametrize("cancel_pending", [False, True])
def test_wait_refuses_foreign_even_when_counters_miss_it_or_deadline_expires(route, sim, cancel_pending):
    sim.load(ALICE, arb.CHAT, "shared")
    event = sim.start(ALICE, "a")
    with pytest.raises(HTTPException) as refused:
        asyncio.run(arun_as(BOB, route._wait_for_model_switch_idle(
            current_request_counted = True, cancel_pending = cancel_pending, timeout_s = 0,
        )))
    assert refused.value.detail["error"] == "gpu_busy"
    assert not event.is_set()
    sim.finish("a")
    asyncio.run(arun_as(BOB, route._wait_for_model_switch_idle(current_request_counted = True)))


def test_wait_discounts_only_own_cancellable_work(route, sim, monkeypatch):
    sim.load(ALICE, arb.CHAT, "shared")
    event = sim.start(ALICE, "a")
    monkeypatch.setattr(keepwarm, "_inflight", 1)
    asyncio.run(arun_as(ALICE, route._wait_for_model_switch_idle(
        current_request_counted = False, cancel_pending = True,
    )))
    assert not event.is_set()
    sim.finish("a")


@pytest.mark.parametrize("previous,target,active", tuple(itertools.product((None, *BACKENDS), BACKENDS, [False, True])))
def test_single_account_arbiter_matches_legacy_decisions(sim, monkeypatch, previous, target, active):
    monkeypatch.setattr(policy, "installation_is_multi_user", lambda: False)
    if previous:
        sim.load(OWNER, previous, "one")
    if active:
        tracker = generations.ActiveGeneration(threading.Event())
        tracker.__enter__()
    try:
        def forbidden_retry():
            pytest.fail("Single-account load must not compute a busy hint")
        monkeypatch.setattr(admission, "estimate_gpu_retry_after", forbidden_retry)
        sim.load(OWNER, target, "two")
        assert sim.evictions == ([(previous, "one")] if previous else [])
        assert arb.current_owner() == target
    finally:
        if active:
            tracker.__exit__(None, None, None)


@pytest.mark.parametrize("active,force,cancel", tuple(itertools.product([False, True], repeat = 3)))
def test_single_account_cancel_matches_legacy_decisions(route, monkeypatch, active, force, cancel):
    monkeypatch.setattr(policy, "installation_is_multi_user", lambda: False)
    event = threading.Event()
    tracker = generations.ActiveGeneration(event, thread_id = "owner-chat")
    if active:
        tracker.__enter__()
    try:
        if active and not force:
            # This 409 already existed: it is the caller's own confirmation prompt.
            with pytest.raises(HTTPException) as refused:
                route._raise_or_cancel_active_generations(force = force, cancel = cancel, action = "Load")
            assert refused.value.detail["error"] == "active_generations"
            assert refused.value.headers is None
        else:
            assert route._raise_or_cancel_active_generations(
                force = force, cancel = cancel, action = "Load",
            ) == int(active and force and cancel)
        assert event.is_set() == (active and force and cancel)
    finally:
        if active:
            tracker.__exit__(None, None, None)


def test_single_account_wait_keeps_post_cancel_timeout(route, monkeypatch):
    monkeypatch.setattr(policy, "installation_is_multi_user", lambda: False)
    monkeypatch.setattr(keepwarm, "_inflight", 1)
    monkeypatch.setattr(generations, "foreign_count", lambda _account: pytest.fail("foreign scan on owner drain"))
    asyncio.run(route._wait_for_model_switch_idle(current_request_counted = False, timeout_s = 0))


def test_borrowing_a_durable_registration_requires_the_same_account():
    event = threading.Event()
    with run_as(ALICE, generations.ActiveGeneration, event, run_id = "same"):
        with run_as(BOB, generations.ActiveGeneration, event, run_id = "same"):
            assert generations.count(ALICE.account_id) == 1
            assert generations.count(BOB.account_id) == 1
        assert generations.count(ALICE.account_id) == 1
        with run_as(ALICE, generations.ActiveGeneration, event, run_id = "same"):
            assert generations.count() == 1
    assert generations.count() == 0


@pytest.mark.parametrize("cancel_queued", [False, True])
def test_real_admission_runs_accounts_concurrently_and_scopes_cancellation(cancel_queued):
    async def scenario():
        queue = admission.get_llama_admission_queue("one-llama-server")
        config = admission.llama_admission_config_from_env()
        entered = {name: asyncio.Event() for name in ("a", "b", "queued")}
        ready = {name: asyncio.Event() for name in entered}
        advance = {name: asyncio.Event() for name in entered}
        events = {name: threading.Event() for name in entered}
        streaming = {}
        reservations = {}

        async def stream(account, name):
            with generations.ActiveGeneration(events[name], run_id = name):
                assert current_account_id() == account.account_id
                reservation = queue.reserve(capacity = 2, config = config, tokens = 32, budget = 128)
                reservations[name] = reservation
                ready[name].set()
                try:
                    lease = reservation.lease_nowait()
                    if lease is None:
                        await advance[name].wait()
                        if events[name].is_set():
                            return
                        lease = await reservation.wait(1)
                    streaming[name] = lease.slot
                    entered[name].set()
                    await advance[name].wait()
                    assert current_account_id() == account.account_id
                finally:
                    streaming.pop(name, None)
                    reservation.cancel()

        tasks = []
        try:
            for account, name in ((ALICE, "a"), (BOB, "b"), (CAROL, "queued")):
                tasks.append(asyncio.create_task(arun_as(account, stream(account, name))))
                await ready[name].wait()
            assert entered["a"].is_set() and entered["b"].is_set()
            assert len(set(streaming.values())) == 2
            assert not entered["queued"].is_set()
            assert queue.snapshot().queued == 1
            if cancel_queued:
                assert generations.cancel_all(CAROL.account_id) == 1
                advance["queued"].set()
                await tasks[2]
                assert set(streaming) == {"a", "b"}
                assert queue.snapshot().queued == 0
            assert run_as(ALICE, generations.cancel_all, ALICE.account_id) == 1
            advance["a"].set()
            await tasks[0]
            assert not events["b"].is_set() and not tasks[1].done()
            if not cancel_queued:
                advance["queued"].set()
                await tasks[2]
                assert entered["queued"].is_set()
            advance["b"].set()
            await tasks[1]
            assert queue.is_idle() and generations.count() == 0
        finally:
            for event in advance.values():
                event.set()
            await asyncio.gather(*tasks, return_exceptions = True)
    asyncio.run(scenario())


@pytest.mark.parametrize("capacity,count,expected", [(2, 0, 15), (2, 2, 15), (2, 3, 30), (2, 7, 60), (1, 20, 120)])
def test_retry_hint_tracks_queue_waves_without_creating_queues(capacity, count, expected):
    async def scenario():
        assert admission.estimate_gpu_retry_after() == 15
        assert not admission._QUEUES
        queue = admission.get_llama_admission_queue("shared-server")
        config = admission.llama_admission_config_from_env()
        reservations = [queue.reserve(capacity = capacity, config = config) for _ in range(count)]
        try:
            assert admission.estimate_gpu_retry_after() == expected
            exc = arb.GpuBusyForAnotherAccountError(arb.CHAT, 1).as_http_exception()
            assert exc.detail["retry_after"] == expected
            assert exc.headers["Retry-After"] == str(expected)
        finally:
            for reservation in reservations:
                reservation.cancel()
    asyncio.run(scenario())


@pytest.mark.parametrize("kind", [arb.CHAT, arb.DIFFUSION, arb.VIDEO])
@pytest.mark.parametrize("multi", [False, True])
def test_global_idle_clock_follows_last_account_activity(monkeypatch, kind, multi):
    monkeypatch.setattr(policy, "installation_is_multi_user", lambda: multi)
    clock = [0.0]
    if kind == arb.CHAT:
        monkeypatch.setattr(keepwarm, "time", SimpleNamespace(monotonic = lambda: clock[0]))
        monkeypatch.setattr(keepwarm, "_last_active", 0.0)
        start, end, idle = keepwarm._note_start, keepwarm._note_end, keepwarm._is_idle
    else:
        monkeypatch.setattr(media_keepwarm, "time", SimpleNamespace(monotonic = lambda: clock[0]))
        tracker = media_keepwarm._Tracker(kind)
        start, end, idle = tracker.note_start, tracker.note_end, tracker.is_idle
    accounts = (ALICE, BOB) if multi else (OWNER, OWNER)
    run_as(accounts[0], start)
    clock[0] = 10.0
    run_as(accounts[1], start)
    clock[0] = 20.0
    run_as(accounts[0], end)
    assert not idle(5)
    clock[0] = 100.0
    assert not idle(5)
    run_as(accounts[1], end)
    clock[0] = 104.0
    assert not idle(5)
    clock[0] = 105.0
    assert idle(5)


@pytest.fixture
def chat_switch(route, monkeypatch):
    backend = SimpleNamespace(
        is_loaded = True, model_identifier = "org/one-GGUF", hf_variant = None,
        _openai_advertised_id = None,
    )
    loads = []
    monkeypatch.setattr(settings, "get_openai_auto_switch_enabled", lambda: True)
    monkeypatch.setattr(resolver, "warm_index_soon", lambda: None)
    monkeypatch.setattr(resolver, "resolve_trusted_cached_local_gguf", lambda name: (name, None, name))
    monkeypatch.setattr(resolver, "local_target_is_gguf", lambda *_args: True)
    monkeypatch.setattr(route, "get_llama_cpp_backend", lambda: backend)
    monkeypatch.setattr(route, "_auto_switch_waiters", {})
    monkeypatch.setattr(route, "_loaded_identity_satisfies", lambda name: backend.model_identifier == name)
    monkeypatch.setattr(route, "_claim_slot_for_non_preview", lambda *_args: None)
    # Visibility is decided by the account model grants, tested on their own.
    monkeypatch.setattr(route.account_access, "require_model_access", lambda *_args, **_kwargs: None)

    async def load(request, *_args, **_kwargs):
        loads.append(request.model_path)
        backend.model_identifier = request.model_path
    monkeypatch.setattr(route, "_load_model_impl", load)

    async def reject(*_args):
        pass
    monkeypatch.setattr(route, "_reject_unservable_model", reject)
    return backend, loads


@pytest.mark.parametrize("path", ["/v1/chat/completions", "/v1/messages"])
def test_chat_auto_switch_keeps_resident_parallel_and_refuses_foreign_swap(route, sim, chat_switch, path):
    backend, loads = chat_switch
    sim.load(ALICE, arb.CHAT, "org/one-GGUF")
    event = sim.start(ALICE, "a")
    request = SimpleNamespace(scope = {}, url = SimpleNamespace(path = path))
    asyncio.run(arun_as(BOB, route._maybe_auto_switch_model("org/one-GGUF", request, BOB.username)))
    assert loads == []
    with pytest.raises(HTTPException) as refused:
        asyncio.run(arun_as(BOB, route._maybe_auto_switch_model("org/two-GGUF", request, BOB.username)))
    assert refused.value.status_code == 409
    assert refused.value.detail["error"]["type"] == "conflict_error"
    if path.endswith("completions"):
        assert refused.value.detail["error"]["code"] == "gpu_busy"
    assert refused.value.headers["Retry-After"] == "15"
    assert loads == [] and not event.is_set()
    assert backend.model_identifier == "org/one-GGUF"
    sim.finish("a")
    asyncio.run(arun_as(BOB, route._maybe_auto_switch_model("org/two-GGUF", request, BOB.username)))
    assert loads == ["org/two-GGUF"]


def test_chat_auto_switch_rechecks_under_lifecycle_gate(route, sim, chat_switch, monkeypatch):
    _backend, loads = chat_switch
    sim.load(ALICE, arb.CHAT, "org/one-GGUF")
    acquire = route._acquire_swap_gate

    async def register_during_prepare():
        await acquire()
        sim.start(ALICE, "late")
    monkeypatch.setattr(route, "_acquire_swap_gate", register_during_prepare)
    request = SimpleNamespace(scope = {}, url = SimpleNamespace(path = "/v1/chat/completions"))
    with pytest.raises(HTTPException) as refused:
        asyncio.run(arun_as(BOB, route._maybe_auto_switch_model("org/two-GGUF", request, BOB.username)))
    assert refused.value.detail["error"]["code"] == "gpu_busy"
    assert loads == []
    assert not route._auto_switch_waiters
    assert not route._auto_switch_process_lock.locked()
    sim.finish("late")


@pytest.fixture
def media_switch(monkeypatch):
    resident = {"loaded": False}
    backend = SimpleNamespace(
        status = lambda: dict(resident), loading_repo_ids = lambda: [],
        generate_progress = lambda: {"active": False}, load_progress = lambda: {"phase": "ready"},
    )
    pick = MediaModelPick("org/image", "org/image", model_kind = "diffusers")
    loads = []
    monkeypatch.setattr(settings, "get_media_auto_switch_enabled", lambda: True)
    monkeypatch.setattr(media, "backend_for", lambda _owner: backend)
    monkeypatch.setattr(media, "resolve_local_media_model", lambda *_args, **_kwargs: pick)
    monkeypatch.setattr(media, "is_edit_only", lambda _pick: False)
    monkeypatch.setattr(media, "load_takes_the_gpu", lambda: True)
    monkeypatch.setattr(media_backends, "load_takes_the_gpu", lambda: True)
    monkeypatch.setattr(media_backends, "other_backend_busy", lambda _owner: False)
    monkeypatch.setattr(media_backends, "chat_busy", lambda *_args: False)
    monkeypatch.setattr(media_keepwarm, "_TRACKERS", {
        owner: media_keepwarm._Tracker(owner) for owner in (arb.DIFFUSION, arb.VIDEO)
    })

    async def require_local(*_args, **_kwargs):
        pass
    monkeypatch.setattr(media, "_require_local", require_local)

    async def load(owner, pick, *_args):
        arb.acquire_for_request(owner, replacing = True)
        loads.append(pick.model_id)
        resident.update(loaded = True, repo_id = pick.model_id, model_kind = "diffusers")
    monkeypatch.setattr(media, "_start_load", load)
    return backend, pick, resident, loads


@pytest.mark.parametrize("owner", [arb.DIFFUSION, arb.VIDEO])
@pytest.mark.parametrize("openai_errors", [False, True])
def test_media_auto_switch_refuses_foreign_work_with_existing_error_shape(sim, media_switch, owner, openai_errors):
    _backend, _pick, _resident, loads = media_switch
    sim.load(ALICE, arb.CHAT, "shared")
    event = sim.start(ALICE, "a")

    def switch():
        return asyncio.run(arun_as(BOB, media.maybe_auto_switch_media_model(
            "org/image", owner = owner, current_subject = BOB.username, openai_errors = openai_errors,
        )))
    with pytest.raises(HTTPException) as refused:
        switch()
    assert refused.value.status_code == 409
    if openai_errors:
        assert refused.value.detail["error"]["code"] == "model_busy"
    else:
        assert isinstance(refused.value.detail, str)
    assert refused.value.headers == {"Retry-After": "15"}
    assert loads == [] and sim.evictions == [] and not event.is_set()
    sim.finish("a")
    switch()
    assert loads == ["org/image"]


def test_media_already_resident_never_checks_account_or_drains(sim, media_switch, monkeypatch):
    _backend, _pick, resident, loads = media_switch
    resident.update(loaded = True, repo_id = "org/image", model_kind = "diffusers")
    sim.load(ALICE, arb.DIFFUSION, "org/image")
    sim.start(ALICE, "a")
    monkeypatch.setattr(policy, "installation_is_multi_user", lambda: pytest.fail("resident hot path"))
    asyncio.run(arun_as(BOB, media.maybe_auto_switch_media_model(
        "org/image", owner = arb.DIFFUSION, current_subject = BOB.username, openai_errors = True,
    )))
    assert loads == []
    sim.finish("a")


def test_media_final_arbiter_refusal_is_converted_after_drain(sim, media_switch, monkeypatch):
    _backend, _pick, _resident, loads = media_switch
    sim.load(ALICE, arb.CHAT, "shared")
    original = media._start_load

    async def late_start(*args):
        sim.start(ALICE, "late")
        return await original(*args)
    monkeypatch.setattr(media, "_start_load", late_start)
    with pytest.raises(HTTPException) as refused:
        asyncio.run(arun_as(BOB, media.maybe_auto_switch_media_model(
            "org/image", owner = arb.DIFFUSION, current_subject = BOB.username, openai_errors = True,
        )))
    assert refused.value.detail["error"]["code"] == "model_busy"
    assert refused.value.headers["Retry-After"] == "15"
    assert loads == [] and sim.evictions == []
    sim.finish("late")


def test_cpu_media_drain_does_not_refuse_unrelated_gpu_work(sim, media_switch, monkeypatch):
    backend, _pick, _resident, _loads = media_switch
    sim.load(ALICE, arb.CHAT, "shared")
    sim.start(ALICE, "a")
    monkeypatch.setattr(media_backends, "load_takes_the_gpu", lambda: False)
    assert asyncio.run(arun_as(BOB, media_backends.drain(
        arb.DIFFUSION, backend, time.monotonic() + 1,
    )))
    sim.finish("a")


@pytest.mark.parametrize("owner", [arb.DIFFUSION, arb.VIDEO])
@pytest.mark.parametrize("resident_owner", [arb.CHAT, arb.DIFFUSION, arb.VIDEO])
def test_actual_media_load_routes_refuse_before_engine_activation_or_load(route, sim, monkeypatch, owner, resident_owner):
    from core.inference import diffusion, diffusion_device, diffusion_engine_router, diffusion_compat, video
    from models.inference import DiffusionLoadRequest, VideoLoadRequest
    from routes import video as video_route

    touched = []
    backend = SimpleNamespace(
        validate_load_request = lambda *_a, **_k: SimpleNamespace(name = "ltx-2", base_repo = None),
        preflight_base_access = lambda *_a, **_k: None,
        assert_precision_available = lambda *_a, **_k: None,
        begin_load = lambda *_a, **_k: touched.append("load"),
    )

    async def ordinal(*_args):
        return None

    monkeypatch.setattr(diffusion, "get_diffusion_backend", lambda: backend)
    monkeypatch.setattr(video, "get_video_backend", lambda: backend)
    monkeypatch.setattr(video, "assert_video_precision_available", lambda *_a, **_k: None)
    monkeypatch.setattr(diffusion_device, "resolve_diffusion_device_target", lambda: SimpleNamespace(device = "cuda"))
    monkeypatch.setattr(diffusion_engine_router, "predict_engine", lambda *_a, **_k: "diffusers")
    monkeypatch.setattr(diffusion_engine_router, "engine_for", lambda *_a: backend)
    monkeypatch.setattr(diffusion_engine_router, "active_engine_name", lambda: "diffusers")
    monkeypatch.setattr(diffusion_engine_router, "select_and_activate_engine", lambda *_a, **_k: touched.append("activate") or backend)
    monkeypatch.setattr(diffusion_compat, "assert_pick_is_not_speech", lambda *_a: None)
    monkeypatch.setattr(route, "_guard_diffusion_load_against_training", lambda: None)
    monkeypatch.setattr(route, "_selected_gpu_ordinal", ordinal)
    monkeypatch.setattr(video_route, "_guard_video_load_against_training", lambda: None)
    monkeypatch.setattr(video_route, "_selected_gpu_ordinal", ordinal)
    sim.load(ALICE, resident_owner, "resident")
    event = sim.start(ALICE, "a")
    if owner == arb.DIFFUSION:
        request = DiffusionLoadRequest(model_path = "org/image", gguf_filename = "model.gguf")
        load = route.load_diffusion_model_gated
    else:
        request = VideoLoadRequest(model_path = "org/video", gguf_filename = "model.gguf")
        load = video_route.load_video_model_gated
    with pytest.raises(HTTPException) as refused:
        asyncio.run(arun_as(BOB, load(request, BOB.username)))
    assert refused.value.status_code == 409
    assert refused.value.detail["error"] == "gpu_busy"
    assert refused.value.headers["Retry-After"] == "15"
    assert touched == [] and sim.evictions == [] and not event.is_set()
    sim.finish("a")


@pytest.mark.parametrize("path", ["/v1/chat/completions", "/v1/messages", "/api/inference/chat"])
def test_chat_final_arbiter_refusal_keeps_endpoint_error_shape(route, sim, chat_switch, monkeypatch, path):
    _backend, loads = chat_switch
    sim.load(ALICE, arb.CHAT, "org/one-GGUF")

    async def late_load(*_args, **_kwargs):
        sim.start(ALICE, "late")
        arb.acquire_for_request(arb.CHAT, replacing = True)
    monkeypatch.setattr(route, "_load_model_impl", late_load)
    request = SimpleNamespace(scope = {}, url = SimpleNamespace(path = path))
    with pytest.raises(HTTPException) as refused:
        asyncio.run(arun_as(BOB, route._maybe_auto_switch_model("org/two-GGUF", request, BOB.username)))
    assert refused.value.status_code == 409
    if path.startswith("/v1/"):
        assert refused.value.detail["error"]["type"] == "conflict_error"
    else:
        assert refused.value.detail["error"] == "gpu_busy"
    assert refused.value.headers["Retry-After"] == "15"
    assert loads == [] and sim.evictions == []
    sim.finish("late")


def test_chat_resident_fast_path_adds_no_account_policy_work(route, chat_switch, monkeypatch):
    monkeypatch.setattr(policy, "installation_is_multi_user", lambda: pytest.fail("resident policy read"))
    monkeypatch.setattr(generations, "foreign_count", lambda _account: pytest.fail("resident foreign scan"))
    request = SimpleNamespace(scope = {}, url = SimpleNamespace(path = "/v1/chat/completions"))
    asyncio.run(arun_as(OWNER, route._maybe_auto_switch_model("org/one-GGUF", request, OWNER.username)))
