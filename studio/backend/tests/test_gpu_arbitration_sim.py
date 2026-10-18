# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Deterministic account/GPU simulations without a GPU: backends are recorders, the arbiter and route guards are real."""

from __future__ import annotations

import asyncio
import itertools
import sqlite3
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
            account,
            generations.ActiveGeneration,
            event,
            thread_id = "same-client-id",
            run_id = name,
            model = self.resident[1],
        )
        tracker.__enter__()
        self.runs[name] = (tracker, event)
        return event

    def finish(self, name):
        tracker, _event = self.runs.pop(name)
        tracker.__exit__(None, None, None)

    def delete(self, account):
        return generations.cancel_all(account.account_id)

    def training(self, account):
        def reserve():
            # Must run BEFORE the training route's destructive cleanup, under its admission gate.
            arb.raise_if_other_accounts_active()
            arb.release(arb.DIFFUSION)
            arb.release(arb.VIDEO)
            self.evict()
            arb.release(arb.CHAT)

        run_as(account, reserve)


@pytest.fixture
def sim(monkeypatch):
    return Simulator(monkeypatch)


def test_a_failed_registration_does_not_take_residency(monkeypatch):
    """``_owner_account`` says who LOADED the resident model, so a registration that raised may not take it."""
    monkeypatch.setitem(arb._EVICTORS, arb.CHAT, lambda: None)
    run_as(ALICE, arb.acquire_for, arb.CHAT, lambda: None)
    assert arb.owner_account() == ALICE.account_id
    with pytest.raises(RuntimeError, match = "out of memory"):
        run_as(
            BOB,
            arb.acquire_for,
            arb.CHAT,
            lambda: (_ for _ in ()).throw(RuntimeError("out of memory")),
        )
    assert arb.owner_account() == ALICE.account_id


def test_a_lone_owner_acquires_without_reading_policy_or_a_database(monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("the arbiter must not read account policy under its lock")

    monkeypatch.setattr(policy, "installation_is_multi_user", unexpected)
    monkeypatch.setattr(policy, "_account_counts", unexpected)
    monkeypatch.setattr(sqlite3, "connect", unexpected)
    monkeypatch.setitem(arb._EVICTORS, arb.CHAT, lambda: None)
    assert run_as(OWNER, arb.acquire_for, arb.CHAT, lambda: "loaded") == "loaded"
    run_as(OWNER, arb.raise_if_other_accounts_active)
    assert arb.owner_account() == OWNER.account_id


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
    # Delete cancels Alice's own run; a refused force leaves it running.
    assert a.is_set() == deleting and not b.is_set()
    assert generations.foreign_count(BOB.account_id) == 1  # cancellation is not completion
    sim.finish("a")
    assert generations.count(BOB.account_id) == 1
    assert sim.load(CAROL, arb.CHAT, "shared") == "reused"
    sim.finish("b")


def test_sim_training_preflight_and_release_do_not_terminate_foreign_chat(sim):
    sim.load(ALICE, arb.CHAT, "shared")
    event = sim.start(ALICE, "a")
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
            ALICE,
            route._raise_or_cancel_active_generations,
            force = force,
            cancel = cancel,
            action = "Replacing the model",
        )
    assert refused.value.status_code == 409
    assert refused.value.detail["error"] == "gpu_busy"
    # Refused for a foreign generation, so the caller's chats are never cancelled.
    assert not mine.is_set()
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
def test_wait_refuses_foreign_even_when_counters_miss_it_or_deadline_expires(
    route, sim, cancel_pending
):
    sim.load(ALICE, arb.CHAT, "shared")
    event = sim.start(ALICE, "a")
    with pytest.raises(HTTPException) as refused:
        asyncio.run(
            arun_as(
                BOB,
                route._wait_for_model_switch_idle(
                    current_request_counted = True,
                    cancel_pending = cancel_pending,
                    timeout_s = 0,
                ),
            )
        )
    assert refused.value.detail["error"] == "gpu_busy"
    assert not event.is_set()
    sim.finish("a")
    asyncio.run(arun_as(BOB, route._wait_for_model_switch_idle(current_request_counted = True)))


def test_wait_discounts_only_own_cancellable_work(route, sim, monkeypatch):
    sim.load(ALICE, arb.CHAT, "shared")
    event = sim.start(ALICE, "a")
    monkeypatch.setattr(keepwarm, "_inflight", 1)
    asyncio.run(
        arun_as(
            ALICE,
            route._wait_for_model_switch_idle(
                current_request_counted = False,
                cancel_pending = True,
            ),
        )
    )
    assert not event.is_set()
    sim.finish("a")


@pytest.mark.parametrize(
    "previous,target,active", tuple(itertools.product((None, *BACKENDS), BACKENDS, [False, True]))
)
def test_single_account_arbiter_matches_legacy_decisions(
    sim, monkeypatch, previous, target, active
):
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
            with pytest.raises(HTTPException) as refused:
                route._raise_or_cancel_active_generations(force = force, cancel = cancel, action = "Load")
            assert refused.value.detail["error"] == "active_generations"
            assert refused.value.headers is None
        else:
            assert route._raise_or_cancel_active_generations(
                force = force,
                cancel = cancel,
                action = "Load",
            ) == int(active and force and cancel)
        assert event.is_set() == (active and force and cancel)
    finally:
        if active:
            tracker.__exit__(None, None, None)


def test_single_account_wait_keeps_post_cancel_timeout(route, monkeypatch):
    monkeypatch.setattr(policy, "installation_is_multi_user", lambda: False)
    monkeypatch.setattr(keepwarm, "_inflight", 1)
    monkeypatch.setattr(
        generations, "foreign_count", lambda _account: pytest.fail("foreign scan on owner drain")
    )
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


@pytest.mark.parametrize(
    "capacity,count,expected", [(2, 0, 15), (2, 2, 15), (2, 3, 30), (2, 7, 60), (1, 20, 120)]
)
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
        is_loaded = True,
        model_identifier = "org/one-GGUF",
        hf_variant = None,
        _openai_advertised_id = None,
    )
    loads = []
    monkeypatch.setattr(settings, "get_openai_auto_switch_enabled", lambda: True)
    monkeypatch.setattr(resolver, "warm_index_soon", lambda: None)
    monkeypatch.setattr(
        resolver, "resolve_trusted_cached_local_gguf", lambda name, **_kw: (name, None, name)
    )
    monkeypatch.setattr(resolver, "local_target_is_gguf", lambda *_args: True)
    monkeypatch.setattr(route, "get_llama_cpp_backend", lambda: backend)
    monkeypatch.setattr(route, "_auto_switch_waiters", {})
    monkeypatch.setattr(
        route, "_loaded_identity_satisfies", lambda name: backend.model_identifier == name
    )
    monkeypatch.setattr(route, "_claim_slot_for_non_preview", lambda *_args: None)
    monkeypatch.setattr(
        route.account_access, "require_model_access", lambda *_args, **_kwargs: None
    )

    async def load(request, *_args, **_kwargs):
        loads.append(request.model_path)
        backend.model_identifier = request.model_path

    monkeypatch.setattr(route, "_load_model_impl", load)

    async def reject(*_args):
        pass

    monkeypatch.setattr(route, "_reject_unservable_model", reject)
    return backend, loads


@pytest.mark.parametrize("path", ["/v1/chat/completions", "/v1/messages"])
def test_chat_auto_switch_keeps_resident_parallel_and_refuses_foreign_swap(
    route, sim, chat_switch, path
):
    backend, loads = chat_switch
    sim.load(ALICE, arb.CHAT, "org/one-GGUF")
    event = sim.start(ALICE, "a")
    request = SimpleNamespace(scope = {}, url = SimpleNamespace(path = path))
    asyncio.run(arun_as(BOB, route._maybe_auto_switch_model("org/one-GGUF", request, BOB.username)))
    assert loads == []
    with pytest.raises(HTTPException) as refused:
        asyncio.run(
            arun_as(BOB, route._maybe_auto_switch_model("org/two-GGUF", request, BOB.username))
        )
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
        asyncio.run(
            arun_as(BOB, route._maybe_auto_switch_model("org/two-GGUF", request, BOB.username))
        )
    assert refused.value.detail["error"]["code"] == "gpu_busy"
    assert loads == []
    assert not route._auto_switch_waiters
    assert not route._auto_switch_process_lock.locked()
    sim.finish("late")


@pytest.fixture
def media_switch(monkeypatch):
    resident = {"loaded": False}
    backend = SimpleNamespace(
        status = lambda: dict(resident),
        loading_repo_ids = lambda: [],
        generate_progress = lambda: {"active": False},
        load_progress = lambda: {"phase": "ready"},
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
    monkeypatch.setattr(
        media_keepwarm,
        "_TRACKERS",
        {owner: media_keepwarm._Tracker(owner) for owner in (arb.DIFFUSION, arb.VIDEO)},
    )

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
def test_media_auto_switch_refuses_foreign_work_with_existing_error_shape(
    sim, media_switch, owner, openai_errors
):
    _backend, _pick, _resident, loads = media_switch
    sim.load(ALICE, arb.CHAT, "shared")
    event = sim.start(ALICE, "a")

    def switch():
        return asyncio.run(
            arun_as(
                BOB,
                media.maybe_auto_switch_media_model(
                    "org/image",
                    owner = owner,
                    current_subject = BOB.username,
                    openai_errors = openai_errors,
                ),
            )
        )

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
    monkeypatch.setattr(
        policy, "installation_is_multi_user", lambda: pytest.fail("resident hot path")
    )
    asyncio.run(
        arun_as(
            BOB,
            media.maybe_auto_switch_media_model(
                "org/image",
                owner = arb.DIFFUSION,
                current_subject = BOB.username,
                openai_errors = True,
            ),
        )
    )
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
        asyncio.run(
            arun_as(
                BOB,
                media.maybe_auto_switch_media_model(
                    "org/image",
                    owner = arb.DIFFUSION,
                    current_subject = BOB.username,
                    openai_errors = True,
                ),
            )
        )
    assert refused.value.detail["error"]["code"] == "model_busy"
    assert refused.value.headers["Retry-After"] == "15"
    assert loads == [] and sim.evictions == []
    sim.finish("late")


def test_cpu_media_drain_does_not_refuse_unrelated_gpu_work(sim, media_switch, monkeypatch):
    backend, _pick, _resident, _loads = media_switch
    sim.load(ALICE, arb.CHAT, "shared")
    sim.start(ALICE, "a")
    monkeypatch.setattr(media_backends, "load_takes_the_gpu", lambda: False)
    assert asyncio.run(
        arun_as(
            BOB,
            media_backends.drain(
                arb.DIFFUSION,
                backend,
                time.monotonic() + 1,
            ),
        )
    )
    sim.finish("a")


# Keep multiple models loaded (When they fit / Always) against a fake VRAM ledger: the arbiter, gpu_residency and the
# chat pricing helpers are real; only the GPU probe, the evictors and the slot reports are fakes.
GIB = 1024


class VramSim:
    def __init__(self, monkeypatch, totals_gb):
        import utils.hardware as hardware
        from core.inference import gpu_residency, model_slots, video_minimax_h3
        from utils.hardware import hardware as hardware_impl

        self.residency = gpu_residency
        self.totals = {gpu: gb * GIB for gpu, gb in enumerate(totals_gb)}
        self.footprint: dict[str, dict[int, int]] = {}
        self.evictions: list[str] = []
        self.probes = 0
        self.index_kind = "physical"
        self.parent = sorted(self.totals)
        self.policy = "off"
        for owner in BACKENDS:
            monkeypatch.setitem(arb._EVICTORS, owner, lambda o = owner: self._evict(o))
            monkeypatch.setitem(
                arb._SLOT_REPORTERS,
                owner,
                lambda o = owner: [
                    {"kind": arb.KIND_LABELS[o], "model": f"{arb.KIND_LABELS[o]}-model"}
                ],
            )
        monkeypatch.setattr(arb, "_residents", {})
        monkeypatch.setattr(arb, "_accounts", {})
        monkeypatch.setattr(arb, "_media_needs", {})
        monkeypatch.setattr(arb, "_release_idle_video_servers", lambda reason: None)
        monkeypatch.setattr(video_minimax_h3, "release_h3_native_servers", lambda reason: 0)
        monkeypatch.setattr(hardware, "get_device", lambda: hardware.DeviceType.CUDA)
        monkeypatch.setattr(hardware, "get_visible_gpu_utilization", self._utilization)
        monkeypatch.setattr(hardware_impl, "get_parent_visible_gpu_ids", lambda: list(self.parent))
        monkeypatch.setattr(gpu_residency, "current_policy", lambda: self.policy)
        monkeypatch.setattr(gpu_residency, "_media_busy", lambda owner: (False, False))
        monkeypatch.setattr(gpu_residency, "chat_load_in_flight", lambda: False)
        # The diffusion planner's own verdict is covered in test_gpu_residency_coexistence.py.
        monkeypatch.setattr(gpu_residency, "_keeps_transformer_resident", lambda *a: True)
        monkeypatch.setattr(model_slots, "eviction_victims", lambda *a, **k: [])

    def _utilization(self):
        self.probes += 1
        return {
            "index_kind": self.index_kind,
            "devices": [
                {"index": gpu, "vram_total_gb": total / GIB, "vram_used_gb": self.used(gpu) / GIB}
                for gpu, total in self.totals.items()
            ],
        }

    def used(self, gpu):
        return sum(held.get(gpu, 0) for held in self.footprint.values())

    def _evict(self, owner):
        self.evictions.append(owner)
        self.footprint.pop(owner, None)

    def load(self, owner, held, **kwargs):
        evicted: list = []

        def register():
            self.footprint[owner] = dict(held)

        arb.acquire_for(owner, register, evicted = evicted, **kwargs)
        return evicted

    def load_image(
        self,
        need_gb,
        gpu = 0,
        headroom_gb = 4,
        **kwargs,
    ):
        """An image load priced at ``need_gb``; idle, only its weights are allocated (the headroom is not)."""
        need, headroom = need_gb * GIB, headroom_gb * GIB
        priced = self.residency.MediaNeed(
            {gpu: need}, weights_mib = need - headroom, headroom_mib = headroom, kind = "image"
        )
        return self.load(arb.DIFFUSION, {gpu: need - headroom}, need_mib = priced, **kwargs)

    def load_chat(
        self,
        gb = 10,
        gpu_ids = (0,),
        **kwargs,
    ):
        held = {gpu: gb * GIB // max(len(gpu_ids or (0,)), 1) for gpu in (gpu_ids or (0,))}
        return self.load(
            arb.CHAT, held, gpu_ids = list(gpu_ids) if gpu_ids is not None else None, **kwargs
        )


@pytest.fixture
def vram(monkeypatch):
    return lambda *totals_gb: VramSim(monkeypatch, totals_gb)


CHAT_SLOT = {"kind": "chat", "model": "chat-model"}
IMAGE_SLOT = {"kind": "image", "model": "image-model"}


@pytest.mark.parametrize("policy", ["fit", "always"])
def test_image_joins_chat_on_one_gpu_when_it_fits(vram, policy):
    v = vram(80)
    v.load_chat(10)
    assert v.load_image(20, policy = policy) == []
    assert v.evictions == [] and set(arb.residents()) == {arb.CHAT, arb.DIFFUSION}
    assert v.residency.media_held_mib() == {0: 20 * GIB}
    assert v.used(0) == 26 * GIB  # 10 GB chat + 16 GB idle image weights


def test_image_evicts_chat_on_one_gpu_when_short_under_fit(vram):
    v = vram(80)
    v.load_chat(10)
    assert v.load_image(75, policy = "fit") == [CHAT_SLOT]
    assert v.evictions == [arb.CHAT] and set(arb.residents()) == {arb.DIFFUSION}
    assert arb.CHAT not in v.footprint


def test_image_short_on_one_gpu_under_always_refuses_and_keeps_chat(vram):
    v = vram(80)
    v.load_chat(10)
    with pytest.raises(arb.MultiModelAlwaysRefusal, match = "is set to Always"):
        v.load_image(75, policy = "always")
    assert v.evictions == [] and set(arb.residents()) == {arb.CHAT}
    assert v.footprint == {arb.CHAT: {0: 10 * GIB}} and v.residency.media_held_mib() == {}


@pytest.mark.parametrize("policy", ["fit", "always"])
def test_image_on_another_gpu_never_evicts_or_probes(vram, policy):
    v = vram(80, 80)
    v.load_chat(10, gpu_ids = (0,))
    assert v.load_image(75, gpu = 1, policy = policy) == []
    assert v.evictions == [] and v.probes == 0
    assert arb.residents() == {arb.CHAT: frozenset({0}), arb.DIFFUSION: frozenset({1})}


def test_image_on_another_gpu_under_off_still_swaps(vram):
    v = vram(80, 80)
    v.load_chat(10, gpu_ids = (0,))
    assert v.load_image(20, gpu = 1) == [CHAT_SLOT]
    assert v.evictions == [arb.CHAT] and set(arb.residents()) == {arb.DIFFUSION}


def _chat_kwargs(
    config,
    placement,
    *,
    primary_can_retry = False,
):
    from routes import inference as inf
    token = inf._primary_can_retry_fit.set(primary_can_retry)
    try:
        return inf._chat_fit_kwargs(config, placement)
    finally:
        inf._primary_can_retry_fit.reset(token)


def _placement(
    gpus = (0,),
    *,
    vulkan = False,
    native_gb = None,
):
    from routes.inference import _LoadPlacement
    resolved = list(gpus) if gpus is not None else None
    return _LoadPlacement(resolved, resolved, vulkan, None, native_required_gb = native_gb)


def test_gguf_chat_joins_an_idle_image_and_its_launch_prices_the_image(vram):
    from core.inference import llama_cpp

    v = vram(80, 80)
    v.policy = "fit"
    v.load_image(20, policy = "fit")
    placement = _placement((0,))
    kwargs = _chat_kwargs(SimpleNamespace(is_gguf = True), placement, primary_can_retry = True)
    assert kwargs == {"policy": "fit", "gpu_ids": [0]}
    assert v.load_chat(10, **kwargs) == []
    assert v.evictions == [] and set(arb.residents()) == {arb.CHAT, arb.DIFFUSION}

    backend = llama_cpp.LlamaCppBackend(manages_processes = False)
    llama_cpp.register_serving_backend(backend)
    try:
        held = backend._other_planned_vram_mib()
    finally:
        llama_cpp.unregister_serving_backend(backend)
    assert held == {
        0: 20 * GIB
    }  # weights plus the headroom the idle image will allocate when it generates
    # A stale free reading on GPU 0 is capped at total less what the image holds; GPU 1 is left alone.
    probed = [(0, 80 * GIB, 80 * GIB), (1, 70 * GIB, 80 * GIB)]
    assert llama_cpp._net_of_held_vram(probed, held) == [
        (0, 60 * GIB, 80 * GIB),
        (1, 70 * GIB, 80 * GIB),
    ]


def test_gguf_chat_with_no_fit_retry_evicts_first_under_fit(vram):
    v = vram(80)
    v.policy = "fit"
    assert _chat_kwargs(SimpleNamespace(is_gguf = True), _placement()) == {
        "policy": "fit",
        "gpu_ids": [0],
        "evict_first": True,
    }


@pytest.mark.parametrize(
    ("policy", "expected", "evicted"),
    [
        ("fit", {"policy": "fit", "gpu_ids": [0], "evict_first": True}, [IMAGE_SLOT]),
        ("always", {"policy": "always", "gpu_ids": [0]}, []),
    ],
)
def test_non_gguf_chat_beside_an_image(vram, policy, expected, evicted):
    v = vram(80)
    v.policy = policy
    v.load_image(20, policy = policy)
    placement = _placement((0,))
    kwargs = _chat_kwargs(SimpleNamespace(is_gguf = False), placement, primary_can_retry = True)
    assert kwargs == expected
    assert v.load_chat(10, **kwargs) == evicted
    assert v.evictions == [arb.DIFFUSION for _ in evicted]
    assert arb.CHAT in arb.residents() and (arb.DIFFUSION in arb.residents()) == (not evicted)


def _audio_cpp_chat(v, native_gb, **placement_kw):
    placement = _placement(native_gb = native_gb, **placement_kw)
    kwargs = _chat_kwargs(SimpleNamespace(is_gguf = False), placement)
    gpus = None if placement.gpu_ids_are_vulkan_ordinals else placement.resolved_gpu_ids
    return kwargs, v.load(arb.CHAT, {0: int(native_gb * GIB)}, **{"gpu_ids": gpus, **kwargs})


@pytest.mark.parametrize("policy", ["fit", "always"])
def test_audio_cpp_chat_joins_a_resident_image_when_native_estimate_fits(vram, policy):
    v = vram(80)
    v.policy = policy
    v.load_image(20, policy = policy)
    kwargs, evicted = _audio_cpp_chat(v, 6.0)
    assert kwargs == {"policy": policy, "gpu_ids": [0], "need_mib": {0: 6144}}
    assert evicted == [] and v.evictions == [] and set(arb.residents()) == {arb.CHAT, arb.DIFFUSION}


def test_audio_cpp_chat_short_beside_an_image_evicts_it_under_fit(vram):
    v = vram(80)
    v.policy = "fit"
    v.load_image(20, policy = "fit")
    kwargs, evicted = _audio_cpp_chat(v, 70.0)
    assert kwargs["need_mib"] == {0: 70 * GIB}
    assert evicted == [IMAGE_SLOT] and v.evictions == [arb.DIFFUSION]
    assert set(arb.residents()) == {arb.CHAT} and v.residency.media_held_mib() == {}


def test_audio_cpp_chat_short_beside_an_image_refuses_under_always(vram):
    v = vram(80)
    v.policy = "always"
    v.load_image(20, policy = "always")
    with pytest.raises(arb.MultiModelAlwaysRefusal, match = "is set to Always"):
        _audio_cpp_chat(v, 70.0)
    assert v.evictions == [] and set(arb.residents()) == {arb.DIFFUSION}


@pytest.mark.parametrize("placement_kw", [{"vulkan": True}, {"gpus": None}])
def test_audio_cpp_chat_without_physical_gpus_evicts_first_under_fit(vram, placement_kw):
    v = vram(80)
    v.policy = "fit"
    v.load_image(20, policy = "fit")
    kwargs, evicted = _audio_cpp_chat(v, 6.0, **placement_kw)
    assert kwargs == {"policy": "fit", "evict_first": True}
    assert evicted == [IMAGE_SLOT] and v.evictions == [arb.DIFFUSION]


def _masked(v):
    # A UUID/MIG mask: rows carry relative indices and there are no numeric parent ids.
    v.index_kind = "relative"
    v.parent = []
    v.load_chat(10, gpu_ids = None)


@pytest.mark.parametrize("priced", [False, True])
def test_masked_host_image_beside_chat_evicts_first_under_fit(vram, priced):
    v = vram(80)
    _masked(v)
    assert v.residency.media_load_need_mib(arb.DIFFUSION, SimpleNamespace(), None) is None
    if priced:
        assert v.load_image(20, policy = "fit") == [CHAT_SLOT]
    else:
        assert v.load(arb.DIFFUSION, {0: 16 * GIB}, policy = "fit") == [CHAT_SLOT]
    assert v.evictions == [arb.CHAT] and set(arb.residents()) == {arb.DIFFUSION}


@pytest.mark.parametrize("priced", [False, True])
def test_masked_host_image_beside_chat_refuses_under_always(vram, priced):
    v = vram(80)
    _masked(v)
    with pytest.raises(arb.MultiModelAlwaysRefusal, match = "is set to Always") as refused:
        if priced:
            v.load_image(20, policy = "always")
        else:
            v.load(arb.DIFFUSION, {0: 16 * GIB}, policy = "always")
    assert refused.value.kind == "chat" and refused.value.model == "chat-model"
    assert v.evictions == [] and set(arb.residents()) == {arb.CHAT}


@pytest.mark.parametrize("owner", [arb.DIFFUSION, arb.VIDEO])
@pytest.mark.parametrize("resident_owner", [arb.CHAT, arb.DIFFUSION, arb.VIDEO])
def test_actual_media_load_routes_refuse_before_engine_activation_or_load(
    route, sim, monkeypatch, owner, resident_owner
):
    from core.inference import (
        diffusion,
        diffusion_device,
        diffusion_engine_router,
        diffusion_compat,
        video,
    )
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
    monkeypatch.setattr(
        diffusion_device, "resolve_diffusion_device_target", lambda: SimpleNamespace(device = "cuda")
    )
    monkeypatch.setattr(diffusion_engine_router, "predict_engine", lambda *_a, **_k: "diffusers")
    monkeypatch.setattr(diffusion_engine_router, "engine_for", lambda *_a: backend)
    monkeypatch.setattr(diffusion_engine_router, "active_engine_name", lambda: "diffusers")
    monkeypatch.setattr(
        diffusion_engine_router,
        "select_and_activate_engine",
        lambda *_a, **_k: touched.append("activate") or backend,
    )
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
def test_chat_final_arbiter_refusal_keeps_endpoint_error_shape(
    route, sim, chat_switch, monkeypatch, path
):
    _backend, loads = chat_switch
    sim.load(ALICE, arb.CHAT, "org/one-GGUF")

    async def late_load(*_args, **_kwargs):
        sim.start(ALICE, "late")
        arb.acquire_for_request(arb.CHAT, replacing = True)

    monkeypatch.setattr(route, "_load_model_impl", late_load)
    request = SimpleNamespace(scope = {}, url = SimpleNamespace(path = path))
    with pytest.raises(HTTPException) as refused:
        asyncio.run(
            arun_as(BOB, route._maybe_auto_switch_model("org/two-GGUF", request, BOB.username))
        )
    assert refused.value.status_code == 409
    if path.startswith("/v1/"):
        assert refused.value.detail["error"]["type"] == "conflict_error"
    else:
        assert refused.value.detail["error"] == "gpu_busy"
    assert refused.value.headers["Retry-After"] == "15"
    assert loads == [] and sim.evictions == []
    sim.finish("late")


def test_chat_resident_fast_path_adds_no_account_policy_work(route, chat_switch, monkeypatch):
    monkeypatch.setattr(
        policy, "installation_is_multi_user", lambda: pytest.fail("resident policy read")
    )
    monkeypatch.setattr(
        generations, "foreign_count", lambda _account: pytest.fail("resident foreign scan")
    )
    request = SimpleNamespace(scope = {}, url = SimpleNamespace(path = "/v1/chat/completions"))
    asyncio.run(
        arun_as(OWNER, route._maybe_auto_switch_model("org/one-GGUF", request, OWNER.username))
    )
