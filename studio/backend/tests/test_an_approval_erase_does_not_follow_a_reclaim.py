# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The approval worker and the idle-slot reclaimer both erase a parked chat's slot. Once one of
them took the cells, the other must not erase the slot again: llama-server defers an erase aimed
at a slot that is processing, so a second one lands on the cache of the chat the first admitted."""

from __future__ import annotations

import ast
import asyncio
import inspect
import textwrap
import threading
import time

from core.inference.llama_preemption import ParticipantState, PreemptionController
import routes.inference as inference
import pytest

pytestmark = pytest.mark.usefixtures("preemption_opted_in")


BASE = "http://127.0.0.1:65091"
SLOT = 1


class _Backend:
    base_url = BASE
    admission_key = "test://approval-erase"
    _kv_cache_unified = True
    context_length = 8192

    def __init__(self):
        self.erased: list = []

    def release_idle_chat_slot(self, base_url, slot):
        self.erased.append((base_url, slot))
        return True


class _ControllerLease:
    def yield_parked_commitment(self, *, charged_at = None):
        return 0


class _ApprovalLease:
    """The approval lease past its `reclaim_would_admit` check: the window the reclaimer races."""

    def __init__(self):
        self.released = 0

    def reclaim_would_admit(self) -> bool:
        return True

    def release_parked_cache(self) -> int:
        self.released += 1
        return 6000


def _controller() -> PreemptionController:
    made = PreemptionController(BASE)
    made.configure(budget = 8192, kv_unified = True, slots = 4)
    made.register("chat", lease = _ControllerLease(), tokens = 6000)
    made.note_tokens("chat", 6000, 6000)
    made.note_measured("chat")
    made.note_state("chat", ParticipantState.PARKED_ON_TOOL)
    return made


def _lift(namespace: dict):
    source = textwrap.dedent(inspect.getsource(inference.produce_openai_chat_completions))
    wanted = {"_erase_and_record", "_reclaim_approval_cache"}
    nodes = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in wanted
    ]
    assert {node.name for node in nodes} == wanted
    exec(compile(ast.Module(body = nodes, type_ignores = []), "<route>", "exec"), namespace)
    return namespace["_reclaim_approval_cache"]


def _worker(monkeypatch, controller, backend, slots):
    monkeypatch.setattr(inference, "get_preemption_controller", lambda key: controller)
    monkeypatch.setattr(
        inference,
        "fetch_llama_slots",
        lambda base, headers = None: slots() if callable(slots) else slots,
    )
    namespace = dict(vars(inference))
    namespace.update(
        llama_backend = backend,
        completion_id = "chat",
        _gguf_decode_lock = threading.Lock(),
        _gguf_decode = {"slot": (BASE, SLOT), "erased": False},
        _reclaim_stop = asyncio.Event(),
    )
    return _lift(namespace), namespace


def test_cells_the_reclaimer_already_took_are_not_erased_again(monkeypatch):
    controller = _controller()
    # The idle-slot reclaimer erased the parked chat's slot and released its commitment.
    assert controller.note_cells_reclaimed(controller.parked_holders()) == 1
    backend = _Backend()
    lease = _ApprovalLease()
    worker, _ = _worker(
        monkeypatch, controller, backend, [{"id": SLOT, "is_processing": False, "id_task": 7}]
    )

    asyncio.run(worker(lease))

    assert backend.erased == [], (
        "the approval worker erased a slot the reclaimer had already emptied; llama-server "
        "defers it onto the chat that slot was handed to"
    )
    assert lease.released == 0


def test_a_slot_another_request_is_processing_is_not_erased(monkeypatch):
    """The reclaimer freed the slot without marking this holder (a partial reclaim), and the
    queued chat it admitted is now decoding there."""
    controller = _controller()
    backend = _Backend()
    lease = _ApprovalLease()
    worker, namespace = _worker(
        monkeypatch, controller, backend, [{"id": SLOT, "is_processing": True, "id_task": 7}]
    )

    asyncio.run(worker(lease))

    assert backend.erased == [], "the erase went to a slot another chat is decoding in"
    assert lease.released == 0, "the commitment went back for cells this worker never erased"
    assert namespace["_gguf_decode"]["erased"] is False


def test_an_idle_slot_still_holding_the_chat_is_erased(monkeypatch):
    controller = _controller()
    backend = _Backend()
    lease = _ApprovalLease()
    worker, namespace = _worker(
        monkeypatch, controller, backend, [{"id": SLOT, "is_processing": False, "id_task": 3}]
    )

    asyncio.run(worker(lease))

    assert backend.erased == [(BASE, SLOT)]
    assert lease.released == 1
    assert namespace["_gguf_decode"]["erased"] is True
    assert controller.participant("chat").cells_reclaimed is True


@pytest.mark.parametrize(
    "reading",
    [None, [], [{"id": SLOT + 1, "is_processing": False, "id_task": 3}]],
    ids = ["unreadable", "empty", "slot-missing"],
)
def test_no_erase_without_a_reading_of_the_slot_idle(monkeypatch, reading):
    """The reclaimer skips a slot its recheck cannot see idle; so must the approval erase, or a
    failed `/slots` read sends it to a slot that may now be decoding another chat."""
    controller = _controller()
    backend = _Backend()
    reads = {"n": 0}

    def _slots():
        # The worker's first read saw the slot idle on this chat's task; the recheck right
        # before the erase does not. A failed first read is retried rather than final.
        reads["n"] += 1
        if reads["n"] == 1:
            return [{"id": SLOT, "is_processing": False, "id_task": 3}]
        return reading

    lease = _ApprovalLease()
    worker, namespace = _worker(monkeypatch, controller, backend, _slots)

    asyncio.run(worker(lease))

    assert reads["n"] == 2
    assert backend.erased == [], "the erase went out with no reading of the slot idle"
    assert lease.released == 0
    assert namespace["_gguf_decode"]["erased"] is False


def test_a_slot_handed_to_another_chat_is_not_erased(monkeypatch):
    """A partial reclaim freed the slot without marking this holder, and the chat it admitted
    there finished: the slot is idle again, but on another task, holding that chat's cache."""
    controller = _controller()
    backend = _Backend()
    state = {"slots": [{"id": SLOT, "is_processing": False, "id_task": 3}]}

    class _Lease(_ApprovalLease):
        def reclaim_would_admit(self) -> bool:
            # Meanwhile the slot ran another chat's task to completion.
            state["slots"] = [{"id": SLOT, "is_processing": False, "id_task": 9}]
            return True

    lease = _Lease()
    worker, namespace = _worker(monkeypatch, controller, backend, lambda: state["slots"])

    asyncio.run(worker(lease))

    assert backend.erased == [], "the erase took the cache of the chat the slot went to"
    assert lease.released == 0
    assert namespace["_gguf_decode"]["erased"] is False


def test_the_erase_waits_for_a_reclaim_in_flight(monkeypatch):
    """A reclaim mid-way through erasing and recording holds the per-load lock; the approval
    erase sent meanwhile would see the holder unmarked and erase the reassigned slot."""
    controller = _controller()
    backend = _Backend()
    lease = _ApprovalLease()
    worker, _ = _worker(
        monkeypatch, controller, backend, [{"id": SLOT, "is_processing": False, "id_task": 3}]
    )
    lock = inference._openai_llama_slot_reclaim_lock(backend)
    holding = threading.Event()
    done = threading.Event()

    def _reclaimer():
        with lock:
            holding.set()
            time.sleep(0.3)
            controller.note_cells_reclaimed(controller.parked_holders())
        done.set()

    thread = threading.Thread(target = _reclaimer)
    thread.start()
    assert holding.wait(5)
    asyncio.run(worker(lease))
    thread.join(5)

    assert done.is_set()
    assert backend.erased == [], "the approval erase did not wait for the reclaim in flight"


def test_one_lock_per_load():
    first, second = _Backend(), _Backend()
    assert inference._openai_llama_slot_reclaim_lock(
        first
    ) is inference._openai_llama_slot_reclaim_lock(second)


def test_every_idle_slot_reclaim_runs_under_the_lock():
    tree = ast.parse(textwrap.dedent(inspect.getsource(inference)))
    parents: dict = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "reclaim_idle_slots"
    ]
    assert len(calls) >= 3
    for call in calls:
        node, guarded = call, False
        while node in parents:
            node = parents[node]
            if isinstance(node, ast.With) and any(
                isinstance(item.context_expr, ast.Call)
                and getattr(item.context_expr.func, "id", None) == "_openai_llama_slot_reclaim_lock"
                for item in node.items
            ):
                guarded = True
                break
        assert guarded, f"reclaim_idle_slots at line {call.lineno} erases outside the lock"


def _other_holder(controller) -> None:
    """A second chat decoding beside the parked one: the chat llama-server can hand the slot to."""
    controller.register("other", lease = _ControllerLease(), tokens = 1000)
    controller.note_tokens("other", 1000, 1000)
    controller.note_measured("other")


def test_a_slot_handed_on_during_the_erase_marks_the_new_holder_for_a_replay(monkeypatch):
    """The reading saw the slot idle on this chat's task, but llama-server started another chat's
    task there before the erase reached it, and defers the erase onto that chat's cache. The
    idle-slot reclaimer rechecks the slot after its erase and marks such holders for a replay;
    the approval erase must too, or the other chat's next round is priced as growth on a cache
    that is gone."""
    controller = _controller()
    _other_holder(controller)
    state = {"slots": [{"id": SLOT, "is_processing": False, "id_task": 3}]}

    class _HandedOn(_Backend):
        def release_idle_chat_slot(self, base_url, slot):
            # Between the reading and the erase the server admitted "other" into the slot.
            state["slots"] = [{"id": SLOT, "is_processing": True, "id_task": 9}]
            return super().release_idle_chat_slot(base_url, slot)

    backend = _HandedOn()
    lease = _ApprovalLease()
    worker, _ = _worker(monkeypatch, controller, backend, lambda: state["slots"])

    asyncio.run(worker(lease))

    assert backend.erased == [(BASE, SLOT)]
    assert (
        controller.participant("other").replay_pending is True
    ), "the erase may have landed on the chat the slot was handed to, and it was not marked"
    assert controller.participant("chat").replay_pending is False


def test_a_slot_still_on_the_chat_after_the_erase_marks_no_one(monkeypatch):
    controller = _controller()
    _other_holder(controller)
    backend = _Backend()
    lease = _ApprovalLease()
    worker, _ = _worker(
        monkeypatch, controller, backend, [{"id": SLOT, "is_processing": False, "id_task": 3}]
    )

    asyncio.run(worker(lease))

    assert backend.erased == [(BASE, SLOT)]
    assert controller.participant("other").replay_pending is False
