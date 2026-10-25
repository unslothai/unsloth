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
    monkeypatch.setattr(inference, "fetch_llama_slots", lambda base, headers = None: slots)
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
