# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A chat parked on a tool-approval prompt gives its cache back once a queued chat needs the
room. The worker that does it reads the slot first; a single failed `/slots` read must not end
it, because the finished task is kept and never restarted, so the queued chat would wait on
room that is never freed."""

from __future__ import annotations

import ast
import asyncio
import inspect
import textwrap
import threading

from core.inference.llama_preemption import ParticipantState, PreemptionController
import routes.inference as inference
import pytest

pytestmark = pytest.mark.usefixtures("preemption_opted_in")


BASE = "http://127.0.0.1:65092"
SLOT = 1


class _Backend:
    base_url = BASE
    admission_key = "test://approval-reread"
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


def test_one_failed_slot_read_does_not_end_the_worker(monkeypatch):
    controller = _controller()
    backend = _Backend()
    lease = _ApprovalLease()
    reads = {"n": 0}

    def _slots():
        reads["n"] += 1
        if reads["n"] == 1:
            return None  # one transient `/slots` failure
        return [{"id": SLOT, "is_processing": False, "id_task": 3}]

    worker, namespace = _worker(monkeypatch, controller, backend, _slots)
    namespace["_APPROVAL_CACHE_RECLAIM_POLL_S"] = 0.01

    asyncio.run(asyncio.wait_for(worker(lease), 5))

    assert backend.erased == [(BASE, SLOT)], (
        "the worker gave up after one failed read; the queued chat waits on room never freed"
    )
    assert lease.released == 1
    assert namespace["_gguf_decode"]["erased"] is True


def test_a_slot_never_read_is_not_erased_and_stop_ends_the_retries(monkeypatch):
    controller = _controller()
    backend = _Backend()
    lease = _ApprovalLease()
    reads = {"n": 0}

    def _slots():
        reads["n"] += 1
        return None

    worker, namespace = _worker(monkeypatch, controller, backend, _slots)
    namespace["_APPROVAL_CACHE_RECLAIM_POLL_S"] = 0.01
    stop = namespace["_reclaim_stop"]

    async def _run():
        task = asyncio.create_task(worker(lease))
        for _ in range(200):
            if reads["n"] >= 3 or task.done():
                break
            await asyncio.sleep(0.01)
        stop.set()
        await asyncio.wait_for(task, 5)

    asyncio.run(_run())

    assert reads["n"] >= 3, "an unreadable slot was not read again"
    assert backend.erased == []
    assert lease.released == 0
    assert namespace["_gguf_decode"]["erased"] is False


def test_a_busy_slot_is_still_refused_after_a_failed_read(monkeypatch):
    controller = _controller()
    backend = _Backend()
    lease = _ApprovalLease()
    reads = {"n": 0}

    def _slots():
        reads["n"] += 1
        if reads["n"] == 1:
            return None
        return [{"id": SLOT, "is_processing": True, "id_task": 7}]

    worker, namespace = _worker(monkeypatch, controller, backend, _slots)
    namespace["_APPROVAL_CACHE_RECLAIM_POLL_S"] = 0.01

    asyncio.run(asyncio.wait_for(worker(lease), 5))

    assert reads["n"] == 2
    assert backend.erased == []
    assert lease.released == 0
