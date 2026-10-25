# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A chat armed in Studio mode whose plan paused a decoding chat waits for that chat to let go
before its own request goes upstream: the victim keeps decoding into its cells until its stream
sees the signal, and a prefill sent meanwhile overruns the cache it is freeing. The unpausable
holders already wait; the armed chat used to return straight from the plan."""

from __future__ import annotations

import asyncio
import threading
import time
from types import SimpleNamespace
import uuid

import pytest

from core.inference import llama_preemption as preemption
from core.inference.llama_preemption import (
    ParticipantState,
    PreemptSignal,
    get_preemption_controller,
)
import routes.inference as inference

pytestmark = pytest.mark.usefixtures("preemption_opted_in")


@pytest.fixture
def armed(monkeypatch):
    monkeypatch.setenv(preemption.PREEMPT_ENV, "1")
    monkeypatch.setattr(inference, "_refresh_residency_before_planning", lambda controller: None)

    async def _no_read(controller):
        return None

    monkeypatch.setattr(inference, "_refresh_residency_before_planning_async", _no_read)
    fake = SimpleNamespace(base_url = f"http://armed-wait-{uuid.uuid4().hex}.test")
    controller = get_preemption_controller(inference._preempt_key(fake))
    controller.configure(budget = 16384, kv_unified = True)
    controller.register("chat", tokens = 15000)
    controller.note_measured("chat")
    controller.note_resident(15000, started_at_seq = controller.residency_epoch())

    def _enrol(*, gen_id, signal, **_kwargs):
        controller.register(gen_id, tokens = 4000, signal = signal)
        return controller

    monkeypatch.setattr(inference, "_openai_llama_preemption_enrol", _enrol)
    yield fake, controller
    for gen_id in list(controller._participants):
        controller.unregister(gen_id)


def _pause_later(controller, after_s):
    def run():
        time.sleep(after_s)
        controller.set_state("chat", ParticipantState.PAUSED)

    thread = threading.Thread(target = run, daemon = True)
    thread.start()
    return thread


def test_the_async_arm_returns_only_once_the_victim_let_go(armed):
    fake, controller = armed

    async def arm():
        _pause_later(controller, 0.3)
        started = time.monotonic()
        policy = await inference._openai_llama_preemption_arm_async(
            llama_backend = fake, gen_id = "new", signal = PreemptSignal()
        )
        return policy, time.monotonic() - started, controller.participant("chat").state

    policy, waited, state = asyncio.run(arm())
    assert policy is not None
    assert state == ParticipantState.PAUSED, "the prefill went upstream while the victim decoded"
    assert waited >= 0.25


def test_the_sync_arm_waits_inline(armed):
    fake, controller = armed
    _pause_later(controller, 0.3)
    started = time.monotonic()
    policy = inference._openai_llama_preemption_arm(
        llama_backend = fake, gen_id = "new", signal = PreemptSignal()
    )
    assert policy is not None
    assert time.monotonic() - started >= 0.25
    assert controller.participant("chat").state == ParticipantState.PAUSED


def test_a_victim_that_never_lets_go_refuses_the_chat(armed, monkeypatch):
    """Its prefill would land in cells the victim still holds: refused as an admission timeout,
    the error a queued chat that never got room ends on, and the enrolment dropped."""
    fake, controller = armed
    monkeypatch.setattr(inference, "_UNPAUSABLE_VICTIM_WAIT_S", 0.1)
    monkeypatch.setattr(inference, "_UNPAUSABLE_VICTIM_HARD_WAIT_S", 0.3)
    with pytest.raises(inference.LlamaAdmissionTimeout):
        inference._openai_llama_preemption_arm(
            llama_backend = fake, gen_id = "new", signal = PreemptSignal()
        )
    assert controller.participant("chat").state == ParticipantState.PREEMPTING
    assert controller.participant("new") is None, "a refused chat stayed counted"


def test_the_async_arm_refuses_the_same_way(armed, monkeypatch):
    fake, controller = armed
    monkeypatch.setattr(inference, "_UNPAUSABLE_VICTIM_WAIT_S", 0.1)
    monkeypatch.setattr(inference, "_UNPAUSABLE_VICTIM_HARD_WAIT_S", 0.3)
    with pytest.raises(inference.LlamaAdmissionTimeout):
        asyncio.run(
            inference._openai_llama_preemption_arm_async(
                llama_backend = fake, gen_id = "new", signal = PreemptSignal()
            )
        )
    assert controller.participant("new") is None, "a refused chat stayed counted"


class _Lease:
    tokens = 4000


class _Reservation:
    def lease_nowait(self):
        return _Lease()


def test_a_tool_round_waits_for_the_chats_its_sweep_paused(armed, monkeypatch):
    """The next round's prefill goes upstream straight after the publish: the same gap."""
    fake, controller = armed
    controller.register("round", tokens = 100, signal = PreemptSignal())
    controller.note_measured("round")
    monkeypatch.setattr(
        inference, "_openai_llama_admission_charged_prompt_tokens", lambda *a, **k: 3000
    )
    _pause_later(controller, 0.3)
    started = time.monotonic()
    inference._openai_llama_publish_round_charge(
        llama_backend = fake,
        gen_id = "round",
        reservation = _Reservation(),
        payload = None,
        conversation = [],
        rendered_tools = None,
        observe_tokens = lambda generated: controller.plan_preemptions(needed = 0, exclude = "round"),
    )
    assert time.monotonic() - started >= 0.25
    assert controller.participant("chat").state == ParticipantState.PAUSED


def test_a_tool_round_whose_victim_never_lets_go_is_refused(armed, monkeypatch):
    """The round's prefill is refused the way a round with no room is, not sent beside the
    victim: the tool loop ends the turn on its partial answer for this refusal."""
    fake, controller = armed
    monkeypatch.setattr(inference, "_UNPAUSABLE_VICTIM_WAIT_S", 0.1)
    monkeypatch.setattr(inference, "_UNPAUSABLE_VICTIM_HARD_WAIT_S", 0.3)
    controller.register("round", tokens = 100, signal = PreemptSignal())
    controller.note_measured("round")
    monkeypatch.setattr(
        inference, "_openai_llama_admission_charged_prompt_tokens", lambda *a, **k: 3000
    )
    with pytest.raises(inference.LlamaAdmissionRecostRefused):
        inference._openai_llama_publish_round_charge(
            llama_backend = fake,
            gen_id = "round",
            reservation = _Reservation(),
            payload = None,
            conversation = [],
            rendered_tools = None,
            observe_tokens = lambda generated: controller.plan_preemptions(needed = 0, exclude = "round"),
        )
    assert controller.participant("chat").state == ParticipantState.PREEMPTING


def test_every_arm_site_answers_a_refusal_as_an_admission_timeout():
    """Each awaited arm sits in a try that turns LlamaAdmissionTimeout into the admission 503,
    not a generic 500 or an escaped exception."""
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(inference))
    parents = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node

    def _names(handler):
        kind = handler.type
        items = kind.elts if isinstance(kind, ast.Tuple) else [kind]
        return {getattr(item, "id", getattr(item, "attr", None)) for item in items}

    sites = 0
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and getattr(node.func, "id", None) == "_openai_llama_preemption_arm_async"
            and isinstance(parents.get(node), ast.Await)
        ):
            continue
        sites += 1
        handled = False
        child, parent = node, parents.get(node)
        while parent is not None and not isinstance(
            parent, (ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            if isinstance(parent, ast.Try) and child in parent.body:
                if any(
                    h.type is not None and "LlamaAdmissionTimeout" in _names(h)
                    for h in parent.handlers
                ):
                    handled = True
                    break
            child, parent = parent, parents.get(parent)
        if not handled and isinstance(parent, ast.AsyncFunctionDef):
            # A helper such as `_arm_anthropic`: every caller must handle it instead.
            callers = [
                call
                for call in ast.walk(tree)
                if isinstance(call, ast.Call) and getattr(call.func, "id", None) == parent.name
            ]
            assert callers, parent.name
            handled = True
            for call in callers:
                c, p = call, parents.get(call)
                ok = False
                while p is not None and not isinstance(p, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if (
                        isinstance(p, ast.Try)
                        and c in p.body
                        and any(
                            h.type is not None and "LlamaAdmissionTimeout" in _names(h)
                            for h in p.handlers
                        )
                    ):
                        ok = True
                        break
                    c, p = p, parents.get(p)
                handled = handled and ok
        assert handled, f"the arm at line {node.lineno} lets a refusal escape as a 500"
    assert sites >= 6
