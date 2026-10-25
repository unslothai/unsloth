# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Every wait for the chats an eviction paused answers the request's Stop, not just the chat arm.

A victim still prefilling cannot pause, so these waits can run a minute. A raw holder
(completions, embeddings, Responses, the passthrough, TTS, a caption), the Anthropic arm and a
tool round's re-cost each waited it out and then ended on a 503 or a refused round, long after
the client had stopped or gone."""

from __future__ import annotations

import ast
import asyncio
import inspect
import threading
import time
from types import SimpleNamespace
import uuid

from fastapi import HTTPException
import pytest

from core.inference import llama_cpp as llama_cpp_mod
from core.inference import llama_preemption as preemption
from core.inference.llama_admission import LlamaAdmissionCancelled
from core.inference.llama_preemption import (
    ParticipantState,
    PreemptSignal,
    get_preemption_controller,
)
import routes.inference as inference

pytestmark = pytest.mark.usefixtures("preemption_opted_in")

_STOP_WITHIN_S = 1.5


@pytest.fixture
def wedged(monkeypatch):
    """A cache held by one decoding chat that, told to pause, never lets go within the test."""
    monkeypatch.setenv(preemption.PREEMPT_ENV, "1")
    monkeypatch.setattr(inference, "_openai_llama_preemption_will_apply", lambda *a, **k: True)
    monkeypatch.setattr(inference, "_openai_llama_admission_budget", lambda *a, **k: 16384)
    monkeypatch.setattr(inference, "_refresh_residency_before_planning", lambda controller: None)

    async def _no_read(controller):
        return None

    monkeypatch.setattr(inference, "_refresh_residency_before_planning_async", _no_read)
    monkeypatch.setattr(inference, "_UNPAUSABLE_VICTIM_WAIT_S", 3.0)
    monkeypatch.setattr(inference, "_UNPAUSABLE_VICTIM_HARD_WAIT_S", 3.0)
    fake = SimpleNamespace(base_url = f"http://eviction-stop-{uuid.uuid4().hex}.test")
    controller = get_preemption_controller(inference._preempt_key(fake))
    controller.configure(budget = 16384, kv_unified = True)
    controller.register("chat", tokens = 15000)
    controller.note_measured("chat")
    controller.note_resident(15000, started_at_seq = controller.residency_epoch())
    yield fake, controller
    for gen_id in list(controller._participants):
        controller.unregister(gen_id)


def _stop_after(seconds):
    stop = threading.Event()
    timer = threading.Timer(seconds, stop.set)
    timer.daemon = True
    timer.start()
    return stop


class _LeavesAfter:
    """A client that disconnects ``seconds`` after its first check."""

    def __init__(self, seconds):
        self._seconds = seconds
        self._first = None

    async def is_disconnected(self):
        now = time.monotonic()
        if self._first is None:
            self._first = now
        return now - self._first >= self._seconds


def _raw_holder(fake, controller, gen_id = "raw"):
    controller.register(gen_id, tokens = 9000, state = ParticipantState.STREAMING_RAW)
    inference._plan_for_unpausable_holder(controller, gen_id, refresh = False)
    assert controller.participant("chat").state == ParticipantState.PREEMPTING
    assert inference._openai_llama_armed_victims_pending(controller, gen_id)
    return gen_id


# -- raw holders: completions, embeddings, Responses, passthrough, TTS, captions -----------------


class TestARawHolder:
    def test_stop_ends_its_wait_for_the_victims(self, wedged):
        fake, controller = wedged

        async def route():
            holder = _raw_holder(fake, controller)
            await inference._openai_llama_let_victims_go(fake, holder, _stop_after(0.2))

        started = time.monotonic()
        with pytest.raises(LlamaAdmissionCancelled):
            asyncio.run(route())
        assert time.monotonic() - started < _STOP_WITHIN_S, "Stop waited for the victim"

    def test_a_disconnect_ends_its_wait_for_the_victims(self, wedged):
        """The streaming completions proxy has no watcher before its upstream response, so the
        wait checks the client itself, as the pre-header send does next."""
        fake, controller = wedged

        async def route():
            holder = _raw_holder(fake, controller)
            await inference._openai_llama_let_victims_go(
                fake, holder, threading.Event(), request = _LeavesAfter(0.2)
            )

        started = time.monotonic()
        with pytest.raises(LlamaAdmissionCancelled):
            asyncio.run(route())
        assert time.monotonic() - started < _STOP_WITHIN_S, "a gone client waited for the victim"

    def test_a_worker_thread_holder_answers_stop_and_is_uncounted(self, wedged):
        """The sync plan (caption / OCR on a worker thread) waits inline."""
        fake, controller = wedged
        started = time.monotonic()
        with pytest.raises(LlamaAdmissionCancelled):
            inference._openai_llama_count_direct_holder(
                fake, 9000, "direct-holder", _stop_after(0.2)
            )
        assert time.monotonic() - started < _STOP_WITHIN_S, "Stop waited for the victim"
        assert set(controller._participants) == {"chat"}, "the stopped holder stayed counted"

    def test_the_passthrough_answers_stop_with_the_admission_499(self, wedged, monkeypatch):
        fake, controller = wedged
        released: list[str] = []
        upstream: list[object] = []
        lease = SimpleNamespace(tokens = 9000, release = lambda: released.append("lease"))
        reservation = SimpleNamespace(
            lease_nowait = lambda: lease, cancel = lambda: released.append("reservation")
        )

        async def reserve(**_kwargs):
            return reservation, None

        async def granted(*_args, **_kwargs):
            return lease

        async def not_cancelled(*_args, **_kwargs):
            return None

        async def counted(*, llama_backend, gen_id, **_kwargs):
            _raw_holder(llama_backend, controller, gen_id)

        async def send(*args, **_kwargs):
            upstream.append(args)

        monkeypatch.setattr(inference, "_openai_llama_admission_reserve_async", reserve)
        monkeypatch.setattr(inference, "_wait_for_openai_admission_non_streaming", granted)
        monkeypatch.setattr(inference, "_raise_if_openai_admission_cancelled", not_cancelled)
        monkeypatch.setattr(inference, "_openai_llama_count_raw_holder_async", counted)
        monkeypatch.setattr(inference, "_openai_passthrough_non_streaming_upstream", send)
        monkeypatch.setattr(inference, "_llama_admission_log", lambda *a, **k: None)

        started = time.monotonic()
        with pytest.raises(HTTPException) as caught:
            asyncio.run(
                inference._openai_passthrough_non_streaming(
                    fake,
                    SimpleNamespace(),
                    "model",
                    monitor_id = "passthrough-stop",
                    request = None,
                    cancel_event = _stop_after(0.2),
                )
            )
        assert caught.value.status_code == 499, caught.value.status_code
        assert time.monotonic() - started < _STOP_WITHIN_S, "Stop waited for the victim"
        assert not upstream, "the request went upstream after Stop"
        assert released == ["lease"]
        assert set(controller._participants) == {"chat"}, "the stopped holder stayed counted"

    def test_the_streaming_passthrough_answers_stop_with_the_admission_499(
        self, wedged, monkeypatch
    ):
        fake, controller = wedged
        fake.context_length = 4096
        released: list[str] = []

        async def counted(*, llama_backend, gen_id, **_kwargs):
            _raw_holder(llama_backend, controller, gen_id)

        monkeypatch.setattr(inference, "_openai_llama_count_raw_holder_async", counted)
        lease = SimpleNamespace(release = lambda: released.append("lease"))
        tracker = SimpleNamespace(__exit__ = lambda *a: released.append("tracker"))
        started = time.monotonic()
        with pytest.raises(HTTPException) as caught:
            asyncio.run(
                inference._openai_passthrough_stream_admitted(
                    None,
                    _stop_after(0.2),
                    fake,
                    SimpleNamespace(model_extra = {}),
                    "model",
                    "chatcmpl-stop",
                    admission_lease = lease,
                    tracker = tracker,
                )
            )
        assert caught.value.status_code == 499, caught.value.status_code
        assert time.monotonic() - started < _STOP_WITHIN_S, "Stop waited for the victim"
        assert released == ["lease", "tracker"], f"the slot or the tracker was kept: {released}"
        assert set(controller._participants) == {"chat"}, "the stopped holder stayed counted"


# -- an armed chat on the Anthropic route -------------------------------------------------------


class TestTheAnthropicArm:
    def test_a_disconnect_ends_the_arms_wait(self, wedged, monkeypatch):
        """A non-streaming /v1/messages registers its cancel event only once admitted, so during
        the arm nothing but the client's own disconnect can end the wait."""
        fake, controller = wedged

        def _enrol(*, gen_id, signal, **_kwargs):
            controller.register(gen_id, tokens = 4000, signal = signal)
            return controller

        monkeypatch.setattr(inference, "_openai_llama_preemption_enrol", _enrol)
        started = time.monotonic()
        with pytest.raises(LlamaAdmissionCancelled):
            asyncio.run(
                inference._openai_llama_preemption_arm_async(
                    request = _LeavesAfter(0.2),
                    llama_backend = fake,
                    gen_id = "msg",
                    signal = PreemptSignal(),
                    cancel_event = threading.Event(),
                )
            )
        assert time.monotonic() - started < _STOP_WITHIN_S, "a gone client waited for the victim"
        assert controller.participant("msg") is None, "the stopped arm stayed enrolled"

    def test_the_route_answers_a_disconnect_during_the_arm_with_a_499(self, monkeypatch):
        from models.inference import AnthropicMessagesRequest

        from .test_llama_preemption_cancel_during_arm import _ledger_is_empty
        from .test_llama_preemption_is_armed_on_chat_surfaces import BASE, _ArmedBackend

        backend = _ArmedBackend(tools = False)
        backend.supports_tool_passthrough = False
        backend.is_vision = False
        backend.count_chat_tokens = lambda *a, **k: 2
        monkeypatch.setattr(inference, "get_llama_cpp_backend", lambda: backend)
        monkeypatch.setattr(inference, "_UNPAUSABLE_VICTIM_WAIT_S", 3.0)
        monkeypatch.setattr(inference, "_UNPAUSABLE_VICTIM_HARD_WAIT_S", 3.0)
        real_plan = inference._openai_llama_preemption_plan
        wedged_at: list[float] = []

        def _plan(controller, *, gen_id, **kwargs):
            policy = real_plan(controller, gen_id = gen_id, **kwargs)
            # A victim still prefilling: told to pause, it holds its cells all wait long.
            controller.register("wedged", tokens = 100)
            controller.set_state("wedged", ParticipantState.PREEMPTING)
            controller.participant(gen_id).awaiting_victims = ["wedged"]
            wedged_at.append(time.monotonic())
            return policy

        monkeypatch.setattr(inference, "_openai_llama_preemption_plan", _plan)

        class _Request:
            state = SimpleNamespace()
            url = SimpleNamespace(path = "/v1/messages")
            method = "POST"

            async def is_disconnected(self):
                # Gone 0.2 s into the arm's wait, connected through admission before it.
                return bool(wedged_at) and time.monotonic() - wedged_at[0] >= 0.2

        async def _run():
            payload = AnthropicMessagesRequest(
                max_tokens = 16, messages = [{"role": "user", "content": "hi"}]
            )
            return await inference.anthropic_messages(
                payload, request = _Request(), current_subject = "t"
            )

        started = time.monotonic()
        try:
            with pytest.raises(HTTPException) as caught:
                asyncio.run(_run())
        finally:
            get_preemption_controller(BASE).unregister("wedged")
        assert wedged_at, "the arm never planned, so this proves nothing"
        assert caught.value.status_code == 499, caught.value.status_code
        assert time.monotonic() - started < 2.0, "a gone client waited for the victim"
        assert not backend.seen, "the request went upstream after the client left"
        _ledger_is_empty(BASE)


# -- a tool round and the plain path's fitted re-cost --------------------------------------------


class _Lease:
    tokens = 4000


class _Reservation:
    def lease_nowait(self):
        return _Lease()


class TestARound:
    def test_stop_ends_a_tool_rounds_wait_as_a_cancellation(self, wedged, monkeypatch):
        """Stop, not a refused round: the tool loop ends the turn as a user Stop on this."""
        fake, controller = wedged
        controller.register("round", tokens = 100, signal = PreemptSignal())
        controller.note_measured("round")
        monkeypatch.setattr(
            inference, "_openai_llama_admission_charged_prompt_tokens", lambda *a, **k: 3000
        )
        started = time.monotonic()
        with pytest.raises(LlamaAdmissionCancelled):
            inference._openai_llama_publish_round_charge(
                llama_backend = fake,
                gen_id = "round",
                reservation = _Reservation(),
                payload = None,
                conversation = [],
                rendered_tools = None,
                observe_tokens = lambda generated: controller.plan_preemptions(
                    needed = 0, exclude = "round"
                ),
                cancel_event = _stop_after(0.2),
            )
        assert time.monotonic() - started < _STOP_WITHIN_S, "Stop waited for the victim"

    def test_the_plain_refit_hands_a_stop_to_the_generator(self, monkeypatch):
        from .test_a_fitted_plain_prompt_is_what_the_ledger_counts import (
            _LONG,
            _controller,
            _refit,
        )

        def stopped(**_kwargs):
            raise LlamaAdmissionCancelled("stopped while waiting for cache room")

        monkeypatch.setattr(inference, "_openai_llama_publish_round_charge", stopped)
        with pytest.raises(LlamaAdmissionCancelled):
            _refit(monkeypatch, _controller(), _LONG, _LONG[-1:])

    def test_a_stopped_plain_refit_sends_nothing_and_ends_as_a_stop(self, monkeypatch):
        from .test_llama_admission_enforced_paths import (
            _SHARE,
            _count_by_length,
            _done,
            _make_backend,
            _sse,
        )

        payloads: list[dict] = []
        backend = _make_backend(monkeypatch, [[_sse({"content": "hi"}), _done()]], payloads)
        monkeypatch.setattr(backend, "count_chat_tokens", _count_by_length)
        history = [
            {"role": "user", "content": f"turn {index} " + "word " * 400} for index in range(10)
        ]

        def _stopped(_messages):
            raise LlamaAdmissionCancelled("stopped while waiting for cache room")

        events = list(
            backend.generate_chat_completion(
                messages = history,
                context_overflow = "truncate_oldest",
                admission_output_allowance = _SHARE,
                on_prompt_fitted = _stopped,
            )
        )
        assert payloads == [], "a stopped refit still sent the fitted prompt"
        assert not [
            e for e in events if isinstance(e, dict) and e.get("type") == "metadata"
        ], "a Stop was reported as a finished turn"


# -- every wait site, by AST ---------------------------------------------------------------------

_WAITS = {
    "_openai_llama_let_victims_go",
    "_openai_llama_preemption_arm_async",
    "_openai_llama_preemption_arm",
    "_openai_llama_publish_round_charge",
}
_INNER_WAITS = {"_openai_llama_wait_for_victims", "_openai_llama_wait_for_armed_victims"}
# Positional slot of the cancel event, where the signature takes one positionally.
_CANCEL_POSITION = {
    "_openai_llama_let_victims_go": 2,
    "_openai_llama_wait_for_victims": 2,
    "_openai_llama_wait_for_armed_victims": 2,
    "_openai_llama_count_direct_holder": 3,
}


def _tree(module):
    tree = ast.parse(inspect.getsource(module))
    parents = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    return tree, parents


def _callee(node):
    return getattr(node.func, "id", getattr(node.func, "attr", None))


def _passes_a_stop(call) -> bool:
    for keyword in call.keywords:
        if keyword.arg == "cancel_event":
            return not (isinstance(keyword.value, ast.Constant) and keyword.value.value is None)
    position = _CANCEL_POSITION.get(_callee(call))
    if position is not None and len(call.args) > position:
        value = call.args[position]
        return not (isinstance(value, ast.Constant) and value.value is None)
    return False


def _enclosing_function(node, parents):
    parent = parents.get(node)
    while parent is not None and not isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef)):
        parent = parents.get(parent)
    return parent


def _handler_names(handler):
    if handler.type is None:
        return {"BaseException"}
    items = handler.type.elts if isinstance(handler.type, ast.Tuple) else [handler.type]
    return {getattr(item, "id", getattr(item, "attr", None)) for item in items}


def _answers_cancellation(handler) -> bool:
    if "LlamaAdmissionCancelled" in _handler_names(handler):
        return True
    # `except Exception as e:` that tells the cancellation apart (the TTS route).
    return any(
        isinstance(node, ast.Name) and node.id == "LlamaAdmissionCancelled"
        for node in ast.walk(handler)
    )


def _handled_in_function(node, parents) -> bool:
    child, parent = node, parents.get(node)
    while parent is not None and not isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef)):
        if isinstance(parent, ast.Try) and child in parent.body:
            if any(_answers_cancellation(h) for h in parent.handlers):
                return True
        child, parent = parent, parents.get(parent)
    return False


def _handled(node, tree, parents, seen = ()) -> bool:
    """Answered by a try around it, or, inside a nested helper, around every call of that helper."""
    if _handled_in_function(node, parents):
        return True
    function = _enclosing_function(node, parents)
    if function is None or function.name in seen or _enclosing_function(function, parents) is None:
        return False
    callers = [
        call
        for call in ast.walk(tree)
        if isinstance(call, ast.Call) and getattr(call.func, "id", None) == function.name
    ]
    return bool(callers) and all(
        _handled(call, tree, parents, (*seen, function.name)) for call in callers
    )


def _swallowed(node, parents) -> bool:
    """A broad handler around it that is reached before any handler naming the cancellation."""
    child, parent = node, parents.get(node)
    while parent is not None and not isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef)):
        if isinstance(parent, ast.Try) and child in parent.body:
            for handler in parent.handlers:
                names = _handler_names(handler)
                if "LlamaAdmissionCancelled" in names:
                    break
                if names & {"Exception", "BaseException", "LlamaAdmissionError"}:
                    return True
        child, parent = parent, parents.get(parent)
    return False


def test_every_eviction_wait_is_handed_the_requests_stop():
    """Each wait for an eviction's victims gets the request's cancel event (or the disconnect it
    is told to watch); a call without one waits out a victim still prefilling."""
    tree, parents = _tree(inference)
    sites: dict[str, int] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _callee(node)
        if name in _WAITS or name in _INNER_WAITS:
            sites[name] = sites.get(name, 0) + 1
            assert _passes_a_stop(node), f"{name} at line {node.lineno} cannot be stopped"
        elif name == "to_thread" and node.args:
            target = getattr(node.args[0], "id", None)
            assert target not in _INNER_WAITS, (
                f"{target} at line {node.lineno} runs in a worker no Stop reaches: use "
                "_await_victims_in_worker"
            )
        elif name == "_await_victims_in_worker":
            sites[name] = sites.get(name, 0) + 1
            assert _passes_a_stop(node), f"the worker wait at line {node.lineno} has no Stop"
        elif name in ("_plan_for_unpausable_holder", "_openai_llama_count_direct_holder"):
            function = _enclosing_function(node, parents)
            # On the event loop the plan never waits; off it, it waits inline.
            if isinstance(function, ast.AsyncFunctionDef) or function.name in (
                "_openai_llama_count_completions_holder",
                "_openai_llama_count_embeddings_holder",
                "_openai_llama_count_raw_holder_async",
            ):
                continue
            assert _passes_a_stop(node), f"{name} at line {node.lineno} waits with no Stop"
    assert sites.get("_openai_llama_let_victims_go", 0) >= 9, sites
    assert sites.get("_openai_llama_publish_round_charge", 0) >= 3, sites
    assert sites.get("_openai_llama_preemption_arm_async", 0) >= 6, sites


def test_every_eviction_wait_answers_a_stop_as_a_cancellation():
    """A route's wait sits in a try that answers LlamaAdmissionCancelled as its own cancellation
    (the 499, or the stream ending), not a 500; a tool round's re-cost lets it reach the tool
    loop, which ends the turn as a Stop."""
    tree, parents = _tree(inference)
    routes = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _callee(node)
        if name in ("_openai_llama_let_victims_go", "_openai_llama_preemption_arm_async"):
            routes += 1
            assert _handled(node, tree, parents), (
                f"{name} at line {node.lineno} lets a Stop escape as a 500"
            )
        elif name == "_openai_llama_publish_round_charge":
            assert not _swallowed(node, parents), (
                f"the round at line {node.lineno} swallows a Stop and goes upstream anyway"
            )
    assert routes >= 15, routes

    llama_tree, llama_parents = _tree(llama_cpp_mod)
    rounds = 0
    for node in ast.walk(llama_tree):
        if isinstance(node, ast.Call) and _callee(node) in (
            "on_prompt_fitted",
            "_call_conversation_grew",
        ):
            rounds += 1
            assert _handled_in_function(node, llama_parents), (
                f"the re-cost at line {node.lineno} of llama_cpp.py treats a Stop as a failure"
            )
    assert rounds >= 3, rounds
