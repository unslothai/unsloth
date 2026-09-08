# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A resumed attempt carries the thought that preceded its prose; Stop ends a pause's wait."""

import threading
import time

import pytest

from core.inference import llama_cpp
from core.inference import llama_preemption as p
from core.inference.llama_cpp import LlamaCppBackend

from .preempt_fakes import (
    DecliningPolicy,
    PreemptRecorder,
    RecordingPolicy,
    delta,
    done,
    finish,
    reasoning,
    run_plain,
    run_tool_loop,
    tool_call,
    web_search_tool,
)


@pytest.mark.parametrize("site", ["plain", "round", "final"])
def test_a_resume_keeps_the_thought_that_preceded_the_prose(monkeypatch, site):
    signal = p.PreemptSignal()
    policy = RecordingPolicy()
    attempt = 1 if site == "final" else 0
    streams = ([[*tool_call()]] if site == "final" else []) + [
        [reasoning("The secret intermediate result is 42."), delta("Therefore the answer"), finish(), done()],
        [delta(" is forty-two."), finish(), done()],
    ]
    recorder = PreemptRecorder(
        monkeypatch,
        streams,
        signal = signal,
        pause_after = {attempt: 2},
        _supports_reasoning = True,
        execute_tool = True,
    )
    if site == "plain":
        run_plain(recorder.backend, signal = signal, policy = policy)
    else:
        run_tool_loop(
            recorder.backend,
            signal = signal,
            policy = policy,
            tools = [web_search_tool()],
            max_tool_iterations = 1 if site == "final" else 5,
            permission_mode = "off",
        )
    tail = recorder.payloads[attempt + 1]["messages"][-1]
    assert tail.get("reasoning_content") == "The secret intermediate result is 42."
    assert tail["content"] == "Therefore the answer"


def test_a_stop_during_a_pause_ends_the_wait_for_room(monkeypatch):
    monkeypatch.setenv(p.PREEMPT_ENV, "1")
    controller = p.PreemptionController("stop-while-paused")
    controller.configure(budget = 4096, kv_unified = True, slots = 2)

    class Lease:
        is_released = False

    signal = p.PreemptSignal()
    controller.register(
        "paused", lease = Lease(), tokens = 1500, signal = signal, state = p.ParticipantState.PAUSED
    )
    controller.register("busy", tokens = 3000)
    policy = p.ControllerPreemptionPolicy(controller, "paused", signal, loop = object())
    cancel = threading.Event()
    entered = threading.Event()
    controller.set_residency_probe(entered.set)
    result = []
    worker = threading.Thread(
        target = lambda: result.append(policy.await_resume(timeout = 5.0, cancel_event = cancel))
    )
    worker.start()
    assert entered.wait(2)
    cancel.set()
    worker.join(2)
    assert not worker.is_alive(), "the waiter outlived the Stop"
    assert result == [False]


def test_the_resume_cap_in_a_tool_round_keeps_the_attempt_and_its_charge(monkeypatch):
    monkeypatch.setattr(p, "DEFAULT_MAX_PREEMPT_RESUMES", 1)
    signal = p.PreemptSignal()
    policy = DecliningPolicy()
    recorder = PreemptRecorder(
        monkeypatch,
        [
            [delta("First preserved sentence. "), finish(), done()],
            [delta("Second sentence with new work. "), finish(), done()],
            [delta("A finished answer."), finish(), done()],
        ],
        signal = signal,
        pause_after = {0: 1, 1: 1},
    )
    events = run_tool_loop(
        recorder.backend,
        signal = signal,
        policy = policy,
        tools = [web_search_tool()],
        max_tool_iterations = 5,
        max_tokens = 100,
        permission_mode = "off",
    )
    assert "declined" in policy.events
    # The cap declines the pause and the final pass extends what both attempts wrote, under
    # what is left of the allowance, instead of starting afresh with the whole cap.
    assert len(recorder.payloads) == 3
    final = recorder.payloads[2]
    assert final.get("continue_final_message") is True
    trailing = final["messages"][-1]
    assert trailing["role"] == "assistant"
    assert trailing["content"] == "First preserved sentence. Second sentence with new work. "
    assert final["max_tokens"] < recorder.payloads[1]["max_tokens"] <= 100
    assert any(isinstance(e, dict) and e.get("type") == "content" for e in events)
    assert not any(isinstance(e, dict) and e.get("reason") == "preempt_gave_up" for e in events)


def test_a_park_before_the_first_token_keeps_the_first_token_deadline_alive(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(llama_cpp.time, "monotonic", lambda: clock[0])

    class Response:
        request = type("Request", (), {"extensions": {"timeout": {}}})()

        def iter_text(self):
            yield ": preempted\n\n"
            for _ in range(601):
                clock[0] += 2.0
                yield ": preempt-keepalive\n\n"
            yield ": resumed\n\n"
            yield delta("Hello", terminator = "\n\n")

    out = list(LlamaCppBackend._iter_text_cancellable(Response(), None, first_token_deadline = 1200.0))
    assert "Hello" in "".join(out)


class _SlowPolicy(RecordingPolicy):
    """Room comes back after a while, as it does behind a queue of chats."""

    def __init__(self, wait_s: float):
        super().__init__()
        self.wait_s = wait_s

    def await_resume(self, timeout = None, *, cancel_event = None):
        self.events.append("await")
        time.sleep(self.wait_s)
        return True


def test_a_pause_says_it_is_still_waiting_so_a_durable_lease_is_renewed(monkeypatch):
    monkeypatch.setattr(llama_cpp, "_PREEMPT_KEEPALIVE_S", 0.05)
    signal = p.PreemptSignal()
    policy = _SlowPolicy(0.3)
    recorder = PreemptRecorder(
        monkeypatch,
        [
            [delta("Once"), delta(" upon")],
            [delta(" a time."), finish(), done()],
        ],
        signal = signal,
        pause_attempts = (0,),
        pause_after = 2,
    )
    events = run_plain(recorder.backend, signal = signal, policy = policy)
    states = [e["state"] for e in events if isinstance(e, dict) and e.get("type") == "preempt"]
    assert states[0] == "paused" and states[-1] == "resumed"
    assert states.count("keepalive") >= 2, "a long wait was silent, and a lease is renewed on sound"
    assert states.index("paused") < states.index("keepalive") < states.index("resumed")
    assert "Once upon a time." in "".join(e for e in events if isinstance(e, str))

    # The routes forward it as the comment the durable producer renews on; the frontend, which
    # reads only its four named signals, ignores it.
    from core.inference import chat_generation_runs as runs
    from routes import inference as routes

    assert (
        routes._OPENAI_PREEMPT_SSE_BY_STATE["keepalive"].strip() == runs._PREEMPT_KEEPALIVE_MARKER
    )
    assert runs._admission_status_chunks(runs._PREEMPT_KEEPALIVE_MARKER + "\n\n") == []


def test_a_policy_that_raises_during_the_wait_raises_on_the_caller():
    class Raising(RecordingPolicy):
        def await_resume(self, timeout = None, *, cancel_event = None):
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match = "boom"):
        list(llama_cpp._await_resume(Raising(), threading.Event()))
