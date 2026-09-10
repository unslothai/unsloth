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
    tool_call_chunk,
    web_search_tool,
)


@pytest.mark.parametrize("site", ["plain", "round", "final"])
def test_a_resume_keeps_the_thought_that_preceded_the_prose(monkeypatch, site):
    signal = p.PreemptSignal()
    policy = RecordingPolicy()
    attempt = 1 if site == "final" else 0
    streams = ([[*tool_call()]] if site == "final" else []) + [
        [
            reasoning("The secret intermediate result is 42."),
            delta("Therefore the answer"),
            finish(),
            done(),
        ],
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

    out = list(
        LlamaCppBackend._iter_text_cancellable(Response(), None, first_token_deadline = 1200.0)
    )
    assert "Hello" in "".join(out)


class _SlowPolicy(RecordingPolicy):
    """Room comes back after a while, as it does behind a queue of chats."""

    def __init__(self, wait_s: float):
        super().__init__()
        self.wait_s = wait_s

    def await_resume(
        self,
        timeout = None,
        *,
        cancel_event = None,
    ):
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
        def await_resume(
            self,
            timeout = None,
            *,
            cancel_event = None,
        ):
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match = "boom"):
        list(llama_cpp._await_resume(Raising(), threading.Event()))


# ── Round 23: what the routed wrapper, a spent cap, a declined pass and a rollback do ────────


def test_stop_reaches_the_controller_through_the_deferred_wrapper():
    # Every route hands the stream a DeferredPreemptionPolicy. Its await_resume took only a
    # timeout, so the caller's keyword raised TypeError and the fallback waited without Stop.
    cancel = threading.Event()
    cancel.set()
    seen: list = []

    class Inner:
        def await_resume(
            self,
            timeout = None,
            *,
            cancel_event = None,
        ):
            seen.append(cancel_event)
            return not (cancel_event is not None and cancel_event.is_set())

    wrapper = p.DeferredPreemptionPolicy(Inner())
    with pytest.raises(StopIteration) as stopped:
        next(llama_cpp._await_resume(wrapper, cancel))
    assert seen == [cancel]
    assert stopped.value.value is False


def test_the_deferred_wrapper_still_serves_an_inner_policy_without_the_keyword():
    class Older:
        def await_resume(self, timeout = None):
            return True

    wrapper = p.DeferredPreemptionPolicy(Older())
    assert wrapper.await_resume(1.0, cancel_event = threading.Event()) is True


@pytest.mark.parametrize("site", ["round", "final"])
def test_a_policy_that_raises_during_the_wait_is_not_a_grant(monkeypatch, site):
    # The lease went back with on_preempted, so decoding on after an exception ran on room
    # nobody booked. The turn ends with its partial, as a refused resume does.
    class FailedResume(RecordingPolicy):
        def await_resume(
            self,
            timeout = None,
            *,
            cancel_event = None,
        ):
            self.events.append("resume-failed")
            raise RuntimeError("resume bookkeeping unavailable")

    final = site == "final"
    signal, policy = p.PreemptSignal(), FailedResume()
    offset = 1 if final else 0
    streams = ([[*tool_call()]] if final else []) + [
        [delta("Partial answer"), finish(), done()],
        [delta(" kept decoding without a grant"), finish(), done()],
    ]
    rec = PreemptRecorder(
        monkeypatch, streams, signal = signal, pause_after = {offset: 1}, execute_tool = final
    )
    events = run_tool_loop(
        rec.backend,
        signal = signal,
        policy = policy,
        tools = [web_search_tool()],
        max_tool_iterations = 1 if final else 5,
        permission_mode = "off",
    )
    states = [e["state"] for e in events if isinstance(e, dict) and e.get("type") == "preempt"]
    assert "resume-failed" in policy.events
    assert "resumed" not in policy.events
    assert len(rec.payloads) == offset + 1, "no second upstream request"
    assert states == ["paused"]
    assert [e for e in events if isinstance(e, dict) and e.get("type") == "metadata"][-1][
        "finish_reason"
    ] == "length"


def test_a_rollback_closes_the_tool_card_it_abandons(monkeypatch):
    # A provisional tool_start went out while the arguments streamed; the pause rolled the
    # attempt back to before the call and the resumed attempt answered in prose, so the card
    # had no result and no run.
    signal = p.PreemptSignal()
    rec = PreemptRecorder(
        monkeypatch,
        [
            [
                tool_call_chunk("call_before_pause", arguments = {"query": "x" * 300}),
                finish("tool_calls"),
                done(),
            ],
            [delta("The resumed answer."), finish(), done()],
        ],
        signal = signal,
        pause_after = {0: 1},
    )
    events = run_tool_loop(
        rec.backend,
        signal = signal,
        policy = RecordingPolicy(),
        tools = [web_search_tool()],
        max_tool_iterations = 5,
        permission_mode = "off",
    )
    kinds = [
        (e["type"], e.get("tool_call_id"))
        for e in events
        if isinstance(e, dict) and e["type"] in ("tool_start", "tool_end", "preempt")
    ]
    assert kinds == [
        ("tool_start", "call_before_pause"),
        ("tool_end", "call_before_pause"),
        ("preempt", None),
        ("preempt", None),
    ]


@pytest.mark.parametrize("site", ["plain", "round", "final"])
def test_a_pause_with_the_callers_cap_spent_ends_the_turn(monkeypatch, site):
    # Four one-token deltas spend a cap of four. Reopening the stream for one floored token
    # went past the cap by that token at all three sites; now the turn ends with `length`.
    signal = p.PreemptSignal()
    streams = [
        [delta("a") for _ in range(4)] + [finish("length"), done()],
        [delta("b"), finish("length"), done()],
    ]
    rec = PreemptRecorder(monkeypatch, streams, signal = signal, pause_after = {0: 4})
    if site == "plain":
        events = run_plain(rec.backend, signal = signal, policy = RecordingPolicy(), max_tokens = 4)
    else:
        events = run_tool_loop(
            rec.backend,
            signal = signal,
            policy = RecordingPolicy(),
            tools = [web_search_tool()],
            max_tokens = 4,
            max_tool_iterations = 0 if site == "final" else 5,
            permission_mode = "off",
        )
    assert [p["max_tokens"] for p in rec.payloads] == [4]
    metadata = [e for e in events if isinstance(e, dict) and e.get("type") == "metadata"]
    assert metadata and metadata[-1]["finish_reason"] == "length"
    assert not any(
        isinstance(e, dict) and e.get("reason") == llama_cpp.PREEMPT_GAVE_UP_REASON for e in events
    ), "nothing was given up: the caller's own cap ended it"
    # Counted once. The interrupted attempt is folded into the accumulator before the
    # terminal event is built, and the event used to add the attempt's own usage and
    # timings on top of that, so a four-token answer reported eight.
    completion = int((metadata[-1].get("usage") or {}).get("completion_tokens") or 0)
    assert completion <= 4, f"the interrupted attempt was counted twice: {completion}"
    predicted_n = int((metadata[-1].get("timings") or {}).get("predicted_n") or 0)
    assert predicted_n <= 4, f"the attempt's timings were folded twice: {predicted_n}"


@pytest.mark.parametrize("site", ["round", "final"])
def test_a_spent_cap_counts_the_interrupted_attempt_once(monkeypatch, site):
    # llama-server reports cumulative timings on every chunk, so the attempt's own reading is
    # in hand when the pause lands. It is folded into the accumulator, and the terminal event
    # then added the same reading again: four tokens reported as eight.
    signal = p.PreemptSignal()
    streams = [
        [
            delta("a", timings = {"predicted_n": i + 1, "predicted_ms": 10.0 * (i + 1)})
            for i in range(4)
        ]
        + [finish("length"), done()],
        [delta("b"), finish("length"), done()],
    ]
    rec = PreemptRecorder(monkeypatch, streams, signal = signal, pause_after = {0: 4})
    events = run_tool_loop(
        rec.backend,
        signal = signal,
        policy = RecordingPolicy(),
        tools = [web_search_tool()],
        max_tokens = 4,
        max_tool_iterations = 0 if site == "final" else 5,
        permission_mode = "off",
    )
    metadata = [e for e in events if isinstance(e, dict) and e.get("type") == "metadata"]
    assert metadata and metadata[-1]["finish_reason"] == "length"
    usage = metadata[-1].get("usage") or {}
    timings = metadata[-1].get("timings") or {}
    assert int(usage.get("completion_tokens") or 0) == 4, usage
    assert int(timings.get("predicted_n") or 0) == 4, timings
    assert abs(float(timings.get("predicted_ms") or 0) - 40.0) < 1e-6, timings


def test_a_declined_pause_hands_the_final_pass_the_whole_partial_and_the_whole_charge(monkeypatch):
    # The final pass extends the declined partial in the prompt, so its snapshots carry it (a
    # non-streaming drain keeps only the last one) and its cap deducts every token that call
    # spent, the granted pause's six as well as the declined seven.
    monkeypatch.setattr(p, "DEFAULT_MAX_PREEMPT_RESUMES", 1)
    signal = p.PreemptSignal()
    rec = PreemptRecorder(
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
        rec.backend,
        signal = signal,
        policy = DecliningPolicy(),
        tools = [web_search_tool()],
        max_tool_iterations = 5,
        max_tokens = 100,
        permission_mode = "off",
    )
    snapshots = [e["text"] for e in events if isinstance(e, dict) and e.get("type") == "content"]
    assert (
        snapshots[-1]
        == "First preserved sentence. Second sentence with new work. A finished answer."
    )
    assert all(later.startswith(earlier) for earlier, later in zip(snapshots, snapshots[1:]))
    assert [p["max_tokens"] for p in rec.payloads] == [100, 94, 87]
    # No turn boundary between the partial and the pass that extends it.
    kinds = [e.get("type") for e in events if isinstance(e, dict)]
    assert "status" not in kinds[kinds.index("content") :]


def test_a_declined_pause_with_the_callers_cap_spent_ends_the_turn(monkeypatch):
    # Four one-token deltas spend a cap of four as the pause is declined. The decline handed
    # the final pass a cap floored at one, a token past the caller's; the turn ends instead.
    monkeypatch.setattr(p, "DEFAULT_MAX_PREEMPT_RESUMES", 0)
    signal = p.PreemptSignal()
    streams = [
        [delta("a") for _ in range(4)] + [finish("length"), done()],
        [delta("b"), finish("length"), done()],
    ]
    rec = PreemptRecorder(monkeypatch, streams, signal = signal, pause_after = {0: 4})
    events = run_tool_loop(
        rec.backend,
        signal = signal,
        policy = DecliningPolicy(),
        tools = [web_search_tool()],
        max_tokens = 4,
        max_tool_iterations = 5,
        permission_mode = "off",
    )
    assert [q["max_tokens"] for q in rec.payloads] == [4]
    metadata = [e for e in events if isinstance(e, dict) and e.get("type") == "metadata"]
    assert metadata and metadata[-1]["finish_reason"] == "length"
    assert int((metadata[-1].get("usage") or {}).get("completion_tokens") or 0) <= 4


@pytest.mark.parametrize("ending", ["declined", "not_resumed"])
def test_the_final_pass_gave_up_counts_the_interrupted_attempt_once(monkeypatch, ending):
    # The refusal never folded the attempt, so a stream with no terminal usage reported none
    # of its tokens; the not-resumed end folded it and then added the reading again.
    if ending == "declined":
        monkeypatch.setattr(p, "DEFAULT_MAX_PREEMPT_RESUMES", 0)
        policy = DecliningPolicy()
    else:
        policy = RecordingPolicy(resume = False)
    signal = p.PreemptSignal()
    streams = [
        [
            delta("a", timings = {"predicted_n": i + 1, "predicted_ms": 10.0 * (i + 1)})
            for i in range(4)
        ]
        + [finish(), done()],
        [delta("b"), finish(), done()],
    ]
    rec = PreemptRecorder(monkeypatch, streams, signal = signal, pause_after = {0: 4})
    events = run_tool_loop(
        rec.backend,
        signal = signal,
        policy = policy,
        tools = [web_search_tool()],
        max_tool_iterations = 0,
        permission_mode = "off",
    )
    assert len(rec.payloads) == 1
    metadata = [e for e in events if isinstance(e, dict) and e.get("type") == "metadata"]
    assert metadata and metadata[-1]["finish_reason"] == "length"
    usage = metadata[-1].get("usage") or {}
    timings = metadata[-1].get("timings") or {}
    assert int(usage.get("completion_tokens") or 0) == 4, usage
    assert int(timings.get("predicted_n") or 0) == 4, timings


def test_the_thought_before_a_pause_survives_a_resumed_turn_that_calls_a_tool(monkeypatch):
    # The resumed attempt went on thinking and then called a tool. Its assistant message
    # carried only the later thought, and the merge replaced the earlier one.
    signal = p.PreemptSignal()
    rec = PreemptRecorder(
        monkeypatch,
        [
            [reasoning("EARLIER_THOUGHT"), finish(), done()],
            [reasoning("LATER_THOUGHT"), tool_call_chunk(), finish("tool_calls"), done()],
            [delta("The answer."), finish(), done()],
        ],
        signal = signal,
        pause_after = {0: 1},
        execute_tool = True,
        _supports_reasoning = True,
    )
    monkeypatch.setattr(rec.backend, "count_chat_tokens", lambda *a, **k: 20)
    run_tool_loop(
        rec.backend,
        signal = signal,
        policy = RecordingPolicy(),
        tools = [web_search_tool()],
        max_tool_iterations = 1,
        permission_mode = "off",
    )
    replay = next(m for m in rec.payloads[-1]["messages"] if m.get("tool_calls"))
    assert replay["reasoning_content"] == "EARLIER_THOUGHTLATER_THOUGHT"


@pytest.mark.parametrize("parallel", ["0", "1"])
def test_two_spellings_of_one_call_run_once_in_a_parallel_round(monkeypatch, parallel):
    # `kernel` heals to {"query": "kernel"}. The ledger keys on the healed call; the parallel
    # gate keyed on the arguments as they arrived, so both ran.
    import json as _json

    monkeypatch.setenv("UNSLOTH_PARALLEL_TOOL_CALLS", parallel)
    raw_calls = [
        {
            "index": i,
            "id": f"c{i}",
            "type": "function",
            "function": {
                "name": "web_search",
                "arguments": args if isinstance(args, str) else _json.dumps(args),
            },
        }
        for i, args in enumerate(["kernel", {"query": "kernel"}])
    ]
    frame = (
        "data: "
        + _json.dumps({"choices": [{"index": 0, "delta": {"tool_calls": raw_calls}}]})
        + "\n"
    )
    rec = PreemptRecorder(
        monkeypatch, [[frame, finish("tool_calls"), done()], [delta("Answer."), finish(), done()]]
    )
    monkeypatch.setattr(rec.backend, "count_chat_tokens", lambda *a, **k: 20)
    ran: list = []

    def execute(name, arguments, **kwargs):
        ran.append((name, arguments))
        return "OK"

    monkeypatch.setattr("core.inference.tools.execute_tool", execute)
    run_tool_loop(
        rec.backend,
        signal = None,
        policy = None,
        tools = [web_search_tool()],
        max_tool_iterations = 1,
        permission_mode = "off",
    )
    assert ran == [("web_search", {"query": "kernel"})]
