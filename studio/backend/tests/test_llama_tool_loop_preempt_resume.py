# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Pausing a tool run and finishing it in one response: the rounds, the synthesised
final answering pass, what a pause may not cost, and what a refusal tells the client."""

from __future__ import annotations

import sys
import threading
from pathlib import Path

from core.inference import llama_preemption as preemption
from core.inference.llama_cpp import PREEMPT_GAVE_UP_REASON, _TOKEN_REPORT_EVERY
from core.inference.llama_preemption import (
    ControllerPreemptionPolicy,
    ParticipantState,
    PreemptionController,
)

from .preempt_fakes import (
    DecliningPolicy as _DecliningPolicy,
    PreemptRecorder,
    RecordingPolicy as _RecordingPolicy,
    delta as _delta,
    done as _done,
    finish as _finish,
    run_tool_loop,
    tool_call as _tool_call_turn,
    tool_call_chunk as _tool_call,
    web_search_tool,
)

for _extra in (str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parent.parent)):
    if _extra not in sys.path:
        sys.path.insert(0, _extra)

_TOOL = web_search_tool(required = True)
_PLAIN_TOOL = web_search_tool()


def _Recorder(
    monkeypatch,
    streams,
    *,
    signal,
    pause_after_attempt = 0,
    pause_attempts = None,
    **kwargs,
):
    return PreemptRecorder(
        monkeypatch,
        streams,
        signal = signal,
        pause_attempts = ({pause_after_attempt} if pause_attempts is None else set(pause_attempts)),
        port = 48847,
        **kwargs,
    )


def _run(
    backend,
    *,
    signal,
    policy,
    tools = None,
    **kwargs,
):
    return run_tool_loop(backend, signal = signal, policy = policy, tools = tools or [_TOOL], **kwargs)


def _two_part(monkeypatch, signal, **kwargs):
    return _Recorder(
        monkeypatch,
        [
            [_delta("Once upon a time"), _finish(), _done()],
            [_delta(" there was a cat."), _finish(), _done()],
        ],
        signal = signal,
        **kwargs,
    )


def _content(events) -> list[str]:
    return [e["text"] for e in events if isinstance(e, dict) and e.get("type") == "content"]


def _gave_up(events) -> list[dict]:
    return [e for e in events if isinstance(e, dict) and e.get("reason") == PREEMPT_GAVE_UP_REASON]


def _metadata(events) -> list[dict]:
    return [e for e in events if isinstance(e, dict) and e.get("type") == "metadata"]


class TestARoundPauses:
    def test_a_paused_attempt_is_reopened_with_the_partial_to_extend(self, monkeypatch):
        signal = preemption.PreemptSignal()
        recorder = _two_part(monkeypatch, signal)
        _run(recorder.backend, signal = signal, policy = _RecordingPolicy())
        assert len(recorder.payloads) == 2, "a paused attempt must be re-opened, not abandoned"
        resumed = recorder.payloads[1]
        assert resumed.get("continue_final_message") is True
        assert resumed.get("add_generation_prompt") is False
        trailing = resumed["messages"][-1]
        assert trailing["role"] == "assistant"
        assert "Once upon a time" in trailing["content"]

    def test_the_signal_is_cleared_before_on_resumed_makes_this_chat_selectable_again(
        self, monkeypatch
    ):
        signal = preemption.PreemptSignal()
        seen: list[bool] = []

        class _Policy(_RecordingPolicy):
            def on_resumed(self):
                # `on_resumed` is what puts this participant back among the candidates. A
                # signal still set here aborts the resumed attempt on its first read, and
                # clearing after it races a sweep that could have chosen it again.
                seen.append(signal.is_set())
                super().on_resumed()

        _run(_two_part(monkeypatch, signal).backend, signal = signal, policy = _Policy())
        assert seen == [False], "the clear ran after the participant became selectable again"
        assert not signal.is_set()
        assert not signal.pending

    def test_a_pause_and_its_resume_are_both_announced_after_the_lease_goes_back(self, monkeypatch):
        signal = preemption.PreemptSignal()
        order: list[str] = []

        class _Policy(_RecordingPolicy):
            def on_preempted(self, checkpoint):
                order.append("on_preempted")
                super().on_preempted(checkpoint)

            def on_resumed(self):
                order.append("on_resumed")
                super().on_resumed()

        recorder = _two_part(monkeypatch, signal)
        for event in recorder.backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "write me a poem"}],
            tools = [_TOOL],
            cancel_event = threading.Event(),
            preempt_event = signal,
            preempt_policy = _Policy(),
        ):
            if isinstance(event, dict) and event.get("type") == "preempt":
                order.append(event["state"])
        assert order == ["on_preempted", "paused", "on_resumed", "resumed"]


class TestAPauseNeverRunsAToolTwice:
    """The one-shot ledger, and how far back a pause is allowed to roll."""

    @staticmethod
    def _executed(monkeypatch):
        calls: list = []

        def _execute(name, arguments, **_kwargs):
            calls.append((name, arguments))
            return f"RESULT<{(arguments or {}).get('query')}>"

        monkeypatch.setattr("core.inference.tools.execute_tool", _execute)
        return calls

    def test_a_completed_round_is_not_repeated_and_its_result_travels_with_the_resume(
        self, monkeypatch
    ):
        calls = self._executed(monkeypatch)
        signal = preemption.PreemptSignal()
        recorder = _Recorder(
            monkeypatch,
            [
                [
                    _tool_call("call_1", "web_search", {"query": "ropes"}),
                    _finish("tool_calls"),
                    _done(),
                ],
                [_delta("Based on the search, a rope"), _finish(), _done()],
                [_delta(" is a balanced tree."), _finish(), _done()],
            ],
            signal = signal,
            pause_after_attempt = 1,
        )
        _run(recorder.backend, signal = signal, policy = _RecordingPolicy())

        assert calls == [
            ("web_search", {"query": "ropes"})
        ], f"the tool ran {len(calls)} times: the resume re-executed a finished round"
        resumed = recorder.payloads[2]
        tool_row = next(m for m in resumed["messages"] if m.get("role") == "tool")
        assert tool_row.get("content") == "RESULT<ropes>"
        assert [
            m for m in resumed["messages"] if m.get("role") == "assistant" and m.get("tool_calls")
        ]
        trailing = resumed["messages"][-1]
        assert "Based on the search, a rope" in (
            trailing.get("content") or ""
        ), "round two's partial was not carried, so it restarts from the tool result"
        assert resumed.get("continue_final_message") is True

    def test_a_resume_is_not_charged_as_a_tool_iteration(self, monkeypatch):
        pauses = 6
        signal = preemption.PreemptSignal()
        recorder = _Recorder(
            monkeypatch,
            [[_delta(f"part {n} "), _finish(), _done()] for n in range(pauses + 1)],
            signal = signal,
            pause_attempts = range(pauses),
        )
        _run(
            recorder.backend,
            signal = signal,
            policy = _RecordingPolicy(),
            max_tool_iterations = 1,
        )
        assert (
            len(recorder.payloads) == pauses + 1
        ), "a paused turn was cut short by the tool-iteration bound"


class TestAPauseBeforeTheFirstToken:
    def test_it_re_issues_the_request_whole(self, monkeypatch):
        signal = preemption.PreemptSignal()
        policy = _RecordingPolicy()
        recorder = _Recorder(
            monkeypatch,
            [
                [_finish(), _done()],  # no content delta at all before the pause
                [_delta("a full answer."), _finish(), _done()],
            ],
            signal = signal,
        )

        def fake_iter(
            response,
            _cancel,
            first_token_deadline = None,
            preempt_event = None,
        ):
            if len(recorder.payloads) - 1 == 0:
                raise preemption.LlamaStreamPreempted
            yield from response.chunks

        monkeypatch.setattr(recorder.backend, "_iter_text_cancellable", fake_iter)
        _run(recorder.backend, signal = signal, policy = policy)

        assert len(recorder.payloads) == 2
        assert not recorder.payloads[1].get(
            "continue_final_message"
        ), "there was no partial, so nothing should be continued"
        assert policy.checkpoints[0].has_resume_point() is False


# ---------------------------------------------------------------- the final answering pass


def _final_pass(
    monkeypatch,
    streams,
    *,
    signal,
    policy,
    pause_attempts = (1,),
    **run_kwargs,
):
    recorder = _Recorder(
        monkeypatch,
        streams,
        signal = signal,
        pause_attempts = pause_attempts,
        execute_tool = True,
    )
    events = run_tool_loop(
        recorder.backend,
        signal = signal,
        policy = policy,
        tools = [_PLAIN_TOOL],
        prompt = "what kernel is current?",
        # One round, so the loop breaks mid-round into the synthesized final pass.
        max_tool_iterations = 1,
        permission_mode = "off",
        **run_kwargs,
    )
    return recorder, events


class _WatchingPolicy(_RecordingPolicy):
    """Records whether the signal was still set when it was made selectable again."""

    def __init__(
        self,
        signal,
        *,
        resume = True,
    ):
        super().__init__(resume = resume)
        self._signal = signal
        self.cleared_before_resume: list[bool] = []

    def on_resumed(self):
        self.cleared_before_resume.append(self._signal.is_set())
        super().on_resumed()


def _paused_final_run(monkeypatch, *, resume = True):
    signal = preemption.PreemptSignal()
    policy = _WatchingPolicy(signal, resume = resume)
    recorder, events = _final_pass(
        monkeypatch,
        [
            _tool_call_turn(),
            [_delta("The current kernel"), _finish(), _done()],
            [_delta(" is 6.10."), _finish(), _done()],
        ],
        signal = signal,
        policy = policy,
    )
    return recorder, policy, signal, events


class TestTheFinalPassPausesAndResumes:
    def test_the_client_is_told_it_is_paused_and_then_resumed(self, monkeypatch):
        _recorder, policy, signal, events = _paused_final_run(monkeypatch)
        assert {"type": "preempt", "state": "paused"} in events, (
            "the final pass swallowed the pause: the user watches a half-written answer "
            "stop dead while the cache waits on cells it will never get back"
        )
        assert events.index({"type": "preempt", "state": "paused"}) < events.index(
            {"type": "preempt", "state": "resumed"}
        )
        assert policy.events == ["preempted", "awaited", "resumed"]
        assert policy.checkpoints[0].visible_text == "The current kernel"
        assert not signal.is_set() and not signal.pending
        assert policy.cleared_before_resume == [False], (
            "the final pass cleared its signal after `on_resumed` had already made this "
            "participant selectable again"
        )

    def test_the_request_is_reopened_with_the_partial_and_the_answer_holds_it_once(
        self, monkeypatch
    ):
        recorder, _policy, _signal, events = _paused_final_run(monkeypatch)
        assert len(recorder.payloads) == 3, (
            "expected the tool round, the paused final pass and its resume; "
            f"got {len(recorder.payloads)}"
        )
        resumed = recorder.payloads[2]
        assert resumed.get("continue_final_message") is True
        assert resumed.get("add_generation_prompt") is False
        assert "The current kernel" in resumed["messages"][-1]["content"], (
            "the partial must go back as the turn to EXTEND, or the model answers from "
            "the top and the user reads the same sentence twice"
        )
        answer = _content(events)[-1]
        assert answer.count("The current kernel") == 1, answer
        assert "is 6.10." in answer, answer

    def test_a_policy_that_gives_up_ends_the_turn_and_says_so(self, monkeypatch):
        recorder, policy, signal, events = _paused_final_run(monkeypatch, resume = False)
        assert policy.events.count("preempted") == 1, "it paused more than once"
        assert not signal.is_set(), "the signal must be cleared before ending the turn"
        assert len(recorder.payloads) == 2, "the final pass was re-opened after a refusal"
        assert _gave_up(events), "the turn ended with no notice of why"
        assert _metadata(events)[-1]["finish_reason"] == "length", (
            "a turn holding a partial has to finish as continuable, which is what the "
            "client resumes from"
        )

    def test_the_allowance_admission_reserved_bounds_the_final_payload(self, monkeypatch):
        signal = preemption.PreemptSignal()
        recorder, _events = _final_pass(
            monkeypatch,
            [_tool_call_turn(), [_delta("w "), _delta("w "), _finish(), _done()]],
            signal = signal,
            policy = None,
            pause_attempts = (),
            max_tokens = 3000,
            admission_output_allowance = 512,
        )
        assert recorder.payloads[-1]["max_tokens"] == 512, (
            "the pass that produces the answer sent an output cap larger than the room "
            f"admission reserved for it: {recorder.payloads[-1]['max_tokens']}"
        )
        assert recorder.payloads[0]["max_tokens"] == 512, "the rounds were already clamped"


class TestTheFinalPassReportsItsGrowth:
    """The sweep is only as good as the thing feeding it."""

    _PER_ATTEMPT = _TOKEN_REPORT_EVERY + 8

    def _answer(self, letter):
        return [_delta(letter) for _ in range(self._PER_ATTEMPT)] + [_finish(), _done()]

    def _run(
        self,
        monkeypatch,
        *,
        on_tokens,
        pause_attempts = (1,),
        policy = None,
    ):
        signal = preemption.PreemptSignal()
        policy = policy if policy is not None else _DecliningPolicy()
        recorder = _Recorder(
            monkeypatch,
            [_tool_call_turn(), self._answer("a"), self._answer("b")],
            signal = signal,
            pause_attempts = pause_attempts,
            pause_after = self._PER_ATTEMPT,
            execute_tool = True,
        )
        events = list(
            recorder.backend.generate_chat_completion_with_tools(
                messages = [{"role": "user", "content": "what kernel is current?"}],
                tools = [_PLAIN_TOOL],
                cancel_event = threading.Event(),
                preempt_event = signal,
                preempt_policy = policy,
                max_tool_iterations = 1,
                permission_mode = "off",
                on_tokens = on_tokens,
            )
        )
        return recorder, policy, events

    def test_the_resumed_attempt_is_counted_from_zero(self, monkeypatch):
        reports: list[int] = []
        recorder, policy, _events = self._run(monkeypatch, on_tokens = reports.append)
        assert len(recorder.payloads) == 3
        assert policy.checkpoints, "the final pass never paused, so nothing here was tested"
        assert reports == [_TOKEN_REPORT_EVERY, _TOKEN_REPORT_EVERY], (
            "the count carried across the resume, so the sweep was told this chat had "
            "grown by both attempts while `note_replayed` had already added the first "
            f"one to its baseline: the same tokens twice. Reported {reports}"
        )

    def test_the_charge_is_the_observed_count_floored_by_the_estimate(self, monkeypatch):
        _recorder, policy, events = self._run(monkeypatch, on_tokens = lambda _n: None)
        checkpoint = policy.checkpoints[0]
        assert checkpoint.visible_text == "a" * self._PER_ATTEMPT
        # One character per chunk, so the four-characters-per-token estimate is a quarter
        # of the truth: the shape of the undercharge on token-dense text.
        assert checkpoint.charged_tokens == self._PER_ATTEMPT, (
            "the pause was charged the character estimate while the attempt's own chunk "
            f"count was known: {checkpoint.charged_tokens} against {self._PER_ATTEMPT}"
        )
        assert checkpoint.charged_tokens >= len(checkpoint.visible_text) // 4
        assert (_metadata(events)[-1].get("usage") or {}).get("completion_tokens", 0) >= (
            self._PER_ATTEMPT
        ), "the paused attempt's tokens were not reported to the caller"

    def test_the_resumed_attempt_does_not_get_a_fresh_output_cap(self, monkeypatch):
        signal = preemption.PreemptSignal()
        recorder = _Recorder(
            monkeypatch,
            [
                [_delta("Once upon a time there was a cat"), _finish(), _done()],
                [_delta(" who slept."), _finish(), _done()],
            ],
            signal = signal,
        )
        _run(recorder.backend, signal = signal, policy = _RecordingPolicy(), max_tokens = 100)
        opened, resumed = recorder.payloads[0], recorder.payloads[1]
        assert opened["max_tokens"] == 100
        assert resumed["max_tokens"] < 100, (
            "the resumed attempt was handed the whole cap again, so the turn may emit "
            f"more than the caller allowed; got {resumed['max_tokens']}"
        )
        assert resumed["max_tokens"] >= 1, "a request for zero tokens returns nothing at all"


class TestGivingUpTellsTheClient:
    def test_a_refused_resume_is_announced_once_though_the_loop_continues(self, monkeypatch):
        signal = preemption.PreemptSignal()
        policy = _RecordingPolicy(resume = False)
        recorder = _two_part(monkeypatch, signal)
        events = _run(recorder.backend, signal = signal, policy = policy)
        assert policy.events.count("preempted") == 1
        # One request, not two: the lease went back with `on_preempted` and this
        # participant is PAUSED, so breaking into the final answering pass would decode
        # on cells the planner had already handed to somebody else.
        assert len(recorder.payloads) == 1
        assert len(_gave_up(events)) == 1, "the tool loop gave up without telling anyone"
        assert _metadata(events)[-1]["finish_reason"] == "length"


class TestADeclinedPauseUnsticksTheParticipant:
    """A pause the stream refuses has to be handed back, not merely ignored."""

    @staticmethod
    def _capped(monkeypatch):
        monkeypatch.setattr(preemption, "DEFAULT_MAX_PREEMPT_RESUMES", 0)

    @staticmethod
    def _declining_run(monkeypatch, *, signal, policy, pause_attempts):
        recorder = PreemptRecorder(
            monkeypatch,
            [_tool_call_turn(), [_delta("The current kernel is 6.10."), _finish(), _done()]],
            signal = signal,
            pause_attempts = pause_attempts,
            request_pressure = False,
            execute_tool = True,
        )
        return recorder, run_tool_loop(
            recorder.backend,
            signal = signal,
            policy = policy,
            tools = [_PLAIN_TOOL],
            prompt = "what kernel is current?",
            max_tool_iterations = 1,
            permission_mode = "off",
        )

    def test_the_participant_decodes_again_and_keeps_its_lease_and_its_cells(self, monkeypatch):
        self._capped(monkeypatch)
        controller = PreemptionController("declined-round")
        controller.configure(budget = 16384, kv_unified = True)
        signal = preemption.PreemptSignal()

        class _Lease:
            preempted = 0

            def preempt(self):
                self.preempted += 1

        lease = _Lease()
        controller.register("other", tokens = 1000)
        participant = controller.register("chat", lease = lease, tokens = 1000, signal = signal)
        assert [v.gen_id for v in controller.plan_preemptions(needed = 16384)] == ["chat"]
        assert participant.consecutive_preemptions == 1
        policy = ControllerPreemptionPolicy(controller, "chat", signal, loop = None)
        _recorder, events = self._declining_run(
            monkeypatch, signal = signal, policy = policy, pause_attempts = (0,)
        )

        assert participant.state == ParticipantState.DECODING, (
            "the refusing chat decoded a whole final answer while PREEMPTING, which no "
            "later sweep can choose and no eviction can reclaim"
        )
        assert participant.preemptable
        assert not signal.is_set() and not signal.pending
        assert lease.preempted == 0, "the tokens were handed back under a stream still using them"
        assert participant.holds_kv and participant.tokens == 1000
        assert participant.consecutive_preemptions == 0, (
            "a pause that never happened must not promote this chat above chats that "
            "really did lose their work"
        )
        assert _content(events), "the turn still has to produce its answer"

    def test_the_final_pass_leaves_nothing_preempting(self, monkeypatch):
        self._capped(monkeypatch)
        controller = PreemptionController("declined-final")
        controller.configure(budget = 16384, kv_unified = True)
        signal = preemption.PreemptSignal()
        controller.register("other", tokens = 1000)
        participant = controller.register("chat", tokens = 1000, signal = signal)
        assert [v.gen_id for v in controller.plan_preemptions(needed = 16384)] == ["chat"]
        policy = ControllerPreemptionPolicy(controller, "chat", signal, loop = None)
        _recorder, events = self._declining_run(
            monkeypatch, signal = signal, policy = policy, pause_attempts = (1,)
        )
        assert (
            participant.state != ParticipantState.PREEMPTING
        ), "the turn ended with the ledger still holding a chosen victim"
        assert not signal.is_set()
        assert _gave_up(events), "the turn ended with no notice of why"
        assert _metadata(events)[-1]["finish_reason"] == "length"


class TestADeclinedContinuationTellsTheClient:
    """A continuation the backend declines must not be retried by the client."""

    def test_the_decline_says_the_retry_would_not_fit_and_still_ends_with_length(self, monkeypatch):
        from test_truncated_answer_continuation import (
            _cut_off_then,
            _done as _tc_done,
            _make_backend,
            _metadata as _tc_metadata,
            _run,
            _sse,
            _texts,
        )

        payloads: list[dict] = []
        backend = _make_backend(
            monkeypatch, _cut_off_then([_sse({"content": " never sent"}), _tc_done()]), payloads
        )
        monkeypatch.setattr(backend, "count_chat_tokens", lambda *_a, **_k: 4096)
        events = _run(backend)

        assert len(payloads) == 1, "the continuation was sent after all"
        # Once: `mergeContextTruncation` on the client SUMS the counters across a turn.
        refusals = [event for event in events if event.get("type") == "context_truncated"]
        assert len(refusals) == 1, refusals
        assert refusals[0]["fits"] is False
        # Nothing was evicted, and a non-zero count here raises "This conversation was
        # compacted" for a compaction that never happened.
        assert refusals[0]["dropped_messages"] == 0
        assert 0 < refusals[0]["prompt_target"] < refusals[0]["context_length"] == 4096
        assert _tc_metadata(events)["finish_reason"] == "length"
        assert "<!DOCTYPE html>" in "".join(_texts(events, "content"))
