# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""An evicted chat resumes from where it was, not from the last user message.

THE REQUIREMENT

A chat that had produced M tokens across T tool rounds when it was preempted must come
back with all of that intact. Restarting it from the user's message throws the model's
work away and repeats every tool's side effects, which is the "silly" outcome the goal
names explicitly. The one concession is a pause that lands INSIDE a tool call: backing up
to the start of that call is acceptable, because nothing has executed yet. Backing up
further is not.

WHAT THIS PINS, ONE SHAPE PER CLASS

  * Mid-reasoning. On a thinking model the opening of a turn is all reasoning and no
    prose, so this is the common case rather than the exotic one. The partial thought
    must go back as `reasoning_content`, so the model re-opens it rather than restarts.
  * After a completed tool round. The assistant's call, its result and the prose that
    followed are all in the conversation the resumed request carries. Nothing is
    re-executed: `execute_tool` runs exactly once for the round that ran.
  * Inside a tool call. The prose before the call is kept; the fragment is what gets
    replayed for the model to finish, and the tool does not run twice.

These are the tests that were missing. `test_llama_tool_loop_preempt_resume.py` covers the
handshake and the plain prose case; nothing covered the two shapes in which work is
actually expensive to lose.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

import pytest  # noqa: E402

from core.inference import llama_preemption as preemption  # noqa: E402

from test_llama_tool_loop_preempt_resume import (  # noqa: E402
    _Recorder,
    _RecordingPolicy,
    _delta,
    _done,
    _finish,
    _run,
)


def _reasoning(text: str) -> str:
    return (
        "data: "
        + json.dumps({"choices": [{"index": 0, "delta": {"reasoning_content": text}}]})
        + "\n"
    )


def _tool_call(call_id: str, name: str, arguments: dict) -> str:
    return (
        "data: "
        + json.dumps(
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": call_id,
                                    "type": "function",
                                    "function": {
                                        "name": name,
                                        "arguments": json.dumps(arguments),
                                    },
                                }
                            ]
                        },
                    }
                ]
            }
        )
        + "\n"
    )


class TestAPauseMidThoughtKeepsTheThought:
    def test_the_partial_reasoning_goes_back_as_reasoning(self, monkeypatch):
        """Preempted with a half-formed thought and no prose yet.

        Carried as `reasoning_content` the backend re-opens the thought. Carried as
        `content` it would be rendered as the answer; dropped, the model starts thinking
        again from nothing, which is the waste.
        """
        signal = preemption.PreemptSignal()
        policy = _RecordingPolicy()
        recorder = _Recorder(
            monkeypatch,
            [
                [_reasoning("Let me think about rope data structures"), _finish(), _done()],
                [_delta("A rope is a balanced tree of strings."), _finish(), _done()],
            ],
            signal = signal,
        )
        recorder.backend._supports_reasoning = True
        _run(recorder.backend, signal = signal, policy = policy)
        assert len(recorder.payloads) == 2
        resumed = recorder.payloads[1]
        trailing = resumed["messages"][-1]
        assert trailing["role"] == "assistant"
        assert "rope data structures" in (
            trailing.get("reasoning_content") or ""
        ), "the thought was not carried back, so the model starts thinking from nothing"
        assert "rope data structures" not in (
            trailing.get("content") or ""
        ), "the thought went back as the ANSWER"

    def test_the_checkpoint_itself_records_the_reasoning(self, monkeypatch):
        signal = preemption.PreemptSignal()
        policy = _RecordingPolicy()
        recorder = _Recorder(
            monkeypatch,
            [
                [_reasoning("thinking hard"), _finish(), _done()],
                [_delta("done"), _finish(), _done()],
            ],
            signal = signal,
        )
        recorder.backend._supports_reasoning = True
        _run(recorder.backend, signal = signal, policy = policy)
        assert policy.checkpoints, "no checkpoint was handed to the policy"
        assert policy.checkpoints[0].reasoning_text.strip() == "thinking hard"


class TestACompletedToolRoundIsNotRepeated:
    """M tokens and T tool calls in, the eviction lands. None of it may be redone."""

    def _executed(self, monkeypatch):
        calls: list = []

        def _execute(name, arguments, **_kwargs):
            calls.append((name, arguments))
            return f"RESULT<{arguments.get('query')}>"

        monkeypatch.setattr("core.inference.tools.execute_tool", _execute)
        return calls

    def test_the_tool_runs_once_and_its_result_travels_with_the_resume(self, monkeypatch):
        """Round one calls a tool and completes. Round two is preempted mid-prose.

        The resumed request must carry round one whole -- the assistant's call, the tool
        result -- and round two's partial as the turn to extend. `execute_tool` must have
        run exactly once, for round one, and not again on the resume.
        """
        calls = self._executed(monkeypatch)
        signal = preemption.PreemptSignal()
        policy = _RecordingPolicy()
        recorder = _Recorder(
            monkeypatch,
            [
                # round 1: the model asks for a tool
                [
                    _tool_call("call_1", "web_search", {"query": "ropes"}),
                    _finish("tool_calls"),
                    _done(),
                ],
                # round 2: prose, preempted partway (attempt index 1)
                [_delta("Based on the search, a rope"), _finish(), _done()],
                # round 2 resumed
                [_delta(" is a balanced tree."), _finish(), _done()],
            ],
            signal = signal,
            pause_after_attempt = 1,
        )
        _run(recorder.backend, signal = signal, policy = policy)

        assert calls == [
            ("web_search", {"query": "ropes"})
        ], f"the tool ran {len(calls)} times: the resume re-executed a finished round"
        assert len(recorder.payloads) == 3
        resumed = recorder.payloads[2]
        roles = [m.get("role") for m in resumed["messages"]]
        assert "tool" in roles, "round one's result was dropped from the resumed history"
        tool_row = next(m for m in resumed["messages"] if m.get("role") == "tool")
        assert tool_row.get("content") == "RESULT<ropes>"
        assistant_calls = [
            m for m in resumed["messages"] if m.get("role") == "assistant" and m.get("tool_calls")
        ]
        assert assistant_calls, "round one's call was dropped from the resumed history"
        trailing = resumed["messages"][-1]
        assert trailing["role"] == "assistant"
        assert "Based on the search, a rope" in (
            trailing.get("content") or ""
        ), "round two's partial was not carried, so it restarts from the tool result"
        assert resumed.get("continue_final_message") is True

    def test_nothing_before_the_pause_is_lost_when_it_lands_on_a_later_round(self, monkeypatch):
        """Two completed rounds, then a pause. Both results must survive, in order."""
        calls = self._executed(monkeypatch)
        signal = preemption.PreemptSignal()
        policy = _RecordingPolicy()
        recorder = _Recorder(
            monkeypatch,
            [
                [
                    _tool_call("call_1", "web_search", {"query": "one"}),
                    _finish("tool_calls"),
                    _done(),
                ],
                [
                    _tool_call("call_2", "web_search", {"query": "two"}),
                    _finish("tool_calls"),
                    _done(),
                ],
                [_delta("Two searches later,"), _finish(), _done()],
                [_delta(" here is the answer."), _finish(), _done()],
            ],
            signal = signal,
            pause_after_attempt = 2,
        )
        _run(recorder.backend, signal = signal, policy = policy)
        assert [a.get("query") for _n, a in calls] == ["one", "two"]
        resumed = recorder.payloads[3]
        tool_rows = [m for m in resumed["messages"] if m.get("role") == "tool"]
        assert [row.get("content") for row in tool_rows] == ["RESULT<one>", "RESULT<two>"]


class TestAPauseInsideAToolCallBacksUpToTheCall:
    """The one place backing up is allowed, and it must not back up further than that."""

    def test_the_prose_before_the_call_survives_and_the_tool_does_not_run_twice(self, monkeypatch):
        """Text-form call, preempted after the markup has started but before it closes.

        What is replayed is the visible prose plus whatever fragment had streamed; the
        model finishes the call on the resume, and the tool then runs ONCE. Losing the
        prose would be backing up to the user message, which is the outcome ruled out.
        """
        calls: list = []

        def _execute(name, arguments, **_kwargs):
            calls.append((name, arguments))
            return "RESULT"

        monkeypatch.setattr("core.inference.tools.execute_tool", _execute)
        signal = preemption.PreemptSignal()
        policy = _RecordingPolicy()
        recorder = _Recorder(
            monkeypatch,
            [
                # prose, then the opening of a text-form call; the pause lands here
                [
                    _delta("First, some context about ropes. "),
                    _delta('<tool_call>{"name": "web_search", "arguments": {"query": '),
                    _finish(),
                    _done(),
                ],
                # resume: the model completes the call
                [
                    _delta('"ropes"}}</tool_call>'),
                    _finish("tool_calls"),
                    _done(),
                ],
                # after the tool: the answer
                [_delta("Ropes are trees."), _finish(), _done()],
            ],
            signal = signal,
            pause_after_attempt = 0,
        )
        _run(recorder.backend, signal = signal, policy = policy)

        resumed = recorder.payloads[1]
        trailing = resumed["messages"][-1]
        assert trailing["role"] == "assistant"
        assert "First, some context about ropes." in (trailing.get("content") or ""), (
            "the prose before the call was dropped: that is backing up to the user "
            "message, not to the call"
        )
        assert len(calls) <= 1, f"the tool ran {len(calls)} times across the pause"
