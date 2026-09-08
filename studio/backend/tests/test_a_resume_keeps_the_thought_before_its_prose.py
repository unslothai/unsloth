# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A pause that lands after the thought AND after the first prose keeps both.

The visible-text resume replayed `content` alone, so a reasoning model's replacement request
was conditioned on a different prefix from the one that produced the answer it is extending:
the model reasons the turn out again, or contradicts the sentence it is finishing.
"""

from __future__ import annotations

import json

from core.inference import llama_preemption as preemption
from core.inference.llama_cpp import LlamaCppBackend
from core.inference.llama_preemption import StreamCheckpoint

from .test_llama_plain_chat_preempt_resume import (
    _Recorder,
    _RecordingPolicy,
    _done,
    _finish,
    _run,
)


def _thought_then_prose(thought: str, prose: str) -> str:
    """One delta carrying the thought and the first prose, as a reasoning model's stream
    reads by the time a pause lands after both."""
    return (
        "data: "
        + json.dumps(
            {"choices": [{"index": 0, "delta": {"reasoning_content": thought, "content": prose}}]}
        )
        + "\n"
    )


def _delta(content: str) -> str:
    return "data: " + json.dumps({"choices": [{"index": 0, "delta": {"content": content}}]}) + "\n"


class TestTheAssembler:
    def test_the_thought_rides_along_with_the_prose(self):
        convo = [{"role": "user", "content": "hi"}]
        out = LlamaCppBackend._assemble_preempt_resume(
            object(),
            convo,
            StreamCheckpoint(visible_text = "Therefore the answer", reasoning_text = "42 it is."),
            "Therefore the answer",
            "42 it is.",
        )
        assert out is True
        assert convo[-1]["content"] == "Therefore the answer"
        assert convo[-1]["reasoning_content"] == "42 it is.", (
            "the thought that produced the prose is the same turn's work; dropped, the "
            "continuation is prompted without it"
        )

    def test_a_thought_an_earlier_pause_left_is_merged_not_replaced(self):
        convo = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "", "reasoning_content": "first half "},
        ]
        LlamaCppBackend._assemble_preempt_resume(
            object(),
            convo,
            StreamCheckpoint(visible_text = "So: ", reasoning_text = "second half"),
            "So: ",
            "second half",
        )
        assert convo[-1]["reasoning_content"] == "first half second half"
        assert convo[-1]["content"] == "So: "
        assert len(convo) == 2

    def test_a_pause_with_no_thought_adds_no_empty_key(self):
        convo = [{"role": "user", "content": "hi"}]
        LlamaCppBackend._assemble_preempt_resume(
            object(),
            convo,
            StreamCheckpoint(visible_text = "Once upon"),
            "Once upon",
            "",
        )
        assert convo[-1]["content"] == "Once upon"
        assert "reasoning_content" not in convo[-1], "a non-reasoning model gains nothing"


class TestAPausedReasoningChat:
    def test_the_replacement_request_carries_the_thought_and_the_prose(self, monkeypatch):
        signal = preemption.PreemptSignal()
        policy = _RecordingPolicy()
        recorder = _Recorder(
            monkeypatch,
            [
                # One chunk: the recorder pauses on the first `data: {` of the attempt, and
                # what this is about is a pause with a thought AND prose behind it.
                [
                    _thought_then_prose(
                        "The secret intermediate result is 42.", "Therefore the answer"
                    ),
                    _finish(),
                    _done(),
                ],
                [_delta(" is forty-two."), _finish(), _done()],
            ],
            signal = signal,
            pause_attempts = (0,),
        )
        recorder.backend._supports_reasoning = True
        _run(recorder.backend, signal = signal, policy = policy)

        assert policy.checkpoints[0].has_resume_point(), "the pause landed after real prose"
        tail = recorder.payloads[1]["messages"][-1]
        assert tail["content"] == "Therefore the answer"
        assert (
            tail.get("reasoning_content") == "The secret intermediate result is 42."
        ), "the resumed attempt is conditioned on the prefix that produced the prose"
