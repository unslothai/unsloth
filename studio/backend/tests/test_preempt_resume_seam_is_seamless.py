# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The text a client sees across a pause is exactly the text the model produced.

WHAT WENT WRONG

Measured on Qwen3.5-4B at ``-c 8192`` with four chats, seed 1234, temperature 0. The solo
reference for one prompt read "Introduction: The Paradigm Shifts"; the same chat, paused
once at 107 characters and resumed, read "Introduction: Theigm Shifts". The pause landed
after "The"; the resumed attempt's first token was " Parad" and the client never got it.

llama-server was not the cause. Aborting a stream and continuing from the characters the
client saw, twelve seams with the prompt cache on and off, gave the reference's next token
every time (`scripts/seam_repro.py`). The token was lost on this side.

Every route consumer diffs cumulative snapshots: ``new = cumulative[len(prev):]`` then
``prev = cumulative``. That is the generator's contract, and a resumed attempt broke it.
It restarted its accumulator at "" so its first snapshot was SHORTER than the last one the
consumer had seen: the diff came out empty, ``prev`` was overwritten with the short
snapshot, and the next diff began after the token that had just been dropped. One token
per resume on the plain path; on the tool loop, whose first emission is a whole buffered
prefix, potentially far more.

THE RULE THESE TESTS PIN

A resumed attempt's snapshots continue the paused attempt's: every string the generator
yields starts with the one before it, across the pause. The assertions replay the routes'
own diff so the check is against what a client would have assembled, not against an
internal accumulator that a refactor could keep monotonic while the yields drift.
"""

from __future__ import annotations

import json
import os
import sys

from core.inference import llama_preemption as preemption

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

from test_llama_plain_chat_preempt_resume import (  # noqa: E402
    _Recorder as _PlainRecorder,
    _RecordingPolicy,
    _delta,
    _done,
    _finish,
    _run as _run_plain,
)
from test_llama_tool_loop_preempt_resume import (  # noqa: E402
    _Recorder as _ToolRecorder,
    _run as _run_tools,
)


def _reasoning(text: str) -> str:
    return (
        "data: "
        + json.dumps({"choices": [{"index": 0, "delta": {"reasoning_content": text}}]})
        + "\n"
    )


def _as_a_route_would(snapshots: list[str]) -> str:
    """The diff every consumer in routes/inference.py applies, verbatim."""
    prev = ""
    out = ""
    for cumulative in snapshots:
        new = cumulative[len(prev) :]
        prev = cumulative
        out += new
    return out


def _plain_snapshots(events) -> list[str]:
    return [event for event in events if isinstance(event, str)]


def _tool_snapshots(events) -> list[str]:
    return [
        event["text"]
        for event in events
        if isinstance(event, dict) and event.get("type") == "content"
    ]


def _assert_monotonic(snapshots: list[str]) -> None:
    for earlier, later in zip(snapshots, snapshots[1:]):
        assert later.startswith(earlier), (
            f"a snapshot went backwards across the pause: {earlier!r} then {later!r}. "
            f"Any consumer diffing these drops text."
        )


class TestThePlainPathSeam:
    def test_the_first_resumed_token_reaches_the_client(self, monkeypatch):
        """The measured failure, byte for byte: "The" paused, " Parad" "igm" resumed."""
        signal = preemption.PreemptSignal()
        policy = _RecordingPolicy()
        recorder = _PlainRecorder(
            monkeypatch,
            [
                [_delta("Introduction: The"), _delta(" Genesis"), _finish(), _done()],
                [_delta(" Parad"), _delta("igm"), _delta(" Shifts"), _finish(), _done()],
            ],
            signal = signal,
        )
        events = _run_plain(recorder.backend, signal = signal, policy = policy)
        snapshots = _plain_snapshots(events)
        assert _as_a_route_would(snapshots) == "Introduction: The Paradigm Shifts"
        _assert_monotonic(snapshots)

    def test_two_pauses_keep_both_prefixes(self, monkeypatch):
        signal = preemption.PreemptSignal()
        policy = _RecordingPolicy()
        recorder = _PlainRecorder(
            monkeypatch,
            [
                [_delta("one"), _finish(), _done()],
                [_delta(" two"), _finish(), _done()],
                [_delta(" three"), _delta(" four"), _finish(), _done()],
            ],
            signal = signal,
            pause_attempts = (0, 1),
        )
        events = _run_plain(recorder.backend, signal = signal, policy = policy)
        snapshots = _plain_snapshots(events)
        assert _as_a_route_would(snapshots) == "one two three four"
        _assert_monotonic(snapshots)
        # And the replay still carries only what was new, never the prefix twice.
        assert recorder.payloads[2]["messages"][-1]["content"] == "one two"

    def test_a_thought_interrupted_mid_way_stays_one_thought(self, monkeypatch):
        """Paused inside <think>, the resumed reasoning must not open a second block."""
        signal = preemption.PreemptSignal()
        policy = _RecordingPolicy()
        recorder = _PlainRecorder(
            monkeypatch,
            [
                [_reasoning("Let me"), _reasoning(" think"), _finish(), _done()],
                [_reasoning(" harder."), _delta("Answer."), _finish(), _done()],
            ],
            signal = signal,
        )
        recorder.backend._supports_reasoning = True
        events = _run_plain(recorder.backend, signal = signal, policy = policy)
        snapshots = _plain_snapshots(events)
        _assert_monotonic(snapshots)
        assert _as_a_route_would(snapshots) == "<think>Let me harder.</think>Answer."

    def test_a_thought_interrupted_then_answered_closes_once(self, monkeypatch):
        """Paused inside <think>, resumed straight into prose: one close, no reopen."""
        signal = preemption.PreemptSignal()
        policy = _RecordingPolicy()
        recorder = _PlainRecorder(
            monkeypatch,
            [
                [_reasoning("Let me"), _finish(), _done()],
                [_delta("Answer."), _delta(" Done."), _finish(), _done()],
            ],
            signal = signal,
        )
        recorder.backend._supports_reasoning = True
        events = _run_plain(recorder.backend, signal = signal, policy = policy)
        snapshots = _plain_snapshots(events)
        _assert_monotonic(snapshots)
        assert _as_a_route_would(snapshots) == "<think>Let me</think>Answer. Done."

    def test_a_pause_mid_thought_resumes_the_thought_not_the_answer(self, monkeypatch):
        """The replay must go back as reasoning_content, or the model reads its own
        half-thought as the start of its answer and the client renders it as such."""
        signal = preemption.PreemptSignal()
        policy = _RecordingPolicy()
        recorder = _PlainRecorder(
            monkeypatch,
            [
                [_reasoning("Let me"), _finish(), _done()],
                [_reasoning(" think."), _delta("Answer."), _finish(), _done()],
            ],
            signal = signal,
        )
        recorder.backend._supports_reasoning = True
        _run_plain(recorder.backend, signal = signal, policy = policy)
        resumed = recorder.payloads[1]["messages"][-1]
        assert resumed["role"] == "assistant"
        assert "<think>" not in (
            resumed.get("content") or ""
        ), "the open thought was replayed as visible content with a literal tag"
        assert resumed.get("reasoning_content") == "Let me"
        assert recorder.payloads[1].get("continue_final_message") is True


class TestTheToolLoopSeam:
    def test_the_first_resumed_emission_reaches_the_client(self, monkeypatch):
        signal = preemption.PreemptSignal()
        policy = _RecordingPolicy()
        recorder = _ToolRecorder(
            monkeypatch,
            [
                [_delta("Introduction: The"), _delta(" Genesis"), _finish(), _done()],
                [_delta(" Parad"), _delta("igm"), _delta(" Shifts"), _finish(), _done()],
            ],
            signal = signal,
        )
        events = _run_tools(recorder.backend, signal = signal, policy = policy)
        snapshots = _tool_snapshots(events)
        assert _as_a_route_would(snapshots) == "Introduction: The Paradigm Shifts"
        _assert_monotonic(snapshots)

    def test_two_pauses_keep_both_prefixes(self, monkeypatch):
        signal = preemption.PreemptSignal()
        policy = _RecordingPolicy()
        recorder = _ToolRecorder(
            monkeypatch,
            [
                [_delta("one"), _finish(), _done()],
                [_delta(" two"), _finish(), _done()],
                [_delta(" three"), _delta(" four"), _finish(), _done()],
            ],
            signal = signal,
            pause_attempts = (0, 1),
        )
        events = _run_tools(recorder.backend, signal = signal, policy = policy)
        snapshots = _tool_snapshots(events)
        assert _as_a_route_would(snapshots) == "one two three four"
        _assert_monotonic(snapshots)

    def test_a_thought_interrupted_mid_way_stays_one_thought(self, monkeypatch):
        signal = preemption.PreemptSignal()
        policy = _RecordingPolicy()
        recorder = _ToolRecorder(
            monkeypatch,
            [
                [_reasoning("Let me"), _reasoning(" think"), _finish(), _done()],
                [_reasoning(" harder."), _delta("Answer."), _finish(), _done()],
            ],
            signal = signal,
        )
        recorder.backend._supports_reasoning = True
        events = _run_tools(recorder.backend, signal = signal, policy = policy)
        snapshots = _tool_snapshots(events)
        _assert_monotonic(snapshots)
        assembled = _as_a_route_would(snapshots)
        assert assembled.count("<think>") == 1, assembled
        assert assembled.endswith("</think>Answer."), assembled
