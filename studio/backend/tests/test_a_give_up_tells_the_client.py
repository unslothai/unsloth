# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A chat that stops waiting for KV room says so. It used to end in silence.

Measured 2026-09-05, four chats on the 35B at -c 8192 with the GPU shared
(logs/studio_gpu0_swap_20260905_154407.log): one chat was evicted while still
prefilling, waited, and its resume was refused. The route then finished the turn with an
empty body and no error at all. The client recorded

    error: None, tokens: 0, chars: 0, wall_s: 163.8

and the GUI rendered a blank assistant turn with no notice and no Continue. A caller
cannot tell that from a model that chose to answer with nothing, which is the one outcome
a scheduler that pauses rather than fails must never produce: the whole reason for pausing
is that the user gets told what happened to their answer.

The notice rides on `context_truncated`, carrying `reason: "preempt_gave_up"`, `fits`
true and `dropped_messages` zero -- see `_preempt_gave_up_event` for why that event and
why those values. This file pins it on the three surfaces it has to reach: the plain chat
stream, the tool loop, and the durable `chat-runs` worker that relays `data:` lines to a
follower.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

from core.inference import llama_preemption as preemption  # noqa: E402
from core.inference.chat_generation_runs import ChatGenerationSupervisor  # noqa: E402
from core.inference.llama_cpp import (  # noqa: E402
    PREEMPT_GAVE_UP_REASON,
    _preempt_gave_up_event,
)
from routes import chat_generation_runs as run_routes  # noqa: E402
from routes import inference  # noqa: E402
from storage import chat_generation_runs_db as runs_db  # noqa: E402

from test_chat_generation_supervisor import durable_run  # noqa: E402, F401
from test_llama_plain_chat_preempt_resume import (  # noqa: E402
    _Recorder as _PlainRecorder,
)
from test_llama_plain_chat_preempt_resume import (  # noqa: E402
    _RecordingPolicy,
    _delta,
    _done,
    _finish,
)
from test_llama_plain_chat_preempt_resume import _run as _run_plain
from test_llama_tool_loop_preempt_resume import (  # noqa: E402
    _Recorder as _ToolRecorder,
)
from test_llama_tool_loop_preempt_resume import _run as _run_tools  # noqa: E402


def _nothing_yet() -> str:
    """A chunk that opens the stream and carries no text.

    The live shape: the chat was still prefilling when it was chosen, so it had produced
    nothing at all when the pause landed. `_Recorder` pauses on the first `data: {` line,
    and this is one that leaves `content_text` empty.
    """
    return "data: " + json.dumps({"choices": [{"index": 0, "delta": {}}]}) + "\n"


def _gave_up(chunks) -> list[dict]:
    return [
        chunk
        for chunk in chunks
        if isinstance(chunk, dict)
        and chunk.get("type") == "context_truncated"
        and chunk.get("reason") == PREEMPT_GAVE_UP_REASON
    ]


def _metadata(chunks) -> list[dict]:
    return [c for c in chunks if isinstance(c, dict) and c.get("type") == "metadata"]


class TestTheEventItself:
    def test_it_says_nothing_was_evicted_and_everything_fitted(self):
        """Both fields are load-bearing on the client and both are true here.

        A non-zero `dropped_messages` raises "This conversation was compacted" for a
        compaction that never happened, and `fits: false` sends the user to the Context
        Length setting for a prompt that was perfectly servable. The cache was busy, not
        small.
        """
        event = _preempt_gave_up_event(4096, 512)
        assert event["type"] == "context_truncated"
        assert event["reason"] == PREEMPT_GAVE_UP_REASON
        assert event["fits"] is True
        assert event["dropped_messages"] == 0
        assert event["context_length"] == 4096
        assert 0 < event["prompt_target"] < 4096

    def test_an_unknown_window_still_produces_a_notice(self):
        """The reason is the payload; the window is context for it. A backend that cannot
        report its context length must still not fall silent."""
        event = _preempt_gave_up_event(None, None)
        assert event["reason"] == PREEMPT_GAVE_UP_REASON
        assert "context_length" not in event


class TestThePlainChatPath:
    def test_a_refused_resume_before_the_first_token_is_not_an_empty_turn(self, monkeypatch):
        """The live failure exactly: nothing decoded, the resume refused, the turn over.

        What the client got was an empty 200. What it must get is a notice naming the
        cause and a terminal `length`, which is the shape it already knows how to resume.
        """
        signal = preemption.PreemptSignal()
        policy = _RecordingPolicy(resume = False)
        recorder = _PlainRecorder(
            monkeypatch,
            [[_nothing_yet(), _finish(), _done()]],
            signal = signal,
        )
        chunks = _run_plain(recorder.backend, signal = signal, policy = policy)

        assert policy.events == ["preempted", "awaited"], "the fixture did not give up"
        assert (
            "".join(c for c in chunks if isinstance(c, str)) == ""
        ), "this test is only about the empty case; it decoded something"
        notices = _gave_up(chunks)
        assert len(notices) == 1, f"expected exactly one notice, got {notices}"
        assert notices[0]["context_length"] == 4096

    def test_the_turn_ends_with_length_so_a_client_can_continue_it(self, monkeypatch):
        signal = preemption.PreemptSignal()
        policy = _RecordingPolicy(resume = False)
        recorder = _PlainRecorder(
            monkeypatch,
            [[_delta("Once upon a time"), _finish(), _done()]],
            signal = signal,
        )
        chunks = _run_plain(recorder.backend, signal = signal, policy = policy)

        assert len(recorder.payloads) == 1, "a refused resume must not re-open the request"
        # The partial is still the answer, unchanged by any of this.
        assert any("Once upon a time" in c for c in chunks if isinstance(c, str))
        assert len(_gave_up(chunks)) == 1
        metadata = _metadata(chunks)
        assert len(metadata) == 1, f"one terminal metadata, got {metadata}"
        assert (
            metadata[0]["finish_reason"] == "length"
        ), "an incomplete turn reported as anything else tells the client it is done"

    def test_the_notice_comes_before_the_end_of_the_turn(self, monkeypatch):
        """Order. A notice after the terminal metadata is a notice a client that stops
        reading at the finish reason never sees."""
        signal = preemption.PreemptSignal()
        policy = _RecordingPolicy(resume = False)
        recorder = _PlainRecorder(
            monkeypatch,
            [[_delta("Once upon a time"), _finish(), _done()]],
            signal = signal,
        )
        chunks = _run_plain(recorder.backend, signal = signal, policy = policy)
        kinds = [c.get("type") for c in chunks if isinstance(c, dict)]
        assert kinds.index("context_truncated") < kinds.index("metadata")
        # And the pause it resolves came first of all.
        assert kinds.index("preempt") < kinds.index("context_truncated")

    def test_a_resume_that_is_granted_says_nothing_of_the_kind(self, monkeypatch):
        """The notice must be a give-up signal, not a pause signal. A chat that paused
        and came back has nothing to apologise for and must not offer Continue."""
        signal = preemption.PreemptSignal()
        policy = _RecordingPolicy()
        recorder = _PlainRecorder(
            monkeypatch,
            [
                [_delta("Once upon a time"), _finish(), _done()],
                [_delta(" there was a cat."), _finish(), _done()],
            ],
            signal = signal,
        )
        chunks = _run_plain(recorder.backend, signal = signal, policy = policy)
        assert _gave_up(chunks) == []
        assert _metadata(chunks)[-1]["finish_reason"] != "length"


class TestTheToolLoopPath:
    def test_a_refused_resume_is_announced_there_too(self, monkeypatch):
        """Every GUI chat carries tools, so this is the surface most users are on.

        Giving up here breaks into the final answering pass rather than ending the
        response, so the turn usually still has text. The notice is owed all the same:
        the client was shown "Paused while another chat finishes" and nothing has
        resolved it.
        """
        signal = preemption.PreemptSignal()
        policy = _RecordingPolicy(resume = False)
        recorder = _ToolRecorder(
            monkeypatch,
            [
                [_delta("Once upon a time"), _finish(), _done()],
                [_delta(" there was a cat."), _finish(), _done()],
            ],
            signal = signal,
        )
        chunks = _run_tools(recorder.backend, signal = signal, policy = policy)

        assert policy.events.count("preempted") == 1
        # Unchanged: the turn is handed to the final pass rather than abandoned.
        assert len(recorder.payloads) == 2
        assert len(_gave_up(chunks)) == 1, "the tool loop gave up without telling anyone"

    def test_it_is_emitted_once_even_though_the_loop_continues(self, monkeypatch):
        """`context_truncated` is not idempotent on the client: `mergeContextTruncation`
        sums the counters across a turn, so a second copy is not a no-op."""
        signal = preemption.PreemptSignal()
        policy = _RecordingPolicy(resume = False)
        recorder = _ToolRecorder(
            monkeypatch,
            [
                [_delta("part one "), _finish(), _done()],
                [_delta("part two"), _finish(), _done()],
            ],
            signal = signal,
        )
        chunks = _run_tools(recorder.backend, signal = signal, policy = policy)
        assert len(_gave_up(chunks)) == 1

    def test_a_granted_resume_stays_quiet(self, monkeypatch):
        signal = preemption.PreemptSignal()
        policy = _RecordingPolicy()
        recorder = _ToolRecorder(
            monkeypatch,
            [
                [_delta("Once upon a time"), _finish(), _done()],
                [_delta(" there was a cat."), _finish(), _done()],
            ],
            signal = signal,
        )
        chunks = _run_tools(recorder.backend, signal = signal, policy = policy)
        assert _gave_up(chunks) == []


async def _follower_stream(after = 0) -> str:
    """Everything a late subscriber is sent for run-1, as raw SSE text."""
    response = await run_routes.chat_generation_events(
        "run-1",
        SimpleNamespace(is_disconnected = AsyncMock(return_value = True)),
        after = after,
        last_event_id = None,
        current_subject = "alice",
    )
    raw = ""
    async for part in response.body_iterator:
        raw += part.decode() if isinstance(part, bytes) else part
    return raw


class TestTheDurableRunPath:
    """The GUI streams plain chats through a durable run, and its worker reads the
    internal stream with `_SSEDecoder`, which keeps `data:` lines and drops comments.

    That is the reason this notice is a `data:` event rather than an SSE comment like
    `: preempt-paused`: the pause needed a special relay written for it before it reached
    a browser on this path at all. A `data:` line needs none.
    """

    @pytest.mark.asyncio
    async def test_the_notice_reaches_a_follower(self, durable_run, monkeypatch):  # noqa: F811
        notice = _preempt_gave_up_event(8192, None)
        wire = inference._context_truncated_sse_chunk(
            "chatcmpl-test",
            "local.gguf",
            {key: value for key, value in notice.items() if key != "type"},
        )

        async def body():
            yield 'data: {"choices": [{"delta": {"content": "half an "}}]}\n\n'
            yield wire
            yield 'data: {"choices": [{"delta": {}, "finish_reason": "length"}]}\n\n'
            yield "data: [DONE]\n\n"

        async def fake(_payload, _request, _subject, *, cancel_on_disconnect):
            return SimpleNamespace(status_code = 200, body_iterator = body())

        monkeypatch.setattr(inference, "produce_openai_chat_completions", fake)
        supervisor = ChatGenerationSupervisor(SimpleNamespace(state = SimpleNamespace()))
        await supervisor._produce("run-1")
        await asyncio.sleep(0)

        payloads = [
            event["payload"] for event in runs_db.list_events("run-1") if event["type"] == "chunk"
        ]
        carried = [p for p in payloads if isinstance(p.get("context_truncated"), dict)]
        assert len(carried) == 1, f"the notice was not persisted: {payloads}"
        assert carried[0]["context_truncated"]["reason"] == PREEMPT_GAVE_UP_REASON
        assert carried[0]["context_truncated"]["fits"] is True

        raw = await _follower_stream()
        assert PREEMPT_GAVE_UP_REASON in raw, (
            "persisted but never relayed: a follower reading this run sees the text stop "
            "and is told nothing, which is the browser-side shape of the same defect"
        )
        run = runs_db.get_run("run-1", "alice")
        assert run["status"] == "completed", "a give-up is not an error"
        assert run["finishReason"] == "length"
