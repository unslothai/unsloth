# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The last stream of a tool run pauses and resumes like the rounds before it.

A tool loop ends in a synthesized answering pass: the rounds break, the tool results are
in the conversation, and one more request writes the reply. That request is routinely the
longest decode of the whole turn, and it feeds the same shared KV cache as everything
else, so it is exactly the generation a sweep under pressure will choose.

It could not be paused. The rounds forwarded ``preempt_event`` into their stream and
handled ``LlamaStreamPreempted``; the final pass forwarded neither, so a chat chosen here
kept decoding while its participant stayed PREEMPTING. That state is outside
``_PREEMPTABLE``, so no later sweep could ask it again, and the cells the planner had
already counted as reclaimed were never released: the chats waiting on them waited for
room that was not coming.

These drive the real loop with fake llama-server streams. The first stream calls a tool,
the loop's one-round budget breaks it into the final pass, and that pass is what pauses.
"""

from __future__ import annotations

import ast
import contextlib
import copy
import json
import pathlib
import threading

from core.inference import llama_preemption as preemption
from core.inference.llama_cpp import LlamaCppBackend


LLAMA_CPP = pathlib.Path(__file__).resolve().parent.parent / "core" / "inference" / "llama_cpp.py"

_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "search",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
    },
}


def _delta(content: str) -> str:
    return "data: " + json.dumps({"choices": [{"index": 0, "delta": {"content": content}}]}) + "\n"


def _finish(reason: str = "stop") -> str:
    return (
        "data: "
        + json.dumps({"choices": [{"index": 0, "delta": {}, "finish_reason": reason}]})
        + "\n"
    )


def _done() -> str:
    return "data: [DONE]\n"


def _tool_call(call_id: str = "call_search") -> list[str]:
    return [
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
                                        "name": "web_search",
                                        "arguments": json.dumps({"query": "kernel"}),
                                    },
                                }
                            ]
                        },
                    }
                ]
            }
        )
        + "\n",
        _done(),
    ]


class _RecordingPolicy:
    """Stands in for the admission side. Records the handshake order."""

    def __init__(self, *, resume = True):
        self.events: list[str] = []
        self.checkpoints: list[preemption.StreamCheckpoint] = []
        self._resume = resume

    def should_preempt(self) -> bool:
        return False

    def on_preempted(self, checkpoint):
        self.events.append("preempted")
        self.checkpoints.append(checkpoint)

    def await_resume(self, timeout = None) -> bool:
        self.events.append("awaited")
        return self._resume

    def on_resumed(self) -> None:
        self.events.append("resumed")


class _Recorder:
    """A backend that pauses the stream of a chosen attempt partway through.

    Attempt 0 is the tool round; attempt 1 is the final answering pass, which is the one
    these tests are about.
    """

    def __init__(
        self,
        monkeypatch,
        streams,
        *,
        signal,
        pause_attempts = (1,),
    ):
        self.payloads: list[dict] = []
        self.signal = signal
        self.pause_attempts = set(pause_attempts)
        self._streams = [list(stream) for stream in streams]
        backend = LlamaCppBackend.__new__(LlamaCppBackend)
        backend._process = object()
        backend._healthy = True
        backend._port = 48847
        backend._api_key = None
        backend._effective_context_length = 4096
        backend._supports_reasoning = False
        backend._reasoning_always_on = False
        backend._reasoning_style = "enable_thinking"
        backend._supports_preserve_thinking = False
        self.backend = backend

        recorder = self

        @contextlib.contextmanager
        def fake_stream_with_retry(
            _client,
            _url,
            payload,
            _cancel_event,
            headers = None,
            first_token_deadline = None,
            preempt_event = None,
        ):
            recorder.payloads.append(copy.deepcopy(payload))
            recorder.opened_with_signal.append(preempt_event)
            stream = recorder._streams.pop(0)
            yield type("FakeResponse", (), {"status_code": 200, "chunks": stream})()

        def fake_iter_text_cancellable(
            response,
            _cancel_event,
            first_token_deadline = None,
            preempt_event = None,
        ):
            attempt = len(recorder.payloads) - 1
            recorder.read_with_signal.append(preempt_event)
            for chunk in response.chunks:
                yield chunk
                if attempt in recorder.pause_attempts and chunk.startswith("data: {"):
                    # Pressure noticed mid-stream, which is when it really is.
                    recorder.signal.request("kv_pressure")
                    raise preemption.LlamaStreamPreempted

        self.opened_with_signal: list[object] = []
        self.read_with_signal: list[object] = []
        monkeypatch.setattr(backend, "_stream_with_retry", fake_stream_with_retry)
        monkeypatch.setattr(backend, "_iter_text_cancellable", fake_iter_text_cancellable)
        monkeypatch.setattr(backend, "_maybe_recover_from_mtp_crash", lambda *_a, **_k: False)
        monkeypatch.setattr(
            "core.inference.tools.execute_tool",
            lambda name, arguments, **_kwargs: "Linux kernel 6.10.",
        )


def _run(recorder, *, signal, policy):
    return list(
        recorder.backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "what kernel is current?"}],
            tools = [_TOOL],
            cancel_event = threading.Event(),
            preempt_event = signal,
            preempt_policy = policy,
            # One round, so the loop breaks mid-round into the synthesized final pass.
            max_tool_iterations = 1,
            permission_mode = "off",
        )
    )


def _paused_final_run(monkeypatch, *, resume = True):
    signal = preemption.PreemptSignal()
    policy = _RecordingPolicy(resume = resume)
    recorder = _Recorder(
        monkeypatch,
        [
            _tool_call(),
            [_delta("The current kernel"), _finish(), _done()],
            [_delta(" is 6.10."), _finish(), _done()],
        ],
        signal = signal,
        pause_attempts = (1,),
    )
    events = _run(recorder, signal = signal, policy = policy)
    return recorder, policy, signal, events


class TestTheFinalPassPauses:
    def test_the_client_is_told_it_is_paused(self, monkeypatch):
        recorder, _policy, _signal, events = _paused_final_run(monkeypatch)
        assert {"type": "preempt", "state": "paused"} in events, (
            "the final pass swallowed the pause: the user watches a half-written answer "
            "stop dead while the cache waits on cells it will never get back"
        )
        assert {"type": "preempt", "state": "resumed"} in events
        assert events.index({"type": "preempt", "state": "paused"}) < events.index(
            {"type": "preempt", "state": "resumed"}
        )

    def test_the_policy_handshake_runs_in_order(self, monkeypatch):
        _recorder, policy, _signal, _events = _paused_final_run(monkeypatch)
        assert policy.events == ["preempted", "awaited", "resumed"]

    def test_the_checkpoint_carries_what_the_final_pass_had_written(self, monkeypatch):
        _recorder, policy, _signal, _events = _paused_final_run(monkeypatch)
        assert policy.checkpoints[0].visible_text == "The current kernel"
        assert policy.checkpoints[0].has_resume_point()
        assert policy.checkpoints[0].resumes == 1

    def test_the_signal_is_cleared_so_the_resume_can_run(self, monkeypatch):
        """Left set, the resumed attempt aborts on its first read and spins."""
        _recorder, _policy, signal, _events = _paused_final_run(monkeypatch)
        assert not signal.is_set()
        assert not signal.pending


class TestTheFinalPassResumes:
    def test_the_request_is_reopened_with_the_partial_to_extend(self, monkeypatch):
        recorder, _policy, _signal, _events = _paused_final_run(monkeypatch)
        assert len(recorder.payloads) == 3, (
            "expected the tool round, the paused final pass and its resume; "
            f"got {len(recorder.payloads)}"
        )
        resumed = recorder.payloads[2]
        assert resumed.get("continue_final_message") is True
        assert resumed.get("add_generation_prompt") is False
        trailing = resumed["messages"][-1]
        assert trailing["role"] == "assistant"
        assert "The current kernel" in trailing["content"], (
            "the partial must go back as the turn to EXTEND, or the model answers from "
            "the top and the user reads the same sentence twice"
        )

    def test_the_answer_holds_the_paused_text_exactly_once(self, monkeypatch):
        _recorder, _policy, _signal, events = _paused_final_run(monkeypatch)
        shown = [event["text"] for event in events if event.get("type") == "content"]
        assert shown, "the resumed final pass produced no answer at all"
        answer = shown[-1]
        assert answer.count("The current kernel") == 1, answer
        assert "is 6.10." in answer, answer

    def test_a_policy_that_gives_up_ends_the_turn_and_says_so(self, monkeypatch):
        """It must not wait forever, it must not pause again, and it must not fall silent.

        Nothing runs after this pass, so a bare return is a blank assistant turn that a
        caller cannot tell from a model that chose to say nothing.
        """
        recorder, policy, signal, events = _paused_final_run(monkeypatch, resume = False)
        assert policy.events.count("preempted") == 1, "it paused more than once"
        assert not signal.is_set(), "the signal must be cleared before ending the turn"
        assert len(recorder.payloads) == 2, "the final pass was re-opened after a refusal"
        assert any(
            event.get("reason") == "preempt_gave_up" for event in events
        ), "the turn ended with no notice of why"
        metadata = [event for event in events if event.get("type") == "metadata"]
        assert metadata and metadata[-1]["finish_reason"] == "length", (
            "a turn holding a partial has to finish as continuable, which is what the "
            "client resumes from"
        )


class TestTheWiringIsThere:
    """Structural, because the absence is what breaks: both calls behaved correctly on
    their own terms, and a pause simply never reached them."""

    @staticmethod
    def _final_pass_source() -> str:
        source = LLAMA_CPP.read_text(encoding = "utf-8")
        head = source.index(
            "with self._open_chat_stream_with_respawn_retry(\n                    stream_payload,"
        )
        return source[head : source.index("buffer += raw_chunk", head)]

    def test_the_final_stream_is_opened_on_the_signal(self):
        assert (
            '{"preempt_event": preempt_event}' in self._final_pass_source().split("as (")[0]
        ), "the final pass opens its stream without the signal it can be preempted by"

    def test_the_final_stream_is_read_on_the_signal(self):
        reader = self._final_pass_source()
        reader = reader[reader.index("_iter_text_cancellable(") :]
        assert '{"preempt_event": preempt_event}' in reader, (
            "the reader is what polls the signal and raises, so without it the pause is "
            "noticed only once the answer is already finished"
        )

    def test_both_pass_the_signal_conditionally(self):
        """A test double written against the old signature must keep working."""
        source = self._final_pass_source()
        assert (
            source.count('{"preempt_event": preempt_event}') == 2
        ), "expected the signal at both sites of the final pass, the stream open and the reader"
        for site in source.split('{"preempt_event": preempt_event}')[:-1]:
            assert site.rstrip().endswith(
                "**({} if preempt_event is None else"
            ), "the signal must be passed only when set, never as a bare keyword"

    def test_the_handler_exists_and_precedes_the_catch_all(self):
        source = LLAMA_CPP.read_text(encoding = "utf-8")
        handler = source.index(
            "except _preemption.LlamaStreamPreempted:",
            source.index(
                "with self._open_chat_stream_with_respawn_retry(\n                    stream_payload,"
            ),
        )
        catch_all = source.index('raise RuntimeError("Lost connection to llama-server")', handler)
        assert handler < catch_all
        block = source[handler:catch_all]
        for step in (
            "preempt_policy.on_preempted(",
            '{"type": "preempt", "state": "paused"}',
            "preempt_policy.await_resume()",
            "preempt_policy.on_resumed()",
            "self._assemble_preempt_resume(",
        ):
            assert step in block, f"the final pass's pause never {step}"
        assert block.index("preempt_event.clear()") < block.index(
            "preempt_policy.on_resumed()"
        ), "the clear must not run after the participant becomes selectable again"

    def test_the_module_still_parses(self):
        """The handler lives deep inside a very long generator; a stray indent there
        would be caught by nothing else in this file."""
        ast.parse(LLAMA_CPP.read_text(encoding = "utf-8"))
