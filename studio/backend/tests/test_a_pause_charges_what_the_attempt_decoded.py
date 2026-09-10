# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""What an aborted attempt produced has to be charged, on both surfaces."""

from __future__ import annotations

import contextlib
import copy
import json
import threading

from core.inference import llama_preemption as preemption
from core.inference.llama_cpp import LlamaCppBackend
import pytest

pytestmark = pytest.mark.usefixtures("preemption_opted_in")


_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "search",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
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


class _Recorder:
    """A backend whose stream pauses itself after a set number of content deltas."""

    def __init__(
        self,
        monkeypatch,
        streams,
        *,
        signal,
        pause_after = 1,
    ):
        self.payloads: list[dict] = []
        self.signal = signal
        self.pause_after = pause_after
        self._streams = [list(stream) for stream in streams]
        self.backend = LlamaCppBackend.__new__(LlamaCppBackend)
        backend = self.backend
        backend._process = object()
        backend._healthy = True
        backend._port = 48851
        backend._api_key = None
        backend._effective_context_length = 4096
        backend._supports_reasoning = False
        backend._reasoning_always_on = False
        backend._reasoning_style = "enable_thinking"
        backend._supports_preserve_thinking = False

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
            yield type(
                "FakeResponse", (), {"status_code": 200, "chunks": recorder._streams.pop(0)}
            )()

        def fake_iter_text_cancellable(
            response,
            _cancel_event,
            first_token_deadline = None,
            preempt_event = None,
        ):
            attempt = len(recorder.payloads) - 1
            seen = 0
            for chunk in response.chunks:
                yield chunk
                if not chunk.startswith("data: {"):
                    continue
                seen += 1
                if attempt == 0 and seen >= recorder.pause_after:
                    recorder.signal.request("kv_pressure")
                    raise preemption.LlamaStreamPreempted

        monkeypatch.setattr(backend, "_stream_with_retry", fake_stream_with_retry)
        monkeypatch.setattr(backend, "_iter_text_cancellable", fake_iter_text_cancellable)
        monkeypatch.setattr(backend, "_maybe_recover_from_mtp_crash", lambda *_a, **_k: False)


class _RecordingPolicy:
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


# Eight CJK tokens. chars // 4 calls this two.
_DENSE = "天地玄黄宇宙洪荒"


class TestTheToolLoopChargesItsPausedAttempt:
    """It charged zero, because it read only the usage a paused stream never sends."""

    def _run(self, backend, *, signal, policy, **kwargs):
        return list(
            backend.generate_chat_completion_with_tools(
                messages = [{"role": "user", "content": "write me a poem"}],
                tools = [_TOOL],
                cancel_event = threading.Event(),
                preempt_event = signal,
                preempt_policy = policy,
                **kwargs,
            )
        )

    def test_the_checkpoint_is_not_charged_zero(self, monkeypatch):
        """Zero skips `note_replayed` entirely -- it is gated on a non-zero charge -- so the
        controller never learns the resumed attempt carries the partial as prompt.
        """
        signal = preemption.PreemptSignal()
        policy = _RecordingPolicy()
        recorder = _Recorder(
            monkeypatch,
            [
                [_delta("Once upon a time there was a cat"), _finish(), _done()],
                [_delta(" who slept."), _finish(), _done()],
            ],
            signal = signal,
        )
        self._run(recorder.backend, signal = signal, policy = policy)

        assert policy.checkpoints[0].visible_text == "Once upon a time there was a cat"
        assert policy.checkpoints[0].charged_tokens > 0, (
            "an attempt that decoded 32 characters was charged nothing, so nothing "
            "re-baselined the ledger and nothing spent the caller's cap"
        )

    def test_the_resumed_attempt_does_not_get_a_fresh_output_cap(self, monkeypatch):
        """The next iteration rebuilds `max_tokens` from the caller's figure, so without an
        explicit continuation cap a request capped at 100 could emit 100 more after every pause.
        """
        signal = preemption.PreemptSignal()
        policy = _RecordingPolicy()
        recorder = _Recorder(
            monkeypatch,
            [
                [_delta("Once upon a time there was a cat"), _finish(), _done()],
                [_delta(" who slept."), _finish(), _done()],
            ],
            signal = signal,
        )
        self._run(recorder.backend, signal = signal, policy = policy, max_tokens = 100)

        opened, resumed = recorder.payloads[0], recorder.payloads[1]
        assert opened["max_tokens"] == 100
        assert resumed["max_tokens"] < 100, (
            "the resumed attempt was handed the whole cap again, so the turn may emit "
            f"more than the caller allowed; got {resumed['max_tokens']}"
        )
        assert resumed["max_tokens"] >= 1, "a request for zero tokens returns nothing at all"


class TestThePlainPathChargesDenseTextByTokens:
    """chars // 4 is an approximation, and on CJK it is wrong by several times."""

    def _run(self, backend, *, signal, policy, **kwargs):
        return list(
            backend.generate_chat_completion(
                messages = [{"role": "user", "content": "写一首诗"}],
                cancel_event = threading.Event(),
                preempt_event = signal,
                preempt_policy = policy,
                **kwargs,
            )
        )

    def test_eight_dense_tokens_are_not_charged_as_two(self, monkeypatch):
        signal = preemption.PreemptSignal()
        policy = _RecordingPolicy()
        # One delta per token, which is what llama-server streams.
        recorder = _Recorder(
            monkeypatch,
            [
                [*[_delta(ch) for ch in _DENSE], _finish(), _done()],
                [_delta("。"), _finish(), _done()],
            ],
            signal = signal,
            pause_after = len(_DENSE),
        )
        self._run(recorder.backend, signal = signal, policy = policy)

        charged = policy.checkpoints[0].charged_tokens
        assert charged >= len(_DENSE), (
            f"{len(_DENSE)} tokens were streamed and {charged} were charged; the "
            "difference is cells the watermark cannot see and output cap the caller "
            "never agreed to"
        )


class TestAGiveUpStillReportsWhatItDecoded:
    """The turn ends on `length` with text on screen, so its usage cannot be zero."""

    def _run(self, backend, *, signal, policy, **kwargs):
        return list(
            backend.generate_chat_completion(
                messages = [{"role": "user", "content": "write me a poem"}],
                cancel_event = threading.Event(),
                preempt_event = signal,
                preempt_policy = policy,
                **kwargs,
            )
        )

    def test_a_first_attempt_give_up_reports_its_completion_tokens(self, monkeypatch):
        signal = preemption.PreemptSignal()
        policy = _RecordingPolicy(resume = False)
        recorder = _Recorder(
            monkeypatch,
            [
                [
                    _delta("Once "),
                    _delta("upon "),
                    _delta("a "),
                    _delta("time"),
                    _finish(),
                    _done(),
                ],
            ],
            signal = signal,
            pause_after = 4,
        )
        items = self._run(recorder.backend, signal = signal, policy = policy)

        metadata = [
            item for item in items if isinstance(item, dict) and item.get("type") == "metadata"
        ]
        assert metadata, "a turn that gave up must still end on a terminal metadata event"
        assert metadata[-1]["finish_reason"] == "length"
        assert metadata[-1]["usage"].get("completion_tokens"), (
            "four deltas were streamed and shown; reporting zero completion tokens for "
            "them corrupts every usage-based client and monitor"
        )


def _opener() -> str:
    """llama-server's first frame: the role, no content."""
    return "data: " + json.dumps({"choices": [{"index": 0, "delta": {"role": "assistant"}}]}) + "\n"


class TestOnlyOutputCountsTowardsTheSweep:
    """The role opener and the finish frame add no cell; counting them let a sweep fire on
    the finish frame and pick the request that had just finished as its victim."""

    def _run(self, backend, *, signal, policy, on_tokens, **kwargs):
        return list(
            backend.generate_chat_completion(
                messages = [{"role": "user", "content": "hi"}],
                cancel_event = threading.Event(),
                preempt_event = signal,
                preempt_policy = policy,
                on_tokens = on_tokens,
                **kwargs,
            )
        )

    def test_the_opener_and_the_finish_frame_do_not_count(self, monkeypatch):
        from core.inference.llama_cpp import _TOKEN_REPORT_EVERY

        signal = preemption.PreemptSignal()
        reports: list[int] = []
        # Opener + (every - 1) content deltas + finish: every frames, every - 1 tokens.
        stream = [_opener()] + [_delta("x")] * (_TOKEN_REPORT_EVERY - 1) + [_finish(), _done()]
        recorder = _Recorder(monkeypatch, [stream], signal = signal, pause_after = 10**6)
        self._run(
            recorder.backend, signal = signal, policy = _RecordingPolicy(), on_tokens = reports.append
        )
        assert reports == [], f"a frame with no output counted: {reports}"

    def test_a_full_batch_of_output_still_reports(self, monkeypatch):
        from core.inference.llama_cpp import _TOKEN_REPORT_EVERY

        signal = preemption.PreemptSignal()
        reports: list[int] = []
        stream = [_opener()] + [_delta("x")] * _TOKEN_REPORT_EVERY + [_finish(), _done()]
        recorder = _Recorder(monkeypatch, [stream], signal = signal, pause_after = 10**6)
        self._run(
            recorder.backend, signal = signal, policy = _RecordingPolicy(), on_tokens = reports.append
        )
        assert reports == [_TOKEN_REPORT_EVERY]

    def test_the_tool_round_counts_only_output_too(self, monkeypatch):
        from core.inference.llama_cpp import _TOKEN_REPORT_EVERY

        def run(stream):
            signal = preemption.PreemptSignal()
            reports: list[int] = []
            recorder = _Recorder(monkeypatch, [stream], signal = signal, pause_after = 10**6)
            list(
                recorder.backend.generate_chat_completion_with_tools(
                    messages = [{"role": "user", "content": "hi"}],
                    tools = [_TOOL],
                    cancel_event = threading.Event(),
                    preempt_event = signal,
                    preempt_policy = _RecordingPolicy(),
                    on_tokens = reports.append,
                )
            )
            return reports

        short = [_opener()] + [_delta("x")] * (_TOKEN_REPORT_EVERY - 1) + [_finish(), _done()]
        assert run(short) == [], "the opener or the finish frame counted on the tool round"
        full = [_opener()] + [_delta("x")] * _TOKEN_REPORT_EVERY + [_finish(), _done()]
        assert run(full) == [_TOKEN_REPORT_EVERY]

    def test_every_reader_shares_the_output_predicate(self):
        import inspect

        from core.inference.llama_cpp import LlamaCppBackend as _B

        predicate = (
            'delta.get("content") or delta.get("reasoning_content") or delta.get("tool_calls")'
        )
        plain = " ".join(inspect.getsource(_B.generate_chat_completion).split())
        tools = " ".join(inspect.getsource(_B.generate_chat_completion_with_tools).split())
        assert plain.count(predicate) == 1
        # The tool round and the final pass.
        assert tools.count(predicate) == 2


def _timed_delta(content: str, *, prompt_n: int, predicted_n: int, predicted_ms: int) -> str:
    chunk = {
        "choices": [{"index": 0, "delta": {"content": content}}],
        "timings": {"prompt_n": prompt_n, "predicted_n": predicted_n, "predicted_ms": predicted_ms},
    }
    return "data: " + json.dumps(chunk) + "\n"


class TestARefusedResumeChargesTheAttemptOnce:
    """The interrupted attempt's decode goes into the accumulators, and the refused ending
    built its metadata from the same reading again: seven tokens reported as fourteen."""

    def test_the_tool_round_reports_the_attempt_once(self, monkeypatch):
        signal = preemption.PreemptSignal()
        policy = _RecordingPolicy(resume = False)
        stream = [
            _timed_delta("x", prompt_n = 100, predicted_n = n, predicted_ms = 10 * n) for n in range(1, 8)
        ] + [_finish(), _done()]
        recorder = _Recorder(monkeypatch, [stream], signal = signal, pause_after = 7)
        items = list(
            recorder.backend.generate_chat_completion_with_tools(
                messages = [{"role": "user", "content": "write me a poem"}],
                tools = [_TOOL],
                cancel_event = threading.Event(),
                preempt_event = signal,
                preempt_policy = policy,
            )
        )
        assert len(recorder.payloads) == 1
        metadata = [i for i in items if isinstance(i, dict) and i.get("type") == "metadata"][-1]
        assert metadata["finish_reason"] == "length"
        assert metadata["usage"]["completion_tokens"] == 7, metadata["usage"]
        assert metadata["usage"]["total_tokens"] == 107, metadata["usage"]
        assert metadata["timings"]["predicted_ms"] == 70, metadata["timings"]

    def test_the_final_pass_folds_before_its_refused_ending(self):
        import inspect

        from core.inference.llama_cpp import LlamaCppBackend as _B

        source = " ".join(inspect.getsource(_B.generate_chat_completion_with_tools).split())
        fold = source.index("_accumulated_completion_tokens += _pre_charged_f")
        assert "yield from _final_pause_gave_up(folded = True)" in source[fold:]
        # The cap-spent ending before the fold reads the whole attempt.
        assert "yield from _final_pause_gave_up()" in source[:fold]


class TestASpentCapIsNotReopened:
    """A pause landing after the caller's cap was spent used to reopen upstream for the
    one-token floor, once per pause; the partial is the answer."""

    def _run(self, backend, *, signal, policy, **kwargs):
        return list(
            backend.generate_chat_completion(
                messages = [{"role": "user", "content": "hi"}],
                cancel_event = threading.Event(),
                preempt_event = signal,
                preempt_policy = policy,
                **kwargs,
            )
        )

    def test_the_partial_ends_the_turn_with_length(self, monkeypatch):
        signal = preemption.PreemptSignal()
        policy = _RecordingPolicy()
        # Eight single-token deltas, paused after the eighth, against a cap of eight.
        recorder = _Recorder(
            monkeypatch,
            [[_delta("x")] * 8 + [_finish(), _done()], [_delta(" more"), _finish(), _done()]],
            signal = signal,
            pause_after = 8,
        )
        events = self._run(recorder.backend, signal = signal, policy = policy, max_tokens = 8)

        assert len(recorder.payloads) == 1, "the spent cap was reopened upstream"
        assert "awaited" not in policy.events, "a spent cap queued for room it cannot use"
        dicts = [e for e in events if isinstance(e, dict)]
        finishes = [e["finish_reason"] for e in dicts if e.get("type") == "metadata"]
        assert finishes[-1] == "length"
        # The caller's own cap ended it, which is not a give-up.
        assert not any(e.get("type") == "context_truncated" for e in dicts)
