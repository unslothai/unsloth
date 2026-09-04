# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A pause has to be tellable from a Stop, at the exact place the stream dies.

Both close the upstream read the same way, through the same watcher and the same
socket shutdown. What separates them is only which exception comes out, and
everything downstream depends on getting that right: ``_LlamaStreamCancelled``
ends the turn and writes ``status="cancelled"`` on the durable run, while a pause
is expected to be caught and resumed. Confusing them either abandons a chat that
was merely waiting or silently resumes one the user stopped.
"""

import threading

import pytest

from core.inference import llama_preemption as preemption
from core.inference.llama_cpp import LlamaCppBackend, _interrupt_event


class _FakeResponse:
    """Enough of httpx.Response for _iter_text_cancellable."""

    def __init__(
        self,
        chunks,
        on_close = None,
    ):
        self._chunks = list(chunks)
        self.closed = False
        self._on_close = on_close
        self.request = None

    def iter_text(self):
        for chunk in self._chunks:
            yield chunk

    def close(self):
        self.closed = True
        if self._on_close is not None:
            self._on_close()


def _drain(
    response,
    *,
    cancel_event = None,
    preempt_event = None,
    deadline_s = 30.0,
):
    import time
    return list(
        LlamaCppBackend._iter_text_cancellable(
            response,
            cancel_event,
            first_token_deadline = time.monotonic() + deadline_s,
            preempt_event = preempt_event,
        )
    )


class TestWhichExceptionComesOut:
    def test_a_pause_raises_preempted(self):
        pause = preemption.PreemptSignal()
        pause.request("kv_pressure")
        response = _FakeResponse(["data: a\n", "data: b\n"])
        with pytest.raises(preemption.LlamaStreamPreempted):
            _drain(response, cancel_event = threading.Event(), preempt_event = pause)
        assert response.closed, "the upstream response must be closed on a pause"

    def test_a_stop_still_ends_quietly(self):
        """Unchanged behaviour: a cancel stops the iteration without raising."""
        cancel = threading.Event()
        cancel.set()
        response = _FakeResponse(["data: a\n"])
        assert _drain(response, cancel_event = cancel) == []
        assert response.closed

    def test_a_stop_during_a_pause_is_still_a_stop(self):
        """A user who pressed Stop while a chat was paused meant Stop. Resuming
        it would restart a turn they abandoned."""
        cancel = threading.Event()
        cancel.set()
        pause = preemption.PreemptSignal()
        pause.request()
        response = _FakeResponse(["data: a\n"])
        # No exception: the cancel path wins and returns.
        assert _drain(response, cancel_event = cancel, preempt_event = pause) == []
        assert response.closed

    def test_a_deferred_pause_does_not_stop_the_stream(self):
        """Tool execution is running; the stream must be left alone."""
        pause = preemption.PreemptSignal()
        response = _FakeResponse(["data: a\n", "data: b\n"])
        with pause.unsafe_window():
            pause.request()
            chunks = _drain(response, cancel_event = threading.Event(), preempt_event = pause)
        assert chunks == ["data: a\n", "data: b\n"]

    def test_no_signal_at_all_is_the_old_path(self):
        response = _FakeResponse(["data: a\n", "data: b\n"])
        assert _drain(response, cancel_event = threading.Event()) == ["data: a\n", "data: b\n"]


class TestThePauseIsSeenBetweenChunks:
    def test_a_pause_raised_mid_stream_stops_the_rest(self):
        """The realistic shape: pressure is noticed while tokens are flowing."""
        pause = preemption.PreemptSignal()
        seen = []

        class _Streaming(_FakeResponse):
            def iter_text(self):
                for index, chunk in enumerate(self._chunks):
                    seen.append(chunk)
                    if index == 1:
                        pause.request("kv_pressure")
                    yield chunk

        response = _Streaming(["a", "b", "c", "d"])
        with pytest.raises(preemption.LlamaStreamPreempted):
            _drain(response, cancel_event = threading.Event(), preempt_event = pause)
        assert "d" not in seen, "the stream kept reading after the pause"


class TestTheCombinedWaitable:
    def test_it_is_the_existing_helper_not_a_new_one(self):
        """Reusing `_CombinedCancelEvent` is why a pause needs no new teardown:
        the watcher thread and the socket shutdown are untouched."""
        from core.inference.llama_cpp import _CombinedCancelEvent

        combined = _interrupt_event(threading.Event(), preemption.PreemptSignal())
        assert isinstance(combined, _CombinedCancelEvent)


class TestTheSignaturesStayBackwardsCompatible:
    """Overrides and test doubles written against the old signatures still work,
    because the new argument is only passed when a pause signal exists."""

    def test_every_funnel_defaults_the_new_argument(self):
        import inspect
        funnels = (
            LlamaCppBackend._iter_text_cancellable,
            LlamaCppBackend._install_cancel_aware_read,
            LlamaCppBackend._stream_with_retry,
            LlamaCppBackend._open_stream,
            LlamaCppBackend._open_chat_stream_with_respawn_retry,
        )
        for funnel in funnels:
            params = inspect.signature(funnel).parameters
            assert "preempt_event" in params, funnel.__name__
            assert params["preempt_event"].default is None, funnel.__name__

    def test_the_tool_loop_accepts_a_policy_and_a_signal(self):
        import inspect
        params = inspect.signature(LlamaCppBackend.generate_chat_completion_with_tools).parameters
        for name in ("preempt_event", "preempt_policy"):
            assert name in params
            assert params[name].default is None

    def test_cancel_only_callers_pass_no_new_argument(self):
        """The guard against the failure this change actually hit: a monkeypatched
        `_stream_with_retry` with the old signature must never see the kwarg."""
        import pathlib
        import re

        source = pathlib.Path(LlamaCppBackend.__module__.replace(".", "/") + ".py")
        text = source.read_text() if source.exists() else ""
        if not text:
            import core.inference.llama_cpp as module
            text = pathlib.Path(module.__file__).read_text()
        # Every forward of the argument is conditional on it existing.
        unconditional = re.findall(r"^\s*preempt_event = preempt_event,\s*$", text, re.M)
        assert (
            not unconditional
        ), "a preempt_event forwarded unconditionally breaks old-signature doubles"
