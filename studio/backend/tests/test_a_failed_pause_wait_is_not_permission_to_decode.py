# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A wait that RAISED never granted anything.

`on_preempted` has already handed the lease back, so treating the exception as a resume let
the next request decode on room nobody booked. The turn ends with its partial instead.

The same handshake carries Stop: a policy that cannot be told about the cancel event sits in
the wait for a chat nobody is reading.
"""

from __future__ import annotations

import inspect
import threading

import pytest

from core.inference import llama_cpp
from core.inference import llama_preemption as preemption
from core.inference.llama_preemption import (
    DeferredPreemptionPolicy,
    NullPreemptionPolicy,
    StreamCheckpoint,
)

from .test_llama_tool_loop_preempt_resume import (
    _Recorder,
    _RecordingPolicy,
    _delta,
    _done,
    _finish,
    _run,
)

pytestmark = pytest.mark.usefixtures("preemption_opted_in")


class _WaitRaises(_RecordingPolicy):
    """Released the lease, then failed in the wait."""

    def await_resume(
        self,
        timeout = None,
        *,
        cancel_event = None,
    ) -> bool:
        self.events.append("awaited")
        raise RuntimeError("the resume wait failed")


class TestTheTurnEndsRatherThanDecoding:
    def test_a_raising_wait_opens_no_second_request(self, monkeypatch):
        signal = preemption.PreemptSignal()
        policy = _WaitRaises()
        recorder = _Recorder(
            monkeypatch,
            [
                [_delta("Once upon a time"), _finish(), _done()],
                [_delta(" there was a cat."), _finish(), _done()],
            ],
            signal = signal,
        )
        _run(recorder.backend, signal = signal, policy = policy)

        assert len(recorder.payloads) == 1, (
            "the wait failed, so nothing granted this chat room: a replacement request "
            "decodes on cells the planner has handed to somebody else"
        )
        assert policy.events == ["preempted", "awaited"], "a failed wait is not a resume"

    def test_the_client_is_told_it_gave_up(self, monkeypatch):
        signal = preemption.PreemptSignal()
        recorder = _Recorder(
            monkeypatch,
            [
                [_delta("Once upon a time"), _finish(), _done()],
                [_delta(" there was a cat."), _finish(), _done()],
            ],
            signal = signal,
        )
        events = _run(recorder.backend, signal = signal, policy = _WaitRaises())

        reasons = [event.get("reason") for event in events if isinstance(event, dict)]
        assert (
            llama_cpp.PREEMPT_GAVE_UP_REASON in reasons
        ), "silence is the one outcome a shared-cache scheduler must never produce"
        assert any(
            isinstance(event, dict) and event.get("finish_reason") == "length" for event in events
        ), "the partial is continuable, and `length` is what says so"

    def test_the_partial_it_streamed_is_still_the_answer(self, monkeypatch):
        signal = preemption.PreemptSignal()
        recorder = _Recorder(
            monkeypatch,
            [
                [_delta("Once upon a time"), _finish(), _done()],
                [_delta(" there was a cat."), _finish(), _done()],
            ],
            signal = signal,
        )
        events = _run(recorder.backend, signal = signal, policy = _WaitRaises())

        text = "".join(
            event.get("text", "")
            for event in events
            if isinstance(event, dict) and event.get("type") == "content"
        )
        assert "Once upon a time" in text


class TestStopReachesTheWait:
    def test_every_policy_takes_the_cancel_keyword(self):
        for policy in (DeferredPreemptionPolicy(), NullPreemptionPolicy()):
            parameters = inspect.signature(policy.await_resume).parameters
            assert "cancel_event" in parameters, type(policy).__name__
            assert parameters["cancel_event"].kind is inspect.Parameter.KEYWORD_ONLY, type(
                policy
            ).__name__

    def test_the_protocol_declares_it_too(self):
        parameters = inspect.signature(preemption.PreemptionPolicy.await_resume).parameters
        assert "cancel_event" in parameters

    def test_the_wrapper_forwards_it(self):
        seen: list = []

        class _Inner:
            def await_resume(
                self,
                timeout = None,
                *,
                cancel_event = None,
            ) -> bool:
                seen.append(cancel_event)
                return True

        event = threading.Event()
        assert DeferredPreemptionPolicy(_Inner()).await_resume(1.0, cancel_event = event) is True
        assert seen == [event]

    def test_an_older_inner_policy_still_works(self):
        """It cannot be told about Stop, but it must not raise into the caller."""

        class _Older:
            def await_resume(self, timeout = None) -> bool:
                return True

        assert (
            DeferredPreemptionPolicy(_Older()).await_resume(1.0, cancel_event = threading.Event())
            is True
        )

    def test_an_unbound_wrapper_never_claims_a_resume(self):
        assert DeferredPreemptionPolicy().await_resume(cancel_event = threading.Event()) is False

    def test_the_generator_helper_no_longer_drops_it(self):
        """`_await_resume` retries without Stop when the keyword raises TypeError, and the
        wrapper is what it holds, so the fallback was the live path."""
        from core.inference.llama_cpp import _await_resume

        event = threading.Event()
        seen: list = []

        class _Policy:
            def await_resume(
                self,
                timeout = None,
                *,
                cancel_event = None,
            ) -> bool:
                seen.append(cancel_event)
                return True

        generator = _await_resume(DeferredPreemptionPolicy(_Policy()), event)
        with pytest.raises(StopIteration) as stopped:
            while True:
                next(generator)
        assert stopped.value.value is True
        assert seen == [event]


class TestAStopDuringThePauseEndsTheWait:
    def test_the_controller_policy_reads_the_event_before_it_asks(self, monkeypatch):
        controller = preemption.PreemptionController("http://cancelled-pause")
        controller.configure(budget = 16384, kv_unified = True, slots = 4)

        class _Lease:
            tokens = 100
            is_released = False

        controller.register("mine", lease = _Lease(), tokens = 100)
        controller.set_state("mine", preemption.ParticipantState.PAUSED)
        policy = preemption.ControllerPreemptionPolicy(
            controller, "mine", preemption.PreemptSignal(), loop = None
        )
        grants: list = []
        monkeypatch.setattr(
            type(controller),
            "try_grant_resume",
            lambda *_args, **_kwargs: grants.append(1) or True,
        )

        event = threading.Event()
        event.set()
        assert policy.await_resume(1.0, cancel_event = event) is False
        assert grants == [], "Stop is read before the grant, or a fast grant skips it"


def test_the_checkpoint_shape_is_unchanged():
    """The handshake above is only meaningful while a pause still carries its partial."""
    checkpoint = StreamCheckpoint(visible_text = "half")
    assert checkpoint.has_resume_point()
