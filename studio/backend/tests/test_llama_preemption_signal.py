# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A pause that lands mid tool call is worse than no pause at all.

Four parallel chats died together on 2026-09-01 because llama.cpp errors EVERY
processing slot when its one unified KV cache overflows. The fix is to pause a
chat instead, but tool execution cannot be interrupted: between the point where
calls materialise and the point where their results are appended, an abort would
either discard the work of tools that already ran or run them twice on resume.

So the signal defers. These pin the deferral semantics, because "the request is
remembered while hidden" is the whole reason deferring is safe.
"""

import threading

from core.inference import llama_preemption as preemption
from core.inference.llama_cpp import _interrupt_event


class TestItAsksAndForgets:
    def test_a_fresh_signal_is_quiet(self):
        assert not preemption.PreemptSignal().is_set()

    def test_a_request_is_visible_and_carries_its_reason(self):
        signal = preemption.PreemptSignal()
        signal.request("kv_pressure")
        assert signal.is_set()
        assert signal.pending
        assert signal.reason == "kv_pressure"

    def test_clearing_forgets_it(self):
        signal = preemption.PreemptSignal()
        signal.request()
        signal.clear()
        assert not signal.is_set()
        assert not signal.pending
        assert signal.reason is None


class TestTheUnsafeWindow:
    """The property tool execution depends on: hidden, but never dropped."""

    def test_a_request_inside_the_window_is_hidden(self):
        signal = preemption.PreemptSignal()
        with signal.unsafe_window():
            signal.request()
            assert not signal.is_set(), "a pause must not land during tool execution"
            assert signal.pending, "but it must not be forgotten either"

    def test_it_arrives_the_moment_the_window_closes(self):
        signal = preemption.PreemptSignal()
        with signal.unsafe_window():
            signal.request()
        assert signal.is_set()

    def test_a_request_made_before_the_window_is_hidden_too(self):
        """It had not been acted on yet, so honouring it now would be the very
        mid-execution abort the window exists to prevent."""
        signal = preemption.PreemptSignal()
        signal.request()
        assert signal.is_set()
        with signal.unsafe_window():
            assert not signal.is_set()
        assert signal.is_set()

    def test_nesting_waits_for_the_outermost(self):
        signal = preemption.PreemptSignal()
        with signal.unsafe_window():
            with signal.unsafe_window():
                signal.request()
                assert not signal.is_set()
            assert not signal.is_set(), "the outer window is still open"
        assert signal.is_set()

    def test_no_request_means_nothing_fires_on_close(self):
        signal = preemption.PreemptSignal()
        with signal.unsafe_window():
            pass
        assert not signal.is_set()

    def test_clearing_inside_the_window_really_forgets(self):
        signal = preemption.PreemptSignal()
        signal.request()
        with signal.unsafe_window():
            signal.clear()
        assert not signal.is_set(), "a cleared request must not resurface on close"

    def test_the_window_closes_even_when_the_body_raises(self):
        signal = preemption.PreemptSignal()
        try:
            with signal.unsafe_window():
                signal.request()
                raise ValueError("tool blew up")
        except ValueError:
            pass
        assert not signal.deferred
        assert signal.is_set()


class TestCancelAndPauseTogether:
    """The stream plumbing takes one event; these two have to share it without
    becoming each other."""

    def test_either_one_interrupts(self):
        cancel = threading.Event()
        pause = preemption.PreemptSignal()
        combined = _interrupt_event(cancel, pause)
        assert not combined.is_set()
        pause.request()
        assert combined.is_set()

    def test_a_cancel_alone_interrupts(self):
        cancel = threading.Event()
        pause = preemption.PreemptSignal()
        combined = _interrupt_event(cancel, pause)
        cancel.set()
        assert combined.is_set()

    def test_one_event_passes_straight_through(self):
        """So a caller with no pause signal keeps exactly the object it had."""
        cancel = threading.Event()
        assert _interrupt_event(cancel, None) is cancel
        assert _interrupt_event(None, None) is None

    def test_a_deferred_pause_does_not_interrupt(self):
        cancel = threading.Event()
        pause = preemption.PreemptSignal()
        combined = _interrupt_event(cancel, pause)
        with pause.unsafe_window():
            pause.request()
            assert not combined.is_set()

    def test_wait_returns_when_the_pause_arrives(self):
        cancel = threading.Event()
        pause = preemption.PreemptSignal()
        combined = _interrupt_event(cancel, pause)
        threading.Timer(0.05, pause.request).start()
        assert combined.wait(timeout = 5.0) is True

    def test_wait_gives_up_without_one(self):
        combined = _interrupt_event(threading.Event(), preemption.PreemptSignal())
        assert combined.wait(timeout = 0.05) is False


class TestTheCheckpoint:
    def test_text_is_a_resume_point(self):
        assert preemption.StreamCheckpoint(visible_text = "Once upon").has_resume_point()

    def test_nothing_generated_is_not(self):
        """`continue_final_message` refuses an empty assistant turn, so such an
        attempt is re-issued whole rather than continued."""
        assert not preemption.StreamCheckpoint().has_resume_point()
        assert not preemption.StreamCheckpoint(visible_text = "   \n ").has_resume_point()

    def test_reasoning_alone_is_not_a_resume_point(self):
        """Reasoning is not replayed as assistant content, so it cannot be the
        prefix a continuation extends."""
        assert not preemption.StreamCheckpoint(reasoning_text = "hmm").has_resume_point()


class TestTheRolloutSwitch:
    def test_on_by_default(self, monkeypatch):
        monkeypatch.delenv(preemption.PREEMPT_ENV, raising = False)
        assert preemption.preemption_enabled() is True

    def test_it_can_be_turned_off(self, monkeypatch):
        for value in ("0", "false", "no", "off", "OFF"):
            monkeypatch.setenv(preemption.PREEMPT_ENV, value)
            assert preemption.preemption_enabled() is False

    def test_nonsense_keeps_the_default(self, monkeypatch):
        monkeypatch.setenv(preemption.PREEMPT_ENV, "maybe")
        assert preemption.preemption_enabled() is True


class TestTheDefaultPolicy:
    def test_it_never_pauses(self):
        """So every existing call site behaves exactly as it did."""
        policy = preemption.NullPreemptionPolicy()
        assert policy.should_preempt() is False
        assert policy.await_resume() is True

    def test_it_satisfies_the_protocol(self):
        assert isinstance(preemption.NullPreemptionPolicy(), preemption.PreemptionPolicy)
