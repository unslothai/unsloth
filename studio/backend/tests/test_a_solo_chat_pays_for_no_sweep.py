# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""One chat on the cache has nobody to preempt, so it pays for no reading of it.

The token callback fires every 32 chunks and read `/slots` behind it: a synchronous HTTP
round trip with a three second timeout, on the reader's own thread, deciding nothing. The
barriers that DO decide something -- admission, a resume wait, a reclaim -- pass `force`
and still read.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.inference import llama_preemption as preemption
import routes.inference as inference

pytestmark = pytest.mark.usefixtures("preemption_opted_in")


def _backend(port: int):
    return SimpleNamespace(
        base_url = f"http://slots-probe-{port}",
        context_length = 16384,
        _kv_cache_context_total = 16384,
        effective_parallel_slots = 4,
        _kv_cache_unified = True,
        _auth_headers = {},
    )


@pytest.fixture
def _slots(monkeypatch):
    calls: list[str] = []

    def _fetch(base, headers = None):
        calls.append(base)
        return [{"id": 0, "is_processing": True, "n_prompt_tokens": 132}]

    monkeypatch.setattr(inference, "fetch_llama_slots", _fetch)
    return calls


def _controller(backend):
    controller = inference.get_preemption_controller(inference._preempt_key(backend))
    controller.configure(budget = 16384, kv_unified = True, slots = 4)
    return controller


class TestTheTokenPath:
    def test_the_growth_is_on_the_ledger_before_the_reading_goes_out(self, monkeypatch):
        """The `/slots` round trip can take its whole three second timeout while the
        server keeps decoding, so the count is recorded first and the reading refines it."""
        backend = _backend(7)
        controller = _controller(backend)
        controller.register("a", tokens = 100, prompt_tokens = 100)
        controller.register("b", tokens = 100, prompt_tokens = 100)
        seen: list[int] = []

        def _fetch(base, headers = None):
            seen.append(controller.participant("a").generated_seen)
            return [{"id": 0, "is_processing": True, "n_prompt_tokens": 132}]

        monkeypatch.setattr(inference, "fetch_llama_slots", _fetch)
        _refresh, observe, _state = inference._openai_llama_residency_observer(
            llama_backend = backend, completion_id = "a"
        )

        observe(150)

        assert seen == [150], "the reading went out before the growth was recorded"

    def test_a_solo_chat_reads_no_slots(self, monkeypatch, _slots):
        backend = _backend(1)
        controller = _controller(backend)
        controller.register("solo", tokens = 100)
        _refresh, observe, _state = inference._openai_llama_residency_observer(
            llama_backend = backend, completion_id = "solo"
        )

        observe(32)

        assert _slots == [], "a chat alone on the cache paid for an HTTP round trip"

    def test_its_ledger_still_moves(self, monkeypatch, _slots):
        """Skipping the reading must not make it invisible to the next chat to arrive."""
        backend = _backend(2)
        controller = _controller(backend)
        controller.register("solo", tokens = 100, prompt_tokens = 100)
        _refresh, observe, _state = inference._openai_llama_residency_observer(
            llama_backend = backend, completion_id = "solo"
        )

        observe(500)

        assert controller.committed_tokens() == 600

    def test_a_second_holder_brings_the_reading_back(self, monkeypatch, _slots):
        backend = _backend(3)
        controller = _controller(backend)
        controller.register("mine", tokens = 100)
        controller.register("theirs", tokens = 100)
        _refresh, observe, _state = inference._openai_llama_residency_observer(
            llama_backend = backend, completion_id = "mine"
        )

        observe(32)

        assert _slots, "with somebody else on the cache the reading decides who stops"

    def test_a_waiter_brings_it_back_too(self, monkeypatch, _slots):
        """A paused chat holds no cells, but the reading is what grants it room."""
        backend = _backend(4)
        controller = _controller(backend)
        controller.register("mine", tokens = 100)
        controller.register("paused", tokens = 100, state = preemption.ParticipantState.PAUSED)
        _refresh, observe, _state = inference._openai_llama_residency_observer(
            llama_backend = backend, completion_id = "mine"
        )

        observe(32)

        assert _slots


class TestTheBarriersStillRead:
    def test_a_forced_refresh_reads_even_alone(self, monkeypatch, _slots):
        """This is the probe admission and the resume wait call."""
        backend = _backend(5)
        controller = _controller(backend)
        controller.register("solo", tokens = 100)
        refresh, _observe, _state = inference._openai_llama_residency_observer(
            llama_backend = backend, completion_id = "solo"
        )

        refresh(controller, force = True)

        assert _slots, "the grant boundary must not decide on a cached figure"


class TestWhatContendedMeans:
    def test_one_holder_is_not_contention(self):
        controller = preemption.PreemptionController("http://contended-1")
        controller.configure(budget = 16384, kv_unified = True, slots = 4)
        controller.register("solo", tokens = 100)
        assert controller.contended() is False

    def test_two_holders_are(self):
        controller = preemption.PreemptionController("http://contended-2")
        controller.configure(budget = 16384, kv_unified = True, slots = 4)
        controller.register("a", tokens = 100)
        controller.register("b", tokens = 100)
        assert controller.contended() is True

    def test_a_paused_chat_counts_even_though_it_holds_nothing(self):
        controller = preemption.PreemptionController("http://contended-3")
        controller.configure(budget = 16384, kv_unified = True, slots = 4)
        controller.register("a", tokens = 100)
        controller.register("b", tokens = 100, state = preemption.ParticipantState.PAUSED)
        controller.set_state("a", preemption.ParticipantState.PAUSED)
        assert controller.contended() is True

    def test_a_backend_the_preemptor_cannot_apply_to_is_never_contended(self):
        controller = preemption.PreemptionController("http://contended-4")
        controller.configure(budget = 16384, kv_unified = False, slots = 4)
        controller.register("a", tokens = 100)
        controller.register("b", tokens = 100)
        assert controller.contended() is False

    def test_the_switch_off_is_not_contention_either(self, monkeypatch):
        monkeypatch.setenv("UNSLOTH_LLAMA_ADMISSION_PREEMPT", "0")
        controller = preemption.PreemptionController("http://contended-5")
        controller.configure(budget = 16384, kv_unified = True, slots = 4)
        controller.register("a", tokens = 100)
        controller.register("b", tokens = 100)
        assert controller.contended() is False
