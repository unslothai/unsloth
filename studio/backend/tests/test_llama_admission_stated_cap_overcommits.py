# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A request that names its own Max Tokens must still decode beside the others."""

import pytest

from routes.inference import (
    _openai_llama_admission_output_allowance as allowance,
    _openai_llama_preemption_will_apply,
)


BUDGET = 16384
SLOTS = 4
SHARE = BUDGET // SLOTS


def _charged(cap, prompt, *, active):
    return allowance(
        cap,
        budget = BUDGET,
        prompt_tokens = prompt,
        context_window = BUDGET,
        share = SHARE,
        preemption_active = active,
    )


class TestTheStatedCapNoLongerSerialises:
    def test_four_chats_fit_where_one_did(self):
        """The measured case, as arithmetic."""
        prompt, cap = 3000, 6000
        before = _charged(cap, prompt, active = False)
        after = _charged(cap, prompt, active = True)
        assert before == cap, "the old behaviour was to charge the cap in full"
        assert (
            BUDGET // (prompt + before) == 1
        ), "which is why four chats at max_tokens 6000 ran one at a time"
        assert (
            BUDGET // (prompt + after) >= SLOTS
        ), f"charged {after}, so only {BUDGET // (prompt + after)} of {SLOTS} fit"

    def test_a_stated_cap_is_charged_the_same_as_an_unstated_one(self):
        """Which is the whole point: the cap stops being an admission decision."""
        prompt = 1000
        assert _charged(6000, prompt, active = True) == _charged(None, prompt, active = True)


class TestTheCasesThatMustNotChange:
    def test_a_small_cap_is_still_its_own_estimate(self):
        """`max_tokens: 50` is charged 50, not the flat allowance."""
        assert _charged(50, 200, active = True) == 50
        assert _charged(50, 200, active = False) == 50

    def test_an_unstated_request_is_untouched(self):
        for prompt in (1, 200, 1000, 3000):
            assert _charged(None, prompt, active = True) == _charged(None, prompt, active = False)

    def test_a_cap_at_or_above_the_window_was_already_unstated(self):
        """`_build_passthrough_payload` sends max_tokens = backend_ctx and "Max" sends the context
        length, so both already meant unstated and neither may change.
        """
        for cap in (BUDGET, BUDGET + 1):
            assert _charged(cap, 3000, active = True) == _charged(None, 3000, active = False)

    def test_the_charge_is_never_zero(self):
        """A zero charge reads as "this request occupies nothing", which would let an unbounded
        number in.
        """
        assert _charged(6000, BUDGET - 1, active = True) >= 1
        assert _charged(1, BUDGET - 1, active = True) >= 1


class TestTheGateIsTheEnforcementItself:
    """The optimism must be switched on by exactly what makes it survivable."""

    class _Backend:
        def __init__(self, unified):
            self._kv_cache_unified = unified

    def test_no_kv_unified_means_no_optimism(self):
        # Without one shared pool a paused slot's cells cannot be purged for anyone else;
        # try_clear_idle_slots is gated on exactly this.
        assert _openai_llama_preemption_will_apply(self._Backend(False), BUDGET) is False

    def test_no_budget_means_no_optimism(self):
        assert _openai_llama_preemption_will_apply(self._Backend(True), 0) is False
        assert _openai_llama_preemption_will_apply(self._Backend(True), None) is False

    def test_the_rollout_switch_turns_it_off(self, monkeypatch):
        """Its documented purpose is to fall back to the wire clamp alone, so it has to take
        admission back with it.
        """
        backend = self._Backend(True)
        monkeypatch.setenv("UNSLOTH_LLAMA_ADMISSION_PREEMPT", "0")
        assert _openai_llama_preemption_will_apply(backend, BUDGET) is False
        monkeypatch.setenv("UNSLOTH_LLAMA_ADMISSION_PREEMPT", "1")
        assert _openai_llama_preemption_will_apply(backend, BUDGET) is True

    def test_admission_accounting_off_turns_it_off(self, monkeypatch):
        """The budget names the cache llama-server allocated whether or not admission is charging
        against it.
        """
        backend = self._Backend(True)
        monkeypatch.setenv("UNSLOTH_LLAMA_ADMISSION_PREEMPT", "1")
        monkeypatch.setenv("UNSLOTH_LLAMA_ADMISSION_KV_BUDGET", "0")
        assert _openai_llama_preemption_will_apply(backend, BUDGET) is False
        monkeypatch.setenv("UNSLOTH_LLAMA_ADMISSION_KV_BUDGET", "1")
        monkeypatch.setenv("UNSLOTH_LLAMA_ADMISSION_CONTROL", "0")
        assert _openai_llama_preemption_will_apply(backend, BUDGET) is False
        monkeypatch.setenv("UNSLOTH_LLAMA_ADMISSION_CONTROL", "1")
        assert _openai_llama_preemption_will_apply(backend, BUDGET) is True

    def test_the_default_is_conservative(self):
        """Every caller that does not pass the flag keeps the old behaviour exactly."""
        assert (
            allowance(6000, budget = BUDGET, prompt_tokens = 3000, context_window = BUDGET, share = SHARE)
            == 6000
        )


class TestTheSumStillHasSomethingHoldingIt:
    def test_capacity_requests_no_longer_fit_and_that_is_deliberate(self):
        """State the overcommit rather than letting it be discovered."""
        prompt = 1000
        charged = _charged(6000, prompt, active = True)
        assert (prompt + charged) * SLOTS <= BUDGET, "a full capacity must still be admitted"
        permitted = (prompt + (BUDGET - prompt)) * SLOTS
        assert (
            permitted > BUDGET
        ), "the cache is meant to be overcommitted now; preemption is the enforcement"
