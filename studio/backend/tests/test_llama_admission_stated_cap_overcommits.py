# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A request that names its own Max Tokens must still decode beside the others.

THE BEHAVIOUR THIS REPLACES

`_openai_llama_admission_output_allowance` charged a stated cap in full. Measured on
`-c 16384 --parallel 4` with four chats at `max_tokens: 6000` and 3000-token prompts: each
was charged 3000 + 6000 + 8 = 9008 against a 14312 ceiling, a second chat did not fit, and
THE FOUR RAN ONE AT A TIME, peaking at 7990 of 16384. Every one of them was individually
entitled to the whole window and none of them got to share it.

That is the goal's requirement failing for anyone who sets a cap: all P users should decode
in parallel up to N - buffer, and charging a stated cap in full answered "only if nobody
states one". It is also why an entire no-tools benchmark reported `preemptions 0` and read
as a clean pass; there was nothing to preempt because nothing was ever overcommitted.

WHY IT IS SAFE NOW AND WAS NOT BEFORE

Charging less than a request may generate is safe only when something reclaims the
difference. The gap between charge and permission was a BUG while arithmetic was the only
defence; it is the design once eviction is real. So the optimism is gated on
`preemption_active`, and the conservative charge remains for every configuration where a
pause cannot happen: the rollout switch off, no `--kv-unified`, or no budget. Serialising is
a real cost and the right one to pay where the alternative is overrunning the cache.
"""

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
        assert BUDGET // (prompt + before) == 1, (
            "which is why four chats at max_tokens 6000 ran one at a time"
        )
        assert BUDGET // (prompt + after) >= SLOTS, (
            f"charged {after}, so only {BUDGET // (prompt + after)} of {SLOTS} fit"
        )

    def test_a_stated_cap_is_charged_the_same_as_an_unstated_one(self):
        """Which is the whole point: the cap stops being an admission decision.

        It remains a WIRE decision. The clamp still holds the request to what it asked
        for; this only stops the reservation pricing the worst case.
        """
        prompt = 1000
        assert _charged(6000, prompt, active = True) == _charged(None, prompt, active = True)


class TestTheCasesThatMustNotChange:
    def test_a_small_cap_is_still_its_own_estimate(self):
        """`max_tokens: 50` is charged 50, not the flat allowance.

        Charging a small request MORE than it can possibly produce would be a different
        way of wasting the cache, and the flat allowance is 1024.
        """
        assert _charged(50, 200, active = True) == 50
        assert _charged(50, 200, active = False) == 50

    def test_an_unstated_request_is_untouched(self):
        for prompt in (1, 200, 1000, 3000):
            assert _charged(None, prompt, active = True) == _charged(None, prompt, active = False)

    def test_a_cap_at_or_above_the_window_was_already_unstated(self):
        """`_build_passthrough_payload` sends max_tokens = backend_ctx and "Max" sends the
        context length, so both already meant unstated and neither may change."""
        for cap in (BUDGET, BUDGET + 1):
            assert _charged(cap, 3000, active = True) == _charged(None, 3000, active = False)

    def test_the_charge_is_never_zero(self):
        """A zero charge reads as "this request occupies nothing", which would let an
        unbounded number in. Even a pathological prompt keeps a floor of 1."""
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
        """Its documented purpose is to fall back to the wire clamp alone, so it has to
        take admission back with it. A switch that disabled eviction while leaving
        admission overcommitted would be strictly worse than either setting."""
        backend = self._Backend(True)
        monkeypatch.setenv("UNSLOTH_LLAMA_ADMISSION_PREEMPT", "0")
        assert _openai_llama_preemption_will_apply(backend, BUDGET) is False
        monkeypatch.setenv("UNSLOTH_LLAMA_ADMISSION_PREEMPT", "1")
        assert _openai_llama_preemption_will_apply(backend, BUDGET) is True

    def test_the_default_is_conservative(self):
        """Every caller that does not pass the flag keeps the old behaviour exactly.

        There are more callers of this than the two that were updated, in tests and in
        paths not yet reviewed, and none of them should silently become optimistic.
        """
        assert allowance(6000, budget = BUDGET, prompt_tokens = 3000,
                         context_window = BUDGET, share = SHARE) == 6000


class TestTheSumStillHasSomethingHoldingIt:
    def test_capacity_requests_no_longer_fit_and_that_is_deliberate(self):
        """State the overcommit rather than letting it be discovered.

        With the optimistic charge, `sum(prompt + permitted)` exceeds the cache by
        construction, because each request is permitted the whole window. That is the vLLM
        shape the goal asks for, and the watermark is what holds it. If this assertion ever
        flips, admission has gone back to dividing the cache and preemption has nothing
        left to do.
        """
        prompt = 1000
        charged = _charged(6000, prompt, active = True)
        assert (prompt + charged) * SLOTS <= BUDGET, "a full capacity must still be admitted"
        permitted = (prompt + (BUDGET - prompt)) * SLOTS
        assert permitted > BUDGET, (
            "the cache is meant to be overcommitted now; preemption is the enforcement"
        )
