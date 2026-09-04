# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A cap of the whole window is not a cap, and pricing it as one serialises the queue.

Studio ships "Max Tokens: Max", which sends max_tokens = the context length, and a client
naming nothing gets the same figure from `_build_passthrough_payload`. #10046 fixed the
tool-loop half of this; these cover the other half, which is every plain chat.
"""

from routes.inference import (
    _OPENAI_LLAMA_ADMISSION_UNSTATED_OUTPUT_TOKENS,
    _openai_llama_admission_output_allowance,
    _openai_llama_admission_tokens,
)


class _Payload:
    def __init__(self, **fields):
        self.__dict__.update(fields)

    def __getattr__(self, _name):
        return None


def _chat(**fields):
    return _Payload(messages = [{"role": "user", "content": "hi"}], **fields)


class TestWhatCountsAsUnstated:
    def test_a_real_cap_is_charged_as_asked(self):
        assert _openai_llama_admission_output_allowance(512, budget = 32768, prompt_tokens = 100) == 512

    def test_no_cap_is_unstated(self):
        assert (
            _openai_llama_admission_output_allowance(None, budget = 32768, prompt_tokens = 100)
            == _OPENAI_LLAMA_ADMISSION_UNSTATED_OUTPUT_TOKENS
        )

    def test_a_cap_of_the_whole_window_is_unstated(self):
        """This is the "Max" setting, and it is the default."""
        assert (
            _openai_llama_admission_output_allowance(32768, budget = 32768, prompt_tokens = 100)
            == _OPENAI_LLAMA_ADMISSION_UNSTATED_OUTPUT_TOKENS
        )

    def test_a_cap_above_the_window_is_unstated(self):
        assert (
            _openai_llama_admission_output_allowance(99999, budget = 32768, prompt_tokens = 100)
            == _OPENAI_LLAMA_ADMISSION_UNSTATED_OUTPUT_TOKENS
        )

    def test_a_cap_just_under_the_window_is_still_a_cap(self):
        """Only the whole window reads as unstated; asking for nearly all of it is a
        statement, and is charged."""
        assert (
            _openai_llama_admission_output_allowance(32767, budget = 32768, prompt_tokens = 100)
            == 32767
        )

    def test_it_never_exceeds_what_is_left(self):
        """On a cache smaller than the allowance, stay inside the budget."""
        assert (
            _openai_llama_admission_output_allowance(None, budget = 2048, prompt_tokens = 1800) == 248
        )
        assert _openai_llama_admission_output_allowance(None, budget = 2048, prompt_tokens = 4000) == 0


class TestTheDefaultChatNoLongerTakesTheWholeCache:
    BUDGET = 32768
    CAPACITY = 4

    def _cost(self, payload):
        return _openai_llama_admission_tokens(payload, budget = self.BUDGET, capacity = self.CAPACITY)

    def test_max_tokens_max_does_not_reserve_the_budget(self):
        cost = self._cost(_chat(max_tokens = self.BUDGET))
        assert cost < self.BUDGET, "Max Tokens = Max still reserves the whole cache"
        # Its FAIR SHARE, not the flat allowance. This asserted `<= 1024 + 100` while the
        # charge was a flat estimate the share could only lower; the charge is now the
        # whole share, because charging less than the wire permits is what made the
        # reservation unsafe. The property this test exists for is unchanged and is the
        # one asserted here: a default chat never reserves the cache, and `capacity` of
        # them still fit (test_four_default_chats_fit_at_once).
        assert cost <= self.BUDGET // self.CAPACITY

    def test_four_default_chats_fit_at_once(self):
        """The change, as the user sees it: four chats on Max have to fit together."""
        cost = self._cost(_chat(max_tokens = self.BUDGET))
        assert cost * 4 <= self.BUDGET, (
            f"four default chats cost {cost * 4} against a {self.BUDGET} cache, so they "
            f"cannot decode together"
        )

    def test_an_uncapped_chat_fits_too(self):
        cost = self._cost(_chat())
        assert cost * 4 <= self.BUDGET

    def test_a_named_cap_is_unaffected(self):
        """Backwards compatible: a stated cap is still priced prompt plus cap."""
        small = self._cost(_chat(max_tokens = 512))
        assert small < self._cost(_chat(max_tokens = 8192))

    def test_max_completion_tokens_spelling_counts_too(self):
        """The supported spelling reaches the same resolver, so "Max" through it counts."""
        assert self._cost(_chat(max_completion_tokens = self.BUDGET)) < self.BUDGET


class TestNothingChangesWhereThereIsNoBudget:
    def test_an_unknown_budget_still_declines_to_price(self):
        assert (
            _openai_llama_admission_tokens(_chat(max_tokens = 4096), budget = None, capacity = 4) is None
        )


class TestMaxIsPerRequestNotPerCache:
    """Under ``--no-kv-unified`` the budget is the aggregate of N private caches while "Max"
    is still one slot's context_length, so measuring the cap against the budget reads that
    default as a real cap and four default chats stop fitting in four caches.
    """

    WINDOW = 4096  # per slot, and what "Max" sends
    BUDGET = WINDOW * 4  # four private caches

    def test_max_is_unstated_against_the_per_request_window(self):
        assert (
            _openai_llama_admission_output_allowance(
                self.WINDOW,
                budget = self.BUDGET,
                prompt_tokens = 100,
                context_window = self.WINDOW,
            )
            == _OPENAI_LLAMA_ADMISSION_UNSTATED_OUTPUT_TOKENS
        )

    def test_four_default_chats_fit_four_private_caches(self):
        cost = _openai_llama_admission_tokens(
            _chat(max_tokens = self.WINDOW),
            budget = self.BUDGET,
            capacity = 4,
            context_window = self.WINDOW,
        )
        assert (
            cost * 4 <= self.BUDGET
        ), f"four default chats cost {cost * 4} against {self.BUDGET}, so the fourth waits"

    def test_a_real_cap_under_the_window_is_still_a_cap(self):
        assert (
            _openai_llama_admission_output_allowance(
                512, budget = self.BUDGET, prompt_tokens = 100, context_window = self.WINDOW
            )
            == 512
        )

    def test_an_unknown_window_falls_back_to_the_budget(self):
        """Which is the unified case, where the two are the same number anyway."""
        assert (
            _openai_llama_admission_output_allowance(
                self.BUDGET, budget = self.BUDGET, prompt_tokens = 100, context_window = None
            )
            == _OPENAI_LLAMA_ADMISSION_UNSTATED_OUTPUT_TOKENS
        )

    def test_the_window_is_read_off_the_backend(self):
        from types import SimpleNamespace

        from routes.inference import _openai_llama_admission_context_window

        backend = SimpleNamespace(context_length = self.WINDOW, _kv_cache_context_total = self.BUDGET)
        assert _openai_llama_admission_context_window(backend) == self.WINDOW

    def test_an_unreadable_window_is_none(self):
        from types import SimpleNamespace

        from routes.inference import _openai_llama_admission_context_window
        for value in (None, 0, -1, "nonsense"):
            assert (
                _openai_llama_admission_context_window(SimpleNamespace(context_length = value))
                is None
            )


class TestTheAllowanceFitsTheAdvertisedSlots:
    """A flat allowance is most of a share on a small cache, so it broke the very thing it
    was added to fix: at 4096 over four slots a default chat cost 1032 and only three ran,
    and at 2048 only one did. Clamped to the share, ``capacity`` of them always fit.
    """

    def _cost(self, budget, capacity):
        return _openai_llama_admission_tokens(
            _chat(max_tokens = budget),
            budget = budget,
            capacity = capacity,
            context_window = budget,
        )

    def test_capacity_default_chats_fit_at_every_cache_size(self):
        for budget in (2048, 4096, 8192, 32768, 262144):
            cost = self._cost(budget, 4)
            assert cost * 4 <= budget, f"{budget} cache admits only {budget // cost} of 4"

    def test_the_charge_scales_with_the_cache(self):
        """A bigger cache buys a bigger share, and the charge follows it.

        This previously asserted the opposite, that a large cache was UNCHANGED, because
        the charge was a flat allowance the share could only lower. That is no longer the
        rule: the charge is the share itself, so it has to scale or a large cache would be
        priced as if it were small and admit far more than it can hold. What must not
        change with the cache size is how MANY chats fit, which is asserted above.
        """
        assert self._cost(262144, 4) > self._cost(32768, 4)
        for budget in (32768, 262144):
            assert self._cost(budget, 4) == budget // 4

    def test_the_share_only_ever_lowers_the_allowance(self):
        base = _openai_llama_admission_output_allowance(
            None, budget = 4096, prompt_tokens = 100, context_window = 4096
        )
        assert (
            _openai_llama_admission_output_allowance(
                None, budget = 4096, prompt_tokens = 100, context_window = 4096, share = 1024
            )
            < base
        )

    def test_a_prompt_past_its_share_keeps_the_flat_allowance(self):
        """Zeroing it would admit a request with no room to generate at all."""
        assert (
            _openai_llama_admission_output_allowance(
                None, budget = 8192, prompt_tokens = 3000, context_window = 8192, share = 2048
            )
            == _OPENAI_LLAMA_ADMISSION_UNSTATED_OUTPUT_TOKENS
        )

    def test_a_named_cap_ignores_the_share(self):
        assert (
            _openai_llama_admission_output_allowance(
                256, budget = 2048, prompt_tokens = 10, context_window = 2048, share = 512
            )
            == 256
        )


class TestTheAllowanceCannotExceedOneSlot:
    """A request cannot occupy more KV than its own slot holds, so the allowance is clamped
    against the WINDOW; clamping against the aggregate budget charged a long-prompt chat for
    more than a slot can physically hold.
    """

    WINDOW = 4096
    BUDGET = WINDOW * 4

    def test_a_long_prompt_shrinks_the_allowance_to_fit_its_slot(self):
        allowance = _openai_llama_admission_output_allowance(
            self.WINDOW,
            budget = self.BUDGET,
            prompt_tokens = 3500,
            context_window = self.WINDOW,
        )
        assert allowance == self.WINDOW - 3500
        assert 3500 + allowance == self.WINDOW, "a request may not exceed one slot"

    def test_four_long_prompt_chats_still_fit(self):
        prompt = 3500
        allowance = _openai_llama_admission_output_allowance(
            self.WINDOW,
            budget = self.BUDGET,
            prompt_tokens = prompt,
            context_window = self.WINDOW,
        )
        assert (prompt + allowance) * 4 <= self.BUDGET

    def test_a_prompt_that_fills_the_slot_gets_nothing(self):
        assert (
            _openai_llama_admission_output_allowance(
                self.WINDOW,
                budget = self.BUDGET,
                prompt_tokens = self.WINDOW,
                context_window = self.WINDOW,
            )
            == 0
        )

    def test_a_short_prompt_still_gets_the_full_allowance(self):
        assert (
            _openai_llama_admission_output_allowance(
                self.WINDOW,
                budget = self.BUDGET,
                prompt_tokens = 100,
                context_window = self.WINDOW,
            )
            == _OPENAI_LLAMA_ADMISSION_UNSTATED_OUTPUT_TOKENS
        )

    def test_unified_is_unchanged(self):
        """window == budget there, so this clamp is the same arithmetic it always was."""
        assert (
            _openai_llama_admission_output_allowance(
                2048, budget = 2048, prompt_tokens = 1800, context_window = 2048
            )
            == 248
        )
