# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A cap of the whole window is not a cap, and pricing it as one serialises the queue.

Studio ships "Max Tokens: Max", which sends max_tokens = the context length, and a client
that names nothing gets the same figure from `_build_passthrough_payload`. So the common
request claims it may write the whole window, and the reservation believed it: one chat
committed the entire cache before generating a token.

Measured on a 262144 cache with --parallel 4 --kv-unified, four chats reached first token
at 0.1 / 2.8 / 4.6 / 8.8s with llamacpp:requests_deferred flat at 0: llama-server had free
slots and never saw requests 2 to 4. #10046 fixed the tool-loop half of this; these cover
the other half, which is every plain chat.
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
        assert _openai_llama_admission_output_allowance(
            512, budget = 32768, prompt_tokens = 100
        ) == 512

    def test_no_cap_is_unstated(self):
        assert _openai_llama_admission_output_allowance(
            None, budget = 32768, prompt_tokens = 100
        ) == _OPENAI_LLAMA_ADMISSION_UNSTATED_OUTPUT_TOKENS

    def test_a_cap_of_the_whole_window_is_unstated(self):
        """This is the "Max" setting, and it is the default."""
        assert _openai_llama_admission_output_allowance(
            32768, budget = 32768, prompt_tokens = 100
        ) == _OPENAI_LLAMA_ADMISSION_UNSTATED_OUTPUT_TOKENS

    def test_a_cap_above_the_window_is_unstated(self):
        assert _openai_llama_admission_output_allowance(
            99999, budget = 32768, prompt_tokens = 100
        ) == _OPENAI_LLAMA_ADMISSION_UNSTATED_OUTPUT_TOKENS

    def test_a_cap_just_under_the_window_is_still_a_cap(self):
        """Only the whole window reads as unstated; asking for nearly all of it is a
        statement, and is charged."""
        assert _openai_llama_admission_output_allowance(
            32767, budget = 32768, prompt_tokens = 100
        ) == 32767

    def test_it_never_exceeds_what_is_left(self):
        """On a cache smaller than the allowance, stay inside the budget."""
        assert _openai_llama_admission_output_allowance(
            None, budget = 2048, prompt_tokens = 1800
        ) == 248
        assert _openai_llama_admission_output_allowance(
            None, budget = 2048, prompt_tokens = 4000
        ) == 0


class TestTheDefaultChatNoLongerTakesTheWholeCache:
    BUDGET = 32768
    CAPACITY = 4

    def _cost(self, payload):
        return _openai_llama_admission_tokens(
            payload, budget = self.BUDGET, capacity = self.CAPACITY
        )

    def test_max_tokens_max_does_not_reserve_the_budget(self):
        cost = self._cost(_chat(max_tokens = self.BUDGET))
        assert cost < self.BUDGET, "Max Tokens = Max still reserves the whole cache"
        assert cost <= _OPENAI_LLAMA_ADMISSION_UNSTATED_OUTPUT_TOKENS + 100

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
        assert _openai_llama_admission_tokens(
            _chat(max_tokens = 4096), budget = None, capacity = 4
        ) is None
