# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A cap of the whole window is not a cap, and pricing it as one serialises the queue.

Studio ships "Max Tokens: Max", and Max sends ``max_tokens`` = the context length. A
client that names nothing gets the same figure a different way: ``_build_passthrough_payload``
fills in ``max_tokens = backend_ctx``. So the overwhelmingly common request arrives
claiming it may write the entire window, and the reservation believed it: one chat
committed the whole cache before generating a token, and the next was refused however
little either actually wrote.

That is a budgeting artefact, not the hardware. Measured on a 262144 cache with
``--parallel 4 --kv-unified``, four chats reached first token at 0.1 / 2.8 / 4.6 / 8.8s
with ``llamacpp:requests_deferred`` flat at 0 the whole time: llama-server had free
slots and never saw requests 2 to 4, because Studio held them.

#10046 fixed the tool-loop half of this (``if tool_loop: return budget``). These cover
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
        """Only "the whole window" is read as unstated. A caller that deliberately asks
        for nearly all of it has said something, and is charged for it."""
        assert _openai_llama_admission_output_allowance(
            32767, budget = 32768, prompt_tokens = 100
        ) == 32767

    def test_it_never_exceeds_what_is_left(self):
        """On a cache smaller than the default allowance, the reservation stays inside the
        budget rather than asking for room that does not exist."""
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
        """The point of the change, stated as the thing the user sees. Four chats on
        Max Tokens = Max have to fit in one cache together, or the fourth waits."""
        cost = self._cost(_chat(max_tokens = self.BUDGET))
        assert cost * 4 <= self.BUDGET, (
            f"four default chats cost {cost * 4} against a {self.BUDGET} cache, so they "
            f"cannot decode together"
        )

    def test_an_uncapped_chat_fits_too(self):
        cost = self._cost(_chat())
        assert cost * 4 <= self.BUDGET

    def test_a_named_cap_is_unaffected(self):
        """Backwards compatibility: a request that states a real cap is priced exactly as
        it was before, prompt plus cap."""
        small = self._cost(_chat(max_tokens = 512))
        assert small < self._cost(_chat(max_tokens = 8192))

    def test_max_completion_tokens_spelling_counts_too(self):
        """The supported spelling reaches the same resolver, so "Max" through it must be
        read as unstated as well."""
        assert self._cost(_chat(max_completion_tokens = self.BUDGET)) < self.BUDGET


class TestNothingChangesWhereThereIsNoBudget:
    def test_an_unknown_budget_still_declines_to_price(self):
        assert _openai_llama_admission_tokens(
            _chat(max_tokens = 4096), budget = None, capacity = 4
        ) is None
