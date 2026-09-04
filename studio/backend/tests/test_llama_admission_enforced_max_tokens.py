# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A reservation nobody enforces is not a reservation.

Admission charged an unstated "Max Tokens: Max" a bounded allowance while the request
sent to llama-server still said the whole window, so the two disagreed by the whole
cache. Measured on 2026-09-01: four tool chats on `-c 16384 --parallel 4 --kv-unified`
were each admitted at their share, all generated into the one shared pool, and
llama-server errored EVERY processing slot at once. Four conversations, lost together.

The bound is the fair share, and the charge is raised to match it exactly. An earlier
revision let the two differ, on the reasoning that the charge should stay optimistic so
more chats fit while the bound only had to be physically safe. That is unsound in
company: a small prompt charged `prompt + 1024` was still permitted its whole share, so
admitting it beside a large-prompt request let the permitted total pass the cache while
the charged total fit. Charging the whole share costs no concurrency, because
`capacity * share <= budget` by construction.
"""

from types import SimpleNamespace

from routes.inference import (
    _OPENAI_LLAMA_ADMISSION_UNSTATED_OUTPUT_TOKENS,
    _openai_llama_admission_enforced_max_tokens,
    _openai_llama_admission_tokens,
)


class _Payload:
    def __init__(self, **fields):
        self.__dict__.update(fields)

    def __getattr__(self, _name):
        return None


def _chat(text = "hi", **fields):
    return _Payload(messages = [{"role": "user", "content": text}], **fields)


def _backend(*, window, total, slots):
    return SimpleNamespace(
        context_length = window,
        _kv_cache_context_total = total,
        effective_parallel_slots = slots,
    )


def _budget(backend):
    from routes.inference import _openai_llama_admission_budget

    return _openai_llama_admission_budget(backend)


def _enforced(payload, backend):
    return _openai_llama_admission_enforced_max_tokens(
        payload, request = None, llama_backend = backend
    )


class TestTheInvariant:
    """The invariant MOVED. It is no longer arithmetic, it is eviction.

    This class used to assert `capacity * (prompt + permitted) <= budget`: admission
    divided the cache so an overrun was impossible by construction. That is safe and it
    is not what was asked for. It held every chat to about `N / slots` -- measured
    2026-09-01, ~4049 tokens an attempt on a 16384 cache, with long answers grinding
    through length continuations and two of four not finishing inside 900s while most of
    the cache sat idle.

    Every chat is now permitted the whole window and the cache is overcommitted on
    purpose, exactly as vLLM admits against the full max_model_len. What keeps the cache
    inside its bounds is the watermark sweep in `llama_preemption`, which sees each n_i
    grow and evicts. Those guarantees are asserted in
    `test_llama_preemption_wiring.py::TestTheWatermarkSweep`, not here.

    What remains true here is narrower and still worth pinning: no single request may be
    permitted more than the window its own slot holds.
    """

    def test_no_request_may_exceed_its_own_window(self):
        for total in (2048, 4096, 8192, 16384, 65536, 262144):
            backend = _backend(window = total, total = total, slots = 4)
            payload = _chat(max_tokens = total)
            enforced = _enforced(payload, backend)
            assert enforced is not None
            assert _prompt_tokens(payload) + enforced <= total, (
                f"{total}: a single request may occupy more than the whole window"
            )

    def test_it_holds_for_a_long_prompt(self):
        backend = _backend(window = 16384, total = 16384, slots = 4)
        payload = _chat("word " * 600, max_tokens = 16384)
        enforced = _enforced(payload, backend)
        assert enforced is not None
        assert _prompt_tokens(payload) + enforced <= 16384

    def test_the_whole_window_is_offered_not_a_share(self):
        """The point of the change: a chat is no longer rationed by the slot count."""
        backend = _backend(window = 16384, total = 16384, slots = 4)
        enforced = _enforced(_chat(max_tokens = 16384), backend)
        assert enforced > 16384 // 4 * 3, (
            f"permitted {enforced} still looks like a share of the cache"
        )

    def test_it_holds_at_other_slot_counts(self):
        for slots in (2, 3, 4, 8):
            backend = _backend(window = 32768, total = 32768, slots = slots)
            payload = _chat(max_tokens = 32768)
            enforced = _enforced(payload, backend)
            assert enforced is not None
            assert _prompt_tokens(payload) + enforced <= 32768
            # And unchanged by how many slots exist: the window is the window.
            assert enforced > 32768 // max(2, slots) * 1.5


class TestWhatIsLeftAlone:
    def test_a_stated_cap_is_never_clamped(self):
        """It is already honest: charged and sent as the same number."""
        backend = _backend(window = 16384, total = 16384, slots = 4)
        assert _enforced(_chat(max_tokens = 512), backend) is None
        assert _enforced(_chat(max_completion_tokens = 2048), backend) is None

    def test_a_single_slot_is_unrestricted(self):
        """One slot owns the whole cache, so there is nothing to divide."""
        backend = _backend(window = 16384, total = 16384, slots = 1)
        assert _enforced(_chat(max_tokens = 16384), backend) is None

    def test_an_unknown_budget_changes_nothing(self):
        backend = SimpleNamespace(context_length = None, effective_parallel_slots = 4)
        assert _enforced(_chat(max_tokens = 4096), backend) is None

    def test_a_shape_with_no_messages_is_left_alone(self):
        """`/completions` takes a prompt string; there is nothing to measure."""
        backend = _backend(window = 16384, total = 16384, slots = 4)
        assert _enforced(_Payload(max_tokens = 16384), backend) is None

    def test_a_private_cache_per_slot_is_unrestricted(self):
        """Under --no-kv-unified the aggregate is N times the window, so a share IS the
        window and no request can overrun anyone else."""
        backend = _backend(window = 4096, total = 16384, slots = 4)
        assert _enforced(_chat(max_tokens = 4096), backend) is None


class TestTheEdges:
    def test_a_prompt_that_fills_the_window_still_gets_a_token(self):
        """Zero would be refused upstream, so the floor is one. Reached now only by a
        prompt approaching the WHOLE window rather than a quarter of it."""
        backend = _backend(window = 16384, total = 16384, slots = 4)
        enforced = _enforced(_chat("word " * 20000, max_tokens = 16384), backend)
        assert enforced == 1

    def test_the_bound_deliberately_exceeds_the_charge(self):
        """They diverge ON PURPOSE now, and that is the whole design.

        The charge is what admission RESERVES so several chats fit; the bound is what a
        single request is physically allowed. Under the divided design these had to be
        equal, because arithmetic was the only defence and a request admitted on less
        than it could use overran the cache. That is no longer how safety is obtained:
        the cache is overcommitted deliberately and the watermark sweep evicts.

        This is the same shape as the bug of 2026-09-01, and the difference is not
        cosmetic. Then, nothing watched the gap. Now the sweep does, on every 32 tokens,
        and it is asserted in TestTheWatermarkSweep. If that sweep is ever removed this
        gap becomes the crash again.
        """
        backend = _backend(window = 16384, total = 16384, slots = 4)
        payload = _chat(max_tokens = 16384)
        charged = _openai_llama_admission_tokens(
            payload, budget = _budget(backend), capacity = 4, context_window = 16384
        )
        enforced = _enforced(payload, backend)
        permitted = _prompt_tokens(payload) + enforced
        assert permitted > charged, (
            f"permitted {permitted} should exceed the charge {charged}: admission "
            "reserves a share so several chats fit, while each may use the window"
        )
        assert enforced > _OPENAI_LLAMA_ADMISSION_UNSTATED_OUTPUT_TOKENS

    def test_the_charge_never_exceeds_what_is_permitted(self):
        """If it did, admission would be reserving room the request cannot use."""
        backend = _backend(window = 16384, total = 16384, slots = 4)
        payload = _chat(max_tokens = 16384)
        charged = _openai_llama_admission_tokens(
            payload, budget = _budget(backend), capacity = 4, context_window = 16384
        )
        assert charged <= _prompt_tokens(payload) + _enforced(payload, backend)


def _prompt_tokens(payload):
    from routes.inference import _openai_llama_admission_prompt_tokens

    return _openai_llama_admission_prompt_tokens(payload) or 0


class TestItReachesTheWireWithoutBecomingTheCallersCap:
    """The bound has to land on the request and nowhere else.

    Charging one figure and sending another is the whole defect, so the allowance must
    reach `payload["max_tokens"]`. But folding it into the caller's `max_tokens` instead
    is its own trap: `_loop_budget_left` reads that as "what the caller allowed" and
    stops continuing once it is spent, which would replace a crash with a silent
    truncation at one share. Both halves are asserted here because the first draft of
    this change did exactly the wrong one.
    """

    def _source(self):
        from pathlib import Path

        import core.inference.llama_cpp as llama_cpp

        return Path(llama_cpp.__file__).read_text()

    def test_both_payload_sites_apply_the_allowance(self):
        source = self._source()
        applied = source.count(
            'payload["max_tokens"] = min(payload["max_tokens"], admission_output_allowance)'
        )
        assert applied == 2, (
            f"expected the plain stream and the tool loop to bound the wire cap, found {applied}"
        )

    def test_the_loop_budget_never_sees_it(self):
        """`_loop_budget_left` answers "did the CALLER cap this", and an admission bound
        is not the caller speaking."""
        lines = self._source().split("\n")
        start = next(i for i, l in enumerate(lines) if "def _loop_budget_left" in l)
        indent = len(lines[start]) - len(lines[start].lstrip())
        body = []
        for line in lines[start + 1:]:
            if line.strip() and (len(line) - len(line.lstrip())) <= indent:
                break
            body.append(line)
        assert body, "could not read the body of _loop_budget_left"
        assert "admission_output_allowance" not in "\n".join(body), (
            "the admission bound leaked into the caller's continuation budget"
        )

    def test_both_entry_points_accept_it(self):
        import inspect

        from core.inference.llama_cpp import LlamaCppBackend

        for name in ("generate_chat_completion", "generate_chat_completion_with_tools"):
            params = inspect.signature(getattr(LlamaCppBackend, name)).parameters
            assert "admission_output_allowance" in params, name
            assert params["admission_output_allowance"].default is None, name


class TestChargedAndPermittedCannotDrift:
    """The bound is only safe if nothing is admitted on less than it may use.

    Found by asking what happens when the allowance floors at 1. A prompt past its share
    is permitted ``prompt + 1``, which is far more than a share, and the defence was that
    such a request is charged more than a share so fewer are admitted. That holds on its
    own, but not in company: a SMALL prompt was charged ``prompt + 1024`` while being
    permitted its whole share, and mixing the two let the permitted total pass the cache
    while the charged total still fit.
    """

    def _charged(self, budget, share, prompt):
        from routes.inference import _openai_llama_admission_output_allowance

        allowance = _openai_llama_admission_output_allowance(
            None,
            budget = budget,
            prompt_tokens = prompt,
            context_window = budget,
            share = share,
        )
        return max(1, min(budget, prompt + allowance))

    def test_the_mixed_set_that_broke_the_invariant(self):
        """Measured before the fix: charged 258774 of 262144, permitted 385750."""
        budget, slots = 262144, 4
        share = budget // slots
        prompts = [1, 65537, 189139, 1]
        admitted, used = [], 0
        for prompt in prompts:
            charged = self._charged(budget, share, prompt)
            if len(admitted) < slots and used + charged <= budget:
                used += charged
                admitted.append(prompt)
        # Against what the wire ACTUALLY permits, which is the window, not the share.
        # Computing `share - prompt` here was the test agreeing with an older design; the
        # clamp says "THE WINDOW, not a share of it" and the permitted total therefore
        # exceeds the cache by construction. That is the vLLM shape the goal asks for:
        # overcommit deliberately, then preempt at a watermark.
        permitted = sum(prompt + max(1, budget - prompt) for prompt in admitted)
        assert permitted > budget, (
            "the cache is meant to be overcommitted now; if this ever holds, admission has "
            "gone back to dividing the window and preemption has nothing left to do"
        )
        # What still has to hold is the thing that actually bounds concurrency.
        assert used <= budget, f"admitted {admitted} charged {used} of {budget}"
        assert len(admitted) <= slots

    def test_the_charge_is_deliberately_less_than_the_permission(self):
        """This asserted the opposite, and the opposite was already false.

        It was `test_nothing_is_admitted_on_less_than_it_may_use`, requiring
        `charged >= permitted` so that `sum(charged) <= budget` implied the permitted total
        fit. It passed only because it computed `permitted` as `prompt + (share - prompt)`
        inline. The code permits `window - prompt`:

            budget=16384  share=4096  prompt=8  charged=4096  permitted=16384

        so the property had already been abandoned, by a factor of four, and the test did
        not notice because it never asked the function.

        The charge is now an admission estimate and nothing more. Preemption enforces the
        cache, which is why it must stay timely: if eviction stops working this becomes the
        crash again, and no arithmetic here will catch it.
        """
        for budget, slots in ((16384, 4), (4096, 4), (2048, 2), (32768, 8), (262144, 4)):
            share = budget // slots
            for prompt in (1, 8, share // 2, share - 2):
                if prompt < 1:
                    continue
                charged = self._charged(budget, share, prompt)
                # It must still be cheap enough that a full capacity fits, which is the
                # property that actually bounds how many chats are admitted at once.
                assert charged * slots <= budget or charged <= share, (
                    f"budget={budget} slots={slots} prompt={prompt}: charged {charged}"
                )
                # And it must not silently become the permission again.
                assert charged < prompt + max(1, budget - prompt)

    def test_a_full_capacity_of_unstated_requests_still_fits(self):
        """Charging the whole share must not cost the concurrency #10070 bought."""
        for budget, slots in ((16384, 4), (4096, 4), (32768, 8), (262144, 4)):
            share = budget // slots
            charged = self._charged(budget, share, 8)
            assert charged * slots <= budget, (
                f"budget={budget} slots={slots}: {slots} small chats charge {charged * slots}"
            )
