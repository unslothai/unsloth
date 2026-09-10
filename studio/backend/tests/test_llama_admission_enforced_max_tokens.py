# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A reservation nobody enforces is not a reservation."""

from types import SimpleNamespace

import pytest

from routes.inference import (
    _OPENAI_LLAMA_ADMISSION_UNSTATED_OUTPUT_TOKENS,
    _OPENAI_LLAMA_ADMISSION_WIRE_RESERVE_TOKENS as _RESERVE,
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


def _backend(
    *,
    window,
    total,
    slots,
    unified = True,
):
    # ``_kv_cache_unified`` as the real backend sets it under --kv-unified: the window is
    # offered only while preemption can reclaim, and that needs one shared pool.
    return SimpleNamespace(
        context_length = window,
        _kv_cache_context_total = total,
        effective_parallel_slots = slots,
        _kv_cache_unified = unified,
    )


def _budget(backend):
    from routes.inference import _openai_llama_admission_budget
    return _openai_llama_admission_budget(backend)


def _enforced(payload, backend):
    return _openai_llama_admission_enforced_max_tokens(payload, request = None, llama_backend = backend)


class TestTheInvariant:
    """No single request may be permitted more than the window its own slot holds."""

    # Opted in per class, not per module: the two classes below assert the SHARE-based
    # bound a default install gets, and the whole-window bound asserted here exists only
    # where a pause can reclaim it.
    pytestmark = pytest.mark.usefixtures("preemption_opted_in")

    def test_no_request_may_exceed_its_own_window(self):
        for total in (2048, 4096, 8192, 16384, 65536, 262144):
            backend = _backend(window = total, total = total, slots = 4)
            payload = _chat(max_tokens = total)
            enforced = _enforced(payload, backend)
            assert enforced is not None
            assert (
                _prompt_tokens(payload) + enforced <= total
            ), f"{total}: a single request may occupy more than the whole window"

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
        assert (
            enforced > 16384 // 4 * 3
        ), f"permitted {enforced} still looks like a share of the cache"

    def test_it_holds_at_other_slot_counts(self):
        for slots in (2, 3, 4, 8):
            backend = _backend(window = 32768, total = 32768, slots = slots)
            payload = _chat(max_tokens = 32768)
            enforced = _enforced(payload, backend)
            assert enforced is not None
            assert _prompt_tokens(payload) + enforced <= 32768
            # And unchanged by how many slots exist: the window is the window.
            assert enforced > 32768 // max(2, slots) * 1.5


class TestThePoolIsNeverFilledToTheLastCell:
    """Two measured costs this module cannot price, both covered by the reserve.

    llama-server stops a sequence on `prompt.n_tokens() + 1 >= slot.n_ctx`, so a request held
    to exactly its share leaves the pool nothing to place its next token in; and the
    estimator prices the message list while llama-server prices the rendered template.

    Measured on b10840 at `-c 16384 --parallel 4 --kv-unified`: four chats summing to exactly
    16384 cells lost every chat in 3 of 6, 4 of 8 and 7 of 12 waves. Four fresh chats on a
    one-line question lost every chat in 6 of 6 with an 8-token reserve, and none in 8 with
    64, which is the measured 38-token template envelope plus margin.

    The reserve is what keeps a full capacity off the last cell, and a full capacity is only
    the question while nothing can reclaim: with the preemptor on, the pool is deliberately
    overcommitted and the controller is the thing that keeps it inside its budget. So these
    are asked with the switch off, which is the regime the measurements above were taken in.
    """

    @pytest.fixture(autouse = True)
    def _no_reclaimer(self, monkeypatch):
        monkeypatch.setenv("UNSLOTH_LLAMA_ADMISSION_PREEMPT", "0")

    def test_a_full_capacity_leaves_the_pool_room_to_step(self):
        for window, slots in ((16384, 4), (16384, 2), (16384, 8), (65536, 4), (4096, 4)):
            backend = _backend(window = window, total = window, slots = slots)
            payload = _chat(max_tokens = window)
            enforced = _enforced(payload, backend)
            assert enforced is not None
            occupancy = (_prompt_tokens(payload) + enforced) * slots
            assert occupancy < window, f"{window}/{slots}: fills the pool to {occupancy}"
            assert (
                window - occupancy >= slots
            ), f"{window}/{slots}: only {window - occupancy} cells left for {slots} sequences"

    def test_a_prompt_inside_the_reserve_of_its_share_does_not_reclaim_it(self):
        """At `share - 1` the fair-share allowance is 1, the reserve takes it below zero and
        the floor of one used to hand back exactly `share`, which is the exact fill that
        loses every chat. Such a prompt does not fit its share, so it is priced like one
        that is over it: a bigger charge, and the queue admits fewer."""
        from routes.inference import (
            _openai_llama_admission_output_allowance as allowance,
            _openai_llama_admission_wire_output_bound as wire_bound,
        )

        window, slots = 16384, 4
        share = window // slots
        for prompt in range(share - _RESERVE, share + 2):
            charged_allowance = allowance(
                None,
                budget = window,
                prompt_tokens = prompt,
                context_window = window,
                share = share,
            )
            charged = max(1, min(window, max(share, prompt + charged_allowance)))
            sent = wire_bound(share = share, prompt_tokens = prompt, window = window, budget = window)
            assert prompt + sent <= charged, (prompt, sent, charged)
            assert prompt + sent != share, f"{prompt}: fills the pool to exactly its share"

    def test_a_prompt_just_clear_of_the_reserve_still_takes_its_share(self):
        """The band is only the reserve wide; below it nothing changes."""
        from routes.inference import _openai_llama_admission_wire_output_bound as wire_bound

        window, slots = 16384, 4
        share = window // slots
        prompt = share - _RESERVE - 1
        sent = wire_bound(share = share, prompt_tokens = prompt, window = window, budget = window)
        assert (prompt + sent) * slots < window
        assert prompt + sent == share - _RESERVE

    def test_the_reserve_is_taken_out_of_the_charge_not_added_to_it(self):
        """The ledger holds `prompt + allowance`; the reserve is room it paid for and did
        not spend. Charging for it would admit fewer chats to buy the same safety."""
        from routes.inference import (
            _openai_llama_admission_output_allowance as allowance,
            _openai_llama_admission_wire_output_bound as wire_bound,
        )

        window, slots = 16384, 4
        share = window // slots
        payload = _chat(max_tokens = window)
        prompt = _prompt_tokens(payload)
        charged = _openai_llama_admission_tokens(
            payload, budget = window, capacity = slots, context_window = window
        )
        # Nothing on top of `prompt + allowance`: the charge is what it was before there
        # was a reserve at all.
        assert charged == prompt + allowance(
            None,
            budget = window,
            prompt_tokens = prompt,
            context_window = window,
            share = share,
        )
        # The wire is the side that gives the cells up, and by exactly the reserve.
        sent = wire_bound(share = share, prompt_tokens = prompt, window = window, budget = window)
        assert sent == share - prompt - _RESERVE

    def test_an_over_share_prompt_keeps_a_usable_allowance(self):
        """The reserve comes off the flat allowance too, and must not floor it."""
        backend = _backend(window = 16384, total = 16384, slots = 4)
        payload = _chat("word " * 4000, max_tokens = 16384)
        enforced = _enforced(payload, backend)
        assert enforced is not None and enforced > 512, enforced


class TestTheMarkupTheBuilderRewrites:
    """Every builder sends `neutralize_control_markup_in_messages(...)`, not the list the
    route priced. A marker in the user's own text becomes ordinary words, so the prompt the
    wire carries is longer than the raw one.

    Measured on b10840 with Qwen3: 32 markers cost 128 more REAL tokens after the rewrite
    (185 -> 313), and 200 cost 800 (857 -> 1657). Pricing the raw list therefore hands back
    an allowance the prompt has already spent, and a full capacity of such requests puts the
    pool back over its budget.

    Asked with the preemptor off, for the reason the class above gives: a full capacity is
    only bounded by the share while nothing can reclaim the overcommit.
    """

    _MARKER = "<|im_start|>"

    @pytest.fixture(autouse = True)
    def _no_reclaimer(self, monkeypatch):
        monkeypatch.setenv("UNSLOTH_LLAMA_ADMISSION_PREEMPT", "0")

    def _bound(self, markers):
        backend = _backend(window = 16384, total = 16384, slots = 4)
        text = "explain this template: " + (self._MARKER + "user hello ") * markers
        return _enforced(_chat(text, max_tokens = 16384), backend)

    def test_a_prompt_full_of_markers_is_priced_after_the_rewrite(self):
        """The rewrite only grows the prompt, so the allowance only shrinks."""
        clean = self._bound(0)
        marked = self._bound(64)
        assert marked < clean, (clean, marked)

    def test_the_wire_figure_counts_the_rewrite(self):
        from routes.inference import _openai_llama_admission_wire_prompt_tokens as wire

        raw = [{"role": "user", "content": (self._MARKER + "user hello ") * 64}]
        plain = [{"role": "user", "content": ("user hello ") * 64}]
        assert wire(raw) > wire(plain), "the neutralised marker is not being charged"

    def test_the_invariant_survives_a_prompt_full_of_markers(self):
        for markers in (0, 32, 64, 200):
            backend = _backend(window = 16384, total = 16384, slots = 4)
            text = "explain this template: " + (self._MARKER + "user hello ") * markers
            payload = _chat(text, max_tokens = 16384)
            bound = _enforced(payload, backend)
            assert bound is not None
            from routes.inference import _openai_llama_admission_wire_prompt_tokens as wire

            sent = wire([{"role": "user", "content": text}])
            assert (sent + bound) * 4 < 16384, f"{markers} markers occupy {(sent + bound) * 4}"


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
        """Under --no-kv-unified the aggregate is N times the window, so a share IS the window and
        no request can overrun anyone else.
        """
        backend = _backend(window = 4096, total = 16384, slots = 4)
        assert _enforced(_chat(max_tokens = 4096), backend) is None


class TestWhereAStatedCapStopsBeingStated:
    """The line the docstring draws, pinned: only a cap STRICTLY BELOW the window is a
    promise to write less than the window. At or above it the caller has promised
    nothing the window did not already say, and ``_openai_llama_admission_tokens``
    charges such a request the unstated allowance, so the wire has to be bounded to
    match or the charge is fiction again."""

    def test_one_token_below_the_window_is_left_alone(self):
        backend = _backend(window = 16384, total = 16384, slots = 4)
        assert _enforced(_chat(max_tokens = 16383), backend) is None

    def test_at_or_above_the_window_is_enforced_like_an_unstated_cap(self):
        backend = _backend(window = 16384, total = 16384, slots = 4)
        unstated = _enforced(_chat(), backend)
        assert _enforced(_chat(max_tokens = 16384), backend) == unstated
        assert _enforced(_chat(max_tokens = 999999), backend) == unstated


class TestTheEdges:
    # The window-sized bound this class measures against is the opted-in one.
    pytestmark = pytest.mark.usefixtures("preemption_opted_in")

    def test_a_prompt_that_fills_the_window_still_gets_a_token(self):
        """Zero would be refused upstream, so the floor is one."""
        backend = _backend(window = 16384, total = 16384, slots = 4)
        enforced = _enforced(_chat("word " * 20000, max_tokens = 16384), backend)
        assert enforced == 1

    def test_the_bound_deliberately_exceeds_the_charge(self):
        """They diverge ON PURPOSE now, and that is the whole design."""
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
    """The bound has to land on the request and nowhere else."""

    def _source(self):
        from pathlib import Path

        import core.inference.llama_cpp as llama_cpp
        return Path(llama_cpp.__file__).read_text()

    def test_both_payload_sites_apply_the_allowance(self):
        source = self._source()
        applied = source.count(
            'payload["max_tokens"] = min(payload["max_tokens"], admission_output_allowance)'
        )
        assert (
            applied == 2
        ), f"expected the plain stream and the tool loop to bound the wire cap, found {applied}"

    def test_the_loop_budget_never_sees_it(self):
        """`_loop_budget_left` answers "did the CALLER cap this", and an admission bound is not the
        caller speaking.
        """
        lines = self._source().split("\n")
        start = next(i for i, l in enumerate(lines) if "def _loop_budget_left" in l)
        indent = len(lines[start]) - len(lines[start].lstrip())
        body = []
        for line in lines[start + 1 :]:
            if line.strip() and (len(line) - len(line.lstrip())) <= indent:
                break
            body.append(line)
        assert body, "could not read the body of _loop_budget_left"
        assert "admission_output_allowance" not in "\n".join(
            body
        ), "the admission bound leaked into the caller's continuation budget"

    def test_both_entry_points_accept_it(self):
        import inspect

        from core.inference.llama_cpp import LlamaCppBackend
        for name in ("generate_chat_completion", "generate_chat_completion_with_tools"):
            params = inspect.signature(getattr(LlamaCppBackend, name)).parameters
            assert "admission_output_allowance" in params, name
            assert params["admission_output_allowance"].default is None, name


class TestChargedAndPermittedCannotDrift:
    """The bound is only safe if nothing is admitted on less than it may use."""

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

        Every prompt here fits its share WITH the wire reserve still in it. One inside that
        band does not fit it, takes the flat allowance and is charged more than a share on
        purpose, so the queue admits fewer of them; that is the class above's subject.
        """
        for budget, slots in ((16384, 4), (4096, 4), (2048, 2), (32768, 8), (262144, 4)):
            share = budget // slots
            for prompt in (1, 8, share // 2, share - _RESERVE - 2):
                if prompt < 1:
                    continue
                charged = self._charged(budget, share, prompt)
                # It must still be cheap enough that a full capacity fits, which is the
                # property that actually bounds how many chats are admitted at once.
                assert (
                    charged * slots <= budget or charged <= share
                ), f"budget={budget} slots={slots} prompt={prompt}: charged {charged}"
                # And it must not silently become the permission again.
                assert charged < prompt + max(1, budget - prompt)

    def test_a_full_capacity_of_unstated_requests_still_fits(self):
        """Charging the whole share must not cost the concurrency #10070 bought."""
        for budget, slots in ((16384, 4), (4096, 4), (32768, 8), (262144, 4)):
            share = budget // slots
            charged = self._charged(budget, share, 8)
            assert (
                charged * slots <= budget
            ), f"budget={budget} slots={slots}: {slots} small chats charge {charged * slots}"


class TestWhenNothingWillReclaim:
    """The window is the ceiling only while preemption can reclaim the overcommit."""

    def test_the_switch_brings_the_share_back(self, monkeypatch):
        monkeypatch.setenv("UNSLOTH_LLAMA_ADMISSION_PREEMPT", "0")
        backend = _backend(window = 16384, total = 16384, slots = 4)
        prompt = _prompt_tokens(_chat("hi"))
        assert _enforced(_chat("hi"), backend) == 4096 - prompt - _RESERVE

    def test_with_the_switch_on_the_window_is_offered(self, monkeypatch):
        monkeypatch.setenv("UNSLOTH_LLAMA_ADMISSION_PREEMPT", "1")
        backend = _backend(window = 16384, total = 16384, slots = 4)
        prompt = _prompt_tokens(_chat("hi"))
        assert _enforced(_chat("hi"), backend) == 16384 - prompt - _RESERVE

    def test_four_shares_fit_the_cache(self, monkeypatch):
        monkeypatch.setenv("UNSLOTH_LLAMA_ADMISSION_PREEMPT", "0")
        backend = _backend(window = 16384, total = 16384, slots = 4)
        prompt = _prompt_tokens(_chat("hi"))
        assert 4 * (prompt + _enforced(_chat("hi"), backend)) <= 16384
