# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A reservation nobody enforces is not a reservation: what each chat is charged, what
it is then permitted on the wire, and why the two are deliberately different."""

from types import SimpleNamespace

import pytest

from routes.inference import (
    _OPENAI_LLAMA_ADMISSION_UNSTATED_OUTPUT_TOKENS,
    _openai_llama_admission_budget,
    _openai_llama_admission_enforced_max_tokens,
    _openai_llama_admission_output_allowance,
    _openai_llama_admission_prompt_tokens,
    _openai_llama_admission_tokens,
    _openai_llama_preemption_will_apply,
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
    # ``_kv_cache_unified`` as the real backend sets it: the window is offered only while
    # preemption can reclaim, and that needs one shared pool.
    return SimpleNamespace(
        context_length = window,
        _kv_cache_context_total = total,
        effective_parallel_slots = slots,
        _kv_cache_unified = unified,
    )


def _enforced(payload, backend):
    return _openai_llama_admission_enforced_max_tokens(payload, request = None, llama_backend = backend)


def _prompt_tokens(payload):
    return _openai_llama_admission_prompt_tokens(payload) or 0


class TestEveryChatIsPermittedItsWholeWindow:
    """The invariant MOVED. It is no longer arithmetic, it is eviction."""

    @pytest.mark.parametrize("total", [2048, 4096, 8192, 16384, 65536, 262144])
    def test_no_request_may_exceed_its_own_window(self, total):
        backend = _backend(window = total, total = total, slots = 4)
        payload = _chat(max_tokens = total)
        enforced = _enforced(payload, backend)
        assert enforced is not None
        assert (
            _prompt_tokens(payload) + enforced <= total
        ), f"{total}: a single request may occupy more than the whole window"

    def test_the_whole_window_is_offered_not_a_share_whatever_the_slot_count(self):
        enforced = _enforced(_chat(max_tokens = 16384), _backend(window = 16384, total = 16384, slots = 4))
        assert enforced > 16384 // 4 * 3, f"permitted {enforced} still looks like a share"
        for slots in (2, 3, 4, 8):
            backend = _backend(window = 32768, total = 32768, slots = slots)
            payload = _chat(max_tokens = 32768)
            enforced = _enforced(payload, backend)
            assert _prompt_tokens(payload) + enforced <= 32768
            # And unchanged by how many slots exist: the window is the window.
            assert enforced > 32768 // max(2, slots) * 1.5

    @pytest.mark.parametrize(
        ("payload", "backend"),
        [
            (_chat(max_tokens = 512), _backend(window = 16384, total = 16384, slots = 4)),
            (_chat(max_completion_tokens = 2048), _backend(window = 16384, total = 16384, slots = 4)),
            (_chat(max_tokens = 16384), _backend(window = 16384, total = 16384, slots = 1)),
            (_Payload(max_tokens = 16384), _backend(window = 16384, total = 16384, slots = 4)),
            (_chat(max_tokens = 4096), _backend(window = 4096, total = 16384, slots = 4)),
            (
                _chat(max_tokens = 4096),
                SimpleNamespace(context_length = None, effective_parallel_slots = 4),
            ),
        ],
    )
    def test_a_stated_cap_a_single_slot_a_private_cache_and_an_unknown_budget_are_left_alone(
        self, payload, backend
    ):
        assert _enforced(payload, backend) is None

    def test_the_bound_exceeds_the_charge_and_the_charge_never_exceeds_the_bound(self):
        backend = _backend(window = 16384, total = 16384, slots = 4)
        payload = _chat(max_tokens = 16384)
        charged = _openai_llama_admission_tokens(
            payload,
            budget = _openai_llama_admission_budget(backend),
            capacity = 4,
            context_window = 16384,
        )
        permitted = _prompt_tokens(payload) + _enforced(payload, backend)
        assert permitted > charged, (
            f"permitted {permitted} should exceed the charge {charged}: admission reserves "
            "a share so several chats fit, while each may use the window"
        )
        assert charged <= permitted
        assert _enforced(payload, backend) > _OPENAI_LLAMA_ADMISSION_UNSTATED_OUTPUT_TOKENS


class TestChargedAndPermittedCannotDrift:
    """The bound is only safe if nothing is admitted on less than it may use."""

    def _charged(self, budget, share, prompt):
        allowance = _openai_llama_admission_output_allowance(
            None, budget = budget, prompt_tokens = prompt, context_window = budget, share = share
        )
        return max(1, min(budget, prompt + allowance))

    def test_the_mixed_set_that_broke_the_invariant(self):
        budget, slots = 262144, 4
        share = budget // slots
        admitted, used = [], 0
        for prompt in (1, 65537, 189139, 1):
            charged = self._charged(budget, share, prompt)
            if len(admitted) < slots and used + charged <= budget:
                used += charged
                admitted.append(prompt)
        # Against what the wire ACTUALLY permits, which is the window, not the share.
        permitted = sum(prompt + max(1, budget - prompt) for prompt in admitted)
        assert permitted > budget, (
            "the cache is meant to be overcommitted now; if this ever holds, admission has "
            "gone back to dividing the window and preemption has nothing left to do"
        )
        assert used <= budget, f"admitted {admitted} charged {used} of {budget}"
        assert len(admitted) <= slots

    @pytest.mark.parametrize(
        ("budget", "slots"), [(16384, 4), (4096, 4), (2048, 2), (32768, 8), (262144, 4)]
    )
    def test_the_charge_is_less_than_the_permission_and_a_full_capacity_still_fits(
        self, budget, slots
    ):
        share = budget // slots
        for prompt in (1, 8, share // 2, share - 2):
            if prompt < 1:
                continue
            charged = self._charged(budget, share, prompt)
            # Cheap enough that a full capacity fits, which is what bounds how many chats
            # are admitted at once.
            assert (
                charged * slots <= budget or charged <= share
            ), f"budget={budget} slots={slots} prompt={prompt}: charged {charged}"
            assert charged < prompt + max(1, budget - prompt)
        assert self._charged(budget, share, 8) * slots <= budget


class TestWhenNothingWillReclaim:
    """The window is the ceiling only while preemption can reclaim the overcommit."""

    def _backend(self):
        return _backend(window = 16384, total = 16384, slots = 4)

    def test_with_the_switch_on_the_window_is_offered(self, monkeypatch):
        monkeypatch.setenv("UNSLOTH_LLAMA_ADMISSION_PREEMPT", "1")
        prompt = _prompt_tokens(_chat("hi"))
        assert _enforced(_chat("hi"), self._backend()) == 16384 - prompt

    def test_an_unpausable_request_is_held_to_its_share_while_the_switch_is_on(self, monkeypatch):
        # never chosen as a victim, so nothing reclaims what it generates past its charge
        monkeypatch.setenv("UNSLOTH_LLAMA_ADMISSION_PREEMPT", "1")
        prompt = _prompt_tokens(_chat("hi"))
        enforced = _openai_llama_admission_enforced_max_tokens(
            _chat("hi"), request = None, llama_backend = self._backend(), pausable = False
        )
        assert enforced == 4096 - prompt

    def test_the_switch_off_brings_the_share_back_and_four_of_them_fit(self, monkeypatch):
        monkeypatch.setenv("UNSLOTH_LLAMA_ADMISSION_PREEMPT", "0")
        prompt = _prompt_tokens(_chat("hi"))
        enforced = _enforced(_chat("hi"), self._backend())
        assert enforced == 4096 - prompt
        assert 4 * (prompt + enforced) <= 16384


# ------------------------------------------------------- a client-stated cap passes through

BUDGET, SLOTS = 16384, 4
SHARE = BUDGET // SLOTS


def _charged(cap, prompt, *, active):
    return _openai_llama_admission_output_allowance(
        cap,
        budget = BUDGET,
        prompt_tokens = prompt,
        context_window = BUDGET,
        share = SHARE,
        preemption_active = active,
    )


class TestAStatedCapNoLongerSerialises:
    def test_four_chats_fit_where_one_did(self):
        prompt, cap = 3000, 6000
        before = _charged(cap, prompt, active = False)
        assert before == cap, "the old behaviour was to charge the cap in full"
        assert (
            BUDGET // (prompt + before) == 1
        ), "which is why four chats at max_tokens 6000 ran one at a time"
        after = _charged(cap, prompt, active = True)
        assert (
            BUDGET // (prompt + after) >= SLOTS
        ), f"charged {after}, so only {BUDGET // (prompt + after)} of {SLOTS} fit"
        assert after == _charged(
            None, prompt, active = True
        ), "a stated cap is charged the same as an unstated one"

    def test_the_cases_that_must_not_change(self):
        assert _charged(50, 200, active = True) == _charged(50, 200, active = False) == 50
        for prompt in (1, 200, 1000, 3000):
            # Pausable: the flat allowance. Unpausable: the rest of the share, which is the cap
            # it is sent, so its reservation covers what it may generate.
            assert _charged(None, prompt, active = True) == min(
                _OPENAI_LLAMA_ADMISSION_UNSTATED_OUTPUT_TOKENS, SHARE - prompt
            )
            assert _charged(None, prompt, active = False) == SHARE - prompt
        for cap in (BUDGET, BUDGET + 1):
            for active in (True, False):
                assert _charged(cap, 3000, active = active) == _charged(
                    None, 3000, active = active
                ), "a cap at or above the window was already unstated"
        assert _charged(6000, BUDGET - 1, active = True) >= 1, "the charge is never zero"
        assert _charged(1, BUDGET - 1, active = True) >= 1

    def test_a_full_capacity_is_still_admitted_and_the_cache_is_deliberately_overcommitted(self):
        prompt = 1000
        charged = _charged(6000, prompt, active = True)
        assert (prompt + charged) * SLOTS <= BUDGET, "a full capacity must still be admitted"
        assert (
            (prompt + (BUDGET - prompt)) * SLOTS > BUDGET
        ), "the cache is meant to be overcommitted now; preemption is the enforcement"


class TestTheGateIsTheEnforcementItself:
    """The optimism must be switched on by exactly what makes it survivable."""

    class _Backend:
        def __init__(self, unified):
            self._kv_cache_unified = unified

    def test_no_kv_unified_and_no_budget_mean_no_optimism(self):
        # Without one shared pool a paused slot's cells cannot be purged for anyone else;
        # try_clear_idle_slots is gated on exactly this.
        assert _openai_llama_preemption_will_apply(self._Backend(False), BUDGET) is False
        assert _openai_llama_preemption_will_apply(self._Backend(True), 0) is False
        assert _openai_llama_preemption_will_apply(self._Backend(True), None) is False

    @pytest.mark.parametrize(
        "switch",
        [
            "UNSLOTH_LLAMA_ADMISSION_PREEMPT",
            "UNSLOTH_LLAMA_ADMISSION_KV_BUDGET",
            "UNSLOTH_LLAMA_ADMISSION_CONTROL",
        ],
    )
    def test_every_switch_that_turns_the_enforcement_off_turns_the_optimism_off(
        self, monkeypatch, switch
    ):
        backend = self._Backend(True)
        assert _openai_llama_preemption_will_apply(backend, BUDGET) is True
        monkeypatch.setenv(switch, "0")
        assert _openai_llama_preemption_will_apply(backend, BUDGET) is False
