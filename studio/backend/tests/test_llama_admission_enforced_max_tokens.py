# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A reservation nobody enforces is not a reservation.

Admission charged an unstated "Max Tokens: Max" a bounded allowance while the request
sent to llama-server still said the whole window, so the two disagreed by the whole
cache. Measured on 2026-09-01: four tool chats on `-c 16384 --parallel 4 --kv-unified`
were each admitted at their share, all generated into the one shared pool, and
llama-server errored EVERY processing slot at once. Four conversations, lost together.

The bound is the fair share rather than the charged figure. The charge is optimistic on
purpose so more chats fit; the bound only has to be physically safe, and at most
`capacity` requests hold a slot at once.
"""

from types import SimpleNamespace

from routes.inference import (
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


def _enforced(payload, backend):
    return _openai_llama_admission_enforced_max_tokens(
        payload, request = None, llama_backend = backend
    )


class TestTheInvariant:
    """`capacity` concurrent requests cannot together exceed the cache."""

    def test_the_configuration_that_lost_four_chats(self):
        backend = _backend(window = 16384, total = 16384, slots = 4)
        payload = _chat(max_tokens = 16384)
        enforced = _enforced(payload, backend)
        assert enforced is not None
        prompt_plus_output = _prompt_tokens(payload) + enforced
        assert prompt_plus_output * 4 <= 16384, (
            f"four chats may occupy {prompt_plus_output * 4} of a 16384 cache"
        )

    def test_it_holds_at_every_cache_size(self):
        for total in (2048, 4096, 8192, 16384, 65536, 262144):
            backend = _backend(window = total, total = total, slots = 4)
            payload = _chat(max_tokens = total)
            enforced = _enforced(payload, backend)
            occupancy = _prompt_tokens(payload) + (enforced if enforced is not None else total)
            assert occupancy * 4 <= total, f"{total}: four chats occupy {occupancy * 4}"

    def test_it_holds_for_a_long_prompt(self):
        backend = _backend(window = 16384, total = 16384, slots = 4)
        payload = _chat("word " * 600, max_tokens = 16384)
        enforced = _enforced(payload, backend)
        assert enforced is not None
        assert (_prompt_tokens(payload) + enforced) * 4 <= 16384

    def test_it_holds_at_other_slot_counts(self):
        for slots in (2, 3, 4, 8):
            backend = _backend(window = 32768, total = 32768, slots = slots)
            payload = _chat(max_tokens = 32768)
            enforced = _enforced(payload, backend)
            assert enforced is not None
            assert (_prompt_tokens(payload) + enforced) * slots <= 32768


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
    def test_a_prompt_past_its_share_still_gets_a_token(self):
        """Zero would be refused upstream. Such a request is charged more than a share,
        so the queue admits fewer than `capacity` of them and the invariant survives."""
        backend = _backend(window = 16384, total = 16384, slots = 4)
        enforced = _enforced(_chat("word " * 4000, max_tokens = 16384), backend)
        assert enforced == 1

    def test_the_bound_is_not_the_charge(self):
        """Clamping to the charged estimate would cut a chat to about a thousand tokens
        for no safety gain. The charge stays optimistic; the bound stays share-sized."""
        backend = _backend(window = 16384, total = 16384, slots = 4)
        payload = _chat(max_tokens = 16384)
        charged = _openai_llama_admission_tokens(
            payload, budget = 16384, capacity = 4, context_window = 16384
        )
        enforced = _enforced(payload, backend)
        assert enforced > charged, f"bound {enforced} should exceed the charge {charged}"

    def test_the_charge_never_exceeds_what_is_permitted(self):
        """If it did, admission would be reserving room the request cannot use."""
        backend = _backend(window = 16384, total = 16384, slots = 4)
        payload = _chat(max_tokens = 16384)
        charged = _openai_llama_admission_tokens(
            payload, budget = 16384, capacity = 4, context_window = 16384
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
