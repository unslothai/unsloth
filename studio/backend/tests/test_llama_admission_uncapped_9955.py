# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""An uncapped request must not reserve the whole KV cache (#9955).

Token accounting closed a real overcommit (#9392), but it charges a request that
names no ``max_tokens`` the rest of the window, because that request really is
forwarded as ``max_tokens = backend_ctx`` and may generate that much. One such
request therefore commits the entire cache and every other caller queues behind
it. The reporter's API card said it exactly:

    SLOTS   1/6 busy - 4 queued
    IN FLIGHT   5

Six slots, five requests in the building, one of them decoding. The clients that
hit this are the ordinary ones: browser translation, and most OpenAI SDK calls,
none of which send a cap.

The fix resolves the omitted cap to the slot's share of the cache before the
request is either reserved or forwarded, so the reservation is honest rather than
pessimistic and ``--parallel N`` serves N of them at once. These cover both halves
of that: the size of the resolved cap, and the reservation it produces.
"""

import asyncio
from types import SimpleNamespace

from core.inference.llama_admission import LlamaAdmissionConfig, LlamaAdmissionQueue

import routes.inference as routes_inference


CTX = 32768
SLOTS = 6
SHARE = CTX // SLOTS


def _backend(
    context_length = CTX,
    slots = SLOTS,
    total = None,
):
    return SimpleNamespace(
        context_length = context_length,
        effective_parallel_slots = slots,
        _kv_cache_context_total = total,
    )


def _uncapped(messages = None):
    """A request shaped like the reporter's: short prompt, no cap of any kind."""
    return SimpleNamespace(
        messages = messages or [{"role": "user", "content": "Translate: hello world"}],
        max_tokens = None,
        max_completion_tokens = None,
    )


def _cap(payload, backend = None):
    return routes_inference._openai_llama_uncapped_max_tokens(
        payload,
        request = None,
        llama_backend = backend if backend is not None else _backend(),
    )


def _cost(payload):
    return routes_inference._openai_llama_admission_tokens(
        payload,
        budget = CTX,
        capacity = SLOTS,
    )


def _run(coro):
    return asyncio.run(coro)


class TestTheResolvedCap:
    def test_an_uncapped_request_is_sized_to_its_slots_share(self):
        payload = _uncapped()
        prompt = routes_inference._openai_llama_admission_prompt_tokens(payload)
        assert _cap(payload) == SHARE - prompt

    def test_a_single_slot_server_is_left_alone(self):
        """Its share IS the whole cache, so there is nothing to unserialise."""
        assert _cap(_uncapped(), _backend(slots = 1)) is None

    def test_an_unreadable_cache_size_is_left_alone(self):
        assert _cap(_uncapped(), _backend(context_length = None)) is None

    def test_a_request_that_names_its_own_cap_keeps_it(self):
        """The helper only ever answers for an omitted cap; a named one is forwarded
        as sent and may still use the whole window."""
        payload = _uncapped()
        payload.max_tokens = 4096
        assert routes_inference._effective_openai_max_tokens(payload) == 4096

    def test_max_completion_tokens_counts_as_named(self):
        payload = _uncapped()
        payload.max_completion_tokens = 4096
        assert routes_inference._effective_openai_max_tokens(payload) == 4096

    def test_a_prompt_that_fills_the_share_is_left_alone(self):
        """Below a usable answer there is nothing to hand back, so the request keeps
        the whole-window default rather than being answered in a few tokens."""
        payload = _uncapped([{"role": "user", "content": "x" * (SHARE * 4)}])
        assert _cap(payload) is None

    def test_the_floor_is_the_boundary(self):
        """One token either side of the minimum, so the guard cannot silently invert."""
        minimum = routes_inference._OPENAI_LLAMA_UNCAPPED_MIN_OUTPUT_TOKENS
        for room, expected in ((minimum, minimum), (minimum - 1, None)):
            payload = _uncapped()
            prompt = routes_inference._openai_llama_admission_prompt_tokens(payload)
            backend = _backend(context_length = (prompt + room) * SLOTS)
            assert _cap(payload, backend) == expected

    def test_a_shape_with_no_messages_is_left_alone(self):
        """Admission charges an unsizeable shape a whole share, so there is no room
        left to hand its output and the two could not be made to agree."""
        assert _cap(SimpleNamespace(prompt = "raw completion text", max_tokens = None)) is None

    def test_slot_only_admission_is_left_alone(self, monkeypatch):
        """No tokens are charged, so a cap here would only shorten answers."""
        monkeypatch.setenv("UNSLOTH_LLAMA_ADMISSION_KV_BUDGET", "0")
        assert _cap(_uncapped()) is None

    def test_admission_off_is_left_alone(self, monkeypatch):
        monkeypatch.setenv("UNSLOTH_LLAMA_ADMISSION_CONTROL", "0")
        assert _cap(_uncapped()) is None


class TestTheReservationItProduces:
    """The half that matters: the cap has to make the reservation land on a share."""

    def test_the_reservation_is_exactly_one_share(self):
        payload = _uncapped()
        payload.max_tokens = _cap(payload)
        assert _cost(payload) == SHARE

    def test_the_unresolved_reservation_was_the_whole_cache(self):
        """The regression itself, for contrast: uncapped, one request owns everything."""
        assert _cost(_uncapped()) == CTX

    def test_every_slot_is_used(self):
        """1/6 busy becomes 6/6. The reporter's case, end to end."""

        async def scenario():
            queue = LlamaAdmissionQueue("test")
            leases = []
            for _ in range(SLOTS):
                payload = _uncapped()
                payload.max_tokens = _cap(payload)
                reservation = queue.reserve(
                    capacity = SLOTS,
                    config = LlamaAdmissionConfig(),
                    tokens = _cost(payload),
                    budget = CTX,
                )
                leases.append(reservation.lease_nowait())
            return leases

        assert all(lease is not None for lease in _run(scenario()))

    def test_the_cache_is_still_not_overcommitted(self):
        """#9392 stays closed. Given a slot to spare, the seventh share does not fit
        the cache and waits, rather than being admitted onto a cache that cannot hold
        it. The slots are deliberately not the limit here: this is the token budget
        refusing, not the pool."""

        async def scenario():
            queue = LlamaAdmissionQueue("test")
            for _ in range(SLOTS + 1):
                payload = _uncapped()
                payload.max_tokens = _cap(payload)
                last = queue.reserve(
                    capacity = SLOTS + 1,
                    config = LlamaAdmissionConfig(),
                    tokens = _cost(payload),
                    budget = CTX,
                )
            return last.lease_nowait()

        assert _run(scenario()) is None
