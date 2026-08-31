# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""An uncapped request must not reserve the whole KV cache (#9955).

Token accounting closed a real overcommit (#9392), but a request naming no
``max_tokens`` is forwarded as ``max_tokens = backend_ctx``, so it is charged the
rest of the window and every other caller queues behind it -- "SLOTS 1/6 busy - 4
queued" on the reporter's card, from the ordinary clients that send no cap.

The fix resolves the omitted cap to the slot's share of the cache before the
request is reserved or forwarded. These cover both halves: the size of the
resolved cap, and the reservation it produces.
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


def _resolve(
    payload,
    backend = None,
    injected = 0,
):
    return routes_inference._openai_llama_uncapped_max_tokens(
        payload,
        request = None,
        llama_backend = backend if backend is not None else _backend(),
        injected_prompt_tokens = injected,
    )


def _cap(payload, backend = None):
    resolved = _resolve(payload, backend)
    return None if resolved is None else resolved.max_tokens


def _cap_with_injection(
    payload,
    injected,
    backend = None,
):
    resolved = _resolve(payload, backend, injected)
    return None if resolved is None else resolved.max_tokens


def _headroom(share = SHARE):
    """The part of a share the resolver leaves for the rendered chat template."""
    return max(
        routes_inference._OPENAI_LLAMA_UNCAPPED_MIN_SHARE_HEADROOM_TOKENS,
        share // routes_inference._OPENAI_LLAMA_UNCAPPED_SHARE_HEADROOM_DIVISOR,
    )


HEADROOM = _headroom()


def _charged(payload):
    """What the cap is sized against: a bound, not the rate admission charges."""
    return routes_inference._openai_llama_admission_prompt_tokens(payload, strict = True)


def _cost(payload, extra = 0):
    return routes_inference._openai_llama_admission_tokens(
        payload,
        budget = CTX,
        capacity = SLOTS,
        extra_prompt_tokens = extra,
    )


def _run(coro):
    return asyncio.run(coro)


class TestTheResolvedCap:
    def test_an_uncapped_request_is_sized_to_its_slots_share(self):
        payload = _uncapped()
        assert _cap(payload) == SHARE - HEADROOM - _charged(payload)

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
        payload = _uncapped([{"role": "user", "content": "x" * SHARE}])
        assert _cap(payload) is None

    def test_the_floor_is_the_boundary(self):
        """One token either side of the minimum, so the guard cannot silently invert."""
        minimum = routes_inference._OPENAI_LLAMA_UNCAPPED_MIN_OUTPUT_TOKENS
        for room, expected in ((minimum, minimum), (minimum - 1, None)):
            payload = _uncapped()
            share = _charged(payload) + room + _headroom(0)
            backend = _backend(context_length = share * SLOTS)
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


class TestPromptInjectedAfterTheSizing:
    """The share has to hold what the route adds after the cap is chosen: the standard
    GGUF path prefixes the current date and sends that, and six slots each overrunning
    by an uncharged injection are the overcommit #9392 closed."""

    def test_the_injection_comes_out_of_the_cap(self):
        payload = _uncapped()
        assert _cap_with_injection(payload, 40) == SHARE - HEADROOM - _charged(payload) - 40

    def test_the_slot_still_holds_one_share(self):
        """What llama-server is actually asked for -- injected prompt plus the cap --
        stays inside the share, so N of them still fit the cache."""
        payload = _uncapped()
        assert _charged(payload) + 40 + _cap_with_injection(payload, 40) == SHARE - HEADROOM

    def test_the_reservation_covers_the_injection_too(self):
        """Charging the cap alone reserves less than the request occupies."""
        payload = _uncapped()
        resolved = _resolve(payload, injected = 40)
        payload.max_tokens = resolved.max_tokens
        assert _cost(payload, resolved.extra_prompt_tokens) == SHARE - HEADROOM

    def test_an_injection_that_eats_the_answer_is_left_alone(self):
        """Same floor as any other prompt that fills the share."""
        payload = _uncapped()
        assert _cap_with_injection(payload, SHARE - HEADROOM - _charged(payload)) is None

    def test_the_date_prompt_is_priced_when_it_is_on(self, monkeypatch):
        monkeypatch.setattr(
            routes_inference,
            "current_date_prompt_line",
            lambda **_: "The current date is 2026-08-31.",
        )
        assert routes_inference._openai_llama_uncapped_injected_date_tokens(None) > 0

    def test_nothing_is_charged_when_it_is_off(self, monkeypatch):
        monkeypatch.setattr(routes_inference, "current_date_prompt_line", lambda **_: "")
        assert routes_inference._openai_llama_uncapped_injected_date_tokens(None) == 0


class TestDenseText:
    """The cap hands out every token the estimate calls free, and a hex prompt tokenises
    near one character per token against the four the dense rate charges: sizing on that
    rate would give a slot room its own prompt sits in, N of them at once."""

    def test_a_blob_is_charged_what_it_can_cost(self):
        payload = _uncapped([{"role": "user", "content": "a1b2c3d4" * 256}])
        dense = routes_inference._openai_llama_admission_prompt_tokens(payload)
        assert _charged(payload) > dense * 2
        assert _cap(payload) == SHARE - HEADROOM - _charged(payload)

    def test_the_reservation_still_lands_on_a_share(self):
        """Admission prices at the dense rate, so the difference has to be handed to it."""
        payload = _uncapped([{"role": "user", "content": "a1b2c3d4" * 256}])
        resolved = _resolve(payload)
        payload.max_tokens = resolved.max_tokens
        assert _cost(payload, resolved.extra_prompt_tokens) == SHARE - HEADROOM

    def test_a_blob_that_fills_the_share_keeps_the_whole_window(self):
        """Pessimism costs an answer, never the cache."""
        payload = _uncapped([{"role": "user", "content": "a1b2c3d4" * (SHARE // 4)}])
        assert _cap(payload) is None


class TestWhatTheShareDoesNotHold:
    """Prompt that reaches llama-server without appearing in the payload estimate."""

    def test_a_multibyte_character_is_charged_its_bytes(self):
        """A code point with no merge falls back to a token per UTF-8 byte, so counting
        characters is not a bound."""
        payload = _uncapped([{"role": "user", "content": "\U0001f600" * 64}])
        ascii_payload = _uncapped([{"role": "user", "content": "x" * 64}])
        assert _charged(payload) > _charged(ascii_payload) * 3

    def test_the_headroom_is_left_unspent(self):
        """The rendered chat template is prompt no estimate of the messages can see."""
        payload = _uncapped()
        resolved = _resolve(payload)
        assert SHARE - (_charged(payload) + resolved.max_tokens) == HEADROOM

    def test_a_request_carrying_tools_is_left_alone(self):
        """The non-streaming passthrough re-sends a tools request under one lease, first
        answer and nudge appended, at the same cap, so one lease would hold two shares."""
        payload = _uncapped()
        payload.tools = [{"type": "function", "function": {"name": "get_weather"}}]
        assert _cap(payload) is None


class TestTheReservationItProduces:
    """The half that matters: the cap has to make the reservation land on a share."""

    def test_the_reservation_is_exactly_one_share(self):
        payload = _uncapped()
        resolved = _resolve(payload)
        payload.max_tokens = resolved.max_tokens
        assert _cost(payload, resolved.extra_prompt_tokens) == SHARE - HEADROOM

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
                resolved = _resolve(payload)
                payload.max_tokens = resolved.max_tokens
                reservation = queue.reserve(
                    capacity = SLOTS,
                    config = LlamaAdmissionConfig(),
                    tokens = _cost(payload, resolved.extra_prompt_tokens),
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
                resolved = _resolve(payload)
                payload.max_tokens = resolved.max_tokens
                last = queue.reserve(
                    capacity = SLOTS + 1,
                    config = LlamaAdmissionConfig(),
                    tokens = _cost(payload, resolved.extra_prompt_tokens),
                    budget = CTX,
                )
            return last.lease_nowait()

        assert _run(scenario()) is None
