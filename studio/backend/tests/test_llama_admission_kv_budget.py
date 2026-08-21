# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Admission must count KV tokens, not just serving slots.

The failure these cover happened live. Two chats generating at once against a model
loaded at ``-c 2048``:

    srv send_error: task id = 1101, error: Context size has been exceeded.
    slot   release: id 0 | task 1101 | n_tokens = 565,  truncated = 0
    srv send_error: task id = 714,  error: Context size has been exceeded.
    slot   release: id 1 | task 714  | n_tokens = 1485, truncated = 0

565 + 1485 = 2050 against a 2048-token cache, ``truncated = 0`` on both: neither request
was too long on its own. llama.cpp killed both tasks, taking a chat reply and a Deep
Research run with it.

The cause is that ``--parallel 4 --kv-unified`` allocates ONE cache of ``n_ctx`` and then
reports ``n_ctx_slot = n_ctx`` to every slot, so four generations can each be admitted
believing they own the whole window.
"""

import asyncio

import pytest

from core.inference.llama_admission import (
    DEFAULT_ADMISSION_KV_BUDGET,
    LlamaAdmissionConfig,
    LlamaAdmissionQueue,
)


def _config(**overrides):
    return LlamaAdmissionConfig(**overrides)


async def _reserve(
    queue,
    *,
    capacity,
    tokens,
    budget,
    config = None,
):
    return queue.reserve(
        capacity = capacity,
        config = config or _config(),
        tokens = tokens,
        budget = budget,
    )


def _run(coro):
    return asyncio.run(coro)


class TestTheBudgetIsEnforced:
    def test_two_requests_that_together_overflow_the_cache_do_not_both_run(self):
        """The live failure, in miniature. 1500 + 1500 against 2048."""

        async def scenario():
            queue = LlamaAdmissionQueue("test")
            first = await _reserve(queue, capacity = 4, tokens = 1500, budget = 2048)
            lease = first.lease_nowait()
            assert lease is not None, "the first request owns the cache"
            second = await _reserve(queue, capacity = 4, tokens = 1500, budget = 2048)
            # A slot is free, but the cache is not. Before token accounting this
            # returned a lease and llama.cpp killed both tasks.
            assert second.lease_nowait() is None
            return queue, lease, second

        queue, lease, second = _run(scenario())
        assert queue.snapshot().committed == 1500

    def test_small_requests_still_run_concurrently(self):
        """The regression guard: this must not become "one request at a time"."""

        async def scenario():
            queue = LlamaAdmissionQueue("test")
            leases = []
            for _ in range(4):
                reservation = await _reserve(queue, capacity = 4, tokens = 400, budget = 2048)
                leases.append(reservation.lease_nowait())
            return leases

        leases = _run(scenario())
        assert all(lease is not None for lease in leases), "4 x 400 fits in 2048"

    def test_a_lone_oversized_request_is_admitted_rather_than_stranded(self):
        """It will be refused by llama-server, with a message naming both counts.
        Refusing it here would strand it forever, since nothing else is running."""

        async def scenario():
            queue = LlamaAdmissionQueue("test")
            reservation = await _reserve(queue, capacity = 4, tokens = 3000, budget = 2048)
            return reservation.lease_nowait()

        assert _run(scenario()) is not None

    def test_releasing_returns_the_tokens(self):
        async def scenario():
            queue = LlamaAdmissionQueue("test")
            first = await _reserve(queue, capacity = 4, tokens = 1500, budget = 2048)
            lease = first.lease_nowait()
            assert queue.snapshot().committed == 1500
            lease.release()
            assert queue.snapshot().committed == 0
            # And the cache is available again.
            second = await _reserve(queue, capacity = 4, tokens = 1500, budget = 2048)
            return second.lease_nowait()

        assert _run(scenario()) is not None

    def test_a_double_release_returns_the_tokens_only_once(self):
        """A second subtraction would drive the pool negative and let the budget
        admit callers the cache cannot hold."""

        async def scenario():
            queue = LlamaAdmissionQueue("test")
            first = await _reserve(queue, capacity = 4, tokens = 1000, budget = 2048)
            lease = first.lease_nowait()
            lease.release()
            lease.release()
            return queue.snapshot()

        assert _run(scenario()).committed == 0


class TestBackwardsCompatibility:
    def test_no_budget_reproduces_slot_only_admission(self):
        """Every caller that does not pass a budget must behave exactly as before."""

        async def scenario():
            queue = LlamaAdmissionQueue("test")
            leases = []
            for _ in range(4):
                reservation = queue.reserve(capacity = 4, config = _config())
                leases.append(reservation.lease_nowait())
            return leases

        leases = _run(scenario())
        assert all(lease is not None for lease in leases)

    def test_the_env_flag_off_restores_slot_only_admission(self):
        async def scenario():
            queue = LlamaAdmissionQueue("test")
            config = _config(kv_budget = False)
            first = await _reserve(
                queue,
                capacity = 4,
                tokens = 1500,
                budget = 2048,
                config = config,
            )
            assert first.lease_nowait() is not None
            second = await _reserve(
                queue,
                capacity = 4,
                tokens = 1500,
                budget = 2048,
                config = config,
            )
            return second.lease_nowait()

        assert _run(scenario()) is not None, "the escape hatch must overcommit as before"

    def test_token_accounting_is_on_by_default(self):
        assert DEFAULT_ADMISSION_KV_BUDGET is True
        assert _config().kv_budget is True

    def test_a_zero_budget_disables_the_check(self):
        async def scenario():
            queue = LlamaAdmissionQueue("test")
            first = await _reserve(queue, capacity = 4, tokens = 5000, budget = 0)
            assert first.lease_nowait() is not None
            second = await _reserve(queue, capacity = 4, tokens = 5000, budget = 0)
            return second.lease_nowait()

        assert _run(scenario()) is not None


class TestTheRouteHelpers:
    def test_the_budget_is_the_backends_own_context_length(self):
        from types import SimpleNamespace

        import routes.inference as routes_inference

        backend = SimpleNamespace(context_length = 2048)
        assert routes_inference._openai_llama_admission_budget(backend) == 2048

    def test_an_unreadable_context_length_means_no_budget(self):
        from types import SimpleNamespace

        import routes.inference as routes_inference
        for value in (None, 0, -1, "nonsense"):
            backend = SimpleNamespace(context_length = value)
            assert routes_inference._openai_llama_admission_budget(backend) is None

    def test_the_cost_is_the_prompt_plus_the_output_allowance(self):
        from types import SimpleNamespace

        import routes.inference as routes_inference

        payload = SimpleNamespace(
            messages = [{"role": "user", "content": "x" * 4000}],
            max_tokens = 256,
        )
        cost = routes_inference._openai_llama_admission_tokens(
            payload,
            budget = 8192,
            capacity = 4,
        )
        assert cost is not None and cost > 256, "the prompt must be counted, not just the output"

    def test_the_cost_is_clamped_to_the_budget(self):
        from types import SimpleNamespace

        import routes.inference as routes_inference

        payload = SimpleNamespace(
            messages = [{"role": "user", "content": "x" * 100_000}],
            max_tokens = 4096,
        )
        cost = routes_inference._openai_llama_admission_tokens(
            payload,
            budget = 2048,
            capacity = 4,
        )
        # Clamped so the queue admits it alone rather than stranding it.
        assert cost == 2048

    def test_a_shape_with_no_messages_reserves_a_fair_share(self):
        from types import SimpleNamespace

        import routes.inference as routes_inference

        payload = SimpleNamespace(prompt = "raw completion text", max_tokens = 128)
        cost = routes_inference._openai_llama_admission_tokens(
            payload,
            budget = 2048,
            capacity = 4,
        )
        # Not the whole budget (that would serialise /completions) and not nothing
        # (that would restore the overcommit).
        assert cost == 512

    def test_no_budget_means_no_cost(self):
        from types import SimpleNamespace

        import routes.inference as routes_inference

        payload = SimpleNamespace(messages = [{"role": "user", "content": "hi"}], max_tokens = 8)
        assert (
            routes_inference._openai_llama_admission_tokens(
                payload,
                budget = None,
                capacity = 4,
            )
            is None
        )


class TestParkedLeasesStillHoldTheirKV:
    """A parked holder gives its SLOT back, not its cache.

    `try_park` hands the slot to the pool while its holder waits on a tool approval, so
    `_held` drops to zero even though llama-server still holds that lease's KV. The
    deadlock escape used to ask whether a slot was held, which made a parked lease look
    like an empty backend and admitted the next caller unconditionally.
    """

    def test_parking_does_not_reopen_the_whole_cache(self):
        async def scenario():
            queue = LlamaAdmissionQueue("test")
            first = await _reserve(queue, capacity = 4, tokens = 1500, budget = 2048)
            lease = first.lease_nowait()
            assert lease is not None
            assert lease.park() is True, "the park budget must allow this"
            # The slot is back, the KV is not.
            assert queue.snapshot().committed == 1500
            second = await _reserve(queue, capacity = 4, tokens = 1500, budget = 2048)
            return queue, second.lease_nowait()

        queue, lease = _run(scenario())
        assert lease is None, "1500 + 1500 against 2048 must not both be admitted"
        assert queue.snapshot().committed == 1500

    def test_a_caller_is_still_admitted_when_nothing_is_committed(self):
        """The escape must survive the fix, or a large lone request deadlocks."""

        async def scenario():
            queue = LlamaAdmissionQueue("test")
            reservation = await _reserve(queue, capacity = 4, tokens = 9999, budget = 2048)
            return reservation.lease_nowait()

        assert _run(scenario()) is not None


class TestTheOutputAllowanceIsCounted:
    def test_max_completion_tokens_is_reserved_like_max_tokens(self):
        """Generation honours max_completion_tokens through
        _effective_openai_max_tokens; admission must reserve the same allowance."""
        from types import SimpleNamespace

        import routes.inference as routes_inference

        messages = [{"role": "user", "content": "x" * 400}]
        with_deprecated = routes_inference._openai_llama_admission_tokens(
            SimpleNamespace(messages = messages, max_tokens = 512),
            budget = 8192,
            capacity = 4,
        )
        with_supported = routes_inference._openai_llama_admission_tokens(
            SimpleNamespace(messages = messages, max_tokens = None, max_completion_tokens = 512),
            budget = 8192,
            capacity = 4,
        )
        assert with_supported == with_deprecated

    def test_a_responses_shape_would_have_fallen_back_to_a_fair_share(self):
        """Why the /v1/responses site now reserves against the translated chat_req: the
        raw model has `input` and `max_output_tokens`, so nothing here can size it."""
        from types import SimpleNamespace

        import routes.inference as routes_inference

        raw = SimpleNamespace(input = "x" * 100_000, max_output_tokens = 4096)
        assert (
            routes_inference._openai_llama_admission_tokens(
                raw,
                budget = 2048,
                capacity = 4,
            )
            == 512
        )


class TestTheWholeRenderedPromptIsCounted:
    """Two more ways the reservation undercounted, both from review.

    An uncapped request reserved no output allowance even though
    `_build_passthrough_payload` then sends `max_tokens = backend_ctx`, so short
    prompts held tiny commitments while each generation could fill the cache. And
    the estimate covered only `messages`, while OpenAI tool definitions are
    rendered into the prompt and Anthropic keeps `system` and `tools` separate
    until they are translated.
    """

    @staticmethod
    def _cost(
        payload,
        budget = 8192,
        capacity = 4,
    ):
        import routes.inference as routes_inference
        return routes_inference._openai_llama_admission_tokens(
            payload,
            budget = budget,
            capacity = capacity,
        )

    def test_an_uncapped_request_reserves_the_rest_of_the_window(self):
        from types import SimpleNamespace
        payload = SimpleNamespace(
            messages = [{"role": "user", "content": "hi"}],
            max_tokens = None,
            max_completion_tokens = None,
        )
        # Generation may run until the window is full, so the reservation says so.
        assert self._cost(payload, budget = 2048) == 2048

    def test_two_uncapped_short_prompts_do_not_both_run(self):
        """The collision this closes: tiny commitments, cache-filling generations."""
        from types import SimpleNamespace

        async def scenario():
            queue = LlamaAdmissionQueue("test")
            payload = SimpleNamespace(
                messages = [{"role": "user", "content": "hi"}],
                max_tokens = None,
                max_completion_tokens = None,
            )
            cost = self._cost(payload, budget = 2048)
            first = await _reserve(queue, capacity = 4, tokens = cost, budget = 2048)
            assert first.lease_nowait() is not None
            second = await _reserve(queue, capacity = 4, tokens = cost, budget = 2048)
            return second.lease_nowait()

        assert _run(scenario()) is None

    def test_a_capped_request_is_unaffected(self):
        from types import SimpleNamespace
        payload = SimpleNamespace(
            messages = [{"role": "user", "content": "hi"}],
            max_tokens = 128,
            max_completion_tokens = None,
        )
        assert self._cost(payload, budget = 2048) < 2048

    def test_tool_schemas_are_counted(self):
        from types import SimpleNamespace

        messages = [{"role": "user", "content": "hi"}]
        bare = self._cost(SimpleNamespace(messages = messages, max_tokens = 16))
        with_tools = self._cost(
            SimpleNamespace(
                messages = messages,
                max_tokens = 16,
                tools = [
                    {
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "description": "d" * 2000,
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }
                ],
            )
        )
        assert with_tools > bare

    def test_an_anthropic_system_block_is_counted(self):
        from types import SimpleNamespace

        messages = [{"role": "user", "content": "hi"}]
        bare = self._cost(SimpleNamespace(messages = messages, max_tokens = 16))
        with_system = self._cost(
            SimpleNamespace(
                messages = messages,
                max_tokens = 16,
                system = "s" * 4000,
            )
        )
        assert with_system > bare

    def test_an_unserialisable_extra_does_not_break_admission(self):
        from types import SimpleNamespace
        payload = SimpleNamespace(
            messages = [{"role": "user", "content": "hi"}],
            max_tokens = 16,
            tools = object(),
        )
        assert self._cost(payload) is not None


class TestToolLoopsReserveTheirUpperBound:
    """One lease covers up to 25 rounds, each larger than the last.

    `generate_chat_completion_with_tools` appends every tool result and re-sends the
    conversation, so a request that starts small can approach the full window while its
    commitment stays at the opening estimate. Another request is then admitted against a
    cache the active rounds have already grown into.
    """

    @staticmethod
    def _cost(
        payload,
        budget = 2048,
        capacity = 4,
        tool_loop = False,
    ):
        import routes.inference as routes_inference
        return routes_inference._openai_llama_admission_tokens(
            payload,
            budget = budget,
            capacity = capacity,
            tool_loop = tool_loop,
        )

    def test_a_tool_request_reserves_the_whole_cache(self):
        """Keyed on the resolved path, not on ``tools``: the loop also opens on
        ``enable_tools``, ``mcp_enabled``, the CLI policy and a checkpoint repair,
        none of which carry a client catalogue."""
        from types import SimpleNamespace

        payload = SimpleNamespace(
            messages = [{"role": "user", "content": "hi"}],
            max_tokens = 16,
            enable_tools = True,
            tools = None,
        )
        assert self._cost(payload, tool_loop = True) == 2048

    def test_two_tool_requests_do_not_run_together(self):
        from types import SimpleNamespace
        async def scenario():
            queue = LlamaAdmissionQueue("test")
            payload = SimpleNamespace(
                messages = [{"role": "user", "content": "hi"}],
                max_tokens = 16,
                enable_tools = True,
                tools = None,
            )
            cost = self._cost(payload, tool_loop = True)
            first = await _reserve(queue, capacity = 4, tokens = cost, budget = 2048)
            assert first.lease_nowait() is not None
            second = await _reserve(queue, capacity = 4, tokens = cost, budget = 2048)
            return second.lease_nowait()

        assert _run(scenario()) is None

    def test_a_request_without_tools_is_unaffected(self):
        """The serialisation is the price of a tool loop, not of every request."""
        from types import SimpleNamespace

        payload = SimpleNamespace(
            messages = [{"role": "user", "content": "hi"}],
            max_tokens = 16,
            tools = None,
        )
        assert self._cost(payload) < 2048

    def test_an_empty_tool_list_is_not_a_tool_loop(self):
        from types import SimpleNamespace
        payload = SimpleNamespace(
            messages = [{"role": "user", "content": "hi"}],
            max_tokens = 16,
            tools = [],
        )
        assert self._cost(payload) < 2048

    def test_a_forwarded_catalogue_is_not_a_tool_loop(self):
        """The passthrough and streaming /v1/responses run ONE generation per HTTP
        call; the client sends the next round itself, with its own reservation."""
        from types import SimpleNamespace

        payload = SimpleNamespace(
            messages = [{"role": "user", "content": "hi"}],
            max_tokens = 16,
            tools = [{"type": "function", "function": {"name": "lookup"}}],
        )
        assert self._cost(payload) < 2048
