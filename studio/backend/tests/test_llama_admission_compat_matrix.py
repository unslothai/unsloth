# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Every backend shape this accounting can be handed, and every old caller of it.

Re-costing changes how a tool loop is admitted, so every install that does NOT get the new
path must behave exactly as it did. Budget and slot count are both derived from the loaded
backend, and that derivation differs per platform and per device:

  * ``capacity`` is ``effective_parallel_slots``, reduced to fit VRAM and forced to 1 on a
    CPU-only or small-VRAM load. A 1-slot backend must be untouched by this change.
  * ``budget`` is ``_kv_cache_context_total``: ``n_ctx`` under ``--kv-unified``,
    ``n_ctx * slots`` without it, None when the context length cannot be read back from
    ``/props``. None means slot-only admission, the pre-#9392 behaviour.

The matrix below is the real portability surface, not a stand-in for other operating
systems: the code added here is stdlib threading and ``time.monotonic`` with no
OS-specific branch, and what varies per platform is which of these numbers arrives.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.inference.llama_admission import (
    DEFAULT_RECOST_WAIT_TIMEOUT_S,
    LlamaAdmissionConfig,
    LlamaAdmissionQueue,
)


def _tokens(payload, *, budget, capacity, tool_loop):
    import routes.inference as routes_inference
    return routes_inference._openai_llama_admission_tokens(
        payload,
        budget = budget,
        capacity = capacity,
        tool_loop = tool_loop,
    )


def _payload(
    *,
    max_tokens = 128,
    tools = True,
    text = "hi",
):
    return SimpleNamespace(
        messages = [{"role": "user", "content": text}],
        max_tokens = max_tokens,
        enable_tools = tools,
        tools = None,
    )


def _lease(
    queue,
    *,
    tokens,
    budget,
    capacity = 4,
    config = None,
):
    reservation = queue.reserve(
        capacity = capacity,
        config = config or LlamaAdmissionConfig(),
        tokens = tokens,
        budget = budget,
    )
    return reservation.lease_nowait()


# (label, slots, n_ctx, kv_unified) -- the shapes a real load actually produces.
BACKENDS = [
    ("cpu-only, 1 slot", 1, 4096, True),
    ("small vram, downshifted", 1, 8192, True),
    ("2 slots unified", 2, 8192, True),
    ("4 slots unified", 4, 4096, True),
    ("8 slots unified", 8, 262144, True),
    ("4 slots NOT unified", 4, 4096, False),
    ("2 slots NOT unified", 2, 8192, False),
]


def _budget(n_ctx, slots, kv_unified):
    """Mirrors _reconcile_effective_ctx_with_server: unified reports one shared cache,
    non-unified reports the aggregate across per-slot windows."""
    return n_ctx * (1 if kv_unified else slots)


@pytest.mark.parametrize("label,slots,n_ctx,kv_unified", BACKENDS)
def test_a_tool_loop_is_never_charged_more_than_the_cache(label, slots, n_ctx, kv_unified):
    budget = _budget(n_ctx, slots, kv_unified)
    cost = _tokens(_payload(), budget = budget, capacity = slots, tool_loop = True)
    assert 1 <= cost <= budget, f"{label}: charged {cost} against a {budget} cache"


@pytest.mark.parametrize("label,slots,n_ctx,kv_unified", BACKENDS)
def test_a_single_slot_backend_behaves_as_it_always_did(label, slots, n_ctx, kv_unified):
    """capacity 1 means share == budget, so a tool loop is charged the whole cache just as
    #9392 charged it. The common shape on CPU-only and on a VRAM-downshifted load."""
    if slots != 1:
        pytest.skip("covered by the multi-slot cases")
    budget = _budget(n_ctx, slots, kv_unified)
    assert _tokens(_payload(), budget = budget, capacity = 1, tool_loop = True) == budget


@pytest.mark.parametrize("label,slots,n_ctx,kv_unified", BACKENDS)
@pytest.mark.asyncio
async def test_the_slots_a_backend_reports_can_all_be_filled(label, slots, n_ctx, kv_unified):
    """On any multi-slot backend, ordinary tool chats fill the slots the backend
    advertises rather than running one at a time."""
    budget = _budget(n_ctx, slots, kv_unified)
    cost = _tokens(_payload(), budget = budget, capacity = slots, tool_loop = True)
    queue = LlamaAdmissionQueue(label)
    admitted = [_lease(queue, tokens = cost, budget = budget, capacity = slots) for _ in range(slots)]
    assert all(
        lease is not None for lease in admitted
    ), f"{label}: only {sum(l is not None for l in admitted)}/{slots} slots usable"
    assert queue.snapshot().committed <= budget


@pytest.mark.parametrize("label,slots,n_ctx,kv_unified", BACKENDS)
def test_max_tokens_on_max_is_not_treated_as_a_promise_to_use_it(label, slots, n_ctx, kv_unified):
    """The UI sends max_tokens = context_length for "Max", which serialises with or
    without tools alike. Fixing that is a separate change; what matters here is that tools
    are not charged MORE than no-tools for the same request."""
    budget = _budget(n_ctx, slots, kv_unified)
    payload = _payload(max_tokens = n_ctx)
    with_tools = _tokens(payload, budget = budget, capacity = slots, tool_loop = True)
    without = _tokens(payload, budget = budget, capacity = slots, tool_loop = False)
    assert with_tools <= max(
        without, budget // max(1, slots)
    ), f"{label}: tools charged {with_tools} vs {without} without"


class TestNothingChangesWhenTheBudgetIsUnknown:
    """``budget`` is None when the context length cannot be read back: the pre-#9392
    slot-only path, which has to stay bit-for-bit what it was."""

    def test_no_budget_means_no_token_cost_at_all(self):
        for tool_loop in (True, False):
            assert _tokens(_payload(), budget = None, capacity = 4, tool_loop = tool_loop) is None
            assert _tokens(_payload(), budget = 0, capacity = 4, tool_loop = tool_loop) is None

    @pytest.mark.asyncio
    async def test_slot_only_admission_still_fills_every_slot(self):
        queue = LlamaAdmissionQueue("no-budget")
        leases = [
            queue.reserve(capacity = 4, config = LlamaAdmissionConfig()).lease_nowait()
            for _ in range(4)
        ]
        assert all(lease is not None for lease in leases)

    @pytest.mark.asyncio
    async def test_recost_waiting_is_a_no_op_without_a_budget(self):
        """It must not block, and must not invent a commitment out of nothing."""
        queue = LlamaAdmissionQueue("no-budget")
        lease = queue.reserve(capacity = 4, config = LlamaAdmissionConfig()).lease_nowait()
        assert lease.recost_waiting(999999, timeout_s = 1.0) is True
        assert queue.snapshot().committed == 0
        assert queue._reparking == 0

    @pytest.mark.asyncio
    async def test_recost_waiting_is_a_no_op_when_admission_is_disabled(self):
        """config.enabled False hands back a lease with no queue behind it."""
        queue = LlamaAdmissionQueue("disabled")
        lease = queue.reserve(capacity = 4, config = LlamaAdmissionConfig(enabled = False)).lease_nowait()
        assert lease.recost_waiting(999999, timeout_s = 1.0) is True
        lease.release()

    @pytest.mark.asyncio
    async def test_the_kv_budget_escape_hatch_still_overcommits(self):
        """UNSLOTH_LLAMA_ADMISSION_KV_BUDGET=0, the documented way out for a backend whose
        reported context length does not match the cache it allocated."""
        queue = LlamaAdmissionQueue("escape")
        config = LlamaAdmissionConfig(kv_budget = False)
        first = _lease(queue, tokens = 1500, budget = 2048, config = config)
        second = _lease(queue, tokens = 1500, budget = 2048, config = config)
        assert first is not None and second is not None
        assert first.recost_waiting(999999, timeout_s = 1.0) is True
        assert queue._reparking == 0


class TestOldCallers:
    """Everything added is keyword-with-default, so code written before this still runs."""

    @pytest.mark.asyncio
    async def test_recost_waiting_without_any_new_keyword(self):
        queue = LlamaAdmissionQueue("compat")
        lease = _lease(queue, tokens = 100, budget = 4096)
        assert lease.recost_waiting(200) is True
        assert queue.snapshot().committed == 200

    @pytest.mark.asyncio
    async def test_the_old_non_blocking_recost_still_exists_and_still_declines(self):
        """Callers that must not block keep the old contract: refuse, never wait."""
        queue = LlamaAdmissionQueue("compat")
        first = _lease(queue, tokens = 3000, budget = 4096)
        second = _lease(queue, tokens = 1000, budget = 4096)
        assert second.recost(4000) is False
        assert queue.snapshot().committed == 4000
        assert queue._reparking == 0, "the non-blocking path must never touch the wait line"

    def test_the_route_recost_helper_accepts_no_cancel_event(self):
        import routes.inference as routes_inference

        # Reservation None is the "not admitted yet" case every call site can hit.
        routes_inference._openai_llama_admission_recost(
            None,
            [{"role": "user", "content": "hi"}],
            request = None,
            llama_backend = SimpleNamespace(context_length = 4096),
        )

    def test_generate_chat_completion_with_tools_still_takes_no_hook(self):
        import inspect

        from core.inference.llama_cpp import LlamaCppBackend

        signature = inspect.signature(LlamaCppBackend.generate_chat_completion_with_tools)
        parameter = signature.parameters["on_conversation_grew"]
        assert parameter.default is None, "the hook must be optional for existing callers"

    def test_the_hook_was_appended_rather_than_inserted(self):
        """No bare ``*`` in this signature, so every parameter is positional-or-keyword and
        inserting one silently rebinds the arguments after it for positional callers, with
        no exception to report it."""
        import inspect

        from core.inference.llama_cpp import LlamaCppBackend

        names = list(
            inspect.signature(LlamaCppBackend.generate_chat_completion_with_tools).parameters
        )
        assert (
            names[-1] == "on_conversation_grew"
        ), f"the hook must be last; signature ends {names[-3:]}"

    def test_the_wait_timeout_has_a_sane_default(self):
        assert DEFAULT_RECOST_WAIT_TIMEOUT_S > 0
        import inspect

        from core.inference.llama_admission import LlamaAdmissionLease

        signature = inspect.signature(LlamaAdmissionLease.recost_waiting)
        assert signature.parameters["timeout_s"].default == DEFAULT_RECOST_WAIT_TIMEOUT_S


class TestNoPersistentStateChanged:
    """An existing install upgrading into this must not need a migration."""

    def test_no_new_environment_variable_is_required(self):
        import os

        from core.inference.llama_admission import llama_admission_config_from_env

        # A completely bare environment must still produce a usable config.
        saved = {k: v for k, v in os.environ.items() if k.startswith("UNSLOTH_")}
        try:
            for key in list(saved):
                os.environ.pop(key, None)
            config = llama_admission_config_from_env()
        finally:
            os.environ.update(saved)
        assert config.enabled is True
        assert config.kv_budget is True

    def test_the_queue_carries_no_serialised_state(self):
        """Nothing about a queue is written to disk, and the slots are the whole of its
        state, so a new field cannot break an old install's stored data."""
        assert "_reparking" in LlamaAdmissionQueue.__slots__
        queue = LlamaAdmissionQueue("fresh")
        assert queue._reparking == 0, "a fresh queue must start with the line open"


class TestTheInjectedToolCatalogueIsCharged:
    """The regression that a live run caught and every unit test missed.

    ``payload.tools`` is what the CLIENT sent. Unsloth's own tool loop resolves Web Search
    and the rest server-side and renders them into the prompt AFTER admission has priced
    the request, so pricing from the payload alone undercounts by the whole catalogue.
    Measured on Qwen3.5-4B-MTP-GGUF, the same user turn is 1716 prompt tokens with tools
    off and 2969 with them on. At ``-c 4096`` that gap is fatal: priced at an equal share
    four tool chats were all admitted and llama.cpp answered every one with ``Context size
    has been exceeded``.
    """

    CATALOG = [
        {
            "type": "function",
            "function": {
                "name": f"tool_{i}",
                "description": "A tool with a realistically wordy description. " * 12,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "What to look up. " * 8},
                        "count": {"type": "integer", "description": "How many. " * 8},
                    },
                    "required": ["query"],
                },
            },
        }
        for i in range(6)
    ]

    def test_the_catalogue_is_added_to_the_estimate(self):
        """Checked on a wide pool, where the floor is small enough that the catalogue sets
        the price. On a narrow one the floor already exceeds the whole request, so masking
        it there is correct."""
        import routes.inference as routes_inference

        budget, capacity = 8192, 16  # share 512, smaller than the catalogue
        payload = _payload(text = "hi")
        without = _tokens(payload, budget = budget, capacity = capacity, tool_loop = True)
        with_catalog = routes_inference._openai_llama_admission_tokens(
            payload,
            budget = budget,
            capacity = capacity,
            tool_loop = True,
            injected_tools = self.CATALOG,
        )
        assert (
            with_catalog > without
        ), f"the catalogue must raise the price: {with_catalog} vs {without}"

    def test_a_catalogue_of_a_realistic_size_is_not_rounded_away(self):
        import routes.inference as routes_inference
        charged = routes_inference._openai_llama_admission_injected_tool_tokens(self.CATALOG)
        assert charged > 500, f"only {charged} tokens charged for a six-tool catalogue"

    @pytest.mark.asyncio
    async def test_four_tool_chats_are_not_admitted_past_a_small_cache(self):
        """The live failure at unit level: four short prompts with a real catalogue cannot
        share 4096 tokens, and admitting all four produced four 500s."""
        import routes.inference as routes_inference

        budget = 4096
        cost = routes_inference._openai_llama_admission_tokens(
            _payload(text = "hi"),
            budget = budget,
            capacity = 4,
            tool_loop = True,
            injected_tools = self.CATALOG,
        )
        queue = LlamaAdmissionQueue("catalog")
        admitted = sum(
            _lease(queue, tokens = cost, budget = budget, capacity = 4) is not None for _ in range(4)
        )
        assert (
            queue.snapshot().committed <= budget
        ), f"admitted {admitted} chats at {cost} each against a {budget} cache"

    def test_no_catalogue_means_no_extra_charge(self):
        """A request that injects nothing must be priced exactly as before."""
        import routes.inference as routes_inference
        for empty in (None, [], ()):
            assert routes_inference._openai_llama_admission_injected_tool_tokens(empty) == 0

    def test_an_unserialisable_catalogue_does_not_break_admission(self):
        import routes.inference as routes_inference
        class Awkward:
            def __repr__(self):
                raise RuntimeError("no")

        # default=str still reaches __repr__, so this must be caught rather than raised.
        assert routes_inference._openai_llama_admission_injected_tool_tokens([Awkward()]) >= 0
