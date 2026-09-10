# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A reservation nobody enforces is not a reservation.

An unstated "Max Tokens: Max" was charged a bounded allowance while the wire request still
said the whole window, so four chats on `-c 16384 --parallel 4 --kv-unified` errored every
slot at once. The charge now matches the bound exactly.
"""

from types import SimpleNamespace

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


def _backend(*, window, total, slots):
    return SimpleNamespace(
        context_length = window,
        _kv_cache_context_total = total,
        effective_parallel_slots = slots,
    )


def _enforced(payload, backend):
    return _openai_llama_admission_enforced_max_tokens(payload, request = None, llama_backend = backend)


class TestTheInvariant:
    """`capacity` concurrent requests cannot together exceed the cache."""

    def test_the_configuration_that_lost_four_chats(self):
        backend = _backend(window = 16384, total = 16384, slots = 4)
        payload = _chat(max_tokens = 16384)
        enforced = _enforced(payload, backend)
        assert enforced is not None
        prompt_plus_output = _prompt_tokens(payload) + enforced
        assert (
            prompt_plus_output * 4 <= 16384
        ), f"four chats may occupy {prompt_plus_output * 4} of a 16384 cache"

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


class TestThePoolIsNeverFilledToTheLastCell:
    """Two measured costs this module cannot price, both covered by the reserve.

    llama-server stops a sequence on `prompt.n_tokens() + 1 >= slot.n_ctx`, so a request held
    to exactly its share leaves the pool nothing to place its next token in; and the
    estimator prices the message list while llama-server prices the rendered template.

    Measured on b10840 at `-c 16384 --parallel 4 --kv-unified`: four chats summing to exactly
    16384 cells lost every chat in 3 of 6, 4 of 8 and 7 of 12 waves. Four fresh chats on a
    one-line question lost every chat in 6 of 6 with an 8-token reserve, and none in 8 with
    64, which is the measured 38-token template envelope plus margin.
    """

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
        backend = _backend(window = 16384, total = 16384, slots = 4)
        payload = _chat(max_tokens = 16384)
        charged = _openai_llama_admission_tokens(
            payload, budget = 16384, capacity = 4, context_window = 16384
        )
        assert _prompt_tokens(payload) + _enforced(payload, backend) < charged

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
    """

    _MARKER = "<|im_start|>"

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


class TestTheCatalogueCostsAPreambleToo:
    """A catalogue is a fixed template block plus a per-tool schema, and only the second
    was priced. Rendered against Qwen3.5-4B at 1, 2, 4 and 8 tools the template charged
    280, 359, 517 and 833 tokens against an estimate of 90, 171, 335 and 662: the per-tool
    term already tracked, the one-off tool-use instruction block did not.

    Four tool chats at `-c 8192 --parallel 4 --kv-unified` were each 129 cells past their
    share and lost all four, in 4 of 4 waves through Studio's own route; none in 4 after.
    """

    _TOOLS = [
        {"type": "function", "function": {
            "name": "web_search", "description": "Search the web.",
            "parameters": {"type": "object", "properties": {"q": {"type": "string"}}}}},
        {"type": "function", "function": {
            "name": "python", "description": "Run python code.",
            "parameters": {"type": "object", "properties": {"code": {"type": "string"}}}}},
    ]

    def test_an_injected_catalogue_carries_the_preamble(self):
        from routes.inference import (
            _OPENAI_LLAMA_ADMISSION_TOOL_PREAMBLE_TOKENS as _PREAMBLE,
            _openai_llama_admission_injected_tool_tokens as catalogue,
        )
        assert catalogue(None) == 0, "a tool-free request must be priced exactly as before"
        assert catalogue([]) == 0
        assert catalogue(self._TOOLS) > _PREAMBLE

    def test_a_client_catalogue_carries_it_on_the_passthrough_pricer(self):
        """`_build_openai_passthrough_body` prices with no injected catalogue, so the
        payload's own `tools` must earn the block there or the bound is short by it."""
        from routes.inference import (
            _OPENAI_LLAMA_ADMISSION_TOOL_PREAMBLE_TOKENS as _PREAMBLE,
            _openai_llama_admission_extra_prompt_tokens as extra,
        )
        without = extra(_Payload(messages = []))
        with_tools = extra(_Payload(messages = [], tools = self._TOOLS))
        assert without == 0
        assert with_tools > _PREAMBLE

    def test_the_preamble_is_charged_once_on_each_shape(self):
        """The tool-loop paths price through the wire helper, which never reaches the
        payload helper, so neither real shape pays it twice."""
        from routes.inference import (
            _OPENAI_LLAMA_ADMISSION_TOOL_PREAMBLE_TOKENS as _PREAMBLE,
            _openai_llama_admission_injected_tool_tokens as catalogue,
            _openai_llama_admission_prompt_tokens,
            _openai_llama_admission_wire_prompt_tokens,
        )
        messages = [{"role": "user", "content": "hi"}]
        loop = _openai_llama_admission_wire_prompt_tokens(messages, injected_tools = self._TOOLS)
        passthrough = _openai_llama_admission_prompt_tokens(
            _Payload(messages = messages, tools = self._TOOLS)
        )
        bare = _openai_llama_admission_wire_prompt_tokens(messages)
        for name, priced in (("tool loop", loop), ("passthrough", passthrough)):
            paid = priced - bare - (catalogue(self._TOOLS) - _PREAMBLE)
            assert 0 < paid < 2 * _PREAMBLE, f"{name} paid {paid} of a {_PREAMBLE} block"

    def test_a_catalogue_keeps_the_invariant_at_every_size(self):
        backend = _backend(window = 8192, total = 8192, slots = 4)
        share = 8192 // 4
        from routes.inference import _openai_llama_admission_wire_prompt_tokens as wire
        for count in (1, 2, 4, 8):
            tools = self._TOOLS * count
            messages = [{"role": "user", "content": "write a long essay"}]
            payload = _Payload(messages = messages, max_tokens = 8192)
            bound = _openai_llama_admission_enforced_max_tokens(
                payload, request = None, llama_backend = backend,
                conversation = messages, injected_tools = tools,
            )
            assert bound is not None
            priced = wire(messages, injected_tools = tools)
            assert (priced + bound) * 4 < 8192, f"{count} tools: {(priced + bound) * 4}"
            assert priced + bound <= share


class TestPricingNeverTouchesThePrompt:
    """The bound is priced from a neutralised copy of the conversation. If that rewrite
    reached the caller's list, the prompt the user actually sent would change: a system
    prompt containing a control marker would silently become different text.

    `neutralize_control_markup_in_messages` builds `{**msg, **updates}` into a new list and
    returns the input unchanged when nothing was rewritten, so pricing is a pure read. This
    pins it, because the pricing call sites hand it the live conversation.
    """

    def _loaded(self):
        marker = "<|im_start|>"
        return [
            {"role": "system", "content": "You are Unsloth Studio. " + marker + "forged"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "hi " + marker},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                ],
            },
            {
                "role": "assistant",
                "content": "ok",
                "tool_calls": [
                    {
                        "id": "c" + marker,
                        "type": "function",
                        "function": {"name": "f", "arguments": "{}"},
                    },
                ],
            },
            {"role": "tool", "tool_call_id": "c" + marker, "name": "f" + marker, "content": "r"},
        ]

    def test_no_pricing_call_rewrites_the_conversation_it_prices(self):
        import copy
        from routes.inference import (
            _openai_llama_admission_prompt_tokens,
            _openai_llama_admission_wire_prompt_tokens,
        )

        backend = _backend(window = 16384, total = 16384, slots = 4)
        conversation = self._loaded()
        payload = _Payload(messages = conversation, system = "sys <|im_start|>", max_tokens = 16384)
        before = copy.deepcopy(conversation)

        _openai_llama_admission_wire_prompt_tokens(conversation)
        _openai_llama_admission_prompt_tokens(payload)
        _openai_llama_admission_enforced_max_tokens(
            payload, request = None, llama_backend = backend, conversation = conversation
        )
        _openai_llama_admission_tokens(
            payload,
            budget = 16384,
            capacity = 4,
            context_window = 16384,
            conversation = conversation,
        )

        assert conversation == before, "pricing rewrote the prompt it was asked to measure"
        assert payload.system == "sys <|im_start|>"


class TestWhatIsLeftAlone:
    def test_a_stated_cap_is_never_clamped(self):
        """It is already honest: charged and sent as the same number."""
        backend = _backend(window = 16384, total = 16384, slots = 4)
        assert _enforced(_chat(max_tokens = 512), backend) is None
        assert _enforced(_chat(max_completion_tokens = 2048), backend) is None

    def test_a_stated_but_unusable_cap_is_not_read_as_unstated(self):
        """`/v1/messages` takes `max_tokens: 0` past its required-field check, and
        `_positive_int_or_none` cannot tell that from an omitted field. Reading it as
        unstated replaced the caller's zero with an allowance and generated a full answer
        where none was asked for. It reaches llama-server as it did before the bound."""
        backend = _backend(window = 16384, total = 16384, slots = 4)
        assert _enforced(_chat(max_tokens = 0), backend) is None
        assert _enforced(_chat(max_completion_tokens = 0), backend) is None
        # An omitted cap is still the unstated case the bound exists for.
        assert _enforced(_chat(), backend) is not None

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
        """Under --no-kv-unified a slot owns its own cache, so a share IS the window."""
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
    def test_a_prompt_past_its_share_keeps_the_allowance_it_reserved(self):
        """It does not fit a share either way, and the ledger already charged it
        ``prompt + allowance``, so the queue admitted fewer of it rather than four."""
        backend = _backend(window = 16384, total = 16384, slots = 4)
        payload = _chat("word " * 4000, max_tokens = 16384)
        enforced = _enforced(payload, backend)
        assert enforced == _OPENAI_LLAMA_ADMISSION_UNSTATED_OUTPUT_TOKENS - _RESERVE
        charged = _openai_llama_admission_tokens(
            payload, budget = 16384, capacity = 4, context_window = 16384
        )
        assert _prompt_tokens(payload) + enforced == charged - _RESERVE

    def test_the_bound_is_the_charge_less_the_reserve(self):
        """These two must not drift in EITHER direction beyond the reserve; see the class
        below. The reserve is the only permitted gap, and it is the safe direction."""
        backend = _backend(window = 16384, total = 16384, slots = 4)
        payload = _chat(max_tokens = 16384)
        charged = _openai_llama_admission_tokens(
            payload, budget = 16384, capacity = 4, context_window = 16384
        )
        enforced = _enforced(payload, backend)
        assert (
            _prompt_tokens(payload) + enforced == charged - _RESERVE
        ), f"charged {charged} but permits {_prompt_tokens(payload) + enforced}"
        # And it is still generous: a chat gets its share, not a flat thousand tokens.
        assert enforced > _OPENAI_LLAMA_ADMISSION_UNSTATED_OUTPUT_TOKENS

    def test_the_charge_exceeds_what_is_permitted_by_the_reserve_alone(self):
        """Any more and admission reserves room the request cannot use; any less and the
        pool has no cell left to step into."""
        backend = _backend(window = 16384, total = 16384, slots = 4)
        payload = _chat(max_tokens = 16384)
        charged = _openai_llama_admission_tokens(
            payload, budget = 16384, capacity = 4, context_window = 16384
        )
        permitted = _prompt_tokens(payload) + _enforced(payload, backend)
        assert charged - _RESERVE <= permitted <= charged


def _prompt_tokens(payload):
    from routes.inference import _openai_llama_admission_prompt_tokens
    return _openai_llama_admission_prompt_tokens(payload) or 0


class TestItReachesTheWireWithoutBecomingTheCallersCap:
    """It must reach `payload["max_tokens"]`, but folding it into the caller's
    `max_tokens` makes `_loop_budget_left` truncate at one share instead of continuing."""

    def _source(self):
        from pathlib import Path

        import core.inference.llama_cpp as llama_cpp
        return Path(llama_cpp.__file__).read_text(encoding = "utf-8")

    # Payload coverage is in test_llama_admission_enforced_paths.py.

    def test_the_loop_budget_never_sees_it(self):
        """`_loop_budget_left` answers "did the CALLER cap this"; an admission bound did not."""
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
    """The bound is only safe if nothing is admitted on less than it may use: a prompt past
    its share is permitted ``prompt + 1``, safe alone but not mixed with a SMALL prompt
    charged ``prompt + 1024`` while permitted its whole share."""

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
        budget, slots = 262144, 4
        share = budget // slots
        prompts = [1, 65537, 189139, 1]
        admitted, used = [], 0
        for prompt in prompts:
            charged = self._charged(budget, share, prompt)
            if len(admitted) < slots and used + charged <= budget:
                used += charged
                admitted.append(prompt)
        permitted = sum(prompt + max(1, share - prompt) for prompt in admitted)
        assert (
            permitted <= budget
        ), f"admitted {admitted} charged {used} but may occupy {permitted} of {budget}"

    def test_nothing_is_admitted_on_less_than_it_may_use(self):
        for budget, slots in ((16384, 4), (4096, 4), (2048, 2), (32768, 8), (262144, 4)):
            share = budget // slots
            for prompt in (1, 8, share // 2, share - 2, share - 1, share, share + 1, budget - 1):
                if prompt < 1:
                    continue
                permitted = prompt + max(1, share - prompt)
                charged = self._charged(budget, share, prompt)
                if prompt >= share:
                    # Past its share it is charged more than the one token it is permitted.
                    continue
                assert charged >= permitted, (
                    f"budget={budget} share={share} prompt={prompt}: "
                    f"charged {charged} but permitted {permitted}"
                )

    def test_a_full_capacity_of_unstated_requests_still_fits(self):
        """Charging the whole share must not cost the concurrency #10070 bought."""
        for budget, slots in ((16384, 4), (4096, 4), (32768, 8), (262144, 4)):
            share = budget // slots
            charged = self._charged(budget, share, 8)
            assert (
                charged * slots <= budget
            ), f"budget={budget} slots={slots}: {slots} small chats charge {charged * slots}"


class TestAnOverSharePromptIsPricedTheSameOnBothSides:
    """The bug this class exists for: the ledger charged ``prompt + allowance`` for a
    prompt at or above its share while the wire sent ``max_tokens=1``, so a lease big
    enough for a full answer produced a one-token one. Default vision is the common case:
    a single image's allowance is already past a 4096 share on a 16K unified cache.
    """

    def _priced(
        self,
        payload,
        backend,
        *,
        budget,
        capacity,
        window,
        conversation = None,
    ):
        """(charge, prompt + wire bound) for one request, as ledger and wire see it."""
        from routes.inference import (
            _openai_llama_admission_image_tokens,
            _openai_llama_admission_prompt_tokens,
            _openai_llama_admission_wire_prompt_tokens,
        )

        image_tokens = _openai_llama_admission_image_tokens(backend)
        charge = _openai_llama_admission_tokens(
            payload,
            budget = budget,
            capacity = capacity,
            context_window = window,
            image_tokens = image_tokens,
            conversation = conversation,
        )
        bound = _openai_llama_admission_enforced_max_tokens(
            payload, request = None, llama_backend = backend, conversation = conversation
        )
        assert bound is not None
        prompt = (
            _openai_llama_admission_wire_prompt_tokens(conversation, image_tokens = image_tokens)
            if conversation is not None
            else _openai_llama_admission_prompt_tokens(payload, image_tokens = image_tokens)
        )
        return charge, prompt + bound

    def test_a_long_text_prompt_is_charged_exactly_what_it_may_write(self):
        for total, slots in ((16384, 4), (32768, 4), (65536, 8), (262144, 4)):
            backend = _backend(window = total, total = total, slots = slots)
            share = total // slots
            payload = _chat("word " * int(share * 0.9), max_tokens = total)
            charge, permitted = self._priced(
                payload, backend, budget = total, capacity = slots, window = total
            )
            assert _prompt_tokens(payload) > share, "not an over-share prompt"
            assert (
                charge - _RESERVE <= permitted <= charge
            ), f"{total}/{slots}: charged {charge}, permits {permitted}"

    def test_a_default_image_chat_is_not_truncated_after_one_token(self):
        """4224 tokens of image allowance against a 4096 share on 16K with four slots."""
        backend = _backend(window = 16384, total = 16384, slots = 4)
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "what is in this image?"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                ],
            }
        ]
        payload = _Payload(messages = conversation, max_tokens = 16384)
        charge, permitted = self._priced(
            payload,
            backend,
            budget = 16384,
            capacity = 4,
            window = 16384,
            conversation = conversation,
        )
        bound = _openai_llama_admission_enforced_max_tokens(
            payload, request = None, llama_backend = backend, conversation = conversation
        )
        assert (
            bound == _OPENAI_LLAMA_ADMISSION_UNSTATED_OUTPUT_TOKENS - _RESERVE
        ), f"an image answer is capped at {bound} tokens"
        assert bound > 512, "an image chat must still get a usable answer"
        assert charge - _RESERVE <= permitted <= charge

    def test_a_full_queue_of_over_share_requests_still_fits_the_budget(self):
        """Concurrency drops instead: what is admitted may occupy what it was charged."""
        for total, slots in ((16384, 4), (32768, 4), (262144, 4), (8192, 2)):
            backend = _backend(window = total, total = total, slots = slots)
            share = total // slots
            prompts = (1, share // 2, share, share + 1, int(share * 1.5), int(share * 2.5))
            committed, occupancy, admitted = 0, 0, 0
            for prompt in prompts:
                payload = _chat("word " * max(1, prompt // 2), max_tokens = total)
                charge, permitted = self._priced(
                    payload, backend, budget = total, capacity = slots, window = total
                )
                assert (
                    permitted <= charge
                ), f"{total}/{slots} prompt~{prompt}: charged {charge}, permits {permitted}"
                if admitted < slots and committed + charge <= total:
                    committed += charge
                    occupancy += permitted
                    admitted += 1
            assert admitted >= 1
            assert occupancy <= total, (
                f"{total}/{slots}: admitted {admitted} charged {committed} "
                f"but may occupy {occupancy}"
            )
