# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Two holes in the KV reservation the estimator has to charge for.

Media: Unsloth's composer sends attachments in the legacy top-level ``image_base64`` /
``audio_base64`` / ``video_base64`` fields, and the generation path splices them into
the prompt AFTER admission is decided. Charging only ``messages`` priced an image at
zero while llama.cpp's mtmd embeddings take real KV positions.

Tool loop: the server-side loop opens on ``enable_tools`` / ``mcp_enabled`` / the CLI
policy / a checkpoint repair, none of which require a client ``tools`` array, so a
predicate keyed on ``payload.tools`` charged Unsloth's own tool traffic the opening
estimate for a lease that runs up to 25 growing rounds.
"""

import base64

from routes.inference import _openai_llama_admission_tokens


class _Payload:
    def __init__(self, **fields):
        self.__dict__.update(fields)

    def __getattr__(self, _name):
        return None


def _image_b64(kib: int = 200) -> str:
    return base64.b64encode(b"\x89PNG" + b"x" * (kib * 1024)).decode()


class TestMediaIsCharged:
    def test_legacy_image_costs_what_the_same_image_inline_costs(self):
        image = _image_b64()
        legacy = _Payload(
            messages = [{"role": "user", "content": "what is this?"}],
            image_base64 = image,
            max_tokens = 128,
        )
        inline = _Payload(
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "what is this?"},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image}"},
                        },
                    ],
                }
            ],
            max_tokens = 128,
        )
        legacy_cost = _openai_llama_admission_tokens(legacy, budget = 4096, capacity = 4)
        inline_cost = _openai_llama_admission_tokens(inline, budget = 4096, capacity = 4)
        # Before the fix this was 139 against 4096: the same request, one spelling
        # charged the whole cache and the other charged its 13 characters of text.
        assert legacy_cost == inline_cost

    def test_two_image_chats_are_not_both_admitted(self):
        """The live failure, with images instead of text."""
        import asyncio

        from core.inference.llama_admission import LlamaAdmissionConfig, LlamaAdmissionQueue

        async def scenario():
            queue = LlamaAdmissionQueue("media")
            config = LlamaAdmissionConfig()
            image = _image_b64(4)
            leases = []
            for _ in range(2):
                payload = _Payload(
                    messages = [{"role": "user", "content": "describe"}],
                    image_base64 = image,
                    max_tokens = 128,
                )
                reservation = queue.reserve(
                    capacity = 4,
                    config = config,
                    budget = 2048,
                    tokens = _openai_llama_admission_tokens(payload, budget = 2048, capacity = 4),
                )
                leases.append(reservation.lease_nowait())
            return leases

        first, second = asyncio.run(scenario())
        assert first is not None, "the first image chat owns the cache"
        # A 4 KiB image is ~5.4k base64 characters, about 1365 tokens on top of a
        # 2048 cache that already holds one such request: the second must queue.
        assert second is None

    def test_audio_and_video_are_charged_too(self):
        clip = _image_b64(8)
        for field in ("audio_base64", "video_base64"):
            payload = _Payload(
                messages = [{"role": "user", "content": "transcribe"}],
                max_tokens = 64,
                **{field: clip},
            )
            cost = _openai_llama_admission_tokens(payload, budget = 65536, capacity = 4)
            assert cost > 2000, f"{field} was charged {cost}, i.e. nothing for the media"


class TestTheToolLoopReservesTheWholeCache:
    def test_a_server_side_loop_without_client_tools_reserves_the_budget(self):
        """enable_tools / mcp_enabled / CLI policy open the loop with no `tools` array."""
        payload = _Payload(
            messages = [{"role": "user", "content": "search my notes"}],
            enable_tools = True,
            max_tokens = 128,
        )
        assert getattr(payload, "tools", None) is None
        assert (
            _openai_llama_admission_tokens(payload, budget = 4096, capacity = 4, tool_loop = True) == 4096
        )

    def test_a_passthrough_forwarding_tools_is_charged_its_own_round(self):
        """One HTTP call is one generation there: the client drives the rounds."""
        payload = _Payload(
            messages = [{"role": "user", "content": "hi"}],
            tools = [
                {
                    "type": "function",
                    "function": {"name": "shell", "parameters": {"type": "object"}},
                }
            ],
            max_tokens = 128,
        )
        cost = _openai_llama_admission_tokens(payload, budget = 4096, capacity = 4)
        assert cost < 4096, "a forwarded catalogue must not serialise the whole cache"


class TestTheBudgetIsTheWholeCacheNotOneSlot:
    """``context_length`` stops being the total once the server has been read back.

    ``_reconcile_effective_ctx_with_server`` adopts the per-slot ``n_ctx`` into
    ``context_length`` and puts the aggregate in ``_kv_cache_context_total``. Without
    ``--kv-unified`` those differ by ``n_parallel``, and budgeting one private cache
    for the whole pool collapses concurrency to a single generation.
    """

    def test_the_partitioned_total_wins_over_one_slot(self):
        from routes.inference import _openai_llama_admission_budget
        backend = _Payload(context_length = 4096, _kv_cache_context_total = 16384)
        assert _openai_llama_admission_budget(backend) == 16384

    def test_a_unified_cache_is_unchanged(self):
        from routes.inference import _openai_llama_admission_budget

        # slots == 1 under --kv-unified, so the total IS the per-request window.
        backend = _Payload(context_length = 8192, _kv_cache_context_total = 8192)
        assert _openai_llama_admission_budget(backend) == 8192

    def test_an_unread_backend_falls_back_to_context_length(self):
        from routes.inference import _openai_llama_admission_budget

        # Nothing read back yet: the two agree, so the fallback is not a guess.
        backend = _Payload(context_length = 8192, _kv_cache_context_total = None)
        assert _openai_llama_admission_budget(backend) == 8192

    def test_a_backend_that_cannot_say_keeps_slot_only_admission(self):
        from routes.inference import _openai_llama_admission_budget
        assert _openai_llama_admission_budget(_Payload()) is None
