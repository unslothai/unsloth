# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Media and tool-loop cases that the KV reservation has to charge for.

Media: Unsloth's composer sends the current image in both a message-level
``image_url`` part and the legacy top-level ``image_base64`` field. The generation
path splices legacy media into the prompt AFTER admission is decided. Admission must
charge the resulting image once, without treating base64 bytes as prompt text.

Tool loop: the server-side loop opens on ``enable_tools`` / ``mcp_enabled`` / the CLI
policy / a checkpoint repair, none of which require a client ``tools`` array, so a
predicate keyed on ``payload.tools`` charged Unsloth's own tool traffic the opening
estimate for a lease that runs up to 25 growing rounds.
"""

import base64

from routes.inference import (
    _OPENAI_LLAMA_ADMISSION_IMAGE_TOKENS,
    _openai_llama_admission_tokens,
)


class _Payload:
    def __init__(self, **fields):
        self.__dict__.update(fields)

    def __getattr__(self, _name):
        return None


def _image_b64(kib: int = 200) -> str:
    return base64.b64encode(b"\x89PNG" + b"x" * (kib * 1024)).decode()


# Two real, decodable, DIFFERENT PNGs. The builders decode and re-encode, unlike the
# estimator, so the synthetic fixture above cannot reach them; and the pair has to
# differ for a distinct-legacy-image case to be distinct at all.
_TINY_PNG = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGNgYGAAAAAEAAH2FzhVAAAAAElFTkSuQmCC"
)
_OTHER_PNG = (
    "iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAIAAAD91JpzAAAAFklEQVR4nGP8z8DAwMDAxMDAwMDAAAANHQED"
    "asKb6QAAAABJRU5ErkJggg=="
)


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
        # Clear of the clamp: at 4096 `max(1, min(budget, ...))` pinned both sides to the
        # budget, so they agreed whatever the estimator did and this proved nothing.
        legacy_cost = _openai_llama_admission_tokens(legacy, budget = 1_000_000, capacity = 4)
        inline_cost = _openai_llama_admission_tokens(inline, budget = 1_000_000, capacity = 4)
        # The wire spelling must not change the commitment. Not to the token: inline
        # really does send a content-part wrapper legacy does not, and the marker is
        # itself a little JSON. What must not survive is the 30x gap between pricing an
        # image at its base64 length and at its text.
        assert abs(legacy_cost - inline_cost) <= 64, (legacy_cost, inline_cost)
        assert max(legacy_cost, inline_cost) < 2 * _OPENAI_LLAMA_ADMISSION_IMAGE_TOKENS

    def test_studio_image_echo_is_charged_once_and_is_bounded(self):
        image = _image_b64(1024)
        inline_messages = [
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
        ]
        dual = _Payload(
            messages = inline_messages,
            image_base64 = image,
            max_tokens = 128,
        )
        inline = _Payload(messages = inline_messages, max_tokens = 128)

        # Clear of the clamp: at 65536 a 1 MiB image priced as prompt text pinned both
        # sides to the budget, so `dual == inline` held on the unfixed estimator too.
        dual_cost = _openai_llama_admission_tokens(dual, budget = 1_000_000, capacity = 4)
        inline_cost = _openai_llama_admission_tokens(inline, budget = 1_000_000, capacity = 4)

        assert dual_cost == inline_cost
        assert dual_cost < 2 * _OPENAI_LLAMA_ADMISSION_IMAGE_TOKENS, (
            "the echo must be charged once, not once per spelling"
        )
        assert dual_cost >= _OPENAI_LLAMA_ADMISSION_IMAGE_TOKENS, (
            "image bytes must be bounded but still charged"
        )

    def test_every_builder_forwards_exactly_what_admission_charged(self):
        """The reservation is only a bound if it counts the images actually sent.

        Studio echoes one image into both spellings. The GGUF builder always dropped the
        echo; `_openai_messages_for_passthrough` (taken when a client sends ``tools`` or
        a ``response_format``) spliced it in regardless, sending two copies against a
        reservation for one: 4417 reserved against 8466 charged on llama-server b10639.

        The echo is the only thing that may be dropped. A legacy image the thread does
        not already hold is a real attachment, so both builders send it and admission
        charges for it -- keyed on the same predicate, or the two answers drift again.
        """
        from models.inference import ChatCompletionRequest
        from routes.inference import (
            _openai_llama_admission_media_tokens,
            _openai_llama_admission_messages_for_estimate,
            _openai_messages_for_gguf_chat,
            _openai_messages_for_passthrough,
        )

        image = _TINY_PNG
        inline_part = {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{image}"},
        }
        shapes = {
            "studio dual": ChatCompletionRequest(
                model = "m",
                max_tokens = 128,
                messages = [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "what is this?"}, inline_part],
                    }
                ],
                image_base64 = image,
            ),
            "inline only": ChatCompletionRequest(
                model = "m",
                max_tokens = 128,
                messages = [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "what is this?"}, inline_part],
                    }
                ],
            ),
            "legacy only": ChatCompletionRequest(
                model = "m",
                max_tokens = 128,
                messages = [{"role": "user", "content": "what is this?"}],
                image_base64 = image,
            ),
            # An older image in history plus a genuinely different one attached to this
            # turn through the legacy field: two images, and both must be charged.
            "history image plus a distinct legacy attachment": ChatCompletionRequest(
                model = "m",
                max_tokens = 128,
                messages = [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "and this?"}, inline_part],
                    }
                ],
                image_base64 = _OTHER_PNG,
            ),
        }

        def _forwarded(messages):
            return sum(
                1
                for msg in messages
                if isinstance(msg.get("content"), list)
                for part in msg["content"]
                if isinstance(part, dict) and part.get("type") == "image_url"
            )

        for name, payload in shapes.items():
            _, message_image_parts = _openai_llama_admission_messages_for_estimate(payload.messages)
            billed = (
                _openai_llama_admission_media_tokens(
                    payload, message_image_parts = message_image_parts
                )
                // _OPENAI_LLAMA_ADMISSION_IMAGE_TOKENS
            )
            for builder_name, messages in (
                ("gguf chat", _openai_messages_for_gguf_chat(payload, True)[0]),
                ("passthrough", _openai_messages_for_passthrough(payload)),
            ):
                assert _forwarded(messages) == billed, (
                    f"{name} via {builder_name}: forwarded {_forwarded(messages)} "
                    f"image(s) but admission charged for {billed}"
                )

    def test_the_allowance_bounds_a_real_projector(self):
        """The per-image charge is an upper bound, not an estimate.

        Measured on llama-server b10639: 4098 KV positions for a 2048x2048 image on
        Qwen3-VL-4B (the 4096-embedding cap plus two mtmd delimiters) and 258 on Gemma 3
        4B, at every resolution and encoded size. A flat 4096 sat below the Qwen figure,
        and reserving less than a request costs is what lets two collide in one cache.
        """
        measured_worst_case = {"qwen3-vl-4b": 4098, "gemma-3-4b": 258}
        for model, tokens in measured_worst_case.items():
            assert _OPENAI_LLAMA_ADMISSION_IMAGE_TOKENS >= tokens, (
                f"per-image allowance under-reserves {model}"
            )

    def test_the_allowance_follows_a_raised_image_token_cap(self):
        """A load can raise the projector ceiling, and the reservation has to follow it.

        ``--image-max-tokens`` is not Unsloth-managed, so ``llama_extra_args`` forwards
        it verbatim. Measured on b10639 with ``--image-max-tokens 8192``: a 4096x4096
        Qwen3-VL image costs 8102, against 4098 at the default. Reserving the default
        against that backend admits concurrent requests the cache cannot hold.
        """
        from routes.inference import (
            _MMPROJ_IMAGE_TOKEN_MAX,
            _openai_llama_admission_image_tokens,
        )

        class _Backend:
            def __init__(
                self,
                extra_args = None,
                projector = None,
            ):
                self._extra_args = extra_args
                self._mmproj_projector_type = projector

        assert _openai_llama_admission_image_tokens(_Backend()) == (
            _OPENAI_LLAMA_ADMISSION_IMAGE_TOKENS
        )
        for spelling in (["--image-max-tokens", "8192"], ["--image-max-tokens=8192"]):
            allowance = _openai_llama_admission_image_tokens(_Backend(spelling))
            assert allowance > _OPENAI_LLAMA_ADMISSION_IMAGE_TOKENS
            assert allowance >= 8102, f"{spelling} under-reserves the measured cost"
        # Junk must not be mistaken for a cap, and must not raise.
        for junk in (["--image-max-tokens"], ["--image-max-tokens", "abc"], ["-c", "40000"]):
            assert _openai_llama_admission_image_tokens(_Backend(junk)) == (
                _OPENAI_LLAMA_ADMISSION_IMAGE_TOKENS
            )

        # Every family llama.cpp gives its own ceiling must be bounded, including the
        # ones far above the default: youtuvl is 62500 and hunyuanvl 16384, so a flat
        # default would have reserved a fraction of what one image really costs.
        for projector, ceiling in _MMPROJ_IMAGE_TOKEN_MAX.items():
            assert _openai_llama_admission_image_tokens(_Backend(projector = projector)) >= ceiling
        assert _openai_llama_admission_image_tokens(_Backend(projector = "youtuvl")) >= 62500
        # A projector with a small ceiling reserves near it rather than the default.
        assert _openai_llama_admission_image_tokens(_Backend(projector = "lfm2")) < 1024
        # An unknown family keeps the default rather than inventing a number.
        assert _openai_llama_admission_image_tokens(_Backend(projector = "nope")) == (
            _OPENAI_LLAMA_ADMISSION_IMAGE_TOKENS
        )
        # The flag is only honoured by dynamic-resolution projectors, so a LOW cap must
        # not talk the reservation below what a fixed-resolution one really costs.
        assert (
            _openai_llama_admission_image_tokens(
                _Backend(["--image-max-tokens", "16"], projector = "qwen3vl_merger")
            )
            >= _MMPROJ_IMAGE_TOKEN_MAX["qwen3vl_merger"]
        )

    def test_two_large_studio_image_chats_can_be_admitted_together(self):
        """A large base64 transport must not turn each vision request into a full-cache lease."""
        import asyncio

        from core.inference.llama_admission import LlamaAdmissionConfig, LlamaAdmissionQueue

        async def scenario():
            queue = LlamaAdmissionQueue("media")
            config = LlamaAdmissionConfig()
            image = _image_b64(1024)
            leases = []
            for _ in range(2):
                payload = _Payload(
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "describe"},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/png;base64,{image}"},
                                },
                            ],
                        }
                    ],
                    image_base64 = image,
                    max_tokens = 128,
                )
                reservation = queue.reserve(
                    capacity = 4,
                    config = config,
                    budget = 12_000,
                    tokens = _openai_llama_admission_tokens(payload, budget = 12_000, capacity = 4),
                )
                leases.append(reservation.lease_nowait())
            admitted = [lease is not None for lease in leases]
            for lease in leases:
                if lease is not None:
                    lease.release()
            return admitted

        assert asyncio.run(scenario()) == [True, True]

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
        # The conservative per-image allowance alone is larger than this tiny cache,
        # so the second request must still queue rather than overcommit it.
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
