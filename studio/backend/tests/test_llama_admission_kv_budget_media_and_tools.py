# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Charge a media image once, and key the tool loop on its server-side triggers, not payload.tools."""

import base64
import copy
import io
import wave

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import routes.inference as inference_route
from auth.authentication import get_current_subject

from models.inference import AnthropicMessagesRequest
from routes.inference import (
    _OPENAI_LLAMA_ADMISSION_IMAGE_TOKENS,
    _openai_llama_admission_media_tokens,
    _openai_llama_admission_messages_for_estimate,
    _openai_llama_admission_tokens,
)
from core.inference.anthropic_compat import anthropic_messages_to_openai
from core.inference.llama_admission import LlamaAdmissionConfig, LlamaAdmissionQueue
from routes.inference import _openai_llama_admission_budget
import asyncio


class _Payload:
    def __init__(self, **fields):
        self.__dict__.update(fields)

    def __getattr__(self, _name):
        return None


def _image_b64(kib: int = 200) -> str:
    return base64.b64encode(b"\x89PNG" + b"x" * (kib * 1024)).decode()


# real, decodable, different PNGs: the builders decode and re-encode, the estimator does not
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
        # clear of the clamp, else both sides pin to the budget and agree regardless
        legacy_cost = _openai_llama_admission_tokens(legacy, budget = 1_000_000, capacity = 4)
        inline_cost = _openai_llama_admission_tokens(inline, budget = 1_000_000, capacity = 4)
        # Wire spelling must not change the commitment (not to the token); the 30x gap must not survive
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

        dual_cost = _openai_llama_admission_tokens(dual, budget = 1_000_000, capacity = 4)
        inline_cost = _openai_llama_admission_tokens(inline, budget = 1_000_000, capacity = 4)

        assert dual_cost == inline_cost
        assert (
            dual_cost < 2 * _OPENAI_LLAMA_ADMISSION_IMAGE_TOKENS
        ), "the echo must be charged once, not once per spelling"
        assert (
            dual_cost >= _OPENAI_LLAMA_ADMISSION_IMAGE_TOKENS
        ), "image bytes must be bounded but still charged"

    def test_every_builder_forwards_exactly_what_admission_charged(self):
        """Every builder must send exactly what admission charged; only the echo of an image may be
        dropped."""
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
        """The per-image allowance must bound the measured worst case, 4098 KV positions for Qwen3-VL-4B."""
        measured_worst_case = {"qwen3-vl-4b": 4098, "gemma-3-4b": 258}
        for model, tokens in measured_worst_case.items():
            assert (
                _OPENAI_LLAMA_ADMISSION_IMAGE_TOKENS >= tokens
            ), f"per-image allowance under-reserves {model}"

    def test_the_allowance_follows_a_raised_image_token_cap(self):
        """A raised --image-max-tokens via llama_extra_args must raise the per-image reservation too."""
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
        for junk in (["--image-max-tokens"], ["--image-max-tokens", "abc"], ["-c", "40000"]):
            assert _openai_llama_admission_image_tokens(_Backend(junk)) == (
                _OPENAI_LLAMA_ADMISSION_IMAGE_TOKENS
            )

        # youtuvl is 62500 and hunyuanvl 16384, far above the default
        for projector, ceiling in _MMPROJ_IMAGE_TOKEN_MAX.items():
            assert _openai_llama_admission_image_tokens(_Backend(projector = projector)) >= ceiling
        assert _openai_llama_admission_image_tokens(_Backend(projector = "youtuvl")) >= 62500
        assert _openai_llama_admission_image_tokens(_Backend(projector = "lfm2")) < 1024
        assert _openai_llama_admission_image_tokens(_Backend(projector = "nope")) == (
            _OPENAI_LLAMA_ADMISSION_IMAGE_TOKENS
        )
        # only dynamic-resolution projectors honour the flag, so a low cap can't undercut these
        assert (
            _openai_llama_admission_image_tokens(
                _Backend(["--image-max-tokens", "16"], projector = "qwen3vl_merger")
            )
            >= _MMPROJ_IMAGE_TOKEN_MAX["qwen3vl_merger"]
        )

    def test_two_large_studio_image_chats_can_be_admitted_together(self):
        """A large base64 transport must not turn each vision request into a full-cache lease."""

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


def _wav_b64(seconds: float, rate: int = 16000) -> str:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * int(seconds * rate))
    return base64.b64encode(buf.getvalue()).decode()


def _mp3_b64(frames: int) -> str:
    # MPEG-1 Layer III, 128 kbps, 44.1 kHz: 417-byte frames of 1152 samples
    frame = b"\xff\xfb\x90\x00" + b"\x00" * 413
    return base64.b64encode(frame * frames).decode()


class TestAnAudioTurnIsChargedByItsDuration:
    """A recording was charged len(base64) // 4, so four seconds of 16 kHz WAV filled a 32768
    cache and every other chat queued behind it. mtmd charges audio by duration, not bytes."""

    class _Backend:
        base_url = "http://llama-audio"
        effective_parallel_slots = 4
        _kv_cache_context_total = 32768
        context_length = 32768
        _mmproj_projector_type = None
        _extra_args = None

    def test_an_audio_chat_does_not_reserve_the_whole_cache(self):
        audio = _wav_b64(20)
        assert len(audio) > 800_000
        budget = 32768
        payload = _Payload(
            messages = [{"role": "user", "content": "transcribe"}],
            audio_base64 = audio,
            max_tokens = 256,
        )
        cost = _openai_llama_admission_tokens(payload, budget = budget, capacity = 4)
        assert cost < budget // 4, f"20 s of audio reserved {cost} of {budget}"

    def test_a_text_chat_is_admitted_beside_an_audio_chat(self):
        async def scenario():
            queue = LlamaAdmissionQueue("media")
            config = LlamaAdmissionConfig()
            leases = []
            for fields in (
                {"audio_base64": _wav_b64(5), "content": "transcribe"},
                {"content": "hi"},
            ):
                payload = _Payload(
                    messages = [{"role": "user", "content": fields.pop("content")}],
                    max_tokens = 256,
                    **fields,
                )
                reservation = queue.reserve(
                    capacity = 4,
                    config = config,
                    budget = 32768,
                    tokens = _openai_llama_admission_tokens(payload, budget = 32768, capacity = 4),
                )
                leases.append(reservation.lease_nowait())
            return leases

        audio, text = asyncio.run(scenario())
        assert audio is not None
        assert text is not None, "a text chat queued behind a 5 s recording"

    @pytest.mark.parametrize(
        "audio, seconds",
        [
            (_wav_b64(1), 1.0),
            ("data:audio/wav;base64," + _wav_b64(45), 45.0),
            (_mp3_b64(383), 383 * 1152 / 44100),
        ],
        ids = ["wav-1s", "wav-data-uri-45s", "mp3-10s"],
    )
    def test_the_charge_bounds_every_audio_projector(self, audio, seconds):
        cost = _openai_llama_admission_media_tokens(_Payload(audio_base64 = audio))
        # 25 embeddings a second; a Whisper encoder pads each clip to a whole 30 s window
        whisper_windows = int(seconds // 30) + 1
        assert cost >= max(25 * seconds, 750 * whisper_windows)
        assert cost <= 25 * (seconds + 30) + 256

    @pytest.mark.parametrize(
        "audio",
        [
            _image_b64(8),
            base64.b64encode(b"RIFF\x00\x00\x00\x00WAVEjunk" + b"\x00" * 4096).decode(),
            "not base64 at all!",
        ],
        ids = ["not-audio", "wav-without-data-chunk", "not-base64"],
    )
    def test_audio_without_a_stated_duration_keeps_the_byte_charge(self, audio):
        cost = _openai_llama_admission_media_tokens(_Payload(audio_base64 = audio))
        assert cost == max(1, len(audio) // 4)

    def test_video_keeps_the_byte_charge(self):
        clip = _wav_b64(20)
        cost = _openai_llama_admission_media_tokens(_Payload(video_base64 = clip, messages = []))
        assert cost == len(clip) // 4

    def test_a_tool_round_does_not_price_the_injected_recording_as_text(self):
        from routes.inference import _openai_llama_admission_recost

        class _Reservation:
            def __init__(self, lease):
                self._lease = lease

            def lease_nowait(self):
                return self._lease

        audio = _wav_b64(3)
        payload = _Payload(
            messages = [{"role": "user", "content": "transcribe"}],
            audio_base64 = audio,
            enable_tools = True,
            max_tokens = 256,
        )
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "transcribe"},
                    {"type": "input_audio", "input_audio": {"data": audio, "format": "wav"}},
                ],
            }
        ]

        async def run():
            queue = LlamaAdmissionQueue("test")
            reservation = queue.reserve(
                capacity = 4,
                config = LlamaAdmissionConfig(),
                budget = 32768,
                tokens = _openai_llama_admission_tokens(
                    payload, budget = 32768, capacity = 4, tool_loop = True
                ),
            )
            lease = reservation.lease_nowait()
            assert lease is not None
            _openai_llama_admission_recost(
                _Reservation(lease),
                conversation,
                request = None,
                llama_backend = self._Backend(),
                payload = payload,
                output_tokens = 256,
            )
            return queue.snapshot().committed

        committed = asyncio.run(run())
        assert committed <= 32768 // 4, f"round zero re-costed a 3 s recording to {committed}"

    def test_the_recording_is_decoded_off_the_event_loop(self, monkeypatch):
        import threading

        import routes.inference as inference_route

        on_loop = []
        estimate = inference_route._openai_llama_admission_estimate

        def spy(**kwargs):
            on_loop.append(threading.current_thread() is threading.main_thread())
            return estimate(**kwargs)

        monkeypatch.setattr(inference_route, "_openai_llama_admission_estimate", spy)
        payload = _Payload(
            messages = [{"role": "user", "content": "transcribe"}],
            audio_base64 = _wav_b64(1),
            max_tokens = 64,
        )

        async def run():
            reservation, _ = await inference_route._openai_llama_admission_reserve_async(
                request = None, llama_backend = self._Backend(), payload = payload
            )
            reservation.cancel()

        asyncio.run(run())
        assert on_loop == [False]

    # an ogg header states no length, so only the WAV it is transcoded to can be measured
    @pytest.mark.parametrize(
        "raw",
        [inference_route._mono_f32_to_wav_bytes(np.zeros(320_000), 16000), b"OggS" * 4**8],
        ids = ["wav", "ogg"],
    )
    def test_an_audio_chat_reserves_a_bound_on_its_duration(self, monkeypatch, raw):
        from .llama_backend_double import FakeLlamaCppBackend

        class _AudioGguf(FakeLlamaCppBackend):
            is_vision = _has_audio_input = True
            _kv_cache_context_total = context_length = 32768

            def generate_chat_completion(self, **_kwargs):
                yield "ok"

            def count_chat_tokens(self, *_args, **_kwargs):
                return 12

            def _request_reasoning_kwargs(self, *_args):
                return None

        charged, tokens = [], inference_route._openai_llama_admission_tokens
        monkeypatch.setattr(inference_route, "get_llama_cpp_backend", _AudioGguf)
        monkeypatch.setattr(
            inference_route,
            "_openai_llama_admission_tokens",
            lambda *a, **kw: charged.append(tokens(*a, **kw)) or charged[-1],
        )
        monkeypatch.setattr(
            inference_route, "_decode_audio_mono", lambda _raw: (np.zeros(20 * 16000), 16000)
        )
        app = FastAPI()
        app.include_router(inference_route.router, prefix = "/v1")
        app.dependency_overrides[get_current_subject] = lambda: "tester"
        body = {
            "max_tokens": 256,
            "messages": [{"role": "user", "content": "hi"}],
            "audio_base64": base64.b64encode(raw).decode(),
        }

        response = TestClient(app).post("/v1/chat/completions", json = body)

        assert response.status_code == 200, response.text
        # 25/s over the clip plus a trailing 30 s window, the wrapper, output, and the prompt
        assert 0 <= charged[0] - (25 * (20 + 30) + 128 + 256) < 32, charged


class TestAnAnthropicImageIsChargedLikeAnyOtherImage:
    """/v1/messages prices from the raw request, so its own image block must be compacted too."""

    def _request(self, data: str):
        return AnthropicMessagesRequest(
            model = "default",
            max_tokens = 128,
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "what is this?"},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": data,
                            },
                        },
                    ],
                }
            ],
        )

    def test_a_big_anthropic_image_costs_what_a_tiny_one_costs(self):
        big = _openai_llama_admission_tokens(
            self._request(_image_b64(1024)), budget = 1_000_000, capacity = 4
        )
        tiny = _openai_llama_admission_tokens(self._request("AAAA"), budget = 1_000_000, capacity = 4)
        assert abs(big - tiny) <= _OPENAI_LLAMA_ADMISSION_IMAGE_TOKENS, (
            f"a 1 MiB Anthropic image was charged {big} against {tiny} for a 4-char one: "
            "the base64 transport is being priced as prompt text"
        )
        assert (
            big >= _OPENAI_LLAMA_ADMISSION_IMAGE_TOKENS
        ), "image bytes must be bounded but still charged"

    def test_a_screenshot_does_not_reserve_the_whole_cache(self):
        budget = 32768
        cost = _openai_llama_admission_tokens(
            self._request(_image_b64(150)), budget = budget, capacity = 4
        )
        assert cost < budget, (
            f"a 150 KiB screenshot was charged {cost} against a {budget}-token cache, so "
            "the queue admits it alone and every other chat waits"
        )

    def test_the_estimate_does_not_carry_the_base64(self):
        data = _image_b64(64)
        estimate_messages, image_parts = _openai_llama_admission_messages_for_estimate(
            self._request(data).messages
        )
        assert image_parts == 1, "the bounded per-image allowance is keyed on this count"
        assert data not in str(estimate_messages)


class TestAToolResultScreenshotIsNotPricedByItsBase64:
    """A nested tool_result screenshot was charged its base64, filling the whole 32768-token cache."""

    def _request(self, data: str):
        return AnthropicMessagesRequest(
            model = "default",
            max_tokens = 128,
            messages = [
                {"role": "user", "content": [{"type": "text", "text": "take a screenshot"}]},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "id": "toolu_01", "name": "screenshot", "input": {}}
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_01",
                            "content": [
                                {"type": "text", "text": "screenshot taken"},
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/png",
                                        "data": data,
                                    },
                                },
                            ],
                        }
                    ],
                },
            ],
        )

    def test_a_screenshot_a_tool_returned_does_not_reserve_the_whole_cache(self):
        budget = 32768
        cost = _openai_llama_admission_tokens(
            self._request(_image_b64(150)), budget = budget, capacity = 4
        )
        assert cost < budget, (
            f"a 150 KiB tool-result screenshot was charged {cost} against a {budget}-token "
            "cache, so the agent that took it runs alone"
        )

    def test_a_big_tool_result_screenshot_costs_what_a_tiny_one_costs(self):
        big = _openai_llama_admission_tokens(
            self._request(_image_b64(1024)), budget = 1_000_000, capacity = 4
        )
        tiny = _openai_llama_admission_tokens(self._request("AAAA"), budget = 1_000_000, capacity = 4)
        assert abs(big - tiny) <= _OPENAI_LLAMA_ADMISSION_IMAGE_TOKENS, (
            f"a 1 MiB tool-result image was charged {big} against {tiny} for a 4-char one: "
            "the base64 transport is being priced as prompt text"
        )

    def test_the_charge_matches_what_the_translation_actually_sends(self):
        """Every forwarded tool image earns an embedding allowance, not a base64 charge."""
        data = _TINY_PNG
        payload = self._request(data)

        estimate_messages, image_parts = _openai_llama_admission_messages_for_estimate(
            payload.messages
        )
        assert data not in str(estimate_messages), "the base64 must not be priced as text"

        sent = anthropic_messages_to_openai(
            [message.model_dump() for message in payload.messages], None
        )
        forwarded = data in str(sent)
        assert image_parts == (1 if forwarded else 0), (
            "admission charges a bounded image allowance exactly when the translation "
            f"sends the image (forwarded={forwarded}, image_parts={image_parts})"
        )

        from types import SimpleNamespace

        from routes.inference import _openai_llama_admission_image_tokens

        text_only = anthropic_messages_to_openai(
            [message.model_dump() for message in payload.messages], None, tool_result_images = False
        )
        assert data not in str(text_only)
        assert _openai_llama_admission_image_tokens(SimpleNamespace(is_vision = False)) == 0

    @pytest.mark.parametrize("source_type", ["base64", "url"])
    @pytest.mark.parametrize("with_text", [False, True])
    def test_each_nested_image_is_compacted_and_charged(self, source_type, with_text):
        payload = self._request(_TINY_PNG)
        blocks = payload.messages[-1].content[0].content
        first = blocks[-1]
        second = copy.deepcopy(first)
        second["source"]["data"] = _OTHER_PNG
        if source_type == "url":
            for block in (first, second):
                block["source"] = {
                    "type": "url",
                    "url": f"data:image/png;base64,{block['source']['data']}",
                }
        blocks[:] = [first, *blocks[:1], second] if with_text else [first, second]
        original = payload.model_dump()

        compact, count = _openai_llama_admission_messages_for_estimate(payload.messages)
        sent = anthropic_messages_to_openai([m.model_dump() for m in payload.messages])
        forwarded = [p for m in sent if m["role"] == "tool" for p in m["content"]]
        assert count == sum(p["type"] == "image_url" for p in forwarded) == 2
        assert _TINY_PNG not in str(compact) and _OTHER_PNG not in str(compact)
        assert payload.model_dump() == original
        assert (
            _openai_llama_admission_tokens(payload, budget = 1_000_000, capacity = 4)
            >= 2 * _OPENAI_LLAMA_ADMISSION_IMAGE_TOKENS
        )


class TestEveryBlockTheTranslationDropsIsPricedTheSameWay:
    """Dropped blocks must be priced by the short note they send, not by their base64 length."""

    def _blocks(self, data: str):
        return {
            "document": {
                "type": "document",
                "source": {"type": "base64", "media_type": "application/pdf", "data": data},
            },
            "search_result": {"type": "search_result", "source": {"data": data}},
            "nested tool_result": {
                "type": "tool_result",
                "tool_use_id": "toolu_02",
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": "image/png", "data": data},
                    }
                ],
            },
        }

    def _request(self, block, *, text_first: bool):
        text = {"type": "text", "text": "the tool answered"}
        content = [text, block] if text_first else [block, text]
        return AnthropicMessagesRequest(
            model = "default",
            max_tokens = 128,
            messages = [
                {"role": "user", "content": [{"type": "text", "text": "use the tool"}]},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "id": "toolu_01", "name": "lookup", "input": {}}
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "toolu_01", "content": content}
                    ],
                },
            ],
        )

    def test_none_of_them_reserve_the_whole_cache(self):
        budget = 32768
        data = _image_b64(150)
        # both orders: a filter that stops at the first block would pass one of them
        for text_first in (True, False):
            for name, block in self._blocks(data).items():
                payload = self._request(block, text_first = text_first)
                cost = _openai_llama_admission_tokens(payload, budget = budget, capacity = 4)
                assert cost < budget, (
                    f"a 150 KiB {name} block (text_first={text_first}) was charged {cost} "
                    f"against a {budget}-token cache, so that agent runs alone"
                )

    def test_the_charge_matches_what_the_translation_actually_sends(self):
        """The charge tracks what the translation actually sends, so dropping whole tool_result
        blocks fails."""
        data = _image_b64(64)
        for text_first in (True, False):
            for name, block in self._blocks(data).items():
                where = f"{name} (text_first={text_first})"
                payload = self._request(block, text_first = text_first)
                estimate_messages, image_parts = _openai_llama_admission_messages_for_estimate(
                    payload.messages
                )
                sent = anthropic_messages_to_openai(
                    [message.model_dump() for message in payload.messages], None
                )
                assert data not in str(sent), f"{where}: the translation now forwards this block"
                assert data not in str(
                    estimate_messages
                ), f"{where}: the transport is being priced as prompt text"
                assert (
                    image_parts == 0
                ), f"{where}: charged {image_parts} image allowances for a dropped block"
                assert "the tool answered" in str(
                    estimate_messages
                ), f"{where}: the text beside it IS sent, so dropping it under-reserves"

    def test_search_result_and_document_text_is_charged_where_it_is_sent(self):
        search_result = {
            "type": "search_result",
            "source": "kb://vault",
            "title": "Vault",
            "content": [{"type": "text", "text": "PURPLE-ELEPHANT-42 " * 200}],
        }
        document = {
            "type": "document",
            "source": {"type": "text", "media_type": "text/plain", "data": "MEMO-BODY " * 200},
            "title": "Memo",
        }
        for block, header in (
            (search_result, "Title: Vault\nSource: kb://vault\n"),
            (document, "Title: Memo\n"),
        ):
            for payload in (
                self._request(block, text_first = True),
                AnthropicMessagesRequest(
                    model = "default",
                    max_tokens = 128,
                    messages = [{"role": "user", "content": [block]}],
                ),
            ):
                sent = anthropic_messages_to_openai([m.model_dump() for m in payload.messages])
                rendered = sent[-1]["content"].rsplit("the tool answered\n", 1)[-1]
                estimate_messages, image_parts = _openai_llama_admission_messages_for_estimate(
                    payload.messages
                )
                assert rendered.startswith(header)
                assert rendered in str(estimate_messages).replace("\\n", "\n")
                assert image_parts == 0
                cost = _openai_llama_admission_tokens(payload, budget = 1_000_000, capacity = 4)
                assert cost > len(rendered) // 8

    def test_a_tool_result_the_translation_does_forward_is_still_charged(self):
        """Forwarded string tool_result content is charged by length, even when it looks like base64."""
        data = _image_b64(150)
        payload = AnthropicMessagesRequest(
            model = "default",
            max_tokens = 128,
            messages = [
                {"role": "user", "content": [{"type": "text", "text": "use the tool"}]},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "id": "toolu_01", "name": "lookup", "input": {}}
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "toolu_01", "content": data}
                    ],
                },
            ],
        )
        sent = anthropic_messages_to_openai(
            [message.model_dump() for message in payload.messages], None
        )
        assert data in str(sent), "the translation stopped forwarding string tool_result content"

        estimate_messages, image_parts = _openai_llama_admission_messages_for_estimate(
            payload.messages
        )
        assert data in str(
            estimate_messages
        ), "content that IS sent was dropped from the estimate, which under-reserves"
        assert image_parts == 0, "a string tool result is prompt text, not an image"

        cost = _openai_llama_admission_tokens(payload, budget = 1_000_000, capacity = 4)
        assert (
            cost > len(data) // 8
        ), f"a {len(data)}-char forwarded tool result was charged only {cost}"


class TestTheToolLoopOpensAtAnEqualShare:
    """A tool loop opens at an equal share and re-costs each round, so the whole cache is not reserved."""

    def test_a_server_side_loop_without_client_tools_is_still_a_tool_loop(self):
        """A loop with no client tools array is still a tool loop; keying on payload.tools
        undercharges it."""
        payload = _Payload(
            messages = [{"role": "user", "content": "search my notes"}],
            enable_tools = True,
            max_tokens = 128,
        )
        assert getattr(payload, "tools", None) is None
        cost = _openai_llama_admission_tokens(payload, budget = 4096, capacity = 4, tool_loop = True)
        assert cost == 1024, cost

    def test_four_tool_chats_fit_the_cache_at_once(self):
        """Four equal shares are exactly the budget, so four tool chats are admitted
        together instead of one at a time."""
        payload = _Payload(
            messages = [{"role": "user", "content": "search my notes"}],
            enable_tools = True,
            max_tokens = 128,
        )
        cost = _openai_llama_admission_tokens(payload, budget = 262144, capacity = 4, tool_loop = True)
        assert cost * 4 <= 262144

    def test_a_tool_loop_bigger_than_its_share_is_charged_what_it_is(self):
        """The share is a floor, not a cap: a run already sending more than a quarter of
        the cache is charged for it, not admitted alongside three others."""
        payload = _Payload(
            messages = [{"role": "user", "content": "x " * 4000}],
            enable_tools = True,
            max_tokens = 128,
        )
        cost = _openai_llama_admission_tokens(payload, budget = 4096, capacity = 4, tool_loop = True)
        assert cost > 1024, cost
        assert cost <= 4096

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
    """Budget the whole pool total, not one slot: without --kv-unified, context_length is only per-slot."""

    def test_the_partitioned_total_wins_over_one_slot(self):
        backend = _Payload(context_length = 4096, _kv_cache_context_total = 16384)
        assert _openai_llama_admission_budget(backend) == 16384

    def test_a_unified_cache_is_unchanged(self):
        # slots == 1 under --kv-unified, so the total IS the per-request window
        backend = _Payload(context_length = 8192, _kv_cache_context_total = 8192)
        assert _openai_llama_admission_budget(backend) == 8192

    def test_an_unread_backend_falls_back_to_context_length(self):
        backend = _Payload(context_length = 8192, _kv_cache_context_total = None)
        assert _openai_llama_admission_budget(backend) == 8192

    def test_a_backend_that_cannot_say_keeps_slot_only_admission(self):
        assert _openai_llama_admission_budget(_Payload()) is None


class TestARoundIsCostedTheSameWayTheReservationWas:
    """A re-cost must count the same terms as the reservation, or it frees room still in use."""

    class _Backend:
        base_url = "http://llama"
        effective_parallel_slots = 4
        _kv_cache_context_total = 4096
        context_length = 4096
        _mmproj_projector_type = None
        _extra_args = None

    class _Reservation:
        def __init__(self, lease):
            self._lease = lease

        def lease_nowait(self):
            return self._lease

    def _round_zero(self, payload, *, output_tokens):
        """Open a tool lease from ``payload``, then re-cost it before it has grown."""
        from routes.inference import (
            _openai_llama_admission_recost,
            _openai_llama_admission_tokens,
        )

        async def _run():
            queue = LlamaAdmissionQueue("test")
            opened = _openai_llama_admission_tokens(
                payload, budget = 4096, capacity = 4, tool_loop = True
            )
            reservation = queue.reserve(
                capacity = 4,
                config = LlamaAdmissionConfig(),
                tokens = opened,
                budget = 4096,
            )
            lease = reservation.lease_nowait()
            assert lease is not None
            _openai_llama_admission_recost(
                self._Reservation(lease),
                payload.messages,
                request = None,
                llama_backend = self._Backend(),
                payload = payload,
                output_tokens = output_tokens,
            )
            return opened, queue.snapshot().committed, queue

        return asyncio.run(_run())

    def test_round_zero_does_not_give_away_a_vision_prompt_s_image_allowance(self):
        """Image parts compact to "[image]" for the text estimate, so the real mtmd cost
        can only come from the compaction count. Discarding it re-costs a vision tool run
        DOWNWARD by the whole allowance on its first round."""
        payload = _Payload(
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "what is in this picture?"},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{_TINY_PNG}"},
                        },
                    ],
                }
            ],
            enable_tools = True,
            max_tokens = 128,
        )
        opened, committed, _ = self._round_zero(payload, output_tokens = 128)
        assert _OPENAI_LLAMA_ADMISSION_IMAGE_TOKENS > 4096 and opened == 4096, opened
        assert committed == opened, (
            f"round zero shrank a correct lease from {opened} to {committed}, "
            f"giving away {opened - committed} tokens of resident image KV"
        )

    def test_round_zero_keeps_an_uncapped_loop_s_output_allowance(self):
        """Round-zero re-cost must not shrink the opening lease; llama-server is already using that room."""
        from routes.inference import (
            _OPENAI_LLAMA_ADMISSION_UNSTATED_OUTPUT_TOKENS,
            _effective_openai_max_tokens,
        )

        payload = _Payload(
            messages = [{"role": "user", "content": "summarise the news"}],
            enable_tools = True,
        )
        assert _effective_openai_max_tokens(payload) is None
        opened, committed, _queue = self._round_zero(
            payload, output_tokens = _effective_openai_max_tokens(payload)
        )
        assert opened < 4096, f"an uncapped loop still opens on the whole {4096} cache ({opened})"
        assert opened <= _OPENAI_LLAMA_ADMISSION_UNSTATED_OUTPUT_TOKENS + 64, opened
        assert (
            committed == opened
        ), f"round zero shrank an uncapped loop from {opened} to {committed}"

    def test_round_zero_keeps_a_top_level_system_prompt(self):
        """Anthropic keeps `system` and `tools` out of `messages` entirely, so for that
        route this is most of the prompt."""
        payload = _Payload(
            messages = [{"role": "user", "content": "hi"}],
            system = "You are a careful assistant that cites its sources. " * 200,
            enable_tools = True,
            max_tokens = 128,
        )
        opened, committed, _ = self._round_zero(payload, output_tokens = 128)
        assert opened > 1024, f"the system text should push this past the share: {opened}"
        assert committed == opened, (
            f"round zero shrank the lease from {opened} to {committed}, dropping the "
            f"non-message prompt the reservation charged"
        )


class TestARoundStopsPayingForAnEvictedClip:
    """Re-cost reads the current conversation, since truncate_oldest can evict the turn that held a clip."""

    class _Backend:
        base_url = "http://llama"
        effective_parallel_slots = 4
        _kv_cache_context_total = 4096
        context_length = 4096
        _mmproj_projector_type = None
        _extra_args = None

    class _Reservation:
        def __init__(self, lease):
            self._lease = lease

        def lease_nowait(self):
            return self._lease

    def _recost(self, payload, conversation):
        from routes.inference import (
            _openai_llama_admission_recost,
            _openai_llama_admission_tokens,
        )
        async def _run():
            queue = LlamaAdmissionQueue("test")
            opened = _openai_llama_admission_tokens(
                payload, budget = 4096, capacity = 4, tool_loop = True
            )
            reservation = queue.reserve(
                capacity = 4,
                config = LlamaAdmissionConfig(),
                tokens = opened,
                budget = 4096,
            )
            lease = reservation.lease_nowait()
            assert lease is not None
            _openai_llama_admission_recost(
                self._Reservation(lease),
                conversation,
                request = None,
                llama_backend = self._Backend(),
                payload = payload,
                output_tokens = 64,
            )
            return queue.snapshot().committed

        return asyncio.run(_run())

    def _payload(self, clip_b64):
        return _Payload(
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video_url",
                            "video_url": {"url": "data:video/mp4;base64," + clip_b64},
                        }
                    ],
                }
            ],
            video_base64 = None,
            audio_base64 = None,
            image_base64 = None,
        )

    def test_the_clip_is_charged_while_the_conversation_still_carries_it(self):
        clip = "A" * 40_000
        kept = [
            {
                "role": "user",
                "content": [{"type": "input_video", "input_video": {"data": clip}}],
            }
        ]
        assert self._recost(self._payload(clip), kept) > self._recost(
            self._payload(clip), [{"role": "user", "content": "text only"}]
        )

    def test_an_evicted_clip_stops_being_charged(self):
        """The whole point: once the turn is gone the round must not still reserve for it."""
        clip = "A" * 40_000
        evicted = self._recost(self._payload(clip), [{"role": "user", "content": "text only"}])
        no_video_at_all = self._recost(
            _Payload(
                messages = [{"role": "user", "content": "text only"}],
                video_base64 = None,
                audio_base64 = None,
                image_base64 = None,
            ),
            [{"role": "user", "content": "text only"}],
        )
        assert evicted == no_video_at_all
