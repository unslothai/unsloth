# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Images are charged, not dropped, when a local GGUF chat is fitted and its tool results sized (#14968)."""

import base64
import io
import os
import sys
import types

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.inference import context_window as cw  # noqa: E402
from core.inference import llama_cpp  # noqa: E402

PIL = pytest.importorskip("PIL.Image")


def _png(width: int, height: int) -> str:
    buffer = io.BytesIO()
    PIL.new("RGB", (width, height)).save(buffer, format = "PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def _image(url: str) -> dict:
    return {"type": "image_url", "image_url": {"url": url}}


def _flat(price: int):
    return lambda part: price


def _backend(
    projector = "qwen3vl_merger",
    extra_args = None,
    vision = True,
):
    return types.SimpleNamespace(
        is_vision = vision,
        _mmproj_projector_type = projector,
        _extra_args = extra_args or [],
    )


def test_an_image_becomes_a_stand_in_of_its_price():
    message = {"role": "user", "content": [{"type": "text", "text": "look"}, _image("x")]}

    (priced,) = cw.messages_with_priced_images([message], _flat(7))

    assert priced["content"][0] == {"type": "text", "text": "look"}
    assert priced["content"][1] == {"type": "text", "text": " the" * 7}
    # The request keeps the real image.
    assert message["content"][1]["type"] == "image_url"


def test_audio_and_video_stay_unpriced():
    message = {
        "role": "user",
        "content": [
            {"type": "text", "text": "hi"},
            {"type": "input_audio", "input_audio": {"data": "AAAA"}},
            {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,AAAA"}},
        ],
    }

    (priced,) = cw.messages_with_priced_images([message], _flat(7))

    assert priced["content"] == [{"type": "text", "text": "hi"}]


def test_a_chat_without_media_is_returned_as_is():
    messages = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]

    assert cw.messages_with_priced_images(messages, _flat(7)) is messages


def test_the_estimate_adds_each_image_price():
    message = {
        "role": "user",
        "content": [{"type": "text", "text": "look"}, _image("x"), _image("y")],
    }

    text_only = cw.estimate_message_tokens_without_unpriced_media(message)

    assert cw.estimate_message_tokens_with_priced_images(message, _flat(500)) == text_only + 1000


def test_image_dimensions_reads_an_inline_png():
    assert cw.image_dimensions(_image(_png(64, 48))) == (64, 48)
    assert cw.image_dimensions({"type": "input_image", "image_url": _png(10, 20)}) == (10, 20)


def test_image_dimensions_gives_up_on_what_it_cannot_read_cheaply():
    assert cw.image_dimensions(_image("https://example.com/a.png")) is None
    assert cw.image_dimensions(_image("data:image/png;base64,bm90IGFuIGltYWdl")) is None


def test_a_qwen3_vl_screenshot_is_priced_from_its_size():
    price = llama_cpp._image_token_price(_backend(), _image(_png(1920, 1080)))

    assert price == 60 * 34 + llama_cpp._IMAGE_WRAPPER_TOKENS


def test_image_max_tokens_caps_the_price():
    backend = _backend(extra_args = ["--image-max-tokens", "1024"])

    price = llama_cpp._image_token_price(backend, _image(_png(1920, 1080)))

    assert price == 1024 + llama_cpp._IMAGE_WRAPPER_TOKENS


def test_the_price_never_exceeds_the_admission_ceiling():
    backend = _backend()
    ceiling = llama_cpp._mcp_image_tokens(backend)

    assert llama_cpp._image_token_price(backend, _image(_png(8000, 8000))) == ceiling


def test_an_unreadable_image_or_unknown_projector_is_charged_the_ceiling():
    unknown = _backend(projector = "some_future_projector")

    assert llama_cpp._image_token_price(_backend(), _image("https://x/y.png")) == (
        llama_cpp._mcp_image_tokens(_backend())
    )
    assert llama_cpp._image_token_price(unknown, _image(_png(64, 64))) == (
        llama_cpp._mcp_image_tokens(unknown)
    )


def test_a_text_only_model_is_charged_nothing():
    assert llama_cpp._image_token_price(_backend(vision = False), _image(_png(64, 64))) == 0


def _screenshot_chat(rounds: int) -> list[dict]:
    """The report's shape: one message with ten screenshots, then a long tool loop."""
    screenshot = _png(1920, 1080)
    messages: list[dict] = [
        {"role": "system", "content": "You are helpful."},
        {
            "role": "user",
            "content": [{"type": "text", "text": "Read these."}]
            + [_image(screenshot) for _ in range(10)],
        },
    ]
    for index in range(rounds):
        call_id = f"call_{index}"
        messages.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": "read_file", "arguments": f'{{"path": "f{index}"}}'},
                    }
                ],
            }
        )
        messages.append({"role": "tool", "tool_call_id": call_id, "content": "line " * 4000})
    return messages


def _fit(messages, backend, priced: bool):
    estimate = (
        backend._estimate_countable_message
        if priced
        else cw.estimate_message_tokens_without_unpriced_media
    )
    countable = backend._countable_messages if priced else cw.messages_without_unpriced_media

    def count(fitted):
        return sum(estimate(message) for message in countable(fitted))

    return cw.fit_rolling_context(
        messages,
        context_length = 65536,
        max_tokens = 32768,
        count_tokens = count,
        estimate_message = estimate,
    )


def test_screenshots_that_outgrow_the_reply_reserve_now_trigger_compaction():
    backend = _backend()
    backend._price_image = lambda part: llama_cpp._image_token_price(backend, part)
    backend._countable_messages = lambda messages: cw.messages_with_priced_images(
        messages, backend._price_image
    )
    backend._estimate_countable_message = (
        lambda message: cw.estimate_message_tokens_with_priced_images(message, backend._price_image)
    )
    messages = _screenshot_chat(rounds = 9)

    _, unpriced = _fit(messages, backend, priced = False)
    fitted, priced = _fit(messages, backend, priced = True)

    # Text alone (~46k) is under the 49k prompt budget, so the old count kept everything.
    assert unpriced is None
    # With the images (~21.7k) charged, older tool rounds go and the screenshots stay.
    assert priced and priced["dropped_messages"] > 0
    assert any(
        isinstance(m.get("content"), list) and m["content"][-1].get("type") == "image_url"
        for m in fitted
    )


def test_the_backend_counts_images_through_its_own_helpers(monkeypatch):
    monkeypatch.setattr(llama_cpp.LlamaCppBackend, "is_vision", property(lambda self: True))
    backend = llama_cpp.LlamaCppBackend.__new__(llama_cpp.LlamaCppBackend)
    backend._mmproj_projector_type = "qwen3vl_merger"
    backend._extra_args = []
    message = {"role": "user", "content": [_image(_png(1920, 1080))]}

    (priced,) = backend._countable_messages([message])

    assert priced["content"][0]["text"] == " the" * (60 * 34 + llama_cpp._IMAGE_WRAPPER_TOKENS)
    assert backend._estimate_countable_message(message) >= 60 * 34
