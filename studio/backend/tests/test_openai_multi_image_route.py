# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Serving every image a user turn carried to a backend whose template can mark them.

Flattening each turn to a string kept one base64 for the whole thread, so a second image on a turn
was refused and an earlier turn's silently replaced by the newest.
"""

import base64
import io
import os
import sys

import pytest
from fastapi import HTTPException
from PIL import Image

from models.inference import ChatMessage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import test_sf_client_tools_passthrough as passthrough  # noqa: E402


def _part(image, container = "PNG") -> dict:
    buffer = io.BytesIO()
    image.save(buffer, format = container)
    encoded = base64.b64encode(buffer.getvalue()).decode()
    return {"type": "image_url", "image_url": {"url": f"data:image/x;base64,{encoded}"}}


def _sized(size: int) -> dict:
    return _part(Image.new("RGB", (size, size), "white"))


def _text(text: str) -> dict:
    return {"type": "text", "text": text}


_NOTHING_TO_SEND = {"type": "image_url", "image_url": {"url": "data:image/png;base64,"}}
_ASK = _text("which is which?")
# Small enough to skip resampling, so nothing but the route itself forces the decode.
_WHOLE = _part(Image.new("RGB", (64, 64), "blue"), container = "JPEG")["image_url"]["url"]
_TRUNCATED = {"type": "image_url", "image_url": {"url": _WHOLE[: len(_WHOLE) - 12]}}
_ACROSS_TURNS = [
    ChatMessage(role = "user", content = [_sized(2), _text("one")]),
    ChatMessage(role = "assistant", content = [_sized(8), _text("mine")]),
    ChatMessage(role = "user", content = [_text("two"), _sized(4)]),
]


def _markers(message) -> list[int]:
    content = message.get("content")
    if not isinstance(content, list):
        return []
    return [i for i, part in enumerate(content) if part.get("type") == "image"]


def _call(
    monkeypatch,
    messages,
    accepts_multiple_images = True,
    **request_kwargs,
):
    backend = passthrough._ScriptedBackend(passthrough._fixed("an answer"))
    backend.models["sf-model"].update(
        is_vision = True,
        chat_template_info = {
            "template": "<tool_call> chatml",
            "renders_image": True,
            "accepts_multiple_images": accepts_multiple_images,
        },
    )
    payload = passthrough._request(messages = messages, stream = False, **request_kwargs)
    passthrough._call(payload, monkeypatch, backend)
    return backend


@pytest.mark.parametrize(
    "messages, sizes, markers",
    [
        ([ChatMessage(role = "user", content = [_sized(2), _sized(4), _ASK])], [2, 4], [[0, 1]]),
        ([ChatMessage(role = "user", content = [_sized(4), _sized(2), _ASK])], [4, 2], [[0, 1]]),
        (_ACROSS_TURNS, [2, 4], [[0], [], [1]]),
    ],
    ids = ["one turn", "one turn, reversed", "earlier turns, past an assistant's own image"],
)
def test_every_image_a_user_turn_carried_is_served_in_document_order_at_that_turn(
    monkeypatch, messages, sizes, markers
):
    """An image on an earlier turn used to be replaced by the newest one, with no error at all."""
    call = _call(monkeypatch, messages).calls[0]

    assert [image.width for image in call["images"]] == sizes
    assert call["image"] is None
    # Per turn, because binding is positional: a marker on the wrong turn is the wrong picture.
    assert [
        _markers(message) for message in call["messages"] if message.get("role") != "system"
    ] == markers


@pytest.mark.parametrize(
    "content, accepts_multiple_images, detail",
    [
        ([_sized(2), _sized(4), _ASK], False, "one image per message"),
        ([_sized(2), _NOTHING_TO_SEND, _ASK], True, "one image per message"),
        ([_TRUNCATED], True, ""),
    ],
    ids = ["a backend that published one image", "an image part with nothing to send", "truncated"],
)
def test_what_cannot_be_served_is_answered_as_a_bad_request(
    monkeypatch, content, accepts_multiple_images, detail
):
    """A part yielding no payload is still an image to the renderer, so the turn stays a 400."""
    turn = [ChatMessage(role = "user", content = content)]
    with pytest.raises(HTTPException) as refusal:
        _call(monkeypatch, turn, accepts_multiple_images = accepts_multiple_images)

    assert refusal.value.status_code == 400
    assert detail in refusal.value.detail


@pytest.mark.parametrize(
    "mode, container", [("CMYK", "JPEG"), ("PA", "TIFF"), ("LAB", "TIFF"), ("RGBA", "TIFF")]
)
def test_an_image_reaches_the_backend_in_a_mode_it_can_be_handed_over_in(
    monkeypatch, mode, container
):
    """The first three reopen in a mode PNG cannot write and must convert; RGBA must not."""
    part = _part(Image.new(mode, (16, 16)), container = container)

    delivered = _call(monkeypatch, [ChatMessage(role = "user", content = [part])]).calls[0]["image"]

    assert delivered.mode == ("RGBA" if mode == "RGBA" else "RGB")
    # The boundary itself, not a claim about which modes it takes.
    delivered.save(io.BytesIO(), format = "PNG")


def test_each_image_is_decoded_once_for_a_request_that_rebuilds_the_conversation(monkeypatch):
    """A client-tool request renders a rebuilt conversation, which used to decode the whole set a
    second time -- twice the work and both full sets resident, on exactly the largest requests."""
    from routes import inference as route

    decoded = []
    real = route._decode_and_resize_image

    def _watch(backend, encoded):
        decoded.append(encoded)
        return real(backend, encoded)

    monkeypatch.setattr(route, "_decode_and_resize_image", _watch)
    turn = [ChatMessage(role = "user", content = [_sized(2), _sized(4), _sized(6), _ASK])]

    _call(monkeypatch, turn, tools = [passthrough.LOOKUP_TOOL])

    assert len(decoded) == len(set(decoded)) == 3


def test_a_16_bit_image_keeps_its_levels_instead_of_clipping_to_white(monkeypatch):
    """Converting reads 0..65535 as 8-bit, so everything above 255 lands on white."""
    source = Image.new("I;16", (2, 2))
    source.putdata([0, 20000, 40000, 65535])
    turn = [ChatMessage(role = "user", content = [_part(source)])]

    delivered = _call(monkeypatch, turn).calls[0]["image"]

    assert [pixel[0] for pixel in delivered.convert("RGB").get_flattened_data()] == [
        0,
        77,
        155,
        255,
    ]
