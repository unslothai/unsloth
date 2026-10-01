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

    # getpixel, like the multi-image test below, because the flattened read is spelled
    # getdata() on Pillow 10 and 11 and get_flattened_data() only from 12. This file runs on
    # both: no-torch-runtime.txt pins pillow 12.3.0 from Python 3.10 and 11.3.0 below it, so
    # the 12-only spelling made this the one test in the file that could not.
    levels = delivered.convert("RGB")
    assert [levels.getpixel(at)[0] for at in ((0, 0), (1, 0), (0, 1), (1, 1))] == [0, 77, 155, 255]


def _as_image(delivered):
    """A served entry, whether it rides as pixels or as the base64 they came in."""
    if isinstance(delivered, str):
        return Image.open(io.BytesIO(base64.b64decode(delivered)))
    return delivered


def test_a_16_bit_image_on_a_multi_image_turn_keeps_its_levels_too(monkeypatch):
    """The single-image path had this guard and the multi-image one did not, which is how the
    route came to hand generation the caller's raw base64 instead of what it had just decoded.
    Only `_decode_and_resize_image` scales 0..65535 down; the worker's own decode does not, so
    the picture arrived clipped to white with every structural assertion still green.

    Asserted on the pixels rather than on the type, so the claim survives a transport that
    carries base64 again -- what must not come back is the lost conversion.
    """
    source = Image.new("I;16", (2, 2))
    source.putdata([0, 20000, 40000, 65535])
    turn = [ChatMessage(role = "user", content = [_part(source), _sized(4), _ASK])]

    served = _call(monkeypatch, turn).calls[0]["images"]

    assert len(served) == 2
    levels = _as_image(served[0]).convert("RGB")
    # getpixel, not the flattened read the single-image test uses: that one is spelled
    # getdata() on Pillow 10 and get_flattened_data() on 12, and this runs on both.
    assert [levels.getpixel(at)[0] for at in ((0, 0), (1, 0), (0, 1), (1, 1))] == [0, 77, 155, 255]


_PDF = {
    "type": "input_document",
    "file_data": "data:application/pdf;base64,JVBERi0xLjQK",
    "filename": "spec.pdf",
}


def test_a_document_part_in_history_never_reaches_the_local_template(monkeypatch):
    """Local templates take text and image only: mistral3's raises "Only text and image blocks
    are supported in message content!", and mlx_inference re-raises that instead of recovering
    whenever the request carries tools or a reasoning knob, so the turn 500s."""
    messages = [
        ChatMessage(role = "user", content = [_text("here is the spec"), _PDF, _sized(2)]),
        ChatMessage(role = "assistant", content = "Got it."),
        ChatMessage(role = "user", content = [_sized(4), _ASK]),
    ]
    call = _call(monkeypatch, messages).calls[0]

    kept = [
        part.get("type")
        for message in call["messages"]
        if isinstance(message.get("content"), list)
        for part in message["content"]
    ]
    assert "input_document" not in kept
    # Dropping the document must not drop the text beside it, nor either picture.
    assert [image.width for image in call["images"]] == [2, 4]
    assert "here is the spec" in str(call["messages"])


def test_a_request_beyond_the_image_budget_is_refused_before_decoding(monkeypatch):
    """Each retained raster is up to 1.92 MB while a solid 800x800 PNG is ~4.8 KB of base64, so
    an unbounded list reaches gigabytes from a body small enough to pass any transport limit."""
    from routes import inference as inference_route

    monkeypatch.setattr(inference_route, "_MAX_SERVED_IMAGES", 3)
    decoded = []
    real = inference_route._decode_and_resize_image

    def counted(backend, encoded):
        decoded.append(encoded)
        return real(backend, encoded)

    monkeypatch.setattr(inference_route, "_decode_and_resize_image", counted)
    content = [_part(Image.new("RGB", (8 + i, 8), "white")) for i in range(4)] + [_ASK]
    with pytest.raises(HTTPException) as exc:
        _call(monkeypatch, [ChatMessage(role = "user", content = content)])
    assert exc.value.status_code == 400
    assert "carries 4 images" in str(exc.value.detail)
    assert "at most 3 are served per request" in str(exc.value.detail)
    # Refused before allocating any of them, which is the point.
    assert decoded == []


def test_a_request_at_the_image_budget_is_served(monkeypatch):
    from routes import inference as inference_route

    monkeypatch.setattr(inference_route, "_MAX_SERVED_IMAGES", 3)
    content = [_part(Image.new("RGB", (8 + i, 8), "white")) for i in range(3)] + [_ASK]
    call = _call(monkeypatch, [ChatMessage(role = "user", content = content)]).calls[0]
    assert [image.width for image in call["images"]] == [8, 9, 10]


def test_the_image_budget_counts_repeats_not_distinct_payloads(monkeypatch):
    """The worker decodes every entry of images_base64, so a payload repeated N times costs N
    rasters there even though this route reuses one PIL object for all of them."""
    from routes import inference as inference_route

    monkeypatch.setattr(inference_route, "_MAX_SERVED_IMAGES", 3)
    same = _part(Image.new("RGB", (8, 8), "white"))
    content = [same, same, same, same, _ASK]
    with pytest.raises(HTTPException) as exc:
        _call(monkeypatch, [ChatMessage(role = "user", content = content)])
    assert exc.value.status_code == 400
    assert "carries 4 images" in str(exc.value.detail)


def test_the_image_budget_refusal_goes_through_the_callers_reject(monkeypatch):
    """_reject fails the open API monitor row before raising; a raw HTTPException would leave
    it reported as running."""
    import asyncio

    from routes import inference as inference_route

    monkeypatch.setattr(inference_route, "_MAX_SERVED_IMAGES", 1)
    seen = []

    def reject(status_code, detail):
        seen.append((status_code, detail))
        return HTTPException(status_code = status_code, detail = detail)

    with pytest.raises(HTTPException):
        asyncio.run(inference_route._decode_request_images(None, ["a", "b"], None, reject = reject))
    assert len(seen) == 1 and seen[0][0] == 400
