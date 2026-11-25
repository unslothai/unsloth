# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Earlier turns' pictures are resent every turn: the transformers / MLX decode is remembered."""

import asyncio
import base64
import hashlib
import threading
from io import BytesIO
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from PIL import Image

from core.inference.orchestrator import PNG_B64_STAMP, InferenceOrchestrator, _encoded_images
from routes import inference as route


def _orchestrator(max_image_side = None):
    backend = InferenceOrchestrator.__new__(InferenceOrchestrator)
    backend.models = {"m": {"is_vision": True, "max_image_side": max_image_side}}
    backend.active_model_name = "m"
    return backend


def _encoded(
    image,
    fmt = "JPEG",
    **save,
):
    buf = BytesIO()
    image.save(buf, format = fmt, **save)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _photo(
    width = 1200,
    height = 900,
    seed = 0,
):
    noise = Image.effect_noise((width, height), 40 + seed).convert("RGB")
    return Image.blend(
        Image.linear_gradient("L").resize((width, height)).convert("RGB"), noise, 0.3
    )


def _fresh_png(image) -> str:
    buf = BytesIO()
    image.save(buf, format = "PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


@pytest.fixture(autouse = True)
def _empty_memo(monkeypatch):
    monkeypatch.setattr(route, "_HISTORY_IMAGE_MEMO", type(route._HISTORY_IMAGE_MEMO)())
    monkeypatch.setattr(route, "_history_image_memo_bytes", 0)


@pytest.fixture
def decodes(monkeypatch):
    seen = []
    real = route._decoded_resized_image

    def counted(backend, encoded):
        seen.append(encoded)
        return real(backend, encoded)

    monkeypatch.setattr(route, "_decoded_resized_image", counted)
    return seen


def _payload(backend, encoded):
    return _encoded_images(
        [route._decode_and_resize_image(backend, encoded)], backend._pil_to_base64
    )[0]


@pytest.mark.parametrize(
    "encoded",
    [
        _encoded(_photo()),
        _encoded(_photo(seed = 1).convert("RGBA"), "PNG"),
        _encoded(_photo(300, 200), "PNG"),
    ],
    ids = ["jpeg", "rgba_png", "small_png"],
)
def test_the_worker_payload_is_the_same_with_and_without_the_memo(encoded, decodes):
    backend = _orchestrator()
    uncached = route._decoded_resized_image(backend, encoded)
    expected = _fresh_png(uncached)

    first = route._decode_and_resize_image(backend, encoded)
    second = route._decode_and_resize_image(backend, encoded)

    assert backend._pil_to_base64(first) == backend._pil_to_base64(second) == expected
    assert first.tobytes() == second.tobytes() == uncached.tobytes()
    assert first.mode == second.mode == "RGB" and first.size == second.size == uncached.size
    # One decode for the two calls (the reference decode above is counted too).
    assert len(decodes) == 2


def test_a_handed_out_picture_is_a_private_copy(decodes):
    backend = _orchestrator()
    encoded = _encoded(_photo())
    first = route._decode_and_resize_image(backend, encoded)
    first.paste((255, 0, 0), (0, 0, 50, 50))
    second = route._decode_and_resize_image(backend, encoded)
    assert first is not second
    assert second.getpixel((0, 0)) != (255, 0, 0)
    assert backend._pil_to_base64(second) == _fresh_png(second)
    assert len(decodes) == 1


def test_a_different_resize_side_misses(decodes):
    encoded = _encoded(_photo(2400, 1800))
    default = route._decode_and_resize_image(_orchestrator(), encoded)
    ocr = route._decode_and_resize_image(_orchestrator(max_image_side = 2048), encoded)
    assert default.size == (800, 600)
    assert ocr.size == (2048, 1536)
    assert len(decodes) == 2
    assert _orchestrator()._pil_to_base64(ocr) == _fresh_png(ocr)


def test_a_backend_that_does_not_name_its_side_is_not_memoized(decodes):
    backend = SimpleNamespace(resize_image = lambda img: img)
    encoded = _encoded(_photo(64, 48))
    route._decode_and_resize_image(backend, encoded)
    route._decode_and_resize_image(backend, encoded)
    assert len(decodes) == 2
    assert not route._HISTORY_IMAGE_MEMO


def test_the_memo_is_bounded_by_count_and_bytes(monkeypatch):
    backend = _orchestrator()
    monkeypatch.setattr(route, "_HISTORY_IMAGE_MEMO_MAX_ENTRIES", 2)
    # Solid colours: entries are near the same size, so the byte budget below holds exactly one.
    for seed in range(4):
        route._decode_and_resize_image(
            backend, _encoded(Image.new("RGB", (64, 48), (seed, 0, 0)), "PNG")
        )
    assert len(route._HISTORY_IMAGE_MEMO) == 2
    assert route._history_image_memo_bytes == sum(v[2] for v in route._HISTORY_IMAGE_MEMO.values())

    budget = max(v[2] for v in route._HISTORY_IMAGE_MEMO.values()) * 3 // 2
    monkeypatch.setattr(route, "_HISTORY_IMAGE_MEMO_MAX_ENTRIES", 64)
    monkeypatch.setattr(route, "_HISTORY_IMAGE_MEMO_MAX_BYTES", budget)
    route._decode_and_resize_image(backend, _encoded(Image.new("RGB", (64, 48), (9, 0, 0)), "PNG"))
    assert len(route._HISTORY_IMAGE_MEMO) == 1
    assert route._history_image_memo_bytes <= budget


def test_the_least_recently_used_picture_is_evicted_first(monkeypatch, decodes):
    backend = _orchestrator()
    monkeypatch.setattr(route, "_HISTORY_IMAGE_MEMO_MAX_ENTRIES", 2)
    a, b, c = (_encoded(_photo(64, 48, seed), "PNG") for seed in range(3))
    route._decode_and_resize_image(backend, a)
    route._decode_and_resize_image(backend, b)
    route._decode_and_resize_image(backend, a)  # a is now the most recent
    route._decode_and_resize_image(backend, c)  # evicts b
    route._decode_and_resize_image(backend, a)
    assert decodes == [a, b, c]


def test_a_corrupt_picture_raises_every_time_and_is_not_remembered(decodes):
    backend = _orchestrator()
    truncated = base64.b64encode(base64.b64decode(_encoded(_photo()))[:4000]).decode("ascii")
    errors = []
    for _ in range(2):
        with pytest.raises(Exception) as exc:
            route._decode_and_resize_image(backend, truncated)
        errors.append((type(exc.value), str(exc.value)))
    assert errors[0] == errors[1]
    assert len(decodes) == 2
    assert not route._HISTORY_IMAGE_MEMO


def test_an_exif_rotated_photo_is_upright_on_a_hit_too():
    exif = Image.Exif()
    exif[0x0112] = 6
    encoded = _encoded(_photo(1200, 600), exif = exif.tobytes())
    backend = _orchestrator()
    expected = _fresh_png(route._decoded_resized_image(backend, encoded))
    first = route._decode_and_resize_image(backend, encoded)
    second = route._decode_and_resize_image(backend, encoded)
    assert first.size == second.size == (400, 800)
    assert backend._pil_to_base64(first) == backend._pil_to_base64(second) == expected


def test_a_stamp_that_no_longer_matches_the_picture_is_ignored():
    image = _photo(64, 48)
    setattr(image, PNG_B64_STAMP, (("RGB", (1, 1)), "stale"))
    assert InferenceOrchestrator._pil_to_base64(image) == _fresh_png(image)


def test_a_multi_image_turn_decodes_only_the_new_picture(decodes):
    backend = _orchestrator()
    history = [_encoded(_photo(640, 480, seed)) for seed in range(3)]
    for turn in range(1, 4):
        asyncio.run(route._decode_request_images(backend, history[:turn]))
    assert decodes == history


def test_durable_run_creation_hashes_images_off_the_event_loop(monkeypatch):
    from routes import chat_generation_runs as runs_route

    url = "data:image/png;base64," + _encoded(_photo(32, 32), "PNG")
    sanitized = {
        "model": "local",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "hi"},
                    {"type": "image_url", "image_url": {"url": url}},
                ],
            },
            {"role": "user", "content": "again"},
        ],
    }
    monkeypatch.setattr(runs_route, "_sanitize_request", lambda payload, request = None: sanitized)
    seen = {}
    real = runs_route.detach_embedded_images

    def watched(payload):
        seen["thread"] = threading.get_ident()
        seen["images"] = real(payload)
        return seen["images"]

    monkeypatch.setattr(runs_route, "detach_embedded_images", watched)

    def stop(_request):
        raise HTTPException(status_code = 418)

    monkeypatch.setattr(route, "_request_used_api_key", stop)

    async def call():
        seen["loop_thread"] = threading.get_ident()
        with pytest.raises(HTTPException) as exc:
            await runs_route.create_chat_generation_run(None, None, "alice")
        assert exc.value.status_code == 418

    asyncio.run(call())
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
    assert seen["thread"] != seen["loop_thread"]
    assert seen["images"] == {digest: url}
    assert sanitized["messages"][0]["content"][1]["image_url"]["url"] == digest
