# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The gallery's PNG encoder writes exactly Pillow's bytes, or steps aside so Pillow writes them."""

from __future__ import annotations

import io

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("PIL")
from PIL import Image  # noqa: E402
from PIL.PngImagePlugin import PngInfo  # noqa: E402

import core.inference.image_gallery as gallery  # noqa: E402
from core.inference import png_encode  # noqa: E402


@pytest.fixture(autouse = True)
def _fresh(monkeypatch, tmp_path):
    monkeypatch.delenv(png_encode.FAST_ENV, raising = False)
    monkeypatch.delenv(gallery.PNG_COMPRESS_LEVEL_ENV, raising = False)
    monkeypatch.setattr(png_encode, "_PROBED", {})
    monkeypatch.setattr(gallery, "studio_root", lambda: tmp_path)


def _fast_here():
    """True where this Python's zlib and Pillow's agree (the probe passes); else every image goes to Pillow.

    Pillow's macOS and Windows wheels can bundle a different zlib, so there the fast path steps aside.
    """
    saved = dict(png_encode._PROBED)
    png_encode._PROBED.clear()
    try:
        return png_encode._proven("RGB", 1)
    finally:
        png_encode._PROBED.clear()
        png_encode._PROBED.update(saved)


FAST_HERE = _fast_here()
needs_fast = pytest.mark.skipif(
    not FAST_HERE, reason = "Pillow's zlib differs from Python's here, so Pillow encodes every image"
)


def _pillow(image, **kwargs):
    buffer = io.BytesIO()
    image.save(buffer, format = "PNG", **kwargs)
    return buffer.getvalue()


def _pixels(
    height,
    width,
    bands,
    seed = 0,
):
    """Smooth rows, flat rows, repeated rows and noise rows, so every filter wins somewhere."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    smooth = np.stack(
        [128 + 90 * np.sin(xx / (13 + 5 * k) + yy / (29 + 3 * k)) for k in range(bands)], -1
    )
    pixels = np.clip(smooth + rng.normal(0, 3, smooth.shape), 0, 255).astype(np.uint8)
    if height > 8:
        pixels[height // 8 : height // 4] = 17
        pixels[height // 4 + 1 : height // 4 + 3] = pixels[height // 4]
        pixels[height // 2 : height // 2 + height // 4] = rng.integers(
            0, 256, pixels[height // 2 : height // 2 + height // 4].shape, dtype = np.uint8
        )
    return pixels[..., 0] if bands == 1 else pixels


def _image(
    mode,
    height,
    width,
    seed = 0,
):
    return Image.fromarray(_pixels(height, width, len(mode), seed), mode)


def _info():
    info = PngInfo()
    info.add_text("unsloth", '{"prompt": "a sloth", "seed": 7}')
    info.add_text("parameters", "a sloth\nSteps: 9, Seed: 7")
    return info


@pytest.mark.parametrize("mode", ["RGB", "RGBA", "L", "LA"])
@pytest.mark.parametrize("size", [(1, 1), (1, 9), (9, 1), (63, 130), (300, 257), (2, 17000)])
@pytest.mark.parametrize("level", [1, 6, 9])
@needs_fast
def test_bytes_match_pillow(mode, size, level):
    image = _image(mode, *size, seed = size[0] * 31 + size[1])
    before = png_encode.engaged
    data = png_encode.encode(image, pnginfo = _info(), compress_level = level)
    assert (
        png_encode._PROBED[(mode, level)] is True
    ), "the probe must hold where Pillow and zlib agree"
    assert data == _pillow(image, pnginfo = _info(), compress_level = level)
    assert png_encode.engaged == before + 1


@needs_fast
def test_bytes_match_pillow_without_text_and_across_idat_buffers():
    # 1500 rows of noise come to several 64 KiB IDAT buffers; a width over 16384 px widens the buffer.
    for size in ((1500, 400), (40, 20000)):
        image = Image.fromarray(
            np.random.default_rng(3).integers(0, 256, (*size, 3), dtype = np.uint8), "RGB"
        )
        assert png_encode.encode(image, compress_level = 1) == _pillow(image, compress_level = 1)


@needs_fast
def test_bytes_match_pillow_for_what_the_image_carries():
    image = _image("RGB", 40, 50)
    image.info["transparency"] = (1, 2, 3)
    image.info["icc_profile"] = bytes(range(128))
    info = PngInfo()
    info.add_text("unsloth", "中文 prompt")  # not latin-1: an iTXt chunk
    info.add(b"prVt", b"after the pixels", after_idat = True)
    data = png_encode.encode(image, pnginfo = info, compress_level = 1)
    assert data is not None
    assert data == _pillow(image, pnginfo = info, compress_level = 1)


@pytest.mark.parametrize("mode", ["RGB", "RGBA", "L", "LA"])
def test_the_probe_spans_more_than_one_idat_buffer(mode):
    data = _pillow(png_encode._probe_image(mode), compress_level = 9)
    assert sum(chunk[0] == b"IDAT" for chunk in png_encode._chunks(data)) >= 2


@needs_fast
def test_lazy_png_source_matches_pillow():
    source = io.BytesIO(_pillow(_image("RGBA", 70, 90), compress_level = 6))
    expected = _pillow(Image.open(io.BytesIO(source.getvalue())), compress_level = 1)
    assert png_encode.encode(Image.open(source), compress_level = 1) == expected


@pytest.mark.parametrize("mode", ["P", "1", "I;16", "CMYK"])
def test_other_modes_are_left_to_pillow(mode):
    assert png_encode.encode(Image.new(mode, (8, 8)), compress_level = 1) is None


def test_level_zero_is_left_to_pillow():
    # Stored blocks follow the encoder's output buffer sizes, which this path does not reproduce.
    assert png_encode.encode(_image("RGB", 64, 64), compress_level = 0) is None


def test_kill_switch(monkeypatch):
    monkeypatch.setenv(png_encode.FAST_ENV, "0")
    assert png_encode.encode(_image("RGB", 64, 64), compress_level = 1) is None


def test_a_failed_probe_keeps_pillow_for_that_mode_and_level(monkeypatch):
    real = png_encode._deflated

    def other_zlib(pixels, bpp, level):
        return real(pixels, bpp, 2 if level == 1 else 1)

    monkeypatch.setattr(png_encode, "_deflated", other_zlib)
    image = _image("RGB", 64, 64)
    assert png_encode.encode(image, compress_level = 1) is None
    assert png_encode._PROBED == {("RGB", 1): False}
    monkeypatch.setattr(png_encode, "_deflated", real)
    # Decided once per process: a later call does not probe again.
    assert png_encode.encode(image, compress_level = 1) is None


@needs_fast
def test_an_encode_error_falls_back(monkeypatch):
    image = _image("RGB", 64, 64)
    assert png_encode.encode(image, compress_level = 1) is not None

    def broken(*args, **kwargs):
        raise MemoryError("no room")

    monkeypatch.setattr(png_encode, "_filtered", broken)
    assert png_encode.encode(image, compress_level = 1) is None


@needs_fast
def test_the_filter_worker_stays_a_few_bands_ahead_of_deflate(monkeypatch):
    # Deflate is the slow side at high levels; a worker free to run ahead would hold the whole filtered image.
    import threading
    import time

    real = png_encode._filtered
    lock = threading.Lock()
    state = {"filtered": 0, "deflated": 0, "lead": 0}

    def counted(*args):
        out = real(*args)
        with lock:
            state["filtered"] += 1
            state["lead"] = max(state["lead"], state["filtered"] - state["deflated"])
        return out

    compressobj = png_encode.zlib.compressobj

    class SlowCompressor:
        def __init__(self, *args):
            self.inner = compressobj(*args)

        def compress(self, data):
            time.sleep(0.01)
            with lock:
                state["deflated"] += 1
            return self.inner.compress(data)

        def flush(self):
            return self.inner.flush()

    image = _image("RGB", 1200, 400)  # 1200 rows of 1201 bytes: 8 bands
    expected = _pillow(image, compress_level = 1)
    assert png_encode.encode(image, compress_level = 1) == expected  # probe with the real zlib
    monkeypatch.setattr(png_encode, "_filtered", counted)
    monkeypatch.setattr(png_encode.zlib, "compressobj", SlowCompressor)
    assert png_encode.encode(image, compress_level = 1) == expected
    assert state["filtered"] == state["deflated"] >= 6
    # The band being deflated plus the ones queued behind it, however slow deflate is.
    assert state["lead"] <= png_encode._AHEAD + 1


def _meta(index = 0):
    return {
        "prompt": "a sloth",
        "width": 300,
        "height": 257,
        "steps": 9,
        "guidance": 0.0,
        "seed": 7 + index,
        "batch_index": index,
        "model": "m",
        "created_at": 100.0,
    }


@pytest.mark.parametrize("level", ["1", "6"])
def test_gallery_writes_pillows_bytes(monkeypatch, level):
    monkeypatch.setenv(gallery.PNG_COMPRESS_LEVEL_ENV, level)
    images = [_image("RGB", 257, 300, seed = i) for i in range(4)]
    before = png_encode.engaged
    records = list(gallery.save_many((image, _meta(i)) for i, image in enumerate(images)))
    # Where the probe fails every image goes to Pillow; the files must be Pillow's bytes either way.
    assert png_encode.engaged == before + (4 if FAST_HERE else 0)
    monkeypatch.setenv(png_encode.FAST_ENV, "0")
    for i, (image, record) in enumerate(zip(images, records)):
        written = gallery.image_path(record["id"]).read_bytes()
        recipe = gallery._recipe(_meta(i))
        assert written == gallery._png_bytes(image, recipe)
        with Image.open(io.BytesIO(written)) as im:
            assert im.tobytes() == image.tobytes()


def test_gallery_falls_back_to_pillow_when_the_fast_path_steps_aside(monkeypatch):
    monkeypatch.setattr(png_encode, "encode", lambda *a, **k: None)
    image = _image("RGB", 64, 64)
    record = gallery.save(image, _meta())
    written = gallery.image_path(record["id"]).read_bytes()
    info = PngInfo()
    recipe = gallery._recipe(_meta())
    import json

    info.add_text("unsloth", json.dumps(recipe))
    info.add_text("parameters", gallery._params_text(recipe))
    assert written == _pillow(image, pnginfo = info, compress_level = 1)


def test_encode_returns_pillows_bytes_or_steps_aside_on_every_platform():
    # Runs everywhere: a served result is always Pillow's bytes, and a failed probe serves nothing.
    image = _image("RGBA", 120, 90, seed = 5)
    data = png_encode.encode(image, pnginfo = _info(), compress_level = 1)
    if FAST_HERE:
        assert data == _pillow(image, pnginfo = _info(), compress_level = 1)
    else:
        assert data is None
