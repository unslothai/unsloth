# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Pillow's exact PNG bytes, with ZipEncode.c's per-row filter choice done in numpy beside deflate.

Used per (mode, level) only after a probe matched Pillow byte for byte in this process; UNSLOTH_IMAGE_PNG_FAST=0 off."""

from __future__ import annotations

import io
import os
import struct
import threading
import zlib
from typing import Any, Iterator, Optional

FAST_ENV = "UNSLOTH_IMAGE_PNG_FAST"
# mode -> (bytes per pixel, IHDR colour type)
_MODES = {"L": (1, 0), "LA": (2, 4), "RGB": (3, 2), "RGBA": (4, 6)}
# Smaller bands made a batch's concurrent encodes queue on the GIL (32 rows of 512 px: 4 at once 1.6x slower).
_BAND_BYTES = 192 * 1024
_AHEAD = 2
_PROBE_LOCK = threading.Lock()
_PROBED: dict[tuple[str, int], bool] = {}
engaged = 0
_ENGAGED_LOCK = threading.Lock()


def enabled() -> bool:
    return (os.environ.get(FAST_ENV) or "").strip().lower() not in ("0", "false", "no", "off")


def _filtered(pixels: Any, bpp: int, start: int, stop: int) -> bytes:
    """Rows [start, stop) as ZipEncode.c filters them: None, Up, Sub, Paeth, each kept only on a strictly lower
    sum of |signed byte|."""
    import numpy as np

    x = pixels[start:stop]
    if start:
        prev = pixels[start - 1 : stop - 1]
    else:
        prev = np.empty_like(x)
        prev[0] = 0
        prev[1:] = x[:-1]
    n, stride = x.shape

    def cost(v: Any) -> Any:
        return np.abs(v.view(np.int8), dtype = np.int16).sum(axis = 1, dtype = np.int64)

    up = x - prev
    sub = x.copy()
    np.subtract(x[:, bpp:], x[:, :-bpp], out = sub[:, bpp:])
    a = np.zeros((n, stride), np.int16)
    a[:, bpp:] = x[:, :-bpp]
    b = prev.astype(np.int16)
    c = np.zeros((n, stride), np.int16)
    c[:, bpp:] = prev[:, :-bpp]
    pa = np.abs(b - c)
    pb = np.abs(a - c)
    pc = np.abs(a + b - 2 * c)
    paeth = x - np.where((pa <= pb) & (pa <= pc), a, np.where(pb <= pc, b, c)).astype(np.uint8)

    out = np.empty((n, stride + 1), np.uint8)
    out[:, 0] = 0
    out[:, 1:] = x
    best = cost(x)
    for code, rows in ((2, up), (1, sub), (4, paeth)):
        score = cost(rows)
        better = score < best
        if better.any():
            best = np.where(better, score, best)
            out[better, 0] = code
            out[better, 1:] = rows[better]
    return out.tobytes()


def _deflated(pixels: Any, bpp: int, level: int) -> Iterator[bytes]:
    height, stride = pixels.shape
    compressor = zlib.compressobj(level, zlib.DEFLATED, 15, 9, zlib.Z_FILTERED)
    rows = max(1, _BAND_BYTES // (stride + 1))
    bands = [(start, min(height, start + rows)) for start in range(0, height, rows)]
    if len(bands) < 3:
        for start, stop in bands:
            yield compressor.compress(_filtered(pixels, bpp, start, stop))
    else:
        from collections import deque
        from concurrent.futures import ThreadPoolExecutor
        from itertools import islice

        # Bounded lead: deflate is slower at high levels, an unbounded worker would hold the whole filtered image.
        with ThreadPoolExecutor(max_workers = 1, thread_name_prefix = "png-filter") as pool:
            todo = iter(bands)
            futures = deque(
                pool.submit(_filtered, pixels, bpp, *band) for band in islice(todo, _AHEAD)
            )
            try:
                while futures:
                    data = futures.popleft().result()
                    band = next(todo, None)
                    if band is not None:
                        futures.append(pool.submit(_filtered, pixels, bpp, *band))
                    yield compressor.compress(data)
            finally:
                for future in futures:
                    future.cancel()
    yield compressor.flush()


def _chunks(data: bytes) -> Optional[list[tuple[bytes, int, int]]]:
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    out = []
    pos = 8
    while pos + 12 <= len(data):
        (length,) = struct.unpack(">I", data[pos : pos + 4])
        out.append((data[pos + 4 : pos + 8], pos, length))
        pos += 12 + length
    return out if pos == len(data) else None


def _framing(image: Any, mode: str, save_kwargs: dict) -> Optional[tuple[bytes, bytes]]:
    """Pillow's bytes around the IDATs: a 1x1 crop (keeps ``info``) saved with the same options, real size in IHDR."""
    buffer = io.BytesIO()
    image.crop((0, 0, 1, 1)).save(buffer, format = "PNG", **save_kwargs)
    data = buffer.getvalue()
    chunks = _chunks(data)
    if not chunks or chunks[0][0] != b"IHDR" or chunks[0][2] != 13:
        return None
    idat = [index for index, chunk in enumerate(chunks) if chunk[0] == b"IDAT"]
    if not idat or idat != list(range(idat[0], idat[-1] + 1)):
        return None
    ihdr = bytearray(data[16:29])
    if bytes(ihdr[8:13]) != bytes((8, _MODES[mode][1], 0, 0, 0)):
        return None
    ihdr[0:8] = struct.pack(">II", image.width, image.height)
    header = (
        data[:8]
        + struct.pack(">I", 13)
        + b"IHDR"
        + bytes(ihdr)
        + struct.pack(">I", zlib.crc32(b"IHDR" + bytes(ihdr)))
    )
    first, last = chunks[idat[0]], chunks[idat[-1]]
    return header + data[33 : first[1]], data[last[1] + 12 + last[2] :]


def _encode(image: Any, mode: str, level: int, save_kwargs: dict) -> Optional[bytes]:
    import numpy as np
    from PIL import ImageFile

    framing = _framing(image, mode, save_kwargs)
    if framing is None:
        return None
    pixels = np.asarray(image)
    bpp = _MODES[mode][0]
    pixels = np.ascontiguousarray(pixels.reshape(image.height, image.width * bpp), dtype = np.uint8)
    # ImageFile._save's buffer: one full IDAT per buffer, the last one short.
    bufsize = max(int(ImageFile.MAXBLOCK), image.width * 4)
    out = io.BytesIO()
    out.write(framing[0])
    pending = bytearray()

    def idat(piece: Any) -> None:
        out.write(struct.pack(">I", len(piece)) + b"IDAT")
        out.write(piece)
        out.write(struct.pack(">I", zlib.crc32(piece, zlib.crc32(b"IDAT"))))

    for data in _deflated(pixels, bpp, level):
        pending += data
        while len(pending) >= bufsize:
            idat(pending[:bufsize])
            del pending[:bufsize]
    if pending:
        idat(pending)
    out.write(framing[1])
    return out.getvalue()


def _probe_image(mode: str) -> Any:
    """Every filter wins some rows; the stream passes zlib's 32 KiB window and one 64 KiB IDAT buffer."""
    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(20261030)
    bands = len(mode)
    height, width = 224, 1028 // bands + 1
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    smooth = np.stack(
        [128 + 100 * np.sin(xx / (11 + 7 * k) + yy / (23 + 5 * k)) for k in range(bands)], -1
    )
    pixels = np.clip(smooth + rng.normal(0, 4, smooth.shape), 0, 255).astype(np.uint8)
    pixels[20:28] = 9
    pixels[40:44] = pixels[39]
    pixels[60:170] = rng.integers(0, 256, pixels[60:170].shape, dtype = np.uint8)
    pixels[180:200, :, :] = (np.arange(width) % 256).astype(np.uint8)[None, :, None]
    if bands == 1:
        pixels = pixels[..., 0]
    return Image.fromarray(pixels, mode)


def _pillow(image: Any, save_kwargs: dict) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format = "PNG", **save_kwargs)
    return buffer.getvalue()


def _proven(mode: str, level: int) -> bool:
    key = (mode, level)
    proven = _PROBED.get(key)
    if proven is not None:
        return proven
    with _PROBE_LOCK:
        proven = _PROBED.get(key)
        if proven is None:
            try:
                from PIL.PngImagePlugin import PngInfo

                info = PngInfo()
                info.add_text("probe", "x" * 40)
                probe = _probe_image(mode)
                kwargs = {"pnginfo": info, "compress_level": level}
                proven = _encode(probe, mode, level, kwargs) == _pillow(probe, kwargs)
            except Exception:  # noqa: BLE001 - unproven means Pillow encodes
                proven = False
            _PROBED[key] = proven
    return proven


def encode(
    image: Any,
    *,
    pnginfo: Any = None,
    compress_level: int,
) -> Optional[bytes]:
    """``image.save(fp, "PNG", pnginfo = ..., compress_level = ...)``'s bytes, or None: the caller uses Pillow."""
    global engaged
    mode = getattr(image, "mode", None)
    if (
        mode not in _MODES
        # Level 0 stored blocks follow Pillow's output buffer size.
        or not 1 <= int(compress_level) <= 9
        or min(getattr(image, "size", (0, 0))) < 1
        or not enabled()
        or not _proven(mode, int(compress_level))
    ):
        return None
    kwargs = {"compress_level": int(compress_level)}
    if pnginfo is not None:
        kwargs["pnginfo"] = pnginfo
    try:
        data = _encode(image, mode, int(compress_level), kwargs)
    except Exception:  # noqa: BLE001 - Pillow's encoder is always correct
        return None
    if data is not None:
        with _ENGAGED_LOCK:
            engaged += 1
    return data
