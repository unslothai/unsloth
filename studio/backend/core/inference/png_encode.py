# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Pillow-exact PNG via numpy row filters (large images: strip-parallel deflate); off with UNSLOTH_IMAGE_PNG_FAST=0; probe-gated per mode."""

from __future__ import annotations

import io
import os
import struct
import threading
import zlib
from typing import Any, Iterator, Optional

FAST_ENV = "UNSLOTH_IMAGE_PNG_FAST"
STRIPS_ENV = "UNSLOTH_IMAGE_PNG_STRIPS"
# mode -> (bytes per pixel, IHDR colour type)
_MODES = {"L": (1, 0), "LA": (2, 4), "RGB": (3, 2), "RGBA": (4, 6)}
# Smaller bands made a batch's concurrent encodes queue on the GIL (32 rows of 512 px: 4 at once 1.6x slower).
_BAND_BYTES = 192 * 1024
_AHEAD = 2
# At or above this many pixel bytes, rows split into strips deflated side by side (pigz layout).
_STRIP_MIN_BYTES = 2 * 1024 * 1024
_STRIP_BYTES = 512 * 1024
_MAX_STRIP_THREADS = 8
_PROBE_LOCK = threading.Lock()
_PROBED: dict[tuple[str, int], bool] = {}
engaged = 0
_ENGAGED_LOCK = threading.Lock()


def enabled() -> bool:
    return (os.environ.get(FAST_ENV) or "").strip().lower() not in ("0", "false", "no", "off")


def _strips_enabled() -> bool:
    return (os.environ.get(STRIPS_ENV) or "").strip().lower() not in ("0", "false", "no", "off")


def _filtered(pixels: Any, bpp: int, start: int, stop: int) -> bytes:
    """ZipEncode.c's choice: None, Up, Sub, Paeth, each kept only on a strictly lower sum of |signed byte|."""
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


def _adler32_combine(first: int, second: int, second_len: int) -> int:
    """zlib's adler32_combine: the Adler-32 of A + B from adler32(A), adler32(B) and len(B)."""
    base = 65521
    rem = second_len % base
    lo = ((first & 0xFFFF) + (second & 0xFFFF) + base - 1) % base
    hi = (rem * (first & 0xFFFF) + (first >> 16) + (second >> 16) + base - rem) % base
    return lo | (hi << 16)


def _strip(
    pixels: Any, bpp: int, level: int, start: int, stop: int, last: bool
) -> tuple[bytes, int, int]:
    """One strip as byte-aligned raw deflate (final block only last), primed with the previous strip's tail."""
    data = _filtered(pixels, bpp, start, stop)
    compressor_args = (level, zlib.DEFLATED, -15, 9, zlib.Z_FILTERED)
    if start:
        stride = pixels.shape[1] + 1
        tail_rows = min(start, -(-32768 // stride))
        tail = _filtered(pixels, bpp, start - tail_rows, start)[-32768:]
        compressor = zlib.compressobj(*compressor_args, zdict = tail)
    else:
        compressor = zlib.compressobj(*compressor_args)
    out = compressor.compress(data) + compressor.flush(zlib.Z_FINISH if last else zlib.Z_SYNC_FLUSH)
    return out, zlib.adler32(data), len(data)


def _strip_deflated(pixels: Any, bpp: int, level: int, threads: int) -> Iterator[bytes]:
    """zlib stream of the strips; bounds follow the image only, so bytes never depend on ``threads``."""
    from concurrent.futures import ThreadPoolExecutor

    height, stride = pixels.shape
    rows = max(1, _STRIP_BYTES // (stride + 1))
    strips = [(start, min(height, start + rows)) for start in range(0, height, rows)]
    empty = zlib.compressobj(level, zlib.DEFLATED, 15, 9, zlib.Z_FILTERED)
    yield (empty.compress(b"") + empty.flush())[:2]
    checksum = 1
    with ThreadPoolExecutor(max_workers = threads, thread_name_prefix = "png-strip") as pool:
        futures = [
            pool.submit(_strip, pixels, bpp, level, start, stop, index == len(strips) - 1)
            for index, (start, stop) in enumerate(strips)
        ]
        try:
            for future in futures:
                data, adler, length = future.result()
                checksum = _adler32_combine(checksum, adler, length)
                yield data
        finally:
            for future in futures:
                future.cancel()
    yield struct.pack(">I", checksum)


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


def _strip_threads(pixel_bytes: int, threads: Optional[int]) -> int:
    if pixel_bytes < _STRIP_MIN_BYTES or not _strips_enabled():
        return 1
    if threads is None:
        threads = getattr(os, "process_cpu_count", os.cpu_count)() or 1
    return max(1, min(int(threads), _MAX_STRIP_THREADS))


def _encode(
    image: Any,
    mode: str,
    level: int,
    save_kwargs: dict,
    threads: int = 1,
) -> Optional[bytes]:
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

    if threads > 1:
        stream = _strip_deflated(pixels, bpp, level, threads)
    else:
        stream = _deflated(pixels, bpp, level)
    for data in stream:
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
    threads: Optional[int] = None,
) -> Optional[bytes]:
    """``image.save(fp, "PNG", ...)``'s bytes, or None (Pillow encodes); large images use strips (same pixels, other bytes)."""
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
        bpp = _MODES[mode][0]
        strip_threads = _strip_threads(image.width * image.height * bpp, threads)
        data = _encode(image, mode, int(compress_level), kwargs, strip_threads)
    except Exception:  # noqa: BLE001 - Pillow's encoder is always correct
        return None
    if data is not None:
        with _ENGAGED_LOCK:
            engaged += 1
    return data
