# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The buffered GGUF string-array walkers must leave the stream exactly where the old
one-read-per-entry loops did, and hand back the same entries, for every array shape."""

from __future__ import annotations

import io
import random
import struct

import pytest

from core.inference.llama_cpp import LlamaCppBackend
from utils.models.gguf_metadata import (
    _STRING_SCAN_CHUNK,
    _STRING_SCAN_MAX,
    _skip_gguf_value,
    skip_gguf_strings,
    take_gguf_strings_by_first_byte,
)


def _entry(raw: bytes) -> bytes:
    return struct.pack("<Q", len(raw)) + raw


def _random_entries(rng: random.Random, n: int) -> list[bytes]:
    out = []
    for _ in range(n):
        x = rng.random()
        if x < 0.03:
            out.append(b"")
        elif x < 0.15:
            out.append(rng.choice([b"<|im_start|>", b"[INST]", b"<think>"]))
        elif x < 0.2:
            out.append(b"<" + bytes(rng.randrange(256) for _ in range(rng.randrange(4))))
        elif x < 0.21:
            out.append(b"y" * rng.choice([255, 4095, 70000]))
        else:
            out.append(bytes(rng.choice(b"abc \xc4\xa0") for _ in range(rng.randrange(1, 9))))
    return out


def _array_cases() -> list[tuple[str, bytes, int]]:
    rng = random.Random(0)
    cases = []
    for n in (0, 1, 5, 3000, 60000):
        body = b"".join(_entry(e) for e in _random_entries(rng, n))
        cases.append((f"n{n}", body + b"TAIL", n))
        if body:
            for cut in (1, 7, len(body) // 2, len(body) - 3):
                cases.append((f"n{n}_cut{cut}", body[:cut], n))
    big = [b"a", b"<" + b"z" * _STRING_SCAN_MAX, b"<" + b"z" * (_STRING_SCAN_MAX - 1), b"b"]
    cases.append(("over_scan_max", b"".join(_entry(e) for e in big) + b"TAIL", len(big)))
    straddle = [b"x" * (_STRING_SCAN_CHUNK - 20), b"<straddles the chunk end>", b"[ok]"]
    cases.append(("straddle", b"".join(_entry(e) for e in straddle) + b"TAIL", len(straddle)))
    for bad in ((1 << 30) + 1, 1 << 40, (1 << 64) - 1):
        body = _entry(b"<a>") + struct.pack("<Q", bad) + _entry(b"<b>")
        cases.append((f"bad_len_{bad}", body, 3))
    return cases


CASES = _array_cases()


def _old_gm_skip(f, alen: int) -> bool:
    for _ in range(alen):
        slen_bytes = f.read(8)
        if len(slen_bytes) < 8:
            return False
        slen = struct.unpack("<Q", slen_bytes)[0]
        if slen > 1 << 30:
            return False
        f.seek(slen, 1)
    return True


def _old_llama_skip(f, alen: int) -> None:
    for _ in range(alen):
        slen = struct.unpack("<Q", f.read(8))[0]
        f.seek(slen, 1)


def _old_llama_take(f, alen: int) -> list:
    kept = []
    for _ in range(alen):
        n = struct.unpack("<Q", f.read(8))[0]
        raw = f.read(n)
        if raw[:1] != b"<" and raw[:1] != b"[":
            continue
        try:
            kept.append(raw.decode("utf-8"))
        except UnicodeDecodeError:
            continue
    return kept


def _outcome(fn, f, *args):
    try:
        result = ("ok", fn(f, *args))
    except Exception as e:  # noqa: BLE001 -- the exception type is part of the contract
        result = ("raised", type(e))
    return result, f.tell()


def _streams(data: bytes, tmp_path):
    path = tmp_path / "array.bin"
    path.write_bytes(data)
    return [lambda: io.BytesIO(data), lambda: open(path, "rb")]


@pytest.mark.parametrize("name,data,count", CASES, ids = [c[0] for c in CASES])
def test_gguf_metadata_skip_matches_entry_loop(name, data, count, tmp_path):
    header = struct.pack("<IQ", 8, count)
    for opener in _streams(header + data, tmp_path):
        with opener() as old, opener() as new:
            assert _outcome(_skip_gguf_value, new, 9) == _outcome(
                lambda f: _old_gm_skip(f, struct.unpack("<IQ", f.read(12))[1]), old
            )


@pytest.mark.parametrize("name,data,count", CASES, ids = [c[0] for c in CASES])
def test_llama_skip_and_read_match_entry_loop(name, data, count, tmp_path):
    header = struct.pack("<IQ", 8, count)
    for opener in _streams(header + data, tmp_path):
        with opener() as old, opener() as new:
            want = _outcome(lambda f: _old_llama_skip(f, struct.unpack("<IQ", f.read(12))[1]), old)
            assert _outcome(LlamaCppBackend._gguf_skip_value, new, 9) == want
        with opener() as old, opener() as new:
            old.seek(12)
            new.seek(12)
            from core.inference.chat_template_helpers import delimiter_shaped_tokens

            want = _outcome(lambda f: delimiter_shaped_tokens(_old_llama_take(f, count)), old)
            assert _outcome(LlamaCppBackend._gguf_read_array_value, new, 8, count) == want


@pytest.mark.parametrize("name,data,count", CASES, ids = [c[0] for c in CASES])
def test_take_returns_every_matching_entry_in_order(name, data, count):
    f = io.BytesIO(data)
    done, kept = take_gguf_strings_by_first_byte(f, count, frozenset(b"<["))
    ref = io.BytesIO(data)
    want = []
    for _ in range(done):
        n = struct.unpack("<Q", ref.read(8))[0]
        raw = ref.read(n)
        if raw[:1] in (b"<", b"["):
            want.append(raw)
    assert kept == want
    assert f.tell() == ref.tell()


def test_walk_reads_in_chunks_not_per_entry():
    data = b"".join(_entry(b"tok%d" % i) for i in range(50000))

    class CountingReader(io.BytesIO):
        reads = 0

        def read(self, *args):
            CountingReader.reads += 1
            return super().read(*args)

    f = CountingReader(data)
    assert skip_gguf_strings(f, 50000) == 50000
    assert f.tell() == len(data)
    assert CountingReader.reads < 10


class _ForwardOnly(io.BytesIO):
    """Skips ahead but cannot seek back, like a pipe that discards; reports that honestly."""

    def seekable(self):
        return False

    def seek(
        self,
        offset,
        whence = 0,
    ):
        target = (
            offset
            if whence == 0
            else self.tell() + offset
            if whence == 1
            else len(self.getvalue()) + offset
        )
        if target < self.tell():
            raise io.UnsupportedOperation("cannot seek back")
        return super().seek(offset, whence)


class _NoSeekableMethod:
    """read / seek / tell without seekable(), as mmap.mmap was before Python 3.13."""

    def __init__(self, data):
        self._f = io.BytesIO(data)
        self.read, self.seek, self.tell = self._f.read, self._f.seek, self._f.tell


def test_streams_that_cannot_rewind_skip_entry_by_entry():
    data = b"".join(_entry(e) for e in (b"<a>", b"b", b"[c]")) + b"TAIL"
    for make in (_ForwardOnly, _NoSeekableMethod):
        f = make(data)
        assert skip_gguf_strings(f, 3) == 0
        assert f.tell() == 0
        f = make(data)
        assert take_gguf_strings_by_first_byte(f, 3, frozenset(b"<[")) == (0, [])
        assert f.tell() == 0
        # The callers' own loops still walk the whole array and stop at the tail.
        f = make(data)
        for _ in range(3 - skip_gguf_strings(f, 3)):
            (n,) = struct.unpack("<Q", f.read(8))
            f.seek(n, 1)
        assert f.read() == b"TAIL"
        assert LlamaCppBackend._gguf_read_array_value(make(data), 8, 3) == (
            LlamaCppBackend._gguf_read_array_value(io.BytesIO(data), 8, 3)
        )


def test_non_seekable_stream_reads_entry_by_entry():
    data = b"".join(_entry(e) for e in (b"<a>", b"b", b"[c]"))

    class NoSeek(io.BytesIO):
        def seekable(self):
            return False

    f = NoSeek(data)
    assert take_gguf_strings_by_first_byte(f, 3, frozenset(b"<[")) == (0, [])
    assert f.tell() == 0
    assert LlamaCppBackend._gguf_read_array_value(NoSeek(data), 8, 3) == (
        LlamaCppBackend._gguf_read_array_value(io.BytesIO(data), 8, 3)
    )
