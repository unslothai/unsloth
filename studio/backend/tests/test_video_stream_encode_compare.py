# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The streamed-frame compare decides exactly as np.array_equal without a chunk-sized temporary."""

from __future__ import annotations

import tracemalloc

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("av")

from core.inference import video_stream_encode as vse  # noqa: E402


def _scratch():
    return np.empty(vse._COMPARE_WORDS, dtype = np.bool_)


@pytest.mark.parametrize("shape", [(3, 16, 24, 3), (2, 7, 5, 3), (1, 1, 1, 3), (5, 64, 64, 3)])
def test_equal_frames_compare_equal(shape):
    rng = np.random.default_rng(0)
    a = rng.integers(0, 256, size = shape, dtype = np.uint8)
    assert vse._frames_equal(a, a.copy(), _scratch()) is True


def test_any_differing_byte_is_found(monkeypatch):
    monkeypatch.setattr(vse, "_COMPARE_WORDS", 16)
    rng = np.random.default_rng(1)
    a = rng.integers(
        0, 256, size = (3, 9, 7, 3), dtype = np.uint8
    )  # 567 bytes: 70 words + 7 tail bytes
    flat = a.size
    for index in (0, 7, 8, 127, 128, 129, flat // 2, 70 * 8 - 1, 70 * 8, flat - 1):
        b = a.copy()
        b.reshape(-1)[index] ^= 1
        assert vse._frames_equal(a, b, np.empty(16, dtype = np.bool_)) is False, index
        assert np.array_equal(a, b) is False


def test_shape_dtype_and_layout_corners():
    a = np.zeros((2, 4, 4, 3), dtype = np.uint8)
    assert vse._frames_equal(a, np.zeros((2, 4, 4, 4), dtype = np.uint8), _scratch()) is False
    assert vse._frames_equal(a, np.zeros((2, 4, 4, 3), dtype = np.int8), _scratch()) is False
    big = np.arange(2 * 4 * 8 * 3, dtype = np.uint8).reshape(2, 4, 8, 3)
    view = big[:, :, ::2]
    assert not view.flags.c_contiguous
    assert vse._frames_equal(view, view.copy(), _scratch()) is True
    other = view.copy()
    other[1, 2, 3, 0] ^= 1
    assert vse._frames_equal(view, other, _scratch()) is False
    # an offset start (not 8-byte aligned) still compares by words
    raw = np.arange(1 + 6 * 8 * 3, dtype = np.uint8)
    off = raw[1:].reshape(6, 8, 3)
    assert vse._frames_equal(off, off.copy(), _scratch()) is True


def _session_with_chunks(chunks):
    class _Vae:
        config = type("C", (), {"patch_size": None})()

    session = vse.StreamEncode(_Vae(), None, 24)
    session.chunks = [(torch.from_numpy(c), None) for c in chunks]
    return session


def _peak_during_matches(session, final):
    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        before = tracemalloc.get_traced_memory()[0]
        ok = session._matches(final)
        return ok, tracemalloc.get_traced_memory()[1] - before
    finally:
        tracemalloc.stop()
        session.close()


def test_matches_allocates_no_chunk_sized_temporary(monkeypatch):
    """Sliced compare peaks at its 128 KiB scratch; the kill switch restores the chunk-sized temporary."""
    rng = np.random.default_rng(2)
    frames = rng.integers(0, 256, size = (6, 256, 512, 3), dtype = np.uint8)
    chunk_bytes = 3 * 256 * 512 * 3
    monkeypatch.delenv(vse.STREAM_COMPARE_ENV, raising = False)
    ok, peak = _peak_during_matches(
        _session_with_chunks([frames[:3].copy(), frames[3:].copy()]),
        torch.from_numpy(frames.copy()),
    )
    assert ok is True
    assert peak < 512 << 10, peak
    monkeypatch.setenv(vse.STREAM_COMPARE_ENV, "0")
    ok, peak = _peak_during_matches(
        _session_with_chunks([frames[:3].copy(), frames[3:].copy()]),
        torch.from_numpy(frames.copy()),
    )
    assert ok is True
    assert peak >= chunk_bytes, peak


@pytest.mark.parametrize("sliced", ["1", "0"])
def test_matches_decides_like_before(monkeypatch, sliced):
    monkeypatch.setenv(vse.STREAM_COMPARE_ENV, sliced)
    rng = np.random.default_rng(3)
    frames = rng.integers(0, 256, size = (5, 32, 48, 3), dtype = np.uint8)
    session = _session_with_chunks([frames[:2].copy(), frames[2:].copy()])
    assert session._matches(torch.from_numpy(frames.copy())) is True
    session.close()
    changed = frames.copy()
    changed[4, 31, 47, 2] ^= 0x80
    session = _session_with_chunks([frames[:2].copy(), frames[2:].copy()])
    assert session._matches(torch.from_numpy(changed)) is False
    assert session.reason == "frames differ"
    session.close()
