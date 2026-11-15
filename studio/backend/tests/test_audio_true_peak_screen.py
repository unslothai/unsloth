# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The screened true peak returns the very float the whole-chunk convolution walk returns."""

from __future__ import annotations

import numpy as np
import pytest

from core.inference import audio_metrics as am


def _walk(data: np.ndarray) -> float:
    """The true peak as every output of every chunk convolved in full (the walk before the screen)."""
    phases = am._oversampler()
    length = phases.shape[1]
    after = (length - 1) // 2
    before = length - 1 - after
    peak = 0.0
    for channel in range(data.shape[1]):
        x = data[:, channel]
        for start in range(0, len(x), am._CHUNK):
            stop = min(start + am._CHUNK, len(x))
            peak = max(peak, float(np.max(np.abs(x[start:stop]))))
            window = am._padded(x, start - before, stop + after)
            for taps in phases:
                peak = max(peak, float(np.max(np.abs(np.convolve(window, taps, mode = "valid")))))
    return peak


def _signals(rng: np.random.Generator):
    rate = 48000
    for n in (1, 5, 11, 12, 13, 47, 1000, 4097, 30011):
        t = np.arange(n) / rate
        yield rng.uniform(-1, 1, (n, 1)).astype(np.float32)
        yield np.stack([0.3 * np.sin(2 * np.pi * 997 * t), 0.5 * np.cos(2 * np.pi * 1000 * t)], 1)
        tone = (0.5 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)[:, None]
        yield tone
        yield np.where(np.sin(2 * np.pi * 100 * t) >= 0, 1.0, -1.0).astype(np.float32)[:, None]
        yield np.zeros((n, 2), np.float32)
        yield np.clip(3.0 * tone, -1.0, 1.0)
        yield (rng.standard_normal((n, 1)) * 1e-30).astype(np.float32)
        yield (rng.standard_normal((n, 2)) * 1e20).astype(np.float32)
        yield np.full((n, 1), 1e-40, np.float32)
        for special in (np.nan, np.inf, -np.inf):
            spiked = rng.uniform(-0.5, 0.5, (n, 1)).astype(np.float32)
            spiked[rng.integers(n)] = special
            yield spiked
        pcm = np.round(rng.standard_normal((n, 2)) * 6000).clip(-32768, 32767) / 32768.0
        yield pcm.astype(np.float32)
        envelope = np.abs(np.sin(2 * np.pi * 3 * t))[:, None] ** 4
        yield (envelope * rng.standard_normal((n, 1)) * 0.2).astype(np.float32)


@pytest.mark.parametrize("chunk", (7, 100, 4096, 1 << 18))
def test_screen_matches_the_full_walk(monkeypatch, chunk):
    monkeypatch.setattr(am, "_CHUNK", chunk)
    rng = np.random.default_rng(chunk)
    for data in _signals(rng):
        expected, got = _walk(data), am.true_peak(data)
        assert repr(got) == repr(expected), (data.shape, data.dtype, expected, got)


def test_many_ties_fall_back_to_the_whole_chunk(monkeypatch):
    # A full-scale square wave ties at every edge, so the screen hands whole chunks to np.convolve.
    calls = []
    real = np.convolve

    def counting(
        a,
        v,
        mode = "full",
    ):
        calls.append(len(a))
        return real(a, v, mode = mode)

    monkeypatch.setattr(np, "convolve", counting)
    t = np.arange(1 << 18) / 48000
    square = np.where(np.sin(2 * np.pi * 1000 * t) >= 0, 1.0, -1.0).astype(np.float32)[:, None]
    assert repr(am.true_peak(square)) == repr(_walk(square))
    assert max(calls) == (1 << 18) + 11


def test_a_long_take_convolves_only_near_its_peak(monkeypatch):
    calls = []
    real = np.convolve

    def counting(
        a,
        v,
        mode = "full",
    ):
        calls.append(len(a))
        return real(a, v, mode = mode)

    rng = np.random.default_rng(3)
    t = np.arange(48000 * 12) / 48000
    speech = (np.abs(np.sin(2 * np.pi * 2 * t)) ** 4 * rng.standard_normal(t.size) * 0.2).astype(
        np.float32
    )
    expected = _walk(speech[:, None])
    monkeypatch.setattr(np, "convolve", counting)
    assert repr(am.true_peak(speech[:, None])) == repr(expected)
    assert calls and max(calls) < 1000
