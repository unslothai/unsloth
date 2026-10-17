# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Measure a take: duration, integrated loudness (ITU-R BS.1770-4), true peak, clipping and the span that holds sound, plus a word match against the text it should say. numpy only, so no new dependency."""

from __future__ import annotations

import functools
import math
import re
import unicodedata
import wave
from pathlib import Path
from typing import Any, Optional

import numpy as np

# BS.1770 gating: 400 ms blocks with 75% overlap, an absolute gate at -70 LUFS and a relative gate 10 LU below the ungated mean.
_BLOCK_S = 0.4
_HOP_S = 0.1
_ABSOLUTE_GATE = -70.0
_RELATIVE_GATE = 10.0
# The K-weighting high-pass rings for well under a second; past this many seconds its impulse response is below float precision.
_IR_SECONDS = 1.0
_CHUNK = 1 << 18
# Clipping: at least 3 samples in a row this close to full scale, on one side, and flat (each within
# one 16-bit step of the last). A clean full-scale sine, even a low one that stays near the rail for
# several samples, keeps moving at its peak, so it never counts.
_CLIP_LEVEL = 0.999
_CLIP_FLAT = 1.0 / 32768
_CLIP_RUN = 3
# Sound span: 20 ms frames louder than 40 dB under the loudest frame, never below -60 dBFS.
_FRAME_S = 0.02
_SPAN_RANGE_DB = 40.0
_SPAN_FLOOR_DB = -60.0


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    """(frames x channels float32 in [-1, 1], sample rate) for an 8/16/24/32-bit PCM WAV."""
    with wave.open(str(path)) as w:
        rate = w.getframerate()
        channels = w.getnchannels()
        width = w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if width == 1:
        data = (np.frombuffer(raw, dtype = np.uint8).astype(np.float32) - 128.0) / 128.0
    elif width == 2:
        data = np.frombuffer(raw, dtype = "<i2").astype(np.float32) / 32768.0
    elif width == 3:
        bytes3 = np.frombuffer(raw, dtype = np.uint8).reshape(-1, 3)
        ints = (
            bytes3[:, 0].astype(np.int32)
            | (bytes3[:, 1].astype(np.int32) << 8)
            | (bytes3[:, 2].astype(np.int32) << 16)
        )
        ints = np.where(ints >= 1 << 23, ints - (1 << 24), ints)
        data = ints.astype(np.float32) / float(1 << 23)
    elif width == 4:
        data = np.frombuffer(raw, dtype = "<i4").astype(np.float32) / float(1 << 31)
    else:
        raise ValueError(f"Unsupported WAV sample width: {width * 8} bits")
    return data.reshape(-1, max(channels, 1)), rate


def _biquad_coefficients(rate: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """The two K-weighting stages (high shelf, then RLB high-pass) at any rate, from the BS.1770 analog prototypes."""
    stages = []
    # Stage 1: high shelf, +4 dB above about 1.7 kHz.
    f0, gain_db, q = 1681.974450955533, 3.999843853973347, 0.7071752369554196
    k = math.tan(math.pi * f0 / rate)
    vh = 10.0 ** (gain_db / 20.0)
    vb = vh**0.4996667741545416
    a0 = 1.0 + k / q + k * k
    b = np.array(
        [(vh + vb * k / q + k * k) / a0, 2.0 * (k * k - vh) / a0, (vh - vb * k / q + k * k) / a0]
    )
    a = np.array([1.0, 2.0 * (k * k - 1.0) / a0, (1.0 - k / q + k * k) / a0])
    stages.append((b, a))
    # Stage 2: second-order high-pass at about 38 Hz.
    f0, q = 38.13547087602444, 0.5003270373238773
    k = math.tan(math.pi * f0 / rate)
    a0 = 1.0 + k / q + k * k
    b = np.array([1.0, -2.0, 1.0])
    a = np.array([1.0, 2.0 * (k * k - 1.0) / a0, (1.0 - k / q + k * k) / a0])
    stages.append((b, a))
    return stages


@functools.lru_cache(maxsize = 8)
def _impulse_response(rate: int) -> np.ndarray:
    """The K-weighting filter's impulse response, truncated where it has died out. Cached per rate:
    the filter runs sample by sample in Python. Callers must not modify it."""
    n = max(int(rate * _IR_SECONDS), 64)
    x = np.zeros(n)
    x[0] = 1.0
    for b, a in _biquad_coefficients(rate):
        y = np.zeros(n)
        x1 = x2 = y1 = y2 = 0.0
        b0, b1, b2 = b
        a1, a2 = a[1], a[2]
        for i in range(n):
            xi = x[i]
            yi = b0 * xi + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
            x2, x1 = x1, xi
            y2, y1 = y1, yi
            y[i] = yi
        x = y
    return x


def _fft_convolve(signal: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """``signal`` (1-D) convolved with ``kernel`` by overlap-add, cut to the signal's length."""
    n = len(signal)
    m = len(kernel)
    size = 1 << int(math.ceil(math.log2(_CHUNK + m - 1)))
    kernel_f = np.fft.rfft(kernel, size)
    out = np.zeros(n + m - 1)
    for start in range(0, n, _CHUNK):
        block = signal[start : start + _CHUNK]
        piece = np.fft.irfft(np.fft.rfft(block, size) * kernel_f, size)[: len(block) + m - 1]
        out[start : start + len(piece)] += piece
    return out[:n]


def integrated_loudness(data: np.ndarray, rate: int) -> Optional[float]:
    """Gated integrated loudness in LUFS, or None when every block sits under the absolute gate (silence, or shorter than one block)."""
    block = int(round(_BLOCK_S * rate))
    hop = int(round(_HOP_S * rate))
    if data.shape[0] < block or block <= 0:
        return None
    kernel = _impulse_response(rate)
    count = 1 + (data.shape[0] - block) // hop
    powers = np.zeros(count)
    for channel in range(data.shape[1]):
        weighted = _fft_convolve(data[:, channel].astype(np.float64), kernel)
        squares = np.concatenate(([0.0], np.cumsum(weighted * weighted)))
        starts = np.arange(count) * hop
        # Channel weights are 1.0 for mono, L, R and C; surround layouts are not something audio runs produce.
        powers += (squares[starts + block] - squares[starts]) / block
    with np.errstate(divide = "ignore"):
        loudness = -0.691 + 10.0 * np.log10(powers)
    kept = powers[loudness > _ABSOLUTE_GATE]
    if kept.size == 0:
        return None
    relative = -0.691 + 10.0 * math.log10(kept.mean()) - _RELATIVE_GATE
    with np.errstate(divide = "ignore"):
        kept = kept[-0.691 + 10.0 * np.log10(kept) > relative]
    if kept.size == 0:
        return None
    return -0.691 + 10.0 * math.log10(kept.mean())


def _oversampler(factor: int = 4, taps_per_phase: int = 12) -> np.ndarray:
    """A windowed-sinc interpolation filter, shaped (factor, taps_per_phase) by phase."""
    length = factor * taps_per_phase
    n = np.arange(length) - (length - 1) / 2.0
    taps = np.sinc(n / factor) * np.kaiser(length, 8.0)
    taps *= factor / taps.sum()
    return taps.reshape(taps_per_phase, factor).T


def true_peak(data: np.ndarray) -> float:
    """The largest absolute sample after 4x oversampling (BS.1770 Annex 2), linear."""
    phases = _oversampler()
    peak = float(np.max(np.abs(data))) if data.size else 0.0
    for channel in range(data.shape[1]):
        x = data[:, channel].astype(np.float64)
        for taps in phases:
            if x.size:
                peak = max(peak, float(np.max(np.abs(np.convolve(x, taps, mode = "same")))))
    return peak


def clipped_samples(data: np.ndarray) -> int:
    """Samples in flat runs of at least ``_CLIP_RUN`` same-sign samples at full scale, per channel."""
    total = 0
    for channel in range(data.shape[1]):
        x = data[:, channel].astype(np.float64)
        side = np.where(x >= _CLIP_LEVEL, 1, np.where(x <= -_CLIP_LEVEL, -1, 0)).astype(np.int8)
        if x.size < 2 or not side.any():
            continue
        # links[i]: sample i + 1 continues a flat run at the rail from sample i.
        links = (side[1:] != 0) & (side[1:] == side[:-1]) & (np.abs(np.diff(x)) <= _CLIP_FLAT)
        edges = np.diff(np.concatenate(([0], links.astype(np.int8), [0])))
        samples = np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1) + 1
        total += int(samples[samples >= _CLIP_RUN].sum())
    return total


def _db(value: float) -> Optional[float]:
    return round(20.0 * math.log10(value), 2) if value > 0 else None


def sound_span(data: np.ndarray, rate: int) -> Optional[tuple[float, float]]:
    """(start_s, end_s) of the part that holds sound, or None for silence."""
    frame = max(int(rate * _FRAME_S), 1)
    count = data.shape[0] // frame
    if count == 0:
        return None
    # Energy over every channel, so channels in opposite phase do not cancel out.
    frames = data[: count * frame].astype(np.float64).reshape(count, frame, data.shape[1])
    rms = np.sqrt(np.mean(frames * frames, axis = (1, 2)))
    loudest = float(rms.max())
    if loudest <= 0:
        return None
    floor = max(20.0 * math.log10(loudest) - _SPAN_RANGE_DB, _SPAN_FLOOR_DB)
    with np.errstate(divide = "ignore"):
        loud = np.nonzero(20.0 * np.log10(rms) > floor)[0]
    if loud.size == 0:
        return None
    return round(float(loud[0]) * frame / rate, 3), round(float(loud[-1] + 1) * frame / rate, 3)


def measure(data: np.ndarray, rate: int) -> dict[str, Any]:
    """The metrics for ``data`` (frames x channels in [-1, 1]) at ``rate``."""
    duration = data.shape[0] / rate if rate else 0.0
    lufs = integrated_loudness(data, rate)
    clipped = clipped_samples(data)
    span = sound_span(data, rate)
    return {
        "duration_s": round(duration, 3),
        "sample_rate": rate,
        "channels": int(data.shape[1]),
        "integrated_lufs": round(lufs, 2) if lufs is not None else None,
        "true_peak_dbtp": _db(true_peak(data)),
        "clipped_samples": clipped,
        "clipping_ratio": round(clipped / data.size, 6) if data.size else 0.0,
        "speech_start_s": span[0] if span else None,
        "speech_end_s": span[1] if span else None,
    }


def measure_file(path: Path) -> dict[str, Any]:
    """``measure`` for a PCM WAV; raises wave.Error, EOFError or ValueError for anything else."""
    data, rate = read_wav(path)
    return measure(data, rate)


# Scripts written without spaces between words: each character counts as a word.
_UNSPACED = re.compile(r"[\u0e00-\u0e7f\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")


def _words(text: str) -> list[str]:
    folded = unicodedata.normalize("NFKC", text or "").casefold()
    folded = "".join(" " if unicodedata.category(ch)[0] in "PS" else ch for ch in folded)
    return _UNSPACED.sub(lambda m: f" {m.group(0)} ", folded).split()


def match_score(expected: str, heard: str) -> float:
    """1 - word error rate after case, punctuation and spacing are normalised, floored at 0."""
    ref = _words(expected)
    hyp = _words(heard)
    if not ref:
        return 1.0 if not hyp else 0.0
    previous = list(range(len(hyp) + 1))
    for i, word in enumerate(ref, 1):
        current = [i] + [0] * len(hyp)
        for j, other in enumerate(hyp, 1):
            current[j] = min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + (word != other),
            )
        previous = current
    return round(max(0.0, 1.0 - previous[-1] / len(ref)), 4)
