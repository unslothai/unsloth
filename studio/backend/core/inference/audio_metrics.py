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
# Every measure walks the audio this many frames at a time, so a 10-minute take never needs
# full-length float64 copies.
_CHUNK = 1 << 18
# Clipping: at least 3 identical samples in a row at full scale, reached or left by a jump a smooth
# peak could not make. PCM rounds a clean peak, even a low tone's, to a flat top of L samples, and the
# curve drops off it by at most about ((L + 1) / (L - 1))**2 quantization steps (3 steps at L = 3,
# 1 from L = 5 in a sweep of clean tones); a clipped waveform hits the rail at speed. The step used is
# at most _CLIP_STEP_MAX, so coarse 8-bit steps (1/128) cannot hide an overdriven tone; 8-bit is too
# coarse to tell a clean low tone's flat top from clipping. 0.99 lets the 8-bit positive rail
# (127/128) count.
_CLIP_LEVEL = 0.99
_CLIP_RUN = 3
_CLIP_EDGE = 2
_CLIP_STEP_MAX = 2.0**-13
# Sound span: 20 ms frames louder than 40 dB under the loudest frame, never below -60 dBFS.
_FRAME_S = 0.02
_SPAN_RANGE_DB = 40.0
_SPAN_FLOOR_DB = -60.0


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    """(frames x channels float32 in [-1, 1], sample rate) for an 8/16/24/32-bit PCM WAV."""
    data, rate, _step = _read_pcm(path)
    return data, rate


def _read_pcm(path: Path) -> tuple[np.ndarray, int, float]:
    """``read_wav`` plus the size of one quantization step."""
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
    return data.reshape(-1, max(channels, 1)), rate, 2.0 ** (1 - 8 * width)


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


def _fft_convolve(signal: np.ndarray, kernel: np.ndarray):
    """``signal`` (1-D) convolved with ``kernel`` by overlap-add, cut to the signal's length and
    yielded in order, one chunk at a time."""
    m = len(kernel)
    size = 1 << int(math.ceil(math.log2(_CHUNK + m - 1)))
    kernel_f = np.fft.rfft(kernel, size)
    tail = np.zeros(m - 1)
    for start in range(0, len(signal), _CHUNK):
        block = signal[start : start + _CHUNK].astype(np.float64)
        piece = np.fft.irfft(np.fft.rfft(block, size) * kernel_f, size)[: len(block) + m - 1]
        piece[: m - 1] += tail
        yield piece[: len(block)]
        tail = piece[len(block) :]


def _padded(x: np.ndarray, start: int, stop: int) -> np.ndarray:
    """``x[start:stop]`` as float64, zero where the range runs past either end."""
    out = np.zeros(stop - start)
    lo, hi = max(start, 0), min(stop, len(x))
    if hi > lo:
        out[lo - start : hi - start] = x[lo:hi]
    return out


def integrated_loudness(data: np.ndarray, rate: int) -> Optional[float]:
    """Gated integrated loudness in LUFS, or None when every block sits under the absolute gate (silence, or shorter than one block)."""
    block = int(round(_BLOCK_S * rate))
    hop = max(int(round(_HOP_S * rate)), 1)
    if data.shape[0] < block or block <= 0:
        return None
    kernel = _impulse_response(rate)
    count = 1 + (data.shape[0] - block) // hop
    starts = np.arange(count) * hop
    # The running sum of squares is only read at block edges, so only those points are kept.
    edges = np.unique(np.concatenate((starts, starts + block)))
    first, last = np.searchsorted(edges, starts), np.searchsorted(edges, starts + block)
    powers = np.zeros(count)
    for channel in range(data.shape[1]):
        sums = np.zeros(edges.size)  # sums[i]: energy of the first edges[i] filtered samples
        total, done = 0.0, 0
        for piece in _fft_convolve(data[:, channel], kernel):
            running = np.cumsum(piece * piece) + total
            lo = np.searchsorted(edges, done, side = "right")
            hi = np.searchsorted(edges, done + piece.size, side = "right")
            sums[lo:hi] = running[edges[lo:hi] - done - 1]
            total, done = float(running[-1]), done + piece.size
        # Channel weights are 1.0 for mono, L, R and C; surround layouts are not something audio runs produce.
        powers += (sums[last] - sums[first]) / block
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
    length = phases.shape[1]
    # np.convolve's "same" output i reads input i - before .. i + after.
    after = (length - 1) // 2
    before = length - 1 - after
    peak = 0.0
    for channel in range(data.shape[1]):
        x = data[:, channel]
        for start in range(0, len(x), _CHUNK):
            stop = min(start + _CHUNK, len(x))
            peak = max(peak, float(np.max(np.abs(x[start:stop]))))
            window = _padded(x, start - before, stop + after)
            for taps in phases:
                peak = max(peak, float(np.max(np.abs(np.convolve(window, taps, mode = "valid")))))
    return peak


def clipped_samples(data: np.ndarray, step: float = 0.0) -> int:
    """Samples in runs of at least ``_CLIP_RUN`` identical full-scale samples, per channel, that the
    waveform reaches or leaves more steeply than a smooth peak could; ``step`` is the quantization
    step (0 for float audio)."""
    total = 0
    for channel in range(data.shape[1]):
        x = data[:, channel]
        n = len(x)
        starts: list[np.ndarray] = []
        lengths: list[np.ndarray] = []
        open_start, open_length = 0, 0  # the run still open at the end of the last chunk
        for start in range(0, n, _CHUNK):
            # One sample of overlap, so a run that crosses chunks stays one run.
            lead = 1 if start else 0
            seg = x[start - lead : start + _CHUNK]
            rail = np.abs(seg) >= _CLIP_LEVEL
            same = np.zeros(len(seg), dtype = bool)
            same[1:] = rail[1:] & rail[:-1] & (seg[1:] == seg[:-1])
            rail, same = rail[lead:], same[lead:]
            # Each rail sample that does not continue the previous one opens a run; id 0 is the
            # run carried in from the last chunk.
            opens = rail & ~same
            ids = np.cumsum(opens)
            counts = np.bincount(ids[rail], minlength = int(ids[-1]) + 1)
            run_starts = np.concatenate(([open_start], np.flatnonzero(opens) + start))
            counts[0] += open_length
            if rail[-1]:
                open_start, open_length = int(run_starts[-1]), int(counts[-1])
                run_starts, counts = run_starts[:-1], counts[:-1]
            else:
                open_start, open_length = 0, 0
            starts.append(run_starts)
            lengths.append(counts)
        starts.append(np.array([open_start]))
        lengths.append(np.array([open_length]))
        first = np.concatenate(starts)
        length = np.concatenate(lengths)
        long_runs = length >= _CLIP_RUN
        first, length = first[long_runs], length[long_runs]
        if not first.size:
            continue
        last = first + length - 1
        top = x[first].astype(np.float64)
        # The larger jump onto or off the run; -1 at an end of the audio, where there is none, and
        # a run that fills the whole channel counts.
        into = np.where(first > 0, np.abs(top - x[np.maximum(first - 1, 0)]), -1.0)
        out = np.where(last < n - 1, np.abs(top - x[np.minimum(last + 1, n - 1)]), -1.0)
        edge = np.maximum(into, out)
        limit = _CLIP_EDGE * min(step, _CLIP_STEP_MAX) * ((length + 1) / (length - 1)) ** 2
        steep = (edge > limit) | (edge < 0)
        total += int(length[steep].sum())
    return total


def _db(value: float) -> Optional[float]:
    return round(20.0 * math.log10(value), 2) if value > 0 else None


def sound_span(data: np.ndarray, rate: int) -> Optional[tuple[float, float]]:
    """(start_s, end_s) of the part that holds sound, or None for silence."""
    if rate <= 0:
        return None
    frame = max(int(rate * _FRAME_S), 1)
    total = data.shape[0]
    count = -(-total // frame)  # the last, shorter frame counts too
    if count == 0:
        return None
    # Energy over every channel, so channels in opposite phase do not cancel out.
    rms = np.empty(count)
    full = total // frame
    step = max(_CHUNK // frame, 1)
    for lo in range(0, full, step):
        hi = min(lo + step, full)
        frames = data[lo * frame : hi * frame].astype(np.float64)
        frames = frames.reshape(hi - lo, frame, data.shape[1])
        rms[lo:hi] = np.sqrt(np.mean(frames * frames, axis = (1, 2)))
    if full < count:
        tail = data[full * frame :].astype(np.float64)
        rms[full] = np.sqrt(np.mean(tail * tail))
    loudest = float(rms.max())
    if loudest <= 0:
        return None
    floor = max(20.0 * math.log10(loudest) - _SPAN_RANGE_DB, _SPAN_FLOOR_DB)
    with np.errstate(divide = "ignore"):
        loud = np.nonzero(20.0 * np.log10(rms) > floor)[0]
    if loud.size == 0:
        return None
    end = min(int(loud[-1] + 1) * frame, total)
    return round(float(loud[0]) * frame / rate, 3), round(end / rate, 3)


def measure(
    data: np.ndarray,
    rate: int,
    step: float = 0.0,
) -> dict[str, Any]:
    """The metrics for ``data`` (frames x channels in [-1, 1]) at ``rate``; ``step`` is the PCM
    quantization step the samples were decoded from (0 for float audio)."""
    duration = data.shape[0] / rate if rate else 0.0
    lufs = integrated_loudness(data, rate)
    clipped = clipped_samples(data, step)
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
    data, rate, step = _read_pcm(path)
    return measure(data, rate, step)


# Scripts written without spaces between words: each character counts as a word.
_UNSPACED = re.compile(r"[\u0e00-\u0e7f\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")


def _words(text: str) -> list[str]:
    folded = unicodedata.normalize("NFKC", text or "").casefold()
    # "don't" and "dont" are one word; other punctuation splits words.
    folded = re.sub(r"(?<=\w)['\u2019](?=\w)", "", folded)
    folded = "".join(" " if unicodedata.category(ch)[0] in "PS" else ch for ch in folded)
    return _UNSPACED.sub(lambda m: f" {m.group(0)} ", folded).split()


# The word alignment compares every expected word with every heard one; past this many pairs
# (about a second of work) there is no score.
_MATCH_MAX_PAIRS = 50_000_000


def match_score(expected: str, heard: str) -> Optional[float]:
    """1 - word error rate after case, punctuation and spacing are normalised, floored at 0.
    None when the two texts are too long to align."""
    ref = _words(expected)
    hyp = _words(heard)
    if not ref:
        return 1.0 if not hyp else 0.0
    if not hyp:
        return 0.0
    if len(ref) * len(hyp) > _MATCH_MAX_PAIRS:
        return None
    ids: dict[str, int] = {}
    ref_ids = [ids.setdefault(word, len(ids)) for word in ref]
    hyp_ids = np.array([ids.setdefault(word, len(ids)) for word in hyp])
    steps = np.arange(len(hyp) + 1)
    previous = steps.copy()
    # One row of the edit-distance table per expected word, a whole row at a time.
    for i, word in enumerate(ref_ids, 1):
        current = np.empty_like(previous)
        current[0] = i
        current[1:] = np.minimum(previous[1:] + 1, previous[:-1] + (hyp_ids != word))
        # An insertion carries a cell to the right: current[j] = min over k <= j of current[k] + j - k.
        previous = np.minimum.accumulate(current - steps) + steps
    return round(max(0.0, 1.0 - int(previous[-1]) / len(ref)), 4)
