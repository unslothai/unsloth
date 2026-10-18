# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Take metrics against signals with known answers."""

from __future__ import annotations

import wave

import numpy as np
import pytest

from core.inference import audio_metrics as am

RATE = 48000


def _sine(
    seconds = 3.0,
    amplitude = 0.1,
    freq = 1000.0,
    rate = RATE,
) -> np.ndarray:
    t = np.arange(int(seconds * rate)) / rate
    return (amplitude * np.sin(2 * np.pi * freq * t))[:, None]


def _write(
    path,
    data,
    rate = RATE,
):
    pcm = np.clip(np.round(data * 32767), -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(data.shape[1])
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())
    return path


def test_a_minus_20_dbfs_sine_reads_its_known_loudness_and_peak(tmp_path):
    report = am.measure_file(_write(tmp_path / "s.wav", _sine()))
    assert report["integrated_lufs"] == pytest.approx(-23.0, abs = 0.05)
    assert report["true_peak_dbtp"] == pytest.approx(-20.0, abs = 0.05)
    assert (report["duration_s"], report["sample_rate"], report["channels"]) == (3.0, RATE, 1)
    assert report["clipped_samples"] == 0


def test_loudness_agrees_with_pyloudnorm():
    pyln = pytest.importorskip("pyloudnorm")
    rng = np.random.default_rng(0)
    data = np.concatenate([_sine(2.0, 0.3, 440.0), _sine(2.0, 0.05, 3000.0)])
    data = data + 0.01 * rng.standard_normal(data.shape)
    expected = pyln.Meter(RATE).integrated_loudness(data[:, 0])
    assert am.integrated_loudness(data, RATE) == pytest.approx(expected, abs = 0.1)
    stereo = np.hstack([data, 0.5 * data])
    expected = pyln.Meter(RATE).integrated_loudness(stereo)
    assert am.integrated_loudness(stereo, RATE) == pytest.approx(expected, abs = 0.1)


def test_hard_clipping_counts_and_a_clean_full_scale_sine_does_not():
    clipped = am.measure(np.clip(3.0 * _sine(1.0, 1.0, 100.0), -1.0, 1.0), RATE)
    assert clipped["clipping_ratio"] > 0.3
    assert am.measure(_sine(1.0, 1.0), RATE)["clipped_samples"] == 0
    # A low full-scale sine lingers near the rail for many samples but never holds one value.
    for freq in (100.0, 10.0):
        assert am.measure(_sine(1.0, 1.0, freq), RATE)["clipped_samples"] == 0
    # Two samples at the rail are a peak, three in a row are clipping.
    burst = np.zeros((100, 1))
    burst[10:12] = 1.0
    assert am.clipped_samples(burst) == 0
    burst[50:53] = -1.0
    assert am.clipped_samples(burst) == 3


def test_silence_padding_gives_the_speech_span_and_silence_gives_none():
    pad = np.zeros((RATE // 2, 1))
    report = am.measure(np.concatenate([pad, _sine(1.0, 0.3), pad]), RATE)
    assert report["speech_start_s"] == pytest.approx(0.5, abs = 0.02)
    assert report["speech_end_s"] == pytest.approx(1.5, abs = 0.02)
    # Channels in opposite phase still hold sound.
    tone = np.concatenate([pad, _sine(1.0, 0.3), pad])
    report = am.measure(np.hstack([tone, -tone]), RATE)
    assert report["speech_start_s"] == pytest.approx(0.5, abs = 0.02)
    # A beep shorter than one 20 ms frame, and sound only in the last partial frame, still count.
    assert am.sound_span(np.full((300, 1), 0.5), RATE) == (0.0, 0.006)
    tail = np.concatenate([np.zeros((960 * 3, 1)), np.full((100, 1), 0.5)])
    assert am.sound_span(tail, RATE) == (0.06, 0.062)
    quiet = am.measure(np.zeros((RATE, 2)), RATE)
    assert (quiet["integrated_lufs"], quiet["true_peak_dbtp"], quiet["speech_start_s"]) == (
        None,
        None,
        None,
    )
    assert quiet["clipping_ratio"] == 0.0


def test_24_bit_wav_reads_like_16_bit(tmp_path):
    data = _sine(0.5, 0.25)
    pcm = np.round(data[:, 0] * (1 << 23)).astype("<i4")
    raw = b"".join(int(v).to_bytes(3, "little", signed = True) for v in pcm)
    with wave.open(str(tmp_path / "s24.wav"), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(3)
        w.setframerate(RATE)
        w.writeframes(raw)
    read, rate = am.read_wav(tmp_path / "s24.wav")
    assert rate == RATE and np.allclose(read, data, atol = 1e-6)


def test_match_score_is_one_minus_wer_after_normalising():
    assert am.match_score("Hello, World!", "hello world") == 1.0
    assert am.match_score("Don’t stop", "dont stop") == 1.0
    assert am.match_score("the quick brown fox", "the quick red fox") == 0.75
    assert am.match_score("one two", "three four five six") == 0.0
    assert am.match_score("", "") == 1.0 and am.match_score("", "words") == 0.0
    # Scripts without spaces count per character.
    assert am.match_score("你好世界", "你好世") == 0.75
    # Too long to align: no score rather than seconds of work.
    assert am.match_score("字" * 20000, "字" * 20000) is None
    jiwer = pytest.importorskip("jiwer")
    ref, hyp = "please call stella ask her", "please call stela ask her to"
    assert am.match_score(ref, hyp) == pytest.approx(1 - jiwer.wer(ref, hyp), abs = 1e-4)


def test_8_bit_clipping_counts_on_both_rails(tmp_path):
    pcm = np.full(100, 128, dtype = np.uint8)
    pcm[10:15], pcm[50:55] = 255, 0
    with wave.open(str(tmp_path / "s8.wav"), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(1)
        w.setframerate(RATE)
        w.writeframes(pcm.tobytes())
    assert am.measure_file(tmp_path / "s8.wav")["clipped_samples"] == 10


def test_chunked_measures_match_one_pass(monkeypatch):
    rng = np.random.default_rng(1)
    noise = 0.3 * rng.standard_normal((RATE * 2 + 17, 2))
    clipped = np.clip(3.0 * _sine(2.0, 1.0, 100.0), -1.0, 1.0)
    whole = am.measure(noise, RATE), am.measure(clipped, RATE)
    # Chunks far smaller than a block, a frame or a clipped run must not change any number.
    for chunk in (4096, 1000, 7):
        monkeypatch.setattr(am, "_CHUNK", chunk)
        assert (am.measure(noise, RATE), am.measure(clipped, RATE)) == whole
