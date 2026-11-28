# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A replayed recording is transcoded once, not on every follow-up turn."""

import base64
import io

import numpy as np
import pytest
import routes.inference as inference_route


@pytest.fixture(autouse = True)
def _empty_cache():
    inference_route._clear_prepared_audio_cache()
    yield
    inference_route._clear_prepared_audio_cache()


def _counting_decoder(monkeypatch, fail_first = 0):
    calls = []

    def decode(raw):
        calls.append(raw)
        if len(calls) <= fail_first:
            raise ValueError("broken container")
        rng = np.random.default_rng(len(raw))
        return rng.standard_normal(16_000 + len(raw)).astype(np.float32) * 0.1, 16_000

    monkeypatch.setattr(inference_route, "_sniff_audio_container", lambda _raw: None)
    monkeypatch.setattr(inference_route, "_decode_audio_mono", decode)
    return calls


def _clip(payload: bytes) -> str:
    return "data:audio/mp4;base64," + base64.b64encode(payload).decode("ascii")


def test_a_repeated_clip_is_decoded_once_and_matches_the_uncached_output(monkeypatch):
    calls = _counting_decoder(monkeypatch)
    clips = [_clip(b"first recording"), _clip(b"second one")]
    first = inference_route._prepare_audio_clips_for_llama(clips)
    second = inference_route._prepare_audio_clips_for_llama(clips)
    assert len(calls) == 2
    expected, transcoded = inference_route._prepare_audio_clips_for_llama_uncached(clips)
    assert transcoded
    assert first == second == expected
    assert all(audio_format == "wav" for _, audio_format in second)


def test_a_caller_mutating_the_result_does_not_change_the_cache(monkeypatch):
    _counting_decoder(monkeypatch)
    clips = [_clip(b"recording")]
    first = inference_route._prepare_audio_clips_for_llama(clips)
    expected = list(first)
    first.clear()
    assert inference_route._prepare_audio_clips_for_llama(clips) == expected


def test_a_different_clip_or_limit_misses(monkeypatch):
    calls = _counting_decoder(monkeypatch)
    inference_route._prepare_audio_clips_for_llama([_clip(b"aaaa")])
    inference_route._prepare_audio_clips_for_llama([_clip(b"aaab")])
    assert len(calls) == 2
    monkeypatch.setattr(inference_route, "_MAX_AUDIO_RAW_BYTES", 20_000)
    capped = inference_route._prepare_audio_clips_for_llama([_clip(b"aaab")])
    assert len(calls) == 3
    assert len(base64.b64decode(capped[0][0])) <= 20_000
    monkeypatch.setattr(inference_route, "_MIN_TRANSCODE_AUDIO_SAMPLE_RATE", 4000)
    inference_route._prepare_audio_clips_for_llama([_clip(b"aaab")])
    assert len(calls) == 4


def test_a_budget_set_by_an_enclosing_decode_misses(monkeypatch):
    calls = _counting_decoder(monkeypatch)
    clips = [_clip(b"recording")]
    inference_route._prepare_audio_clips_for_llama(clips)
    inference_route._decode_within(10.0, inference_route._prepare_audio_clips_for_llama, clips)
    assert len(calls) == 2


def test_only_one_entry_is_kept(monkeypatch):
    calls = _counting_decoder(monkeypatch)
    a, b = [_clip(b"recording a")], [_clip(b"recording bbbb")]
    for clips in (a, b, a):
        inference_route._prepare_audio_clips_for_llama(clips)
    assert len(calls) == 3
    assert len(inference_route._PREPARED_AUDIO_CACHE) == 1


def test_errors_are_not_cached(monkeypatch):
    calls = _counting_decoder(monkeypatch, fail_first = 2)
    clips = [_clip(b"recording")]
    for _ in range(2):
        with pytest.raises(ValueError, match = "broken container"):
            inference_route._prepare_audio_clips_for_llama(clips)
    assert not inference_route._PREPARED_AUDIO_CACHE
    inference_route._prepare_audio_clips_for_llama(clips)
    assert len(calls) == 3


def test_too_short_audio_is_refused_every_time(monkeypatch):
    calls = []
    monkeypatch.setattr(inference_route, "_sniff_audio_container", lambda _raw: None)
    monkeypatch.setattr(
        inference_route,
        "_decode_audio_mono",
        lambda raw: calls.append(raw) or (np.zeros(0, np.float32), 16_000),
    )
    for _ in range(2):
        with pytest.raises(inference_route._AudioTooShortError):
            inference_route._prepare_audio_clips_for_llama([_clip(b"x")])
    assert len(calls) == 2


def test_passthrough_only_requests_are_not_kept(monkeypatch):
    monkeypatch.setattr(inference_route, "_sniff_audio_container", lambda _raw: "wav")
    monkeypatch.setattr(inference_route, "_passthrough_audio_seconds", lambda *_a: 1.0)
    clip = base64.b64encode(b"RIFF....WAVE").decode("ascii")
    assert inference_route._prepare_audio_clips_for_llama([clip]) == [(clip, "wav")]
    assert not inference_route._PREPARED_AUDIO_CACHE


def test_a_real_flac_round_trips_byte_identically():
    sf = pytest.importorskip("soundfile")
    buffer = io.BytesIO()
    tone = 0.2 * np.sin(np.linspace(0, 440 * 2 * np.pi, 44_100)).astype(np.float32)
    sf.write(buffer, np.stack([tone, tone[::-1]], axis = 1), 44_100, format = "FLAC")
    clips = [_clip(buffer.getvalue())]
    expected, _ = inference_route._prepare_audio_clips_for_llama_uncached(clips)
    assert inference_route._prepare_audio_clips_for_llama(clips) == expected
    assert inference_route._PREPARED_AUDIO_CACHE
    assert inference_route._prepare_audio_clips_for_llama(clips) == expected
