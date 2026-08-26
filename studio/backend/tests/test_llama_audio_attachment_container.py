# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import base64
import io
import sys

import av
import numpy as np
import routes.inference as inference_route


def _id3_prefix() -> bytes:
    # ID3v2.4, no payload and no footer.
    return b"ID3\x04\x00\x00\x00\x00\x00\x00"


def test_audio_container_sniffer_distinguishes_mpeg_layers_and_adts():
    # MPEG-1 Layer III, MPEG-1 Layer II, and ADTS AAC respectively.
    assert inference_route._sniff_audio_container(b"\xff\xfb\x90\x64") == "mp3"
    assert inference_route._sniff_audio_container(b"\xff\xfd\x90\x64") is None
    assert inference_route._sniff_audio_container(b"\xff\xf1\x50\x80") is None


def test_audio_container_sniffer_checks_the_frame_after_id3():
    assert inference_route._sniff_audio_container(_id3_prefix() + b"\xff\xfb\x90\x64") == "mp3"
    assert inference_route._sniff_audio_container(_id3_prefix() + b"\xff\xfd\x90\x64") is None
    assert inference_route._sniff_audio_container(_id3_prefix() + b"\xff\xf1\x50\x80") is None


def test_aac_and_mp2_are_transcoded_instead_of_forwarded_as_mp3(monkeypatch):
    decoded = []
    monkeypatch.setattr(
        inference_route,
        "_decode_audio_mono",
        lambda raw: (decoded.append(raw) or object(), 16_000),
    )
    monkeypatch.setattr(
        inference_route,
        "_fit_transcoded_audio_to_wav_cap",
        lambda array, sample_rate: (array, sample_rate),
    )
    monkeypatch.setattr(
        inference_route,
        "_mono_f32_to_wav_bytes",
        lambda _array, _sample_rate: b"transcoded wav",
    )

    for raw in (b"\xff\xfd\x90\x64", b"\xff\xf1\x50\x80"):
        encoded, format_name = inference_route._prepare_audio_for_llama(
            base64.b64encode(raw).decode("ascii")
        )
        assert format_name == "wav"
        assert base64.b64decode(encoded) == b"transcoded wav"

    assert decoded == [b"\xff\xfd\x90\x64", b"\xff\xf1\x50\x80"]


def _encode_wma() -> bytes:
    buffer = io.BytesIO()
    with av.open(buffer, mode = "w", format = "asf") as output:
        stream = output.add_stream("wmav2", rate = 44_100)
        stream.layout = "stereo"
        stream.bit_rate = 128_000
        indexes = np.arange(8_192)
        mono = np.sin(indexes * 2 * np.pi * 440 / 44_100).astype(np.float32)
        frame = av.AudioFrame.from_ndarray(
            np.stack([mono, mono]),
            format = "fltp",
            layout = "stereo",
        )
        frame.sample_rate = 44_100
        for packet in stream.encode(frame):
            output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
    return buffer.getvalue()


def _encode_amr() -> bytes:
    buffer = io.BytesIO()
    with av.open(buffer, mode = "w", format = "amr") as output:
        stream = output.add_stream("amr_nb", rate = 8_000)
        stream.layout = "mono"
        stream.bit_rate = 12_200
        indexes = np.arange(3_200)
        samples = (np.sin(indexes * 2 * np.pi * 440 / 8_000) * 16_000).astype(np.int16)
        frame = av.AudioFrame.from_ndarray(
            samples.reshape(1, -1),
            format = "s16",
            layout = "mono",
        )
        frame.sample_rate = 8_000
        for packet in stream.encode(frame):
            output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
    return buffer.getvalue()


def test_wma_and_amr_decode_with_no_librosa(monkeypatch):
    # Match the GGUF-only installation: PyAV is present, librosa is not. Force
    # libsndfile out too so the test proves the PyAV fallback owns both formats.
    monkeypatch.setitem(sys.modules, "soundfile", None)
    monkeypatch.setitem(sys.modules, "librosa", None)

    for raw, expected_rate in ((_encode_wma(), 44_100), (_encode_amr(), 8_000)):
        samples, sample_rate = inference_route._decode_audio_mono(raw)
        assert sample_rate == expected_rate
        assert samples.ndim == 1
        assert samples.dtype == np.float32
        assert np.max(np.abs(samples)) > 0


def test_pyav_duration_limit_stops_before_concatenating(monkeypatch):
    raw = _encode_amr()
    monkeypatch.setitem(sys.modules, "soundfile", None)
    monkeypatch.setitem(sys.modules, "librosa", None)
    monkeypatch.setattr(inference_route, "_MAX_AUDIO_SECONDS", 0)

    def concatenate_after_limit(*_args, **_kwargs):
        raise AssertionError("duration-limited audio must not be concatenated")

    monkeypatch.setattr(np, "concatenate", concatenate_after_limit)
    try:
        inference_route._decode_audio_mono(raw)
    except inference_route._DecodedAudioTooLongError:
        pass
    else:
        raise AssertionError("expected PyAV to stop at the decoded-duration limit")
