# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import base64

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
