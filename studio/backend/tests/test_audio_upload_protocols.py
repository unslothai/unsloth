# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Uploaded audio for speech-to-text and chat attachments is decoded from its own bytes only."""

from __future__ import annotations

import pytest

av = pytest.importorskip("av")
np = pytest.importorskip("numpy")

import routes.inference as inference_route
from core.inference import stt_sidecar

from .test_video_minimax_h3_reference_protocols import _playlist, _real_wav_bytes


@pytest.fixture
def host_wav(tmp_path, monkeypatch):
    (tmp_path / "host.wav").write_bytes(_real_wav_bytes())
    monkeypatch.chdir(tmp_path)


def test_transcription_upload_playlist_does_not_read_a_host_file(host_wav):
    with pytest.raises(stt_sidecar.SttAudioDecodeError):
        stt_sidecar._decode_audio_bounded(_playlist("host.wav"))
    assert len(stt_sidecar._decode_audio_bounded(_real_wav_bytes())) == 16000


def test_chat_audio_attachment_playlist_does_not_read_a_host_file(host_wav):
    with pytest.raises(av.error.FFmpegError):
        inference_route._decode_audio_mono_with_av(_playlist("host.wav"))
    samples, rate = inference_route._decode_audio_mono_with_av(_real_wav_bytes())
    assert rate == 8000 and len(samples) == 8000
