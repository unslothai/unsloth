# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""MiniMax-H3 reference uploads are decoded from their own bytes, never from files they name."""

from __future__ import annotations

import io

import pytest

from core.inference import video_minimax_h3 as h3

from .test_video_gallery import _real_mp4_bytes, _real_mp4_with_audio

av = pytest.importorskip("av")
np = pytest.importorskip("numpy")


def _real_wav_bytes(seconds: float = 1.0, rate: int = 8000) -> bytes:
    buf = io.BytesIO()
    with av.open(buf, "w", format = "wav") as out:
        stream = out.add_stream("pcm_s16le", rate = rate, layout = "mono")
        frame = av.AudioFrame.from_ndarray(
            np.full((1, int(seconds * rate)), 1000, dtype = np.int16), format = "s16", layout = "mono"
        )
        frame.sample_rate = rate
        for packet in stream.encode(frame):
            out.mux(packet)
        for packet in stream.encode(None):
            out.mux(packet)
    return buf.getvalue()


def _playlist(name: str) -> bytes:
    # FFmpeg probes this as the concat demuxer, which opens the named file relative to the cwd.
    return f"ffconcat version 1.0\nfile '{name}'\n".encode()


@pytest.fixture
def host_files(tmp_path, monkeypatch):
    (tmp_path / "host.wav").write_bytes(_real_wav_bytes())
    (tmp_path / "host.mp4").write_bytes(_real_mp4_bytes(frames = 24))
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_reference_audio_playlist_does_not_read_a_host_file(host_files):
    with pytest.raises(av.error.FFmpegError):
        h3.decode_h3_reference_audio(_playlist("host.wav"))


@pytest.mark.parametrize("trim", [(None, None), (0.0, 2.5)])
def test_reference_video_playlist_does_not_read_a_host_file(host_files, trim):
    with pytest.raises(av.error.FFmpegError):
        h3.decode_h3_reference_video(
            _playlist("host.mp4"), trim_start_seconds = trim[0], trim_end_seconds = trim[1]
        )


def test_reference_video_ordinal_fallback_does_not_read_a_host_file(host_files):
    with pytest.raises(av.error.FFmpegError):
        h3._decode_h3_video_trim_by_ordinal(_playlist("host.mp4"), av, (0.0, 2.5))


def test_every_reference_open_allows_no_protocols(monkeypatch):
    seen = []
    real_open = av.open

    def recording_open(*args, **kwargs):
        seen.append(kwargs.get("options"))
        return real_open(*args, **kwargs)

    clip, wav = _real_mp4_with_audio(seconds = 3), _real_wav_bytes()
    monkeypatch.setattr(av, "open", recording_open)

    frames, waveform, _ = h3.decode_h3_reference_video(clip)
    assert frames and waveform is not None
    frames, _, _ = h3.decode_h3_reference_video(clip, trim_start_seconds = 0.0, trim_end_seconds = 2.5)
    assert frames
    assert h3._decode_h3_video_trim_by_ordinal(clip, av, (0.0, 2.5))[0]
    waveform, rate = h3.decode_h3_reference_audio(wav)
    assert rate == 8000 and len(waveform) == 8000

    assert seen and all(opts == {"protocol_whitelist": "none"} for opts in seen), seen
