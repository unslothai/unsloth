# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Every upload decoder refuses an HLS or DASH playlist before any decoder sees the bytes."""

from __future__ import annotations

import base64
import os
import sys
import types

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

import routes.inference as inference_route
from core.inference import llama_video_input
from utils.media_playlist import PlaylistRefused

HLS = b"#EXTM3U\n#EXT-X-VERSION:3\n#EXT-X-TARGETDURATION:1\n"
DASH = (
    b'<?xml version="1.0"?>\n<MPD xmlns="urn:mpeg:dash:schema:mpd:2011" '
    b'profiles="urn:mpeg:dash:profile:isoff-on-demand:2011"><Period/></MPD>\n'
)
PLAYLISTS = pytest.mark.parametrize("payload", [HLS, DASH], ids = ["hls", "dash"])
# Leading bytes of real containers; the decoders are mocked, so only the playlist check reads them.
VALID_MEDIA = {
    "wav": b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00",
    "mp3": b"ID3\x04\x00\x00\x00\x00\x00\x00\xff\xfb\x90\x00",
    "flac": b"fLaC\x00\x00\x00\x22",
    "ogg": b"OggS\x00\x02\x00\x00\x00\x00",
    "m4a": b"\x00\x00\x00\x20ftypM4A \x00\x00\x00\x00",
    "mp4": b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00",
    "webm": b"\x1aE\xdf\xa3\x9fB\x86\x81\x01webm",
    "mov": b"\x00\x00\x00\x14ftypqt  \x00\x00\x00\x00",
}
VALID = pytest.mark.parametrize("payload", list(VALID_MEDIA.values()), ids = list(VALID_MEDIA))


def _b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode()


@pytest.fixture
def studio_home(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio"))


@pytest.fixture
def fake_torchaudio(monkeypatch):
    calls = []

    def info(path):
        calls.append(("info", path))
        return types.SimpleNamespace(sample_rate = 16000, num_frames = 16000, num_channels = 1)

    def load(path, num_frames = -1):
        calls.append(("load", path))
        return torch.zeros(1, 16000), 16000

    monkeypatch.setitem(sys.modules, "torchaudio", types.SimpleNamespace(info = info, load = load))
    return calls


@pytest.fixture
def fake_torchcodec(monkeypatch):
    """torchaudio 2.9+ without librosa: info() is gone and the last resort is torchcodec."""
    calls = []

    class AudioDecoder:
        def __init__(self, path):
            calls.append(path)
            self.metadata = types.SimpleNamespace(
                sample_rate = 16000, duration_seconds = 1.0, num_channels = 1
            )

        def get_samples_played_in_range(self, start, stop):
            return types.SimpleNamespace(data = torch.zeros(1, 16000))

    def info(path):
        raise AttributeError("torchaudio.info was removed")

    def no_decoder(raw):
        raise RuntimeError("no soundfile, PyAV or librosa")

    decoders = types.ModuleType("torchcodec.decoders")
    decoders.AudioDecoder = AudioDecoder
    torchcodec = types.ModuleType("torchcodec")
    torchcodec.decoders = decoders
    monkeypatch.setitem(sys.modules, "torchcodec", torchcodec)
    monkeypatch.setitem(sys.modules, "torchcodec.decoders", decoders)
    monkeypatch.setitem(sys.modules, "torchaudio", types.SimpleNamespace(info = info, load = None))
    monkeypatch.setattr(inference_route, "_decode_audio_mono", no_decoder)
    return calls


@PLAYLISTS
def test_torchaudio_never_opens_a_playlist(studio_home, fake_torchaudio, payload):
    with pytest.raises(PlaylistRefused, match = "Playlists are not accepted"):
        inference_route._decode_audio_base64(_b64(payload))
    assert fake_torchaudio == []


@PLAYLISTS
def test_torchcodec_fallback_never_opens_a_playlist(studio_home, fake_torchcodec, payload):
    with pytest.raises(PlaylistRefused, match = "Playlists are not accepted"):
        inference_route._decode_audio_base64(_b64(payload))
    assert fake_torchcodec == []


@VALID
def test_torchaudio_still_decodes_real_audio(studio_home, fake_torchaudio, payload):
    assert inference_route._decode_audio_base64(_b64(payload)).shape == (16000,)
    assert [name for name, _ in fake_torchaudio] == ["info", "load"]


@VALID
def test_torchcodec_fallback_still_decodes_real_audio(studio_home, fake_torchcodec, payload):
    assert inference_route._decode_audio_base64(_b64(payload)).shape == (16000,)
    assert len(fake_torchcodec) == 1


@pytest.fixture
def mono_decoders(monkeypatch):
    calls = []

    def decode(name):
        def run(raw):
            calls.append(name)
            return np.zeros(8000, dtype = np.float32), 8000

        return run

    monkeypatch.setattr(inference_route, "_decode_audio_mono_with_soundfile", decode("soundfile"))
    monkeypatch.setattr(inference_route, "_decode_audio_mono_with_av", decode("av"))
    return calls


@PLAYLISTS
def test_mono_decoder_refuses_a_playlist_before_any_backend(mono_decoders, payload):
    with pytest.raises(PlaylistRefused, match = "Playlists are not accepted"):
        inference_route._decode_audio_mono(payload)
    assert mono_decoders == []


@VALID
def test_mono_decoder_still_decodes_real_audio(mono_decoders, payload):
    samples, rate = inference_route._decode_audio_mono(payload)
    assert rate == 8000 and len(samples) == 8000
    assert mono_decoders == ["soundfile"]


@pytest.fixture
def no_ffmpeg(monkeypatch):
    monkeypatch.setattr(llama_video_input.shutil, "which", lambda name: None)


@PLAYLISTS
def test_llama_video_refuses_a_playlist_without_ffmpeg(no_ffmpeg, payload):
    with pytest.raises(PlaylistRefused, match = "Playlists are not accepted"):
        llama_video_input.shrink_video_for_llama(_b64(payload), max_bytes = 1 << 20)


@pytest.mark.parametrize("kind", ["mp4", "webm", "mov"])
def test_llama_video_without_ffmpeg_still_passes_a_clip_through(no_ffmpeg, kind):
    clip = _b64(VALID_MEDIA[kind])
    assert llama_video_input.shrink_video_for_llama(clip, max_bytes = 1 << 20) == clip


@PLAYLISTS
def test_mlx_video_clip_refuses_a_playlist_before_writing(tmp_path, monkeypatch, payload):
    import tempfile

    from core.inference import mlx_inference

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    with pytest.raises(PlaylistRefused, match = "Playlists are not accepted"):
        mlx_inference._write_video_clip(_b64(payload))
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("kind", ["mp4", "webm", "mov"])
def test_mlx_video_clip_still_writes_a_real_clip(tmp_path, monkeypatch, kind):
    import tempfile

    from core.inference import mlx_inference

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    path = mlx_inference._write_video_clip(_b64(VALID_MEDIA[kind]))
    try:
        with open(path, "rb") as handle:
            assert handle.read() == VALID_MEDIA[kind]
    finally:
        os.unlink(path)
