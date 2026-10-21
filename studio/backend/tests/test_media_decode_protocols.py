# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Audio decodes and the chat-video shrink never let a probed playlist open what it names."""

from __future__ import annotations

import base64
import io
import json
import sys
from pathlib import Path

import pytest

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

av = pytest.importorskip("av")
np = pytest.importorskip("numpy")

from core.inference import audio_inputs, llama_video_input  # noqa: E402
from utils.datasets import audio_decode  # noqa: E402


def _wav_bytes(seconds: float = 0.5, rate: int = 8000) -> bytes:
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
    # Probed as the concat demuxer, which opens the named file relative to the cwd.
    return f"ffconcat version 1.0\nfile '{name}'\n".encode()


@pytest.fixture
def host_wav(tmp_path, monkeypatch):
    (tmp_path / "host.wav").write_bytes(_wav_bytes())
    monkeypatch.chdir(tmp_path)


def test_dataset_audio_buffer_playlist_does_not_read_a_host_file(host_wav):
    with pytest.raises((av.error.FFmpegError, ValueError)):
        audio_decode._decode_with_av(io.BytesIO(_playlist("host.wav")))
    samples, rate = audio_decode._decode_with_av(io.BytesIO(_wav_bytes()))
    assert rate == 8000 and len(samples) == 4000


def test_speech_encode_buffer_playlist_does_not_read_a_host_file(host_wav):
    with pytest.raises((av.error.FFmpegError, ValueError)):
        audio_inputs.encode_wav(_playlist("host.wav"), "mp3")
    assert audio_inputs.encode_wav(_wav_bytes(), "mp3")[0]


@pytest.mark.parametrize(
    "module", [audio_inputs, audio_decode], ids = ["audio_inputs", "audio_decode"]
)
def test_a_saved_playlist_upload_does_not_read_a_host_file(module, host_wav, tmp_path):
    # A path may open `file`, so the playlist itself is refused before FFmpeg probes it.
    upload = tmp_path / "upload.wav"
    upload.write_bytes(_playlist("host.wav"))
    with pytest.raises(ValueError):
        module._av_open(av, str(upload))


@pytest.mark.parametrize(
    "module", [audio_inputs, audio_decode], ids = ["audio_inputs", "audio_decode"]
)
def test_a_saved_dash_manifest_upload_is_refused(module, host_wav, tmp_path):
    from utils.media_playlist import PlaylistRefused

    upload = tmp_path / "upload.m4a"
    upload.write_bytes(
        b'<?xml version="1.0"?>\n<MPD xmlns="urn:mpeg:dash:schema:mpd:2011" '
        b'profiles="urn:mpeg:dash:profile:isoff-on-demand:2011"><Period><AdaptationSet>'
        b"<Representation><BaseURL>host.wav</BaseURL></Representation></AdaptationSet></Period></MPD>\n"
    )
    with pytest.raises(PlaylistRefused):
        module._av_open(av, str(upload))


@pytest.mark.parametrize(
    "module", [audio_inputs, audio_decode], ids = ["audio_inputs", "audio_decode"]
)
def test_path_sources_may_open_files_only(module, tmp_path, monkeypatch):
    seen = []
    real_open = av.open

    def spy(source, *args, **kwargs):
        seen.append(kwargs.get("options"))
        return real_open(source, *args, **kwargs)

    path = tmp_path / "clip.wav"
    path.write_bytes(_wav_bytes())
    monkeypatch.setattr(av, "open", spy)

    with module._av_open(av, str(path)) as container:
        assert container.streams.audio
    # PyAV 19 refuses metadata_errors and the helper retries, so every attempt is checked.
    assert seen and all(options == {"protocol_whitelist": "file"} for options in seen)


def test_chat_video_ffmpeg_and_ffprobe_get_a_file_only_whitelist(monkeypatch):
    calls = []

    class _Result:
        def __init__(
            self,
            returncode,
            stdout = b"",
        ):
            self.returncode, self.stdout, self.stderr = returncode, stdout, b""

    def run(argv, _timeout):
        calls.append(list(argv))
        if Path(argv[0]).name == "ffprobe":
            geometry = {
                "streams": [{"width": 4000, "height": 4000, "avg_frame_rate": "10/1"}],
                "format": {"duration": "5.0"},
            }
            return _Result(0, json.dumps(geometry).encode())
        return _Result(1)

    monkeypatch.setattr(llama_video_input.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(llama_video_input, "_run", run)

    clip = base64.b64encode(b"not really a video").decode("ascii")
    assert llama_video_input.shrink_video_for_llama(clip, 1 << 20) == clip

    assert [Path(argv[0]).name for argv in calls] == ["ffprobe", "ffmpeg"]
    for argv in calls:
        i = argv.index("-protocol_whitelist")
        assert argv[i + 1] == "file"
        source = argv.index("-i") + 1 if "-i" in argv else len(argv) - 1
        assert i < source


@pytest.mark.parametrize(
    "head",
    [
        b"#EXTM3U\n#EXTINF:1,\n/etc/hostname\n",
        b"\xef\xbb\xbf  ffconcat version 1.0\nfile '/etc/hostname'\n",
        # DASH is probed by "<MPD" anywhere in the buffer, after an XML prolog here.
        b'<?xml version="1.0"?>\n<MPD xmlns="urn:mpeg:dash:schema:mpd:2011" '
        b'profiles="urn:mpeg:dash:profile:isoff-on-demand:2011"><Period><AdaptationSet>'
        b"<Representation><BaseURL>host.wav</BaseURL></Representation></AdaptationSet></Period></MPD>\n",
        b"<!--" + b" " * 8192 + b'--><mpd profiles="URN:MPEG:DASH:PROFILE:ISOFF-LIVE:2011"></mpd>',
    ],
)
def test_a_playlist_upload_is_refused_before_ffmpeg_sees_it(tmp_path, head):
    from utils.media_playlist import PlaylistRefused, refuse_playlist

    path = tmp_path / "upload.mp4"
    path.write_bytes(head)
    for source in (str(path), head, io.BytesIO(head)):
        with pytest.raises(PlaylistRefused):
            refuse_playlist(source)
    # Real media headers pass, as does "<MPD" text that FFmpeg would not probe as DASH.
    refuse_playlist(b"\x00\x00\x00\x18ftypmp42")
    refuse_playlist(b"ID3\x04\x00\x00\x00\x00\x00\x10TIT2 <MPD> notes")
    refuse_playlist(b"\x00" + head)
    refuse_playlist(b"RIFF\x00\x00\x00\x00WAVEfmt ")


def test_shrink_refuses_a_playlist_clip(monkeypatch):
    from core.inference import llama_video_input
    from utils.media_playlist import PlaylistRefused

    clip = base64.b64encode(b"#EXTM3U\n/etc/hostname\n").decode()
    monkeypatch.setattr(llama_video_input.shutil, "which", lambda name: f"/usr/bin/{name}")
    with pytest.raises(PlaylistRefused):
        llama_video_input.shrink_video_for_llama(clip, 1 << 20)
