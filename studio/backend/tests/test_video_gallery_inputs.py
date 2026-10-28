# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A gallery video as an input: one frame as PNG, or its audio track as an audio input."""

from __future__ import annotations

import io

import pytest
from PIL import Image

from core.inference import audio_inputs, video_gallery
from utils.account_context import run_as

from .test_account_media_isolation import ALICE, BOB, _client, isolated  # noqa: F401
from .test_video_gallery import _real_mp4_bytes, _real_mp4_with_audio

pytest.importorskip("av")


def _clip(data: bytes, prompt: str = "a grey ramp") -> str:
    meta = {
        "prompt": prompt,
        "width": 64,
        "height": 48,
        "num_frames": 8,
        "fps": 8,
        "duration_s": 1.0,
        "steps": 1,
        "guidance": 1.0,
        "seed": 7,
        "model": "test/none",
        "created_at": "2026-01-01T00:00:00+00:00",
    }
    return run_as(ALICE, video_gallery.save, data, meta)["id"]


def _grey(png: bytes) -> int:
    image = Image.open(io.BytesIO(png)).convert("L")
    return image.getpixel((image.width // 2, image.height // 2))


def test_frames_pick_first_last_and_a_time():
    # Frame i is grey (i * 30): 8 frames at 8 fps, so 0.5 s shows frame 4.
    video_id = _clip(_real_mp4_bytes(frames = 8, size = (64, 48), rate = 8))
    url = f"/api/inference/video/gallery/{video_id}/frame"
    with _client(ALICE) as client:
        for at, grey in (("first", 0), ("last", 210), ("0.5", 120), ("0", 0)):
            response = client.get(url, params = {"at": at})
            assert response.status_code == 200, response.text
            assert response.headers["content-type"] == "image/png"
            assert abs(_grey(response.content) - grey) <= 12, (at, _grey(response.content))
        assert Image.open(io.BytesIO(client.get(url).content)).size == (64, 48)
        small = client.get(url, params = {"at": "last", "max_px": 32})
        assert Image.open(io.BytesIO(small.content)).size == (32, 24)
        past = client.get(url, params = {"at": "5"})
        assert past.status_code == 400
        assert "past the end" in past.json()["detail"]
        for bad in ("middle", "-1", "nan"):
            assert client.get(url, params = {"at": bad}).status_code == 400
        assert client.get(url, params = {"max_px": 8}).status_code == 422
    with _client(BOB) as client:
        assert client.get(url).status_code == 404


def test_frame_times_count_from_a_stream_that_does_not_start_at_zero(tmp_path):
    # Frame i is grey (i * 30) at 8 fps, with timestamps starting at 1 s: 0.5 s is still frame 4.
    av = pytest.importorskip("av")
    np = pytest.importorskip("numpy")
    path = tmp_path / "offset.mp4"
    with av.open(str(path), "w", format = "mp4") as out:
        stream = out.add_stream("mpeg4", rate = 8)
        stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
        for i in range(8):
            frame = av.VideoFrame.from_ndarray(
                np.full((48, 64, 3), i * 30, dtype = np.uint8), format = "rgb24"
            )
            frame.pts = i + 8
            for packet in stream.encode(frame):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)
    with av.open(str(path)) as src:
        assert src.streams.video[0].start_time  # the fixture really is offset
    for at, grey in (("first", 0), ("0.5", 120), ("0", 0), ("last", 210)):
        assert abs(_grey(video_gallery.frame_png(path, at)) - grey) <= 12, at


def test_a_clip_that_cannot_be_read_gets_a_plain_error_without_server_paths(monkeypatch, tmp_path):
    video_id = _clip(b"not an mp4 at all, only bytes")
    url = f"/api/inference/video/gallery/{video_id}/frame"
    with _client(ALICE) as client:
        broken = client.get(url)
        assert broken.status_code == 400
        assert broken.json()["detail"] == "This clip could not be decoded."
        # The file can go away between the ownership check and the decode.
        monkeypatch.setattr(video_gallery, "owned_video_path", lambda _id: tmp_path / "gone.mp4")
        gone = client.get(url)
    assert gone.status_code == 404
    assert gone.json()["detail"] == "Video not found."


def test_audio_track_becomes_a_trimmed_idempotent_audio_input():
    video_id = _clip(_real_mp4_with_audio(seconds = 1))
    url = f"/api/inference/video/gallery/{video_id}/audio-input"
    body = {"start_s": 0.25, "end_s": 0.75}
    with _client(ALICE) as client:
        first = client.post(url, json = body)
        assert first.status_code == 201, first.text
        record = first.json()
        assert abs(record["duration_s"] - 0.5) <= 0.02
        assert record["name"] == "a grey ramp (audio).wav"
        again = client.post(url, json = body)
        assert again.status_code == 200
        assert again.json()["id"] == record["id"]
        assert client.get(record["url"]).status_code == 200
        whole = client.post(url, json = {})
        assert whole.status_code == 201
        assert whole.json()["duration_s"] >= 0.95
        past = client.post(url, json = {"start_s": 5})
        assert past.status_code == 400
        assert "past the end" in past.json()["detail"]
        for invalid in ({"start_s": 0.5, "end_s": 0.5}, {"start_s": -1}, {"name": "x"}):
            assert client.post(url, json = invalid).status_code == 422
    with _client(BOB) as client:
        assert client.post(url, json = body).status_code == 404
    leftovers = list(run_as(ALICE, audio_inputs.inputs_dir).glob(".*.tmp"))
    assert leftovers == []


def test_an_audio_input_name_keeps_a_prompt_with_slashes_and_line_breaks():
    video_id = _clip(_real_mp4_with_audio(seconds = 1), prompt = "cats/dogs\nat dusk")
    with _client(ALICE) as client:
        response = client.post(f"/api/inference/video/gallery/{video_id}/audio-input", json = {})
    assert response.status_code == 201, response.text
    assert response.json()["name"] == "cats dogs at dusk (audio).wav"


def test_a_silent_video_has_no_audio_track_to_extract():
    video_id = _clip(_real_mp4_bytes())
    with _client(ALICE) as client:
        response = client.post(f"/api/inference/video/gallery/{video_id}/audio-input", json = {})
    assert response.status_code == 400
    assert response.json()["detail"].startswith("This video has no audio track")
    assert list(run_as(ALICE, audio_inputs.inputs_dir).iterdir()) == []


def test_a_frame_png_keeps_pillows_default_encoding(monkeypatch, tmp_path):
    # MCP previews and inputs gate on these bytes (INLINE_CAP, MAX_IMAGE_BYTES), so a faster but
    # larger zlib level could flip an inline preview to a link. The gallery level must not apply.
    from core.inference import image_gallery

    av = pytest.importorskip("av")
    path = tmp_path / "clip.mp4"
    path.write_bytes(_real_mp4_bytes(frames = 4, size = (64, 48), rate = 4))
    monkeypatch.delenv(image_gallery.PNG_COMPRESS_LEVEL_ENV, raising = False)
    with av.open(str(path)) as src:
        image = next(src.decode(src.streams.video[0])).to_image()
    expected = io.BytesIO()
    image.save(expected, format = "PNG")
    assert video_gallery.frame_png(path, "first") == expected.getvalue()
    monkeypatch.setenv(image_gallery.PNG_COMPRESS_LEVEL_ENV, "1")
    assert video_gallery.frame_png(path, "first") == expected.getvalue()
