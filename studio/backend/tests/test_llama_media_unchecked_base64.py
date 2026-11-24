# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Media reaching llama-server must be the bytes Studio checked: its own base64 decoder reads
unpadded, prefixed and bare payloads that Python's would skip, and its FFmpeg opens playlists."""

from __future__ import annotations

import base64
import io

import pytest
from fastapi import HTTPException
from PIL import Image

from core.inference import llama_video_input
from utils.media_playlist import PlaylistRefused

PLAYLIST = b"ffconcat version 1.0\nfile Pictures/scan-0.png\nduration 2\nfile Pictures/scan-0.png\n"


def _padded_playlist() -> str:
    body = PLAYLIST
    while not base64.b64encode(body).endswith(b"=="):
        body += b"\n"
    return base64.b64encode(body).decode("ascii")


@pytest.fixture(autouse = True)
def _with_tools(monkeypatch):
    monkeypatch.setattr(llama_video_input.shutil, "which", lambda name: f"/usr/bin/{name}")


def test_unpadded_playlist_clip_is_refused():
    with pytest.raises(PlaylistRefused):
        llama_video_input.shrink_video_for_llama(_padded_playlist().rstrip("="), 1 << 20)


def test_doubled_data_header_never_forwards_the_playlist(monkeypatch):
    monkeypatch.setattr(llama_video_input, "video_tools", lambda *_a, **_k: None)
    clip = "data:video/mp4;base64," + _padded_playlist()
    try:
        forwarded = llama_video_input.shrink_video_for_llama(clip, 1 << 20)
    except (PlaylistRefused, ValueError):
        return
    assert not base64.b64decode(forwarded).lstrip().startswith(b"ffconcat")


def test_a_canonical_clip_is_forwarded_unchanged(monkeypatch):
    monkeypatch.setattr(llama_video_input, "video_tools", lambda *_a, **_k: None)
    clip = base64.b64encode(b"\x1aE\xdf\xa3" + b"\x00" * 64).decode("ascii")
    assert llama_video_input.shrink_video_for_llama(clip, 1 << 20) == clip


def test_an_unpadded_clip_is_forwarded_as_the_bytes_checked(monkeypatch):
    monkeypatch.setattr(llama_video_input, "video_tools", lambda *_a, **_k: None)
    raw = b"\x1aE\xdf\xa3" + b"\x00" * 65
    clip = base64.b64encode(raw).decode("ascii").rstrip("=")
    assert base64.b64decode(llama_video_input.shrink_video_for_llama(clip, 1 << 20)) == raw


def _image_part(url: str) -> dict:
    return {"type": "image_url", "image_url": {"url": url}}


def test_bare_base64_playlist_image_is_refused():
    from routes.inference import _normalize_openai_image_parts_for_llama

    part = _image_part(base64.b64encode(PLAYLIST).decode("ascii"))
    with pytest.raises(HTTPException) as err:
        _normalize_openai_image_parts_for_llama([{"role": "user", "content": [part]}])
    assert err.value.status_code == 400


def test_bare_base64_image_still_reaches_the_model():
    from routes.inference import _normalize_openai_image_parts_for_llama

    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (200, 10, 10)).save(buf, format = "PNG")
    part = _image_part(base64.b64encode(buf.getvalue()).decode("ascii"))
    _normalize_openai_image_parts_for_llama([{"role": "user", "content": [part]}])
    url = part["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")
    with Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))) as img:
        assert img.getpixel((0, 0)) == (200, 10, 10)


def test_an_id3_tag_does_not_hide_a_playlist():
    from utils.media_playlist import refuse_playlist

    body = b"#EXTM3U\n/srv/meetings/board-call.flac\n"
    tag = b"ID3\x04\x00\x00\x00\x00\x00\x00"
    with pytest.raises(PlaylistRefused):
        refuse_playlist(tag + body)
    # A tag holding the words is still just a tag.
    refuse_playlist(b"ID3\x04\x00\x00\x00\x00\x00\x10TIT2 <MPD> notes")


def test_decision_api_refuses_playlists_in_audio_and_state():
    from routes import systemone

    clip = "data:audio/wav;base64," + base64.b64encode(PLAYLIST).decode("ascii")
    with pytest.raises(HTTPException) as err:
        systemone._validate_audio([clip])
    assert err.value.status_code == 422
    state = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64," + _padded_playlist().rstrip("=")},
                },
                {"type": "text", "text": "what colour"},
            ],
        }
    ]
    with pytest.raises(HTTPException) as err:
        systemone._refuse_state_playlists(state)
    assert err.value.status_code == 422
    systemone._refuse_state_playlists([{"role": "user", "content": "plain text"}])
