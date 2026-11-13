# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A Hub dataset clip that is really a playlist must never reach FFmpeg as a local file."""

import uuid

import pytest

MP4_HEAD = b"\x00\x00\x00\x20ftypisom\x00\x00\x02\x00isomiso2avc1mp41"


def _patch_hub(monkeypatch, tmp_path, body):
    import huggingface_hub

    class Api:
        def list_repo_files(
            self,
            repo_id,
            repo_type = None,
            revision = None,
        ):
            return ["videos/clip00.mp4"]

    def download(repo_id, filename, **kw):
        path = tmp_path / filename
        path.parent.mkdir(parents = True, exist_ok = True)
        path.write_bytes(body)
        return str(path)

    monkeypatch.setattr(huggingface_hub, "HfApi", Api)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)


def _resolved(rows_or_messages, **kw):
    from utils.datasets.format_conversion import own_message_videos

    # A fresh repo name per call: conversions are cached per dataset.
    out = own_message_videos(
        rows_or_messages, dataset_name = f"someone/clips{uuid.uuid4().hex}", **kw
    )
    return [
        part["video"]
        for row in out
        for message in row["messages"]
        for part in message["content"]
        if part.get("type") == "video"
    ]


def _messages():
    return [{"messages": [{"role": "user", "content": [{"type": "video", "video": "clip00.mp4"}]}]}]


@pytest.mark.parametrize(
    "body",
    [
        b"#EXTM3U\n#EXT-X-TARGETDURATION:1\n#EXTINF:1,\nhttp://169.254.169.254/latest/meta-data/\n",
        b"ffconcat version 1.0\nfile 'http://127.0.0.1:8888/'\n",
        b'<?xml version="1.0"?><MPD></MPD>',
    ],
)
def test_a_hub_clip_that_is_a_playlist_is_refused(monkeypatch, tmp_path, body):
    _patch_hub(monkeypatch, tmp_path, body)
    with pytest.raises(ValueError, match = "playlist"):
        _resolved(_messages())


def test_a_real_hub_clip_still_resolves(monkeypatch, tmp_path):
    _patch_hub(monkeypatch, tmp_path, MP4_HEAD)
    assert _resolved(_messages()) == [str(tmp_path / "videos/clip00.mp4")]


def test_flat_rows_refuse_a_playlist_clip(monkeypatch, tmp_path):
    from utils.datasets.format_conversion import convert_video_to_vlm_format

    _patch_hub(monkeypatch, tmp_path, b"#EXTM3U\nhttp://169.254.169.254/\n")
    rows = [{"video": "videos/clip00.mp4", "answer": "a"}]
    try:
        out = convert_video_to_vlm_format(
            rows, "video", "answer", dataset_name = f"someone/clips{uuid.uuid4().hex}"
        )
    except ValueError:
        return
    paths = [
        p["video"]
        for r in out
        for m in r["messages"]
        for p in m["content"]
        if p.get("type") == "video"
    ]
    assert str(tmp_path / "videos/clip00.mp4") not in paths
