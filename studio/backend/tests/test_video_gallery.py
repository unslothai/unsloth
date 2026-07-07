# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unit tests for the disk-backed video gallery: MP4 + JSON-sidecar round-trips,
listing order, safe id handling, orphan-pair skipping, and delete/clear."""

from __future__ import annotations

import json
import os
from pathlib import Path

import core.inference.video_gallery as gallery


import pytest


@pytest.fixture(autouse = True)
def _tmp_gallery(monkeypatch, tmp_path):
    # Point the gallery at a throwaway root instead of ~/.unsloth/studio.
    monkeypatch.setattr(gallery, "studio_root", lambda: tmp_path)


def _mp4(tag = b"\x00\x00\x00\x18ftypmp42"):
    # Not a real container; the gallery treats the bytes as opaque payload.
    return tag


def _meta(**over):
    base = {
        "prompt": "a sloth surfing",
        "negative_prompt": None,
        "width": 1024,
        "height": 576,
        "num_frames": 49,
        "steps": 30,
        "guidance": 6.0,
        "seed": 7,
        "model": "unsloth/some-video-model",
        "created_at": 100.0,
    }
    base.update(over)
    return base


def test_save_writes_pair_and_round_trips():
    record = gallery.save(_mp4(), _meta())
    assert record["id"] and record["url"].endswith(f"{record['id']}/file")

    # Both files of the pair exist: the mp4 payload and the json recipe sidecar.
    directory = gallery.gallery_dir()
    assert (directory / f"{record['id']}.mp4").is_file()
    sidecar = directory / f"{record['id']}.json"
    assert json.loads(sidecar.read_text(encoding = "utf-8"))["prompt"] == "a sloth surfing"

    listed = gallery.list_videos()
    assert len(listed) == 1
    assert listed[0]["prompt"] == "a sloth surfing" and listed[0]["seed"] == 7
    # Meta fields survive the sidecar round-trip untouched.
    assert listed[0]["num_frames"] == 49 and listed[0]["model"] == "unsloth/some-video-model"


def test_url_shape():
    record = gallery.save(_mp4(), _meta())
    assert record["url"] == f"/api/inference/video/gallery/{record['id']}/file"


def _save_with_mtime(prompt: str, t: float) -> dict:
    record = gallery.save(_mp4(), _meta(prompt = prompt, created_at = t))
    # Listing orders by mp4 mtime; set it explicitly so a tight test loop can't tie it.
    os.utime(gallery.gallery_dir() / f"{record['id']}.mp4", (t, t))
    return record


def test_list_is_newest_first():
    old = _save_with_mtime("old", 100.0)
    new = _save_with_mtime("new", 200.0)
    assert [r["id"] for r in gallery.list_videos()] == [new["id"], old["id"]]


def test_list_paginates_with_limit_offset():
    # 5 videos, newest (t=4) first.
    for i in range(5):
        _save_with_mtime(f"p{i}", float(i))
    page1 = gallery.list_videos(limit = 2, offset = 0)
    page2 = gallery.list_videos(limit = 2, offset = 2)
    assert [r["prompt"] for r in page1] == ["p4", "p3"]
    assert [r["prompt"] for r in page2] == ["p2", "p1"]
    # limit=None still returns everything from the offset.
    assert len(gallery.list_videos()) == 5
    assert len(gallery.list_videos(offset = 4)) == 1


def test_video_path_rejects_unsafe_ids():
    # Traversal / bad chars / absolute paths never resolve to a path.
    assert gallery.video_path("../../etc/passwd") is None
    assert gallery.video_path("/etc/passwd") is None
    assert gallery.video_path("a/b") is None
    assert gallery.video_path("missing") is None


def test_video_path_returns_mp4_for_saved_id():
    record = gallery.save(_mp4(), _meta())
    path = gallery.video_path(record["id"])
    assert path is not None and path.name == f"{record['id']}.mp4"


def test_delete_removes_both_files():
    record = gallery.save(_mp4(), _meta(prompt = "a"))
    gallery.save(_mp4(), _meta(prompt = "b"))
    directory = gallery.gallery_dir()
    assert gallery.delete(record["id"]) is True
    # Both halves of the pair are gone.
    assert not (directory / f"{record['id']}.mp4").exists()
    assert not (directory / f"{record['id']}.json").exists()
    assert gallery.delete(record["id"]) is False  # already gone
    assert len(gallery.list_videos()) == 1


def test_delete_keeps_sidecar_listable_when_mp4_unlink_fails(monkeypatch):
    # delete() must remove the MP4 FIRST: list_videos globs *.mp4 but requires a readable sidecar,
    # so if the sidecar were dropped first and the mp4 unlink then failed (a Windows lock from a
    # concurrent stream/transcode), the still-present mp4 would vanish from the gallery with no way
    # to retry. Simulate the mp4 unlink failing and assert the video stays listable (sidecar kept).
    record = gallery.save(_mp4(), _meta(prompt = "keep"))
    directory = gallery.gallery_dir()
    mp4 = directory / f"{record['id']}.mp4"
    sidecar = directory / f"{record['id']}.json"

    real_unlink = Path.unlink

    def _fail_on_mp4(self, *a, **k):
        if self.suffix == ".mp4":
            raise PermissionError("mp4 locked")
        return real_unlink(self, *a, **k)

    # Patch Path.unlink (what delete() actually calls) rather than os.unlink: on Python 3.10
    # Path.unlink dispatches through a cached _accessor bound to os.unlink at import, so patching
    # os.unlink there has no effect and the mp4 delete would wrongly succeed. Scope it to its own
    # context so undoing it does NOT also revert the autouse fixture's studio_root redirect (both
    # share the function-scoped monkeypatch); otherwise list_videos below would read the real home.
    with pytest.MonkeyPatch.context() as m:
        m.setattr(Path, "unlink", _fail_on_mp4)
        assert gallery.delete(record["id"]) is False  # mp4 unlink failed
    # The sidecar was NOT dropped, so the record is still listable and the user can retry.
    assert sidecar.exists() and mp4.exists()
    assert [r["prompt"] for r in gallery.list_videos()] == ["keep"]
    assert gallery.delete(record["id"]) is True  # retry now succeeds


def test_clear_returns_count():
    gallery.save(_mp4(), _meta(prompt = "a"))
    gallery.save(_mp4(), _meta(prompt = "b"))
    assert gallery.clear() == 2
    assert gallery.list_videos() == []
    # No stray sidecars left behind after a clear.
    assert list(gallery.gallery_dir().glob("*.json")) == []


def test_list_skips_orphan_mp4_without_sidecar():
    # An MP4 with no readable json sidecar (a hand-dropped file) is not a record.
    orphan = gallery.gallery_dir() / "orphan.mp4"
    orphan.write_bytes(_mp4())
    gallery.save(_mp4(), _meta(prompt = "ours"))
    listed = gallery.list_videos()
    assert [r["prompt"] for r in listed] == ["ours"]


def test_list_skips_orphan_sidecar_without_mp4():
    # A json sidecar with no MP4 alongside it is never surfaced (listing globs mp4s).
    orphan = gallery.gallery_dir() / "lonely.json"
    orphan.write_text(json.dumps(_meta(prompt = "no video")), encoding = "utf-8")
    gallery.save(_mp4(), _meta(prompt = "ours"))
    listed = gallery.list_videos()
    assert [r["prompt"] for r in listed] == ["ours"]


def test_orphan_mp4_in_window_does_not_drop_valid_videos():
    # An orphan MP4 sorting INTO the requested page must not consume a window slot
    # and drop a valid video that sorts after it: paging is over readable records.
    _save_with_mtime("p2", 100.0)
    orphan = gallery.gallery_dir() / "zzz_orphan.mp4"
    orphan.write_bytes(_mp4())  # newest by mtime (set below), sorts first
    os.utime(orphan, (300.0, 300.0))
    _save_with_mtime("p1", 200.0)
    # First page of 2 must still return both real videos, not [p1] (orphan eating a slot).
    page1 = gallery.list_videos(limit = 2, offset = 0)
    assert [r["prompt"] for r in page1] == ["p1", "p2"]


def test_list_skips_corrupt_sidecar():
    # A sidecar that is not valid JSON is treated as a foreign/orphan mp4 and skipped.
    directory = gallery.gallery_dir()
    (directory / "broken.mp4").write_bytes(_mp4())
    (directory / "broken.json").write_text("{not json", encoding = "utf-8")
    gallery.save(_mp4(), _meta(prompt = "ours"))
    listed = gallery.list_videos()
    assert [r["prompt"] for r in listed] == ["ours"]


def _real_mp4_bytes() -> bytes:
    # A real (tiny) MP4 for the transcode tests: 8 frames of flat color at
    # 32x32, encoded with mpeg4 (bundled in every PyAV build, unlike libx264).
    av = pytest.importorskip("av")
    np = pytest.importorskip("numpy")
    import io

    buf = io.BytesIO()
    with av.open(buf, "w", format = "mp4") as out:
        stream = out.add_stream("mpeg4", rate = 8)
        stream.width = 32
        stream.height = 32
        stream.pix_fmt = "yuv420p"
        for i in range(8):
            frame = av.VideoFrame.from_ndarray(
                np.full((32, 32, 3), i * 30, dtype = np.uint8), format = "rgb24"
            )
            for packet in stream.encode(frame):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)
    return buf.getvalue()


def test_transcode_gif_and_webm_produce_real_containers():
    record = gallery.save(_real_mp4_bytes(), _meta())
    gif = gallery.transcode(record["id"], "gif")
    assert gif is not None and gif.startswith(b"GIF8")
    webm = gallery.transcode(record["id"], "webm")
    # EBML magic: WebM is a Matroska container.
    assert webm is not None and webm[:4] == b"\x1a\x45\xdf\xa3"


def test_transcode_unknown_id_and_bad_format():
    assert gallery.transcode("does-not-exist", "gif") is None
    record = gallery.save(_real_mp4_bytes(), _meta())
    with pytest.raises(ValueError):
        gallery.transcode(record["id"], "avi")
