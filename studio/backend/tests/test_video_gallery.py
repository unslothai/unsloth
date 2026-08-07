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
        "fps": 24,
        "duration_s": 2.0,
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


def test_owned_video_path_serves_only_owned_clips():
    # A hand-dropped orphan MP4 resolves via video_path (safe stem, on disk) but must NOT be served: owned_video_path
    # applies the same sidecar check as delete/clear, so serve and export cannot stream a clip the listing hides.
    orphan = gallery.gallery_dir() / "recording.mp4"
    orphan.write_bytes(_mp4())
    assert gallery.video_path("recording") is not None  # resolvable...
    assert gallery.owned_video_path("recording") is None  # ...but not ours to serve

    ours = gallery.save(_mp4(), _meta(prompt = "ours"))
    assert gallery.owned_video_path(ours["id"]) is not None
    assert gallery.owned_video_path("../../etc/passwd") is None
    assert gallery.owned_video_path("missing") is None


def test_transcode_refuses_orphan_mp4():
    # Export shares the /file resolver, so a guessed stem for an orphan MP4 must not be re-encoded out either.
    orphan = gallery.gallery_dir() / "recording.mp4"
    orphan.write_bytes(_real_mp4_bytes())
    assert gallery.transcode("recording", "gif") is None
    assert gallery.transcode("recording", "webm") is None


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
    # delete() must remove the MP4 FIRST: list_videos globs *.mp4 but needs a readable sidecar, so dropping the sidecar first and then failing
    # the mp4 unlink (a Windows lock) would hide a still-present mp4 with no way to retry. Fail the mp4 unlink and assert the video stays listable.
    record = gallery.save(_mp4(), _meta(prompt = "keep"))
    directory = gallery.gallery_dir()
    mp4 = directory / f"{record['id']}.mp4"
    sidecar = directory / f"{record['id']}.json"

    real_unlink = Path.unlink

    def _fail_on_mp4(self, *a, **k):
        if self.suffix == ".mp4":
            raise PermissionError("mp4 locked")
        return real_unlink(self, *a, **k)

    # Patch Path.unlink (what delete() calls) not os.unlink: on 3.10 Path.unlink goes through a cached _accessor bound at import. Scoped to its own
    # context so undoing it does not revert the autouse fixture's studio_root redirect (shared monkeypatch), which would make list_videos read the real home.
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


def test_clear_preserves_orphan_mp4():
    # An orphan / foreign MP4 is invisible to list_videos; clear must remove the owned pair without destroying it.
    foreign = gallery.gallery_dir() / "recording.mp4"
    foreign.write_bytes(_mp4())
    gallery.save(_mp4(), _meta(prompt = "ours"))
    assert gallery.clear() == 1
    assert foreign.exists()
    assert gallery.list_videos() == []


def test_delete_ignores_orphan_mp4():
    # A per-id delete must refuse an MP4 we do not own (no readable sidecar).
    foreign = gallery.gallery_dir() / "recording.mp4"
    foreign.write_bytes(_mp4())
    assert gallery.delete("recording") is False
    assert foreign.exists()


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
    # An orphan MP4 sorting INTO the requested page must not consume a window slot: paging is over readable records.
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


def test_list_skips_invalid_utf8_sidecar():
    # Invalid UTF-8 raises UnicodeDecodeError, not an OSError: one corrupt sidecar is skipped, it does not 500 the listing.
    directory = gallery.gallery_dir()
    (directory / "badbytes.mp4").write_bytes(_mp4())
    (directory / "badbytes.json").write_bytes(b"\xff\xfe{}")
    gallery.save(_mp4(), _meta(prompt = "ours"))
    assert [r["prompt"] for r in gallery.list_videos()] == ["ours"]


def test_clear_preserves_mp4_with_present_but_invalid_sidecar():
    # A hand-dropped MP4 whose sidecar parses but lacks the required recipe keys is hidden by list_videos, so clear must spare it.
    directory = gallery.gallery_dir()
    (directory / "foreign.mp4").write_bytes(_mp4())
    (directory / "foreign.json").write_text("{}", encoding = "utf-8")
    gallery.save(_mp4(), _meta(prompt = "ours"))
    assert gallery.clear() == 1
    assert (directory / "foreign.mp4").exists()


def test_delete_refuses_mp4_with_present_but_invalid_sidecar():
    # The gallery never surfaced a record missing required keys, so a guessed id must not destroy it.
    directory = gallery.gallery_dir()
    (directory / "foreign.mp4").write_bytes(_mp4())
    (directory / "foreign.json").write_text(
        json.dumps({"prompt": "x"}), encoding = "utf-8"
    )  # partial sidecar (no width/seed/...)
    assert gallery.delete("foreign") is False
    assert (directory / "foreign.mp4").exists()


def test_valid_callback_paginates_over_accepted_records():
    # ``valid`` must filter before pagination, else a leading bad record returns a short page with more remaining and stalls scroll.
    _save_with_mtime("BAD", 300.0)  # newest, sorts first
    _save_with_mtime("g1", 200.0)
    _save_with_mtime("g2", 100.0)

    def _valid(rec):
        return rec.get("prompt") != "BAD"

    page = gallery.list_videos(limit = 2, offset = 0, valid = _valid)
    assert [r["prompt"] for r in page] == ["g1", "g2"]
    assert len(gallery.list_videos(limit = 3, offset = 0, valid = _valid)) == 2


def test_valid_callback_leading_bad_records_do_not_stall_at_offset_zero():
    # Every record in the first window is schema-invalid: the pager must look past them so has_more is False and the client advances.
    for i in range(3):
        _save_with_mtime(f"BAD{i}", 300.0 - i)
    _save_with_mtime("good", 10.0)

    def _valid(rec):
        return not str(rec.get("prompt", "")).startswith("BAD")

    records = gallery.list_videos(limit = 2, offset = 0, valid = _valid)
    assert [r["prompt"] for r in records] == ["good"]


def test_save_leaves_no_orphan_mp4_when_sidecar_publish_fails(monkeypatch):
    # If the sidecar (the pair's commit marker) fails to publish, the MP4 must not be stranded as an invisible orphan.
    real_replace = gallery.os.replace
    calls = {"n": 0}

    def _replace(src, dst, *a, **k):
        calls["n"] += 1
        if calls["n"] == 2:  # the sidecar publish
            raise OSError("simulated sidecar failure")
        return real_replace(src, dst, *a, **k)

    monkeypatch.setattr(gallery.os, "replace", _replace)
    with pytest.raises(OSError, match = "simulated sidecar failure"):
        gallery.save(_mp4(), _meta())
    # No mp4, no sidecar, no temp files -- the whole record was rolled back.
    assert list(gallery.gallery_dir().iterdir()) == []
    assert gallery.list_videos() == []


def _real_mp4_bytes(
    frames: int = 8,
    size: int = 32,
    rate: int = 8,
) -> bytes:
    # A real tiny MP4 for the transcode tests: flat-color frames in mpeg4 (bundled in every PyAV build, unlike libx264).
    av = pytest.importorskip("av")
    np = pytest.importorskip("numpy")
    import io

    buf = io.BytesIO()
    with av.open(buf, "w", format = "mp4") as out:
        stream = out.add_stream("mpeg4", rate = rate)
        stream.width = size
        stream.height = size
        stream.pix_fmt = "yuv420p"
        for i in range(frames):
            frame = av.VideoFrame.from_ndarray(
                np.full((size, size, 3), (i * 30) % 256, dtype = np.uint8), format = "rgb24"
            )
            for packet in stream.encode(frame):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)
    return buf.getvalue()


def _real_mp4_with_audio(
    seconds: int = 1,
    size: int = 32,
    rate: int = 8,
) -> bytes:
    # An LTX-2-shaped clip: video plus a synchronized 440 Hz audio track, so the WebM export can be checked for the track.
    av = pytest.importorskip("av")
    np = pytest.importorskip("numpy")
    import io
    import math
    from fractions import Fraction

    arate = 44100
    buf = io.BytesIO()
    with av.open(buf, "w", format = "mp4") as out:
        video = out.add_stream("mpeg4", rate = rate)
        video.width = video.height = size
        video.pix_fmt = "yuv420p"
        audio = out.add_stream("aac", rate = arate)
        audio.layout = "stereo"
        for i in range(seconds * rate):
            frame = av.VideoFrame.from_ndarray(
                np.full((size, size, 3), (i * 30) % 256, dtype = np.uint8), format = "rgb24"
            )
            for packet in video.encode(frame):
                out.mux(packet)
        written = 0
        while written < seconds * arate:
            count = min(1024, seconds * arate - written)
            tone = np.array(
                [
                    int(20000 * math.sin(2 * math.pi * 440 * (written + k) / arate))
                    for k in range(count)
                ],
                dtype = np.int16,
            )
            # Packed s16 is one interleaved row.
            frame = av.AudioFrame.from_ndarray(
                np.repeat(tone, 2).reshape(1, count * 2), format = "s16", layout = "stereo"
            )
            frame.sample_rate = arate
            frame.pts = written
            frame.time_base = Fraction(1, arate)
            for packet in audio.encode(frame):
                out.mux(packet)
            written += count
        for packet in video.encode():
            out.mux(packet)
        for packet in audio.encode():
            out.mux(packet)
    return buf.getvalue()


def test_webm_export_keeps_the_audio_track():
    """WebM is offered as the web-embed format, so a clip generated with synchronized audio (LTX-2)
    must not come back mute: the exporter has to mux the track as Opus, WebM's audio codec."""
    av = pytest.importorskip("av")
    import io

    record = gallery.save(_real_mp4_with_audio(), _meta())
    webm = gallery.transcode(record["id"], "webm")
    assert webm is not None and webm[:4] == b"\x1a\x45\xdf\xa3"
    with av.open(io.BytesIO(webm)) as container:
        kinds = {(s.type, s.codec_context.name) for s in container.streams}
    assert ("video", "vp9") in kinds
    assert ("audio", "opus") in kinds
    samples = 0
    with av.open(io.BytesIO(webm)) as container:
        for frame in container.decode(audio = 0):
            samples += frame.samples
    # A full second of 48 kHz audio survived (Opus pads its last 20 ms frame).
    assert samples >= 48000, samples


def test_webm_export_still_works_without_an_audio_encoder(monkeypatch):
    # A PyAV build with no libopus must keep exporting the video rather than failing the download.
    av = pytest.importorskip("av")
    import io

    # The refusal is injected by wrapping the container av.open() returns, NOT by patching OutputContainer.add_stream: that
    # is a C extension type and PyAV 17 (the 3.10 CI leg) raises "cannot set 'add_stream' attribute of immutable type".
    real_open = av.open

    class _NoOpusContainer:
        """Delegates to the real output container, but refuses the Opus encoder."""

        def __init__(self, inner):
            self._inner = inner

        def add_stream(
            self,
            codec_name = None,
            *args,
            **kwargs,
        ):
            if codec_name == "libopus":
                raise ValueError("unknown encoder 'libopus'")
            return self._inner.add_stream(codec_name, *args, **kwargs)

        def __getattr__(self, name):
            return getattr(self._inner, name)

        def __enter__(self):
            self._inner.__enter__()
            return self

        def __exit__(self, *exc):
            return self._inner.__exit__(*exc)

    def _open(
        file,
        mode = "r",
        *args,
        **kwargs,
    ):
        inner = real_open(file, mode, *args, **kwargs)
        return _NoOpusContainer(inner) if mode == "w" else inner

    monkeypatch.setattr(av, "open", _open)
    record = gallery.save(_real_mp4_with_audio(), _meta())
    webm = gallery.transcode(record["id"], "webm")
    assert webm is not None and webm[:4] == b"\x1a\x45\xdf\xa3"
    with av.open(io.BytesIO(webm)) as container:
        assert [s.type for s in container.streams] == ["video"]


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


def test_transcode_to_file_writes_a_temp_file_the_caller_owns():
    # The route streams the export from disk instead of materialising it: the caps allow 2048x2048 x 1024 frames, and holding
    # a VP9 export of that size as bytes (then again in the response) let concurrent exports exhaust the process.
    record = gallery.save(_real_mp4_bytes(), _meta())
    for fmt, magic in (("webm", b"\x1a\x45\xdf\xa3"), ("gif", b"GIF8")):
        path = gallery.transcode_to_file(record["id"], fmt)
        assert path is not None and path.is_file(), fmt
        assert path.suffix == f".{fmt}"
        assert path.read_bytes()[: len(magic)] == magic
        # It is a temp file, NOT something inside the gallery: deleting it must not touch the clip.
        path.unlink()
        assert gallery.video_path(record["id"]) is not None
    assert gallery.transcode_to_file("does-not-exist", "webm") is None


def test_transcode_to_file_leaves_no_temp_file_when_the_encode_fails(monkeypatch):
    # A half-written export must not accumulate in the temp dir on a host with no VP9 encoder.
    import tempfile

    from core.inference import video_gallery as vg

    record = gallery.save(_real_mp4_bytes(), _meta())

    def _boom(src, dest):
        dest.write_bytes(b"partial")
        raise RuntimeError("WebM export failed (libvpx-vp9 unavailable?)")

    monkeypatch.setattr(vg, "_transcode_webm", _boom)
    before = set(Path(tempfile.gettempdir()).glob("unsloth-export-*"))
    with pytest.raises(RuntimeError):
        vg.transcode_to_file(record["id"], "webm")
    assert set(Path(tempfile.gettempdir()).glob("unsloth-export-*")) == before


def test_gif_export_bounds_frames_and_edge(monkeypatch):
    """Every kept frame is held as a paletted image before the encoder runs, so an unbounded walk
    is a memory bomb: the generate request allows 2048x2048 for 1024 frames, and at the 12 fps
    target the step is 1, which is over 4 GB of frames plus the GIF buffer. Cap both axes."""
    import io as _io

    from core.inference import video_gallery as vg

    Image = pytest.importorskip("PIL.Image")
    monkeypatch.setattr(vg, "_GIF_MAX_EDGE", 16)
    monkeypatch.setattr(vg, "_GIF_MAX_FRAMES", 4)

    record = gallery.save(_real_mp4_bytes(frames = 24, size = 64, rate = 12), _meta())
    gif = vg._transcode_gif(vg.gallery_dir() / f"{record['id']}.mp4")

    assert gif.startswith(b"GIF8")
    with Image.open(_io.BytesIO(gif)) as im:
        assert max(im.size) <= 16, im.size
        frames = 1
        try:
            while True:
                im.seek(im.tell() + 1)
                frames += 1
        except EOFError:
            pass
    assert frames <= 4, frames
