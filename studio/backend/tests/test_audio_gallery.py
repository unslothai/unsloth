# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unit tests for the disk-backed audio gallery: WAV + JSON-sidecar round-trips,
listing order, safe id handling, orphan-pair skipping, and delete/clear."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import core.inference.audio_gallery as gallery

import pytest


@pytest.fixture(autouse = True)
def _tmp_gallery(monkeypatch, tmp_path):
    # Point the gallery at a throwaway root instead of ~/.unsloth/studio.
    monkeypatch.setattr(gallery, "studio_root", lambda: tmp_path)


def _wav(tag = b"RIFF\x24\x00\x00\x00WAVEfmt "):
    # Not a real container; the gallery treats the bytes as opaque payload.
    return tag


def _meta(**over):
    base = {
        "prompt": "hello from a sloth",
        "model": "unsloth/orpheus-3b-0.1-ft",
        "audio_type": "snac",
        "sample_rate": 24000,
        "duration_s": 1.5,
        "created_at": "2026-08-06T00:00:00Z",
    }
    base.update(over)
    return base


def test_save_writes_pair_and_round_trips():
    record = gallery.save(_wav(), _meta())
    assert record["id"] and record["url"].endswith(f"{record['id']}/file")

    # Both files of the pair exist: the wav payload and the json recipe sidecar.
    directory = gallery.gallery_dir()
    assert (directory / f"{record['id']}.wav").is_file()
    sidecar = directory / f"{record['id']}.json"
    assert json.loads(sidecar.read_text(encoding = "utf-8"))["prompt"] == "hello from a sloth"

    listed = gallery.list_audio()
    assert len(listed) == 1
    assert listed[0]["prompt"] == "hello from a sloth"
    # Meta fields survive the sidecar round-trip untouched.
    assert listed[0]["sample_rate"] == 24000 and listed[0]["audio_type"] == "snac"


def test_url_shape():
    record = gallery.save(_wav(), _meta())
    assert record["url"] == f"/api/inference/audio/gallery/{record['id']}/file"


def _save_with_mtime(prompt: str, t: float) -> dict:
    record = gallery.save(_wav(), _meta(prompt = prompt))
    # Listing orders by wav mtime; set it explicitly so a tight test loop can't tie it.
    os.utime(gallery.gallery_dir() / f"{record['id']}.wav", (t, t))
    return record


def test_list_is_newest_first():
    old = _save_with_mtime("old", 100.0)
    new = _save_with_mtime("new", 200.0)
    assert [r["id"] for r in gallery.list_audio()] == [new["id"], old["id"]]


def test_list_paginates_with_limit_offset():
    for i in range(5):
        _save_with_mtime(f"p{i}", float(i))
    page1 = gallery.list_audio(limit = 2, offset = 0)
    page2 = gallery.list_audio(limit = 2, offset = 2)
    assert [r["prompt"] for r in page1] == ["p4", "p3"]
    assert [r["prompt"] for r in page2] == ["p2", "p1"]
    # limit=None still returns everything from the offset.
    assert len(gallery.list_audio()) == 5
    assert len(gallery.list_audio(offset = 4)) == 1


def test_cursor_pagination_does_not_skip_after_earlier_clip_is_deleted():
    records = [_save_with_mtime(prompt, float(i)) for i, prompt in enumerate("DCBA", 1)]
    page1 = gallery.list_audio_page(limit = 3)
    visible1 = page1[:2]
    assert [record["prompt"] for record, _ in visible1] == ["A", "B"]

    # Removing A shifts every offset, but the exclusive B cursor still starts at C.
    assert gallery.delete(records[-1]["id"]) is True
    page2 = gallery.list_audio(limit = 2, before = visible1[-1][1])
    assert [record["prompt"] for record in page2] == ["C", "D"]


def test_audio_path_rejects_unsafe_ids():
    # Traversal / bad chars / absolute paths never resolve to a path.
    assert gallery.audio_path("../../etc/passwd") is None
    assert gallery.audio_path("/etc/passwd") is None
    assert gallery.audio_path("a/b") is None
    assert gallery.audio_path("missing") is None


def test_audio_path_returns_wav_for_saved_id():
    record = gallery.save(_wav(), _meta())
    path = gallery.audio_path(record["id"])
    assert path is not None and path.name == f"{record['id']}.wav"


def test_owned_audio_path_serves_only_owned_clips():
    # A hand-dropped orphan WAV resolves via audio_path (safe stem, on disk) but must NOT be
    # served: owned_audio_path applies the same sidecar check as delete/clear.
    orphan = gallery.gallery_dir() / "recording.wav"
    orphan.write_bytes(_wav())
    assert gallery.audio_path("recording") is not None  # resolvable...
    assert gallery.owned_audio_path("recording") is None  # ...but not ours to serve

    ours = gallery.save(_wav(), _meta(prompt = "ours"))
    assert gallery.owned_audio_path(ours["id"]) is not None
    assert gallery.owned_audio_path("../../etc/passwd") is None
    assert gallery.owned_audio_path("missing") is None


def test_gallery_file_route_streams_the_owned_wav(monkeypatch):
    from fastapi.responses import FileResponse
    from routes.inference import get_gallery_audio_file

    record = gallery.save(_wav(), _meta())
    monkeypatch.setattr(
        Path,
        "read_bytes",
        lambda self: pytest.fail("the route must not buffer the WAV before responding"),
    )
    response = asyncio.run(
        get_gallery_audio_file(record["id"], current_subject = "tester")
    )

    assert isinstance(response, FileResponse)
    assert Path(response.path) == gallery.gallery_dir() / f"{record['id']}.wav"
    assert response.media_type == "audio/wav"
    assert response.headers["cache-control"] == "private, max-age=31536000, immutable"


def test_delete_removes_both_files():
    record = gallery.save(_wav(), _meta(prompt = "a"))
    gallery.save(_wav(), _meta(prompt = "b"))
    directory = gallery.gallery_dir()
    assert gallery.delete(record["id"]) is True
    # Both halves of the pair are gone.
    assert not (directory / f"{record['id']}.wav").exists()
    assert not (directory / f"{record['id']}.json").exists()
    assert gallery.delete(record["id"]) is False  # already gone
    assert len(gallery.list_audio()) == 1


def test_delete_keeps_sidecar_listable_when_wav_unlink_fails(monkeypatch):
    # delete() must remove the WAV FIRST: list_audio globs *.wav but needs a readable sidecar,
    # so dropping the sidecar first and then failing the wav unlink would hide a still-present
    # wav with no way to retry.
    record = gallery.save(_wav(), _meta(prompt = "keep"))
    directory = gallery.gallery_dir()
    wav = directory / f"{record['id']}.wav"
    sidecar = directory / f"{record['id']}.json"

    real_unlink = Path.unlink

    def _fail_on_wav(self, *a, **k):
        if self.suffix == ".wav":
            raise PermissionError("wav locked")
        return real_unlink(self, *a, **k)

    # Scoped so undoing it does not revert the autouse fixture's studio_root redirect.
    with pytest.MonkeyPatch.context() as m:
        m.setattr(Path, "unlink", _fail_on_wav)
        assert gallery.delete(record["id"]) is False  # wav unlink failed
    # The sidecar was NOT dropped, so the record is still listable and the user can retry.
    assert sidecar.exists() and wav.exists()
    assert [r["prompt"] for r in gallery.list_audio()] == ["keep"]
    assert gallery.delete(record["id"]) is True  # retry now succeeds


def test_clear_returns_count():
    gallery.save(_wav(), _meta(prompt = "a"))
    gallery.save(_wav(), _meta(prompt = "b"))
    assert gallery.clear() == 2
    assert gallery.list_audio() == []
    # No stray sidecars left behind after a clear.
    assert list(gallery.gallery_dir().glob("*.json")) == []


def test_clear_preserves_orphan_wav():
    # An orphan / foreign WAV is invisible to list_audio; clear must remove the owned pair without destroying it.
    foreign = gallery.gallery_dir() / "recording.wav"
    foreign.write_bytes(_wav())
    gallery.save(_wav(), _meta(prompt = "ours"))
    assert gallery.clear() == 1
    assert foreign.exists()
    assert gallery.list_audio() == []


def test_delete_ignores_orphan_wav():
    # A per-id delete must refuse a WAV we do not own (no readable sidecar).
    foreign = gallery.gallery_dir() / "recording.wav"
    foreign.write_bytes(_wav())
    assert gallery.delete("recording") is False
    assert foreign.exists()


def test_list_skips_orphan_wav_without_sidecar():
    orphan = gallery.gallery_dir() / "orphan.wav"
    orphan.write_bytes(_wav())
    gallery.save(_wav(), _meta(prompt = "ours"))
    assert [r["prompt"] for r in gallery.list_audio()] == ["ours"]


def test_list_skips_orphan_sidecar_without_wav():
    orphan = gallery.gallery_dir() / "lonely.json"
    orphan.write_text(json.dumps(_meta(prompt = "no audio")), encoding = "utf-8")
    gallery.save(_wav(), _meta(prompt = "ours"))
    assert [r["prompt"] for r in gallery.list_audio()] == ["ours"]


def test_orphan_wav_in_window_does_not_drop_valid_clips():
    # An orphan WAV sorting INTO the requested page must not consume a window slot: paging is over readable records.
    _save_with_mtime("p2", 100.0)
    orphan = gallery.gallery_dir() / "zzz_orphan.wav"
    orphan.write_bytes(_wav())
    os.utime(orphan, (300.0, 300.0))
    _save_with_mtime("p1", 200.0)
    page1 = gallery.list_audio(limit = 2, offset = 0)
    assert [r["prompt"] for r in page1] == ["p1", "p2"]


def test_list_skips_corrupt_sidecar():
    directory = gallery.gallery_dir()
    (directory / "broken.wav").write_bytes(_wav())
    (directory / "broken.json").write_text("{not json", encoding = "utf-8")
    gallery.save(_wav(), _meta(prompt = "ours"))
    assert [r["prompt"] for r in gallery.list_audio()] == ["ours"]


def test_list_skips_invalid_utf8_sidecar():
    # Invalid UTF-8 raises UnicodeDecodeError, not an OSError: one corrupt sidecar is skipped, it does not 500 the listing.
    directory = gallery.gallery_dir()
    (directory / "badbytes.wav").write_bytes(_wav())
    (directory / "badbytes.json").write_bytes(b"\xff\xfe{}")
    gallery.save(_wav(), _meta(prompt = "ours"))
    assert [r["prompt"] for r in gallery.list_audio()] == ["ours"]


def test_clear_preserves_wav_with_present_but_invalid_sidecar():
    # A hand-dropped WAV whose sidecar parses but lacks the required recipe keys is hidden by list_audio, so clear must spare it.
    directory = gallery.gallery_dir()
    (directory / "foreign.wav").write_bytes(_wav())
    (directory / "foreign.json").write_text("{}", encoding = "utf-8")
    gallery.save(_wav(), _meta(prompt = "ours"))
    assert gallery.clear() == 1
    assert (directory / "foreign.wav").exists()


def test_delete_refuses_wav_with_present_but_invalid_sidecar():
    # The gallery never surfaced a record missing required keys, so a guessed id must not destroy it.
    directory = gallery.gallery_dir()
    (directory / "foreign.wav").write_bytes(_wav())
    (directory / "foreign.json").write_text(
        json.dumps({"prompt": "x"}), encoding = "utf-8"
    )  # partial sidecar (no model/sample_rate/...)
    assert gallery.delete("foreign") is False
    assert (directory / "foreign.wav").exists()


def test_valid_callback_paginates_over_accepted_records():
    # ``valid`` must filter before pagination, else a leading bad record returns a short page and stalls scroll.
    _save_with_mtime("BAD", 300.0)  # newest, sorts first
    _save_with_mtime("g1", 200.0)
    _save_with_mtime("g2", 100.0)

    def _valid(rec):
        return rec.get("prompt") != "BAD"

    page = gallery.list_audio(limit = 2, offset = 0, valid = _valid)
    assert [r["prompt"] for r in page] == ["g1", "g2"]
    assert len(gallery.list_audio(limit = 3, offset = 0, valid = _valid)) == 2


def test_save_leaves_no_orphan_wav_when_sidecar_publish_fails(monkeypatch):
    # If the sidecar (the pair's commit marker) fails to publish, the WAV must not be stranded as an invisible orphan.
    real_replace = gallery.os.replace
    calls = {"n": 0}

    def _replace(src, dst, *a, **k):
        calls["n"] += 1
        if calls["n"] == 2:  # the sidecar publish
            raise OSError("simulated sidecar failure")
        return real_replace(src, dst, *a, **k)

    monkeypatch.setattr(gallery.os, "replace", _replace)
    with pytest.raises(OSError, match = "simulated sidecar failure"):
        gallery.save(_wav(), _meta())
    # No wav, no sidecar, no temp files: the whole record was rolled back.
    assert list(gallery.gallery_dir().iterdir()) == []
    assert gallery.list_audio() == []
