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
    monkeypatch.setattr(gallery, "studio_root", lambda: tmp_path)


def _wav(tag = b"RIFF\x24\x00\x00\x00WAVEfmt "):
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

    directory = gallery.gallery_dir()
    assert (directory / f"{record['id']}.wav").is_file()
    sidecar = directory / f"{record['id']}.json"
    assert json.loads(sidecar.read_text(encoding = "utf-8"))["prompt"] == "hello from a sloth"

    listed = gallery.list_audio()
    assert len(listed) == 1
    assert listed[0]["prompt"] == "hello from a sloth"
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
    assert len(gallery.list_audio()) == 5
    assert len(gallery.list_audio(offset = 4)) == 1


def test_cursor_pagination_does_not_skip_after_earlier_clip_is_deleted():
    records = [_save_with_mtime(prompt, float(i)) for i, prompt in enumerate("DCBA", 1)]
    page1 = gallery.list_audio_page(limit = 3)
    visible1 = page1[:2]
    assert [record["prompt"] for record, _ in visible1] == ["A", "B"]

    assert gallery.delete(records[-1]["id"]) is True
    page2 = gallery.list_audio(limit = 2, before = visible1[-1][1])
    assert [record["prompt"] for record in page2] == ["C", "D"]


def test_audio_path_rejects_unsafe_ids():
    assert gallery.audio_path("../../etc/passwd") is None
    assert gallery.audio_path("/etc/passwd") is None
    assert gallery.audio_path("a/b") is None
    assert gallery.audio_path("missing") is None


def test_audio_path_returns_wav_for_saved_id():
    record = gallery.save(_wav(), _meta())
    path = gallery.audio_path(record["id"])
    assert path is not None and path.name == f"{record['id']}.wav"


def test_owned_audio_path_serves_only_owned_clips():
    orphan = gallery.gallery_dir() / "recording.wav"
    orphan.write_bytes(_wav())
    assert gallery.audio_path("recording") is not None
    assert gallery.owned_audio_path("recording") is None

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
    response = asyncio.run(get_gallery_audio_file(record["id"], current_subject = "tester"))

    assert isinstance(response, FileResponse)
    assert Path(response.path) == gallery.gallery_dir() / f"{record['id']}.wav"
    assert response.media_type == "audio/wav"
    assert response.headers["cache-control"] == "private, max-age=31536000, immutable"


def test_delete_removes_both_files():
    record = gallery.save(_wav(), _meta(prompt = "a"))
    gallery.save(_wav(), _meta(prompt = "b"))
    directory = gallery.gallery_dir()
    assert gallery.delete(record["id"]) is True
    assert not (directory / f"{record['id']}.wav").exists()
    assert not (directory / f"{record['id']}.json").exists()
    assert gallery.delete(record["id"]) is False
    assert len(gallery.list_audio()) == 1


def test_delete_keeps_sidecar_listable_when_wav_unlink_fails(monkeypatch):
    # Remove the WAV first so a failed unlink does not hide it from list_audio.
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
        assert gallery.delete(record["id"]) is False
    assert sidecar.exists() and wav.exists()
    assert [r["prompt"] for r in gallery.list_audio()] == ["keep"]
    assert gallery.delete(record["id"]) is True


def test_clear_returns_count():
    gallery.save(_wav(), _meta(prompt = "a"))
    gallery.save(_wav(), _meta(prompt = "b"))
    assert gallery.clear() == 2
    assert gallery.list_audio() == []
    assert list(gallery.gallery_dir().glob("*.json")) == []


def test_clear_preserves_orphan_wav():
    foreign = gallery.gallery_dir() / "recording.wav"
    foreign.write_bytes(_wav())
    gallery.save(_wav(), _meta(prompt = "ours"))
    assert gallery.clear() == 1
    assert foreign.exists()
    assert gallery.list_audio() == []


def test_delete_ignores_orphan_wav():
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
    directory = gallery.gallery_dir()
    (directory / "badbytes.wav").write_bytes(_wav())
    (directory / "badbytes.json").write_bytes(b"\xff\xfe{}")
    gallery.save(_wav(), _meta(prompt = "ours"))
    assert [r["prompt"] for r in gallery.list_audio()] == ["ours"]


def test_clear_preserves_wav_with_present_but_invalid_sidecar():
    directory = gallery.gallery_dir()
    (directory / "foreign.wav").write_bytes(_wav())
    (directory / "foreign.json").write_text("{}", encoding = "utf-8")
    gallery.save(_wav(), _meta(prompt = "ours"))
    assert gallery.clear() == 1
    assert (directory / "foreign.wav").exists()


def test_delete_refuses_wav_with_present_but_invalid_sidecar():
    directory = gallery.gallery_dir()
    (directory / "foreign.wav").write_bytes(_wav())
    (directory / "foreign.json").write_text(json.dumps({"prompt": "x"}), encoding = "utf-8")
    assert gallery.delete("foreign") is False
    assert (directory / "foreign.wav").exists()


def test_valid_callback_paginates_over_accepted_records():
    # ``valid`` must filter before pagination, else a leading bad record returns a short page and stalls scroll.
    _save_with_mtime("BAD", 300.0)
    _save_with_mtime("g1", 200.0)
    _save_with_mtime("g2", 100.0)

    def _valid(rec):
        return rec.get("prompt") != "BAD"

    page = gallery.list_audio(limit = 2, offset = 0, valid = _valid)
    assert [r["prompt"] for r in page] == ["g1", "g2"]
    assert len(gallery.list_audio(limit = 3, offset = 0, valid = _valid)) == 2


def test_save_leaves_no_orphan_wav_when_sidecar_publish_fails(monkeypatch):
    # Sidecar is the commit marker; a failed publish must roll back the WAV.
    real_replace = gallery.os.replace
    calls = {"n": 0}

    def _replace(src, dst, *a, **k):
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError("simulated sidecar failure")
        return real_replace(src, dst, *a, **k)

    monkeypatch.setattr(gallery.os, "replace", _replace)
    with pytest.raises(OSError, match = "simulated sidecar failure"):
        gallery.save(_wav(), _meta())
    assert list(gallery.gallery_dir().iterdir()) == []
    assert gallery.list_audio() == []


def test_a_nonnumeric_cap_disables_pruning(monkeypatch):
    """The documented contract: "off" means off. Restoring the default for a value the
    operator did set would delete recordings they had asked to keep."""
    from core.inference import audio_gallery

    monkeypatch.setenv("UNSLOTH_AUDIO_GALLERY_MAX_CLIPS", "off")
    assert audio_gallery._max_clips() == 0
    monkeypatch.setenv("UNSLOTH_AUDIO_GALLERY_MAX_CLIPS", "0")
    assert audio_gallery._max_clips() == 0
    monkeypatch.setenv("UNSLOTH_AUDIO_GALLERY_MAX_CLIPS", "5")
    assert audio_gallery._max_clips() == 5
    monkeypatch.delenv("UNSLOTH_AUDIO_GALLERY_MAX_CLIPS")
    assert audio_gallery._max_clips() == audio_gallery._DEFAULT_MAX_CLIPS


def test_records_carry_default_archived_flag():
    _save_with_mtime("a", 100.0)
    assert gallery.list_audio()[0]["archived"] is False


def test_archived_clips_leave_the_default_listing():
    keep = _save_with_mtime("keep", 100.0)
    shelved = _save_with_mtime("shelved", 200.0)
    assert gallery.set_flags(shelved["id"], archived = True)["archived"] is True
    assert [r["id"] for r in gallery.list_audio()] == [keep["id"]]
    archived = gallery.list_audio(archived = True)
    assert [r["id"] for r in archived] == [shelved["id"]]
    assert archived[0]["archived"] is True


def test_restoring_puts_a_clip_back_in_history():
    record = _save_with_mtime("a", 100.0)
    gallery.set_flags(record["id"], archived = True)
    gallery.set_flags(record["id"], archived = False)
    assert [r["id"] for r in gallery.list_audio()] == [record["id"]]
    assert gallery.list_audio(archived = True) == []


def test_archived_clips_do_not_consume_a_page_slot():
    for i in range(4):
        record = _save_with_mtime(f"a{i}", 100.0 + i)
        if i % 2 == 0:
            gallery.set_flags(record["id"], archived = True)
    assert [r["prompt"] for r in gallery.list_audio(limit = 2)] == ["a3", "a1"]
    assert [r["prompt"] for r in gallery.list_audio(archived = True)] == ["a2", "a0"]


def test_archived_shelf_paginates_by_cursor():
    records = [_save_with_mtime(prompt, float(i)) for i, prompt in enumerate("DCBA", 1)]
    for record in records:
        gallery.set_flags(record["id"], archived = True)
    page1 = gallery.list_audio_page(limit = 2, archived = True)
    assert [record["prompt"] for record, _ in page1] == ["A", "B"]
    page2 = gallery.list_audio(limit = 2, before = page1[-1][1], archived = True)
    assert [record["prompt"] for record in page2] == ["C", "D"]


def test_set_flags_refuses_unowned_ids():
    (gallery.gallery_dir() / "foreign.wav").write_bytes(_wav())
    assert gallery.set_flags("foreign", archived = True) is None
    assert gallery.set_flags("../../etc/passwd", archived = True) is None
    assert gallery.set_flags("missing", archived = True) is None


def test_delete_prunes_the_flag_entry():
    from core.inference import gallery_flags

    record = _save_with_mtime("a", 100.0)
    gallery.set_flags(record["id"], archived = True)
    assert gallery.delete(record["id"]) is True
    assert gallery_flags.read(gallery.gallery_dir()) == {}


def test_clear_spares_archived_clips():
    active = _save_with_mtime("active", 100.0)
    shelved = _save_with_mtime("shelved", 200.0)
    gallery.set_flags(shelved["id"], archived = True)
    assert gallery.clear() == 1
    assert [r["id"] for r in gallery.list_audio(archived = True)] == [shelved["id"]]
    assert gallery.audio_path(active["id"]) is None


def test_clear_can_include_archived_clips():
    from core.inference import gallery_flags

    record = _save_with_mtime("shelved", 100.0)
    gallery.set_flags(record["id"], archived = True)
    assert gallery.clear(include_archived = True) == 1
    assert gallery.list_audio(archived = True) == []
    assert gallery_flags.read(gallery.gallery_dir()) == {}


def test_clear_refuses_when_the_flag_store_cannot_be_read():
    from core.inference import gallery_flags

    record = _save_with_mtime("shelved", 100.0)
    gallery.set_flags(record["id"], archived = True)
    (gallery.gallery_dir() / ".flags.json").write_text("corrupt", encoding = "utf-8")
    with pytest.raises(gallery_flags.FlagsUnavailable):
        gallery.clear()
    assert gallery.audio_path(record["id"]) is not None


def test_clear_all_replaces_an_unreadable_store():
    _save_with_mtime("a", 100.0)
    (gallery.gallery_dir() / ".flags.json").write_text("corrupt", encoding = "utf-8")
    assert gallery.clear(include_archived = True) == 1
    _save_with_mtime("b", 200.0)
    assert gallery.clear() == 1


def test_prune_spares_archived_clips(monkeypatch):
    monkeypatch.setenv("UNSLOTH_AUDIO_GALLERY_MAX_CLIPS", "2")
    shelved = _save_with_mtime("shelved", 100.0)
    gallery.set_flags(shelved["id"], archived = True)
    _save_with_mtime("b", 200.0)
    _save_with_mtime("c", 300.0)
    newest = gallery.save(_wav(), _meta(prompt = "d"))
    assert gallery.audio_path(shelved["id"]) is not None
    assert [r["prompt"] for r in gallery.list_audio()] == ["d", "c"]
    assert gallery.audio_path(newest["id"]) is not None


def test_flags_route_archives_and_restores():
    from fastapi import HTTPException
    from models.inference import AudioGalleryFlagsPatch
    from routes.inference import update_gallery_audio_flags

    record = gallery.save(_wav(), _meta())
    archived = asyncio.run(
        update_gallery_audio_flags(
            record["id"], AudioGalleryFlagsPatch(archived = True), current_subject = "tester"
        )
    )
    assert archived.archived is True
    assert gallery.list_audio() == []
    restored = asyncio.run(
        update_gallery_audio_flags(
            record["id"], AudioGalleryFlagsPatch(archived = False), current_subject = "tester"
        )
    )
    assert restored.archived is False
    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(
            update_gallery_audio_flags(
                "missing", AudioGalleryFlagsPatch(archived = True), current_subject = "tester"
            )
        )
    assert excinfo.value.status_code == 404


def test_clear_route_refuses_with_an_unreadable_store():
    from fastapi import HTTPException
    from routes.inference import clear_gallery_audio

    record = gallery.save(_wav(), _meta())
    gallery.set_flags(record["id"], archived = True)
    (gallery.gallery_dir() / ".flags.json").write_text("corrupt", encoding = "utf-8")
    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(clear_gallery_audio(current_subject = "tester"))
    assert excinfo.value.status_code == 503
    assert gallery.audio_path(record["id"]) is not None


def test_prune_skips_when_the_flag_store_cannot_be_read(monkeypatch):
    # The prune deletes on "not archived", so an unreadable store must stop it as it stops clear().
    monkeypatch.setenv("UNSLOTH_AUDIO_GALLERY_MAX_CLIPS", "2")
    shelved = _save_with_mtime("shelved", 100.0)
    gallery.set_flags(shelved["id"], archived = True)
    _save_with_mtime("b", 200.0)
    _save_with_mtime("c", 300.0)
    (gallery.gallery_dir() / ".flags.json").write_text("corrupt", encoding = "utf-8")
    gallery.save(_wav(), _meta(prompt = "d"))
    assert gallery.audio_path(shelved["id"]) is not None
    assert len(list(gallery.gallery_dir().glob("*.wav"))) == 4


def test_prune_spares_a_clip_archived_after_its_snapshot(monkeypatch):
    # Victims are chosen under the lock, so a racing archive is honoured.
    from core.inference import gallery_flags

    doomed = _save_with_mtime("doomed", 100.0)
    _save_with_mtime("b", 200.0)
    _save_with_mtime("c", 300.0)

    real = gallery._list_audio_entries
    fired = []

    def racing(*args, **kwargs):
        entries = real(*args, **kwargs)
        if not fired:
            fired.append(True)
            gallery_flags.set_flags_locked(gallery.gallery_dir(), doomed["id"], archived = True)
        return entries

    monkeypatch.setattr(gallery, "_list_audio_entries", racing)
    monkeypatch.setenv("UNSLOTH_AUDIO_GALLERY_MAX_CLIPS", "2")
    gallery.save(_wav(), _meta(prompt = "d"))

    assert gallery.audio_path(doomed["id"]) is not None
    assert [r["prompt"] for r in gallery.list_audio(archived = True)] == ["doomed"]


def test_prune_stops_when_the_cross_process_lock_is_unavailable(monkeypatch):
    import contextlib

    from core.inference import gallery_flags

    doomed = _save_with_mtime("doomed", 100.0)
    _save_with_mtime("b", 200.0)
    _save_with_mtime("c", 300.0)

    @contextlib.contextmanager
    def unlocked(_directory):
        yield False

    real_read = gallery_flags.read_trusted

    def racing_read(directory):
        flags = real_read(directory)
        gallery_flags.set_flags_locked(directory, doomed["id"], archived = True)
        return flags

    monkeypatch.setattr(gallery_flags, "_file_lock", unlocked)
    monkeypatch.setattr(gallery_flags, "read_trusted", racing_read)
    monkeypatch.setenv("UNSLOTH_AUDIO_GALLERY_MAX_CLIPS", "2")
    gallery.save(_wav(), _meta(prompt = "d"))

    assert gallery.audio_path(doomed["id"]) is not None


def test_clear_stops_when_the_cross_process_lock_is_unavailable(monkeypatch):
    import contextlib

    from core.inference import gallery_flags

    record = _save_with_mtime("active", 100.0)

    @contextlib.contextmanager
    def unlocked(_directory):
        yield False

    monkeypatch.setattr(gallery_flags, "_file_lock", unlocked)
    with pytest.raises(gallery_flags.FlagsUnavailable):
        gallery.clear()
    assert gallery.audio_path(record["id"]) is not None


def test_pinned_clips_lead_history():
    old = _save_with_mtime("old", 100.0)
    new = _save_with_mtime("new", 200.0)
    assert gallery.set_flags(old["id"], pinned = True)["pinned"] is True
    assert [r["id"] for r in gallery.list_audio()] == [old["id"], new["id"]]
    gallery.set_flags(old["id"], pinned = False)
    assert [r["id"] for r in gallery.list_audio()] == [new["id"], old["id"]]


def test_records_carry_the_listing_sort_key():
    record = _save_with_mtime("a", 150.0)
    assert gallery.list_audio()[0]["order_at"] == 150.0
    assert record["pinned"] is False


def test_move_places_a_clip_after_its_new_neighbour():
    a, b, c = (_save_with_mtime(p, t) for p, t in (("a", 300.0), ("b", 200.0), ("c", 100.0)))
    gallery.move(c["id"], a["id"])
    assert [r["prompt"] for r in gallery.list_audio()] == ["a", "c", "b"]
    gallery.move(b["id"], None)
    assert [r["prompt"] for r in gallery.list_audio()] == ["b", "a", "c"]


def test_move_among_pins_pins_the_clip():
    a, b, c = (_save_with_mtime(p, t) for p, t in (("a", 300.0), ("b", 200.0), ("c", 100.0)))
    gallery.set_flags(a["id"], pinned = True)
    gallery.set_flags(b["id"], pinned = True)
    moved = gallery.move(c["id"], None)
    assert moved["pinned"] is True
    assert [r["prompt"] for r in gallery.list_audio()][0] == "c"


def test_move_refuses_unknown_or_archived_clips():
    a = _save_with_mtime("a", 100.0)
    b = _save_with_mtime("b", 200.0)
    assert gallery.move("missing", None) is None
    gallery.set_flags(a["id"], archived = True)
    assert gallery.move(a["id"], None) is None
    with pytest.raises(KeyError):
        gallery.move(b["id"], "not-on-the-shelf")


def test_cursor_pages_through_pins_and_dragged_clips():
    clips = [_save_with_mtime(f"p{i}", float(i)) for i in range(1, 6)]
    gallery.set_flags(clips[0]["id"], pinned = True)
    gallery.move(clips[1]["id"], clips[3]["id"])
    full = [r["id"] for r in gallery.list_audio()]
    seen, before = [], None
    while True:
        page = gallery.list_audio_page(limit = 2, before = before)
        seen += [r["id"] for r, _ in page]
        if len(page) < 2:
            break
        before = page[-1][1]
    assert seen == full


def test_prune_goes_by_age_and_spares_pins(monkeypatch):
    monkeypatch.setenv("UNSLOTH_AUDIO_GALLERY_MAX_CLIPS", "2")
    oldest = _save_with_mtime("oldest", 100.0)
    gallery.set_flags(oldest["id"], pinned = True)
    b = _save_with_mtime("b", 200.0)
    c = _save_with_mtime("c", 300.0)
    gallery.move(c["id"], b["id"])
    gallery.save(_wav(), _meta(prompt = "d"))
    assert gallery.audio_path(oldest["id"]) is not None
    assert gallery.audio_path(c["id"]) is not None
    assert gallery.audio_path(b["id"]) is None


def test_list_route_round_trips_a_pinned_cursor():
    from routes.inference import list_gallery_audio

    clips = [_save_with_mtime(f"p{i}", float(i)) for i in range(1, 4)]
    for clip in clips:
        gallery.set_flags(clip["id"], pinned = True)
    page1 = asyncio.run(list_gallery_audio(limit = 2, current_subject = "tester"))
    assert page1.has_more and page1.next_before_pin is not None
    page2 = asyncio.run(
        list_gallery_audio(
            limit = 2,
            before_mtime = page1.next_before_mtime,
            before_id = page1.next_before_id,
            before_pin = page1.next_before_pin,
            current_subject = "tester",
        )
    )
    ids = [c.id for c in page1.audio] + [c.id for c in page2.audio]
    assert sorted(ids) == sorted(c["id"] for c in clips) and len(set(ids)) == 3


def test_project_route_copies_the_wav(monkeypatch, tmp_path):
    import storage.studio_db as studio_db
    from models.inference import GalleryProjectRequest
    from routes.inference import add_gallery_audio_to_project

    root = tmp_path / "Projects" / "demo"
    (root / "sandbox").mkdir(parents = True)
    monkeypatch.setattr(
        studio_db,
        "ensure_chat_project_workspace",
        lambda pid: {"id": pid, "rootPath": str(root), "sandboxPath": str(root / "sandbox")},
    )
    record = gallery.save(_wav(), _meta())
    result = asyncio.run(
        add_gallery_audio_to_project(
            record["id"], GalleryProjectRequest(project_id = "p1"), current_subject = "tester"
        )
    )
    dest = root / "sandbox" / "audio" / f"{record['id']}.wav"
    assert result.path == str(dest) and result.already is False
    assert dest.read_bytes() == _wav()


def test_list_route_resolves_a_cursor_sent_without_its_pin_rank():
    from routes.inference import list_gallery_audio

    clips = [_save_with_mtime(f"p{i}", float(i)) for i in range(1, 5)]
    for clip in clips[:3]:
        gallery.set_flags(clip["id"], pinned = True)
    page1 = asyncio.run(list_gallery_audio(limit = 2, current_subject = "tester"))
    page2 = asyncio.run(
        list_gallery_audio(
            limit = 2,
            before_mtime = page1.next_before_mtime,
            before_id = page1.next_before_id,
            current_subject = "tester",
        )
    )
    ids = [c.id for c in page1.audio] + [c.id for c in page2.audio]
    assert sorted(ids) == sorted(c["id"] for c in clips)


def test_a_new_clip_records_its_workflow():
    from routes.inference import _persist_tts_clip

    speech = _persist_tts_clip(_wav(), 24000, "hi", "kokoro", "audiocpp_tts")
    song = _persist_tts_clip(_wav(), 44100, "a song", "ace-step", "audiocpp_music")
    sidecars = {
        clip["id"]: json.loads(
            (gallery.gallery_dir() / f"{clip['id']}.json").read_text(encoding = "utf-8")
        )
        for clip in (speech, song)
    }
    assert sidecars[speech["id"]]["workflow"] == "speak"
    assert sidecars[song["id"]]["workflow"] == "music"
    assert {r["id"]: r["workflow"] for r in gallery.list_audio()} == {
        speech["id"]: "speak",
        song["id"]: "music",
    }


def test_an_old_clip_takes_its_workflow_from_its_audio_type():
    from models.inference import AudioGalleryItem

    speech = gallery.save(_wav(), _meta())
    song = gallery.save(_wav(), _meta(audio_type = "minimax_music3"))
    listed = {r["id"]: r for r in gallery.list_audio()}
    assert listed[speech["id"]]["workflow"] == "speak"
    assert listed[song["id"]]["workflow"] == "music"
    assert AudioGalleryItem(**listed[song["id"]]).workflow == "music"
    assert gallery.set_flags(song["id"], pinned = True)["workflow"] == "music"


def test_a_scoped_clear_keeps_the_other_workflows_clips():
    speech = _save_with_mtime("speech", 100.0)
    song = gallery.save(_wav(), _meta(audio_type = "audiocpp_music"))
    gallery.set_flags(song["id"], pinned = True)
    assert gallery.clear(workflow = "speak") == 1
    assert gallery.audio_path(speech["id"]) is None
    remaining = gallery.list_audio()
    assert [r["id"] for r in remaining] == [song["id"]] and remaining[0]["pinned"] is True
    assert gallery.clear(workflow = "music") == 1
    assert gallery.list_audio() == []


def test_a_scoped_clear_with_archived_keeps_the_other_workflows_flags():
    shelved_song = gallery.save(_wav(), _meta(audio_type = "audiocpp_music"))
    gallery.set_flags(shelved_song["id"], archived = True)
    _save_with_mtime("speech", 100.0)
    assert gallery.clear(include_archived = True, workflow = "speak") == 1
    assert [r["id"] for r in gallery.list_audio(archived = True)] == [shelved_song["id"]]


def test_the_clear_route_scopes_to_a_workflow():
    from routes.inference import clear_gallery_audio

    gallery.save(_wav(), _meta())
    song = gallery.save(_wav(), _meta(audio_type = "audiocpp_music"))
    assert asyncio.run(clear_gallery_audio(workflow = "speak", current_subject = "tester")) == {
        "removed": 1
    }
    assert [r["id"] for r in gallery.list_audio()] == [song["id"]]
    assert asyncio.run(clear_gallery_audio(current_subject = "tester")) == {"removed": 1}


def test_a_clone_clip_keeps_its_workflow_and_run_fields_and_survives_a_speak_clear():
    from models.inference import AudioGalleryItem
    from routes.inference import _persist_tts_clip

    run = {"voice_id": "v" * 32, "settings": {"reference_text_used": True}, "source_clip_id": None}
    speech = gallery.save(_wav(), _meta())
    clone = _persist_tts_clip(_wav(), 24000, "hi", "qwen3-base", "audiocpp_tts", run, "clone")
    meta = json.loads((gallery.gallery_dir() / f"{clone['id']}.json").read_text(encoding = "utf-8"))
    assert meta["workflow"] == "clone" and "source_clip_id" not in meta
    (item,) = [AudioGalleryItem(**r) for r in gallery.list_audio() if r["id"] == clone["id"]]
    assert (
        item.workflow == "clone"
        and item.voice_id == "v" * 32
        and item.settings["reference_text_used"] is True
    )
    assert gallery.set_flags(clone["id"], pinned = True)["workflow"] == "clone"
    assert gallery.clear(workflow = "speak") == 1
    assert gallery.audio_path(speech["id"]) is None
    assert [r["id"] for r in gallery.list_audio()] == [clone["id"]]


def test_an_edit_keeps_its_workflow_and_a_scoped_clear_takes_its_source_clips():
    from models.inference import AudioGalleryItem
    from routes.inference import _persist_tts_clip, clear_gallery_audio

    speech = gallery.save(_wav(), _meta())
    source = gallery.save(
        _wav(),
        _meta(
            model = "Recording",
            audio_type = "recording",
            workflow = "edit",
            role = "source",
        ),
    )
    edited = _persist_tts_clip(
        _wav(),
        24000,
        "edited",
        "m",
        "audiocpp_tts",
        {"role": "output", "source_clip_id": source["id"]},
        "edit",
    )
    listed = {r["id"]: r for r in gallery.list_audio()}
    assert listed[source["id"]]["workflow"] == listed[edited["id"]]["workflow"] == "edit"
    assert AudioGalleryItem(**listed[edited["id"]]).source_clip_id == source["id"]
    assert gallery.set_flags(edited["id"], pinned = True)["workflow"] == "edit"
    cleared = asyncio.run(clear_gallery_audio(workflow = "edit", current_subject = "tester"))
    assert cleared == {"removed": 2}
    assert [r["id"] for r in gallery.list_audio()] == [speech["id"]]


def test_the_inputs_and_voices_folders_never_list_as_clips():
    gallery.save(_wav(), _meta())
    for folder in ("inputs", "voices"):
        directory = gallery.gallery_dir() / folder
        directory.mkdir(exist_ok = True)
        (directory / "abc.wav").write_bytes(_wav())
        (directory / "abc.json").write_text(json.dumps(_meta()), encoding = "utf-8")
    assert len(gallery.list_audio()) == 1
    assert gallery.clear() == 1
    assert (gallery.gallery_dir() / "inputs" / "abc.wav").is_file()


def test_only_a_hidden_edit_source_is_kept_for_the_clip_that_plays_it(monkeypatch):
    reference = gallery.save(_wav(), _meta(workflow = "speak"))
    gallery.save(_wav(), _meta(workflow = "clone", source_clip_id = reference["id"]))
    assert gallery.clear(workflow = "speak") == 1
    monkeypatch.setenv("UNSLOTH_AUDIO_GALLERY_MAX_CLIPS", "1")
    reference = _save_with_mtime("reference", 1000)
    clone = gallery.save(_wav(), _meta(workflow = "clone", source_clip_id = reference["id"]))
    gallery._prune_to_cap()
    assert [r["id"] for r in gallery.list_audio()] == [clone["id"]]


def _stem_meta(**over):
    return _meta(
        audio_type = "audiocpp_sep",
        workflow = "separate",
        sample_rate = 44100,
        role = "vocals",
        group_id = "g" * 32,
        settings = {"stems": ["vocals", "instrumental"], "num_overlap": None},
        **over,
    )


def test_save_file_copies_across_filesystems(tmp_path, monkeypatch):
    import errno

    src = tmp_path / "vocals.wav"
    src.write_bytes(_wav())
    real_replace = os.replace

    def cross_device(a, b):
        if Path(a) == src:
            raise OSError(errno.EXDEV, "Invalid cross-device link")
        return real_replace(a, b)

    monkeypatch.setattr(gallery.os, "replace", cross_device)
    record = gallery.save_file(src, _stem_meta())
    assert gallery.audio_path(record["id"]).read_bytes() == _wav()
    assert not list(gallery.gallery_dir().glob(".*.tmp"))


def test_save_file_rolls_back_when_the_sidecar_fails(tmp_path, monkeypatch):
    src = tmp_path / "vocals.wav"
    src.write_bytes(_wav())
    gallery.gallery_dir()

    def broken(self, *_args, **_kwargs):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(Path, "write_text", broken)
    with pytest.raises(OSError):
        gallery.save_file(src, _stem_meta())
    monkeypatch.undo()
    monkeypatch.setattr(gallery, "studio_root", lambda: tmp_path)
    assert gallery.list_audio() == []
    assert not [p for p in gallery.gallery_dir().iterdir() if p.suffix in (".wav", ".json", ".tmp")]


def test_save_file_prunes_only_when_asked(tmp_path, monkeypatch):
    pruned = []
    monkeypatch.setattr(gallery, "_prune_to_cap", lambda: pruned.append(1) or 0)
    for prune in (False, True):
        src = tmp_path / f"{prune}.wav"
        src.write_bytes(_wav())
        gallery.save_file(src, _stem_meta(), prune = prune)
    assert pruned == [1]


def test_prune_keeps_or_drops_a_group_whole(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_AUDIO_GALLERY_MAX_CLIPS", "3")

    def group(gid, t):
        ids = []
        for i, role in enumerate(("vocals", "drums", "bass", "other")):
            src = tmp_path / f"{gid}{role}.wav"
            src.write_bytes(_wav())
            meta = {**_stem_meta(), "group_id": gid, "role": role}
            record = gallery.save_file(src, meta, prune = False)
            os.utime(gallery.gallery_dir() / f"{record['id']}.wav", (t + i, t + i))
            ids.append(record["id"])
        return ids

    old = group("a" * 32, 100.0)
    new = group("b" * 32, 200.0)
    gallery._prune_to_cap()
    assert all(gallery.audio_path(i) is not None for i in new)
    assert all(gallery.audio_path(i) is None for i in old)


def test_a_separate_scoped_clear_spares_speak_and_clone():
    speech = gallery.save(_wav(), _meta())
    clone = gallery.save(_wav(), _meta(audio_type = "audiocpp_tts", workflow = "clone"))
    stem = gallery.save(_wav(), _stem_meta())
    assert gallery.clear(workflow = "separate") == 1
    assert gallery.audio_path(stem["id"]) is None
    assert {r["id"] for r in gallery.list_audio()} == {speech["id"], clone["id"]}


def test_a_convert_clip_keeps_its_workflow_meta_and_scoped_clear():
    from models.inference import AudioGalleryItem
    from routes.inference import clear_gallery_audio, _persist_tts_clip

    speech = gallery.save(_wav(), _meta())
    convert = _persist_tts_clip(
        _wav(),
        16000,
        "take.wav → Manthos",
        "rvc",
        "audiocpp_tts",
        {
            "role": "output",
            "source_input_id": "i" * 32,
            "source_name": "take.wav",
            "reference_name": "Manthos",
            "target_builtin": "manthos",
            "settings": {"mode": "speech", "pitch": 3, "pitch_auto": False, "options": {}},
        },
        "convert",
    )
    listed = {r["id"]: r for r in gallery.list_audio()}
    item = AudioGalleryItem(**listed[convert["id"]])
    assert item.workflow == "convert" and item.prompt == "take.wav → Manthos"
    assert (item.source_input_id, item.source_name) == ("i" * 32, "take.wav")
    assert (item.reference_name, item.target_builtin) == ("Manthos", "manthos")
    assert gallery.set_flags(convert["id"], pinned = True)["workflow"] == "convert"
    assert asyncio.run(clear_gallery_audio(workflow = "convert", current_subject = "tester")) == {
        "removed": 1
    }
    assert [r["id"] for r in gallery.list_audio()] == [speech["id"]]


def test_a_kept_source_goes_with_its_clip(tmp_path):
    source = tmp_path / "source.wav"
    source.write_bytes(_wav())
    kept = gallery.save(_wav(), _meta(), source)
    plain = gallery.save(_wav(), _meta())
    assert kept["source_saved"] is True and "source_saved" not in plain
    copy = gallery.gallery_dir() / f"{kept['id']}.source.wav"
    assert copy.read_bytes() == source.read_bytes()
    assert gallery.owned_source_path(kept["id"]) == copy
    assert gallery.owned_source_path(plain["id"]) is None
    assert sorted(r["id"] for r in gallery.list_audio()) == sorted([kept["id"], plain["id"]])
    assert gallery.move(plain["id"], None)["id"] == plain["id"]
    gallery.set_flags(kept["id"], archived = True)
    assert gallery.clear() == 1 and copy.is_file()
    assert gallery.clear(include_archived = True) == 1
    assert not copy.exists()
    again = gallery.save(_wav(), _meta(), source)
    assert gallery.delete(again["id"])
    assert not (gallery.gallery_dir() / f"{again['id']}.source.wav").exists()


def test_deleting_the_last_edit_takes_its_hidden_original():
    source = gallery.save(_wav(), _meta(workflow = "edit", role = "source"))
    first, second = (
        gallery.save(_wav(), _meta(workflow = "edit", role = "output", source_clip_id = source["id"]))
        for _ in range(2)
    )
    assert gallery.delete(first["id"]) is True
    assert gallery.audio_path(source["id"]) is not None  # the second edit still plays it
    assert gallery.delete(second["id"]) is True
    assert gallery.audio_path(source["id"]) is None


def test_deleting_a_clip_made_from_a_listed_clip_keeps_that_clip():
    take = gallery.save(_wav(), _meta())
    converted = gallery.save(_wav(), _meta(workflow = "convert", source_clip_id = take["id"]))
    assert gallery.delete(converted["id"]) is True
    assert gallery.audio_path(take["id"]) is not None


def test_a_group_delete_takes_only_that_runs_clips():
    from fastapi import HTTPException
    from routes.inference import delete_gallery_audio_group

    stems = [
        gallery.save(_wav(), _meta(workflow = "separate", group_id = "g1", role = r))
        for r in ("vocals", "drums")
    ]
    other = gallery.save(_wav(), _meta(workflow = "separate", group_id = "g2", role = "vocals"))
    loose = gallery.save(_wav(), _meta())
    assert asyncio.run(delete_gallery_audio_group("g1", current_subject = "tester")) == {"removed": 2}
    assert all(gallery.audio_path(stem["id"]) is None for stem in stems)
    assert {r["id"] for r in gallery.list_audio()} == {other["id"], loose["id"]}
    assert gallery.delete_group("") == 0
    with pytest.raises(HTTPException) as missing:
        asyncio.run(delete_gallery_audio_group("g1", current_subject = "tester"))
    assert missing.value.status_code == 404


def test_a_group_delete_spares_the_runs_archived_clips():
    # A stem restored from the archive leaves its siblings archived; deleting the run on the
    # Separate page, which lists only active clips, must not take them.
    stems = [
        gallery.save(_wav(), _meta(workflow = "separate", group_id = "g1", role = r))
        for r in ("vocals", "drums", "bass")
    ]
    for stem in stems[1:]:
        gallery.set_flags(stem["id"], archived = True)
    assert gallery.delete_group("g1") == 1
    assert gallery.audio_path(stems[0]["id"]) is None
    assert {r["id"] for r in gallery.list_audio(archived = True)} == {s["id"] for s in stems[1:]}


def test_the_group_route_refuses_with_an_unreadable_store():
    from fastapi import HTTPException
    from routes.inference import delete_gallery_audio_group

    stem = gallery.save(_wav(), _meta(workflow = "separate", group_id = "g1", role = "vocals"))
    gallery.set_flags(stem["id"], archived = True)
    (gallery.gallery_dir() / ".flags.json").write_text("corrupt", encoding = "utf-8")
    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(delete_gallery_audio_group("g1", current_subject = "tester"))
    assert excinfo.value.status_code == 503
    assert gallery.audio_path(stem["id"]) is not None


def test_an_archive_racing_a_group_delete_is_never_lost(monkeypatch):
    import threading

    from core.inference import gallery_flags

    stems = [
        gallery.save(_wav(), _meta(workflow = "separate", group_id = "g1", role = r))
        for r in ("vocals", "drums")
    ]
    target = stems[1]["id"]
    archived: list = []
    real_read_trusted = gallery_flags.read_trusted

    def read_then_archive(directory):
        flags = real_read_trusted(directory)
        worker = threading.Thread(
            target = lambda: archived.append(gallery.set_flags(target, archived = True))
        )
        worker.start()
        worker.join(timeout = 0.5)
        return flags

    monkeypatch.setattr(gallery_flags, "read_trusted", read_then_archive)
    gallery.delete_group("g1")
    monkeypatch.setattr(gallery_flags, "read_trusted", real_read_trusted)
    for _ in range(50):
        if archived:
            break
        threading.Event().wait(0.1)
    # The archive either lands first and spares the stem, or waits and finds it gone.
    assert archived and (archived[0] is None) == (gallery.audio_path(target) is None)
