# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import json

import pytest

from core.inference import gallery_flags, transcript_gallery as gallery


@pytest.fixture(autouse = True)
def isolated_gallery(monkeypatch, tmp_path):
    monkeypatch.setattr(gallery, "studio_root", lambda: tmp_path)


def save(title = "speech.wav"):
    return gallery.save({"text": "Hello 世界", "model": "tiny", "duration": 3.5}, title)


def test_round_trip_preserves_text_and_origin():
    record = save("some/folder/speech.wav")
    assert record["title"] == "speech.wav"
    assert gallery.get(record["id"]) == record
    assert gallery.list_transcripts()["transcripts"] == [record]


def test_cursor_survives_new_records_and_deletion():
    first, second = save(), save()
    page = gallery.list_transcripts(limit = 1)
    assert page["transcripts"] == [second]
    gallery.delete(second["id"])
    save()
    assert gallery.list_transcripts(before = page["next_cursor"])["transcripts"] == [first]


def test_clear_keeps_archived_transcripts():
    keep, remove = save(), save()
    gallery.set_archived(keep["id"], True)
    assert gallery.clear() == 1
    assert gallery.get(remove["id"]) is None
    assert gallery.list_transcripts(archived = True)["transcripts"][0]["id"] == keep["id"]
    gallery.set_archived(keep["id"], False)
    assert gallery.list_transcripts()["transcripts"][0]["id"] == keep["id"]


def test_clear_does_not_guess_when_archive_flags_are_corrupt():
    record = save()
    gallery.set_archived(record["id"], True)
    flags_path = gallery_flags._store_path(gallery.gallery_dir())
    flags_path.write_text("broken")
    with pytest.raises(gallery_flags.FlagsUnavailable):
        gallery.clear()
    assert (gallery.gallery_dir() / f"{record['id']}.json").exists()


@pytest.mark.parametrize(
    "contents",
    ["broken", json.dumps({"version": 1, "items": {}, "unreadable": True})],
)
def test_listing_recovers_when_archive_flags_are_unavailable(contents):
    record = save()
    gallery_flags._store_path(gallery.gallery_dir()).write_text(contents)
    assert gallery.list_transcripts()["transcripts"] == [record]
    with pytest.raises(gallery_flags.FlagsUnavailable):
        gallery.clear()
    assert gallery.get(record["id"])["text"] == record["text"]


def test_foreign_files_and_unsafe_ids_are_untouched(tmp_path):
    directory = gallery.gallery_dir()
    foreign = directory / ("a" * 32 + ".json")
    foreign.write_text(json.dumps({"text": "foreign"}))
    outside = tmp_path / "private.json"
    outside.write_text("secret")
    assert gallery.get("../private") is None
    assert not gallery.delete("../private")
    assert gallery.clear() == 0
    assert foreign.exists() and outside.exists()


def test_atomic_save_failure_does_not_leave_history_entry(monkeypatch):
    def fail(*args):
        raise OSError("disk full")

    monkeypatch.setattr(gallery.os, "replace", fail)
    with pytest.raises(OSError):
        save()
    assert gallery.list_transcripts()["transcripts"] == []
    assert not list(gallery.gallery_dir().glob("*.tmp"))


def test_account_roots_are_separate(monkeypatch, tmp_path):
    owner = save()
    monkeypatch.setattr(gallery, "is_owner_context", lambda: False)
    monkeypatch.setattr(gallery, "account_path", lambda name: tmp_path / "account" / name)
    monkeypatch.setattr(gallery, "ensure_account_dir", gallery.ensure_dir)
    assert gallery.get(owner["id"]) is None
    account = save()
    monkeypatch.setattr(gallery, "is_owner_context", lambda: True)
    assert gallery.get(account["id"]) is None


def test_archive_and_delete_survive_a_filesystem_without_flock(monkeypatch):
    """A lock we cannot take must not cost the user archiving, only clearing.

    ``clear`` deletes on the strength of a flag, so it fails closed. Archiving and
    deleting one named transcript do not, and audio_gallery.set_flags deliberately takes
    the plain lock for exactly that reason. Taking require_file_lock here made both a 500
    on any mount where flock is unavailable, while the same account's audio clips worked.
    """
    import contextlib

    @contextlib.contextmanager
    def unlockable(directory):
        yield False

    record = save()
    monkeypatch.setattr(gallery_flags, "_file_lock", unlockable)

    assert gallery.set_archived(record["id"], True)["archived"] is True
    assert gallery.delete(record["id"]) is True
    with pytest.raises(gallery_flags.FlagsUnavailable):
        gallery.clear()
