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
    # The list carries summaries: the record plus its segment and word counts.
    assert gallery.list_transcripts()["transcripts"] == [
        {**record, "segment_count": 0, "has_words": False}
    ]


def test_cursor_survives_new_records_and_deletion():
    first, second = save(), save()
    page = gallery.list_transcripts(limit = 1)
    assert page["transcripts"] == [gallery.summary(second)]
    gallery.delete(second["id"])
    save()
    assert gallery.list_transcripts(before = page["next_cursor"])["transcripts"] == [
        gallery.summary(first)
    ]


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
    assert gallery.list_transcripts()["transcripts"] == [gallery.summary(record)]
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


DETAILS = {
    "segments": [
        {"start": 0.12, "end": 3.14, "text": "Welcome back.", "speaker": "S01"},
        {"start": 3.21, "end": 7.82, "text": "Check the forecast.", "speaker": "S02"},
    ],
    "words": [{"start": 0.12, "end": 0.5, "word": "Welcome"}],
    "speakers": [{"id": "S01", "label": "Speaker 1"}, {"id": "S02", "label": "Speaker 2"}],
    "source": {"kind": "input", "id": "c" * 32, "name": "meeting.webm"},
    "timestamps": True,
}


def save_details(**overrides):
    result = {"text": "Welcome back. Check the forecast.", "model": "moss", "duration": 7.9}
    return gallery.save({**result, **DETAILS, **overrides}, "meeting.webm")


def test_an_old_record_reads_exactly_as_before():
    record = {
        "id": "d" * 32,
        "title": "old.wav",
        "text": "old words",
        "model": "tiny",
        "duration": 1.0,
        "language": None,
        "created_at": "2026-01-01T00:00:00+00:00",
        "archived": False,
    }
    (gallery.gallery_dir() / f"{record['id']}.json").write_text(json.dumps(record))
    assert gallery.get(record["id"]) == record
    assert gallery.get(record["id"]) == record
    (row,) = gallery.list_transcripts()["transcripts"]
    assert row == {**record, "segment_count": 0, "has_words": False}


def test_details_round_trip_and_the_list_carries_counts_only():
    record = save_details()
    for key, value in DETAILS.items():
        assert record[key] == value
    assert gallery.get(record["id"]) == record
    (row,) = gallery.list_transcripts()["transcripts"]
    assert "segments" not in row and "words" not in row
    assert row["segment_count"] == 2 and row["has_words"] is True
    assert row["speakers"] == DETAILS["speakers"] and row["source"] == DETAILS["source"]
    # A source is named by id: no path is ever stored.
    stored = (gallery.gallery_dir() / f"{record['id']}.json").read_text(encoding = "utf-8")
    assert "/" not in json.loads(stored)["source"]["id"]


def test_malformed_details_are_dropped_and_the_record_survives():
    record = save_details(
        segments = [
            {"start": "soon", "end": 1, "text": "bad time"},
            {"start": 5, "end": 1, "text": "backwards"},
            {"start": 0, "end": 1, "text": "kept"},
            "not a segment",
        ],
        words = [{"start": 0, "end": 1, "word": "ok"}] * 100_001,
        speakers = [{"id": "S01", "label": "Speaker 1"}, {"id": 3}],
        speaker_names = {"S01": "x" * 41, "S99": "ghost"},
        source = {"kind": "path", "id": "/etc/passwd", "name": "x"},
    )
    assert record["segments"] == [{"start": 0.0, "end": 1.0, "text": "kept"}]
    assert "words" not in record and "speaker_names" not in record and "source" not in record
    assert record["speakers"] == [{"id": "S01", "label": "Speaker 1"}]
    # A hand-edited file loses only its bad optional keys on read.
    path = gallery.gallery_dir() / f"{record['id']}.json"
    data = json.loads(path.read_text(encoding = "utf-8"))
    data["segments"] = {"not": "a list"}
    data["speaker_names"] = {"S01": 7}
    path.write_text(json.dumps(data))
    read = gallery.get(record["id"])
    assert read["text"] == record["text"] and "segments" not in read and "speaker_names" not in read
    assert "timestamps" not in read


def test_speaker_names_are_validated_cleared_and_written_atomically(monkeypatch):
    record = save_details()
    named = gallery.set_speaker_names(record["id"], {"S01": "  Alice ", "S02": "Bob"})
    assert named["speaker_names"] == {"S01": "Alice", "S02": "Bob"}
    assert gallery.get(record["id"])["speaker_names"] == {"S01": "Alice", "S02": "Bob"}
    assert gallery.set_speaker_names(record["id"], {"S02": None})["speaker_names"] == {
        "S01": "Alice"
    }
    with pytest.raises(gallery.TranscriptPatchError, match = "no speaker"):
        gallery.set_speaker_names(record["id"], {"S09": "Ghost"})
    with pytest.raises(gallery.TranscriptPatchError, match = "40"):
        gallery.set_speaker_names(record["id"], {"S01": "x" * 41})
    assert gallery.get(record["id"])["speaker_names"] == {"S01": "Alice"}
    assert "speaker_names" not in gallery.set_speaker_names(record["id"], {"S01": ""})
    # An archived transcript stays archived through a rename.
    gallery.set_archived(record["id"], True)
    assert gallery.set_speaker_names(record["id"], {"S01": "Al"})["archived"] is True

    def fail(*args):
        raise OSError("disk full")

    monkeypatch.setattr(gallery.os, "replace", fail)
    with pytest.raises(OSError):
        gallery.set_speaker_names(record["id"], {"S01": "Never"})
    assert gallery.get(record["id"])["speaker_names"] == {"S01": "Al"}
    assert not list(gallery.gallery_dir().glob(".*.tmp"))


def test_unknown_and_unsafe_ids_have_no_speakers_to_name(tmp_path):
    assert gallery.set_speaker_names("../private", {"S01": "x"}) is None
    assert gallery.set_speaker_names("e" * 32, {"S01": "x"}) is None
    assert gallery.get("../private") is None
    foreign = gallery.gallery_dir() / ("f" * 32 + ".json")
    foreign.write_text(json.dumps({"text": "foreign"}))
    assert gallery.get("f" * 32) is None
    assert gallery.set_speaker_names("f" * 32, {"S01": "x"}) is None
