# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The listing reuses each transcript's row until its file changes, with output unchanged."""

import json
import os
import threading

import pytest

from core.inference import gallery_flags, transcript_gallery as gallery


@pytest.fixture(autouse = True)
def isolated_gallery(monkeypatch, tmp_path):
    monkeypatch.setattr(gallery, "studio_root", lambda: tmp_path)


@pytest.fixture
def loads(monkeypatch):
    calls = []
    real = gallery._load

    def counting(path, transcript_id):
        if not counting.paused:
            calls.append(transcript_id)
        return real(path, transcript_id)

    counting.paused = False
    monkeypatch.setattr(gallery, "_load", counting)
    return calls


def save(text = "Hello", speakers = True):
    details = {}
    if speakers:
        details = {
            "segments": [{"start": 0, "end": 1, "text": text, "speaker": "S0"}],
            "words": [{"start": 0, "end": 1, "word": text}],
            "speakers": [{"id": "S0", "label": "Speaker 1"}],
            "source": {"kind": "input", "id": "abc", "name": "a.wav"},
        }
    return gallery.save({"text": text, "model": "tiny", "duration": 1.0, **details}, "a.wav")


def uncached(
    limit = 50,
    before = None,
    archived = False,
):
    """The listing as it was computed before rows were cached."""
    directory = gallery.gallery_dir()
    flags = gallery_flags.read(directory)
    records = []
    for path in directory.glob("*.json"):
        record = gallery._read(directory, path.stem)
        if record is None or gallery_flags.is_archived(flags, path.stem) != archived:
            continue
        if before is not None and f"{record['created_at']}|{record['id']}" >= before:
            continue
        record["archived"] = archived
        records.append(gallery.summary(record))
    records.sort(key = lambda record: (record["created_at"], record["id"]), reverse = True)
    visible = records[:limit]
    cursor = f"{visible[-1]['created_at']}|{visible[-1]['id']}" if len(records) > limit else None
    return {"transcripts": visible, "next_cursor": cursor}


def same(*args):
    listed = json.dumps(gallery.list_transcripts(*args))
    counting = gallery._load
    if hasattr(counting, "paused"):
        counting.paused = True
    try:
        expected = json.dumps(uncached(*args))
    finally:
        if hasattr(counting, "paused"):
            counting.paused = False
    assert listed == expected


def test_a_warm_listing_parses_nothing_and_a_change_parses_one_file(loads):
    first, second = save("one"), save("two")
    same()
    assert sorted(loads) == sorted([first["id"], second["id"]])
    loads.clear()
    same()
    same(1)
    same(1, gallery.list_transcripts(1)["next_cursor"])
    assert loads == []
    gallery.set_speaker_names(first["id"], {"S0": "Ann"})
    loads.clear()
    same()
    assert loads == [first["id"]]


def test_archive_state_is_read_fresh_without_parsing(loads):
    record = save()
    same()
    loads.clear()
    gallery.set_archived(record["id"], True)
    loads.clear()
    assert gallery.list_transcripts()["transcripts"] == []
    assert gallery.list_transcripts(archived = True)["transcripts"][0]["archived"] is True
    same(50, None, True)
    assert loads == []
    gallery.set_archived(record["id"], False)
    loads.clear()
    assert gallery.list_transcripts()["transcripts"][0]["archived"] is False
    assert loads == []


def test_returned_rows_are_independent_of_the_cache():
    save()
    page = gallery.list_transcripts()
    row = page["transcripts"][0]
    row["title"] = "changed"
    row["archived"] = True
    row["speakers"].append({"id": "x", "label": "x"})
    row["source"].clear()
    same()


def test_in_place_edits_deletes_and_corruption_are_seen():
    record = save()
    path = gallery.gallery_dir() / f"{record['id']}.json"
    same()
    stored = json.loads(path.read_text(encoding = "utf-8"))
    stored["title"] = "a much longer title than before"
    path.write_text(json.dumps(stored), encoding = "utf-8")
    same()
    assert gallery.list_transcripts()["transcripts"][0]["title"] == stored["title"]
    path.write_text("{broken", encoding = "utf-8")
    same()
    assert gallery.list_transcripts()["transcripts"] == []
    path.write_text(json.dumps(stored), encoding = "utf-8")
    same()
    gallery.delete(record["id"])
    same()
    assert gallery.list_transcripts()["transcripts"] == []


@pytest.mark.skipif(os.name == "nt" or os.geteuid() == 0, reason = "needs POSIX permissions")
def test_an_unreadable_file_is_not_remembered_as_missing():
    record = save()
    path = gallery.gallery_dir() / f"{record['id']}.json"
    os.chmod(path, 0)
    try:
        assert gallery.list_transcripts()["transcripts"] == []
    finally:
        os.chmod(path, 0o600)
    same()
    assert gallery.list_transcripts()["transcripts"][0]["id"] == record["id"]


def test_accounts_never_share_rows(monkeypatch, tmp_path):
    record = save("owner")
    owner_file = gallery.gallery_dir() / f"{record['id']}.json"
    monkeypatch.setattr(gallery, "is_owner_context", lambda: False)
    monkeypatch.setattr(gallery, "account_path", lambda name: tmp_path / "account" / name)
    monkeypatch.setattr(gallery, "ensure_account_dir", gallery.ensure_dir)
    other = gallery.gallery_dir() / owner_file.name
    stored = json.loads(owner_file.read_text(encoding = "utf-8"))
    other.write_text(json.dumps({**stored, "title": "other"}), encoding = "utf-8")
    assert gallery.list_transcripts()["transcripts"][0]["title"] == "other"
    same()
    monkeypatch.setattr(gallery, "is_owner_context", lambda: True)
    assert gallery.list_transcripts()["transcripts"][0]["title"] == "a.wav"
    same()


def test_the_cache_stays_bounded_and_correct(monkeypatch):
    monkeypatch.setattr(gallery, "_row_cache", type(gallery._row_cache)())
    monkeypatch.setattr(gallery, "_row_cache_bytes", 0)
    monkeypatch.setattr(gallery, "_ROW_CACHE_MAX", 3)
    for index in range(6):
        save(f"text {index}")
    same()
    assert len(gallery._row_cache) == 3
    monkeypatch.setattr(gallery, "_ROW_CACHE_BUDGET", 2 * 4096 + 100)
    save()
    same()
    assert len(gallery._row_cache) <= 2
    assert gallery._row_cache_bytes == sum(entry[2] for entry in gallery._row_cache.values())
    monkeypatch.setattr(gallery, "_ROW_CACHE_ITEM_MAX", 10)
    gallery._row_cache.clear()
    monkeypatch.setattr(gallery, "_row_cache_bytes", 0)
    same()
    assert len(gallery._row_cache) == 0


def test_hand_edited_non_scalar_values_list_but_are_not_kept():
    record = save(speakers = False)
    path = gallery.gallery_dir() / f"{record['id']}.json"
    stored = json.loads(path.read_text(encoding = "utf-8"))
    path.write_text(json.dumps({**stored, "duration": {"odd": [1]}}), encoding = "utf-8")
    same()
    page = gallery.list_transcripts()
    page["transcripts"][0]["duration"]["odd"].append(2)
    same()
    assert str(path) not in gallery._row_cache


def test_concurrent_writers_leave_the_listing_exact():
    records = [save(f"start {index}") for index in range(8)]
    errors = []

    def write(index):
        try:
            for turn in range(15):
                gallery.set_speaker_names(records[index]["id"], {"S0": f"n{turn}" * (turn % 3 + 1)})
                gallery.set_archived(records[index]["id"], turn % 2 == 0)
        except Exception as exc:  # pragma: no cover - reported below
            errors.append(exc)

    def read():
        try:
            for _ in range(15):
                for archived in (False, True):
                    gallery.list_transcripts(3, None, archived)
        except Exception as exc:  # pragma: no cover - reported below
            errors.append(exc)

    threads = [threading.Thread(target = write, args = (i,)) for i in range(8)]
    threads += [threading.Thread(target = read) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    for archived in (False, True):
        same(50, None, archived)
        same(2, None, archived)
