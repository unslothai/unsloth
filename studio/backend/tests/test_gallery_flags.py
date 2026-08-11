# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unit tests for the gallery pin/archive flag store: patch semantics, the fail-safe read,
atomic writes and orphan pruning."""

from __future__ import annotations

import json
import math
import os

import pytest

import core.inference.gallery_flags as flags


@pytest.fixture
def gdir(tmp_path):
    d = tmp_path / "images"
    d.mkdir()
    return d


def _store(directory):
    return directory / ".flags.json"


def test_unknown_id_reads_as_no_flags(gdir):
    items = flags.read(gdir)
    assert items == {}
    assert flags.flags_for(items, "nope") == {"pinned": False, "archived": False}
    assert flags.is_archived(items, "nope") is False


def test_missing_store_is_not_created_by_reading(gdir):
    flags.read(gdir)
    assert not _store(gdir).exists()


def test_set_and_read_back_each_flag(gdir):
    assert flags.set_flags(gdir, "a", pinned = True) == {"pinned": True, "archived": False}
    assert flags.set_flags(gdir, "b", archived = True) == {"pinned": False, "archived": True}
    items = flags.read(gdir)
    assert flags.flags_for(items, "a") == {"pinned": True, "archived": False}
    assert flags.is_archived(items, "b") is True


def test_none_leaves_the_other_flag_alone(gdir):
    flags.set_flags(gdir, "a", pinned = True, archived = True)
    # Patch only `archived`; the pin must survive.
    assert flags.set_flags(gdir, "a", archived = False) == {"pinned": True, "archived": False}


def test_toggling_everything_off_removes_the_entry(gdir):
    flags.set_flags(gdir, "a", pinned = True)
    flags.set_flags(gdir, "a", pinned = False)
    # No residue: an id back at its defaults should not keep a row.
    assert flags.read(gdir) == {}


def test_pin_rank_orders_most_recently_pinned_first(gdir):
    flags.set_flags(gdir, "first", pinned = True)
    flags.set_flags(gdir, "second", pinned = True)
    items = flags.read(gdir)
    assert flags.pin_rank(items, "second") > flags.pin_rank(items, "first")
    # An unpinned id must sort behind every pinned one.
    assert flags.pin_rank(items, "unpinned") == float("-inf")


def test_a_coarse_clock_still_orders_two_pins(gdir, monkeypatch):
    # Windows advances time.time() in ~16 ms steps, so two pins a click apart read the same wall
    # clock and the pinned group loses the order the client serializes its PATCHes to preserve.
    import time as _time

    monkeypatch.setattr(_time, "time", lambda: 1000.0)
    flags.set_flags(gdir, "first", pinned = True)
    flags.set_flags(gdir, "second", pinned = True)
    flags.set_flags(gdir, "third", pinned = True)
    items = flags.read(gdir)
    ranks = [flags.pin_rank(items, i) for i in ("first", "second", "third")]
    assert ranks[0] < ranks[1] < ranks[2], ranks
    # Re-pinning an already pinned id still moves it to the front of the group.
    flags.set_flags(gdir, "first", pinned = True)
    items = flags.read(gdir)
    assert flags.pin_rank(items, "first") > flags.pin_rank(items, "third")


def test_a_pin_never_stores_a_non_finite_timestamp(gdir):
    # The monotonic nudge must not be able to manufacture the value _pinned_at refuses. A store
    # holding the largest finite float nudges to infinity, which would read back as unpinned and
    # leave the store untrusted, so the default clear stops working after a successful PATCH.
    import sys

    _store(gdir).write_text(
        json.dumps({"version": 1, "items": {"huge": {"pinned_at": sys.float_info.max}}}),
        encoding = "utf-8",
    )
    assert flags.set_flags(gdir, "a", pinned = True) == {"pinned": True, "archived": False}
    items = flags.read_trusted(gdir)  # must not raise: the store is still readable
    assert flags.flags_for(items, "a")["pinned"] is True
    assert math.isfinite(flags.pin_rank(items, "a"))


@pytest.mark.parametrize(
    "raw",
    [
        "not json at all",
        "[]",  # right type, wrong shape
        '{"version": 1, "items": []}',  # items must be a mapping
        '{"version": 99, "items": {"a": {}}}',  # unknown schema version
    ],
)
def test_a_corrupt_store_degrades_to_no_flags(gdir, raw):
    # Losing a pin beats refusing to list the gallery, so every unreadable store reads empty.
    _store(gdir).write_text(raw, encoding = "utf-8")
    assert flags.read(gdir) == {}


def test_a_corrupt_store_is_overwritten_by_the_next_write(gdir):
    _store(gdir).write_text("garbage", encoding = "utf-8")
    flags.set_flags(gdir, "a", pinned = True)
    items = flags.read(gdir)
    assert set(items) == {"a"}
    assert flags.flags_for(items, "a") == {"pinned": True, "archived": False}


def test_a_non_dict_entry_reads_as_no_flags(gdir):
    _store(gdir).write_text(
        json.dumps({"version": 1, "items": {"a": "hand edited"}}), encoding = "utf-8"
    )
    items = flags.read(gdir)
    assert flags.flags_for(items, "a") == {"pinned": False, "archived": False}


def test_forget_prunes_only_the_named_ids(gdir):
    flags.set_flags(gdir, "keep", pinned = True)
    flags.set_flags(gdir, "drop", archived = True)
    flags.forget(gdir, ["drop", "never-existed"])
    items = flags.read(gdir)
    assert set(items) == {"keep"}


def test_forget_on_an_empty_store_writes_nothing(gdir):
    flags.forget(gdir, ["a"])
    assert not _store(gdir).exists()


def test_writes_leave_no_temp_files_behind(gdir):
    flags.set_flags(gdir, "a", pinned = True)
    flags.forget(gdir, ["a"])
    # The tmp is renamed into place, so only the store (and its lock) may remain.
    leftovers = {p.name for p in gdir.iterdir()} - {".flags.json", ".flags.json.lock"}
    assert leftovers == set()


def test_read_trusted_raises_on_a_corrupt_store(gdir):
    _store(gdir).write_text("garbage", encoding = "utf-8")
    with pytest.raises(flags.FlagsUnavailable):
        flags.read_trusted(gdir)


def test_read_trusted_accepts_a_missing_store(gdir):
    # No store yet genuinely means nothing is flagged, which is safe to act on.
    assert flags.read_trusted(gdir) == {}


def test_set_flags_raises_when_the_store_cannot_be_written(gdir, monkeypatch):
    def _boom(*a, **k):
        raise OSError("read-only filesystem")

    monkeypatch.setattr(flags.os, "replace", _boom)
    with pytest.raises(OSError):
        flags.set_flags(gdir, "a", pinned = True)
    # The failed write leaves no temp behind.
    assert [p.name for p in gdir.iterdir() if p.name.startswith(".flags.json.tmp")] == []


def test_forget_stays_best_effort_when_the_store_cannot_be_written(gdir, monkeypatch):
    flags.set_flags(gdir, "a", pinned = True)
    real = flags.os.replace
    monkeypatch.setattr(flags.os, "replace", lambda *a, **k: (_ for _ in ()).throw(OSError("nope")))
    # The media is already deleted by this point, so a stale row must not raise into the caller.
    flags.forget(gdir, ["a"])
    monkeypatch.setattr(flags.os, "replace", real)


def test_a_corrupt_store_is_replaced_rather_than_blocking_new_flags(gdir):
    # Refusing here would leave the user unable to pin anything until they hand-fixed the file.
    _store(gdir).write_text("[]", encoding = "utf-8")
    flags.set_flags(gdir, "a", archived = True)
    assert flags.is_archived(flags.read(gdir), "a") is True


def test_a_store_rebuilt_from_illegible_contents_stays_untrusted(gdir):
    # The write above must not be blocked, but the file it leaves behind is not evidence: the old
    # contents were never read, so "nothing else is archived" is a guess. Trusting the replacement
    # is what let an unrelated pin hand every previously archived image to the next clear().
    _store(gdir).write_text("[]", encoding = "utf-8")
    flags.set_flags(gdir, "a", archived = True)
    with pytest.raises(flags.FlagsUnavailable):
        flags.read_trusted(gdir)
    # And it stays that way across further writes, rather than being laundered clean by the next one.
    flags.set_flags(gdir, "b", pinned = True)
    with pytest.raises(flags.FlagsUnavailable):
        flags.read_trusted(gdir)


def test_a_malformed_entry_taints_the_whole_store_for_trusted_reads(gdir):
    # One bad value would otherwise be filtered out silently, reading as "this id is not archived",
    # which is enough for clear() to delete an archived file.
    _store(gdir).write_text(
        json.dumps({"version": 1, "items": {"ok": {"archived": True}, "bad": "corrupt"}}),
        encoding = "utf-8",
    )
    with pytest.raises(flags.FlagsUnavailable):
        flags.read_trusted(gdir)
    # The fail-safe reader still degrades quietly, so listing keeps working.
    assert flags.read(gdir) == {"ok": {"archived": True}}


def test_exclusive_serializes_against_set_flags(gdir):
    # clear() decides from a snapshot then unlinks; an archive landing in that window must wait,
    # not slip in and leave the file deleted after its PATCH reported success.
    import threading

    started = threading.Event()
    landed = threading.Event()

    def _archive():
        started.set()
        flags.set_flags(gdir, "a", archived = True)
        landed.set()

    with flags.exclusive(gdir):
        worker = threading.Thread(target = _archive)
        worker.start()
        started.wait(timeout = 5)
        # Held: the writer cannot land while the section is open.
        assert not landed.wait(timeout = 0.5)
    worker.join(timeout = 5)
    assert landed.is_set()
    assert flags.is_archived(flags.read(gdir), "a") is True


def test_forget_locked_does_not_deadlock_inside_exclusive(gdir):
    # The cross-process lock is per descriptor, so a nested forget() would block on the lock its
    # own caller holds. clear() uses forget_locked for exactly this reason.
    flags.set_flags(gdir, "a", pinned = True)
    with flags.exclusive(gdir):
        flags.forget_locked(gdir, ["a"])
    assert flags.read(gdir) == {}


@pytest.mark.parametrize(
    "pinned_at",
    [
        10**400,  # JSON ints are unbounded; this overflows float()
        -(10**400),
        float("nan"),
        float("inf"),
        "2026-01-01",  # wrong type entirely
        True,  # bool is an int subclass, but not a timestamp
    ],
)
def test_an_unusable_pin_time_reads_as_unpinned_instead_of_raising(gdir, pinned_at):
    # These are read on every listing, and the store's contract is to degrade rather than raise.
    _store(gdir).write_text(
        json.dumps({"version": 1, "items": {"a": {"pinned_at": pinned_at}}}), encoding = "utf-8"
    )
    items = flags.read(gdir)
    assert flags.pin_rank(items, "a") == float("-inf")
    assert flags.flags_for(items, "a")["pinned"] is False


def test_an_unusable_pin_time_does_not_hide_the_archived_flag(gdir):
    _store(gdir).write_text(
        json.dumps({"version": 1, "items": {"a": {"pinned_at": 10**400, "archived": True}}}),
        encoding = "utf-8",
    )
    assert flags.is_archived(flags.read(gdir), "a") is True


def test_a_write_repairs_a_store_with_a_malformed_entry(gdir):
    # Merging the bad entry back would leave every later clear() refused until someone fixed the
    # file by hand, which is the opposite of what a pin action should cost the user.
    _store(gdir).write_text(
        json.dumps({"version": 1, "items": {"good": {"archived": True}, "bad": "corrupt"}}),
        encoding = "utf-8",
    )
    flags.set_flags(gdir, "new", pinned = True)
    # Trusted again, so a default clear is no longer blocked.
    items = flags.read_trusted(gdir)
    assert set(items) == {"good", "bad", "new"}
    # The readable flags survived the repair.
    assert flags.is_archived(items, "good") is True
    assert flags.flags_for(items, "new")["pinned"] is True
    # The unreadable one is kept on the archive shelf rather than handed to the next clear().
    assert flags.is_archived(items, "bad") is True


@pytest.mark.parametrize("archived", [None, 1, "yes", []])
def test_a_non_bool_archived_is_refused_rather_than_read_as_active(gdir, archived):
    # Every reader turns a non-bool into "not archived", which is what clear() deletes on, so the
    # store has to refuse instead of handing the file over.
    _store(gdir).write_text(
        json.dumps({"version": 1, "items": {"a": {"archived": archived}}}), encoding = "utf-8"
    )
    with pytest.raises(flags.FlagsUnavailable):
        flags.read_trusted(gdir)


def test_an_unusable_pin_time_also_costs_the_store_its_trust(gdir):
    _store(gdir).write_text(
        json.dumps({"version": 1, "items": {"a": {"pinned_at": 10**400}}}), encoding = "utf-8"
    )
    with pytest.raises(flags.FlagsUnavailable):
        flags.read_trusted(gdir)


def test_a_write_repairs_a_bad_field_without_dropping_the_archive(gdir):
    # Dropping the whole entry over its pin time would hand an archived item to the next clear().
    _store(gdir).write_text(
        json.dumps({"version": 1, "items": {"a": {"pinned_at": 10**400, "archived": True}}}),
        encoding = "utf-8",
    )
    flags.set_flags(gdir, "b", pinned = True)
    items = flags.read_trusted(gdir)
    assert flags.is_archived(items, "a") is True
    assert flags.flags_for(items, "a")["pinned"] is False


def test_a_write_never_repairs_an_archive_into_an_active_item(gdir):
    # Dropping the unreadable flag would leave the store trusted and the item active, so the next
    # default clear() would delete a file that was on the archive shelf. Resolve it the safe way.
    _store(gdir).write_text(
        json.dumps({"version": 1, "items": {"a": {"archived": None}}}), encoding = "utf-8"
    )
    flags.set_flags(gdir, "b", pinned = True)
    items = flags.read_trusted(gdir)
    assert flags.is_archived(items, "a") is True
    assert flags.flags_for(items, "b")["pinned"] is True


def test_an_absent_archived_key_is_not_treated_as_damage(gdir):
    # Unarchiving removes the key, so absent means active and must stay active through a repair.
    _store(gdir).write_text(
        json.dumps({"version": 1, "items": {"a": {"pinned_at": 10**400}}}), encoding = "utf-8"
    )
    flags.set_flags(gdir, "b", pinned = True)
    assert flags.is_archived(flags.read_trusted(gdir), "a") is False


def test_archived_false_is_a_shape_we_write_and_stays_trusted(gdir):
    _store(gdir).write_text(
        json.dumps({"version": 1, "items": {"a": {"archived": False, "pinned_at": 1.0}}}),
        encoding = "utf-8",
    )
    items = flags.read_trusted(gdir)
    assert flags.flags_for(items, "a") == {"pinned": True, "archived": False}


def test_a_filesystem_that_cannot_lock_still_completes_the_write(gdir, monkeypatch):
    # Some network filesystems refuse to lock. Acquisition already tolerated that, but the matching
    # unlock did not, so the store was written and the call still raised, failing a PATCH whose
    # work had landed (and, through clear(), one that had already deleted files).
    # Whichever primitive this platform uses: fcntl does not exist on Windows, and importing it
    # unconditionally failed the test there rather than exercising the branch that runs.
    if os.name == "nt":
        import msvcrt as locking
        primitive = "locking"
    else:
        import fcntl as locking
        primitive = "flock"

    def _unsupported(*_args):
        raise OSError(45, "Operation not supported")

    monkeypatch.setattr(locking, primitive, _unsupported)
    assert flags.set_flags(gdir, "a", pinned = True) == {"pinned": True, "archived": False}
    flags.forget(gdir, ["a"])
    with flags.exclusive(gdir):
        pass
    assert flags.read(gdir) == {}
