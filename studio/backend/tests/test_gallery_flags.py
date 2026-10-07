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
    assert flags.set_flags(gdir, "a", archived = False) == {"pinned": True, "archived": False}


def test_toggling_everything_off_removes_the_entry(gdir):
    flags.set_flags(gdir, "a", pinned = True)
    flags.set_flags(gdir, "a", pinned = False)
    # Ids back at their defaults must not leave a row.
    assert flags.read(gdir) == {}


def test_pin_rank_orders_most_recently_pinned_first(gdir):
    flags.set_flags(gdir, "first", pinned = True)
    flags.set_flags(gdir, "second", pinned = True)
    items = flags.read(gdir)
    assert flags.pin_rank(items, "second") > flags.pin_rank(items, "first")
    assert flags.pin_rank(items, "unpinned") == float("-inf")


def test_a_coarse_clock_still_orders_two_pins(gdir, monkeypatch):
    # Windows time.time() has ~16 ms resolution, so back-to-back pins can share a timestamp.
    import time as _time

    monkeypatch.setattr(_time, "time", lambda: 1000.0)
    flags.set_flags(gdir, "first", pinned = True)
    flags.set_flags(gdir, "second", pinned = True)
    flags.set_flags(gdir, "third", pinned = True)
    items = flags.read(gdir)
    ranks = [flags.pin_rank(items, i) for i in ("first", "second", "third")]
    assert ranks[0] < ranks[1] < ranks[2], ranks
    # Re-pinning an already pinned id moves it to the front.
    flags.set_flags(gdir, "first", pinned = True)
    items = flags.read(gdir)
    assert flags.pin_rank(items, "first") > flags.pin_rank(items, "third")


def test_a_pin_never_stores_a_non_finite_timestamp(gdir):
    # Nudging the max finite float overflows to inf, which reads back as unpinned.
    import sys

    _store(gdir).write_text(
        json.dumps({"version": 1, "items": {"huge": {"pinned_at": sys.float_info.max}}}),
        encoding = "utf-8",
    )
    assert flags.set_flags(gdir, "a", pinned = True) == {"pinned": True, "archived": False}
    items = flags.read_trusted(gdir)
    assert flags.flags_for(items, "a")["pinned"] is True
    assert math.isfinite(flags.pin_rank(items, "a"))


@pytest.mark.parametrize(
    "raw",
    [
        "not json at all",
        "[]",
        '{"version": 1, "items": []}',
        '{"version": 99, "items": {"a": {}}}',
    ],
)
def test_a_corrupt_store_degrades_to_no_flags(gdir, raw):
    # Losing a pin beats failing to list the gallery, so unreadable stores read empty.
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
    leftovers = {p.name for p in gdir.iterdir()} - {".flags.json", ".flags.json.lock"}
    assert leftovers == set()


def test_read_trusted_raises_on_a_corrupt_store(gdir):
    _store(gdir).write_text("garbage", encoding = "utf-8")
    with pytest.raises(flags.FlagsUnavailable):
        flags.read_trusted(gdir)


def test_read_trusted_accepts_a_missing_store(gdir):
    assert flags.read_trusted(gdir) == {}


def test_set_flags_raises_when_the_store_cannot_be_written(gdir, monkeypatch):
    def _boom(*a, **k):
        raise OSError("read-only filesystem")

    monkeypatch.setattr(flags.os, "replace", _boom)
    with pytest.raises(OSError):
        flags.set_flags(gdir, "a", pinned = True)
    assert [p.name for p in gdir.iterdir() if p.name.startswith(".flags.json.tmp")] == []


def test_forget_stays_best_effort_when_the_store_cannot_be_written(gdir, monkeypatch):
    flags.set_flags(gdir, "a", pinned = True)
    real = flags.os.replace
    monkeypatch.setattr(flags.os, "replace", lambda *a, **k: (_ for _ in ()).throw(OSError("nope")))
    # Media is already deleted by now, so a stale row must not raise.
    flags.forget(gdir, ["a"])
    monkeypatch.setattr(flags.os, "replace", real)


def test_a_corrupt_store_is_replaced_rather_than_blocking_new_flags(gdir):
    _store(gdir).write_text("[]", encoding = "utf-8")
    flags.set_flags(gdir, "a", archived = True)
    assert flags.is_archived(flags.read(gdir), "a") is True


def test_a_store_rebuilt_from_illegible_contents_stays_untrusted(gdir):
    # Writing over an unread store must not mark it trusted, or clear() could delete archived items.
    _store(gdir).write_text("[]", encoding = "utf-8")
    flags.set_flags(gdir, "a", archived = True)
    with pytest.raises(flags.FlagsUnavailable):
        flags.read_trusted(gdir)
    flags.set_flags(gdir, "b", pinned = True)
    with pytest.raises(flags.FlagsUnavailable):
        flags.read_trusted(gdir)


def test_a_malformed_entry_taints_the_whole_store_for_trusted_reads(gdir):
    # A silently dropped bad value reads as not archived, letting clear() delete the file.
    _store(gdir).write_text(
        json.dumps({"version": 1, "items": {"ok": {"archived": True}, "bad": "corrupt"}}),
        encoding = "utf-8",
    )
    with pytest.raises(flags.FlagsUnavailable):
        flags.read_trusted(gdir)
    assert flags.read(gdir) == {"ok": {"archived": True}}


def test_exclusive_serializes_against_set_flags(gdir):
    # clear() snapshots then unlinks; a concurrent archive must wait, not slip in between.
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
        assert not landed.wait(timeout = 0.5)
    worker.join(timeout = 5)
    assert landed.is_set()
    assert flags.is_archived(flags.read(gdir), "a") is True


def test_forget_locked_does_not_deadlock_inside_exclusive(gdir):
    # The cross-process lock is per descriptor, so nested forget() would self-deadlock; use forget_locked.
    flags.set_flags(gdir, "a", pinned = True)
    with flags.exclusive(gdir):
        flags.forget_locked(gdir, ["a"])
    assert flags.read(gdir) == {}


@pytest.mark.parametrize(
    "pinned_at",
    [
        10**400,  # overflows float()
        -(10**400),
        float("nan"),
        float("inf"),
        "2026-01-01",
        True,  # bool is an int subclass
    ],
)
def test_an_unusable_pin_time_reads_as_unpinned_instead_of_raising(gdir, pinned_at):
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
    # A pin write must repair the bad entry, not merge it back and block every later clear().
    _store(gdir).write_text(
        json.dumps({"version": 1, "items": {"good": {"archived": True}, "bad": "corrupt"}}),
        encoding = "utf-8",
    )
    flags.set_flags(gdir, "new", pinned = True)
    items = flags.read_trusted(gdir)
    assert set(items) == {"good", "bad", "new"}
    assert flags.is_archived(items, "good") is True
    assert flags.flags_for(items, "new")["pinned"] is True
    # Unreadable entries stay archived so clear() never deletes them.
    assert flags.is_archived(items, "bad") is True


@pytest.mark.parametrize("archived", [None, 1, "yes", []])
def test_a_non_bool_archived_is_refused_rather_than_read_as_active(gdir, archived):
    # Readers treat non-bool as not archived (what clear() deletes on), so the store must refuse.
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
    _store(gdir).write_text(
        json.dumps({"version": 1, "items": {"a": {"pinned_at": 10**400, "archived": True}}}),
        encoding = "utf-8",
    )
    flags.set_flags(gdir, "b", pinned = True)
    items = flags.read_trusted(gdir)
    assert flags.is_archived(items, "a") is True
    assert flags.flags_for(items, "a")["pinned"] is False


def test_a_write_never_repairs_an_archive_into_an_active_item(gdir):
    _store(gdir).write_text(
        json.dumps({"version": 1, "items": {"a": {"archived": None}}}), encoding = "utf-8"
    )
    flags.set_flags(gdir, "b", pinned = True)
    items = flags.read_trusted(gdir)
    assert flags.is_archived(items, "a") is True
    assert flags.flags_for(items, "b")["pinned"] is True


def test_an_absent_archived_key_is_not_treated_as_damage(gdir):
    # Unarchiving removes the key, so absent means active.
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
    # Some network filesystems refuse to lock; unlock must tolerate that too.
    # fcntl is absent on Windows, so use the platform's primitive.
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


def _shelf(gdir, mtimes):
    """The shelf in listing order, as the galleries sort it."""
    items = flags.read(gdir)
    pairs = list(mtimes.items())
    pairs.sort(
        key = lambda p: (flags.pin_rank(items, p[0]), flags.order_rank(items, p[0], p[1])),
        reverse = True,
    )
    return [i for i, _ in pairs]


def _move(gdir, mtimes, item_id, after_id):
    with flags.exclusive(gdir):
        items = flags.read(gdir)
        ordered = sorted(
            mtimes.items(),
            key = lambda p: (flags.pin_rank(items, p[0]), flags.order_rank(items, p[0], p[1])),
            reverse = True,
        )
        return flags.place_locked(gdir, item_id, ordered, after_id = after_id)


MTIMES = {"a": 400.0, "b": 300.0, "c": 200.0, "d": 100.0}


def test_an_undragged_shelf_sorts_newest_first(gdir):
    assert _shelf(gdir, MTIMES) == ["a", "b", "c", "d"]


@pytest.mark.parametrize(
    "item, after, expected",
    [
        ("d", None, ["d", "a", "b", "c"]),
        ("a", "d", ["b", "c", "d", "a"]),
        ("a", "b", ["b", "a", "c", "d"]),
        ("d", "a", ["a", "d", "b", "c"]),
        ("b", "c", ["a", "c", "b", "d"]),
    ],
)
def test_a_drag_lands_after_its_neighbour(gdir, item, after, expected):
    assert _move(gdir, MTIMES, item, after) == {"pinned": False, "archived": False}
    assert _shelf(gdir, MTIMES) == expected


def test_only_the_moved_item_is_rewritten(gdir):
    _move(gdir, MTIMES, "d", "a")
    items = flags.read(gdir)
    assert set(items) == {"d"}
    assert 300.0 < flags.order_at(items, "d") < 400.0


def test_new_media_still_lands_first_after_a_drag_to_the_front(gdir):
    _move(gdir, MTIMES, "d", None)
    later = {**MTIMES, "new": flags.order_at(flags.read(gdir), "d") + 1}
    assert _shelf(gdir, later)[:2] == ["new", "d"]


def test_repeated_drops_in_one_gap_keep_their_order(gdir):
    mtimes = dict(MTIMES)
    for _ in range(5):
        _move(gdir, mtimes, "d", "a")
        _move(gdir, mtimes, "c", "a")
    assert _shelf(gdir, mtimes) == ["a", "c", "d", "b"]


def test_dropping_between_pins_pins_and_orders_it(gdir):
    flags.set_flags(gdir, "c", pinned = True)
    flags.set_flags(gdir, "a", pinned = True)
    assert _move(gdir, MTIMES, "d", "a")["pinned"] is True
    assert _shelf(gdir, MTIMES) == ["a", "d", "c", "b"]


def test_dropping_at_the_front_of_a_pinned_shelf_pins_it(gdir):
    flags.set_flags(gdir, "c", pinned = True)
    assert _move(gdir, MTIMES, "b", None)["pinned"] is True
    assert _shelf(gdir, MTIMES)[:2] == ["b", "c"]


def test_dropping_among_unpinned_items_unpins_it(gdir):
    flags.set_flags(gdir, "a", pinned = True)
    assert _move(gdir, MTIMES, "a", "c")["pinned"] is False
    assert _shelf(gdir, MTIMES) == ["b", "c", "a", "d"]


@pytest.mark.parametrize("item, pinned", [("a", True), ("c", False)])
def test_the_seam_between_groups_keeps_the_items_state(gdir, item, pinned):
    flags.set_flags(gdir, "b", pinned = True)
    flags.set_flags(gdir, "a", pinned = True)
    assert _move(gdir, MTIMES, item, "b")["pinned"] is pinned
    shelf = _shelf(gdir, MTIMES)
    assert shelf.index(item) == shelf.index("b") + 1


def test_a_later_pin_still_leads_a_dragged_pin(gdir):
    flags.set_flags(gdir, "c", pinned = True)
    _move(gdir, MTIMES, "d", None)
    flags.set_flags(gdir, "b", pinned = True)
    assert _shelf(gdir, MTIMES)[:3] == ["b", "d", "c"]


def test_an_unknown_neighbour_is_refused(gdir):
    with pytest.raises(KeyError):
        _move(gdir, MTIMES, "a", "gone")
    assert not _store(gdir).exists()


@pytest.mark.parametrize("order_at", ["x", True, None, 10**400])
def test_an_unusable_order_key_reads_as_undragged_and_costs_trust(gdir, order_at):
    _store(gdir).write_text(json.dumps({"version": 1, "items": {"d": {"order_at": order_at}}}))
    items = flags.read(gdir)
    assert flags.order_at(items, "d") is None
    assert _shelf(gdir, MTIMES) == ["a", "b", "c", "d"]
    assert flags.is_trusted(gdir) is False
