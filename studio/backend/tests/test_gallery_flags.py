# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unit tests for the gallery pin/archive flag store: patch semantics, the fail-safe read,
atomic writes and orphan pruning."""

from __future__ import annotations

import json

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
