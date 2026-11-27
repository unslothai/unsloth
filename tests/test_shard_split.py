# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""tests/_shared/shard_split.py: the parts of a split shard are disjoint, cover every test, keep xdist groups whole,
and do not depend on collection order."""

import os
import random
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "_shared"))
import shard_split  # noqa: E402


class _Marker:
    def __init__(self, name):
        self.args, self.kwargs = (name,), {}


class _Item:
    def __init__(
        self,
        nodeid,
        group = None,
    ):
        self.nodeid, self._group = nodeid, group

    def get_closest_marker(self, name):
        return _Marker(self._group) if name == "xdist_group" and self._group else None


class _Hook:
    def __init__(self):
        self.deselected = []

    def pytest_deselected(self, items):
        self.deselected.extend(items)


class _Config:
    def __init__(self):
        self.hook = _Hook()


def _items():
    items = [_Item(f"tests/test_a_gloo.py::test_case[{i}]") for i in range(23)]
    items += [_Item(f"tests/test_b_gloo.py::test_shared_{i}@b", group = "b") for i in range(7)]
    items += [_Item(f"tests/test_c_gloo.py::test_{i}") for i in range(11)]
    return items


def _run(
    items,
    spec,
    monkeypatch,
    durations = None,
):
    monkeypatch.setenv(shard_split.ENV, spec)
    monkeypatch.setattr(shard_split, "_durations", lambda: {} if durations is None else durations)
    config, kept = _Config(), list(items)
    shard_split.select(config, kept)
    return kept, config.hook.deselected


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_the_parts_are_disjoint_and_cover_every_test(n, monkeypatch):
    items = _items()
    seen = []
    for k in range(1, n + 1):
        kept, dropped = _run(items, f"{k}/{n}", monkeypatch)
        assert len(kept) + len(dropped) == len(items)
        seen.extend(item.nodeid for item in kept)
    assert sorted(seen) == sorted(item.nodeid for item in items)


def test_an_xdist_group_stays_in_one_part(monkeypatch):
    items = _items()
    owners = {
        k
        for k in range(1, 4)
        for item in _run(items, f"{k}/3", monkeypatch)[0]
        if item.get_closest_marker("xdist_group")
    }
    assert len(owners) == 1


def test_the_parts_are_balanced_and_spread_one_tests_parametrizations(monkeypatch):
    items = _items()
    sizes = [len(_run(items, f"{k}/3", monkeypatch)[0]) for k in range(1, 4)]
    assert max(sizes) - min(sizes) <= 7, sizes  # one whole group of 7 at most
    for k in range(1, 4):
        assert any("test_case[" in item.nodeid for item in _run(items, f"{k}/3", monkeypatch)[0])


def test_collection_order_does_not_change_the_parts(monkeypatch):
    items = _items()
    shuffled = list(items)
    random.Random(0).shuffle(shuffled)
    for k in range(1, 4):
        a = sorted(item.nodeid for item in _run(items, f"{k}/3", monkeypatch)[0])
        b = sorted(item.nodeid for item in _run(shuffled, f"{k}/3", monkeypatch)[0])
        assert a == b


def test_unset_keeps_everything(monkeypatch):
    monkeypatch.delenv(shard_split.ENV, raising = False)
    items = _items()
    kept = list(items)
    shard_split.select(_Config(), kept)
    assert kept == items


def test_measured_seconds_balance_the_parts(monkeypatch):
    """One 100 s test gets a part to itself; the 1 s tests share the other."""
    items = [_Item("tests/test_a_gloo.py::test_slow")] + [
        _Item(f"tests/test_a_gloo.py::test_fast[{i}]") for i in range(30)
    ]
    durations = {"tests/test_a_gloo.py": {"*": 1, "test_slow": 100}}
    first = [item.nodeid for item in _run(items, "1/2", monkeypatch, durations)[0]]
    assert first == ["tests/test_a_gloo.py::test_slow"]


def test_the_checked_in_weights_load():
    durations = shard_split._durations()
    assert durations, shard_split.DURATIONS
    for path, seconds in durations.items():
        assert path.startswith("tests/") and path.endswith(".py"), path
        assert "*" in seconds and all(isinstance(v, int) and v >= 1 for v in seconds.values()), path


@pytest.mark.parametrize("spec", ["0/2", "3/2", "two/3", "1"])
def test_a_bad_spec_is_refused(spec):
    with pytest.raises(ValueError):
        shard_split.parse(spec)
