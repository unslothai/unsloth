# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The cache walk must size exactly what the original per-entry walk sized, and a slow
cache must not hold the Settings page once it has been measured."""

import os
import sys
import threading
import time
from pathlib import Path
from typing import Iterable, Optional

import pytest

from utils import cache_inventory
from utils.cache_inventory import LinkLedger, definition_for


def _reference_measure_tree(path: Path, ledger: LinkLedger) -> int:
    """The walk as it was before it was tuned, kept here as the oracle."""
    count = 0
    stack = [path]
    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as entries:
                for entry in entries:
                    count += 1
                    if entry.is_symlink():
                        continue
                    if entry.is_dir(follow_symlinks = False):
                        if cache_inventory._descendable(entry):
                            stack.append(Path(entry.path))
                        continue
                    cache_inventory._record_entry(entry, ledger)
        except OSError:
            continue
    return count


def _reference_measure_root(root: Path, patterns: Optional[Iterable[str]] = None) -> tuple:
    ledger = LinkLedger()
    entries = 0
    try:
        if root.is_symlink() or cache_inventory._is_junction(root) or not root.is_dir():
            return 0, 0
    except OSError:
        return 0, 0
    try:
        with os.scandir(root) as scan:
            for entry in scan:
                if not cache_inventory._matching(entry.name, patterns):
                    continue
                entries += 1
                if entry.is_symlink():
                    continue
                if entry.is_dir(follow_symlinks = False):
                    if cache_inventory._descendable(entry):
                        _reference_measure_tree(Path(entry.path), ledger)
                    continue
                cache_inventory._record_entry(entry, ledger)
    except OSError:
        pass
    return ledger.freeable_bytes(), entries


def _blob(path: Path, size: int) -> Path:
    path.parent.mkdir(parents = True, exist_ok = True)
    path.write_bytes(b"x" * size)
    return path


@pytest.fixture
def awkward_tree(tmp_path):
    """Nested dirs, symlinks in and out, hard links in and out, and dirs the walk cannot read."""
    root = tmp_path / "cache"
    outside = tmp_path / "outside"
    _blob(outside / "big.bin", 1 << 20)
    _blob(root / "top.bin", 11)
    for archive in range(4):
        for depth in range(3):
            nested = (
                root / "archive-v0" / f"a{archive}" / Path(*[f"d{d}" for d in range(depth + 1)])
            )
            _blob(nested / f"f{archive}_{depth}.py", 100 + archive * 10 + depth)
    os.symlink(outside, root / "archive-v0" / "linked_dir")
    os.symlink(outside / "big.bin", root / "archive-v0" / "a0" / "linked_file.bin")
    os.symlink(root / "missing", root / "dangling")
    # Both links inside: counted once. One link outside: not reclaimable, so not counted.
    inner = _blob(root / "wheels" / "pkg" / "inner.bin", 5000)
    os.link(inner, root / "archive-v0" / "a1" / "inner_link.bin")
    shared = _blob(root / "wheels" / "pkg" / "shared.bin", 7000)
    os.link(shared, outside / "venv_link.bin")
    # A three-link inode with two links in two different top-level dirs and one outside.
    triple = _blob(root / "builds" / "t.bin", 3000)
    os.link(triple, root / "wheels" / "t_link.bin")
    os.link(triple, outside / "t_out.bin")
    # Unlistable, and listable but not searchable.
    _blob(root / "locked" / "hidden.bin", 900)
    _blob(root / "noexec" / "child.bin", 800)
    _blob(root / "archive-v0" / "a2" / "deep_locked" / "deeper" / "x.bin", 700)
    os.chmod(root / "locked", 0)
    os.chmod(root / "noexec", 0o600)
    os.chmod(root / "archive-v0" / "a2" / "deep_locked", 0)
    yield root
    for path in (root / "locked", root / "noexec", root / "archive-v0" / "a2" / "deep_locked"):
        os.chmod(path, 0o755)


@pytest.mark.skipif(sys.platform == "win32", reason = "symlinks and chmod 0 need POSIX semantics")
@pytest.mark.parametrize("patterns", [None, ("archive-v0", "wheels"), ("*.bin",)])
def test_the_walk_sizes_exactly_what_the_reference_walk_sizes(awkward_tree, patterns):
    expected = _reference_measure_root(awkward_tree, patterns)
    assert cache_inventory._measure_root(awkward_tree, patterns = patterns) == expected
    if patterns is None:
        # Sanity on the oracle itself: the outside-linked and unreadable bytes are not counted.
        nested = sum(100 + a * 10 + d for a in range(4) for d in range(3))
        # Root bypasses permission checks (capabilities(7) CAP_DAC_OVERRIDE), so it reads the locked dirs.
        locked = 900 + 800 + 700 if getattr(os, "geteuid", lambda: -1)() == 0 else 0
        assert expected == (11 + nested + 5000 + locked, 7)


@pytest.mark.skipif(sys.platform == "win32", reason = "symlinks need POSIX semantics")
def test_the_tree_walk_counts_entries_like_the_reference(awkward_tree):
    target = awkward_tree / "archive-v0"
    reference, tuned = LinkLedger(), LinkLedger()
    assert cache_inventory._measure_tree(target, tuned) == _reference_measure_tree(
        target, reference
    )
    assert tuned.freeable_bytes() == reference.freeable_bytes()


def test_the_walk_matches_the_reference_on_a_portable_tree(tmp_path):
    """Runs on Windows too, where the awkward tree's symlinks and chmod are unavailable."""
    root = tmp_path / "cache"
    outside = tmp_path / "outside"
    _blob(root / "top.bin", 11)
    for archive in range(3):
        _blob(root / f"a{archive}" / "d0" / "d1" / f"f{archive}.py", 100 + archive)
    inner = _blob(root / "wheels" / "inner.bin", 5000)
    os.link(inner, root / "a1" / "inner_link.bin")
    shared = _blob(root / "wheels" / "shared.bin", 7000)
    _blob(outside / "keep.txt", 1)
    os.link(shared, outside / "venv_link.bin")
    for patterns in (None, ("a*",), ("*.bin",)):
        assert cache_inventory._measure_root(root, patterns = patterns) == _reference_measure_root(
            root, patterns
        )


def test_a_junction_below_the_top_level_is_still_not_walked(tmp_path, monkeypatch):
    root = tmp_path / "cache"
    _blob(root / "a" / "b" / "junction" / "elsewhere.bin", 4096)
    _blob(root / "a" / "kept.bin", 3)
    monkeypatch.setattr(cache_inventory, "_is_junction", lambda path: Path(path).name == "junction")
    assert cache_inventory._measure_root(root) == (3, 1)


@pytest.fixture
def clean_memo(monkeypatch):
    monkeypatch.setattr(cache_inventory, "_size_cache", {})
    monkeypatch.setattr(cache_inventory, "_size_epochs", {})
    monkeypatch.setattr(cache_inventory, "_size_flights", {})
    # A background walk an earlier test left running (a real cache walk under load) must not hold these up.
    monkeypatch.setattr(cache_inventory, "_background_walk_lock", threading.Lock())


def _expire(key: str, *, walk_seconds: float) -> None:
    begun, _stored, entry, *_rest = cache_inventory._size_cache[key]
    past = time.monotonic() - cache_inventory._INVENTORY_TTL_SECONDS - 1
    cache_inventory._size_cache[key] = (past - walk_seconds, past, entry, walk_seconds)


def test_an_expired_slow_cache_answers_at_once_and_refreshes_behind(monkeypatch, clean_memo):
    definition = definition_for("uv")
    sizes = iter([10, 20])
    release = threading.Event()
    walked = threading.Event()

    def describe(target):
        size = next(sizes)
        if size == 20:
            walked.set()
            release.wait(10)
        return {"key": target.key, "size_bytes": size}

    monkeypatch.setattr(cache_inventory, "describe_cache", describe)
    assert cache_inventory._described(definition, refresh = False)["size_bytes"] == 10
    _expire("uv", walk_seconds = cache_inventory._BACKGROUND_WALK_SECONDS + 5)

    started = time.monotonic()
    assert cache_inventory._described(definition, refresh = False)["size_bytes"] == 10
    assert time.monotonic() - started < 1.0
    assert walked.wait(5), "no background walk was started"
    # A second read while that walk runs neither waits nor starts another.
    assert cache_inventory._described(definition, refresh = False)["size_bytes"] == 10
    release.set()
    for _ in range(100):
        if cache_inventory._size_cache["uv"][2]["size_bytes"] == 20:
            break
        time.sleep(0.05)
    assert cache_inventory._described(definition, refresh = False)["size_bytes"] == 20


def test_a_studio_download_or_delete_is_never_answered_from_the_older_walk(monkeypatch, clean_memo):
    """Downloads and deletes call invalidate_hf_cache_scans; Settings must then walk, as base did after the TTL."""
    from hub.utils.inventory_scan import invalidate_hf_cache_scans
    for key in ("hf_hub", "hf_xet", "hf_datasets", "hf_assets"):
        definition = definition_for(key)
        sizes = iter([100, 999])
        monkeypatch.setattr(
            cache_inventory,
            "describe_cache",
            lambda target, sizes = sizes: {"key": target.key, "size_bytes": next(sizes)},
        )
        assert cache_inventory._described(definition, refresh = False)["size_bytes"] == 100
        _expire(key, walk_seconds = cache_inventory._BACKGROUND_WALK_SECONDS + 5)
        invalidate_hf_cache_scans()
        assert cache_inventory._described(definition, refresh = False)["size_bytes"] == 999, key


def test_an_expired_quick_cache_is_still_walked_in_the_request(monkeypatch, clean_memo):
    definition = definition_for("pip")
    sizes = iter([10, 20])
    monkeypatch.setattr(
        cache_inventory,
        "describe_cache",
        lambda target: {"key": target.key, "size_bytes": next(sizes)},
    )
    assert cache_inventory._described(definition, refresh = False)["size_bytes"] == 10
    _expire("pip", walk_seconds = 0.01)
    assert cache_inventory._described(definition, refresh = False)["size_bytes"] == 20


def test_recheck_waits_for_a_fresh_walk_even_for_a_slow_cache(monkeypatch, clean_memo):
    definition = definition_for("triton")
    sizes = iter([10, 20])
    monkeypatch.setattr(
        cache_inventory,
        "describe_cache",
        lambda target: {"key": target.key, "size_bytes": next(sizes)},
    )
    assert cache_inventory._described(definition, refresh = False)["size_bytes"] == 10
    _expire("triton", walk_seconds = 60)
    assert cache_inventory._described(definition, refresh = True)["size_bytes"] == 20


def test_a_background_walk_overtaken_by_a_purge_does_not_store_its_size(monkeypatch, clean_memo):
    definition = definition_for("uv")
    sizes = iter([10, 20, 0])
    walked = threading.Event()
    release = threading.Event()

    def describe(target):
        size = next(sizes)
        if size == 20:
            walked.set()
            release.wait(10)
        return {"key": target.key, "size_bytes": size}

    monkeypatch.setattr(cache_inventory, "describe_cache", describe)
    cache_inventory._described(definition, refresh = False)
    _expire("uv", walk_seconds = 30)
    assert cache_inventory._described(definition, refresh = False)["size_bytes"] == 10
    assert walked.wait(5)
    cache_inventory.invalidate_cache_size("uv")
    release.set()
    # The purge's own read waits for the stale walk, then walks again rather than keep 20.
    assert cache_inventory._described(definition, refresh = False)["size_bytes"] == 0


def test_background_walks_run_one_at_a_time(monkeypatch, clean_memo):
    running: list = []
    peak: list = []
    lock = threading.Lock()
    first_pass = {"uv": True, "triton": True}

    def describe(target):
        if first_pass.pop(target.key, False):
            return {"key": target.key, "size_bytes": 1}
        with lock:
            running.append(target.key)
            peak.append(len(running))
        time.sleep(0.2)
        with lock:
            running.remove(target.key)
        return {"key": target.key, "size_bytes": 2}

    monkeypatch.setattr(cache_inventory, "describe_cache", describe)
    for key in ("uv", "triton"):
        cache_inventory._described(definition_for(key), refresh = False)
        _expire(key, walk_seconds = 30)
    for key in ("uv", "triton"):
        assert cache_inventory._described(definition_for(key), refresh = False)["size_bytes"] == 1
    for _ in range(100):
        if all(cache_inventory._size_cache[key][2]["size_bytes"] == 2 for key in ("uv", "triton")):
            break
        time.sleep(0.05)
    assert peak == [1, 1]


def test_the_inventory_starts_background_walks_after_its_own(monkeypatch, clean_memo):
    order: list = []
    real_refresh = cache_inventory._refresh_in_background
    monkeypatch.setattr(
        cache_inventory,
        "describe_cache",
        lambda target: (
            order.append(("walk", target.key)),
            {
                "key": target.key,
                "size_bytes": 1,
                "present": False,
                "purgeable": False,
                "opt_in": False,
            },
        )[1],
    )
    monkeypatch.setattr(cache_inventory, "_free_bytes", lambda: None)
    monkeypatch.setattr(cache_inventory, "_total_disk_bytes", lambda: None)
    cache_inventory.cache_inventory()
    for definition in cache_inventory.CACHE_DEFINITIONS:
        _expire(definition.key, walk_seconds = 30 if definition.key == "uv" else 0.01)
    order.clear()
    monkeypatch.setattr(
        cache_inventory,
        "_refresh_in_background",
        lambda definition: (order.append(("background", definition.key)), real_refresh(definition)),
    )
    cache_inventory.cache_inventory()
    for _ in range(100):
        if ("walk", "uv") in order:
            break
        time.sleep(0.05)
    foreground = [key for kind, key in order if kind == "walk" and key != "uv"]
    assert len(foreground) == len(cache_inventory.CACHE_DEFINITIONS) - 1
    assert order.index(("background", "uv")) == len(foreground)
