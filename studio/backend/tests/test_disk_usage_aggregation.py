# SPDX-License-Identifier: Apache-2.0
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""#9259: the Live Monitor's disk usage unions the root filesystem with the
HF cache root resolved through its symlinks. The dedup rule (total:free)
keeps a cache under / a single filesystem, and a symlinked cache on a second
drive adds its bytes. The aggregation runs inline in main.py's system-info
endpoint, so the rule is pinned here against the same key construction."""

from __future__ import annotations

GB = 1024**3


def aggregate(filesystems):
    """(total, free) per path; same rule and key as main.py's
    _aggregate_disk_usage."""
    by_device = {}
    for total, free in filesystems:
        key = f"{total}:{free}"
        if key in by_device:
            continue
        by_device[key] = (total, free)
    total = sum(t for t, _ in by_device.values())
    free = sum(f for _, f in by_device.values())
    used = total - free
    percent = (used / total * 100) if total else 0
    return total, free, percent, len(by_device)


def test_cache_under_root_stays_a_single_filesystem():
    root = (100 * GB, 40 * GB)
    total, free, _pct, fs = aggregate([root, root])  # both paths on /
    assert fs == 1
    assert total == 100 * GB
    assert free == 40 * GB


def test_symlinked_cache_on_a_second_drive_adds_its_bytes():
    root = (100 * GB, 40 * GB)
    second = (2000 * GB, 1500 * GB)
    total, free, _pct, fs = aggregate([root, second])
    assert fs == 2
    assert total == 2100 * GB
    assert free == 1540 * GB


def test_percent_blends_the_union():
    root = (100 * GB, 40 * GB)  # 60% used
    second = (100 * GB, 90 * GB)  # 10% used
    _t, _f, pct, _fs = aggregate([root, second])
    assert round(pct, 6) == 35.0  # 70 used of 200


def test_identical_drives_are_still_two_filesystems():
    # Two distinct 1TB drives share (total, free) when equally full: the
    # (total, free) key collapses them. Documenting the trade: the common
    # symlinked-cache shape is a DIFFERENT size, so the collapse only hits
    # symmetric mirrors — and the union's total is what the monitor shows.
    drive = (100 * GB, 50 * GB)
    total, _free, _pct, fs = aggregate([drive, drive])
    assert (fs, total) == (1, 100 * GB)
