# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""#9259: Live Monitor disk usage covers a HF cache symlinked onto another drive."""

import os
import sys
from collections import namedtuple
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils import system_disk  # noqa: E402

GB = 1024**3
_Usage = namedtuple("_Usage", "total used free")


@pytest.fixture
def second_drive(tmp_path, monkeypatch):
    """`/` and tmp_path are drive 1; tmp_path/second is drive 2. Drive 1's free shrinks per read,
    as it does while a download writes."""
    second = tmp_path / "second"
    (second / "hf").mkdir(parents = True)
    reads = {"root": 0}

    def on_second(path):
        return Path(path).resolve().is_relative_to(second.resolve())

    def fake_usage(path):
        if on_second(path):
            return _Usage(2000 * GB, 500 * GB, 1500 * GB)
        reads["root"] += 1
        return _Usage(100 * GB, 60 * GB, 40 * GB - reads["root"])

    monkeypatch.setattr(system_disk, "_device", lambda p: 2 if on_second(p) else 1)
    monkeypatch.setattr(system_disk.shutil, "disk_usage", fake_usage)
    return second


def test_symlinked_cache_on_second_drive_adds_its_bytes(second_drive, tmp_path):
    link = tmp_path / "home_cache"
    link.symlink_to(second_drive / "hf", target_is_directory = True)
    usage = system_disk.system_disk_usage([link])
    assert usage.filesystems == 2
    assert usage.total == 2100 * GB
    assert usage.free == 1540 * GB - 1


def test_cache_on_root_volume_is_counted_once(second_drive, tmp_path):
    # Keyed on (total, free) this double counted: free moved between the two reads.
    usage = system_disk.system_disk_usage([tmp_path])
    assert usage.filesystems == 1
    assert usage.total == 100 * GB


def test_missing_cache_dir_uses_its_parent_volume(second_drive):
    usage = system_disk.system_disk_usage([second_drive / "not" / "created"])
    assert usage.filesystems == 2


def test_unreadable_cache_keeps_the_root_reading(monkeypatch):
    real = system_disk._device
    root = Path(os.path.abspath(os.sep))

    def flaky(path):
        if Path(path) != root:
            raise PermissionError(path)
        return real(path)

    monkeypatch.setattr(system_disk, "_device", flaky)
    assert system_disk.system_disk_usage(["/definitely/elsewhere"]).filesystems == 1


def test_single_disk_matches_psutil():
    psutil = pytest.importorskip("psutil")
    root = os.path.abspath(os.sep)
    usage = system_disk.system_disk_usage([root])
    expected = psutil.disk_usage(root)
    assert (usage.filesystems, usage.total) == (1, expected.total)
    assert abs(usage.percent - expected.percent) < 0.5
