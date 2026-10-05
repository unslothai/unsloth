# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""#9259: the Live Monitor reports the drive holding the HF cache when it is not the system disk."""

import os
import sys
from collections import namedtuple
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils import system_disk  # noqa: E402

GB = 10**9
_Usage = namedtuple("_Usage", "total used free")


@pytest.fixture
def second_drive(tmp_path, monkeypatch):
    """Everything under tmp_path/second is drive 2; the rest, `/` included, is drive 1."""
    second = (tmp_path / "second").resolve()
    (second / "hf" / "hub").mkdir(parents = True)

    def on_second(path):
        return Path(path).resolve().is_relative_to(second)

    def fake_usage(path):
        assert on_second(path), f"measured {path}, not the models drive"
        return _Usage(2000 * GB, 500 * GB, 1500 * GB)

    monkeypatch.setattr(system_disk, "_device", lambda p: 2 if on_second(p) else 1)
    monkeypatch.setattr(system_disk.shutil, "disk_usage", fake_usage)
    return second


def test_symlinked_cache_reports_the_drive_holding_the_bytes(second_drive, tmp_path):
    link = tmp_path / "huggingface"
    link.symlink_to(second_drive / "hf", target_is_directory = True)
    assert system_disk.models_disk_usage(link / "hub") == {
        "total_gb": 2000.0,
        "free_gb": 1500.0,
        "percent_used": 25.0,
    }


def test_cache_on_the_system_disk_adds_nothing(second_drive, tmp_path):
    assert system_disk.models_disk_usage(tmp_path / "hub") is None


def test_cache_not_created_yet_uses_its_parent_volume(second_drive):
    assert system_disk.models_disk_usage(second_drive / "new" / "hub")["total_gb"] == 2000.0


def test_unreadable_cache_is_omitted(monkeypatch):
    def denied(path):
        raise PermissionError(path)

    monkeypatch.setattr(system_disk, "_device", denied)
    assert system_disk.models_disk_usage(Path("/somewhere/hub")) is None


def test_active_cache_on_root_volume_is_none_on_a_single_disk_host(tmp_path, monkeypatch):
    monkeypatch.setattr(system_disk, "_hub_cache", lambda: Path(os.path.abspath(os.sep)) / "x")
    assert system_disk.models_disk_usage() is None
