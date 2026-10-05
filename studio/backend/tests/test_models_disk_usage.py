# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""#9259: the Live Monitor reports the drive holding the HF cache when it is not the system disk."""

import os
import sys
import threading
import time
from collections import namedtuple
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils import system_disk  # noqa: E402

GB = 10**9
_Usage = namedtuple("_Usage", "total used free")


@pytest.fixture
def second_drive(tmp_path, monkeypatch):
    """Everything under tmp_path/second is drive 2; the rest, `/` included, is drive 1.

    Matched on the literal path, so only a caller that resolved the symlink lands on drive 2."""
    second = (tmp_path / "second").resolve()
    (second / "hf" / "hub").mkdir(parents = True)

    def on_second(path):
        return Path(path).is_relative_to(second)

    def fake_usage(path):
        if on_second(path):
            return _Usage(2000 * GB, 500 * GB, 1500 * GB)
        return _Usage(100 * GB, 60 * GB, 40 * GB)

    def fake_device(path):
        os.stat(path)  # real errors: missing, permission denied
        return 2 if on_second(path) else 1

    monkeypatch.setattr(system_disk, "_device", fake_device)
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


def test_cache_dir_missing_behind_a_symlink_still_finds_the_target_drive(second_drive, tmp_path):
    link = tmp_path / "huggingface"
    link.symlink_to(second_drive / "hf", target_is_directory = True)
    assert system_disk.models_disk_usage(link / "not-created" / "hub")["total_gb"] == 2000.0


def test_cache_on_the_system_disk_adds_nothing(second_drive, tmp_path):
    assert system_disk.models_disk_usage(tmp_path / "hub") is None


def test_cache_not_created_yet_uses_its_parent_volume(second_drive):
    assert system_disk.models_disk_usage(second_drive / "new" / "hub")["total_gb"] == 2000.0


def test_volume_sharing_the_system_pool_adds_nothing(second_drive, monkeypatch):
    # APFS Data volume / btrfs subvolume: own st_dev, same pool as `/`.
    pool = _Usage(2000 * GB, 1200 * GB, 800 * GB)
    shared = _Usage(2000 * GB, 300 * GB, 800 * GB - 5 * 10**6)
    monkeypatch.setattr(
        system_disk.shutil,
        "disk_usage",
        lambda p: shared if Path(p).is_relative_to(second_drive) else pool,
    )
    assert system_disk.models_disk_usage(second_drive / "hf" / "hub") is None


@pytest.mark.skipif(
    not os.environ.get("GITHUB_ACTIONS") or sys.platform == "win32",
    reason = "hosted Linux / macOS runners have one disk (Windows checks out on D:, home on C:)",
)
def test_hosted_runner_home_cache_is_the_system_disk():
    # macos runners put home on the APFS Data volume (own st_dev, same container as `/`).
    assert system_disk.models_disk_usage(Path.home() / ".cache" / "huggingface" / "hub") is None


def test_unreadable_cache_is_omitted(monkeypatch):
    def denied(path):
        raise PermissionError(path)

    monkeypatch.setattr(system_disk, "_device", denied)
    assert system_disk.models_disk_usage(Path("/somewhere/hub")) is None


def test_active_cache_on_root_volume_is_none_on_a_single_disk_host(tmp_path, monkeypatch):
    monkeypatch.setattr(system_disk, "_hub_cache", lambda: Path(os.path.abspath(os.sep)) / "x")
    assert system_disk.models_disk_usage() is None


@pytest.mark.skipif(
    sys.platform == "win32" or os.geteuid() == 0, reason = "needs POSIX permission bits"
)
def test_unreadable_cache_under_a_readable_volume_is_not_its_parent(second_drive):
    # Python 3.14's Path.exists is False for EACCES too; climbing on it reported the parent volume.
    locked = second_drive / "locked"
    (locked / "hub").mkdir(parents = True)
    locked.chmod(0)
    try:
        assert system_disk.models_disk_usage(locked / "hub") is None
    finally:
        locked.chmod(0o755)


@pytest.fixture
def fresh_cache(monkeypatch):
    monkeypatch.setattr(system_disk, "_readings", {})
    monkeypatch.setattr(system_disk, "_probes", {})
    monkeypatch.setattr(system_disk, "_FIRST_WAIT_S", 0.2)


def test_hung_cache_volume_never_blocks_the_poll(fresh_cache, monkeypatch):
    release, calls = threading.Event(), []

    def hung(cache = None):
        calls.append(cache)
        release.wait(10)
        return {"total_gb": 1.0}

    monkeypatch.setattr(system_disk, "_cache_key", lambda: "studio:/nfs")
    monkeypatch.setattr(system_disk, "models_disk_usage", hung)
    for _ in range(5):
        start = time.monotonic()
        assert system_disk.cached_models_disk_usage() is None
        assert time.monotonic() - start < 1.0
    assert len(calls) == 1, "one probe in flight, not one per poll"
    release.set()
    for _ in range(50):
        if not system_disk._probes:
            break
        time.sleep(0.05)
    assert system_disk.cached_models_disk_usage() == {"total_gb": 1.0}


def test_cache_path_is_resolved_off_the_request_thread(fresh_cache, monkeypatch):
    def blocking_resolve():
        raise AssertionError("resolved on the request thread")

    monkeypatch.setattr(system_disk, "_hub_cache", blocking_resolve)
    monkeypatch.setattr(system_disk, "_cache_key", lambda: "studio:/nfs")
    monkeypatch.setattr(system_disk, "models_disk_usage", lambda cache = None: {"ok": True})
    assert system_disk.cached_models_disk_usage() == {"ok": True}


def test_changed_models_folder_is_reprobed(fresh_cache, monkeypatch):
    folder = {"key": "a"}
    monkeypatch.setattr(system_disk, "_cache_key", lambda: folder["key"])
    monkeypatch.setattr(system_disk, "models_disk_usage", lambda cache = None: {"key": folder["key"]})
    assert system_disk.cached_models_disk_usage() == {"key": "a"}
    folder["key"] = "b"
    assert system_disk.cached_models_disk_usage() == {"key": "b"}


def test_switching_away_from_a_hung_folder_probes_the_new_one(fresh_cache, monkeypatch):
    release = threading.Event()
    folder = {"key": "nfs"}

    def probe(cache = None):
        key = folder["key"]
        if key == "nfs":
            release.wait(10)
        return {"key": key}

    monkeypatch.setattr(system_disk, "_cache_key", lambda: folder["key"])
    monkeypatch.setattr(system_disk, "models_disk_usage", probe)
    assert system_disk.cached_models_disk_usage() is None
    folder["key"] = "local"
    assert system_disk.cached_models_disk_usage() == {"key": "local"}
    # The slow probe landing late must not evict the active folder's reading.
    release.set()
    for _ in range(50):
        if not system_disk._probes:
            break
        time.sleep(0.05)
    monkeypatch.setattr(
        system_disk, "models_disk_usage", lambda cache = None: pytest.fail("re-probed")
    )
    assert system_disk.cached_models_disk_usage() == {"key": "local"}
