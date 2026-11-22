# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Per-scan memo: lookups repeat inside one inventory scan only, never across scans."""

import asyncio
import os
import sys
import threading
from pathlib import Path

import pytest

_backend = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _backend)

from utils import hf_cache_settings
from utils.paths import storage_roots
from utils.scan_memo import in_scan_memo, memoized, scan_memo


def test_memoized_outside_a_scope_always_computes():
    calls = []
    assert memoized("k", lambda: calls.append(1) or len(calls)) == 1
    assert memoized("k", lambda: calls.append(1) or len(calls)) == 2


def test_memoized_inside_a_scope_computes_once_and_ends_with_it():
    calls = []
    compute = lambda: calls.append(1) or len(calls)
    with scan_memo():
        assert memoized("k", compute) == 1
        assert memoized("k", compute) == 1
        assert memoized("other", compute) == 2
        with scan_memo():
            assert memoized("k", compute) == 1
    assert memoized("k", compute) == 3


def test_a_raise_is_not_cached():
    calls = []

    def compute():
        calls.append(1)
        if len(calls) == 1:
            raise OSError("transient")
        return "ok"

    with scan_memo():
        with pytest.raises(OSError):
            memoized("k", compute)
        assert memoized("k", compute) == "ok"
        assert memoized("k", compute) == "ok"
    assert len(calls) == 2


def test_async_scope_reaches_to_thread_workers_and_closes():
    calls = []

    def lookup():
        return memoized("k", lambda: calls.append(threading.get_ident()) or len(calls))

    @in_scan_memo
    async def scan():
        first = await asyncio.to_thread(lookup)
        second = await asyncio.to_thread(lookup)
        return first, second

    assert asyncio.run(scan()) == (1, 1)
    assert asyncio.run(scan()) == (2, 2)
    assert lookup() == 3


def test_a_thread_that_outlives_the_scope_stops_using_it():
    import contextvars

    calls = []
    compute = lambda: calls.append(1) or len(calls)
    with scan_memo():
        assert memoized("k", compute) == 1
        leaked = contextvars.copy_context()
    # A copied context still holds the dict, as a task or thread started in the scan would.
    assert leaked.run(memoized, "k", compute) == 2
    assert leaked.run(memoized, "k", compute) == 3

    def new_scan():
        with scan_memo():
            return memoized("k", compute), memoized("k", compute)

    # A scan begun from that context gets a fresh memo of its own.
    assert leaked.run(new_scan) == (4, 4)


def test_sync_decorator_keeps_name_and_result():
    @in_scan_memo
    def scan(value):
        return [memoized("k", lambda: value), memoized("k", lambda: value + 1)]

    assert scan.__name__ == "scan"
    assert scan(1) == [1, 1]
    assert scan(5) == [5, 5]


def test_studio_root_follows_env_between_scans(tmp_path, monkeypatch):
    first, second = tmp_path / "a", tmp_path / "b"
    first.mkdir()
    second.mkdir()
    monkeypatch.delenv("STUDIO_HOME", raising = False)
    monkeypatch.delenv("UNSLOTH_HOME", raising = False)
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(first))
    with scan_memo():
        assert storage_roots.studio_root() == first.resolve()
        # Keyed on the value, so a change inside a scan is still seen.
        monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(second))
        assert storage_roots.studio_root() == second.resolve()
    assert storage_roots.studio_root() == second.resolve()


def test_studio_root_symlink_retarget_is_seen_by_the_next_scan(tmp_path, monkeypatch):
    first, second, link = tmp_path / "a", tmp_path / "b", tmp_path / "link"
    first.mkdir()
    second.mkdir()
    try:
        link.symlink_to(first, target_is_directory = True)
    except (OSError, NotImplementedError):
        pytest.skip(
            reason = "creating a symlink needs privileges this host lacks (Windows without Developer Mode)"
        )
    monkeypatch.delenv("STUDIO_HOME", raising = False)
    monkeypatch.delenv("UNSLOTH_HOME", raising = False)
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(link))
    with scan_memo():
        assert storage_roots.studio_root() == first.resolve()
    link.unlink()
    link.symlink_to(second, target_is_directory = True)
    with scan_memo():
        assert storage_roots.studio_root() == second.resolve()
    assert storage_roots.studio_root() == second.resolve()


def test_hf_cache_setting_write_is_seen_by_the_next_scan(monkeypatch, tmp_path):
    stored = {"value": None}
    monkeypatch.setattr(hf_cache_settings, "_EXPLICIT_CACHE_ENV", {})
    monkeypatch.setattr(hf_cache_settings, "_portable_cache_home", lambda: None)
    monkeypatch.setattr(
        hf_cache_settings,
        "_stored_cache_home",
        lambda: Path(stored["value"]) if stored["value"] else None,
    )
    reads = []
    real = hf_cache_settings._get_hf_cache_paths_uncached
    monkeypatch.setattr(
        hf_cache_settings,
        "_get_hf_cache_paths_uncached",
        lambda: reads.append(1) or real(),
    )
    with scan_memo():
        default = hf_cache_settings.get_hf_cache_paths()
        assert hf_cache_settings.get_hf_cache_paths() is default
    assert len(reads) == 1
    stored["value"] = str(tmp_path / "custom")
    assert hf_cache_settings.get_hf_cache_paths().cache_home == tmp_path / "custom"
    with scan_memo():
        assert hf_cache_settings.get_hf_cache_paths().cache_home == tmp_path / "custom"


def test_a_cache_invalidation_mid_scan_is_seen_by_the_same_scan(monkeypatch, tmp_path):
    from hub.utils import inventory_scan

    stored = {"value": None}
    monkeypatch.setattr(hf_cache_settings, "_EXPLICIT_CACHE_ENV", {})
    monkeypatch.setattr(hf_cache_settings, "_portable_cache_home", lambda: None)
    monkeypatch.setattr(
        hf_cache_settings,
        "_stored_cache_home",
        lambda: Path(stored["value"]) if stored["value"] else None,
    )
    with scan_memo():
        before = hf_cache_settings.get_hf_cache_paths()
        stored["value"] = str(tmp_path / "custom")
        assert hf_cache_settings.get_hf_cache_paths() is before
        inventory_scan.invalidate_hf_cache_scans()
        assert hf_cache_settings.get_hf_cache_paths().cache_home == tmp_path / "custom"


def test_known_hub_caches_returns_a_fresh_list_per_call(monkeypatch, tmp_path):
    monkeypatch.setattr(hf_cache_settings, "_known_hf_hub_caches", lambda: [tmp_path])
    with scan_memo():
        first = hf_cache_settings.known_hf_hub_caches()
        first.append("mutated")
        assert hf_cache_settings.known_hf_hub_caches() == [tmp_path]


def test_registered_model_root_sees_a_new_scan_folder_on_the_next_scan(tmp_path, monkeypatch):
    from storage import studio_db
    from utils.models import model_config

    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(studio_db, "_schema_ready", set())
    # macOS tmp_path lives under /private/var, which the scan-folder denylist refuses.
    monkeypatch.setattr(studio_db, "_denied_path_prefixes", lambda: [])
    folder = tmp_path / "models"
    (folder / "m").mkdir(parents = True)
    target = str(folder / "m")
    with scan_memo():
        assert model_config._registered_custom_model_root(target) is None
        studio_db.add_scan_folder(str(folder))
        # Inside one scan the folder list is a snapshot.
        assert model_config._registered_custom_model_root(target) is None
    assert model_config._registered_custom_model_root(target) == Path(os.path.abspath(folder))
    with scan_memo():
        assert model_config._registered_custom_model_root(target) == Path(os.path.abspath(folder))
    studio_db.close_wal_keeper()
