# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A custom folder behind a hung mount must not hold the model list hostage."""

import asyncio
import os
import threading

import pytest

from hub.services.models import local_inventory
from utils.paths import scan_folder_health as health
from utils.paths.scan_folder_health import (
    STATUS_OK,
    STATUS_PERMISSION_DENIED,
    STATUS_SLOW,
    annotate_scan_folders,
    refresh_failed_scan_folders,
)


@pytest.fixture(autouse = True)
def _fast_budget(monkeypatch):
    monkeypatch.setattr(local_inventory, "_CUSTOM_FOLDER_SCAN_TIMEOUT_S", 0.2)
    local_inventory._custom_folder_scans.clear()
    health._failed.clear()
    yield
    health._failed.clear()
    local_inventory._custom_folder_scans.clear()


def _status(path: str) -> str:
    return annotate_scan_folders([{"path": path}])[0]["status"]


def test_a_stuck_folder_is_skipped_marked_slow_and_recovers():
    release = threading.Event()
    calls = []

    def stuck():
        calls.append(1)
        release.wait(5)
        return ["row"]

    assert asyncio.run(local_inventory._run_custom_folder_bounded("/mnt/nas", stuck)) is None
    assert _status("/mnt/nas") == STATUS_SLOW
    # Still stuck: the next listing skips it at once instead of parking a second thread on the mount.
    assert local_inventory.run_custom_folder_bounded_sync("/mnt/nas", stuck) is None
    assert len(calls) == 1
    release.set()
    local_inventory._custom_folder_scans["/mnt/nas"][1].result(5)
    assert _status("/mnt/nas") == STATUS_OK
    assert local_inventory.run_custom_folder_bounded_sync("/mnt/nas", lambda: ["row"]) == ["row"]


def test_overlapping_listings_cannot_grow_threads_on_a_hung_folder():
    release = threading.Event()
    started = []

    def stuck():
        started.append(1)
        release.wait(5)

    async def overlapping():
        tasks = []
        for _ in range(8):
            tasks.append(
                asyncio.create_task(local_inventory._run_custom_folder_bounded("/mnt/nas", stuck))
            )
            await asyncio.sleep(0.08)
        return await asyncio.gather(*tasks)

    try:
        assert asyncio.run(overlapping()) == [None] * 8
        # Only requests inside the first budget window start a thread; later ones see it stuck.
        assert len(started) <= 4
    finally:
        release.set()


def test_stuck_folders_cannot_starve_a_healthy_one():
    release = threading.Event()
    try:
        for index in range(6):
            assert (
                local_inventory.run_custom_folder_bounded_sync(
                    f"/mnt/stuck{index}", lambda: release.wait(5)
                )
                is None
            )
        assert local_inventory.run_custom_folder_bounded_sync("/models", lambda: ["m"]) == ["m"]
    finally:
        release.set()


def test_healthy_folders_scan_concurrently_and_raise_as_before():
    async def both():
        return await asyncio.gather(
            local_inventory._run_custom_folder_bounded("/a", lambda: ["a"]),
            local_inventory._run_custom_folder_bounded("/a", lambda: ["a2"]),
        )

    assert asyncio.run(both()) == [["a"], ["a2"]]

    def denied():
        raise PermissionError(13, "denied")

    with pytest.raises(PermissionError):
        local_inventory.run_custom_folder_bounded_sync("/b", denied)


def test_a_finished_slow_scan_keeps_a_failure_it_recorded(monkeypatch):
    health._record("/c", STATUS_SLOW, "/c")
    health._record("/c", STATUS_PERMISSION_DENIED, "/c")
    health.clear_scan_slow("/c")
    assert _status("/c") == STATUS_PERMISSION_DENIED


def test_the_folder_list_does_not_reprobe_a_slow_folder(monkeypatch):
    health._record("/mnt/nas", STATUS_SLOW, "/mnt/nas")

    def hang(*_args, **_kwargs):
        raise AssertionError("probing a hung mount would hang GET /scan-folders")

    monkeypatch.setattr(health, "probe_folder", hang)
    refresh_failed_scan_folders([{"path": "/mnt/nas"}])
    assert _status("/mnt/nas") == STATUS_SLOW


def test_macos_case_variants_are_one_folder(tmp_path, monkeypatch):
    from hub.storage import scan_folders
    from storage import studio_db

    monkeypatch.setattr(scan_folders, "_denied_path_prefixes", lambda: [])
    monkeypatch.setattr(scan_folders.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(scan_folders, "macos_volume_ignores_case", lambda _path: True)
    folder = tmp_path / "Models"
    folder.mkdir()
    real = str(folder.resolve())
    conn = studio_db.get_connection()
    try:
        conn.execute(
            "INSERT INTO scan_folders (path, created_at, recursive) VALUES (?, ?, ?)",
            (real.upper(), "2026-01-01", 1),
        )
        conn.commit()
    finally:
        conn.close()
    row, inserted = scan_folders.add_scan_folder_with_status(real)
    assert not inserted
    assert row["path"] == real.upper() and row["recursive"]
    assert len(scan_folders.list_scan_folders()) == 1


def _block_on(hung: str, release: threading.Event, identity):
    def resolve(raw):
        if str(raw) == hung:
            release.wait(5)
        return identity(raw)

    return resolve


def test_a_hung_folder_cannot_stall_the_hub_folder_list(monkeypatch, tmp_path):
    release = threading.Event()
    healthy = str(tmp_path)
    monkeypatch.setattr(
        local_inventory,
        "list_scan_folders",
        lambda: [{"id": 1, "path": "/mnt/nas"}, {"id": 2, "path": healthy}],
    )
    monkeypatch.setattr(
        local_inventory,
        "_inventory_path_identity",
        _block_on("/mnt/nas", release, local_inventory._inventory_path_identity),
    )
    try:
        folders = asyncio.run(asyncio.wait_for(local_inventory._load_custom_folders(), 2))
        assert [folder["_identity"] for folder in folders] == [
            os.path.normcase("/mnt/nas"),
            local_inventory._inventory_path_identity(healthy),
        ]
        assert _status("/mnt/nas") == STATUS_SLOW
    finally:
        release.set()


def test_a_hung_folder_cannot_stall_the_compat_model_list(monkeypatch, tmp_path):
    from routes import models as models_route

    release = threading.Event()
    sources = models_route._CompatLocalInventorySources(
        tmp_path / "active", tmp_path / "legacy", tmp_path / "default", (), ()
    )
    monkeypatch.setattr(
        "storage.studio_db.list_scan_folders", lambda: [{"id": 1, "path": "/mnt/nas"}]
    )
    monkeypatch.setattr(
        models_route,
        "_compat_inventory_path_identity",
        _block_on("/mnt/nas", release, models_route._compat_inventory_path_identity),
    )
    monkeypatch.setattr(models_route, "collect_local_models", lambda *_args, **_kwargs: [])

    async def listing_and_loop():
        ticks = 0

        async def tick():
            nonlocal ticks
            while True:
                ticks += 1
                await asyncio.sleep(0.01)

        ticker = asyncio.create_task(tick())
        try:
            rows = await asyncio.wait_for(
                models_route._shared_compat_local_inventory_scan(tmp_path / "models", sources), 2
            )
        finally:
            ticker.cancel()
        return rows, ticks

    try:
        rows, ticks = asyncio.run(listing_and_loop())
        assert rows == [] and ticks > 5
        assert _status("/mnt/nas") == STATUS_SLOW
    finally:
        release.set()
