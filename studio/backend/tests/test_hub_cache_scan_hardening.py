# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
import sys
import threading
import types
from types import SimpleNamespace

if "structlog" not in sys.modules:

    class _DummyLogger:
        def __getattr__(self, _name):
            return lambda *args, **kwargs: None

    sys.modules["structlog"] = types.SimpleNamespace(
        BoundLogger = _DummyLogger,
        get_logger = lambda *args, **kwargs: _DummyLogger(),
    )

import routes.datasets as datasets_route
from utils import hf_cache_scan


def test_dataset_inventory_uses_shared_hf_cache_scan(monkeypatch, tmp_path):
    calls = 0
    repo_path = tmp_path / "datasets--Org--Data"
    repo_path.mkdir()
    repo = SimpleNamespace(
        repo_id = "Org/Data",
        repo_type = "dataset",
        repo_path = repo_path,
        size_on_disk = 123,
        revisions = [SimpleNamespace(files = [])],
    )

    def all_hf_cache_scans():
        nonlocal calls
        calls += 1
        return [SimpleNamespace(cache_dir = tmp_path, repos = [repo])]

    monkeypatch.setattr(
        datasets_route.hf_cache_scan, "all_hf_cache_scans", all_hf_cache_scans
    )
    monkeypatch.setattr(
        datasets_route.hf_cache_scan, "partial_repo_ids", lambda *_args: set()
    )
    monkeypatch.setattr(datasets_route, "_scan_hub_dataset_cache_dirs", lambda: [])
    monkeypatch.setattr(datasets_route, "_scan_processed_dataset_caches", lambda: [])

    rows = datasets_route._scan_hf_dataset_caches()

    assert calls == 1
    assert rows == [
        {
            "repo_id": "Org/Data",
            "size_bytes": 123,
            "cache_path": str(repo_path),
            "partial": False,
        }
    ]


def test_delete_cached_dataset_invalidates_shared_scan_cache(monkeypatch):
    invalidations = 0

    def invalidate():
        nonlocal invalidations
        invalidations += 1

    monkeypatch.setattr(
        datasets_route,
        "resolve_cached_repo_id_case",
        lambda repo_id, repo_type: repo_id,
    )
    monkeypatch.setattr(datasets_route._registry, "begin_delete", lambda _key: True)
    monkeypatch.setattr(datasets_route._registry, "end_delete", lambda _key: None)
    monkeypatch.setattr(
        datasets_route,
        "_delete_cached_dataset_blocking",
        lambda repo_id: {"deleted": repo_id},
    )
    monkeypatch.setattr(
        datasets_route.hf_cache_scan, "invalidate_hf_cache_scans", invalidate
    )

    result = asyncio.run(
        datasets_route.delete_cached_dataset("Org/Data", current_subject = "test")
    )

    assert result == {"deleted": "Org/Data"}
    assert invalidations == 1


def test_all_hf_cache_scans_waiters_reraise_owner_failure(monkeypatch):
    error = RuntimeError("scan failed")
    owner_entered = threading.Event()
    release_owner = threading.Event()
    waiter_waiting = threading.Event()
    owner_errors: list[BaseException] = []
    waiter_errors: list[BaseException] = []

    def reset_scan_state():
        with hf_cache_scan._hf_cache_scans_lock:
            hf_cache_scan._hf_cache_scans_flight = None
            hf_cache_scan._hf_cache_scans_result = None
            hf_cache_scan._hf_cache_scans_cached_at = 0.0

    def compute_all_hf_cache_scans():
        owner_entered.set()
        assert release_owner.wait(timeout = 2)
        raise error

    def call_owner():
        try:
            hf_cache_scan.all_hf_cache_scans()
        except BaseException as exc:
            owner_errors.append(exc)

    def call_waiter():
        try:
            hf_cache_scan.all_hf_cache_scans()
        except BaseException as exc:
            waiter_errors.append(exc)

    reset_scan_state()
    monkeypatch.setattr(
        hf_cache_scan, "_compute_all_hf_cache_scans", compute_all_hf_cache_scans
    )

    owner = threading.Thread(target = call_owner)
    owner.start()
    assert owner_entered.wait(timeout = 2)

    with hf_cache_scan._hf_cache_scans_lock:
        flight = hf_cache_scan._hf_cache_scans_flight
    assert flight is not None
    event_wait = flight.event.wait

    def wait_wrapper(*args, **kwargs):
        waiter_waiting.set()
        return event_wait(*args, **kwargs)

    flight.event.wait = wait_wrapper

    waiter = threading.Thread(target = call_waiter)
    waiter.start()
    assert waiter_waiting.wait(timeout = 2)

    release_owner.set()
    owner.join(timeout = 2)
    waiter.join(timeout = 2)
    reset_scan_state()

    assert owner_errors == [error]
    assert waiter_errors == [error]


def test_stall_watchdog_marker_ignores_completed_blobs(monkeypatch, tmp_path):
    root = tmp_path / "hub"
    repo = root / "models--Org--Model"
    blobs = repo / "blobs"
    blobs.mkdir(parents = True)
    (blobs / "completed").write_bytes(b"x" * 1024)
    incomplete = blobs / "active.incomplete"
    incomplete.write_bytes(b"abc")

    monkeypatch.setattr(hf_cache_scan, "_hf_cache_root", lambda create = False: root)

    marker = hf_cache_scan._repo_incomplete_blob_progress_marker("model", "Org/Model")
    assert marker is not None
    assert marker[0] == 1
    assert marker[1] == 3

    incomplete.unlink()
    assert (
        hf_cache_scan._repo_incomplete_blob_progress_marker("model", "Org/Model")
        is None
    )


def test_purge_repo_cache_dirs_removes_config_only_partial_repo(monkeypatch, tmp_path):
    root = tmp_path / "hub"
    repo = root / "models--Org--Partial"
    snapshot = repo / "snapshots" / "rev"
    blobs = repo / "blobs"
    snapshot.mkdir(parents = True)
    blobs.mkdir()
    (snapshot / "config.json").write_text("{}")
    (blobs / "config").write_text("{}")

    monkeypatch.setattr(hf_cache_scan, "_hf_cache_roots", lambda: [root])

    assert hf_cache_scan.purge_repo_cache_dirs("model", "org/partial") is True
    assert not repo.exists()
