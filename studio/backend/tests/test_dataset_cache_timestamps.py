# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""S9 for PR #9642: cached datasets need a timestamp for Recent to mean anything.

/api/hub/datasets/cached carried no time field, while local recipe and upload
datasets carried updated_at, so every cached Hub dataset sorted below every
local one whatever the date.

Deliberately not beside its subject in hub/tests/: studio-backend-ci runs
`pytest tests/` from studio/backend, and hub/tests is a sibling of that path, so
nothing there is collected. A guard that never runs is not a guard.
"""

import os
from types import SimpleNamespace

import pytest

from hub.schemas.datasets import CachedDatasetItem
from hub.services.datasets import cache_inventory


def _stub_hf_scan(monkeypatch, repos):
    monkeypatch.setattr(
        cache_inventory,
        "_collect_hf_cache_scans",
        lambda: ([SimpleNamespace(repos = repos)], {"/cache"}),
    )
    monkeypatch.setattr(
        cache_inventory.hf_cache_scan,
        "is_snapshot_partial",
        lambda *_args, **_kwargs: False,
    )
    monkeypatch.setattr(
        cache_inventory,
        "_raw_dataset_cache_has_data",
        lambda *_args: True,
    )
    monkeypatch.setattr(cache_inventory, "_scan_hub_dataset_cache_dirs", lambda: [])
    monkeypatch.setattr(cache_inventory, "_scan_processed_dataset_caches", lambda: [])
    monkeypatch.setattr(cache_inventory, "_scan_app_processed_dataset_caches", lambda: [])


def test_schema_carries_the_timestamp():
    assert "last_modified" in CachedDatasetItem.__annotations__
    # Unset rather than 0, or an unreadable cache sorts as 1970.
    assert CachedDatasetItem(repo_id = "Org/Data").last_modified is None


def test_hf_scan_reports_the_repo_timestamp_in_seconds(monkeypatch):
    _stub_hf_scan(
        monkeypatch,
        [
            SimpleNamespace(
                repo_id = "Org/Data",
                repo_type = "dataset",
                repo_path = "/cache/datasets--Org--Data",
                size_on_disk = 100,
                last_modified = 1_700_000_000.5,
                revisions = [SimpleNamespace(files = [], commit_hash = "abc")],
            )
        ],
    )

    rows = cache_inventory._scan_hf_dataset_caches()

    assert len(rows) == 1
    # POSIX seconds, as the cached-model scan emits, so one normalizer covers both.
    assert rows[0]["last_modified"] == pytest.approx(1_700_000_000.5)


def test_the_key_is_omitted_when_no_mtime_is_readable(monkeypatch):
    # Broken symlink, clockless share, or a cache deleted mid-scan. Still listed,
    # just undated.
    _stub_hf_scan(
        monkeypatch,
        [
            SimpleNamespace(
                repo_id = "Org/Data",
                repo_type = "dataset",
                repo_path = "/definitely/not/on/disk/datasets--Org--Data",
                size_on_disk = 100,
                revisions = [SimpleNamespace(files = [], commit_hash = "abc")],
            )
        ],
    )

    rows = cache_inventory._scan_hf_dataset_caches()

    assert len(rows) == 1
    assert "last_modified" not in rows[0]


def test_a_non_positive_mtime_is_dropped_rather_than_reported_as_1970(monkeypatch):
    _stub_hf_scan(
        monkeypatch,
        [
            SimpleNamespace(
                repo_id = "Org/Data",
                repo_type = "dataset",
                repo_path = "/definitely/not/on/disk/datasets--Org--Data",
                size_on_disk = 100,
                last_modified = 0.0,
                revisions = [SimpleNamespace(files = [], commit_hash = "abc")],
            )
        ],
    )

    assert "last_modified" not in cache_inventory._scan_hf_dataset_caches()[0]


def test_falls_back_to_stat_when_the_library_reports_nothing(monkeypatch, tmp_path):
    cache_dir = tmp_path / "datasets--Org--Data"
    (cache_dir / "snapshots").mkdir(parents = True)
    # Both candidates: the fallback takes the newest, or a freshly created parent
    # directory would dominate.
    os.utime(cache_dir / "snapshots", (1_700_000_000, 1_700_000_000))
    os.utime(cache_dir, (1_690_000_000, 1_690_000_000))

    _stub_hf_scan(
        monkeypatch,
        [
            SimpleNamespace(
                repo_id = "Org/Data",
                repo_type = "dataset",
                repo_path = str(cache_dir),
                size_on_disk = 100,
                revisions = [SimpleNamespace(files = [], commit_hash = "abc")],
            )
        ],
    )

    rows = cache_inventory._scan_hf_dataset_caches()

    assert rows[0]["last_modified"] == pytest.approx(1_700_000_000, abs = 2)


def test_a_merge_keeps_the_newer_of_the_two_timestamps(monkeypatch):
    _stub_hf_scan(
        monkeypatch,
        [
            SimpleNamespace(
                repo_id = "Org/Data",
                repo_type = "dataset",
                repo_path = "/cache/datasets--Org--Data",
                size_on_disk = 100,
                last_modified = 1_700_000_000.0,
                revisions = [SimpleNamespace(files = [], commit_hash = "abc")],
            )
        ],
    )
    # The processed-cache scan describes the same dataset, more recently touched.
    monkeypatch.setattr(
        cache_inventory,
        "_scan_processed_dataset_caches",
        lambda: [
            {
                "repo_id": "org/data",
                "size_bytes": 250,
                "cache_path": "/processed/org___data",
                "processed_cache": True,
                "partial": False,
                "last_modified": 1_800_000_000.0,
            }
        ],
    )

    rows = cache_inventory._scan_hf_dataset_caches()

    assert rows[0]["last_modified"] == pytest.approx(1_800_000_000.0)


def test_a_merge_does_not_lose_a_timestamp_the_other_row_lacks(monkeypatch):
    _stub_hf_scan(
        monkeypatch,
        [
            SimpleNamespace(
                repo_id = "Org/Data",
                repo_type = "dataset",
                repo_path = "/cache/datasets--Org--Data",
                size_on_disk = 100,
                last_modified = 1_700_000_000.0,
                revisions = [SimpleNamespace(files = [], commit_hash = "abc")],
            )
        ],
    )
    monkeypatch.setattr(
        cache_inventory,
        "_scan_processed_dataset_caches",
        lambda: [
            {
                "repo_id": "org/data",
                "size_bytes": 250,
                "cache_path": "/processed/org___data",
                "processed_cache": True,
                "partial": False,
            }
        ],
    )

    rows = cache_inventory._scan_hf_dataset_caches()

    assert rows[0]["last_modified"] == pytest.approx(1_700_000_000.0)


def test_hf_scan_keeps_a_newer_timestamp_from_a_smaller_duplicate(monkeypatch):
    def repo(size, last_modified):
        return SimpleNamespace(
            repo_id = "Org/Data",
            repo_type = "dataset",
            repo_path = f"/cache-{size}/datasets--Org--Data",
            size_on_disk = size,
            last_modified = last_modified,
            revisions = [SimpleNamespace(files = [], commit_hash = str(size))],
        )

    _stub_hf_scan(
        monkeypatch,
        [repo(200, 1_700_000_000.0), repo(100, 1_900_000_000.0)],
    )

    rows = cache_inventory._scan_hf_dataset_caches()

    assert rows[0]["size_bytes"] == 200
    assert rows[0]["last_modified"] == pytest.approx(1_900_000_000.0)


def test_fallback_scan_keeps_a_newer_timestamp_from_a_smaller_duplicate(monkeypatch, tmp_path):
    larger_root = tmp_path / "larger"
    newer_root = tmp_path / "newer"
    for root, modified in (
        (larger_root, 1_700_000_000),
        (newer_root, 1_900_000_000),
    ):
        cache_dir = root / "datasets--Org--Data"
        snapshots = cache_dir / "snapshots"
        snapshots.mkdir(parents = True)
        os.utime(cache_dir, (modified, modified))
        os.utime(snapshots, (modified, modified))

    monkeypatch.setattr(
        cache_inventory,
        "_hf_hub_cache_roots",
        lambda: [larger_root, newer_root],
    )
    monkeypatch.setattr(
        cache_inventory,
        "_directory_stats",
        lambda path: (200 if "larger" in path.parts else 100, 0.0),
    )
    monkeypatch.setattr(
        cache_inventory,
        "_hub_dataset_snapshot_count",
        lambda _path: 1,
    )
    monkeypatch.setattr(
        cache_inventory.hf_cache_scan,
        "is_snapshot_partial",
        lambda *_args: False,
    )
    monkeypatch.setattr(cache_inventory, "_raw_dataset_cache_has_data", lambda *_args: True)

    rows = cache_inventory._scan_hub_dataset_cache_dirs()

    assert rows[0]["size_bytes"] == 200
    assert rows[0]["last_modified"] == pytest.approx(1_900_000_000.0)


def test_fallback_scan_uses_the_newest_payload_mtime(monkeypatch, tmp_path):
    root = tmp_path / "hub"
    cache_dir = root / "datasets--Org--Data"
    snapshots = cache_dir / "snapshots"
    blobs = cache_dir / "blobs"
    snapshots.mkdir(parents = True)
    blobs.mkdir()
    payload = blobs / "sha256"
    payload.write_bytes(b"payload")
    os.utime(payload, (1_900_000_000, 1_900_000_000))
    os.utime(snapshots, (1_700_000_000, 1_700_000_000))
    os.utime(cache_dir, (1_700_000_000, 1_700_000_000))

    monkeypatch.setattr(cache_inventory, "_hf_hub_cache_roots", lambda: [root])
    monkeypatch.setattr(
        cache_inventory,
        "_hub_dataset_snapshot_count",
        lambda _path: 1,
    )
    monkeypatch.setattr(
        cache_inventory.hf_cache_scan,
        "is_snapshot_partial",
        lambda *_args: False,
    )
    monkeypatch.setattr(
        cache_inventory,
        "_raw_dataset_cache_has_data",
        lambda *_args: True,
    )

    rows = cache_inventory._scan_hub_dataset_cache_dirs()

    assert rows[0]["last_modified"] == pytest.approx(1_900_000_000.0)


def test_processed_scan_keeps_a_newer_timestamp_from_a_smaller_duplicate(monkeypatch, tmp_path):
    larger_root = tmp_path / "larger-processed"
    newer_root = tmp_path / "newer-processed"
    for root, size, modified in (
        (larger_root, 200, 1_700_000_000),
        (newer_root, 100, 1_900_000_000),
    ):
        cache_dir = root / "Org___Data"
        cache_dir.mkdir(parents = True)
        (cache_dir / "data.arrow").write_bytes(b"x" * size)
        os.utime(cache_dir, (modified, modified))

    monkeypatch.setattr(
        cache_inventory,
        "_hf_datasets_cache_roots",
        lambda: [larger_root, newer_root],
    )
    monkeypatch.setattr(
        cache_inventory,
        "processed_dataset_cache_has_artifacts",
        lambda _path: True,
    )

    rows = cache_inventory._scan_processed_dataset_caches()

    assert rows[0]["size_bytes"] == 200
    assert rows[0]["last_modified"] == pytest.approx(1_900_000_000.0)


def test_processed_scan_uses_the_newest_nested_artifact_mtime(monkeypatch, tmp_path):
    root = tmp_path / "processed"
    cache_dir = root / "Org___Data"
    build_dir = cache_dir / "default" / "1.0.0" / "build"
    build_dir.mkdir(parents = True)
    artifact = build_dir / "data.arrow"
    artifact.write_bytes(b"payload")
    os.utime(artifact, (1_900_000_000, 1_900_000_000))
    os.utime(cache_dir, (1_700_000_000, 1_700_000_000))

    monkeypatch.setattr(
        cache_inventory,
        "_hf_datasets_cache_roots",
        lambda: [root],
    )
    monkeypatch.setattr(
        cache_inventory,
        "processed_dataset_cache_has_artifacts",
        lambda _path: True,
    )

    rows = cache_inventory._scan_processed_dataset_caches()

    assert rows[0]["last_modified"] == pytest.approx(1_900_000_000.0)


def test_recent_order_is_now_derivable_from_the_payload(monkeypatch):
    # The whole point: two cached datasets, and the newer one sorts first.
    _stub_hf_scan(
        monkeypatch,
        [
            SimpleNamespace(
                repo_id = "Org/Older",
                repo_type = "dataset",
                repo_path = "/cache/datasets--Org--Older",
                size_on_disk = 100,
                last_modified = 1_700_000_000.0,
                revisions = [SimpleNamespace(files = [], commit_hash = "a")],
            ),
            SimpleNamespace(
                repo_id = "Org/Newer",
                repo_type = "dataset",
                repo_path = "/cache/datasets--Org--Newer",
                size_on_disk = 100,
                last_modified = 1_900_000_000.0,
                revisions = [SimpleNamespace(files = [], commit_hash = "b")],
            ),
        ],
    )

    rows = cache_inventory._scan_hf_dataset_caches()
    by_recent = sorted(rows, key = lambda row: -(row.get("last_modified") or 0.0))

    assert [row["repo_id"] for row in by_recent] == ["Org/Newer", "Org/Older"]
