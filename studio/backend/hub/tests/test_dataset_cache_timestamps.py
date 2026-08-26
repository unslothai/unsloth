# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""S9 for PR #9642: cached datasets need a timestamp for Recent to mean anything.

Before this, /api/hub/datasets/cached carried no time field at all, while local
recipe and upload datasets carried updated_at. Sorting the two together put every
cached Hub dataset below every local one no matter which was newer.
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
    monkeypatch.setattr(cache_inventory, "_raw_dataset_cache_has_data", lambda *_args: True)
    monkeypatch.setattr(cache_inventory, "_scan_hub_dataset_cache_dirs", lambda: [])
    monkeypatch.setattr(cache_inventory, "_scan_processed_dataset_caches", lambda: [])
    monkeypatch.setattr(cache_inventory, "_scan_app_processed_dataset_caches", lambda: [])


def test_schema_carries_the_timestamp():
    assert "last_modified" in CachedDatasetItem.__annotations__
    # Unset rather than 0: the frontend must be able to tell "no readable mtime"
    # from "modified at the epoch", or an unreadable cache sorts as 1970.
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
    # POSIX seconds, the same unit the cached-model scan emits, so one
    # normalizer on the frontend covers both.
    assert rows[0]["last_modified"] == pytest.approx(1_700_000_000.5)


def test_the_key_is_omitted_when_no_mtime_is_readable(monkeypatch):
    # A broken snapshot symlink on Windows, a share with no clock, or a cache
    # deleted mid-scan. The row must still be listed, just undated.
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
    # Both candidates, because the fallback reports the newest of them and a
    # freshly created parent directory would otherwise dominate.
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
