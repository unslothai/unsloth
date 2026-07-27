# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import errno
import os
import sys
import time
import types
from pathlib import Path

import pytest

from hub.utils import dataset_cache, hf_cache_state


@pytest.fixture(autouse = True)
def _known_cache_root(monkeypatch, tmp_path):
    monkeypatch.setattr(hf_cache_state, "hf_cache_roots", lambda: [tmp_path])


def _dataset_repo(root: Path, repo_id: str, snapshot: str = "rev") -> tuple[Path, Path]:
    repo_root = root / f"datasets--{repo_id.replace('/', '--')}"
    snap = repo_root / "snapshots" / snapshot
    snap.mkdir(parents = True)
    return repo_root, snap


def _patch_cache_dirs(monkeypatch, repo_id: str, repo_roots: list[Path]) -> None:
    monkeypatch.setattr(
        dataset_cache,
        "iter_repo_cache_dirs",
        lambda repo_type, requested: iter(repo_roots)
        if repo_type == "dataset" and requested == repo_id
        else iter([]),
    )


def _fake_datasets(monkeypatch):
    calls: list[dict] = []

    class DownloadConfig:
        def __init__(self, *, local_files_only):
            self.local_files_only = local_files_only

    module = types.ModuleType("datasets")
    module.DownloadConfig = DownloadConfig
    module.load_dataset = lambda **kwargs: calls.append(kwargs) or {"loaded": True}
    cache_safe = types.ModuleType("utils.datasets.cache_safe")
    cache_safe.load_dataset_cache_safe = module.load_dataset
    monkeypatch.setitem(sys.modules, "datasets", module)
    monkeypatch.setitem(sys.modules, "utils.datasets.cache_safe", cache_safe)
    return calls


def test_latest_cached_dataset_snapshot_prefers_selected_cache_path(monkeypatch, tmp_path):
    repo_id = "Org/Data"
    selected_root, selected_snap = _dataset_repo(tmp_path, repo_id)
    other_root, other_snap = _dataset_repo(tmp_path / "other", repo_id)
    (selected_snap / "train.parquet").write_bytes(b"selected")
    (other_snap / "train.parquet").write_bytes(b"other")
    _patch_cache_dirs(monkeypatch, repo_id, [other_root])

    assert (
        dataset_cache.latest_cached_dataset_snapshot(repo_id, str(selected_root))
        == selected_snap.resolve()
    )


def test_dataset_snapshot_rejects_foreign_paths(tmp_path):
    foreign = tmp_path / "not-a-cache" / "snapshots" / "rev"
    foreign.mkdir(parents = True)
    (foreign / "train.parquet").write_bytes(b"x")

    assert dataset_cache.dataset_snapshot_from_cache_path(str(foreign), "Org/Data") is None


def test_dataset_snapshot_rejects_lookalike_outside_known_cache(monkeypatch, tmp_path):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    repo_root, _ = _dataset_repo(tmp_path / "outside", "Org/Data")
    monkeypatch.setattr(hf_cache_state, "hf_cache_roots", lambda: [allowed])

    assert (
        dataset_cache.dataset_snapshot_from_cache_path(str(repo_root), "Org/Data")
        is None
    )


def test_dataset_snapshot_ignores_symlinks_outside_cache(tmp_path):
    repo_root, snapshot = _dataset_repo(tmp_path, "Org/Data", "rev")
    escaped = tmp_path / "escaped"
    escaped.mkdir()
    (escaped / "train.parquet").write_bytes(b"x")
    link = repo_root / "snapshots" / "linked"
    link.symlink_to(escaped, target_is_directory = True)
    future = time.time() + 3600
    os.utime(escaped, (future, future))

    assert (
        dataset_cache.dataset_snapshot_from_cache_path(str(repo_root), "Org/Data")
        == snapshot.resolve()
    )


def test_dataset_snapshots_directory_symlink_cannot_escape_cache(tmp_path):
    repo_root = tmp_path / "datasets--Org--Data"
    repo_root.mkdir()
    external = tmp_path / "external-snapshots"
    snapshot = external / "rev"
    snapshot.mkdir(parents = True)
    (snapshot / "train.parquet").write_bytes(b"x")
    (repo_root / "snapshots").symlink_to(external, target_is_directory = True)

    assert dataset_cache.dataset_snapshot_from_cache_path(str(repo_root), "Org/Data") is None


def test_refs_main_preferred_over_newer_mtime_snapshot(monkeypatch, tmp_path):
    repo_id = "Org/Data"
    repo_root, pinned = _dataset_repo(tmp_path, repo_id, "commit-old")
    newer = repo_root / "snapshots" / "commit-new"
    newer.mkdir()
    (pinned / "train.parquet").write_bytes(b"old")
    (newer / "train.parquet").write_bytes(b"new")
    os.utime(pinned, (1_000, 1_000))
    os.utime(newer, (2_000, 2_000))
    refs = repo_root / "refs"
    refs.mkdir()
    (refs / "main").write_text("commit-old")
    _patch_cache_dirs(monkeypatch, repo_id, [repo_root])

    assert dataset_cache.latest_cached_dataset_snapshot(repo_id) == pinned.resolve()

    (refs / "main").write_text("commit-missing")
    assert dataset_cache.latest_cached_dataset_snapshot(repo_id) == newer.resolve()


def test_cached_snapshot_load_preserves_hf_subset_and_split(monkeypatch, tmp_path):
    repo_id = "Org/Data"
    repo_root, snapshot = _dataset_repo(tmp_path, repo_id)
    (snapshot / "README.md").write_text("---\nconfigs: []\n---\n")
    calls = _fake_datasets(monkeypatch)

    result = dataset_cache.load_cached_hf_dataset(
        repo_id,
        str(repo_root),
        subset = "english",
        split = "validation",
        token = "hf_test",
    )

    assert result == {"loaded": True}
    assert len(calls) == 1
    assert calls[0]["path"] == str(snapshot.resolve())
    assert calls[0]["name"] == "english"
    assert calls[0]["split"] == "validation"
    assert calls[0]["token"] == "hf_test"
    assert calls[0]["download_config"].local_files_only is True
    assert "data_files" not in calls[0]


def test_processed_cache_load_uses_selected_cache_root(monkeypatch, tmp_path):
    repo_id = "Org/Data"
    processed_root = tmp_path / "processed"
    processed = processed_root / "Org___Data"
    processed.mkdir(parents = True)
    monkeypatch.setenv("HF_DATASETS_CACHE", str(processed_root))
    calls = _fake_datasets(monkeypatch)

    resolved = dataset_cache.latest_cached_dataset_path(repo_id, str(processed))
    result = dataset_cache.load_cached_hf_dataset(
        repo_id,
        str(resolved),
        subset = None,
        split = "train",
    )

    assert resolved == processed.resolve()
    assert result == {"loaded": True}
    assert calls[0]["path"] == repo_id
    assert calls[0]["split"] == "train"
    assert calls[0]["cache_dir"] == str(processed_root.resolve())
    assert calls[0]["download_config"].local_files_only is True


def test_processed_cache_is_discovered_without_selected_path(monkeypatch, tmp_path):
    repo_id = "Org/Data"
    processed_root = tmp_path / "processed"
    processed = processed_root / "Org___Data"
    processed.mkdir(parents = True)
    monkeypatch.setenv("HF_DATASETS_CACHE", str(processed_root))

    assert dataset_cache.latest_cached_dataset_path(repo_id) == processed.resolve()
    assert (
        dataset_cache.latest_cached_dataset_path(repo_id, str(tmp_path / "stale"))
        == processed.resolve()
    )


def test_processed_cache_rejects_lookalike_outside_known_root(monkeypatch, tmp_path):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    foreign = tmp_path / "foreign" / "Org___Data"
    foreign.mkdir(parents = True)
    monkeypatch.setenv("HF_DATASETS_CACHE", str(allowed))

    assert dataset_cache.processed_dataset_cache_path(str(foreign), "Org/Data") is None


@pytest.mark.parametrize(
    "error",
    [
        FileNotFoundError("missing"),
        PermissionError("unreadable"),
        RuntimeError("safetensor header is corrupt"),
        OSError("Can't load tokenizer for '/cache/snapshot'"),
        OSError(errno.EIO, "I/O error"),
        ConnectionError("Offline mode is enabled"),
    ],
)
def test_cache_artifact_errors_are_retryable(error):
    assert dataset_cache.is_cache_artifact_error(error) is True


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("CUDA out of memory"),
        ValueError("unsupported model architecture"),
        RuntimeError("remote code approval required"),
        TypeError("unexpected keyword argument 'local_files_only'"),
        RuntimeError("this model does not support safetensors"),
        RuntimeError("optimizer state is corrupt"),
        OSError(errno.EMFILE, "too many open files"),
        OSError(errno.ENOMEM, "out of memory"),
    ],
)
def test_non_cache_failures_are_not_retryable(error):
    assert dataset_cache.is_cache_artifact_error(error) is False
