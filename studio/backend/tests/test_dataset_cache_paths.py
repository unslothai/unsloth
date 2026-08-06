# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import errno
import json
import os
import sys
import time
import types
from pathlib import Path

import pytest

from hub.utils import (
    dataset_cache,
    dataset_processed_cache,
    download_manifest,
    hf_cache_state,
    state_dir,
)


@pytest.fixture(autouse = True)
def _known_cache_root(monkeypatch, tmp_path):
    monkeypatch.setattr(hf_cache_state, "hf_cache_roots", lambda: [tmp_path])
    monkeypatch.setattr(
        "utils.paths.storage_roots.cache_root",
        lambda: tmp_path / "app-cache",
    )


def _dataset_repo(
    root: Path,
    repo_id: str,
    snapshot: str = "rev",
) -> tuple[Path, Path]:
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

    assert dataset_cache.dataset_snapshot_from_cache_path(str(repo_root), "Org/Data") is None


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
    assert calls[0]["cache_dir"].startswith(
        str(tmp_path / "app-cache" / "hf-datasets" / "snapshot-loads")
    )
    assert "streaming" not in calls[0]
    assert "data_files" not in calls[0]


def test_cached_snapshot_row_limit_streams_without_processed_cache(monkeypatch, tmp_path):
    repo_id = "Org/Data"
    repo_root, snapshot = _dataset_repo(tmp_path, repo_id)
    (snapshot / "train.parquet").write_bytes(b"rows")
    calls: list[dict] = []
    take_limits: list[int] = []
    materialized: list[tuple[list[dict], object, object, object]] = []
    features = object()
    split_identity = object()

    class SplitInfo:
        def __init__(self, *, name):
            self.name = name

    class SplitDict(dict):
        pass

    copied_info = types.SimpleNamespace(splits = None)

    class DownloadConfig:
        def __init__(self, *, local_files_only):
            self.local_files_only = local_files_only

    class Stream:
        def __init__(self):
            self.features = features
            self.info = types.SimpleNamespace(copy = lambda: copied_info)
            self.split = split_identity

        def take(self, limit):
            take_limits.append(limit)
            return iter([{"text": "one"}, {"text": "two"}, {"text": "three"}][:limit])

    class Dataset:
        @classmethod
        def from_list(
            cls,
            rows,
            *,
            features = None,
            info = None,
            split = None,
        ):
            materialized.append((rows, features, info, split))
            return {"rows": rows}

    def load_dataset(**kwargs):
        assert Path(kwargs["cache_dir"]).is_dir()
        calls.append(kwargs)
        return {"train": Stream(), "validation": object()}

    module = types.ModuleType("datasets")
    module.Dataset = Dataset
    module.DownloadConfig = DownloadConfig
    module.SplitDict = SplitDict
    module.SplitInfo = SplitInfo
    module.load_dataset = load_dataset
    monkeypatch.setitem(sys.modules, "datasets", module)
    monkeypatch.setattr(
        dataset_cache,
        "prepare_app_processed_dataset_cache",
        lambda *_args, **_kwargs: pytest.fail(
            "bounded snapshot loads must not prepare Arrow cache"
        ),
    )

    result = dataset_cache.load_cached_hf_dataset(
        repo_id,
        str(repo_root),
        subset = "english",
        split = "train",
        token = "hf_test",
        row_limit = 2,
    )

    assert result == {"rows": [{"text": "one"}, {"text": "two"}]}
    assert take_limits == [2]
    assert materialized == [
        ([{"text": "one"}, {"text": "two"}], features, copied_info, split_identity)
    ]
    assert calls[0]["path"] == str(snapshot.resolve())
    assert calls[0]["name"] == "english"
    assert "split" not in calls[0]
    assert calls[0]["token"] == "hf_test"
    assert calls[0]["streaming"] is True
    assert calls[0]["download_config"].local_files_only is True
    assert not Path(calls[0]["cache_dir"]).exists()
    assert list(copied_info.splits) == ["train", "validation"]
    assert copied_info.splits["train"].name == "train"
    assert not (tmp_path / "app-cache").exists()


def test_cached_snapshot_row_limit_preserves_empty_stream_schema(monkeypatch, tmp_path):
    repo_id = "Org/Data"
    repo_root, snapshot = _dataset_repo(tmp_path, repo_id)
    (snapshot / "train.parquet").write_bytes(b"rows")
    features = {"text": object(), "score": object()}
    materialized: list[dict] = []

    class DownloadConfig:
        def __init__(self, *, local_files_only):
            self.local_files_only = local_files_only

    class Stream:
        info = None
        split = "train"

        def __init__(self):
            self.features = features

        def take(self, limit):
            return iter(())

    class Dataset:
        @classmethod
        def from_dict(cls, mapping, **kwargs):
            materialized.append({"mapping": mapping, **kwargs})
            return {"rows": []}

        @classmethod
        def from_list(cls, rows, **kwargs):
            pytest.fail("empty typed streams must use Dataset.from_dict")

    module = types.ModuleType("datasets")
    module.Dataset = Dataset
    module.DownloadConfig = DownloadConfig
    module.load_dataset = lambda **kwargs: {"train": Stream()}
    monkeypatch.setitem(sys.modules, "datasets", module)

    result = dataset_cache.load_cached_hf_dataset(
        repo_id,
        str(repo_root),
        subset = None,
        split = "train",
        row_limit = 2,
    )

    assert result == {"rows": []}
    assert materialized == [
        {
            "mapping": {"text": [], "score": []},
            "features": features,
            "info": None,
            "split": "train",
        }
    ]


def test_cached_snapshot_row_limit_keeps_bracketed_split_eager(monkeypatch, tmp_path):
    repo_id = "Org/Data"
    repo_root, _ = _dataset_repo(tmp_path, repo_id)
    calls = _fake_datasets(monkeypatch)

    result = dataset_cache.load_cached_hf_dataset(
        repo_id,
        str(repo_root),
        subset = None,
        split = "train[:2]",
        row_limit = 2,
    )

    assert result == {"loaded": True}
    assert "streaming" not in calls[0]
    assert calls[0]["cache_dir"].startswith(
        str(tmp_path / "app-cache" / "hf-datasets" / "snapshot-loads")
    )


def test_cached_snapshot_row_limit_reports_missing_split_for_hub_fallback(monkeypatch, tmp_path):
    repo_id = "Org/Data"
    repo_root, snapshot = _dataset_repo(tmp_path, repo_id)
    (snapshot / "train.parquet").write_bytes(b"rows")
    calls: list[dict] = []

    class DownloadConfig:
        def __init__(self, *, local_files_only):
            self.local_files_only = local_files_only

    def load_dataset(**kwargs):
        calls.append(kwargs)
        return {"train": object()}

    module = types.ModuleType("datasets")
    module.DownloadConfig = DownloadConfig
    module.load_dataset = load_dataset
    monkeypatch.setitem(sys.modules, "datasets", module)

    with pytest.raises(ValueError) as raised:
        dataset_cache.load_cached_hf_dataset(
            repo_id,
            str(repo_root),
            subset = None,
            split = "validation",
            row_limit = 2,
        )

    assert str(raised.value) == "Unknown split \"validation\". Should be one of ['train']."
    assert (
        dataset_cache.dataset_cache_fallback_allowed(
            raised.value,
            require_exact = False,
            revision = "dataset-commit",
        )
        is True
    )
    assert not Path(calls[0]["cache_dir"]).exists()


@pytest.mark.parametrize("row_limit", [0, -1, True])
def test_cached_dataset_row_limit_must_be_positive_integer(row_limit):
    with pytest.raises(ValueError, match = "row_limit must be a positive integer"):
        dataset_cache.load_cached_hf_dataset(
            "Org/Data",
            None,
            subset = None,
            split = "train",
            row_limit = row_limit,
        )


def _metadata_manifest(
    repo_id: str,
    hub_cache: Path,
    commit_hash: str,
    expected_files: list[download_manifest.ExpectedFile],
    *,
    version: int = 2,
    metadata_derived: bool = True,
) -> download_manifest.Manifest:
    return download_manifest.Manifest(
        repo_type = "dataset",
        repo_id = repo_id,
        variant = None,
        started_at = "",
        expected_files = tuple(expected_files),
        hub_cache = str(hub_cache.resolve()),
        version = version,
        commit_hash = commit_hash if metadata_derived else None,
        metadata_derived = metadata_derived,
    )


def test_complete_dataset_snapshot_requires_exact_metadata_commit(monkeypatch, tmp_path):
    repo_id = "Org/Data"
    repo_root, snapshot = _dataset_repo(tmp_path, repo_id, "commit-a")
    payload = b"rows"
    (snapshot / "train.parquet").write_bytes(payload)
    manifest = _metadata_manifest(
        repo_id,
        tmp_path,
        snapshot.name,
        [download_manifest.ExpectedFile("train.parquet", len(payload))],
    )
    monkeypatch.setattr(
        download_manifest,
        "read_dataset_completion",
        lambda *_args, **_kwargs: manifest,
    )

    assert (
        dataset_cache.complete_dataset_snapshot_path(str(snapshot), repo_id) == snapshot.resolve()
    )

    mismatched = _metadata_manifest(
        repo_id,
        tmp_path,
        "commit-b",
        [download_manifest.ExpectedFile("train.parquet", len(payload))],
    )
    monkeypatch.setattr(
        download_manifest,
        "read_dataset_completion",
        lambda *_args, **_kwargs: mismatched,
    )
    assert dataset_cache.complete_dataset_snapshot_path(str(snapshot), repo_id) is None


@pytest.mark.parametrize(
    ("version", "metadata_derived"),
    [(1, False), (2, False)],
)
def test_legacy_or_disk_derived_manifest_cannot_attest_dataset(
    monkeypatch, tmp_path, version, metadata_derived
):
    repo_id = "Org/Data"
    _, snapshot = _dataset_repo(tmp_path, repo_id, "commit-a")
    (snapshot / "train.parquet").write_bytes(b"rows")
    manifest = _metadata_manifest(
        repo_id,
        tmp_path,
        snapshot.name,
        [download_manifest.ExpectedFile("train.parquet", 4)],
        version = version,
        metadata_derived = metadata_derived,
    )
    monkeypatch.setattr(
        download_manifest,
        "read_dataset_completion",
        lambda *_args, **_kwargs: manifest,
    )

    assert dataset_cache.complete_dataset_snapshot_path(str(snapshot), repo_id) is None


def test_complete_dataset_snapshot_rejects_external_file_symlink(monkeypatch, tmp_path):
    repo_id = "Org/Data"
    _, snapshot = _dataset_repo(tmp_path, repo_id, "commit-a")
    external = tmp_path / "external.parquet"
    external.write_bytes(b"rows")
    (snapshot / "train.parquet").symlink_to(external)
    manifest = _metadata_manifest(
        repo_id,
        tmp_path,
        snapshot.name,
        [download_manifest.ExpectedFile("train.parquet", 4)],
    )
    monkeypatch.setattr(
        download_manifest,
        "read_dataset_completion",
        lambda *_args, **_kwargs: manifest,
    )

    assert dataset_cache.complete_dataset_snapshot_path(str(snapshot), repo_id) is None


def test_complete_dataset_snapshot_rejects_cross_snapshot_symlink(monkeypatch, tmp_path):
    repo_id = "Org/Data"
    _, snapshot = _dataset_repo(tmp_path, repo_id, "commit-a")
    _, other_snapshot = _dataset_repo(tmp_path, repo_id, "commit-b")
    (other_snapshot / "train.parquet").write_bytes(b"rows")
    (snapshot / "train.parquet").symlink_to(os.path.join(os.pardir, "commit-b", "train.parquet"))
    manifest = _metadata_manifest(
        repo_id,
        tmp_path,
        snapshot.name,
        [download_manifest.ExpectedFile("train.parquet", 4)],
    )
    monkeypatch.setattr(
        download_manifest,
        "read_dataset_completion",
        lambda *_args, **_kwargs: manifest,
    )

    assert dataset_cache.complete_dataset_snapshot_path(str(snapshot), repo_id) is None


def test_complete_dataset_snapshot_accepts_hub_blob_symlink(monkeypatch, tmp_path):
    repo_id = "Org/Data"
    repo_root, snapshot = _dataset_repo(tmp_path, repo_id, "commit-a")
    blob = repo_root / "blobs" / "blob"
    blob.parent.mkdir()
    blob.write_bytes(b"rows")
    (snapshot / "train.parquet").symlink_to(os.path.relpath(blob, snapshot))
    manifest = _metadata_manifest(
        repo_id,
        tmp_path,
        snapshot.name,
        [download_manifest.ExpectedFile("train.parquet", 4)],
    )
    monkeypatch.setattr(
        download_manifest,
        "read_dataset_completion",
        lambda *_args, **_kwargs: manifest,
    )

    assert (
        dataset_cache.complete_dataset_snapshot_path(str(snapshot), repo_id) == snapshot.resolve()
    )


def test_newer_download_preserves_older_complete_snapshot(monkeypatch, tmp_path):
    repo_id = "Org/Data"
    repo_root, older = _dataset_repo(tmp_path, repo_id, "commit-old")
    newer = repo_root / "snapshots" / "commit-new"
    newer.mkdir()
    for snapshot in (older, newer):
        (snapshot / "train.parquet").write_bytes(b"rows")
    monkeypatch.setattr(state_dir, "cache_root", lambda: tmp_path / "state")
    monkeypatch.setattr(
        "utils.hf_cache_settings.get_hf_cache_paths",
        lambda: types.SimpleNamespace(hub_cache = tmp_path),
    )
    expected = [download_manifest.ExpectedFile("train.parquet", 4)]

    assert download_manifest.write_dataset_completion(
        repo_id,
        older.name,
        expected,
        hub_cache = tmp_path,
    )
    assert download_manifest.write_manifest(
        "dataset",
        repo_id,
        None,
        expected,
        commit_hash = older.name,
        metadata_derived = True,
        hub_cache = tmp_path,
    )
    assert download_manifest.write_dataset_completion(
        repo_id,
        newer.name,
        expected,
        hub_cache = tmp_path,
    )
    assert download_manifest.write_manifest(
        "dataset",
        repo_id,
        None,
        expected,
        commit_hash = newer.name,
        metadata_derived = True,
        hub_cache = tmp_path,
    )

    assert dataset_cache.complete_dataset_snapshot_path(str(older), repo_id) == older.resolve()
    assert dataset_cache.complete_dataset_snapshot_path(str(newer), repo_id) == newer.resolve()


def test_dataset_completion_isolated_and_purged_by_hub_cache(monkeypatch, tmp_path):
    repo_id = "Org/Data"
    cache_a = tmp_path / "cache-a"
    cache_b = tmp_path / "cache-b"
    _, snapshot_a = _dataset_repo(cache_a, repo_id, "same-commit")
    _, snapshot_b = _dataset_repo(cache_b, repo_id, "same-commit")
    (snapshot_a / "train.parquet").write_bytes(b"four")
    (snapshot_b / "train.parquet").write_bytes(b"five!")
    monkeypatch.setattr(state_dir, "cache_root", lambda: tmp_path / "state")
    monkeypatch.setattr(hf_cache_state, "hf_cache_roots", lambda: [cache_a, cache_b])
    monkeypatch.setattr(
        "utils.hf_cache_settings.get_hf_cache_paths",
        lambda: types.SimpleNamespace(hub_cache = cache_a),
    )

    assert download_manifest.write_dataset_completion(
        repo_id,
        snapshot_a.name,
        [download_manifest.ExpectedFile("train.parquet", 4)],
        hub_cache = cache_a,
    )
    assert download_manifest.write_dataset_completion(
        repo_id,
        snapshot_b.name,
        [download_manifest.ExpectedFile("train.parquet", 5)],
        hub_cache = cache_b,
    )
    assert (
        dataset_cache.complete_dataset_snapshot_path(str(snapshot_a), repo_id)
        == snapshot_a.resolve()
    )
    assert (
        dataset_cache.complete_dataset_snapshot_path(str(snapshot_b), repo_id)
        == snapshot_b.resolve()
    )

    assert (
        download_manifest.purge_all_state_for_repo(
            "dataset",
            repo_id,
            hub_cache = cache_a,
        )
        > 0
    )
    assert dataset_cache.complete_dataset_snapshot_path(str(snapshot_a), repo_id) is None
    assert (
        dataset_cache.complete_dataset_snapshot_path(str(snapshot_b), repo_id)
        == snapshot_b.resolve()
    )


def test_dataset_completion_cache_ownership_uses_platform_case_rules(monkeypatch, tmp_path):
    hub_cache = tmp_path / "Case-Sensitive-Input"
    monkeypatch.setattr(state_dir, "cache_root", lambda: tmp_path / "state")
    monkeypatch.setattr(
        download_manifest.os.path,
        "normcase",
        lambda value: str(value).casefold(),
    )

    assert download_manifest.write_dataset_completion(
        "Org/Data",
        "commit-a",
        [download_manifest.ExpectedFile("train.parquet", 4)],
        hub_cache = hub_cache,
    )
    assert (
        download_manifest.read_dataset_completion(
            "Org/Data",
            "commit-a",
            hub_cache = tmp_path / "case-sensitive-input",
        )
        is not None
    )


def test_dataset_completion_bounds_long_state_filenames(monkeypatch, tmp_path):
    hub_cache = tmp_path / "hub"
    hub_cache.mkdir()
    repo_id = f"{'a' * 96}/{'b' * 96}"
    commit_hash = "c" * 240
    monkeypatch.setattr(state_dir, "cache_root", lambda: tmp_path / "state")

    assert download_manifest.write_dataset_completion(
        repo_id,
        commit_hash,
        [download_manifest.ExpectedFile("train.parquet", 1)],
        hub_cache = hub_cache,
    )
    paths = list((tmp_path / "state" / "hub-state" / "manifests").rglob("*.json"))

    assert len(paths) == 1
    assert len(paths[0].name.encode("utf-8")) <= 255
    assert len(f".{paths[0].name}.tmp-00000000".encode("utf-8")) <= 255
    assert (
        download_manifest.read_dataset_completion(
            repo_id,
            commit_hash,
            hub_cache = hub_cache,
        )
        is not None
    )


@pytest.mark.parametrize(
    "payload",
    [
        "{",
        '{"version":999,"repo_type":"dataset","repo_id":"Org/Data"}',
    ],
)
def test_dataset_completion_corrupt_schema_fails_closed(monkeypatch, tmp_path, payload):
    hub_cache = tmp_path / "hub"
    hub_cache.mkdir()
    monkeypatch.setattr(state_dir, "cache_root", lambda: tmp_path / "state")
    assert download_manifest.write_dataset_completion(
        "Org/Data",
        "commit-a",
        [download_manifest.ExpectedFile("train.parquet", 1)],
        hub_cache = hub_cache,
    )
    paths = list((tmp_path / "state" / "hub-state" / "manifests").rglob("*.json"))
    assert len(paths) == 1
    paths[0].write_text(payload, encoding = "utf-8")

    assert (
        download_manifest.read_dataset_completion(
            "Org/Data",
            "commit-a",
            hub_cache = hub_cache,
        )
        is None
    )


def test_dataset_completion_rejects_boolean_file_size(monkeypatch, tmp_path):
    hub_cache = tmp_path / "hub"
    hub_cache.mkdir()
    monkeypatch.setattr(state_dir, "cache_root", lambda: tmp_path / "state")
    assert download_manifest.write_dataset_completion(
        "Org/Data",
        "commit-a",
        [download_manifest.ExpectedFile("train.parquet", 1)],
        hub_cache = hub_cache,
    )
    paths = list((tmp_path / "state" / "hub-state" / "manifests").rglob("*.json"))
    assert len(paths) == 1
    payload = json.loads(paths[0].read_text(encoding = "utf-8"))
    payload["expected_files"][0]["size"] = False
    paths[0].write_text(json.dumps(payload), encoding = "utf-8")

    assert (
        download_manifest.read_dataset_completion(
            "Org/Data",
            "commit-a",
            hub_cache = hub_cache,
        )
        is None
    )


def test_preview_snapshot_returns_immutable_revision_with_cache_pin(monkeypatch, tmp_path):
    repo_id = "Org/Data"
    repo_root, snapshot = _dataset_repo(tmp_path, repo_id, "commit-preview")
    (snapshot / "train.parquet").write_bytes(b"rows")
    monkeypatch.setattr(
        download_manifest,
        "read_dataset_completion",
        lambda *_args, **_kwargs: None,
    )

    pin, revision = dataset_cache.training_dataset_cache_pin(
        repo_id,
        str(repo_root),
    )

    assert pin == snapshot.resolve()
    assert revision == "commit-preview"


def test_training_cache_pin_prefers_processed_cache_without_explicit_path(monkeypatch, tmp_path):
    repo_id = "Org/Data"
    repo_root, snapshot = _dataset_repo(tmp_path, repo_id, "commit-preview")
    (snapshot / "README.md").write_text("metadata only", encoding = "utf-8")
    processed_root = tmp_path / "processed"
    processed = processed_root / "Org___Data"
    output = processed / "default" / "0.0.0" / "build-hash"
    output.mkdir(parents = True)
    (output / "dataset_info.json").write_text("{}", encoding = "utf-8")
    (output / "data-train.arrow").write_bytes(b"\xff\xff\xff\xff")
    monkeypatch.setenv("HF_DATASETS_CACHE", str(processed_root))
    monkeypatch.setattr(
        dataset_cache,
        "hf_datasets_cache_roots",
        lambda: [processed_root.resolve()],
    )
    _patch_cache_dirs(monkeypatch, repo_id, [repo_root])

    pin, revision = dataset_cache.training_dataset_cache_pin(repo_id)
    explicit_pin, explicit_revision = dataset_cache.training_dataset_cache_pin(
        repo_id,
        str(repo_root),
    )

    assert pin == processed.resolve()
    assert revision is None
    assert explicit_pin == snapshot.resolve()
    assert explicit_revision == "commit-preview"


def test_training_cache_pin_ignores_empty_processed_cache(monkeypatch, tmp_path):
    repo_id = "Org/Data"
    repo_root, snapshot = _dataset_repo(tmp_path, repo_id, "commit-preview")
    (snapshot / "train.parquet").write_bytes(b"rows")
    processed_root = tmp_path / "processed"
    processed = processed_root / "Org___Data"
    output = processed / "default" / "0.0.0" / "build-hash"
    output.mkdir(parents = True)
    (output / "dataset_info.json").write_text("{}", encoding = "utf-8")
    (output / "data-train.arrow").write_bytes(b"")
    monkeypatch.setenv("HF_DATASETS_CACHE", str(processed_root))
    monkeypatch.setattr(
        dataset_cache,
        "hf_datasets_cache_roots",
        lambda: [processed_root.resolve()],
    )
    _patch_cache_dirs(monkeypatch, repo_id, [repo_root])

    pin, revision = dataset_cache.training_dataset_cache_pin(repo_id)

    assert pin == snapshot.resolve()
    assert revision == "commit-preview"


def test_training_cache_pin_ignores_incomplete_processed_cache(monkeypatch, tmp_path):
    repo_id = "Org/Data"
    repo_root, snapshot = _dataset_repo(tmp_path, repo_id, "commit-preview")
    (snapshot / "train.parquet").write_bytes(b"rows")
    processed_root = tmp_path / "processed"
    processed = processed_root / "Org___Data"
    output = processed / "default" / "0.0.0" / "build-hash.incomplete"
    output.mkdir(parents = True)
    (output / "dataset_info.json").write_text("{}", encoding = "utf-8")
    (output / "data-train.arrow").write_bytes(b"\xff\xff\xff\xff")
    monkeypatch.setattr(
        dataset_cache,
        "hf_datasets_cache_roots",
        lambda: [processed_root.resolve()],
    )
    _patch_cache_dirs(monkeypatch, repo_id, [repo_root])

    pin, revision = dataset_cache.training_dataset_cache_pin(repo_id)

    assert pin == snapshot.resolve()
    assert revision == "commit-preview"


def test_app_processed_cache_is_deterministic_and_discoverable(monkeypatch, tmp_path):
    repo_id = "Org/Data"
    _, snapshot = _dataset_repo(tmp_path, repo_id, "commit-a")
    (snapshot / "train.parquet").write_bytes(b"rows")

    first = dataset_processed_cache.prepare_app_processed_dataset_cache(
        repo_id,
        snapshot,
    )
    dataset_processed_cache.mark_app_processed_dataset_cache_complete(first)
    second = dataset_processed_cache.prepare_app_processed_dataset_cache(
        repo_id,
        snapshot,
    )

    assert second.path == first.path
    assert second.cache_dir == first.cache_dir
    assert second.complete is True
    assert list(dataset_processed_cache.iter_app_processed_dataset_caches()) == [second]


def test_app_processed_cache_rejects_symlinked_parent(monkeypatch, tmp_path):
    repo_id = "Org/Data"
    _, snapshot = _dataset_repo(tmp_path, repo_id, "commit-a")
    (snapshot / "train.parquet").write_bytes(b"rows")
    configured_root = tmp_path / "configured-cache"
    external = tmp_path / "external"
    configured_root.mkdir()
    external.mkdir()
    (configured_root / "hf-datasets").symlink_to(
        external,
        target_is_directory = True,
    )
    monkeypatch.setattr(
        "utils.paths.storage_roots.cache_root",
        lambda: configured_root,
    )

    with pytest.raises(OSError, match = "Dataset cache root is unavailable"):
        dataset_processed_cache.prepare_app_processed_dataset_cache(
            repo_id,
            snapshot,
        )

    assert not (external / "snapshot-loads").exists()


def test_app_processed_cache_does_not_scan_or_delete_through_parent_symlink(monkeypatch, tmp_path):
    repo_id = "Org/Data"
    _, snapshot = _dataset_repo(tmp_path, repo_id, "commit-a")
    (snapshot / "train.parquet").write_bytes(b"rows")
    external_root = tmp_path / "external-cache"
    monkeypatch.setattr(
        "utils.paths.storage_roots.cache_root",
        lambda: external_root,
    )
    entry = dataset_processed_cache.prepare_app_processed_dataset_cache(
        repo_id,
        snapshot,
    )
    dataset_processed_cache.mark_app_processed_dataset_cache_complete(entry)
    configured_root = tmp_path / "configured-cache"
    configured_root.mkdir()
    (configured_root / "hf-datasets").symlink_to(
        external_root / "hf-datasets",
        target_is_directory = True,
    )
    monkeypatch.setattr(
        "utils.paths.storage_roots.cache_root",
        lambda: configured_root,
    )

    assert list(dataset_processed_cache.iter_app_processed_dataset_caches()) == []
    assert dataset_processed_cache.delete_app_processed_dataset_caches(repo_id) == (False, [])
    assert entry.path.is_dir()


def test_dataset_completion_v2_round_trips_metadata_commit(monkeypatch, tmp_path):
    hub_cache = tmp_path / "hub"
    hub_cache.mkdir()
    monkeypatch.setattr(state_dir, "cache_root", lambda: tmp_path / "state")
    monkeypatch.setattr(
        "utils.hf_cache_settings.get_hf_cache_paths",
        lambda: types.SimpleNamespace(hub_cache = hub_cache),
    )

    assert download_manifest.write_dataset_completion(
        "Org/Data",
        "commit-a",
        [download_manifest.ExpectedFile("train.parquet", 4)],
        "http",
        hub_cache = hub_cache,
    )
    manifest = download_manifest.read_dataset_completion(
        "Org/Data",
        "commit-a",
        hub_cache = hub_cache,
    )

    assert manifest is not None
    assert manifest.version == 2
    assert manifest.commit_hash == "commit-a"
    assert manifest.metadata_derived is True


def test_download_manifest_stays_readable_by_pre_pr_v1_reader(monkeypatch, tmp_path):
    hub_cache = tmp_path / "hub"
    hub_cache.mkdir()
    monkeypatch.setattr(state_dir, "cache_root", lambda: tmp_path / "state")

    assert download_manifest.write_manifest(
        "dataset",
        "Org/Data",
        None,
        [download_manifest.ExpectedFile("train.parquet", 4)],
        "http",
        hub_cache = hub_cache,
        commit_hash = "commit-a",
        metadata_derived = True,
    )
    path = download_manifest.manifest_path(
        "dataset",
        "Org/Data",
        None,
        hub_cache = hub_cache,
    )
    assert path is not None
    payload = json.loads(path.read_text(encoding = "utf-8"))

    # This is the complete compatibility contract the pre-PR reader used.
    assert payload["version"] == 1
    assert payload["expected_files"] == [{"path": "train.parquet", "size": 4}]

    # The current reader can still consume the additive revision attestation,
    # which the interrupted-download recovery path needs.
    manifest = download_manifest.read_manifest(
        "dataset",
        "Org/Data",
        None,
        hub_cache = hub_cache,
    )
    assert manifest is not None
    assert manifest.version == 1
    assert manifest.commit_hash == "commit-a"
    assert manifest.metadata_derived is True


def test_startup_migrates_existing_ordinary_v2_manifests_across_cache_scopes(
    monkeypatch, tmp_path
):
    active_cache = tmp_path / "active-hub"
    inactive_cache = tmp_path / "inactive-hub"
    active_cache.mkdir()
    inactive_cache.mkdir()
    monkeypatch.setattr(state_dir, "cache_root", lambda: tmp_path / "state")
    monkeypatch.setattr(
        "utils.hf_cache_settings.get_hf_cache_paths",
        lambda: types.SimpleNamespace(hub_cache = active_cache),
    )

    assert download_manifest.write_manifest(
        "model",
        "Org/Model",
        "Q4_K_M",
        [download_manifest.ExpectedFile("model.gguf", 8)],
        "http",
        hub_cache = active_cache,
        _schema_version = 2,
    )
    assert download_manifest.write_manifest(
        "dataset",
        "Org/Data",
        None,
        [download_manifest.ExpectedFile("train.parquet", 4)],
        "http",
        hub_cache = inactive_cache,
        commit_hash = "commit-a",
        metadata_derived = True,
        _schema_version = 2,
    )

    assert download_manifest.migrate_ordinary_v2_manifests_for_downgrade() == 2
    for repo_type, repo_id, variant, hub_cache in (
        ("model", "Org/Model", "Q4_K_M", active_cache),
        ("dataset", "Org/Data", None, inactive_cache),
    ):
        path = download_manifest.manifest_path(
            repo_type,
            repo_id,
            variant,
            hub_cache = hub_cache,
        )
        assert path is not None
        payload = json.loads(path.read_text(encoding = "utf-8"))
        assert payload["version"] == 1
        assert payload["expected_files"]

    dataset_manifest = download_manifest.read_manifest(
        "dataset",
        "Org/Data",
        None,
        hub_cache = inactive_cache,
    )
    assert dataset_manifest is not None
    assert dataset_manifest.version == 1
    assert dataset_manifest.commit_hash == "commit-a"
    assert dataset_manifest.metadata_derived is True
    assert download_manifest.migrate_ordinary_v2_manifests_for_downgrade() == 0


def test_manifest_migration_prefilter_does_not_read_v1_body():
    reads = []

    class PrefixReader:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self, size):
            reads.append(size)
            if len(reads) > 1:
                raise AssertionError("v1 body should not be read")
            return b'{\n  "version": 1,'

    class ManifestPath:
        def open(self, mode):
            assert mode == "rb"
            return PrefixReader()

    assert download_manifest._read_migration_payload(ManifestPath()) is None
    assert reads == [download_manifest._MANIFEST_MIGRATION_PREFIX_BYTES]


def test_startup_manifest_migration_preserves_dataset_completion_v2(monkeypatch, tmp_path):
    hub_cache = tmp_path / "hub"
    hub_cache.mkdir()
    monkeypatch.setattr(state_dir, "cache_root", lambda: tmp_path / "state")

    assert download_manifest.write_dataset_completion(
        "Org/Data",
        "commit-a",
        [download_manifest.ExpectedFile("train.parquet", 4)],
        "http",
        hub_cache = hub_cache,
    )
    assert download_manifest.migrate_ordinary_v2_manifests_for_downgrade() == 0

    [path] = list((tmp_path / "state" / "hub-state" / "manifests").rglob("*.json"))
    assert json.loads(path.read_text(encoding = "utf-8"))["version"] == 2
    assert (
        download_manifest.read_dataset_completion(
            "Org/Data",
            "commit-a",
            hub_cache = hub_cache,
        )
        is not None
    )


@pytest.mark.parametrize("invalid_record", ["unsafe_path", "mismatched_identity"])
def test_startup_manifest_migration_leaves_untrusted_v2_records_untouched(
    monkeypatch, tmp_path, invalid_record
):
    hub_cache = tmp_path / "hub"
    hub_cache.mkdir()
    monkeypatch.setattr(state_dir, "cache_root", lambda: tmp_path / "state")

    assert download_manifest.write_manifest(
        "model",
        "Org/Model",
        None,
        [download_manifest.ExpectedFile("model.safetensors", 8)],
        hub_cache = hub_cache,
        _schema_version = 2,
    )
    path = download_manifest.manifest_path(
        "model",
        "Org/Model",
        None,
        hub_cache = hub_cache,
    )
    assert path is not None
    payload = json.loads(path.read_text(encoding = "utf-8"))
    if invalid_record == "unsafe_path":
        payload["expected_files"][0]["path"] = "../outside.safetensors"
    else:
        payload["repo_id"] = "Other/Model"
    path.write_text(json.dumps(payload), encoding = "utf-8")

    assert download_manifest.migrate_ordinary_v2_manifests_for_downgrade() == 0
    assert json.loads(path.read_text(encoding = "utf-8"))["version"] == 2


def test_startup_manifest_migration_skips_oversized_record(monkeypatch, tmp_path):
    hub_cache = tmp_path / "hub"
    hub_cache.mkdir()
    monkeypatch.setattr(state_dir, "cache_root", lambda: tmp_path / "state")

    assert download_manifest.write_manifest(
        "model",
        "Org/Model",
        None,
        [download_manifest.ExpectedFile("model.safetensors", 8)],
        hub_cache = hub_cache,
        _schema_version = 2,
    )
    path = download_manifest.manifest_path(
        "model",
        "Org/Model",
        None,
        hub_cache = hub_cache,
    )
    assert path is not None
    monkeypatch.setattr(
        download_manifest,
        "_MANIFEST_MIGRATION_MAX_BYTES",
        path.stat().st_size - 1,
    )

    assert download_manifest.migrate_ordinary_v2_manifests_for_downgrade() == 0
    assert json.loads(path.read_text(encoding = "utf-8"))["version"] == 2


def test_startup_manifest_migration_write_failure_keeps_v2_record(monkeypatch, tmp_path):
    hub_cache = tmp_path / "hub"
    hub_cache.mkdir()
    monkeypatch.setattr(state_dir, "cache_root", lambda: tmp_path / "state")

    assert download_manifest.write_manifest(
        "dataset",
        "Org/Data",
        None,
        [download_manifest.ExpectedFile("train.parquet", 4)],
        hub_cache = hub_cache,
        _schema_version = 2,
    )
    path = download_manifest.manifest_path(
        "dataset",
        "Org/Data",
        None,
        hub_cache = hub_cache,
    )
    assert path is not None
    monkeypatch.setattr(
        download_manifest,
        "_atomic_write_json",
        lambda *_args, **_kwargs: False,
    )

    assert download_manifest.migrate_ordinary_v2_manifests_for_downgrade() == 0
    assert json.loads(path.read_text(encoding = "utf-8"))["version"] == 2


def test_manifest_compatibility_migration_runs_after_orphan_reaping():
    source = (Path(__file__).resolve().parent.parent / "main.py").read_text(encoding = "utf-8")

    reaper = source.index("reap_hub_orphan_workers()")
    migration = source.index("migrate_ordinary_v2_manifests_for_downgrade()")
    startup_complete = source.index("\n    yield\n", migration)

    assert reaper < migration < startup_complete


def test_dataset_completion_rejects_missing_recorded_hub_cache(monkeypatch, tmp_path):
    hub_cache = tmp_path / "hub"
    hub_cache.mkdir()
    monkeypatch.setattr(state_dir, "cache_root", lambda: tmp_path / "state")
    assert download_manifest.write_dataset_completion(
        "Org/Data",
        "commit-a",
        [download_manifest.ExpectedFile("train.parquet", 4)],
        hub_cache = hub_cache,
    )
    [path] = list((tmp_path / "state" / "hub-state" / "manifests").rglob("*.json"))
    payload = json.loads(path.read_text(encoding = "utf-8"))
    payload["hub_cache"] = None
    path.write_text(json.dumps(payload), encoding = "utf-8")

    assert (
        download_manifest.read_dataset_completion(
            "Org/Data",
            "commit-a",
            hub_cache = hub_cache,
        )
        is None
    )


@pytest.mark.parametrize("variant", [False, 0, [], {}])
def test_manifest_v2_rejects_non_string_variant(monkeypatch, tmp_path, variant):
    hub_cache = tmp_path / "hub"
    hub_cache.mkdir()
    monkeypatch.setattr(state_dir, "cache_root", lambda: tmp_path / "state")
    assert download_manifest.write_dataset_completion(
        "Org/Data",
        "commit-a",
        [download_manifest.ExpectedFile("train.parquet", 2)],
        hub_cache = hub_cache,
    )
    [path] = list((tmp_path / "state" / "hub-state" / "manifests").rglob("*.json"))
    payload = json.loads(path.read_text(encoding = "utf-8"))
    payload["variant"] = variant
    path.write_text(json.dumps(payload), encoding = "utf-8")

    assert (
        download_manifest.read_dataset_completion(
            "Org/Data",
            "commit-a",
            hub_cache = hub_cache,
        )
        is None
    )


@pytest.mark.parametrize("state_kind", ["manifest", "cancel_marker"])
def test_repo_state_purge_uses_enumerated_variant_path(monkeypatch, tmp_path, state_kind):
    hub_cache = tmp_path / "hub"
    hub_cache.mkdir()
    monkeypatch.setattr(state_dir, "cache_root", lambda: tmp_path / "state")
    if state_kind == "manifest":
        assert download_manifest.write_manifest(
            "model",
            "Org/Model",
            "Q4_K_M",
            [download_manifest.ExpectedFile("model.gguf", 4)],
            hub_cache = hub_cache,
        )
        path = download_manifest.manifest_path(
            "model",
            "Org/Model",
            "Q4_K_M",
            hub_cache = hub_cache,
        )
    else:
        assert download_manifest.write_cancel_marker(
            "model",
            "Org/Model",
            "Q4_K_M",
            "http",
            hub_cache = hub_cache,
        )
        path = download_manifest.marker_path(
            "model",
            "Org/Model",
            "Q4_K_M",
            hub_cache = hub_cache,
        )
    assert path is not None
    payload = json.loads(path.read_text(encoding = "utf-8"))
    payload["variant"] = "Q8_0"
    path.write_text(json.dumps(payload), encoding = "utf-8")

    assert (
        download_manifest.purge_all_state_for_repo(
            "model",
            "Org/Model",
            hub_cache = hub_cache,
        )
        == 1
    )
    assert not path.exists()


@pytest.mark.parametrize(
    "path_value",
    [
        "../train.parquet",
        "%2e%2e/train.parquet",
        "/train.parquet",
        "C:train.parquet",
        "C:\\train.parquet",
        "\\\\server\\share\\train.parquet",
        "nested\\train.parquet",
        "file:stream",
        "nested/file:stream",
        "file%3Astream",
        ".",
        "\x00train.parquet",
    ],
)
def test_manifest_expected_path_rejects_cross_platform_traversal(path_value):
    assert download_manifest.expected_path_is_safe(path_value) is False


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
        row_limit = 2,
    )

    assert resolved == processed.resolve()
    assert result == {"loaded": True}
    assert calls[0]["path"] == repo_id
    assert calls[0]["split"] == "train"
    assert calls[0]["cache_dir"] == str(processed_root.resolve())
    assert calls[0]["download_config"].local_files_only is True
    assert "streaming" not in calls[0]


def test_processed_cache_is_discovered_without_selected_path(monkeypatch, tmp_path):
    repo_id = "Org/Data"
    processed_root = tmp_path / "processed"
    processed = processed_root / "Org___Data"
    output = processed / "default" / "0.0.0" / "build-hash"
    output.mkdir(parents = True)
    (output / "dataset_info.json").write_text("{}", encoding = "utf-8")
    (output / "data-train.arrow").write_bytes(b"\xff\xff\xff\xff")
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
        ValueError("Either model_file or model_proto must be specified."),
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


def test_unknown_dataset_split_error_is_dataset_fallback_only_through_exception_chain():
    missing_split = ValueError("Unknown split \"validation\". Should be one of ['train'].")
    wrapped = RuntimeError("cached dataset load failed")
    wrapped.__cause__ = missing_split

    assert dataset_cache.is_cache_artifact_error(wrapped) is False
    assert (
        dataset_cache.dataset_cache_fallback_allowed(
            wrapped,
            require_exact = False,
            revision = None,
        )
        is True
    )


@pytest.mark.parametrize(
    "message",
    [
        'Unknown split "validation".',
        'Unknown split "validation". Should be one of train.',
        "Unknown split 'validation'. Should be one of ['train'].",
    ],
)
def test_unknown_dataset_split_lookalikes_are_not_retryable(message):
    error = ValueError(message)
    assert dataset_cache.is_cache_artifact_error(error) is False
    assert (
        dataset_cache.dataset_cache_fallback_allowed(
            error,
            require_exact = False,
            revision = None,
        )
        is False
    )


def test_unknown_dataset_split_fallback_preserves_exact_and_offline_gates(monkeypatch):
    error = ValueError("Unknown split \"validation\". Should be one of ['train'].")
    monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)
    monkeypatch.delenv("HF_DATASETS_OFFLINE", raising = False)

    assert (
        dataset_cache.dataset_cache_fallback_allowed(
            error,
            require_exact = False,
            revision = "dataset-commit",
        )
        is True
    )
    assert (
        dataset_cache.dataset_cache_fallback_allowed(
            error,
            require_exact = True,
            revision = None,
        )
        is False
    )

    monkeypatch.setenv("HF_DATASETS_OFFLINE", "true")
    assert (
        dataset_cache.dataset_cache_fallback_allowed(
            error,
            require_exact = False,
            revision = "dataset-commit",
        )
        is False
    )
