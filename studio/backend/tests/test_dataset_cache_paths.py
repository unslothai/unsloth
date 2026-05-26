# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import importlib.util
import sys
import types
from pathlib import Path

if "loggers" not in sys.modules:

    class _DummyLogger:
        def __getattr__(self, _name):
            return lambda *args, **kwargs: None

    sys.modules["loggers"] = types.SimpleNamespace(
        get_logger = lambda *args, **kwargs: _DummyLogger(),
    )

from utils import hf_cache_scan

_CACHE_PATHS_FILE = (
    Path(__file__).resolve().parent.parent / "utils" / "datasets" / "cache_paths.py"
)
_spec = importlib.util.spec_from_file_location(
    "cache_paths_under_test",
    _CACHE_PATHS_FILE,
)
assert _spec and _spec.loader
cache_paths = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cache_paths)
cached_dataset_training_files = cache_paths.cached_dataset_training_files
latest_cached_dataset_snapshot = cache_paths.latest_cached_dataset_snapshot


def _dataset_repo(root: Path, repo_id: str, snapshot: str = "rev") -> tuple[Path, Path]:
    repo_root = root / f"datasets--{repo_id.replace('/', '--')}"
    snap = repo_root / "snapshots" / snapshot
    snap.mkdir(parents = True)
    return repo_root, snap


def test_latest_cached_dataset_snapshot_prefers_selected_cache_path(
    monkeypatch, tmp_path
):
    repo_id = "Org/Data"
    selected_root, selected_snap = _dataset_repo(tmp_path / "selected", repo_id)
    other_root, other_snap = _dataset_repo(tmp_path / "other", repo_id)
    (selected_snap / "train.parquet").write_bytes(b"selected")
    (other_snap / "train.parquet").write_bytes(b"other")

    monkeypatch.setattr(
        hf_cache_scan,
        "iter_repo_cache_dirs",
        lambda repo_type, requested: iter([other_root])
        if repo_type == "dataset" and requested == repo_id
        else iter([]),
    )

    assert (
        latest_cached_dataset_snapshot(repo_id, str(selected_root))
        == selected_snap.resolve()
    )


def test_cached_dataset_training_files_filters_requested_split(monkeypatch, tmp_path):
    repo_id = "Org/Data"
    repo_root, snap = _dataset_repo(tmp_path, repo_id)
    train = snap / "train-00000-of-00001.parquet"
    validation = snap / "validation-00000-of-00001.parquet"
    test = snap / "test-00000-of-00001.parquet"
    train.write_bytes(b"train")
    validation.write_bytes(b"validation")
    test.write_bytes(b"test")

    monkeypatch.setattr(
        hf_cache_scan,
        "iter_repo_cache_dirs",
        lambda repo_type, requested: iter([repo_root])
        if repo_type == "dataset" and requested == repo_id
        else iter([]),
    )

    assert cached_dataset_training_files(
        repo_id,
        str(repo_root),
        subset = None,
        train_split = "train",
    ) == [str(train)]
    assert cached_dataset_training_files(
        repo_id,
        str(repo_root),
        subset = None,
        train_split = "validation",
    ) == [str(validation)]


def test_cached_dataset_training_files_filters_selected_subset(monkeypatch, tmp_path):
    repo_id = "Org/Data"
    repo_root, snap = _dataset_repo(tmp_path, repo_id)
    en = snap / "en" / "train.parquet"
    fr = snap / "fr" / "train.parquet"
    en.parent.mkdir()
    fr.parent.mkdir()
    en.write_bytes(b"en")
    fr.write_bytes(b"fr")

    monkeypatch.setattr(
        hf_cache_scan,
        "iter_repo_cache_dirs",
        lambda repo_type, requested: iter([repo_root])
        if repo_type == "dataset" and requested == repo_id
        else iter([]),
    )

    assert cached_dataset_training_files(
        repo_id,
        str(repo_root),
        subset = "fr",
        train_split = "train",
    ) == [str(fr)]
