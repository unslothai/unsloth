# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for the S3 dataset loader (core.training.s3_dataset).

boto3 is optional and may be absent in CI, so the S3 client is mocked: a fake
client provides a paginator over a synthetic bucket listing and writes files on
download_file. No network or real AWS credentials are involved.
"""

import importlib.util
import os
from pathlib import Path

import pytest

# Load the modules under test directly by path. Importing them through their
# packages (core.training / models) would execute heavy package __init__ chains
# (structlog, torch, …) that aren't needed for these unit tests.
_BACKEND = Path(__file__).resolve().parents[1]


def _load(mod_name, rel_path):
    spec = importlib.util.spec_from_file_location(mod_name, _BACKEND / rel_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


s3_dataset = _load("s3_dataset", "core/training/s3_dataset.py")
S3Config = _load("models_training_s3", "models/training.py").S3Config


class _FakePaginator:
    def __init__(self, keys):
        self._keys = keys

    def paginate(self, **kwargs):
        prefix = kwargs.get("Prefix")
        contents = [
            {"Key": k} for k in self._keys if prefix is None or k.startswith(prefix)
        ]
        # Emit in two pages to exercise pagination handling.
        mid = len(contents) // 2
        yield {"Contents": contents[:mid]}
        yield {"Contents": contents[mid:]}


class _FakeS3Client:
    def __init__(self, keys):
        self._keys = keys
        self.downloaded = []

    def get_paginator(self, name):
        assert name == "list_objects_v2"
        return _FakePaginator(self._keys)

    def download_file(self, bucket, key, local_path, **kwargs):
        self.downloaded.append((bucket, key, local_path))
        callback = kwargs.get("Callback")
        if callback is not None:
            callback(1)
        with open(local_path, "w", encoding = "utf-8") as f:
            f.write(f"content-of:{key}")


@pytest.fixture
def fake_client(monkeypatch):
    """Force boto3_available True and stub the client builder."""
    keys = [
        "datasets/train.parquet",
        "datasets/extra.parquet",
        "datasets/notes.txt",  # filtered out (unsupported)
        "datasets/subdir/",  # directory placeholder, skipped
        "other/ignore.parquet",  # filtered out by prefix
    ]
    client = _FakeS3Client(keys)
    monkeypatch.setattr(s3_dataset, "boto3_available", lambda: True)
    monkeypatch.setattr(s3_dataset, "_build_s3_client", lambda cfg: client)
    return client


def _cfg(**overrides):
    base = {
        "bucket": "my-bucket",
        "region": "us-east-1",
        "prefix": "datasets/",
        "access_key_id": "AKIA_TEST",
        "secret_access_key": "secret",
        "use_iam_role": False,
    }
    base.update(overrides)
    return base


def test_downloads_only_supported_files_under_prefix(fake_client, tmp_path):
    files = s3_dataset.download_s3_dataset(_cfg(), dest_dir = str(tmp_path))
    names = sorted(os.path.basename(f) for f in files)
    # txt is unsupported, the directory placeholder is skipped, and the
    # "other/" key is excluded by the prefix filter.
    assert names == ["extra.parquet", "train.parquet"]
    for f in files:
        assert os.path.exists(f)


def test_allows_json_and_jsonl_family(monkeypatch, tmp_path):
    client = _FakeS3Client(["datasets/train.json", "datasets/extra.jsonl"])
    monkeypatch.setattr(s3_dataset, "boto3_available", lambda: True)
    monkeypatch.setattr(s3_dataset, "_build_s3_client", lambda cfg: client)

    files = s3_dataset.download_s3_dataset(_cfg(), dest_dir = str(tmp_path))

    assert sorted(os.path.basename(f) for f in files) == ["extra.jsonl", "train.json"]


def test_ignores_common_json_metadata_files(monkeypatch, tmp_path):
    client = _FakeS3Client(
        [
            "datasets/train.parquet",
            "datasets/schema.json",
            "datasets/metadata.json",
            "datasets/dataset_info.json",
        ]
    )
    monkeypatch.setattr(s3_dataset, "boto3_available", lambda: True)
    monkeypatch.setattr(s3_dataset, "_build_s3_client", lambda cfg: client)

    files = s3_dataset.download_s3_dataset(_cfg(), dest_dir = str(tmp_path))

    assert [os.path.basename(f) for f in files] == ["train.parquet"]


def test_raises_when_prefix_contains_mixed_formats(monkeypatch, tmp_path):
    client = _FakeS3Client(["datasets/train.parquet", "datasets/stray.csv"])
    monkeypatch.setattr(s3_dataset, "boto3_available", lambda: True)
    monkeypatch.setattr(s3_dataset, "_build_s3_client", lambda cfg: client)

    with pytest.raises(ValueError, match = "mixed dataset formats"):
        s3_dataset.download_s3_dataset(_cfg(), dest_dir = str(tmp_path))

    assert client.downloaded == []


def test_raises_when_no_supported_files(monkeypatch, tmp_path):
    client = _FakeS3Client(["datasets/readme.txt"])
    monkeypatch.setattr(s3_dataset, "boto3_available", lambda: True)
    monkeypatch.setattr(s3_dataset, "_build_s3_client", lambda cfg: client)
    with pytest.raises(ValueError, match = "No supported dataset files"):
        s3_dataset.download_s3_dataset(_cfg(), dest_dir = str(tmp_path))


def test_raises_when_boto3_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(s3_dataset, "boto3_available", lambda: False)
    with pytest.raises(RuntimeError, match = "requires boto3"):
        s3_dataset.download_s3_dataset(_cfg(), dest_dir = str(tmp_path))


def test_basename_collisions_are_disambiguated(monkeypatch, tmp_path):
    # Two keys share a basename under different sub-prefixes.
    client = _FakeS3Client(["datasets/a/train.parquet", "datasets/b/train.parquet"])
    monkeypatch.setattr(s3_dataset, "boto3_available", lambda: True)
    monkeypatch.setattr(s3_dataset, "_build_s3_client", lambda cfg: client)
    files = s3_dataset.download_s3_dataset(_cfg(), dest_dir = str(tmp_path))
    assert len(files) == 2
    assert len(set(files)) == 2  # no overwrite


def test_basename_collision_skips_existing_generated_suffix(monkeypatch, tmp_path):
    client = _FakeS3Client(
        [
            "datasets/a/train.parquet",
            "datasets/b/train_1.parquet",
            "datasets/c/train.parquet",
        ]
    )
    monkeypatch.setattr(s3_dataset, "boto3_available", lambda: True)
    monkeypatch.setattr(s3_dataset, "_build_s3_client", lambda cfg: client)

    files = s3_dataset.download_s3_dataset(_cfg(), dest_dir = str(tmp_path))

    assert [os.path.basename(f) for f in files] == [
        "train.parquet",
        "train_1.parquet",
        "train_2.parquet",
    ]
    assert len(set(files)) == 3
    assert (tmp_path / "train_1.parquet").read_text(encoding = "utf-8") == (
        "content-of:datasets/b/train_1.parquet"
    )
    assert (tmp_path / "train_2.parquet").read_text(encoding = "utf-8") == (
        "content-of:datasets/c/train.parquet"
    )


def test_download_handle_cleans_owned_temp_dir(monkeypatch, tmp_path):
    target_dir = tmp_path / "owned-download"
    client = _FakeS3Client(["datasets/train.parquet"])
    monkeypatch.setattr(s3_dataset, "boto3_available", lambda: True)
    monkeypatch.setattr(s3_dataset, "_build_s3_client", lambda cfg: client)
    monkeypatch.setattr(s3_dataset.tempfile, "mkdtemp", lambda prefix: str(target_dir))

    download = s3_dataset.prepare_s3_dataset_download(_cfg())

    assert target_dir.exists()
    assert download.files == [str(target_dir / "train.parquet")]
    download.cleanup()
    assert not target_dir.exists()


def test_dest_dir_is_not_removed_by_cleanup(monkeypatch, tmp_path):
    client = _FakeS3Client(["datasets/train.parquet"])
    monkeypatch.setattr(s3_dataset, "boto3_available", lambda: True)
    monkeypatch.setattr(s3_dataset, "_build_s3_client", lambda cfg: client)

    download = s3_dataset.prepare_s3_dataset_download(_cfg(), dest_dir = str(tmp_path))

    download.cleanup()
    assert tmp_path.exists()
    assert (tmp_path / "train.parquet").exists()


def test_cancel_callback_aborts_and_removes_temp_dir(monkeypatch, tmp_path):
    target_dir = tmp_path / "cancelled-download"
    client = _FakeS3Client(["datasets/train.parquet"])
    monkeypatch.setattr(s3_dataset, "boto3_available", lambda: True)
    monkeypatch.setattr(s3_dataset, "_build_s3_client", lambda cfg: client)
    monkeypatch.setattr(s3_dataset.tempfile, "mkdtemp", lambda prefix: str(target_dir))
    calls = 0

    def cancel_after_download_starts():
        nonlocal calls
        calls += 1
        return calls >= 4

    with pytest.raises(s3_dataset.S3DownloadCancelled):
        s3_dataset.prepare_s3_dataset_download(
            _cfg(),
            cancel_callback = cancel_after_download_starts,
        )

    assert not target_dir.exists()


# ── S3Config model (camelCase aliases + credential validation) ──


def test_s3config_accepts_camelcase_aliases():
    cfg = S3Config.model_validate(
        {
            "bucket": "b",
            "region": "eu-west-1",
            "accessKeyId": "AKIA",
            "secretAccessKey": "shh",
        }
    )
    assert cfg.access_key_id == "AKIA"
    assert cfg.secret_access_key == "shh"
    # model_dump() yields snake_case for the loader.
    assert cfg.model_dump()["access_key_id"] == "AKIA"


def test_s3config_accepts_snake_case():
    cfg = S3Config.model_validate(
        {"bucket": "b", "access_key_id": "AKIA", "secret_access_key": "shh"}
    )
    assert cfg.access_key_id == "AKIA"


def test_s3config_requires_credentials_or_iam():
    with pytest.raises(ValueError):
        S3Config.model_validate({"bucket": "b"})


def test_s3config_iam_role_needs_no_keys():
    cfg = S3Config.model_validate({"bucket": "b", "useIamRole": True})
    assert cfg.use_iam_role is True
