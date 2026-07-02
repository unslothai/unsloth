# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
S3 dataset loader.

Downloads dataset files (parquet / json / jsonl / csv) from an AWS S3 bucket
to a local temp directory so the existing local-file dataset path can consume
them. boto3 is an optional dependency and is imported lazily — callers should
gate on :func:`boto3_available` before invoking the loader.

The S3 config dict mirrors ``models.training.S3Config.model_dump()`` (snake_case
keys): bucket, region, prefix, access_key_id, secret_access_key, use_iam_role.
Credentials are read once to build the client and never logged or persisted.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from importlib.util import find_spec
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# Extensions the local-file loader (UnslothTrainer._loader_for_files) understands.
SUPPORTED_EXTENSIONS = (".parquet", ".json", ".jsonl", ".csv")
_JSON_EXTENSIONS = (".json", ".jsonl")
_IGNORED_METADATA_FILENAMES = {
    "dataset_info.json",
    "metadata.json",
    "schema.json",
    "state.json",
}


class S3DownloadCancelled(RuntimeError):
    """Raised when the caller cancels an S3 dataset download."""


class S3DatasetDownload:
    def __init__(
        self,
        files: list[str],
        temp_dir: Optional[str] = None,
    ):
        self.files = files
        self.temp_dir = temp_dir

    def cleanup(self) -> None:
        if not self.temp_dir:
            return
        shutil.rmtree(self.temp_dir, ignore_errors = True)
        self.temp_dir = None


def boto3_available() -> bool:
    """True if boto3 can be imported (without importing it)."""
    return find_spec("boto3") is not None


def _build_s3_client(s3_config: dict):
    """Create a boto3 S3 client from the config dict.

    Uses explicit access keys when provided, otherwise falls back to the
    default credential chain (IAM role / instance profile / env / shared creds).
    """
    import boto3  # lazy: optional dependency

    region = s3_config.get("region") or "us-east-1"
    use_iam_role = bool(s3_config.get("use_iam_role"))
    access_key_id = s3_config.get("access_key_id")
    secret_access_key = s3_config.get("secret_access_key")

    if not use_iam_role and access_key_id and secret_access_key:
        return boto3.client(
            "s3",
            region_name = region,
            aws_access_key_id = access_key_id,
            aws_secret_access_key = secret_access_key,
        )
    # IAM role / instance profile / ambient credentials
    return boto3.client("s3", region_name = region)


def _list_dataset_keys(client, bucket: str, prefix: Optional[str]) -> list[str]:
    """List object keys under ``prefix`` that have a supported data extension."""
    paginator = client.get_paginator("list_objects_v2")
    list_kwargs = {"Bucket": bucket}
    if prefix:
        list_kwargs["Prefix"] = prefix

    keys: list[str] = []
    for page in paginator.paginate(**list_kwargs):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/"):
                continue  # directory placeholder
            if os.path.basename(key).lower() in _IGNORED_METADATA_FILENAMES:
                continue
            if key.lower().endswith(SUPPORTED_EXTENSIONS):
                keys.append(key)
    return keys


def _extension_family(key: str) -> str:
    ext = os.path.splitext(key)[1].lower()
    if ext in _JSON_EXTENSIONS:
        return "json"
    return ext.lstrip(".")


def _validate_single_extension_family(keys: list[str]) -> None:
    families: list[str] = []
    for key in keys:
        family = _extension_family(key)
        if family not in families:
            families.append(family)

    if len(families) <= 1:
        return

    raise ValueError(
        "S3 prefix contains mixed dataset formats "
        f"({', '.join(families)}). Keep one dataset format under the selected prefix."
    )


def _unique_local_path(target_dir: str, filename: str, used_paths: set[str]) -> str:
    """Return an unused flattened path for an S3 object basename."""
    stem, ext = os.path.splitext(filename)
    candidate = os.path.join(target_dir, filename)
    suffix = 1
    while candidate in used_paths or os.path.exists(candidate):
        candidate = os.path.join(target_dir, f"{stem}_{suffix}{ext}")
        suffix += 1
    used_paths.add(candidate)
    return candidate


def _raise_if_cancelled(cancel_callback: Optional[Callable[[], bool]]) -> None:
    if cancel_callback is not None and cancel_callback():
        raise S3DownloadCancelled("S3 dataset download cancelled")


def prepare_s3_dataset_download(
    s3_config: dict,
    dest_dir: Optional[str] = None,
    cancel_callback: Optional[Callable[[], bool]] = None,
) -> S3DatasetDownload:
    """Download supported dataset files from S3 to a local directory.

    Returns the local files plus the owned temporary directory, when one was
    created. Call ``cleanup()`` after the dataset loader has materialized data.

    Raises ``RuntimeError`` if boto3 is missing, and ``ValueError`` if the
    bucket/prefix contains no supported dataset files.
    """
    if not boto3_available():
        raise RuntimeError("S3 dataset loading requires boto3. Install it with: pip install boto3")

    bucket = s3_config.get("bucket")
    if not bucket:
        raise ValueError("s3_config.bucket is required")
    prefix = s3_config.get("prefix")

    _raise_if_cancelled(cancel_callback)
    client = _build_s3_client(s3_config)

    keys = _list_dataset_keys(client, bucket, prefix)
    _raise_if_cancelled(cancel_callback)
    if not keys:
        where = f"s3://{bucket}/{prefix}" if prefix else f"s3://{bucket}"
        raise ValueError(
            f"No supported dataset files ({', '.join(SUPPORTED_EXTENSIONS)}) "
            f"found under {where}"
        )

    _validate_single_extension_family(keys)

    owns_temp_dir = dest_dir is None
    target_dir = dest_dir or tempfile.mkdtemp(prefix = "unsloth_s3_dataset_")
    try:
        os.makedirs(target_dir, exist_ok = True)

        local_files: list[str] = []
        used_paths: set[str] = set()
        for key in keys:
            _raise_if_cancelled(cancel_callback)
            filename = os.path.basename(key)
            local_path = _unique_local_path(target_dir, filename, used_paths)
            download_kwargs = {}
            if cancel_callback is not None:
                download_kwargs["Callback"] = lambda _bytes: _raise_if_cancelled(cancel_callback)
            client.download_file(bucket, key, local_path, **download_kwargs)
            _raise_if_cancelled(cancel_callback)
            local_files.append(local_path)
    except Exception:
        if owns_temp_dir:
            shutil.rmtree(target_dir, ignore_errors = True)
        raise

    logger.info(
        "Downloaded %d dataset file(s) from s3://%s/%s to %s",
        len(local_files),
        bucket,
        prefix or "",
        target_dir,
    )
    return S3DatasetDownload(
        files = local_files,
        temp_dir = target_dir if owns_temp_dir else None,
    )


def download_s3_dataset(
    s3_config: dict,
    dest_dir: Optional[str] = None,
    cancel_callback: Optional[Callable[[], bool]] = None,
) -> list[str]:
    download = prepare_s3_dataset_download(
        s3_config,
        dest_dir = dest_dir,
        cancel_callback = cancel_callback,
    )
    return download.files
