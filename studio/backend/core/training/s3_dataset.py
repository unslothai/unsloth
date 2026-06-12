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
import tempfile
from importlib.util import find_spec
from typing import Optional

logger = logging.getLogger(__name__)

# Extensions the local-file loader (UnslothTrainer._loader_for_files) understands.
SUPPORTED_EXTENSIONS = (".parquet", ".json", ".jsonl", ".csv")


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
            if key.lower().endswith(SUPPORTED_EXTENSIONS):
                keys.append(key)
    return keys


def download_s3_dataset(s3_config: dict, dest_dir: Optional[str] = None) -> list[str]:
    """Download supported dataset files from S3 to a local directory.

    Returns the list of absolute local file paths (one per downloaded object),
    suitable for passing through the existing local-file dataset loader.

    Raises ``RuntimeError`` if boto3 is missing, and ``ValueError`` if the
    bucket/prefix contains no supported dataset files.
    """
    if not boto3_available():
        raise RuntimeError(
            "S3 dataset loading requires boto3. Install it with: pip install boto3"
        )

    bucket = s3_config.get("bucket")
    if not bucket:
        raise ValueError("s3_config.bucket is required")
    prefix = s3_config.get("prefix")

    client = _build_s3_client(s3_config)

    keys = _list_dataset_keys(client, bucket, prefix)
    if not keys:
        where = f"s3://{bucket}/{prefix}" if prefix else f"s3://{bucket}"
        raise ValueError(
            f"No supported dataset files ({', '.join(SUPPORTED_EXTENSIONS)}) "
            f"found under {where}"
        )

    target_dir = dest_dir or tempfile.mkdtemp(prefix = "unsloth_s3_dataset_")
    os.makedirs(target_dir, exist_ok = True)

    local_files: list[str] = []
    for key in keys:
        # Flatten the key to a filename, keeping the basename and extension.
        filename = os.path.basename(key)
        local_path = os.path.join(target_dir, filename)
        # Disambiguate collisions from different prefixes sharing a basename.
        if local_path in local_files:
            stem, ext = os.path.splitext(filename)
            local_path = os.path.join(target_dir, f"{stem}_{len(local_files)}{ext}")
        client.download_file(bucket, key, local_path)
        local_files.append(local_path)

    logger.info(
        "Downloaded %d dataset file(s) from s3://%s/%s to %s",
        len(local_files),
        bucket,
        prefix or "",
        target_dir,
    )
    return local_files
