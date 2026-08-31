# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
S3 dataset loader.

Downloads dataset files (parquet / json / jsonl / csv) from an AWS S3 bucket
to a local temp directory so the existing local-file dataset path can consume
them. boto3 is an optional dependency and is imported lazily — callers should
gate on :func:`boto3_available` before invoking the loader.

Audio datasets (#4539) are a manifest plus the audio files it references. Audio
keys under the prefix are downloaded too, preserving their key structure so the
manifest's relative references stay true, and the manifest's audio values are
rewritten to the materialized absolute paths — the local loader hands them to
``datasets.Audio``, which opens path strings verbatim. Audio references a
parquet manifest carries cannot be rewritten without pyarrow, so those must
already be absolute or resolvable on this host.

The S3 config dict mirrors ``models.training.S3Config.model_dump()`` (snake_case
keys): bucket, region, prefix, access_key_id, secret_access_key, use_iam_role.
Credentials are read once to build the client and never logged or persisted.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import shutil
import tempfile
from importlib.util import find_spec
from typing import Callable, Optional
from utils.datasets.format_detection import _AUDIO_EXTENSIONS
from utils.paths.path_utils import drop_shadowed_appledouble_names

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
# Manifest formats whose audio references this module can rewrite in place.
_REWRITABLE_MANIFEST_EXTENSIONS = (".json", ".jsonl", ".csv")


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
    import boto3

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


def _list_dataset_keys(client, bucket: str, prefix: Optional[str]) -> tuple[list[str], list[str]]:
    """Supported dataset keys under ``prefix``, as (manifest keys, audio keys)."""
    paginator = client.get_paginator("list_objects_v2")
    list_kwargs = {"Bucket": bucket}
    if prefix:
        list_kwargs["Prefix"] = prefix

    keys: list[str] = []
    audio_keys: list[str] = []
    for page in paginator.paginate(**list_kwargs):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/"):
                continue
            if os.path.basename(key).lower() in _IGNORED_METADATA_FILENAMES:
                continue
            if key.lower().endswith(SUPPORTED_EXTENSIONS):
                keys.append(key)
            elif key.lower().endswith(_AUDIO_EXTENSIONS):
                audio_keys.append(key)
    # A Mac sync uploads Finder metadata under the shard's own extension and
    # _validate_single_extension_family cannot see it, so a key is dropped only when the object it
    # would describe is in the same listing.
    return (
        drop_shadowed_appledouble_names(keys),
        drop_shadowed_appledouble_names(audio_keys),
    )


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


def _download_one(client, bucket, key, local_path, cancel_callback) -> None:
    download_kwargs = {}
    if cancel_callback is not None:
        download_kwargs["Callback"] = lambda _bytes: _raise_if_cancelled(cancel_callback)
    client.download_file(bucket, key, local_path, **download_kwargs)
    _raise_if_cancelled(cancel_callback)


def _key_relative_to_prefix(key: str, prefix: Optional[str]) -> str:
    if prefix and key.startswith(prefix):
        key = key[len(prefix) :]
    return key.lstrip("/")


def _download_structured(
    client, bucket: str, prefix: Optional[str], keys: list[str], target_dir: str, cancel_callback
) -> dict[str, str]:
    """Download ``keys`` mirroring their prefix-relative layout. Returns key -> local path."""
    local_by_key: dict[str, str] = {}
    for key in keys:
        _raise_if_cancelled(cancel_callback)
        relative = _key_relative_to_prefix(key, prefix)
        parts = [part for part in relative.split("/") if part not in ("", ".")]
        # ".." is a legal literal in an S3 key, and this layout joins keys into
        # filesystem paths. Bucket contents are external input: refuse rather
        # than let a listing write above the download directory.
        if ".." in parts or not parts:
            raise ValueError(
                f"S3 key {key!r} contains '..' or empty path segments and cannot "
                "be downloaded into the dataset directory."
            )
        local_path = os.path.join(target_dir, *parts)
        os.makedirs(os.path.dirname(local_path) or target_dir, exist_ok = True)
        _download_one(client, bucket, key, local_path, cancel_callback)
        local_by_key[key] = local_path
    return local_by_key


def _rewrite_audio_references(
    manifest_local_by_key: dict[str, str],
    audio_local_by_key: dict[str, str],
    bucket: str,
    prefix: Optional[str],
) -> None:
    """Point each manifest's audio references at the downloaded local files.

    A reference resolves as the prefix-relative key, the full ``s3://`` URI, or
    a path relative to the manifest's own directory — whichever downloaded.
    Anything unmatched is left exactly as written: it may be an absolute path
    or URL that is somebody else's contract to satisfy.
    """
    lookup: dict[str, str] = {}
    for key, local_path in audio_local_by_key.items():
        lookup[_key_relative_to_prefix(key, prefix)] = local_path
        lookup[f"s3://{bucket}/{key}"] = local_path

    for manifest_key, manifest_path in manifest_local_by_key.items():
        if not manifest_path.lower().endswith(_REWRITABLE_MANIFEST_EXTENSIONS):
            continue  # parquet: cannot rewrite without pyarrow (see module docstring)
        manifest_dir = os.path.dirname(_key_relative_to_prefix(manifest_key, prefix))

        def resolve(value):
            if not isinstance(value, str) or not value.lower().endswith(_AUDIO_EXTENSIONS):
                return None
            direct = lookup.get(value)
            if direct is not None:
                return direct
            relative_to_manifest = os.path.normpath(os.path.join(manifest_dir, value))
            return lookup.get(relative_to_manifest.replace(os.sep, "/"))

        if manifest_path.lower().endswith(".csv"):
            _rewrite_csv_manifest(manifest_path, resolve)
        else:
            _rewrite_json_manifest(manifest_path, resolve)


def _rewrite_row(row, resolve) -> None:
    """Rewrite one manifest row's audio references in place."""
    if not isinstance(row, dict):
        return
    for column, value in row.items():
        replacement = resolve(value)
        if replacement is not None:
            row[column] = replacement
        elif isinstance(value, dict):
            # The HF undecoded-audio shape: {"path": ..., "bytes": ...}.
            replacement = resolve(value.get("path"))
            if replacement is not None:
                value["path"] = replacement


def _rewrite_json_manifest(manifest_path: str, resolve) -> None:
    if manifest_path.lower().endswith(".jsonl"):
        with open(manifest_path, encoding = "utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        for row in rows:
            _rewrite_row(row, resolve)
        with open(manifest_path, "w", encoding = "utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii = False) + "\n")
        return
    with open(manifest_path, encoding = "utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        return  # column-oriented JSON: leave it alone rather than guess
    for row in data:
        _rewrite_row(row, resolve)
    with open(manifest_path, "w", encoding = "utf-8") as f:
        json.dump(data, f, ensure_ascii = False)


def _rewrite_csv_manifest(manifest_path: str, resolve) -> None:
    with open(manifest_path, encoding = "utf-8", newline = "") as f:
        rows = list(csv.reader(f))
    changed = False
    for row in rows:
        for index, cell in enumerate(row):
            replacement = resolve(cell)
            if replacement is not None:
                row[index] = replacement
                changed = True
    if not changed:
        return
    with open(manifest_path, "w", encoding = "utf-8", newline = "") as f:
        csv.writer(f).writerows(rows)


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

    keys, audio_keys = _list_dataset_keys(client, bucket, prefix)
    _raise_if_cancelled(cancel_callback)
    where = f"s3://{bucket}/{prefix}" if prefix else f"s3://{bucket}"
    if not keys:
        if audio_keys:
            raise ValueError(
                f"Found {len(audio_keys)} audio file(s) under {where} but no manifest. "
                "An audio dataset needs a JSON/JSONL/CSV manifest beside the audio, "
                "with a column of audio paths and a column of transcriptions."
            )
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
        if audio_keys:
            # Keys keep their prefix-relative structure so the manifest's
            # relative references stay true; unique keys cannot collide.
            manifest_local_by_key = _download_structured(
                client, bucket, prefix, keys, target_dir, cancel_callback
            )
            audio_local_by_key = _download_structured(
                client, bucket, prefix, audio_keys, target_dir, cancel_callback
            )
            local_files = list(manifest_local_by_key.values())
            _rewrite_audio_references(manifest_local_by_key, audio_local_by_key, bucket, prefix)
        else:
            # The tabular-only layout keeps its flat, collision-renamed shape.
            used_paths: set[str] = set()
            for key in keys:
                _raise_if_cancelled(cancel_callback)
                filename = os.path.basename(key)
                local_path = _unique_local_path(target_dir, filename, used_paths)
                _download_one(client, bucket, key, local_path, cancel_callback)
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
