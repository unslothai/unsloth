# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Local dataset upload and listing services."""

from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path

from fastapi import HTTPException, UploadFile

from hub.schemas.datasets import (
    LocalDatasetItem,
    LocalDatasetsResponse,
    UploadDatasetResponse,
)
from hub.utils.paths import dataset_uploads_root, ensure_dir, recipe_datasets_root

# Tabular formats are preferred over archives for Tier 1 preview: archives
# (e.g. images.zip) load as ImageFolder with synthetic columns that don't
# match the real schema.
_TABULAR_EXTS = (".parquet", ".json", ".jsonl", ".csv", ".tsv", ".arrow")
_ARCHIVE_EXTS = (".tar", ".tar.gz", ".tgz", ".gz", ".zst", ".zip", ".txt")
DATA_EXTS = _TABULAR_EXTS + _ARCHIVE_EXTS
LOCAL_FILE_EXTS = (".json", ".jsonl", ".csv", ".parquet")
LOCAL_UPLOAD_EXTS = {".csv", ".json", ".jsonl", ".parquet"}
LOCAL_UPLOAD_CHUNK_BYTES = 1024 * 1024
LOCAL_UPLOAD_MAX_BYTES = 500 * 1024 * 1024
LOCAL_DATASETS_ROOT = recipe_datasets_root()
DATASET_UPLOAD_DIR = dataset_uploads_root()


def _safe_read_metadata(path: Path) -> dict | None:
    try:
        payload = json.loads(path.read_text(encoding = "utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _safe_read_rows_from_metadata(payload: dict | None) -> int | None:
    if not payload:
        return None
    for key in ("actual_num_records", "target_num_records"):
        value = payload.get(key)
        if isinstance(value, int):
            return value
    return None


def _safe_read_metadata_summary(payload: dict | None) -> dict | None:
    if not payload:
        return None

    actual_num_records = (
        payload.get("actual_num_records")
        if isinstance(payload.get("actual_num_records"), int)
        else None
    )
    target_num_records = (
        payload.get("target_num_records")
        if isinstance(payload.get("target_num_records"), int)
        else actual_num_records
    )

    columns: list[str] | None = None
    schema = payload.get("schema")
    if isinstance(schema, dict):
        columns = [str(key) for key in schema.keys()]
    if not columns:
        stats = payload.get("column_statistics")
        if isinstance(stats, list):
            derived = [
                str(item.get("column_name"))
                for item in stats
                if isinstance(item, dict) and item.get("column_name")
            ]
            columns = derived or None

    parquet_files_count = None
    file_paths = payload.get("file_paths")
    if isinstance(file_paths, dict):
        parquet_files = file_paths.get("parquet-files")
        if isinstance(parquet_files, list):
            parquet_files_count = len(parquet_files)

    total_num_batches = (
        payload.get("total_num_batches")
        if isinstance(payload.get("total_num_batches"), int)
        else parquet_files_count
    )
    num_completed_batches = (
        payload.get("num_completed_batches")
        if isinstance(payload.get("num_completed_batches"), int)
        else total_num_batches
    )

    return {
        "actual_num_records": actual_num_records,
        "target_num_records": target_num_records,
        "total_num_batches": total_num_batches,
        "num_completed_batches": num_completed_batches,
        "columns": columns,
    }


def _safe_mtime(path: Path) -> float | None:
    try:
        return path.stat().st_mtime
    except OSError:
        return None


def _display_uploaded_dataset_name(path: Path) -> str:
    stem = path.stem
    prefix, sep, rest = stem.partition("_")
    if sep and len(prefix) == 32 and all(c in "0123456789abcdef" for c in prefix):
        return f"{rest}{path.suffix}"
    return path.name


def _build_recipe_dataset_items() -> list[LocalDatasetItem]:
    if not LOCAL_DATASETS_ROOT.exists():
        return []

    items: list[LocalDatasetItem] = []
    for entry in LOCAL_DATASETS_ROOT.iterdir():
        if not entry.is_dir() or not entry.name.startswith("recipe_"):
            continue
        parquet_dir = entry / "parquet-files"
        if not parquet_dir.exists() or not any(parquet_dir.glob("*.parquet")):
            continue

        rows = None
        metadata_summary = None
        metadata_path = entry / "metadata.json"
        if metadata_path.exists():
            metadata_payload = _safe_read_metadata(metadata_path)
            rows = _safe_read_rows_from_metadata(metadata_payload)
            metadata_summary = _safe_read_metadata_summary(metadata_payload)

        items.append(
            LocalDatasetItem(
                id = entry.name,
                label = entry.name,
                path = str(parquet_dir.resolve()),
                source = "recipe",
                rows = rows,
                updated_at = _safe_mtime(entry),
                metadata = metadata_summary,
            )
        )

    return items


def _build_uploaded_dataset_items() -> list[LocalDatasetItem]:
    if not DATASET_UPLOAD_DIR.exists():
        return []

    items: list[LocalDatasetItem] = []
    for path in DATASET_UPLOAD_DIR.iterdir():
        if not path.is_file() or path.suffix.lower() not in LOCAL_UPLOAD_EXTS:
            continue
        try:
            if path.stat().st_size == 0:
                continue
        except OSError:
            continue
        label = _display_uploaded_dataset_name(path)
        items.append(
            LocalDatasetItem(
                id = path.name,
                label = label,
                path = str(path.resolve()),
                source = "upload",
                updated_at = _safe_mtime(path),
            )
        )
    return items


def _build_local_dataset_items() -> list[LocalDatasetItem]:
    items = _build_recipe_dataset_items() + _build_uploaded_dataset_items()
    items.sort(key = lambda item: item.updated_at or 0, reverse = True)
    return items


def _stream_file_preview_slice(path: Path, preview_size: int):
    """Stream the first ``preview_size`` rows so a large file is never fully parsed into Arrow; returns ``(Dataset, None)`` or ``None`` if empty/unsupported."""
    from itertools import islice

    from datasets import Dataset, load_dataset

    name = path.name.lower()
    if name.endswith((".json", ".jsonl")):
        loader = "json"
    elif name.endswith((".csv", ".tsv")):
        loader = "csv"
    elif name.endswith(".parquet"):
        loader = "parquet"
    elif name.endswith(".arrow"):
        loader = "arrow"
    elif name.endswith(".txt"):
        loader = "text"
    else:
        return None

    streamed = load_dataset(
        loader,
        data_files = str(path),
        split = "train",
        streaming = True,
    )
    rows = list(islice(streamed, preview_size))
    if not rows:
        return None
    return Dataset.from_list(rows), None


def _load_local_preview_slice(
    *, dataset_path: Path, train_split: str, preview_size: int
):
    # Non-streaming loads take the cached builder lock; use the EACCES-safe wrapper.
    from utils.datasets.cache_safe import load_dataset_cache_safe as load_dataset

    if dataset_path.is_dir():
        parquet_dir = (
            dataset_path / "parquet-files"
            if (dataset_path / "parquet-files").exists()
            else dataset_path
        )
        parquet_files = sorted(parquet_dir.glob("*.parquet"))
        if parquet_files:
            dataset = load_dataset(
                "parquet",
                data_files = [str(path) for path in parquet_files],
                split = train_split,
            )
            total_rows = len(dataset)
            preview_slice = dataset.select(range(min(preview_size, total_rows)))
            return preview_slice, total_rows

        candidate_files: list[Path] = []
        for ext in LOCAL_FILE_EXTS:
            candidate_files.extend(sorted(dataset_path.glob(f"*{ext}")))
        if not candidate_files:
            raise HTTPException(
                status_code = 400,
                detail = "Unsupported local dataset directory (expected parquet/json/jsonl/csv files)",
            )
        dataset_path = candidate_files[0]

    suffix = dataset_path.suffix.lower()
    # Parquet/Arrow give a cheap exact total_rows via len()+select; JSON/CSV
    # carry no such metadata, so stream them and report total_rows=None.
    if suffix == ".parquet":
        dataset = load_dataset(
            "parquet", data_files = str(dataset_path), split = train_split
        )
        total_rows = len(dataset)
        preview_slice = dataset.select(range(min(preview_size, total_rows)))
        return preview_slice, total_rows

    if suffix in (".json", ".jsonl", ".csv"):
        preview = _stream_file_preview_slice(dataset_path, preview_size)
        if preview is None:
            raise HTTPException(
                status_code = 400,
                detail = "Dataset appears to be empty or could not be read",
            )
        return preview

    raise HTTPException(
        status_code = 400, detail = f"Unsupported file format: {dataset_path.suffix}"
    )


def _sanitize_filename(filename: str) -> str:
    name = Path(filename).name.strip().replace("\x00", "")
    if not name:
        return "dataset_upload"
    return name


def _upload_too_large(size_bytes: int) -> HTTPException:
    return HTTPException(
        status_code = 413,
        detail = (
            f"Upload is too large "
            f"({size_bytes:,} bytes; max {LOCAL_UPLOAD_MAX_BYTES:,})."
        ),
    )


async def upload_dataset_response(file: UploadFile) -> UploadDatasetResponse:
    filename = _sanitize_filename(file.filename or "dataset_upload")
    ext = Path(filename).suffix.lower()
    if ext not in LOCAL_UPLOAD_EXTS:
        allowed = ", ".join(sorted(LOCAL_UPLOAD_EXTS))
        raise HTTPException(
            status_code = 400,
            detail = f"Unsupported file type: {ext}. Allowed: {allowed}",
        )

    declared_size = getattr(file, "size", None)
    if isinstance(declared_size, int) and declared_size > LOCAL_UPLOAD_MAX_BYTES:
        raise _upload_too_large(declared_size)

    ensure_dir(DATASET_UPLOAD_DIR)
    stem = Path(filename).stem
    stored_name = f"{uuid.uuid4().hex}_{stem}{ext}"
    stored_path = DATASET_UPLOAD_DIR / stored_name

    written = 0
    try:
        with open(stored_path, "wb") as f:
            while chunk := await file.read(LOCAL_UPLOAD_CHUNK_BYTES):
                written += len(chunk)
                if written > LOCAL_UPLOAD_MAX_BYTES:
                    raise _upload_too_large(written)
                await asyncio.to_thread(f.write, chunk)
    except Exception:
        stored_path.unlink(missing_ok = True)
        raise

    if written == 0:
        stored_path.unlink(missing_ok = True)
        raise HTTPException(status_code = 400, detail = "Empty upload payload")

    return UploadDatasetResponse(filename = filename, stored_path = str(stored_path))


def list_local_datasets_response() -> LocalDatasetsResponse:
    return LocalDatasetsResponse(datasets = _build_local_dataset_items())
