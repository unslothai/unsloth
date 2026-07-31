# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Datasets API routes."""

import json
import sys
from contextlib import suppress
from pathlib import Path
from typing import Optional
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile

backend_path = Path(__file__).parent.parent.parent
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))

from auth.authentication import get_current_subject
from hub.dependencies import get_hf_token
from models.datasets import LocalDatasetItem, LocalDatasetsResponse, UploadDatasetResponse
from utils.paths import dataset_uploads_root, ensure_dir, recipe_datasets_root
from utils.upload_limits import get_upload_limit_bytes, get_upload_limit_label

router = APIRouter()

LOCAL_UPLOAD_EXTS = {".csv", ".json", ".jsonl", ".parquet"}
# sync: training dataset upload limits are exposed by /api/settings/upload-limit
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


def _build_local_dataset_items() -> list[LocalDatasetItem]:
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

        try:
            updated_at = entry.stat().st_mtime
        except OSError:
            updated_at = None

        items.append(
            LocalDatasetItem(
                id = entry.name,
                label = entry.name,
                path = str(parquet_dir.resolve()),
                rows = rows,
                updated_at = updated_at,
                metadata = metadata_summary,
            )
        )

    items.sort(key = lambda item: item.updated_at or 0, reverse = True)
    return items


def _sanitize_filename(filename: str) -> str:
    name = Path(filename).name.strip().replace("\x00", "")
    if not name:
        return "dataset_upload"
    return name


@router.post("/upload", response_model = UploadDatasetResponse)
async def upload_dataset(
    file: UploadFile, current_subject: str = Depends(get_current_subject)
) -> UploadDatasetResponse:
    filename = _sanitize_filename(file.filename or "dataset_upload")
    ext = Path(filename).suffix.lower()
    if ext not in LOCAL_UPLOAD_EXTS:
        allowed = ", ".join(sorted(LOCAL_UPLOAD_EXTS))
        raise HTTPException(
            status_code = 400,
            detail = f"Unsupported file type: {ext}. Allowed: {allowed}",
        )

    ensure_dir(DATASET_UPLOAD_DIR)
    stem = Path(filename).stem
    stored_name = f"{uuid4().hex}_{stem}{ext}"
    stored_path = DATASET_UPLOAD_DIR / stored_name

    # Stream to disk in chunks to avoid holding the whole file in memory. The
    # route-level cap gives a clear training-dataset error and avoids leaving
    # oversized partial files in the Unsloth uploads directory.
    upload_limit_bytes = get_upload_limit_bytes()
    total_bytes = 0
    upload_complete = False
    try:
        with open(stored_path, "wb") as f:
            while chunk := await file.read(1024 * 1024):
                total_bytes += len(chunk)
                if total_bytes > upload_limit_bytes:
                    raise HTTPException(
                        status_code = 413,
                        detail = (
                            "Training dataset upload too large. "
                            f"Maximum is {get_upload_limit_label()}."
                        ),
                    )
                f.write(chunk)
        upload_complete = True
    finally:
        if not upload_complete:
            with suppress(OSError):
                stored_path.unlink(missing_ok = True)

    if stored_path.stat().st_size == 0:
        stored_path.unlink(missing_ok = True)
        raise HTTPException(status_code = 400, detail = "Empty upload payload")

    return UploadDatasetResponse(filename = filename, stored_path = str(stored_path))


@router.get("/local", response_model = LocalDatasetsResponse)
def list_local_datasets(
    current_subject: str = Depends(get_current_subject),
) -> LocalDatasetsResponse:
    return LocalDatasetsResponse(datasets = _build_local_dataset_items())


@router.get("/download-progress")
async def get_dataset_download_progress(
    repo_id: str = Query(..., description = "HuggingFace dataset repo ID, e.g. 'unsloth/LaTeX_OCR'"),
    hf_token: Optional[str] = Depends(get_hf_token),
    current_subject: str = Depends(get_current_subject),
):
    """Compatibility route backed by the shared multi-cache progress service."""
    from hub.services.datasets import downloads

    return await downloads.get_dataset_download_progress_response(
        repo_id,
        hf_token = hf_token,
    )
