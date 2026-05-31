# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Seed inspect endpoints for data recipe."""

from __future__ import annotations

import base64
import binascii
import json
import os
import re
from itertools import islice
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, HTTPException, UploadFile, File as FastAPIFile, Form

try:
    from data_designer_unstructured_seed.chunking import (
        build_multi_file_preview_rows,
        build_unstructured_preview_rows,
        normalize_unstructured_text,
        resolve_chunking,
    )
except ImportError:
    build_multi_file_preview_rows = None
    build_unstructured_preview_rows = None
    normalize_unstructured_text = None
    resolve_chunking = None
from core.data_recipe.jsonable import to_preview_jsonable
from utils.paths import ensure_dir, seed_uploads_root, unstructured_uploads_root

from models.data_recipe import (
    SeedInspectRequest,
    SeedInspectResponse,
    SeedInspectUploadRequest,
    UnstructuredFileUploadResponse,
)

router = APIRouter()

DATA_EXTS = (".parquet", ".jsonl", ".json", ".csv")
DEFAULT_SPLIT = "train"
LOCAL_UPLOAD_EXTS = {".csv", ".json", ".jsonl"}
UNSTRUCTURED_ALLOWED_EXTS = {".pdf", ".docx", ".txt", ".md"}
SEED_UPLOAD_DIR = seed_uploads_root()
UNSTRUCTURED_UPLOAD_ROOT = unstructured_uploads_root()
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
MAX_TOTAL_SIZE = 100 * 1024 * 1024  # 100MB

_SAFE_ID_RE = re.compile(r"^[a-zA-Z0-9_-]+$")


def _validate_safe_id(value: str, label: str) -> str:
    if not value or not _SAFE_ID_RE.match(value):
        raise HTTPException(
            400, f"Invalid {label}: must be alphanumeric/dash/underscore only"
        )
    return value


def _serialize_preview_value(value: Any) -> Any:
    return to_preview_jsonable(value)


def _serialize_preview_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {str(key): _serialize_preview_value(value) for key, value in row.items()}
        for row in rows
    ]


def _normalize_optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    trimmed = value.strip()
    return trimmed if trimmed else None


def _list_hf_data_files(*, dataset_name: str, token: str | None) -> list[str]:
    try:
        from huggingface_hub import HfApi
        from huggingface_hub.utils import HfHubHTTPError
    except ImportError:
        return []
    try:
        api = HfApi()
        repo_files = api.list_repo_files(dataset_name, repo_type = "dataset", token = token)
        return [file for file in repo_files if file.lower().endswith(DATA_EXTS)]
    except (HfHubHTTPError, OSError, ValueError):
        return []


def _select_best_file(data_files: list[str], split: str = DEFAULT_SPLIT) -> str | None:
    if not data_files:
        return None
    split_lower = split.lower()

    def score(path: str) -> tuple[int, int]:
        name = path.lower()
        if f"/{split_lower}/" in name:
            return (0, len(path))
        if (
            f"_{split_lower}." in name
            or f"-{split_lower}." in name
            or f"/{split_lower}." in name
            or f"/{split_lower}_" in name
            or f"/{split_lower}-" in name
        ):
            return (1, len(path))
        return (2, len(path))

    return sorted(data_files, key = score)[0]


def _resolve_seed_hf_path(
    dataset_name: str, data_files: list[str], split: str = DEFAULT_SPLIT
) -> str | None:
    selected = _select_best_file(data_files, split)
    if not selected:
        return None

    ext = Path(selected).suffix.lower()
    if ext not in DATA_EXTS:
        return f"datasets/{dataset_name}/{selected}"

    parent = Path(selected).parent.as_posix()
    if not parent or parent == ".":
        return f"datasets/{dataset_name}/**/*{ext}"
    return f"datasets/{dataset_name}/{parent}/**/*{ext}"


def _build_stream_load_kwargs(
    *,
    dataset_name: str,
    split: str,
    subset: str | None,
    token: str | None,
    data_file: str | None = None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "path": dataset_name,
        "split": split,
        "streaming": True,
        "trust_remote_code": False,
    }
    if data_file:
        kwargs["data_files"] = [data_file]
    if subset:
        kwargs["name"] = subset
    if token:
        kwargs["token"] = token
    return kwargs


def _load_preview_rows(
    *,
    load_dataset_fn,
    load_kwargs: dict[str, Any],
    preview_size: int,
) -> list[dict[str, Any]]:
    streamed_ds = load_dataset_fn(**load_kwargs)
    return [row for row in islice(streamed_ds, preview_size)]


def _extract_columns(rows: list[dict[str, Any]]) -> list[str]:
    columns_seen: dict[str, None] = {}
    for row in rows:
        for key in row.keys():
            columns_seen[str(key)] = None
    return list(columns_seen.keys())


def _sanitize_filename(filename: str) -> str:
    name = Path(filename).name.strip().replace("\x00", "")
    if not name:
        return "seed_upload"
    return name


def _decode_base64_payload(content_base64: str) -> bytes:
    raw = content_base64.strip()
    if "," in raw and raw.lower().startswith("data:"):
        raw = raw.split(",", 1)[1]
    try:
        return base64.b64decode(raw, validate = True)
    except binascii.Error as exc:
        raise HTTPException(status_code = 400, detail = "invalid base64 payload") from exc


def _read_preview_rows_from_local_file(
    path: Path, preview_size: int
) -> list[dict[str, Any]]:
    try:
        import pandas as pd
    except ImportError as exc:
        raise HTTPException(
            status_code = 500, detail = f"seed inspect dependencies unavailable: {exc}"
        ) from exc

    ext = path.suffix.lower()
    try:
        if ext == ".csv":
            df = pd.read_csv(path, nrows = preview_size, encoding = "utf-8-sig")
            df.columns = df.columns.str.strip()
            unnamed = [c for c in df.columns if c == "" or c.startswith("Unnamed:")]
            if unnamed:
                df = df.drop(columns = unnamed)
                full_df = pd.read_csv(path, encoding = "utf-8-sig")
                full_df.columns = full_df.columns.str.strip()
                full_df = full_df.drop(columns = unnamed)
                tmp_csv = path.with_suffix(".tmp.csv")
                full_df.to_csv(tmp_csv, index = False, encoding = "utf-8")
                tmp_csv.replace(path)
        elif ext == ".jsonl":
            df = pd.read_json(path, lines = True).head(preview_size)
        elif ext == ".json":
            try:
                df = pd.read_json(path).head(preview_size)
            except ValueError:
                df = pd.read_json(path, lines = True).head(preview_size)
        else:
            raise HTTPException(status_code = 422, detail = f"unsupported file type: {ext}")
    except HTTPException:
        raise
    except (ValueError, OSError) as exc:
        raise HTTPException(
            status_code = 422, detail = f"seed inspect failed: {exc}"
        ) from exc

    rows = df.to_dict(orient = "records")
    return _serialize_preview_rows(rows)


def _read_preview_rows_from_unstructured_file(
    *,
    path: Path,
    preview_size: int,
    chunk_size: int | None,
    chunk_overlap: int | None,
) -> list[dict[str, Any]]:
    if resolve_chunking is None or build_unstructured_preview_rows is None:
        raise HTTPException(
            500,
            "Unstructured seed support not available (missing data_designer_unstructured_seed)",
        )
    size, overlap = resolve_chunking(chunk_size, chunk_overlap)
    try:
        rows = build_unstructured_preview_rows(
            source_path = path,
            preview_size = preview_size,
            chunk_size = size,
            chunk_overlap = overlap,
        )
    except (FileNotFoundError, RuntimeError, ValueError, OSError) as exc:
        raise HTTPException(
            status_code = 422, detail = f"seed inspect failed: {exc}"
        ) from exc
    return _serialize_preview_rows(rows)


def _read_preview_rows_from_multi_files(
    *,
    block_id: str,
    file_ids: list[str],
    file_names: list[str],
    preview_size: int,
    chunk_size: int | None,
    chunk_overlap: int | None,
) -> list[dict[str, str]]:
    if build_multi_file_preview_rows is None:
        raise HTTPException(
            500,
            "Unstructured seed support not available (missing data_designer_unstructured_seed)",
        )

    _validate_safe_id(block_id, "block_id")
    block_dir = UNSTRUCTURED_UPLOAD_ROOT / block_id
    file_entries: list[tuple[Path, str]] = []
    for fid, fname in zip(file_ids, file_names):
        extracted = block_dir / f"{fid}.extracted.txt"
        if not extracted.exists():
            raise HTTPException(
                404, f"Extracted text not found for file: {fname} (id: {fid})"
            )
        file_entries.append((extracted, fname))

    return build_multi_file_preview_rows(
        file_entries = file_entries,
        preview_size = preview_size,
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap,
    )


@router.post("/seed/inspect", response_model = SeedInspectResponse)
def inspect_seed_dataset(payload: SeedInspectRequest) -> SeedInspectResponse:
    dataset_name = payload.dataset_name.strip()
    if not dataset_name or dataset_name.count("/") < 1:
        raise HTTPException(
            status_code = 400,
            detail = "dataset_name must be a Hugging Face repo id like org/repo",
        )

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise HTTPException(
            status_code = 500, detail = f"seed inspect dependencies unavailable: {exc}"
        ) from exc

    split = _normalize_optional_text(payload.split) or DEFAULT_SPLIT
    subset = _normalize_optional_text(payload.subset)
    token = _normalize_optional_text(payload.hf_token)
    preview_size = int(payload.preview_size)

    preview_rows: list[dict[str, Any]] = []
    data_files = _list_hf_data_files(dataset_name = dataset_name, token = token)

    selected_file = _select_best_file(data_files, split)
    if selected_file:
        try:
            single_file_kwargs = _build_stream_load_kwargs(
                dataset_name = dataset_name,
                split = split,
                subset = subset,
                token = token,
                data_file = selected_file,
            )
            preview_rows = _load_preview_rows(
                load_dataset_fn = load_dataset,
                load_kwargs = single_file_kwargs,
                preview_size = preview_size,
            )
        except (ValueError, OSError, RuntimeError):
            preview_rows = []

    if not preview_rows:
        try:
            split_kwargs = _build_stream_load_kwargs(
                dataset_name = dataset_name,
                split = split,
                subset = subset,
                token = token,
            )
            preview_rows = _load_preview_rows(
                load_dataset_fn = load_dataset,
                load_kwargs = split_kwargs,
                preview_size = preview_size,
            )
        except (ValueError, OSError, RuntimeError) as exc:
            raise HTTPException(
                status_code = 422, detail = f"seed inspect failed: {exc}"
            ) from exc

    if not preview_rows:
        raise HTTPException(
            status_code = 422, detail = "dataset appears empty or unreadable"
        )
    preview_rows = _serialize_preview_rows(preview_rows)
    columns = _extract_columns(preview_rows)

    if not data_files:
        resolved_path = f"datasets/{dataset_name}/**/*.parquet"
    else:
        resolved_path = _resolve_seed_hf_path(dataset_name, data_files, split)
        if not resolved_path:
            raise HTTPException(
                status_code = 422, detail = "unable to resolve seed dataset path"
            )

    return SeedInspectResponse(
        dataset_name = dataset_name,
        resolved_path = resolved_path,
        columns = columns,
        preview_rows = preview_rows,
        split = split,
        subset = subset,
    )


def _extract_text_from_file(file_path: Path, ext: str) -> str:
    """Extract text from uploaded file based on extension, converting to markdown where possible."""
    if ext in {".txt", ".md"}:
        raw = file_path.read_text(encoding = "utf-8", errors = "ignore")
    elif ext == ".pdf":
        import pymupdf4llm

        raw = pymupdf4llm.to_markdown(
            str(file_path), write_images = False, show_progress = False, use_ocr = False
        )
    elif ext == ".docx":
        import mammoth

        with open(str(file_path), "rb") as f:
            result = mammoth.convert_to_markdown(f)
            raw = result.value
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    if normalize_unstructured_text is None:
        return raw
    return normalize_unstructured_text(raw)


def _get_block_total_size(block_dir: Path, file_ids: list[str]) -> int:
    """Sum raw upload sizes for tracked file IDs only."""
    if not block_dir.exists() or not file_ids:
        return 0
    id_set = set(file_ids)
    total = 0
    for f in block_dir.iterdir():
        if not f.is_file():
            continue
        if f.name.endswith(".extracted.txt") or f.name.endswith(".meta.json"):
            continue
        stem = f.name.split(".")[0]
        if stem in id_set:
            total += f.stat().st_size
    return total


@router.post("/seed/upload-unstructured-file")
async def upload_unstructured_file(
    file: UploadFile = FastAPIFile(...),
    block_id: str = Form(...),
    existing_file_ids: str = Form(""),
) -> UnstructuredFileUploadResponse:
    _validate_safe_id(block_id, "block_id")

    tracked_ids = [fid.strip() for fid in existing_file_ids.split(",") if fid.strip()]

    original_filename = file.filename or "upload"
    ext = Path(original_filename).suffix.lower()
    if ext not in UNSTRUCTURED_ALLOWED_EXTS:
        raise HTTPException(
            400,
            f"Unsupported file type: {ext}. Allowed: {', '.join(sorted(UNSTRUCTURED_ALLOWED_EXTS))}",
        )

    content = await file.read()
    size_bytes = len(content)

    if size_bytes == 0:
        raise HTTPException(400, "Empty file not allowed")

    if size_bytes > MAX_FILE_SIZE:
        raise HTTPException(
            413, f"File too large ({size_bytes} bytes). Maximum is 50MB."
        )

    block_dir = UNSTRUCTURED_UPLOAD_ROOT / block_id
    ensure_dir(block_dir)
    current_total = _get_block_total_size(block_dir, file_ids = tracked_ids)
    if current_total + size_bytes > MAX_TOTAL_SIZE:
        raise HTTPException(
            413, f"Total upload limit ({MAX_TOTAL_SIZE // (1024 * 1024)}MB) exceeded"
        )

    file_id = uuid4().hex
    raw_path = block_dir / f"{file_id}{ext}"
    raw_path.write_bytes(content)

    extracted_path = block_dir / f"{file_id}.extracted.txt"
    try:
        extracted_text = _extract_text_from_file(raw_path, ext)
        if not extracted_text or not extracted_text.strip():
            raw_path.unlink(missing_ok = True)
            return UnstructuredFileUploadResponse(
                file_id = file_id,
                filename = original_filename,
                size_bytes = size_bytes,
                status = "error",
                error = "No extractable text found in file",
            )
        extracted_path.write_text(extracted_text, encoding = "utf-8")
    except Exception as e:
        raw_path.unlink(missing_ok = True)
        extracted_path.unlink(missing_ok = True)
        return UnstructuredFileUploadResponse(
            file_id = file_id,
            filename = original_filename,
            size_bytes = size_bytes,
            status = "error",
            error = f"Text extraction failed: {type(e).__name__}: {e}",
        )

    try:
        meta_path = block_dir / f"{file_id}.meta.json"
        meta_path.write_text(
            json.dumps(
                {"original_filename": original_filename, "size_bytes": size_bytes}
            ),
            encoding = "utf-8",
        )
    except OSError:
        raw_path.unlink(missing_ok = True)
        extracted_path.unlink(missing_ok = True)
        return UnstructuredFileUploadResponse(
            file_id = file_id,
            filename = original_filename,
            size_bytes = size_bytes,
            status = "error",
            error = "Failed to save file metadata",
        )

    return UnstructuredFileUploadResponse(
        file_id = file_id,
        filename = original_filename,
        size_bytes = size_bytes,
        status = "ok",
    )


@router.delete("/seed/unstructured-file/{block_id}/{file_id}")
async def remove_unstructured_file(block_id: str, file_id: str):
    _validate_safe_id(block_id, "block_id")
    _validate_safe_id(file_id, "file_id")

    block_dir = UNSTRUCTURED_UPLOAD_ROOT / block_id
    if not block_dir.exists():
        raise HTTPException(404, "Block not found")

    deleted = False
    for f in block_dir.iterdir():
        stem = f.name.split(".")[0]
        if stem == file_id:
            f.unlink(missing_ok = True)
            deleted = True

    if not deleted:
        raise HTTPException(404, "File not found")
    try:
        if not any(block_dir.iterdir()):
            block_dir.rmdir()
    except OSError:
        pass

    return {"status": "ok"}


@router.post("/seed/inspect-upload", response_model = SeedInspectResponse)
def inspect_seed_upload(payload: SeedInspectUploadRequest) -> SeedInspectResponse:
    if payload.file_ids is not None:
        if len(payload.file_ids) == 0:
            raise HTTPException(400, "file_ids must not be empty")
        _validate_safe_id(payload.block_id, "block_id")
        for fid in payload.file_ids:
            _validate_safe_id(fid, "file_id")
        preview_rows = _read_preview_rows_from_multi_files(
            block_id = payload.block_id,
            file_ids = payload.file_ids,
            file_names = payload.file_names,
            preview_size = payload.preview_size,
            chunk_size = payload.unstructured_chunk_size,
            chunk_overlap = payload.unstructured_chunk_overlap,
        )
        columns = ["chunk_text", "source_file"] if preview_rows else []
        resolved_paths = [
            str(UNSTRUCTURED_UPLOAD_ROOT / payload.block_id / f"{fid}.extracted.txt")
            for fid in payload.file_ids
        ]
        return SeedInspectResponse(
            dataset_name = "unstructured_seed",
            resolved_path = resolved_paths[0] if resolved_paths else "",
            resolved_paths = resolved_paths,
            columns = columns,
            preview_rows = _serialize_preview_rows(preview_rows),
        )

    seed_source_type = _normalize_optional_text(payload.seed_source_type) or "local"
    filename = _sanitize_filename(payload.filename)
    ext = Path(filename).suffix.lower()
    # Legacy single-file unstructured path only supports .txt/.md
    # PDF/DOCX extraction uses the multi-file upload endpoint instead
    _LEGACY_UNSTRUCTURED_EXTS = {".txt", ".md"}
    if seed_source_type == "unstructured":
        if ext not in _LEGACY_UNSTRUCTURED_EXTS:
            allowed = ", ".join(sorted(_LEGACY_UNSTRUCTURED_EXTS))
            raise HTTPException(
                status_code = 400,
                detail = f"unsupported file type: {ext}. allowed: {allowed}",
            )
    else:
        if ext not in LOCAL_UPLOAD_EXTS:
            allowed = ", ".join(sorted(LOCAL_UPLOAD_EXTS))
            raise HTTPException(
                status_code = 400,
                detail = f"unsupported file type: {ext}. allowed: {allowed}",
            )

    file_bytes = _decode_base64_payload(payload.content_base64)
    if not file_bytes:
        raise HTTPException(status_code = 400, detail = "empty upload payload")
    if len(file_bytes) > MAX_FILE_SIZE:
        raise HTTPException(status_code = 413, detail = "file too large (max 50MB)")

    ensure_dir(SEED_UPLOAD_DIR)
    stored_name = f"{uuid4().hex}_{filename}"
    stored_path = SEED_UPLOAD_DIR / stored_name
    stored_path.write_bytes(file_bytes)

    if seed_source_type == "unstructured":
        preview_rows = _read_preview_rows_from_unstructured_file(
            path = stored_path,
            preview_size = int(payload.preview_size),
            chunk_size = payload.unstructured_chunk_size,
            chunk_overlap = payload.unstructured_chunk_overlap,
        )
    else:
        preview_rows = _read_preview_rows_from_local_file(
            stored_path,
            int(payload.preview_size),
        )
    if not preview_rows:
        raise HTTPException(
            status_code = 422, detail = "dataset appears empty or unreadable"
        )
    columns = _extract_columns(preview_rows)

    return SeedInspectResponse(
        dataset_name = filename,
        resolved_path = str(stored_path),
        columns = columns,
        preview_rows = preview_rows,
        split = None,
        subset = None,
    )


@router.get("/seed/github/env-token")
def get_github_env_token_status() -> dict:
    """Report whether the server has a GH_TOKEN / GITHUB_TOKEN env var.

    The value is never returned; the UI uses this to tell the user they
    can leave the token field blank.
    """
    has_token = bool(os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN"))
    return {"has_token": has_token}
