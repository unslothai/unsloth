# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Datasets API routes
"""

import asyncio
import base64
import io
import json
import os
import sys
import threading
import time
from collections import OrderedDict
from pathlib import Path
from uuid import uuid4
from typing import Optional
from fastapi import APIRouter, Body, Depends, Header, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field
import structlog
from loggers import get_logger


_dataset_size_cache: "OrderedDict[str, tuple[int, bool, str]]" = OrderedDict()
_dataset_size_neg_cache: "OrderedDict[str, float]" = OrderedDict()
_DATASET_SIZE_CACHE_MAX = 256
_DATASET_SIZE_NEG_TTL = 60.0
_DATASET_SIZE_TIMEOUT_SECONDS = 5.0
_dataset_size_cache_lock = threading.Lock()


def _get_dataset_size_cached(repo_id: str, hf_token: Optional[str] = None) -> int:
    token_fp = hf_cache_scan.token_fingerprint(hf_token)
    with _dataset_size_cache_lock:
        cached = _dataset_size_cache.get(repo_id)
        if cached is not None:
            size, restricted, cached_fp = cached
            # A gated/private repo's size is only served back to the token that
            # fetched it: a tokenless caller couldn't fetch it, and another token
            # may have no access to it at all.
            if not restricted or cached_fp == token_fp:
                _dataset_size_cache.move_to_end(repo_id)
                return size
        # A token may unlock a gated/private repo that failed anonymously, so a
        # tokened lookup ignores a prior negative-cache entry.
        if not hf_token:
            neg_ts = _dataset_size_neg_cache.get(repo_id)
            if neg_ts is not None and (time.monotonic() - neg_ts) < _DATASET_SIZE_NEG_TTL:
                return 0
    try:
        from huggingface_hub import HfApi

        info = HfApi(token = hf_token).dataset_info(
            repo_id,
            files_metadata = True,
            timeout = _DATASET_SIZE_TIMEOUT_SECONDS,
        )
        total = sum(s.size for s in info.siblings if getattr(s, "size", None))
        restricted = bool(
            getattr(info, "private", False) or getattr(info, "gated", False)
        )
    except Exception:
        if not hf_token:
            with _dataset_size_cache_lock:
                _dataset_size_neg_cache[repo_id] = time.monotonic()
                _dataset_size_neg_cache.move_to_end(repo_id)
                while len(_dataset_size_neg_cache) > _DATASET_SIZE_CACHE_MAX:
                    _dataset_size_neg_cache.popitem(last = False)
        return 0
    with _dataset_size_cache_lock:
        _dataset_size_cache[repo_id] = (total, restricted, token_fp)
        _dataset_size_cache.move_to_end(repo_id)
        _dataset_size_neg_cache.pop(repo_id, None)
        while len(_dataset_size_cache) > _DATASET_SIZE_CACHE_MAX:
            _dataset_size_cache.popitem(last = False)
    return total


# Add backend directory to path
backend_path = Path(__file__).parent.parent.parent
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))

# Import dataset utilities
from utils.datasets import check_dataset_format
from utils.datasets.cache_paths import (
    cached_dataset_candidates as _shared_cached_dataset_candidates,
    latest_cached_dataset_snapshot as _shared_latest_cached_dataset_snapshot,
)
from auth.authentication import get_current_subject

router = APIRouter()
logger = get_logger(__name__)


from models.datasets import (
    AiAssistMappingRequest,
    AiAssistMappingResponse,
    CheckFormatRequest,
    CheckFormatResponse,
    LocalDatasetItem,
    LocalDatasetsResponse,
    UploadDatasetResponse,
)
from utils.paths import (
    dataset_uploads_root,
    ensure_dir,
    is_valid_repo_id as _is_valid_repo_id,
    recipe_datasets_root,
    resolve_cached_repo_id_case,
    resolve_dataset_path,
)
from utils import hf_cache_scan


def _serialize_preview_value(value):
    """make it json safe for client preview ⊂(◉‿◉)つ"""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    try:
        from PIL.Image import Image as PILImage

        if isinstance(value, PILImage):
            buffer = io.BytesIO()
            value.convert("RGB").save(buffer, format = "JPEG", quality = 85)
            return {
                "type": "image",
                "mime": "image/jpeg",
                "width": value.width,
                "height": value.height,
                "data": base64.b64encode(buffer.getvalue()).decode("ascii"),
            }
    except Exception:
        pass

    if isinstance(value, dict):
        return {str(key): _serialize_preview_value(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return [_serialize_preview_value(item) for item in value]

    return str(value)


def _serialize_preview_rows(rows):
    return [
        {str(key): _serialize_preview_value(value) for key, value in dict(row).items()}
        for row in rows
    ]


# --- Endpoints ---

# Recognized data-file extensions for the single-file fallback approach.
# Tabular formats are preferred over archives for Tier 1 preview because
# archives (e.g. images.zip) may be loaded as ImageFolder datasets with
# synthetic columns (image/label) that don't match the real dataset schema.
_TABULAR_EXTS = (".parquet", ".json", ".jsonl", ".csv", ".tsv", ".arrow")
_ARCHIVE_EXTS = (".tar", ".tar.gz", ".tgz", ".gz", ".zst", ".zip", ".txt")
DATA_EXTS = _TABULAR_EXTS + _ARCHIVE_EXTS
LOCAL_FILE_EXTS = (".json", ".jsonl", ".csv", ".parquet")
LOCAL_UPLOAD_EXTS = {".csv", ".json", ".jsonl", ".parquet"}
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


def _load_local_preview_slice(
    *, dataset_path: Path, train_split: str, preview_size: int
):
    from datasets import load_dataset

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
        else:
            candidate_files: list[Path] = []
            for ext in LOCAL_FILE_EXTS:
                candidate_files.extend(sorted(dataset_path.glob(f"*{ext}")))
            if not candidate_files:
                raise HTTPException(
                    status_code = 400,
                    detail = "Unsupported local dataset directory (expected parquet/json/jsonl/csv files)",
                )
            dataset_path = candidate_files[0]

    if dataset_path.suffix in [".json", ".jsonl"]:
        dataset = load_dataset("json", data_files = str(dataset_path), split = train_split)
    elif dataset_path.suffix == ".csv":
        dataset = load_dataset("csv", data_files = str(dataset_path), split = train_split)
    elif dataset_path.suffix == ".parquet":
        dataset = load_dataset(
            "parquet", data_files = str(dataset_path), split = train_split
        )
    else:
        raise HTTPException(
            status_code = 400, detail = f"Unsupported file format: {dataset_path.suffix}"
        )

    total_rows = len(dataset)
    preview_slice = dataset.select(range(min(preview_size, total_rows)))
    return preview_slice, total_rows


def _latest_cached_dataset_snapshot(
    repo_id: str,
    local_path: Optional[str] = None,
) -> Optional[Path]:
    return _shared_latest_cached_dataset_snapshot(repo_id, local_path)


def _cached_dataset_candidates(
    snapshot: Path,
    *,
    subset: Optional[str],
    train_split: str,
) -> list[Path]:
    return _shared_cached_dataset_candidates(
        snapshot,
        subset = subset,
        train_split = train_split,
        extensions = DATA_EXTS,
        preferred_extensions = _TABULAR_EXTS,
    )


def _load_cached_file_preview_slice(path: Path, preview_size: int):
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


def _load_cached_hf_preview_slice(
    request: CheckFormatRequest,
    preview_size: int,
):
    if not _is_valid_repo_id(request.dataset_name):
        return None
    snapshot = _latest_cached_dataset_snapshot(
        request.dataset_name,
        request.local_path,
    )
    if snapshot is None:
        return None
    train_split = request.train_split or "train"
    for candidate in _cached_dataset_candidates(
        snapshot,
        subset = request.subset,
        train_split = train_split,
    ):
        try:
            preview = _load_cached_file_preview_slice(candidate, preview_size)
        except Exception as exc:
            logger.debug("Cached dataset preview failed for %s: %s", candidate, exc)
            continue
        if preview is not None:
            return preview
    return None


def _load_processed_hf_preview_slice(
    request: CheckFormatRequest,
    preview_size: int,
):
    if not _is_valid_repo_id(request.dataset_name):
        return None
    try:
        from datasets import DownloadConfig, load_dataset
    except Exception:
        return None

    load_kwargs = {
        "path": request.dataset_name,
        "split": request.train_split or "train",
        "download_config": DownloadConfig(local_files_only = True),
    }
    if request.subset:
        load_kwargs["name"] = request.subset
    if request.hf_token:
        load_kwargs["token"] = request.hf_token

    dataset = load_dataset(**load_kwargs)
    total_rows = len(dataset)
    preview_slice = dataset.select(range(min(preview_size, total_rows)))
    return preview_slice, total_rows


def _load_any_cached_hf_preview_slice(
    request: CheckFormatRequest,
    preview_size: int,
):
    cached_preview = _load_cached_hf_preview_slice(request, preview_size)
    if cached_preview is not None:
        return cached_preview
    try:
        return _load_processed_hf_preview_slice(request, preview_size)
    except Exception as exc:
        logger.debug(
            "Processed dataset cache preview failed for %s: %s",
            request.dataset_name,
            exc,
        )
        return None


def _sanitize_filename(filename: str) -> str:
    name = Path(filename).name.strip().replace("\x00", "")
    if not name:
        return "dataset_upload"
    return name


@router.post("/upload", response_model = UploadDatasetResponse)
async def upload_dataset(
    file: UploadFile,
    current_subject: str = Depends(get_current_subject),
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

    # Stream file to disk in chunks to avoid holding entire file in memory
    with open(stored_path, "wb") as f:
        while chunk := await file.read(1024 * 1024):
            f.write(chunk)

    if stored_path.stat().st_size == 0:
        stored_path.unlink(missing_ok = True)
        raise HTTPException(status_code = 400, detail = "Empty upload payload")

    return UploadDatasetResponse(filename = filename, stored_path = str(stored_path))


@router.get("/local", response_model = LocalDatasetsResponse)
def list_local_datasets(
    current_subject: str = Depends(get_current_subject),
) -> LocalDatasetsResponse:
    return LocalDatasetsResponse(datasets = _build_local_dataset_items())


def _collect_hf_cache_scans() -> tuple[list, set[str]]:
    scans = hf_cache_scan.all_hf_cache_scans()
    seen_roots = {
        str(cache_dir)
        for cache_dir in (getattr(scan, "cache_dir", None) for scan in scans)
        if cache_dir is not None
    }
    return scans, seen_roots


def _hf_hub_cache_roots() -> list[Path]:
    """Return Hub cache roots that may contain ``datasets--owner--repo`` dirs."""
    from utils.paths import legacy_hf_cache_dir, hf_default_cache_dir

    roots: list[Path] = []
    seen: set[str] = set()

    def _add(path: Optional[Path]) -> None:
        if path is None or not path.is_dir():
            return
        try:
            resolved = str(path.resolve())
        except OSError:
            return
        if resolved in seen:
            return
        seen.add(resolved)
        roots.append(path)

    try:
        from huggingface_hub.constants import HF_HUB_CACHE

        _add(Path(HF_HUB_CACHE))
    except Exception:
        pass

    hf_hub_cache = os.environ.get("HF_HUB_CACHE")
    if hf_hub_cache:
        _add(Path(hf_hub_cache).expanduser())

    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        _add(Path(hf_home).expanduser() / "hub")

    _add(legacy_hf_cache_dir())
    _add(hf_default_cache_dir())
    return roots


def _repo_id_from_hub_dataset_dir(name: str) -> str | None:
    if not name.startswith("datasets--"):
        return None
    encoded = name.removeprefix("datasets--")
    owner, sep, repo = encoded.partition("--")
    if not sep or not owner or not repo:
        return None
    repo_id = f"{owner}/{repo}"
    return repo_id if _is_valid_repo_id(repo_id) else None


def _directory_size(path: Path) -> int:
    total = 0
    try:
        for entry in path.rglob("*"):
            try:
                if entry.is_file() and not entry.is_symlink():
                    total += entry.stat().st_size
            except OSError:
                continue
    except OSError:
        return 0
    return total


def _hub_dataset_snapshot_count(path: Path) -> int:
    snapshots = path / "snapshots"
    try:
        return sum(1 for entry in snapshots.iterdir() if entry.is_dir())
    except OSError:
        return 0


def _scan_hub_dataset_cache_dirs() -> list[dict]:
    """Fallback scanner for HF Hub dataset cache directories.

    ``scan_cache_dir()`` can fail or skip repos when one cache entry is partially
    corrupt. The model scanner has several fallback paths already; datasets need
    the same resilience so the On Device tab reflects what is actually on disk.
    """
    seen_lower: dict[str, dict] = {}
    for root in _hf_hub_cache_roots():
        try:
            entries = [entry for entry in root.iterdir() if entry.is_dir()]
        except OSError:
            continue
        for entry in entries:
            repo_id = _repo_id_from_hub_dataset_dir(entry.name)
            if repo_id is None:
                continue
            size_bytes = _directory_size(entry / "blobs")
            if size_bytes <= 0:
                size_bytes = _directory_size(entry)
            if size_bytes <= 0:
                continue
            key = repo_id.lower()
            existing = seen_lower.get(key)
            row = {
                "repo_id": repo_id,
                "size_bytes": size_bytes,
                "cache_path": str(entry.resolve()),
                "partial": _hub_dataset_snapshot_count(entry) == 0,
            }
            if existing is None or size_bytes > existing["size_bytes"]:
                seen_lower[key] = row
    return sorted(seen_lower.values(), key = lambda c: c["repo_id"])


def _hf_datasets_cache_roots() -> list[Path]:
    roots: list[Path] = []
    seen: set[str] = set()

    def _add(path: Optional[Path]) -> None:
        if path is None or not path.is_dir():
            return
        try:
            resolved = str(path.resolve())
        except OSError:
            return
        if resolved in seen:
            return
        seen.add(resolved)
        roots.append(path)

    env_cache = os.environ.get("HF_DATASETS_CACHE")
    if env_cache:
        _add(Path(env_cache).expanduser())

    try:
        from datasets import config as datasets_config

        _add(Path(datasets_config.HF_DATASETS_CACHE))
    except Exception:
        pass

    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        _add(Path(hf_home).expanduser() / "datasets")

    xdg_cache = Path(
        os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")
    ).expanduser()
    _add(xdg_cache / "huggingface" / "datasets")
    return roots


def _repo_id_from_datasets_cache_dir(name: str) -> str | None:
    if "___" not in name:
        return None
    owner, repo = name.split("___", 1)
    repo_id = f"{owner}/{repo}"
    return repo_id if _is_valid_repo_id(repo_id) else None


def _processed_dataset_cache_size(path: Path) -> int:
    total = 0
    try:
        for entry in path.rglob("*"):
            try:
                if entry.is_file():
                    total += entry.stat().st_size
            except OSError:
                continue
    except OSError:
        return 0
    return total


def _looks_like_processed_dataset_cache(path: Path) -> bool:
    try:
        for entry in path.rglob("*"):
            if not entry.is_file():
                continue
            if entry.name in {"dataset_info.json", "state.json"}:
                return True
            if entry.suffix == ".arrow":
                return True
    except OSError:
        return False
    return False


def _scan_processed_dataset_caches() -> list[dict]:
    """Return HF datasets-library cache rows keyed by repo_id.

    `datasets.load_dataset()` stores processed Arrow caches separately from the
    Hub snapshot cache. Older Studio training runs can therefore have usable
    on-device datasets that `huggingface_hub.scan_cache_dir()` never reports.
    """
    seen_lower: dict[str, dict] = {}
    for root in _hf_datasets_cache_roots():
        try:
            entries = [entry for entry in root.iterdir() if entry.is_dir()]
        except OSError:
            continue
        for entry in entries:
            repo_id = _repo_id_from_datasets_cache_dir(entry.name)
            if repo_id is None:
                continue
            if not _looks_like_processed_dataset_cache(entry):
                continue
            size_bytes = _processed_dataset_cache_size(entry)
            if size_bytes <= 0:
                continue
            key = repo_id.lower()
            existing = seen_lower.get(key)
            if existing is None or size_bytes > existing["size_bytes"]:
                seen_lower[key] = {
                    "repo_id": repo_id,
                    "size_bytes": size_bytes,
                    "cache_path": str(entry.resolve()),
                    "processed_cache": True,
                    "partial": False,
                }
    return sorted(seen_lower.values(), key = lambda c: c["repo_id"])


def _scan_hf_dataset_caches() -> list[dict]:
    """Walk active + legacy + default HF caches; return one row per cached dataset repo."""
    scans, seen_roots = _collect_hf_cache_scans()

    seen_lower: dict[str, dict] = {}
    inspected = 0
    for hf_cache in scans:
        for repo_info in hf_cache.repos:
            inspected += 1
            try:
                # repo_type is an enum-like string ("dataset"/"model"/"space")
                # — compare against str(...) to avoid quirks if the library
                # ever switches to an Enum.
                if str(repo_info.repo_type) != "dataset":
                    continue
                total_size = int(getattr(repo_info, "size_on_disk", 0) or 0)
                if total_size == 0:
                    total_size = sum(
                        int(f.size_on_disk or 0)
                        for rev in repo_info.revisions
                        for f in rev.files
                    )
                key = repo_info.repo_id.lower()
                existing = seen_lower.get(key)
                if existing is None or total_size > existing["size_bytes"]:
                    seen_lower[key] = {
                        "repo_id": repo_info.repo_id,
                        "size_bytes": total_size,
                        "cache_path": str(repo_info.repo_path),
                    }
            except Exception as exc:
                label = getattr(repo_info, "repo_id", "<unknown>")
                logger.warning("Skipping cached dataset repo %s: %s", label, exc)

    partial = hf_cache_scan.partial_repo_ids(
        "dataset", (row["repo_id"] for row in seen_lower.values()),
    )
    for row in seen_lower.values():
        row["partial"] = row["repo_id"] in partial
    for row in _scan_hub_dataset_cache_dirs():
        key = row["repo_id"].lower()
        existing = seen_lower.get(key)
        if existing is None:
            seen_lower[key] = row
        else:
            existing["size_bytes"] = max(existing["size_bytes"], row["size_bytes"])
            existing["cache_path"] = existing.get("cache_path") or row.get("cache_path")
            existing["partial"] = bool(existing.get("partial")) or bool(
                row.get("partial")
            )
    for row in _scan_processed_dataset_caches():
        key = row["repo_id"].lower()
        existing = seen_lower.get(key)
        if existing is None:
            seen_lower[key] = row
        else:
            existing["size_bytes"] = max(existing["size_bytes"], row["size_bytes"])
    logger.info(
        "Cached dataset scan: roots=%d inspected=%d returned=%d",
        len(seen_roots) or len(scans), inspected, len(seen_lower),
    )
    return sorted(seen_lower.values(), key = lambda c: c["repo_id"])


@router.get("/cached")
async def list_cached_datasets(
    current_subject: str = Depends(get_current_subject),
):
    """List dataset repos already downloaded into the HF cache."""
    try:
        return {"cached": await asyncio.to_thread(_scan_hf_dataset_caches)}
    except Exception as exc:
        logger.error("Error listing cached datasets: %s", exc, exc_info = True)
        raise HTTPException(
            status_code = 500,
            detail = "Failed to read the local dataset cache.",
        ) from exc


@router.delete("/cached")
async def delete_cached_dataset(
    repo_id: str = Body(..., embed = True),
    current_subject: str = Depends(get_current_subject),
):
    """Remove a cached dataset repo from the HF cache."""
    if not _is_valid_repo_id(repo_id):
        raise HTTPException(status_code = 400, detail = "Invalid repo_id format")

    repo_key = resolve_cached_repo_id_case(repo_id, repo_type = "dataset")
    if not _registry.begin_delete(repo_key):
        raise HTTPException(
            status_code = 400,
            detail = "Cancel the active download before deleting.",
        )
    try:
        return await asyncio.to_thread(_delete_cached_dataset_blocking, repo_id)
    finally:
        _registry.end_delete(repo_key)
        hf_cache_scan.invalidate_hf_cache_scans()


def _delete_cached_dataset_blocking(repo_id: str) -> dict:
    scans, _seen_roots = _collect_hf_cache_scans()

    deleted = False
    for hf_cache in scans:
        for repo_info in hf_cache.repos:
            if str(repo_info.repo_type) != "dataset":
                continue
            if repo_info.repo_id.lower() != repo_id.lower():
                continue
            try:
                strategy = hf_cache.delete_revisions(
                    *(rev.commit_hash for rev in repo_info.revisions)
                )
                strategy.execute()
                deleted = True
            except Exception as exc:
                logger.error(
                    "Failed deleting cached dataset %s: %s", repo_id, exc,
                )
                raise HTTPException(
                    status_code = 500,
                    detail = f"Failed to delete dataset: {exc}",
                ) from exc

    processed_deleted = _delete_processed_dataset_cache(repo_id)
    if not deleted and not processed_deleted:
        if hf_cache_scan.purge_partial_repo("dataset", repo_id):
            return {"status": "deleted", "repo_id": repo_id}
        raise HTTPException(status_code = 404, detail = "Dataset not found in cache")
    return {"status": "deleted", "repo_id": repo_id}


def _delete_processed_dataset_cache(repo_id: str) -> bool:
    import shutil

    target = repo_id.replace("/", "___").lower()
    deleted = False
    for root in _hf_datasets_cache_roots():
        try:
            entries = [entry for entry in root.iterdir() if entry.is_dir()]
        except OSError:
            continue
        for entry in entries:
            if entry.name.lower() != target:
                continue
            try:
                shutil.rmtree(entry)
                deleted = True
            except Exception as exc:
                logger.error(
                    "Failed deleting processed dataset cache %s: %s",
                    repo_id,
                    exc,
                )
                raise HTTPException(
                    status_code = 500,
                    detail = f"Failed to delete dataset: {exc}",
                ) from exc
    return deleted


@router.get("/download-progress")
async def get_dataset_download_progress(
    repo_id: str = Query(
        ..., description = "HuggingFace dataset repo ID, e.g. 'unsloth/LaTeX_OCR'"
    ),
    expected_bytes: int = Query(
        0, description = "Expected total download size in bytes",
    ),
    hf_token: Optional[str] = Header(None, alias = "X-HF-Token"),
    current_subject: str = Depends(get_current_subject),
):
    """Return download progress for a HuggingFace dataset repo.

    Mirrors ``GET /api/models/download-progress`` but scans the
    ``datasets--owner--name`` cache directory under HF_HUB_CACHE.
    Modern ``datasets``/``huggingface_hub`` caches both raw model and
    raw dataset blobs in HF_HUB_CACHE; the ``datasets`` library writes
    its processed Arrow shards elsewhere, but the in-progress *download*
    bytes are observable here. Returns ``cache_path`` so the UI can
    show users where the dataset blobs landed on disk.
    """
    _empty = {
        "downloaded_bytes": 0,
        "expected_bytes": max(expected_bytes, 0),
        "progress": 0,
        "cache_path": None,
    }
    def _compute():
        if not _is_valid_repo_id(repo_id):
            return _empty

        force_active = _registry.get_job(repo_id).state in {"running", "cancelling"}
        readings: list[hf_cache_scan.CacheProgressReading] = []

        for entry in hf_cache_scan.preferred_repo_cache_dirs(
            "dataset",
            repo_id,
            force_active = force_active,
        ):
            completed_bytes = 0
            in_progress_bytes = 0
            cache_path = hf_cache_scan.resolve_hf_cache_realpath(entry)
            blobs_dir = entry / "blobs"
            if blobs_dir.is_dir():
                try:
                    blob_entries = list(blobs_dir.iterdir())
                except OSError:
                    blob_entries = []
                for f in blob_entries:
                    # Skip a blob that vanished mid-poll (renamed out of
                    # *.incomplete) rather than zeroing the whole reading.
                    try:
                        if not f.is_file():
                            continue
                        if f.name.endswith(".incomplete"):
                            in_progress_bytes += f.stat().st_size
                        else:
                            completed_bytes += f.stat().st_size
                    except OSError:
                        continue
            readings.append((completed_bytes, in_progress_bytes, cache_path))

        selected_reading = hf_cache_scan.select_best_cache_progress(readings)
        if selected_reading is None:
            return _empty

        completed_bytes, in_progress_bytes, cache_path = selected_reading
        downloaded_bytes = completed_bytes + in_progress_bytes
        if downloaded_bytes == 0:
            return {**_empty, "cache_path": cache_path}

        expected_total = max(expected_bytes, 0)
        if expected_total <= 0:
            expected_total = _get_dataset_size_cached(repo_id, hf_token)
        if expected_total <= 0:
            return {
                "downloaded_bytes": downloaded_bytes,
                "expected_bytes": 0,
                "progress": 0,
                "cache_path": cache_path,
            }

        # Same 95% completion threshold as the model endpoint -- HF blob
        # dedup makes completed_bytes drift slightly under expected_bytes,
        # and inter-file gaps would otherwise look like "done".
        if completed_bytes >= expected_total * 0.95:
            progress = 1.0
        else:
            progress = min(downloaded_bytes / expected_total, 0.99)
        return {
            "downloaded_bytes": downloaded_bytes,
            "expected_bytes": expected_total,
            "progress": round(progress, 3),
            "cache_path": cache_path,
        }

    try:
        return await asyncio.to_thread(_compute)
    except Exception as e:
        logger.warning(f"Error checking dataset download progress for {repo_id}: {e}")
        return _empty


class DownloadDatasetRequest(BaseModel):
    """Body for ``POST /api/datasets/download``.

    The HuggingFace token travels in the ``X-HF-Token`` request header so it
    never appears in browser devtools payload tabs, request-body access logs,
    or exception reporters that capture request bodies.
    """

    repo_id: str = Field(..., description = "HuggingFace dataset repo ID")
    use_xet: bool = Field(
        False,
        description = "Enable Xet parallel chunked transport. Default False uses HTTP Range-resume.",
    )


class DatasetDownloadJobStatus(BaseModel):
    """Live state of a background dataset download job."""

    state: str = Field(
        ...,
        description = "'idle' | 'running' | 'complete' | 'error' | 'cancelled'",
    )
    error: Optional[str] = Field(None, description = "Error message if state == 'error'")


_registry = hf_cache_scan.DownloadRegistry()


def _dataset_status(repo_id: str) -> DatasetDownloadJobStatus:
    state = _registry.get_job(repo_id)
    return DatasetDownloadJobStatus(state = state.state, error = state.error)


def terminate_active_dataset_downloads() -> None:
    _registry.terminate_all("dataset download")


@router.post("/download", status_code = 202)
async def download_dataset(
    body: DownloadDatasetRequest,
    hf_token: Optional[str] = Header(None, alias = "X-HF-Token", max_length = 512),
    current_subject: str = Depends(get_current_subject),
):
    """Start a background download for a HuggingFace dataset."""
    repo_id = body.repo_id.strip()
    if not _is_valid_repo_id(repo_id):
        raise HTTPException(
            status_code = 400,
            detail = f"Invalid repo_id: {repo_id!r}",
        )
    # Canonicalize so two different-cased paste-ins share one job + cache dir.
    repo_id = resolve_cached_repo_id_case(repo_id, repo_type = "dataset")

    transport = (
        hf_cache_scan.TRANSPORT_XET if body.use_xet else hf_cache_scan.TRANSPORT_HTTP
    )
    unavailable_reason = hf_cache_scan.download_transport_unavailable_reason(transport)
    if unavailable_reason is not None:
        raise HTTPException(status_code = 400, detail = unavailable_reason)

    claimed, claim_state = _registry.claim(repo_id, transport)
    generation = _registry.current_generation(repo_id)
    if not claimed:
        # A rejected claim for this repo's own in-flight job is pollable; one
        # blocked by an in-progress delete leaves no job, so flag it.
        return {
            "repo_id": repo_id,
            "state": claim_state,
            "accepted": _registry.adoptable(repo_id),
            "generation": generation,
        }

    backend_dir = Path(__file__).resolve().parent.parent
    try:
        proc = hf_cache_scan.spawn_worker(
            ["--repo-id", repo_id, "--dataset"],
            hf_token,
            backend_dir,
            use_xet = body.use_xet,
        )
    except Exception as e:
        scrubbed = hf_cache_scan.scrub_secrets(str(e), hf_token = hf_token)
        logger.error(f"Failed to spawn dataset download worker for {repo_id}: {scrubbed}", exc_info = True)
        _registry.set_job(repo_id, "error", scrubbed)
        raise HTTPException(
            status_code = 500,
            detail = f"Failed to start dataset download: {scrubbed}",
        ) from e

    if not _registry.register_process(repo_id, proc):
        hf_cache_scan.kill_and_reap_process(
            proc,
            label = f"dataset {repo_id}",
            logger = logger,
        )
    else:
        worker_token = hf_token
        def _watch() -> None:
            hf_cache_scan.finalize_worker_exit(
                _registry,
                repo_id,
                proc,
                hf_token = worker_token,
                label = repo_id,
                log_prefix = "Dataset download",
                logger = logger,
                repo_type = None if body.use_xet else "dataset",
                repo_id = None if body.use_xet else repo_id,
            )
            if _registry.get_job(repo_id).state in ("error", "cancelled"):
                hf_cache_scan.purge_empty_marker_dir("dataset", repo_id)
            hf_cache_scan.invalidate_hf_cache_scans()

        threading.Thread(
            target = _watch,
            name = f"hf-dataset-download-watch-{repo_id}",
            daemon = True,
        ).start()

    return {
        "repo_id": repo_id,
        "state": _registry.get_job(repo_id).state,
        "accepted": True,
        "generation": generation,
    }


class CancelDatasetDownloadRequest(BaseModel):
    repo_id: str = Field(..., description = "HuggingFace dataset repo ID")
    generation: Optional[int] = Field(None, description = "Download generation")


@router.post("/download/cancel", status_code = 202)
async def cancel_dataset_download(
    body: CancelDatasetDownloadRequest,
    current_subject: str = Depends(get_current_subject),
):
    """Cancel an in-flight dataset download (SIGKILL; HF cache resumes on next download)."""
    repo_id = body.repo_id.strip()
    if not _is_valid_repo_id(repo_id):
        raise HTTPException(
            status_code = 400, detail = f"Invalid repo_id: {repo_id!r}",
        )
    repo_id = resolve_cached_repo_id_case(repo_id, repo_type = "dataset")

    proc = _registry.get_process(repo_id)
    if proc is None or proc.poll() is not None:
        # The worker may be mid-spawn (claim→register window): arm a pending
        # cancel so register_process kills it on arrival.
        if _registry.mark_pending_cancel(repo_id, body.generation):
            return {"repo_id": repo_id, "state": "cancelling"}
        return {"repo_id": repo_id, "state": _registry.get_job(repo_id).state}

    if not _registry.request_cancel(repo_id, proc, body.generation):
        return {"repo_id": repo_id, "state": _registry.get_job(repo_id).state}

    try:
        proc.kill()
    except ProcessLookupError:
        pass
    except Exception as e:
        logger.warning(f"Cancel SIGKILL for dataset {repo_id} failed: {e}")

    return {"repo_id": repo_id, "state": "cancelling"}


@router.get("/download-status", response_model = DatasetDownloadJobStatus)
async def get_dataset_download_status(
    repo_id: str = Query(..., description = "HuggingFace dataset repo ID"),
    current_subject: str = Depends(get_current_subject),
):
    """Return the latest state of a background dataset download job."""
    repo_id = repo_id.strip()
    if not _is_valid_repo_id(repo_id):
        return DatasetDownloadJobStatus(state = "idle")
    repo_id = resolve_cached_repo_id_case(repo_id, repo_type = "dataset")
    return _dataset_status(repo_id)


@router.get("/transport-status")
async def get_dataset_transport_status(
    repo_id: str = Query(..., description = "HuggingFace dataset repo ID"),
    current_subject: str = Depends(get_current_subject),
):
    """Return last transport used for this dataset + whether any partial
    blobs exist + whether that partial supports byte-level resume.

    See ``models.get_model_transport_status`` for the semantics of
    ``resumable`` — XET partials are reported via ``has_partial`` but
    are not byte-level resumable.
    """
    repo_id = repo_id.strip()
    if not _is_valid_repo_id(repo_id):
        return {"has_partial": False, "last_transport": None, "resumable": False}
    return {
        "has_partial": hf_cache_scan.has_active_incomplete_blobs("dataset", repo_id),
        "last_transport": hf_cache_scan.read_active_transport_marker("dataset", repo_id),
        "resumable": hf_cache_scan.is_resumable_partial("dataset", repo_id),
    }


@router.post("/check-format", response_model = CheckFormatResponse)
def check_format(
    request: CheckFormatRequest,
    current_subject: str = Depends(get_current_subject),
):
    """
    Check if a dataset requires manual column mapping.

    Strategy for HuggingFace datasets:
      1. list_repo_files → pick the first data file → load_dataset(data_files=[…])
         Avoids resolving thousands of files; typically ~2-4 s.
      2. Full streaming load_dataset as a last-resort fallback.

    Local files are loaded directly.

    Using a plain `def` (not async) so FastAPI runs this in a thread-pool,
    preventing any blocking IO from freezing the event loop.
    """
    try:
        from itertools import islice
        from datasets import Dataset, load_dataset
        from utils.datasets import format_dataset

        PREVIEW_SIZE = 10

        logger.info(f"Checking format for dataset: {request.dataset_name}")

        dataset_path = resolve_dataset_path(request.dataset_name)
        total_rows = None

        if dataset_path.exists():
            # ── Local file ──────────────────────────────────────────
            train_split = request.train_split or "train"
            preview_slice, total_rows = _load_local_preview_slice(
                dataset_path = dataset_path,
                train_split = train_split,
                preview_size = PREVIEW_SIZE,
            )
        else:
            # ── HuggingFace dataset ─────────────────────────────────
            # Tier 1: list_repo_files → load only the first data file
            cached_preview = (
                _load_any_cached_hf_preview_slice(request, PREVIEW_SIZE)
                if request.prefer_local_cache
                else None
            )
            if cached_preview is not None:
                preview_slice, total_rows = cached_preview
            elif request.prefer_local_cache:
                raise HTTPException(
                    status_code = 404,
                    detail = "Dataset is not available in the local cache.",
                )
            else:
                preview_slice = None

                try:
                    from huggingface_hub import HfApi

                    api = HfApi()
                    repo_files = api.list_repo_files(
                        request.dataset_name,
                        repo_type = "dataset",
                        token = request.hf_token or None,
                    )
                    data_files = [
                        f for f in repo_files if any(f.endswith(ext) for ext in DATA_EXTS)
                    ]

                    # Prefer tabular formats over archives (e.g. images.zip → ImageFolder
                    # with synthetic image/label columns that don't match the real schema).
                    tabular_files = [
                        f
                        for f in data_files
                        if any(f.endswith(ext) for ext in _TABULAR_EXTS)
                    ]
                    candidates = tabular_files or data_files

                    # When a subset is specified, narrow to files whose name matches
                    # (e.g. subset="testmini" → prefer "testmini.parquet").
                    if request.subset and candidates:
                        subset_matches = [
                            f for f in candidates if request.subset in Path(f).stem
                        ]
                        if subset_matches:
                            candidates = subset_matches

                    if candidates:
                        first_file = candidates[0]
                        logger.info(f"Tier 1: loading single file {first_file}")
                        load_kwargs = {
                            "path": request.dataset_name,
                            "data_files": [first_file],
                            "split": "train",
                            "streaming": True,
                        }
                        if request.hf_token:
                            load_kwargs["token"] = request.hf_token

                        streamed_ds = load_dataset(**load_kwargs)
                        rows = list(islice(streamed_ds, PREVIEW_SIZE))
                        if rows:
                            preview_slice = Dataset.from_list(rows)
                except Exception as e:
                    logger.warning(f"Tier 1 (single-file) failed: {e}")

            if preview_slice is None:
                # Tier 2: full streaming (resolves all files — slow for large repos)
                logger.info("Tier 2: falling back to full streaming load_dataset")
                try:
                    load_kwargs = {
                        "path": request.dataset_name,
                        "split": request.train_split,
                        "streaming": True,
                    }
                    if request.subset:
                        load_kwargs["name"] = request.subset
                    if request.hf_token:
                        load_kwargs["token"] = request.hf_token

                    streamed_ds = load_dataset(**load_kwargs)

                    rows = list(islice(streamed_ds, PREVIEW_SIZE))
                    if not rows:
                        raise HTTPException(
                            status_code = 400,
                            detail = "Dataset appears to be empty or could not be streamed",
                        )

                    preview_slice = Dataset.from_list(rows)
                    total_rows = None
                except Exception:
                    cached_preview = _load_any_cached_hf_preview_slice(
                        request,
                        PREVIEW_SIZE,
                    )
                    if cached_preview is None:
                        raise
                    preview_slice, total_rows = cached_preview

        # Run lightweight format check on the preview slice
        result = check_dataset_format(preview_slice, is_vlm = request.is_vlm)

        logger.info(
            f"Format check result: requires_mapping={result['requires_manual_mapping']}, format={result['detected_format']}, is_image={result.get('is_image', False)}"
        )

        # Generate preview samples
        preview_samples = None
        if not result["requires_manual_mapping"]:
            if result.get("suggested_mapping"):
                # Heuristic-detected: show raw data so columns match the API response.
                # Processing (column stripping) happens at training time, not preview.
                preview_samples = _serialize_preview_rows(preview_slice)
            else:
                try:
                    format_result = format_dataset(
                        preview_slice,
                        format_type = "auto",
                        num_proc = None,  # Only 10 preview rows -- no need for multiprocessing
                    )
                    processed = format_result["dataset"]
                    preview_samples = _serialize_preview_rows(processed)
                except Exception as e:
                    logger.warning(
                        f"Processed preview generation failed (non-fatal): {e}"
                    )
                    preview_samples = _serialize_preview_rows(preview_slice)
        else:
            preview_samples = _serialize_preview_rows(preview_slice)

        # Collect warnings: from check_dataset_format + URL-based image detection
        warning = result.get("warning")
        image_col = result.get("detected_image_column")
        if image_col and image_col in (result.get("columns") or []):
            try:
                sample_val = preview_slice[0][image_col]
                if isinstance(sample_val, str) and sample_val.startswith(
                    ("http://", "https://")
                ):
                    url_warning = (
                        "This dataset contains image URLs instead of embedded images. "
                        "Images will be downloaded during training, which may be slow for large datasets."
                    )
                    logger.info(f"URL-based image column detected: {image_col}")
                    warning = f"{warning} {url_warning}" if warning else url_warning
            except Exception:
                pass

        return CheckFormatResponse(
            requires_manual_mapping = result["requires_manual_mapping"],
            detected_format = result["detected_format"],
            columns = result["columns"],
            is_image = result.get("is_image", False),
            is_audio = result.get("is_audio", False),
            multimodal_columns = result.get("multimodal_columns"),
            suggested_mapping = result.get("suggested_mapping"),
            detected_image_column = result.get("detected_image_column"),
            detected_audio_column = result.get("detected_audio_column"),
            detected_text_column = result.get("detected_text_column"),
            detected_speaker_column = result.get("detected_speaker_column"),
            preview_samples = preview_samples,
            total_rows = total_rows,
            warning = warning,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error checking dataset format: {e}", exc_info = True)
        raise HTTPException(
            status_code = 500, detail = f"Failed to check dataset format: {str(e)}"
        )


@router.post("/ai-assist-mapping", response_model = AiAssistMappingResponse)
def ai_assist_mapping(
    request: AiAssistMappingRequest,
    current_subject: str = Depends(get_current_subject),
):
    """
    Run LLM-assisted dataset conversion advisor (user-triggered).

    Multi-pass analysis using a 7B helper model:
      Pass 1: Classify dataset type from HF card + samples
      Pass 2: Generate conversion strategy (system prompt, templates)
      Pass 3: Validate conversion quality

    Falls back to simple column classification if the advisor fails.
    """
    try:
        from utils.datasets.llm_assist import llm_conversion_advisor

        # Truncate sample values for the LLM prompt
        truncated = [
            {col: str(s.get(col, ""))[:200] for col in request.columns}
            for s in request.samples[:5]
        ]

        result = llm_conversion_advisor(
            column_names = request.columns,
            samples = truncated,
            dataset_name = request.dataset_name,
            hf_token = request.hf_token,
            model_name = request.model_name,
            model_type = request.model_type,
        )

        if result and result.get("success"):
            return AiAssistMappingResponse(
                success = True,
                suggested_mapping = result.get("suggested_mapping"),
                system_prompt = result.get("system_prompt"),
                user_template = result.get("user_template"),
                assistant_template = result.get("assistant_template"),
                label_mapping = result.get("label_mapping"),
                dataset_type = result.get("dataset_type"),
                is_conversational = result.get("is_conversational"),
                user_notification = result.get("user_notification"),
            )

        return AiAssistMappingResponse(
            success = False,
            warning = "AI could not determine column roles. Please assign them manually.",
        )

    except Exception as e:
        logger.error(f"AI assist mapping failed: {e}", exc_info = True)
        raise HTTPException(status_code = 500, detail = f"AI assist failed: {str(e)}")
