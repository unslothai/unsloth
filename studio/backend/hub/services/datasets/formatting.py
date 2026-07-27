# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Dataset preview, format-check, and mapping-assist services."""

from __future__ import annotations

import base64
import errno
import io
import re
from pathlib import Path
from typing import Optional

from fastapi import HTTPException
from loggers import get_logger

from hub.schemas.datasets import (
    AiAssistMappingRequest,
    AiAssistMappingResponse,
    CheckFormatRequest,
    CheckFormatResponse,
)
from hub.services.datasets.local import (
    DATA_EXTS,
    _TABULAR_EXTS,
    _load_local_preview_slice,
    _stream_file_preview_slice,
)
from hub.utils.dataset_cache import (
    cached_dataset_candidates as _shared_cached_dataset_candidates,
    latest_cached_dataset_snapshot as _shared_latest_cached_dataset_snapshot,
    split_label_matches as _split_label_matches,
)
from hub.utils import download_registry
from hub.utils.dataset_format import check_dataset_format, format_dataset_preview
from hub.utils.hf_errors import hf_error_status
from hub.utils.paths import (
    is_valid_repo_id as _is_valid_repo_id,
    resolve_dataset_path,
)

logger = get_logger(__name__)

_BINARY_IMAGE_PREVIEW_MAX_BYTES = 10 * 1024 * 1024
_IMAGE_PREVIEW_MAX_PIXELS = 16_000_000
_IMAGE_PREVIEW_THUMBNAIL_SIZE = (512, 512)


def _image_pixel_count(image) -> int:
    width = max(int(getattr(image, "width", 0) or 0), 0)
    height = max(int(getattr(image, "height", 0) or 0), 0)
    return width * height


def _pil_image_has_transparency(image) -> bool:
    if "A" in image.getbands():
        extrema = image.getchannel("A").getextrema()
        return bool(extrema and extrema[0] < 255)
    if image.mode == "P":
        transparency = image.info.get("transparency")
        if transparency is None:
            return False
        if isinstance(transparency, bytes):
            return any(alpha < 255 for alpha in transparency)
        return True
    return False


def _serialize_pil_image(image):
    pixel_count = _image_pixel_count(image)
    if pixel_count > _IMAGE_PREVIEW_MAX_PIXELS:
        return (
            f"<image preview omitted, {image.width}x{image.height} pixels "
            f"exceeds {_IMAGE_PREVIEW_MAX_PIXELS:,} pixel limit>"
        )

    preview = image.copy()
    preview.thumbnail(_IMAGE_PREVIEW_THUMBNAIL_SIZE)
    buffer = io.BytesIO()
    if _pil_image_has_transparency(preview):
        preview.save(buffer, format = "PNG")
        mime = "image/png"
    else:
        preview.convert("RGB").save(buffer, format = "JPEG", quality = 85)
        mime = "image/jpeg"
    return {
        "type": "image",
        "mime": mime,
        "width": preview.width,
        "height": preview.height,
        "data": base64.b64encode(buffer.getvalue()).decode("ascii"),
    }


def _serialize_binary_value(data):
    if len(data) > _BINARY_IMAGE_PREVIEW_MAX_BYTES:
        return (
            f"<binary data omitted, {len(data)} bytes exceeds "
            f"{_BINARY_IMAGE_PREVIEW_MAX_BYTES:,} byte preview limit>"
        )

    try:
        from PIL import Image as PILImageModule
        with PILImageModule.open(io.BytesIO(data)) as image:
            return _serialize_pil_image(image)
    except Exception:
        return f"<binary data, {len(data)} bytes>"


def _serialize_preview_value(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, (bytes, bytearray, memoryview)):
        return _serialize_binary_value(value)

    try:
        from PIL.Image import Image as PILImage
        if isinstance(value, PILImage):
            return _serialize_pil_image(value)
    except Exception:
        pass

    if isinstance(value, dict):
        # Undecoded HF Image/Audio cells are {"bytes": b"...", "path": ...}.
        raw = value.get("bytes")
        if isinstance(raw, (bytes, bytearray, memoryview)) and not (
            value.keys() - {"bytes", "path"}
        ):
            return _serialize_binary_value(raw)
        return {str(key): _serialize_preview_value(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return [_serialize_preview_value(item) for item in value]

    return str(value)


def _serialize_preview_rows(rows):
    return [
        {str(key): _serialize_preview_value(value) for key, value in dict(row).items()}
        for row in rows
    ]


def _latest_cached_dataset_snapshot(
    repo_id: str, local_path: Optional[str] = None
) -> Optional[Path]:
    return _shared_latest_cached_dataset_snapshot(repo_id, local_path)


def _cached_dataset_candidates(
    snapshot: Path, *, subset: Optional[str], train_split: str
) -> list[Path]:
    return _shared_cached_dataset_candidates(
        snapshot,
        subset = subset,
        train_split = train_split,
        extensions = DATA_EXTS,
        preferred_extensions = _TABULAR_EXTS,
    )


def _repo_file_label_tokens(path: str) -> set[str]:
    return {token for token in re.split(r"[^a-z0-9]+", path.lower()) if token}


def _repo_file_matches_label(path: str, label: str) -> bool:
    return label.strip().lower() in _repo_file_label_tokens(path)


def _repo_file_matches_split(path: str, split: str) -> bool:
    return _split_label_matches(path, split)


def _select_tier1_repo_file(
    files: list[str], *, subset: Optional[str], train_split: str
) -> Optional[str]:
    data_files = sorted(f for f in files if any(f.lower().endswith(ext) for ext in DATA_EXTS))
    if not data_files:
        return None
    tabular_files = [f for f in data_files if any(f.lower().endswith(ext) for ext in _TABULAR_EXTS)]
    candidates = tabular_files or data_files
    if subset:
        candidates = [f for f in candidates if _repo_file_matches_label(f, subset)]
        if not candidates:
            return None
    candidates = [f for f in candidates if _repo_file_matches_split(f, train_split)]
    return candidates[0] if candidates else None


def _load_cached_hf_preview_slice(request: CheckFormatRequest, preview_size: int):
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
            preview = _stream_file_preview_slice(candidate, preview_size)
        except Exception as exc:
            logger.debug("Cached dataset preview failed for %s: %s", candidate, exc)
            continue
        if preview is not None:
            return preview
    return None


def _load_processed_hf_preview_slice(
    request: CheckFormatRequest,
    preview_size: int,
    hf_token: Optional[str] = None,
):
    if not _is_valid_repo_id(request.dataset_name):
        return None
    try:
        from datasets import DownloadConfig

        # Non-streaming loads take the cached builder lock; use the EACCES-safe wrapper.
        from utils.datasets.cache_safe import load_dataset_cache_safe as load_dataset
    except Exception:
        return None

    load_kwargs = {
        "path": request.dataset_name,
        "split": request.train_split or "train",
        "download_config": DownloadConfig(local_files_only = True),
    }
    if request.subset:
        load_kwargs["name"] = request.subset
    if hf_token:
        load_kwargs["token"] = hf_token

    dataset = load_dataset(**load_kwargs)
    total_rows = len(dataset)
    preview_slice = dataset.select(range(min(preview_size, total_rows)))
    return preview_slice, total_rows


def _load_any_cached_hf_preview_slice(
    request: CheckFormatRequest,
    preview_size: int,
    hf_token: Optional[str] = None,
):
    cached_preview = _load_cached_hf_preview_slice(request, preview_size)
    if cached_preview is not None:
        return cached_preview
    try:
        return _load_processed_hf_preview_slice(request, preview_size, hf_token)
    except Exception as exc:
        logger.debug(
            "Processed dataset cache preview failed for %s: %s",
            request.dataset_name,
            exc,
        )
        return None


def check_format_response(
    request: CheckFormatRequest, hf_token: Optional[str] = None
) -> CheckFormatResponse:
    """
    Check if a dataset requires manual column mapping.

    HF datasets: tier 1 loads a single requested split/subset file (avoids
    resolving thousands of files); tier 2 falls back to full streaming. Local
    files load directly. Plain `def` so FastAPI runs the blocking IO in a
    thread-pool.
    """
    try:
        from itertools import islice

        PREVIEW_SIZE = 10

        logger.info(f"Checking format for dataset: {request.dataset_name}")

        try:
            dataset_path = resolve_dataset_path(request.dataset_name)
        except ValueError as e:
            # Malformed path (null bytes, '..', outside roots) is a client error:
            # surface 400 rather than the generic 500 below.
            raise HTTPException(status_code = 400, detail = str(e)) from e
        total_rows = None

        if dataset_path.exists():
            train_split = request.train_split or "train"
            preview_slice, total_rows = _load_local_preview_slice(
                dataset_path = dataset_path,
                train_split = train_split,
                preview_size = PREVIEW_SIZE,
            )
        else:
            from datasets import Dataset, load_dataset

            # Tier 1: list_repo_files → load only the first data file
            cached_preview = (
                _load_any_cached_hf_preview_slice(request, PREVIEW_SIZE, hf_token)
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
                        token = hf_token or None,
                    )
                    train_split = request.train_split or "train"
                    first_file = _select_tier1_repo_file(
                        repo_files,
                        subset = request.subset,
                        train_split = train_split,
                    )
                    if first_file:
                        logger.info(f"Tier 1: loading single file {first_file}")
                        load_kwargs = {
                            "path": request.dataset_name,
                            "data_files": {train_split: [first_file]},
                            "split": train_split,
                            "streaming": True,
                        }
                        if hf_token:
                            load_kwargs["token"] = hf_token

                        streamed_ds = load_dataset(**load_kwargs)
                        rows = list(islice(streamed_ds, PREVIEW_SIZE))
                        if rows:
                            preview_slice = Dataset.from_list(rows)
                except Exception as e:
                    logger.warning(
                        "Tier 1 (single-file) failed: %s",
                        download_registry.scrub_secrets(str(e), hf_token = hf_token),
                    )

            if preview_slice is None:
                # Tier 2: full streaming (resolves all files — slow for large repos)
                logger.info("Tier 2: falling back to full streaming load_dataset")
                try:
                    load_kwargs = {
                        "path": request.dataset_name,
                        "split": request.train_split or "train",
                        "streaming": True,
                    }
                    if request.subset:
                        load_kwargs["name"] = request.subset
                    if hf_token:
                        load_kwargs["token"] = hf_token

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
                        hf_token,
                    )
                    if cached_preview is None:
                        raise
                    preview_slice, total_rows = cached_preview

        result = check_dataset_format(preview_slice, is_vlm = request.is_vlm)

        logger.info(
            f"Format check result: requires_mapping={result['requires_manual_mapping']}, format={result['detected_format']}, is_image={result.get('is_image', False)}"
        )

        preview_samples = None
        if not result["requires_manual_mapping"]:
            if result.get("suggested_mapping"):
                # Heuristic-detected: show raw data so columns match the response;
                # column stripping happens at training time, not preview.
                preview_samples = _serialize_preview_rows(preview_slice)
            else:
                try:
                    processed = format_dataset_preview(preview_slice)
                    preview_samples = _serialize_preview_rows(processed)
                except Exception as e:
                    logger.warning(f"Processed preview generation failed (non-fatal): {e}")
                    preview_samples = _serialize_preview_rows(preview_slice)
        else:
            preview_samples = _serialize_preview_rows(preview_slice)

        # Collect warnings: from check_dataset_format + URL-based image detection
        warning = result.get("warning")
        image_col = result.get("detected_image_column")
        if image_col and image_col in (result.get("columns") or []):
            try:
                sample_val = preview_slice[0][image_col]
                if isinstance(sample_val, str) and sample_val.startswith(("http://", "https://")):
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
        scrubbed = download_registry.scrub_secrets(str(e), hf_token = hf_token)
        # Missing/gated/bad-token and malformed names are client errors, not 500s.
        status = hf_error_status(e)
        if (
            status is None
            and isinstance(e, OSError)
            and getattr(e, "errno", None) == errno.ENAMETOOLONG
        ):
            status, scrubbed = 400, "Invalid dataset name"
        elif status is None and isinstance(e, FileNotFoundError):
            # datasets raises DatasetNotFoundError (FileNotFoundError) for missing/gated.
            status = 404
        elif status is None and isinstance(e, ValueError):
            status = 400
        if status is not None:
            raise HTTPException(status_code = status, detail = scrubbed)
        logger.error("Error checking dataset format: %s", scrubbed)
        raise HTTPException(
            status_code = 500,
            detail = "Failed to check dataset format: " + scrubbed,
        )


def ai_assist_mapping_response(
    request: AiAssistMappingRequest, hf_token: Optional[str] = None
) -> AiAssistMappingResponse:
    """
    Run the LLM-assisted dataset conversion advisor (user-triggered).

    Multi-pass analysis with a 7B helper model: classify dataset type, generate
    a conversion strategy, then validate it. Falls back to simple column
    classification if the advisor fails.
    """
    try:
        from hub.utils.llm_assist import llm_conversion_advisor

        truncated = [
            {col: str(s.get(col, ""))[:200] for col in request.columns} for s in request.samples[:5]
        ]

        result = llm_conversion_advisor(
            column_names = request.columns,
            samples = truncated,
            dataset_name = request.dataset_name,
            hf_token = hf_token,
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
                warning = result.get("warning"),
            )

        return AiAssistMappingResponse(
            success = False,
            warning = "AI could not determine column roles. Please assign them manually.",
        )

    except Exception as e:
        scrubbed = download_registry.scrub_secrets(str(e), hf_token = hf_token)
        status = hf_error_status(e)
        if status is None and isinstance(e, FileNotFoundError):
            status = 404
        elif status is None and isinstance(e, ValueError):
            status = 400
        if status is not None:
            raise HTTPException(status_code = status, detail = scrubbed)
        logger.error("AI assist mapping failed: %s", scrubbed)
        raise HTTPException(
            status_code = 500,
            detail = "AI assist failed: " + scrubbed,
        )
