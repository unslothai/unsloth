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
    dataset_snapshot_from_cache_path as _shared_dataset_snapshot_from_cache_path,
    latest_cached_dataset_path as _shared_latest_cached_dataset_path,
    latest_cached_dataset_snapshot as _shared_latest_cached_dataset_snapshot,
    load_cached_hf_dataset as _shared_load_cached_hf_dataset,
    split_label_matches as _split_label_matches,
)
from hub.utils import download_registry
from hub.utils.dataset_format import check_dataset_format, format_dataset_preview
from hub.utils.hf_errors import hf_error_status
from hub.utils.paths import (
    is_valid_repo_id as _is_valid_repo_id,
    normalize_path,
    resolve_dataset_path,
)
from hub.utils.hf_tokens import is_anonymous
from utils.utils import anonymous_and_offline
from utils.datasets.audio_decode import ensure_audio_decoding
from utils.paths.path_utils import drop_shadowed_appledouble_names

logger = get_logger(__name__)

_BINARY_IMAGE_PREVIEW_MAX_BYTES = 10 * 1024 * 1024
_IMAGE_PREVIEW_MAX_PIXELS = 16_000_000
_IMAGE_PREVIEW_THUMBNAIL_SIZE = (512, 512)
_LOCAL_CACHE_MISS_ERROR_CODE = "dataset_local_cache_miss"
_MISSING_DATASET_DETAIL = "This dataset is no longer on disk. Add it again or pick another dataset."


def _is_local_dataset_ref(dataset_name: str) -> bool:
    normalized = normalize_path(str(dataset_name or "").strip())
    return Path(normalized).expanduser().is_absolute()


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


def _serialize_decoded_audio(value):
    """Summarise a decoded Audio cell the way binary cells are summarised."""
    samples = value.get("array") or []
    rate = value.get("sampling_rate")
    try:
        seconds = len(samples) / rate if rate else None
    except (TypeError, ZeroDivisionError):
        seconds = None
    detail = f"{len(samples)} samples"
    if rate:
        detail += f" @ {rate} Hz"
    if seconds is not None:
        detail += f", {seconds:.1f}s"
    return f"<audio, {detail}>"


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
        # A decoded Audio cell becomes one float per sample under the soundfile fallback, so ten preview
        # rows of a few seconds each are tens of MB of JSON and the client dies rendering it.
        if "sampling_rate" in value and isinstance(value.get("array"), (list, tuple)):
            return _serialize_decoded_audio(value)
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
    if local_path:
        return _shared_dataset_snapshot_from_cache_path(local_path, repo_id)
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


def _repo_file_has_other_common_split(path: str, train_split: str) -> bool:
    requested = train_split.strip().lower()
    return any(
        label != requested and _repo_file_matches_split(path, label)
        for label in ("train", "validation", "valid", "dev", "eval", "test")
    )


def _select_tier1_repo_file(
    files: list[str],
    *,
    subset: Optional[str],
    train_split: str,
    allow_unlabeled_fallback: bool = False,
) -> Optional[str]:
    # "._train.parquet" sorts first and would be handed to the single-file preview load.
    data_files = sorted(
        f
        for f in drop_shadowed_appledouble_names(list(files))
        if any(f.lower().endswith(ext) for ext in DATA_EXTS)
    )
    if not data_files:
        return None
    tabular_files = [f for f in data_files if any(f.lower().endswith(ext) for ext in _TABULAR_EXTS)]
    candidates = tabular_files or data_files
    if subset:
        candidates = [f for f in candidates if _repo_file_matches_label(f, subset)]
        if not candidates:
            return None
    split_candidates = [f for f in candidates if _repo_file_matches_split(f, train_split)]
    if split_candidates:
        return split_candidates[0]
    if (
        allow_unlabeled_fallback
        and len(candidates) == 1
        and not _repo_file_has_other_common_split(candidates[0], train_split)
    ):
        return candidates[0]
    return None


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
    local_path = request.local_path
    if not local_path:
        cached_path = _shared_latest_cached_dataset_path(request.dataset_name)
        if cached_path is None:
            return None
        local_path = str(cached_path)
    dataset = _shared_load_cached_hf_dataset(
        request.dataset_name,
        local_path,
        subset = request.subset,
        split = request.train_split or "train",
        token = hf_token,
    )
    total_rows = len(dataset)
    preview_slice = dataset.select(range(min(preview_size, total_rows)))
    return preview_slice, total_rows


def _load_any_cached_hf_preview_slice(
    request: CheckFormatRequest,
    preview_size: int,
    hf_token: Optional[str] = None,
):
    # Both paths return real rows off disk without asking the Hub: the raw slice reads the
    # snapshot, the processed one loads with local_files_only=True and drops the falsy
    # sentinel. Refuse the whole disk route here; the handler then answers 404.
    if is_anonymous(hf_token):
        return None
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
    request: CheckFormatRequest,
    hf_token: Optional[str] = None,
    *,
    allow_unlabeled_tier1_fallback: bool = False,
) -> CheckFormatResponse:
    """
    Check if a dataset requires manual column mapping.

    HF datasets: tier 1 loads a single requested split/subset file (avoids
    resolving thousands of files); tier 2 falls back to full streaming. Local
    files load directly. Plain `def` so FastAPI runs the blocking IO in a
    thread-pool. The deprecated alias opts into the single-file fallback that
    its previous implementation used, preserving source column order when the
    only data filename has no split label.
    """
    try:
        from itertools import islice

        PREVIEW_SIZE = 10

        logger.info(f"Checking format for dataset: {request.dataset_name}")

        # An audio column decodes on the first preview row, so this precedes every tier.
        ensure_audio_decoding()

        try:
            dataset_path = resolve_dataset_path(request.dataset_name)
        except ValueError as e:
            # Malformed path (null bytes, '..', outside roots) is a client error: surface 400, not 500.
            raise HTTPException(status_code = 400, detail = str(e)) from e
        total_rows = None

        dataset_exists = dataset_path.exists()
        if not dataset_exists and _is_local_dataset_ref(request.dataset_name):
            raise HTTPException(status_code = 404, detail = _MISSING_DATASET_DETAIL)

        # Offline `datasets` answers a streaming load from its cache without authorizing,
        # and Tier 2 runs on the default prefer_local_cache=false, ahead of that guard.
        if anonymous_and_offline(hf_token) and not dataset_exists:
            raise HTTPException(
                status_code = 404,
                detail = "This request cannot be authorized without network access.",
            )
        if dataset_exists:
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
                    detail = {
                        "code": _LOCAL_CACHE_MISS_ERROR_CODE,
                        "message": "Dataset is not available in the local cache.",
                    },
                )
            else:
                preview_slice = None

                try:
                    from huggingface_hub import HfApi

                    # No token on the constructor: list_repo_files is given it explicitly
                    # and that argument wins.
                    api = HfApi()
                    repo_files = api.list_repo_files(
                        request.dataset_name,
                        repo_type = "dataset",
                        token = hf_token,
                    )
                    train_split = request.train_split or "train"
                    first_file = _select_tier1_repo_file(
                        repo_files,
                        subset = request.subset,
                        train_split = train_split,
                        allow_unlabeled_fallback = allow_unlabeled_tier1_fallback,
                    )
                    if first_file:
                        logger.info(f"Tier 1: loading single file {first_file}")
                        load_kwargs = {
                            "path": request.dataset_name,
                            "data_files": {train_split: [first_file]},
                            "split": train_split,
                            "streaming": True,
                            "token": hf_token,
                        }

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
                # Tier 2: full streaming (resolves all files - slow for large repos)
                logger.info("Tier 2: falling back to full streaming load_dataset")
                try:
                    load_kwargs = {
                        "path": request.dataset_name,
                        "split": request.train_split or "train",
                        "streaming": True,
                        "token": hf_token,
                    }
                    if request.subset:
                        load_kwargs["name"] = request.subset

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
                # Heuristic-detected: show raw data so columns match the response (stripping happens at training).
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
            chat_column = result.get("chat_column"),
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
