# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Training API routes
"""

import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from typing import Dict, Optional, Any
import structlog
from loggers import get_logger
import asyncio
from datetime import datetime
import uuid as _uuid

# Add backend directory to path.
backend_path = Path(__file__).parent.parent.parent
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))

try:
    from core.training import get_training_backend
    from core.training.resume import (
        can_resume_run,
        get_resume_checkpoint_path,
        normalize_resume_output_dir,
        training_run_config,
    )
    from storage.studio_db import get_resumable_run_by_output_dir
    from utils.models.model_config import detect_gguf_model, load_model_defaults
    from utils.paths import is_local_path, normalize_path, resolve_dataset_path
except ImportError:
    # Fallback: parent directory.
    parent_backend = backend_path.parent / "backend"
    if str(parent_backend) not in sys.path:
        sys.path.insert(0, str(parent_backend))
    from core.training import get_training_backend
    from core.training.resume import (
        can_resume_run,
        get_resume_checkpoint_path,
        normalize_resume_output_dir,
        training_run_config,
    )
    from storage.studio_db import get_resumable_run_by_output_dir
    from utils.models.model_config import detect_gguf_model, load_model_defaults
    from utils.paths import is_local_path, normalize_path, resolve_dataset_path

# Auth
from auth.authentication import authenticated_via_api_key, get_current_subject

from utils.utils import canonical_model_repo_id, hf_env_offline, log_and_http_error

from models import (
    TrainingStartRequest,
    TrainingStartRequestStatus,
    TrainingJobResponse,
    TrainingStatus,
    TrainingProgress,
)
from models.responses import TrainingStopResponse, TrainingMetricsResponse
from pydantic import BaseModel as PydanticBaseModel, ValidationError as PydanticValidationError


class TrainingStopRequest(PydanticBaseModel):
    save: bool = True
    expected_job_id: Optional[str] = None


class TrainingResetRequest(PydanticBaseModel):
    expected_job_id: Optional[str] = None


router = APIRouter()
logger = get_logger(__name__)

_LOCAL_MODEL_PROBE_LIMIT = 2000
_REMOTE_MODEL_METADATA_TIMEOUT_SECONDS = 5.0
_REMOTE_MODEL_METADATA_RETRY_TIMEOUT_SECONDS = 10.0


class _LocalModelProbeIncomplete(RuntimeError):
    pass


def _model_preflight_error(status_code: int, code: str, message: str) -> HTTPException:
    return HTTPException(
        status_code = status_code,
        detail = {"code": code, "message": message},
    )


def _http_exception_error(exc: HTTPException) -> tuple[str, Optional[str]]:
    detail = exc.detail
    if isinstance(detail, dict):
        message = detail.get("message")
        code = detail.get("code")
        if isinstance(message, str) and message:
            return message, code if isinstance(code, str) and code else None
    return str(detail), None


@dataclass(frozen = True)
class _ModelPreflightResult:
    model_name: str
    model_local_path: Optional[str]
    cached_model_pin: Optional[tuple[str, str]]


# Consecutive 1s polls without a step update that count as a stall. Applied only
# once stepping: the pre-first-step phase (model load + tokenization) can take far
# longer, and timing out there made a healthy long-prep run look frozen.
_PROGRESS_STALL_TIMEOUT_POLLS = 1800  # ~30 min at 1 poll/sec


def _stop_training_if_active(backend, *, save: bool, expected_job_id: Optional[str]):
    from core.training.lifecycle import training_lifecycle_guard
    with training_lifecycle_guard():
        is_active = backend.is_training_active()
        stopped = (
            backend.stop_training(
                save = save,
                expected_job_id = expected_job_id,
            )
            if is_active
            else False
        )
    return is_active, stopped


def _validate_local_dataset_paths(paths: list[str], label: str = "Local dataset") -> list[str]:
    """Resolve and validate a list of local dataset paths. Returns validated absolute paths."""
    validated = []
    missing = []
    for dataset_path in paths:
        dataset_file = resolve_dataset_path(dataset_path)
        if not dataset_file.exists():
            missing.append(f"{dataset_path} (resolved: {dataset_file})")
            continue
        logger.info(f"Found {label.lower()} file: {dataset_file}")
        validated.append(str(dataset_file))

    if missing:
        missing_detail = "; ".join(missing[:3])
        raise HTTPException(
            status_code = 400,
            detail = f"{label} not found: {missing_detail}",
        )
    return validated


def _start_request_response(record) -> TrainingJobResponse:
    return TrainingJobResponse(
        job_id = record.job_id,
        status = "error" if record.state == "rejected" else "queued",
        message = record.message,
        error = record.error,
        error_code = record.error_code,
    )


def _reject_start_request(
    backend,
    start_request_id: Optional[str],
    message: str,
    error_code: Optional[str] = None,
) -> None:
    if backend is None or start_request_id is None:
        return
    backend.resolve_start_request(
        start_request_id,
        state = "rejected",
        message = message,
        error = message,
        error_code = error_code,
    )


def _observe_training_start_task(task: asyncio.Task[bool]) -> None:
    if not task.cancelled():
        task.exception()


def _is_indexed_model_weight_name(name: str, expected_suffix: str) -> bool:
    lower = name.lower()
    return lower.endswith(expected_suffix) and not lower.startswith(
        ("adapter_model", "optimizer", "scheduler", "rng_state", "scaler")
    )


_MODEL_WEIGHT_CANDIDATES = (
    ("model.safetensors", None),
    ("model.safetensors.index.json", ".safetensors"),
    ("pytorch_model.bin", None),
    ("pytorch_model.bin.index.json", ".bin"),
)
_SHARDED_WEIGHT_NAME_PATTERN = re.compile(
    r"^(?P<family>.+)-(?P<part>\d+)-of-(?P<total>\d+)(?P<suffix>(?:\.[^.]+)+)$",
    re.IGNORECASE,
)


def _is_nonempty_file(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except (OSError, ValueError):
        return False


def _has_complete_indexed_weights(path: Path, index_name: str, expected_suffix: str) -> bool:
    snapshot = os.path.abspath(os.path.normpath(str(path)))
    snapshot_key = os.path.normcase(snapshot)
    try:
        index_text = (path / index_name).read_text(encoding = "utf-8-sig")
        payload = json.loads(index_text)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    weight_map = payload.get("weight_map") if isinstance(payload, dict) else None
    if not isinstance(weight_map, dict):
        return False
    shards = list(weight_map.values())
    if not shards or not all(isinstance(shard, str) and shard for shard in shards):
        return False
    families: dict[tuple[str, str, str, int], set[int]] = {}
    for shard in set(shards):
        joined = os.path.normpath(os.path.join(snapshot, shard))
        joined_key = os.path.normcase(joined)
        contained = joined_key == snapshot_key or joined_key.startswith(snapshot_key + os.sep)
        shard_path = Path(joined)
        if (
            not contained
            or not _is_indexed_model_weight_name(
                shard_path.name,
                expected_suffix,
            )
            or not _is_nonempty_file(shard_path)
        ):
            return False
        match = _SHARDED_WEIGHT_NAME_PATTERN.fullmatch(shard_path.name)
        if match is not None:
            part = int(match.group("part"))
            total = int(match.group("total"))
            if total < 1 or part < 1 or part > total:
                return False
            family = (
                os.path.normcase(str(shard_path.parent)),
                match.group("family").casefold(),
                match.group("suffix").casefold(),
                total,
            )
            families.setdefault(family, set()).add(part)
    return all(len(parts) == family[3] for family, parts in families.items())


def _has_trainable_local_weights(path: Path) -> bool:
    try:
        if not path.is_dir():
            return False
        config_text = (path / "config.json").read_text(encoding = "utf-8-sig")
        config = json.loads(config_text)
        if not isinstance(config, dict):
            return False
        for name, index_suffix in _MODEL_WEIGHT_CANDIDATES:
            candidate = path / name
            if not candidate.is_file():
                continue
            if index_suffix is None:
                return _is_nonempty_file(candidate)
            return _has_complete_indexed_weights(path, name, index_suffix)
        return False
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False


def _has_adapter_metadata(path: Path) -> bool:
    return path.is_dir() and (path / "adapter_config.json").is_file()


def _remote_untrainable_model_format(model_name: str, hf_token: Optional[str]) -> Optional[str]:
    from huggingface_hub import model_info as hf_model_info
    from hub.utils.hf_errors import hf_error_status

    repo_id = canonical_model_repo_id(model_name)
    timeouts = (
        _REMOTE_MODEL_METADATA_TIMEOUT_SECONDS,
        _REMOTE_MODEL_METADATA_RETRY_TIMEOUT_SECONDS,
    )
    for attempt, timeout in enumerate(timeouts):
        try:
            info = hf_model_info(
                repo_id,
                token = hf_token,
                timeout = timeout,
            )
            break
        except Exception as error:
            status_code = hf_error_status(error)
            if status_code is None:
                upstream_status = getattr(
                    getattr(error, "response", None),
                    "status_code",
                    None,
                )
                if isinstance(upstream_status, int) and 500 <= upstream_status < 600:
                    status_code = upstream_status
            transient_status = status_code in (408, 429) or (
                isinstance(status_code, int) and 500 <= status_code < 600
            )
            if status_code in (401, 403):
                raise _model_preflight_error(
                    422,
                    "hf_model_access_denied",
                    (
                        "Hugging Face denied access to this model. Add a valid Hugging Face "
                        "token with repository access and accept any required access terms, "
                        "then try again."
                    ),
                ) from error
            retry_available = attempt + 1 < len(timeouts)
            if transient_status:
                if retry_available:
                    continue
                if status_code == 429:
                    raise _model_preflight_error(
                        429,
                        "hf_model_verification_rate_limited",
                        "Hugging Face model verification is rate-limited. Retry shortly.",
                    ) from error
            elif status_code is not None:
                raise _model_preflight_error(
                    status_code,
                    "hf_model_verification_failed",
                    (
                        "The Hugging Face model could not be verified. "
                        "Check the repository ID and your access token."
                    ),
                ) from error
            elif retry_available:
                continue
            logger.warning(
                "Could not inspect remote model files for %s after retry (%s)",
                model_name,
                type(error).__name__,
            )
            raise _model_preflight_error(
                503,
                "hf_model_metadata_unavailable",
                (
                    "Hugging Face model metadata is temporarily unavailable. "
                    "Retry before starting training."
                ),
            ) from error

    root_files: set[str] = set()
    has_gguf = False
    for sibling in getattr(info, "siblings", None) or ():
        name = getattr(sibling, "rfilename", None)
        if not isinstance(name, str):
            continue
        normalized = name.replace("\\", "/")
        if normalized.casefold().endswith(".gguf"):
            has_gguf = True
        if "/" not in normalized:
            root_files.add(normalized)
    if "adapter_config.json" in root_files:
        return "adapter"
    has_trainable_weights = any(name in root_files for name, _ in _MODEL_WEIGHT_CANDIDATES)
    if has_gguf and not has_trainable_weights:
        return "gguf"
    return None


def _detect_local_gguf(path: Path) -> Optional[str]:
    try:
        try:
            is_directory = path.is_dir()
        except OSError:
            if path.suffix.lower() == ".gguf":
                return detect_gguf_model(str(path))
            raise
        if not is_directory:
            return detect_gguf_model(str(path))
        for index, entry in enumerate(path.rglob("*"), start = 1):
            if index > _LOCAL_MODEL_PROBE_LIMIT:
                raise _LocalModelProbeIncomplete
            if entry.suffix.lower() != ".gguf":
                continue
            detected = detect_gguf_model(str(entry))
            if detected is not None:
                return detected
    except _LocalModelProbeIncomplete:
        raise
    except (OSError, RuntimeError) as error:
        raise _LocalModelProbeIncomplete from error
    return None


def _reject_untrainable_model_request(
    request: TrainingStartRequest, actual_model_repo_id: Optional[str] = None
) -> _ModelPreflightResult:
    model_format = (request.model_format or "").strip().lower()
    if model_format == "gguf":
        raise HTTPException(
            status_code = 400,
            detail = "GGUF models are inference-only and cannot be trained.",
        )
    if model_format == "adapter":
        raise HTTPException(
            status_code = 400,
            detail = "Adapter models are inference-only and cannot be trained as base models.",
        )
    path: Optional[Path] = None
    model_name = request.model_name
    model_local_path: Optional[str] = None
    cached_model_pin: Optional[tuple[str, str]] = None
    if is_local_path(request.model_name):
        try:
            path = Path(normalize_path(request.model_name)).expanduser().resolve(strict = True)
        except (OSError, RuntimeError, ValueError) as error:
            raise HTTPException(
                status_code = 400,
                detail = "Local model path was not found or could not be accessed.",
            ) from error
        model_name = str(path)
        if request.model_local_path:
            model_local_path = (
                model_name
                if request.model_local_path == request.model_name
                else normalize_path(request.model_local_path)
            )
    else:
        model_local_path = (
            normalize_path(request.model_local_path) if request.model_local_path else None
        )
        from hub.utils.hf_cache_state import latest_snapshot_from_cache_path

        snapshot = None
        offline_mode = hf_env_offline()
        if request.resume_from_checkpoint and request.model_snapshot_path:
            snapshot = latest_snapshot_from_cache_path(
                request.model_snapshot_path,
                "model",
                canonical_model_repo_id(actual_model_repo_id or request.model_name),
                ("config.json", "adapter_config.json"),
            )
        elif request.model_known_cached or request.model_local_path or offline_mode:
            from core.training.training import _resolve_model_snapshot
            snapshot = _resolve_model_snapshot(
                request.model_name,
                model_local_path,
            )
        if snapshot:
            path = Path(snapshot)
            if offline_mode and not request.resume_from_checkpoint:
                cached_model_pin = (
                    canonical_model_repo_id(request.model_name),
                    snapshot,
                )
    metadata_error: Optional[HTTPException] = None
    if path is None:
        try:
            remote_format = _remote_untrainable_model_format(
                request.model_name,
                request.hf_token or None,
            )
        except HTTPException as error:
            metadata_error = error
            from core.training.training import _resolve_model_snapshot

            snapshot = _resolve_model_snapshot(
                request.model_name,
                model_local_path,
            )
            if snapshot is None:
                raise
            path = Path(snapshot)
            cached_model_pin = (
                canonical_model_repo_id(request.model_name),
                snapshot,
            )
        else:
            if remote_format is None:
                return _ModelPreflightResult(model_name, model_local_path, cached_model_pin)
            if remote_format == "gguf":
                raise HTTPException(
                    status_code = 400,
                    detail = "GGUF-only remote models are inference-only and cannot be trained.",
                )
            raise HTTPException(
                status_code = 400,
                detail = ("Adapter models are inference-only and cannot be trained as base models."),
            )
    has_trainable_weights = _has_trainable_local_weights(path)
    if has_trainable_weights:
        return _ModelPreflightResult(model_name, model_local_path, cached_model_pin)
    if _has_adapter_metadata(path):
        raise HTTPException(
            status_code = 400,
            detail = (
                "Adapter-only local models are inference-only and cannot be trained as base models."
            ),
        )
    try:
        has_gguf = _detect_local_gguf(path) is not None
    except _LocalModelProbeIncomplete:
        raise HTTPException(
            status_code = 400,
            detail = (
                "The local model directory is too large or could not be read safely. "
                "Select its snapshot directory containing config.json and trainable weights."
            ),
        )
    if has_gguf:
        raise HTTPException(
            status_code = 400,
            detail = "GGUF-only local models are inference-only and cannot be trained.",
        )
    if metadata_error is not None:
        raise metadata_error
    raise HTTPException(
        status_code = 400,
        detail = "The selected local model does not contain trainable weights.",
    )


_RESUME_DATASET_DEFAULTS = {
    "hf_dataset": None,
    "local_datasets": [],
    "local_eval_datasets": [],
    "format_type": "",
    "subset": None,
    "train_split": "train",
    "eval_split": None,
    "dataset_streaming": False,
    "dataset_slice_start": None,
    "dataset_slice_end": None,
    "custom_format_mapping": None,
    "is_dataset_image": False,
    "is_dataset_audio": False,
    "is_embedding": False,
}
_RESUME_CACHE_FIELDS = (
    "model_known_cached",
    "model_local_path",
    "model_format",
    "model_snapshot_path",
    "dataset_known_cached",
    "dataset_local_path",
    "dataset_snapshot_path",
)
_RESUME_CHECKPOINT_STRUCTURE_FIELDS = (
    "load_in_4bit",
    "use_lora",
    "lora_r",
    "lora_alpha",
    "lora_dropout",
    "target_modules",
    "gradient_checkpointing",
    "use_rslora",
    "use_loftq",
    "use_dora",
    "finetune_vision_layers",
    "finetune_language_layers",
    "finetune_attention_modules",
    "finetune_mlp_modules",
    "optim",
    "lr_scheduler_type",
    "embedding_learning_rate",
)


def _normalized_optional_string(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value or None


def _prepare_resume_resource_provenance(
    request: TrainingStartRequest, resume_run: dict
) -> tuple[Optional[str], bool, bool, bool, Optional[str]]:
    stored = training_run_config(resume_run)
    from core.training.provenance import (
        ExactResumeResourcesUnavailable,
        RESOURCE_PROVENANCE_KEY,
        exact_resume_resource_requirements,
        resource_provenance_is_complete,
    )

    try:
        requires_exact_model, requires_exact_dataset = exact_resume_resource_requirements(stored)
    except ExactResumeResourcesUnavailable:
        raise HTTPException(
            status_code = 409,
            detail = (
                "The source run has invalid resource provenance or its exact snapshots "
                "are no longer available."
            ),
        )
    stored_model = _normalized_optional_string(resume_run.get("model_name")) or (
        _normalized_optional_string(stored.get("model_name"))
    )
    if stored_model is None:
        raise HTTPException(
            status_code = 409,
            detail = (
                "The source run does not contain model provenance and cannot be resumed safely."
            ),
        )
    if request.model_name != stored_model:
        raise HTTPException(
            status_code = 409,
            detail = "The selected model does not match the model used by the source run.",
        )

    stored_training_type = stored.get("training_type")
    if isinstance(stored_training_type, str) and stored_training_type:
        if request.training_type != stored_training_type:
            raise HTTPException(
                status_code = 409,
                detail = "The training type does not match the source run.",
            )
        request.training_type = stored_training_type
    for field in _RESUME_CHECKPOINT_STRUCTURE_FIELDS:
        if field not in stored:
            continue
        value = stored[field]
        setattr(request, field, list(value) if isinstance(value, list) else value)

    stored_dataset = _normalized_optional_string(stored.get("hf_dataset"))
    requested_dataset = _normalized_optional_string(request.hf_dataset)
    if requested_dataset != stored_dataset:
        raise HTTPException(
            status_code = 409,
            detail = "The selected dataset does not match the dataset used by the source run.",
        )

    request.model_name = stored_model
    for field, default in _RESUME_DATASET_DEFAULTS.items():
        value = stored.get(field, default)
        setattr(request, field, list(value) if isinstance(value, list) else value)
    request.s3_config = None
    for field in _RESUME_CACHE_FIELDS:
        default = False if field.endswith("_known_cached") else None
        setattr(request, field, stored.get(field, default))
    try:
        validated_request = TrainingStartRequest.model_validate(
            {field: getattr(request, field) for field in TrainingStartRequest.model_fields}
        )
    except PydanticValidationError as error:
        raise HTTPException(
            status_code = 409,
            detail = (
                "The source run contains invalid training configuration and cannot "
                "be resumed safely."
            ),
        ) from error
    for field in TrainingStartRequest.model_fields:
        setattr(request, field, getattr(validated_request, field))
    marker = stored.get(RESOURCE_PROVENANCE_KEY)
    model_load_mode = (
        _normalized_optional_string(marker.get("model_load_mode"))
        if requires_exact_model and isinstance(marker, dict)
        else None
    )
    return (
        _normalized_optional_string(stored.get("actual_model_repo_id")),
        requires_exact_model,
        requires_exact_dataset,
        resource_provenance_is_complete(stored),
        model_load_mode,
    )


@router.get("/hardware")
async def get_hardware_utilization(current_subject: str = Depends(get_current_subject)):
    """
    Live snapshot of GPU hardware utilization for the active backend.

    Polled by the frontend during training.
    """
    from utils.hardware import get_gpu_utilization
    return get_gpu_utilization()


@router.get("/hardware/visible")
async def get_visible_hardware_utilization(current_subject: str = Depends(get_current_subject)):
    from utils.hardware import get_visible_gpu_utilization

    # Off the event loop: the ROCm fallbacks shell out (Windows perf counters, sysfs) and the System view polls this route.
    return await asyncio.to_thread(get_visible_gpu_utilization)


@router.get("/start-requests/{start_request_id}", response_model = TrainingStartRequestStatus)
async def get_training_start_request(
    start_request_id: str, current_subject: str = Depends(get_current_subject)
):
    backend = get_training_backend()
    record = backend.get_start_request(start_request_id)
    if record is None:
        raise HTTPException(status_code = 404, detail = "Training start request not found")
    return TrainingStartRequestStatus(
        start_request_id = record.start_request_id,
        job_id = record.job_id,
        state = record.state,
        message = record.message,
        error = record.error,
        error_code = record.error_code,
    )


@router.post("/start-requests/{start_request_id}/acknowledge")
async def acknowledge_training_start_request(
    start_request_id: str, current_subject: str = Depends(get_current_subject)
):
    backend = get_training_backend()
    if not backend.acknowledge_start_request(start_request_id):
        raise HTTPException(
            status_code = 409,
            detail = "Training start request is not ready to acknowledge",
        )
    return {"status": "ok"}


@router.post("/start")
async def start_training(
    request: TrainingStartRequest,
    current_subject: str = Depends(get_current_subject),
    via_api_key: bool = Depends(authenticated_via_api_key),
):
    """
    Start a training job.

    Initiates training in the background and returns immediately. Use /status
    to check progress.
    """
    backend = None
    reserved_start_request_id = None
    start_task: Optional[asyncio.Task[bool]] = None
    try:
        logger.info(f"Starting training job with model: {request.model_name}")
        backend = get_training_backend()
        job_id = f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{_uuid.uuid4().hex[:8]}"
        if request.start_request_id:
            reservation, record = backend.reserve_start_request(
                request.start_request_id,
                job_id,
            )
            if reservation == "existing":
                return _start_request_response(record)
            if reservation == "conflict":
                return _start_request_response(record)
            reserved_start_request_id = request.start_request_id

        # When Unsloth is driven as an inference API (API-key auth), refuse to start
        # training while a request is in flight: training frees VRAM by unloading
        # the chat model, which would kill the stream. The Unsloth UI (session auth)
        # still starts training and coexists/frees VRAM as before. (A mixed UI+API
        # session is not yet special-cased.)
        if via_api_key is True:
            from core.inference.llama_keepwarm import other_inference_request_count
            if other_inference_request_count(current_request_counted = False) > 0:
                raise HTTPException(
                    status_code = 409,
                    detail = (
                        "Cannot start training over the API while an inference request is in "
                        "progress. Wait for it to finish, or start training from the Unsloth UI."
                    ),
                )

        # No in-process ensure_transformers_version(): the subprocess
        # (worker.py) activates the correct version before importing ML libs.

        # A consented latest-transformers install stage-and-swaps .venv_t5_latest;
        # a worker spawned mid-swap could activate a half-replaced sidecar.
        from utils.transformers_latest import is_install_in_progress

        if is_install_in_progress():
            raise HTTPException(
                status_code = 409,
                detail = ("A transformers installation is in progress. Retry when it completes."),
            )

        # S3 dataset loading needs the optional boto3 dependency. Reject early
        # with a clear message so credentials are never accepted and then
        # silently dropped on a host without boto3 installed.
        if request.s3_config is not None and not request.resume_from_checkpoint:
            from core.training.s3_dataset import boto3_available
            if not boto3_available():
                raise HTTPException(
                    status_code = 501,
                    detail = "S3 dataset loading requires boto3. Install it with: pip install boto3",
                )

        # Check before mutating state.
        if await asyncio.to_thread(backend.is_training_active):
            existing_job_id: Optional[str] = getattr(backend, "current_job_id", "")
            _reject_start_request(
                backend,
                reserved_start_request_id,
                "Training already active",
            )
            return TrainingJobResponse(
                job_id = existing_job_id or "",
                status = "error",
                message = (
                    "Training is already in progress. "
                    "Stop current training before starting a new one."
                ),
                error = "Training already active",
            )

        resume_output_dir: Optional[str] = None
        resume_run: Optional[dict] = None
        resume_actual_model_repo_id: Optional[str] = None
        resume_model_load_mode: Optional[str] = None
        resume_requires_exact_resources = False
        resume_requires_exact_model = False
        resume_requires_exact_dataset = False
        if request.resume_from_checkpoint:
            try:
                resume_output_dir = await asyncio.to_thread(
                    normalize_resume_output_dir,
                    request.resume_from_checkpoint,
                )
            except ValueError as e:
                # Deliberate user-facing validation message.
                validation_message = str(e)
                raise HTTPException(status_code = 400, detail = validation_message)

            resume_run = await asyncio.to_thread(
                get_resumable_run_by_output_dir,
                resume_output_dir,
            )
            if not resume_run or not await asyncio.to_thread(can_resume_run, resume_run):
                raise HTTPException(
                    status_code = 400,
                    detail = "Resume checkpoint must belong to a stopped or errored run with complete saved trainer state.",
                )
            resume_checkpoint = await asyncio.to_thread(
                get_resume_checkpoint_path,
                resume_output_dir,
            )
            if not resume_checkpoint:
                raise HTTPException(
                    status_code = 400,
                    detail = "Resume checkpoint must include saved trainer state.",
                )
            request.resume_from_checkpoint = resume_checkpoint
            (
                resume_actual_model_repo_id,
                resume_requires_exact_model,
                resume_requires_exact_dataset,
                resume_requires_exact_resources,
                resume_model_load_mode,
            ) = await asyncio.to_thread(
                _prepare_resume_resource_provenance,
                request,
                resume_run,
            )

        if request.local_datasets:
            request.local_datasets = _validate_local_dataset_paths(
                request.local_datasets, "Local dataset"
            )
        if request.local_eval_datasets and request.eval_steps > 0:
            request.local_eval_datasets = _validate_local_dataset_paths(
                request.local_eval_datasets, "Local eval dataset"
            )

        # Validate streaming-mode compatibility before any expensive work.
        # Streaming is supported only for Hugging Face text datasets.
        if request.dataset_streaming:
            if not request.hf_dataset:
                raise HTTPException(
                    status_code = 400,
                    detail = "dataset_streaming requires hf_dataset; streaming is not supported for local datasets.",
                )
            if request.is_dataset_image or request.is_dataset_audio:
                raise HTTPException(
                    status_code = 400,
                    detail = "dataset_streaming is not supported for vision or audio datasets.",
                )
            if request.is_embedding:
                raise HTTPException(
                    status_code = 400,
                    detail = "dataset_streaming is not supported for embedding training; the embedding loader needs the full dataset.",
                )
            from utils.hardware import hardware as _hw

            if _hw.DEVICE == _hw.DeviceType.MLX:
                raise HTTPException(
                    status_code = 400,
                    detail = "dataset_streaming is not yet supported on Apple Silicon (MLX); the MLX loader materializes the full dataset.",
                )
            if request.max_steps is None or request.max_steps <= 0:
                raise HTTPException(
                    status_code = 422,
                    detail = "dataset_streaming requires max_steps > 0 because streaming datasets have no known length.",
                )
            if request.train_on_completions:
                raise HTTPException(
                    status_code = 422,
                    detail = "dataset_streaming is not supported with train_on_completions yet.",
                )
            if request.eval_steps > 0:
                train_split = request.train_split or "train"
                if not request.eval_split or request.eval_split == train_split:
                    raise HTTPException(
                        status_code = 422,
                        detail = "dataset_streaming with evaluation requires a separate eval_split.",
                    )
            # Streaming is HF-only: reject when the request also carries a local
            # dataset path or an S3 config; those sources cannot be streamed via
            # HF's streaming loader.
            if request.local_datasets:
                raise HTTPException(
                    status_code = 400,
                    detail = (
                        "dataset_streaming is HF-only; remove local_datasets / S3 source. "
                        "Streaming is not supported with local file paths."
                    ),
                )
            if request.s3_config is not None:
                raise HTTPException(
                    status_code = 400,
                    detail = (
                        "dataset_streaming is HF-only; remove local_datasets / S3 source. "
                        "Streaming is not supported with S3 datasets."
                    ),
                )
            if request.dataset_known_cached or request.dataset_local_path:
                raise HTTPException(
                    status_code = 422,
                    detail = (
                        "dataset_streaming streams from the Hub and cannot use the local "
                        "dataset cache; disable streaming to train from the cached copy."
                    ),
                )

        model_preflight = await asyncio.to_thread(
            _reject_untrainable_model_request,
            request,
            resume_actual_model_repo_id,
        )
        cached_model_pin = model_preflight.cached_model_pin
        training_actual_model_repo_id = resume_actual_model_repo_id
        training_model_snapshot_path = request.model_snapshot_path
        if cached_model_pin is not None:
            training_actual_model_repo_id, training_model_snapshot_path = cached_model_pin

        # Convert request to backend kwargs.
        training_kwargs = {
            "model_name": model_preflight.model_name,
            "project_name": request.project_name,
            "training_type": request.training_type,
            "hf_token": request.hf_token or "",
            "load_in_4bit": request.load_in_4bit,
            "max_seq_length": request.max_seq_length,
            "vision_image_size": request.vision_image_size,
            "hf_dataset": request.hf_dataset or "",
            "model_known_cached": request.model_known_cached,
            "model_local_path": model_preflight.model_local_path,
            "model_format": request.model_format,
            "model_snapshot_path": training_model_snapshot_path,
            "actual_model_repo_id": training_actual_model_repo_id,
            "resume_model_load_mode": resume_model_load_mode,
            "dataset_known_cached": request.dataset_known_cached,
            "dataset_local_path": request.dataset_local_path,
            "dataset_snapshot_path": request.dataset_snapshot_path,
            "local_datasets": request.local_datasets,
            "local_eval_datasets": request.local_eval_datasets,
            "format_type": request.format_type,
            "subset": request.subset,
            "train_split": request.train_split,
            "dataset_streaming": request.dataset_streaming,
            "eval_split": request.eval_split,
            "eval_steps": request.eval_steps,
            "dataset_slice_start": request.dataset_slice_start,
            "dataset_slice_end": request.dataset_slice_end,
            "custom_format_mapping": request.custom_format_mapping,
            "num_epochs": request.num_epochs,
            "learning_rate": request.learning_rate,
            "embedding_learning_rate": request.embedding_learning_rate,
            "batch_size": request.batch_size,
            "gradient_accumulation_steps": request.gradient_accumulation_steps,
            "warmup_steps": request.warmup_steps,
            "warmup_ratio": request.warmup_ratio,
            "max_steps": request.max_steps,
            "save_steps": request.save_steps,
            "weight_decay": request.weight_decay,
            "max_grad_norm": request.max_grad_norm,
            "max_grad_value": request.max_grad_value,
            "max_grad_leaf_norm": request.max_grad_leaf_norm,
            "cast_norm_output_to_input_dtype": request.cast_norm_output_to_input_dtype,
            "random_seed": request.random_seed,
            "packing": request.packing,
            "optim": request.optim,
            "lr_scheduler_type": request.lr_scheduler_type,
            "use_lora": request.use_lora,
            "lora_r": request.lora_r,
            "lora_alpha": request.lora_alpha,
            "lora_dropout": request.lora_dropout,
            "target_modules": request.target_modules if request.target_modules else None,
            "gradient_checkpointing": request.gradient_checkpointing.strip()
            if request.gradient_checkpointing and request.gradient_checkpointing.strip()
            else "unsloth",
            "use_rslora": request.use_rslora,
            "use_loftq": request.use_loftq,
            "use_dora": request.use_dora,
            "train_on_completions": request.train_on_completions,
            "finetune_vision_layers": request.finetune_vision_layers,
            "finetune_language_layers": request.finetune_language_layers,
            "finetune_attention_modules": request.finetune_attention_modules,
            "finetune_mlp_modules": request.finetune_mlp_modules,
            "is_dataset_image": request.is_dataset_image,
            "is_dataset_audio": request.is_dataset_audio,
            "is_embedding": request.is_embedding,
            "enable_wandb": request.enable_wandb,
            "wandb_token": request.wandb_token or "",
            "wandb_project": request.wandb_project or "",
            "enable_tensorboard": request.enable_tensorboard,
            "tensorboard_dir": request.tensorboard_dir or "",
            "output_dir": resume_output_dir,
            "resume_from_checkpoint": request.resume_from_checkpoint,
            "require_exact_resume_resources": resume_requires_exact_resources,
            "require_exact_model_resource": resume_requires_exact_model,
            "require_exact_dataset_resource": resume_requires_exact_dataset,
            "require_validated_model_snapshot": cached_model_pin is not None,
            "trust_remote_code": request.trust_remote_code,
            "approved_remote_code_fingerprint": request.approved_remote_code_fingerprint,
            "subject": current_subject,
            "gpu_ids": request.gpu_ids,
            "s3_config": request.s3_config.model_dump() if request.s3_config else None,
        }

        # Latest-sidecar models size and train 16-bit (same flip as chat load):
        # 4-bit is disabled for brand-new architectures, so VRAM coexistence
        # checks must not underestimate against a load the worker will refuse.
        from core.training.provenance import (
            ExactResumeResourcesUnavailable,
            effective_training_load_in_4bit,
        )

        if training_kwargs["load_in_4bit"]:
            from core.training.training import resolve_training_model_load_target

            try:
                model_load_target = await asyncio.to_thread(
                    resolve_training_model_load_target,
                    training_kwargs,
                )
            except ExactResumeResourcesUnavailable as exc:
                raise HTTPException(status_code = 409, detail = str(exc))
            try:
                effective_load_in_4bit = await asyncio.to_thread(
                    effective_training_load_in_4bit,
                    training_kwargs,
                    model_load_target,
                    training_kwargs["hf_token"] or None,
                )
            except ExactResumeResourcesUnavailable as exc:
                raise HTTPException(status_code = 409, detail = str(exc))
            if not effective_load_in_4bit:
                training_kwargs["load_in_4bit"] = False
                logger.info(
                    "Latest-transformers sidecar active for %s - sizing and "
                    "training in 16-bit (4-bit is disabled for brand-new "
                    "architectures)",
                    model_load_target,
                )

        # Training page has no trust_remote_code toggle, so honor the YAML default
        # -- but only for genuine first-party (unsloth/nvidia) Hub repos, never a
        # local path or a name merely starting with "unsloth/".
        if not training_kwargs["trust_remote_code"]:
            from utils.security.trusted_org import is_trusted_org_repo

            model_defaults = load_model_defaults(request.model_name)
            yaml_trust = model_defaults.get("training", {}).get("trust_remote_code", False)
            if yaml_trust and is_trusted_org_repo(
                request.model_name, hf_token = request.hf_token or None
            ):
                logger.info(f"YAML config sets trust_remote_code=True for {request.model_name}")
                training_kwargs["trust_remote_code"] = True
            elif yaml_trust:
                logger.warning(
                    "YAML sets trust_remote_code=True for %s but it is not a trusted "
                    "first-party repo; leaving disabled (user can opt in explicitly).",
                    request.model_name,
                )

        # Free VRAM for training: stop export, unload chat unless it can coexist.
        # A before_spawn hook -> runs only after start_training's guards pass, so
        # we never tear down chat/export VRAM for a start that is then refused.
        def _free_vram_for_training() -> None:
            try:
                from core.export import get_export_backend
                exp_backend = get_export_backend()
                # Tear down the export subprocess whenever an export is in flight,
                # not just once a checkpoint is loaded: during the load phase
                # current_checkpoint is still unset while the worker is already
                # allocating GPU memory, so gate on is_export_active() too.
                if exp_backend.current_checkpoint or exp_backend.is_export_active():
                    logger.info("Shutting down export subprocess to free GPU memory for training")
                    exp_backend._shutdown_subprocess()
                    exp_backend.current_checkpoint = None
                    exp_backend.is_vision = False
                    exp_backend.is_peft = False
            except Exception as e:
                logger.warning("Could not shut down export subprocess: %s", e)

            try:
                from routes.training_vram import (
                    can_keep_chat_during_training,
                    coordinate_models_for_training,
                )

                def _can_keep_resident_models():
                    return can_keep_chat_during_training(
                        model_name = training_kwargs["model_name"],
                        hf_token = training_kwargs["hf_token"],
                        training_type = training_kwargs["training_type"],
                        load_in_4bit = training_kwargs["load_in_4bit"],
                        batch_size = training_kwargs["batch_size"],
                        max_seq_length = training_kwargs["max_seq_length"],
                        lora_rank = training_kwargs["lora_r"],
                        target_modules = training_kwargs["target_modules"],
                        gradient_checkpointing = training_kwargs["gradient_checkpointing"],
                        optimizer = training_kwargs["optim"],
                        gpu_ids = training_kwargs["gpu_ids"],
                    )

                freed = coordinate_models_for_training(_can_keep_resident_models)
                if freed:
                    logger.info("Freed models for training: %s", freed)
            except Exception as e:
                logger.warning("Inference/training memory coordination failed; proceeding: %s", e)

        # The hook runs only once start guards pass -> VRAM freed iff training starts.
        from utils.transformers_version import SidecarSwapInProgress

        def _run_backend_start() -> bool:
            try:
                success = backend.start_training(
                    job_id = job_id,
                    start_request_id = request.start_request_id,
                    before_spawn = _free_vram_for_training,
                    resume_source_run_id = resume_run["id"] if resume_run else None,
                    **training_kwargs,
                )
            except (SidecarSwapInProgress, ExactResumeResourcesUnavailable) as exc:
                _reject_start_request(backend, reserved_start_request_id, str(exc))
                raise
            except ValueError as exc:
                _reject_start_request(backend, reserved_start_request_id, str(exc))
                raise
            except Exception:
                _reject_start_request(
                    backend,
                    reserved_start_request_id,
                    "Failed to start training",
                )
                raise

            if success:
                if reserved_start_request_id is not None:
                    backend.resolve_start_request(
                        reserved_start_request_id,
                        state = "accepted",
                        message = "Training job queued and starting in subprocess",
                    )
            else:
                progress_error = backend.trainer.training_progress.error
                _reject_start_request(
                    backend,
                    reserved_start_request_id,
                    progress_error or "Failed to start training subprocess",
                )
            return success

        try:
            start_task = asyncio.create_task(asyncio.to_thread(_run_backend_start))
            success = await asyncio.shield(start_task)
        except SidecarSwapInProgress as exc:
            # Expected loss of the race against a sidecar install: a retryable
            # 409 matching the route-entry guard, not an internal error.
            raise HTTPException(status_code = 409, detail = str(exc))
        except ExactResumeResourcesUnavailable as exc:
            raise HTTPException(status_code = 409, detail = str(exc))

        if not success:
            progress_error = backend.trainer.training_progress.error
            failure_message = progress_error or "Failed to start training subprocess"
            return TrainingJobResponse(
                job_id = backend.current_job_id or "",
                status = "error",
                message = failure_message,
                error = progress_error or "subprocess_start_failed",
            )

        return TrainingJobResponse(
            job_id = job_id,
            status = "queued",
            message = "Training job queued and starting in subprocess",
            error = None,
        )

    except asyncio.CancelledError:
        if start_task is None:
            _reject_start_request(
                backend,
                reserved_start_request_id,
                "Training start was cancelled",
            )
        else:
            start_task.add_done_callback(_observe_training_start_task)
        raise
    except HTTPException as exc:
        # Deliberate rejections (S3 not implemented, resume validation) must
        # reach the client with their original status, not a generic 500.
        detail, error_code = _http_exception_error(exc)
        _reject_start_request(
            backend,
            reserved_start_request_id,
            detail,
            error_code,
        )
        raise
    except ValueError as e:
        logger.warning("Rejected training GPU selection: %s", e)
        # Deliberate user-facing GPU-selection validation message.
        validation_message = str(e)
        _reject_start_request(backend, reserved_start_request_id, validation_message)
        raise HTTPException(status_code = 400, detail = validation_message)
    except Exception as e:
        _reject_start_request(
            backend,
            reserved_start_request_id,
            "Failed to start training",
        )
        raise log_and_http_error(
            e,
            500,
            "Failed to start training",
            event = "training.start_failed",
            log = logger,
        )


@router.post("/stop", response_model = TrainingStopResponse)
async def stop_training(
    body: TrainingStopRequest = TrainingStopRequest(),
    current_subject: str = Depends(get_current_subject),
):
    """
    Stop the currently running training job.

    Body:
        save (bool): If True (default), save the model at the current checkpoint.
    """
    try:
        backend = get_training_backend()
        is_active, stopped = await asyncio.to_thread(
            _stop_training_if_active,
            backend,
            save = body.save,
            expected_job_id = body.expected_job_id,
        )
        logger.info("Stop requested: save=%s is_active=%s", body.save, is_active)
        if not is_active or not stopped:
            return TrainingStopResponse(
                status = "idle", message = "No training job is currently running"
            )

        return TrainingStopResponse(
            status = "stopped",
            message = "Stop requested. Training will stop at the next safe step.",
        )

    except Exception as e:
        raise log_and_http_error(
            e,
            500,
            "Failed to stop training",
            event = "training.stop_failed",
            log = logger,
        )


@router.post("/reset")
async def reset_training(
    body: Optional[TrainingResetRequest] = None, current_subject: str = Depends(get_current_subject)
):
    """Reset training state so the user can return to configuration."""
    try:
        backend = get_training_backend()
        result = await asyncio.to_thread(
            backend.reset_training_state,
            expected_job_id = body.expected_job_id if body is not None else None,
        )
        if result == "superseded":
            return {"status": "superseded"}
        if result == "active":
            raise HTTPException(
                status_code = 409,
                detail = "Training is still running. Stop training and wait for it to finish before resetting.",
            )
        return {"status": "ok"}
    except HTTPException:
        raise
    except Exception as e:
        raise log_and_http_error(
            e,
            500,
            "Failed to reset training",
            event = "training.reset_failed",
            log = logger,
        )


@router.get("/status")
async def get_training_status(current_subject: str = Depends(get_current_subject)):
    """
    Get the current training status.
    """
    try:
        backend = get_training_backend()
        job_id: str = getattr(backend, "current_job_id", "") or ""
        start_request_id: Optional[str] = getattr(backend, "current_start_request_id", None)
        start_request = backend.status_start_request()
        start_request_state = start_request.state if start_request is not None else None
        if start_request is not None and start_request.state in {"pending", "rejected"}:
            job_id = "" if start_request.state == "rejected" else start_request.job_id
            start_request_id = start_request.start_request_id

        is_active = await asyncio.to_thread(backend.is_training_active)
        if is_active and backend.current_start_request_id:
            active_start_request = backend.get_start_request(backend.current_start_request_id)
            if active_start_request is not None:
                start_request = active_start_request
                start_request_state = active_start_request.state
                job_id = active_start_request.job_id
                start_request_id = active_start_request.start_request_id

        try:
            progress = backend.trainer.get_training_progress()
        except Exception:
            progress = None

        status_message = (
            getattr(progress, "status_message", None) if progress else None
        ) or "Ready to train"
        error_message = getattr(progress, "error", None) if progress else None
        if start_request is not None and start_request.state == "pending":
            status_message = start_request.message
            error_message = None
        elif start_request is not None and start_request.state == "rejected":
            status_message = start_request.message
            error_message = start_request.error or start_request.message

        trainer_stopped = getattr(backend, "_should_stop", False)

        # Derive high-level phase
        if start_request_state == "pending":
            phase = "configuring"
        elif start_request_state == "rejected":
            phase = "error"
        elif error_message:
            phase = "error"
        elif is_active:
            msg_lower = status_message.lower()
            if "loading" in msg_lower or "importing" in msg_lower:
                phase = "loading_model"
            elif any(k in msg_lower for k in ["preparing", "initializing", "configuring"]):
                phase = "configuring"
            else:
                phase = "training"
        elif trainer_stopped:
            phase = "stopped"
        elif progress and getattr(progress, "is_completed", False):
            phase = "completed"
        else:
            phase = "idle"

        details = None
        if progress:
            details = {
                "epoch": getattr(progress, "epoch", 0),
                "step": getattr(progress, "step", 0),
                "total_steps": getattr(progress, "total_steps", 0),
                "loss": getattr(progress, "loss", None),
                "learning_rate": getattr(progress, "learning_rate", None),
            }
            # Always present: an explicit null tells the client to drop a cached
            # path (stop without save clears the run's output_dir).
            details["output_dir"] = getattr(backend, "_output_dir", None) or None

        # Metric history for chart recovery after SSE reconnection.
        metric_history = None
        if backend.step_history:
            metric_history = {
                "steps": list(backend.step_history),
                "loss": list(backend.loss_history),
                "lr": list(backend.lr_history),
                "grad_norm": list(getattr(backend, "grad_norm_history", [])),
                "grad_norm_steps": list(getattr(backend, "grad_norm_step_history", [])),
                "eval_loss": list(backend.eval_loss_history),
                "eval_steps": list(backend.eval_step_history),
            }

        return TrainingStatus(
            job_id = job_id,
            start_request_id = start_request_id,
            start_request_state = start_request_state,
            phase = phase,
            is_training_running = is_active,
            eval_enabled = backend.eval_enabled,
            message = status_message,
            error = error_message,
            details = details,
            metric_history = metric_history,
        )

    except Exception as e:
        raise log_and_http_error(
            e,
            500,
            "Failed to get training status",
            event = "training.status_failed",
            log = logger,
        )


@router.get("/metrics", response_model = TrainingMetricsResponse)
async def get_training_metrics(current_subject: str = Depends(get_current_subject)):
    """
    Get training metrics (loss, learning rate, steps).
    """
    try:
        backend = get_training_backend()

        loss_history = backend.loss_history
        lr_history = backend.lr_history
        step_history = backend.step_history
        grad_norm_history = getattr(backend, "grad_norm_history", [])
        grad_norm_step_history = getattr(backend, "grad_norm_step_history", [])

        current_loss = loss_history[-1] if loss_history else None
        current_lr = lr_history[-1] if lr_history else None
        current_step = step_history[-1] if step_history else None

        return TrainingMetricsResponse(
            loss_history = loss_history,
            lr_history = lr_history,
            step_history = step_history,
            grad_norm_history = grad_norm_history,
            grad_norm_step_history = grad_norm_step_history,
            current_loss = current_loss,
            current_lr = current_lr,
            current_step = current_step,
        )

    except Exception as e:
        raise log_and_http_error(
            e,
            500,
            "Failed to get training metrics",
            event = "training.metrics_failed",
            log = logger,
        )


@router.get("/progress")
async def stream_training_progress(
    request: Request, current_subject: str = Depends(get_current_subject)
):
    """
    Stream training progress via Server-Sent Events (SSE).

    Real-time progress with reconnection support per the SSE spec:
      - `id:` per event so the browser tracks position.
      - `retry:` to control reconnection interval.
      - Named `event:` types (progress, heartbeat, complete, error).
      - Reads `Last-Event-ID` on reconnect to replay missed steps.
    """
    # Read Last-Event-ID header for reconnection resume.
    last_event_id = request.headers.get("last-event-id")
    resume_from_step: Optional[int] = None
    if last_event_id is not None:
        try:
            resume_from_step = int(last_event_id)
            # Fires on every reconnect (each tab switch); the meaningful signal is
            # the "replayed N missed steps" line below, logged only when N > 0.
            logger.debug(f"SSE reconnect: resuming from step {resume_from_step}")
        except ValueError:
            logger.warning(f"Invalid Last-Event-ID: {last_event_id}")

    async def event_generator():
        backend = get_training_backend()
        job_id: str = getattr(backend, "current_job_id", "") or ""

        # ── Helpers ──────────────────────────────────────────────
        def build_progress(
            step: int,
            loss: Optional[float],
            learning_rate: Optional[float],
            total_steps: int,
            epoch: Optional[float] = None,
            progress: Optional[Any] = None,
            grad_norm_override: Optional[float] = None,
            eval_loss_override: Optional[float] = None,
        ) -> TrainingProgress:
            total = max(total_steps, 0)
            if step < 0 or total == 0:
                progress_percent = 0.0
            else:
                progress_percent = float(step) / float(total) * 100.0 if total > 0 else 0.0

            # Pull values from the progress object if available.
            elapsed_seconds = getattr(progress, "elapsed_seconds", None) if progress else None
            eta_seconds = getattr(progress, "eta_seconds", None) if progress else None
            grad_norm = grad_norm_override
            if grad_norm is None and progress:
                grad_norm = getattr(progress, "grad_norm", None)
            num_tokens = getattr(progress, "num_tokens", None) if progress else None
            eval_loss = eval_loss_override
            if eval_loss is None and progress:
                eval_loss = getattr(progress, "eval_loss", None)

            return TrainingProgress(
                job_id = job_id,
                step = step,
                total_steps = total,
                loss = loss,
                learning_rate = learning_rate,
                progress_percent = progress_percent,
                epoch = epoch,
                elapsed_seconds = elapsed_seconds,
                eta_seconds = eta_seconds,
                grad_norm = grad_norm,
                num_tokens = num_tokens,
                eval_loss = eval_loss,
            )

        def format_sse(
            data: str,
            event: str = "progress",
            event_id: Optional[int] = None,
        ) -> str:
            """Format a single SSE message with id/event/data fields."""
            lines = []
            if event_id is not None:
                lines.append(f"id: {event_id}")
            lines.append(f"event: {event}")
            lines.append(f"data: {data}")
            lines.append("")  # trailing blank line
            lines.append("")  # double newline terminates the event
            return "\n".join(lines)

        # ── Retry directive ──────────────────────────────────────
        # Reconnect after 3 seconds if the connection drops.
        yield "retry: 3000\n\n"

        # ── Replay missed steps on reconnect ─────────────────────
        if resume_from_step is not None and backend.step_history:
            replayed = 0
            grad_norm_by_step = {
                step_val: grad_val
                for step_val, grad_val in zip(
                    getattr(backend, "grad_norm_step_history", []),
                    getattr(backend, "grad_norm_history", []),
                )
            }
            for i, step_val in enumerate(backend.step_history):
                if step_val > resume_from_step:
                    loss_val = backend.loss_history[i] if i < len(backend.loss_history) else None
                    lr_val = backend.lr_history[i] if i < len(backend.lr_history) else None
                    tp_replay = getattr(
                        getattr(backend, "trainer", None), "training_progress", None
                    )
                    total_replay = (
                        getattr(tp_replay, "total_steps", step_val) if tp_replay else step_val
                    )
                    epoch_replay = getattr(tp_replay, "epoch", None) if tp_replay else None
                    payload = build_progress(
                        step_val,
                        loss_val,
                        lr_val,
                        total_replay,
                        epoch_replay,
                        progress = tp_replay,
                        grad_norm_override = grad_norm_by_step.get(step_val),
                    )
                    yield format_sse(payload.model_dump_json(), event = "progress", event_id = step_val)
                    replayed += 1
            if replayed:
                logger.info(f"SSE reconnect: replayed {replayed} missed steps")

        # ── Initial status (only on fresh connections) ───────────
        if resume_from_step is None:
            is_active = await asyncio.to_thread(backend.is_training_active)
            tp = getattr(getattr(backend, "trainer", None), "training_progress", None)
            initial_total_steps = getattr(tp, "total_steps", 0) if tp else 0
            initial_epoch = getattr(tp, "epoch", None) if tp else None

            initial_progress = build_progress(
                step = 0,
                loss = None,
                learning_rate = None,
                total_steps = initial_total_steps,
                epoch = initial_epoch,
                progress = tp,
            )
            yield format_sse(initial_progress.model_dump_json(), event = "progress", event_id = 0)

            # If not active, send final state and exit
            if not is_active:
                _live = (getattr(tp, "step", 0) or 0) if tp else 0
                if backend.step_history or _live > 0:
                    final_step = backend.step_history[-1] if backend.step_history else 0
                    final_loss = backend.loss_history[-1] if backend.loss_history else None
                    final_lr = backend.lr_history[-1] if backend.lr_history else None
                    # Histories skip non-finite steps; report the live step with
                    # loss=None instead of the last finite pair.
                    if _live > final_step:
                        final_step = _live
                        final_loss = getattr(tp, "loss", None)
                        final_lr = getattr(tp, "learning_rate", final_lr)
                    final_total_steps = getattr(tp, "total_steps", final_step) if tp else final_step
                    final_epoch = getattr(tp, "epoch", None) if tp else None
                    payload = build_progress(
                        final_step,
                        final_loss,
                        final_lr,
                        final_total_steps,
                        final_epoch,
                        progress = tp,
                    )
                    yield format_sse(
                        payload.model_dump_json(), event = "complete", event_id = final_step
                    )
                else:
                    yield format_sse(
                        build_progress(-1, None, None, 0, progress = tp).model_dump_json(),
                        event = "complete",
                        event_id = 0,
                    )
                return

        # ── Live polling loop ────────────────────────────────────
        last_step = resume_from_step if resume_from_step is not None else -1
        no_update_count = 0
        # The stall timeout applies only once the run is stepping (pre-step prep
        # may legitimately emit no step for a long time). On reconnect to an
        # already-stepping run, seed from the resume point / history, else a worker
        # that hangs after step N never times out for a client that reconnects past it.
        seen_live_step = (resume_from_step is not None and resume_from_step > 0) or bool(
            backend.step_history
        )

        while await asyncio.to_thread(backend.is_training_active):
            # Client gone: end the generator without falling through to the final
            # "complete" frame, which a buffered/proxy consumer could otherwise read
            # as a finished run while training is still active.
            if await request.is_disconnected():
                return
            try:
                tp_inner = getattr(getattr(backend, "trainer", None), "training_progress", None)
                live_step = (getattr(tp_inner, "step", 0) or 0) if tp_inner else 0
                if backend.step_history or live_step > 0:
                    current_step = backend.step_history[-1] if backend.step_history else 0
                    current_loss = backend.loss_history[-1] if backend.loss_history else None
                    current_lr = backend.lr_history[-1] if backend.lr_history else None
                    # Histories skip non-finite steps; follow the live progress
                    # step and report its loss (None until it recovers).
                    if live_step > current_step:
                        current_step = live_step
                        current_loss = getattr(tp_inner, "loss", None)
                        current_lr = getattr(tp_inner, "learning_rate", current_lr)
                    current_total_steps = (
                        getattr(tp_inner, "total_steps", current_step) if tp_inner else current_step
                    )
                    current_epoch = getattr(tp_inner, "epoch", None) if tp_inner else None

                    # Only send if the step changed.
                    if current_step != last_step:
                        progress_payload = build_progress(
                            current_step,
                            current_loss,
                            current_lr,
                            current_total_steps,
                            current_epoch,
                            progress = tp_inner,
                        )
                        yield format_sse(
                            progress_payload.model_dump_json(),
                            event = "progress",
                            event_id = current_step,
                        )
                        last_step = current_step
                        no_update_count = 0
                        seen_live_step = True
                    else:
                        no_update_count += 1
                        # Heartbeat every 10 seconds.
                        if no_update_count % 10 == 0:
                            heartbeat_payload = build_progress(
                                current_step,
                                current_loss,
                                current_lr,
                                current_total_steps,
                                current_epoch,
                                progress = tp_inner,
                            )
                            yield format_sse(
                                heartbeat_payload.model_dump_json(),
                                event = "heartbeat",
                                event_id = current_step,
                            )
                else:
                    # No steps yet, but training is active (model loading, etc.).
                    no_update_count += 1
                    if no_update_count % 5 == 0:
                        # Pull total_steps + status so the frontend can show
                        # "Tokenizing…" etc.
                        tp_prep = getattr(
                            getattr(backend, "trainer", None),
                            "training_progress",
                            None,
                        )
                        prep_total = getattr(tp_prep, "total_steps", 0) if tp_prep else 0
                        preparing_payload = build_progress(
                            0,
                            None,
                            None,
                            prep_total,
                            progress = tp_prep,
                        )
                        yield format_sse(
                            preparing_payload.model_dump_json(),
                            event = "heartbeat",
                            event_id = 0,
                        )

                # Fires only once stepping: a long pre-first-step prep phase is not
                # a stall, and ending the stream there made a healthy run look frozen.
                if seen_live_step and no_update_count > _PROGRESS_STALL_TIMEOUT_POLLS:
                    logger.warning("Progress stream timeout - no updates received")
                    tp_timeout = getattr(
                        getattr(backend, "trainer", None), "training_progress", None
                    )
                    timeout_payload = build_progress(last_step, None, None, 0, progress = tp_timeout)
                    yield format_sse(
                        timeout_payload.model_dump_json(),
                        event = "error",
                        event_id = last_step if last_step >= 0 else 0,
                    )
                    break

                await asyncio.sleep(1)  # Poll every second

            except Exception as e:
                logger.error(f"Error in progress stream: {e}", exc_info = True)
                tp_error = getattr(getattr(backend, "trainer", None), "training_progress", None)
                error_payload = build_progress(0, None, None, 0, progress = tp_error)
                yield format_sse(
                    error_payload.model_dump_json(),
                    event = "error",
                    event_id = last_step if last_step >= 0 else 0,
                )
                break

        # ── Final "complete" event ───────────────────────────────
        final_step = backend.step_history[-1] if backend.step_history else last_step
        final_loss = backend.loss_history[-1] if backend.loss_history else None
        final_lr = backend.lr_history[-1] if backend.lr_history else None
        final_tp = getattr(getattr(backend, "trainer", None), "training_progress", None)
        # If the run ended on a non-finite stretch, report the live step with
        # loss=None instead of rolling back to the last finite pair.
        _final_live_step = (getattr(final_tp, "step", 0) or 0) if final_tp else 0
        if _final_live_step > (final_step if final_step is not None else -1):
            final_step = _final_live_step
            final_loss = getattr(final_tp, "loss", None)
            final_lr = getattr(final_tp, "learning_rate", final_lr)
        final_total_steps = getattr(final_tp, "total_steps", final_step) if final_tp else final_step
        final_epoch = getattr(final_tp, "epoch", None) if final_tp else None
        final_payload = build_progress(
            final_step,
            final_loss,
            final_lr,
            final_total_steps,
            final_epoch,
            progress = final_tp,
        )
        yield format_sse(
            final_payload.model_dump_json(),
            event = "complete",
            event_id = final_step if final_step >= 0 else 0,
        )

    return StreamingResponse(
        event_generator(),
        media_type = "text/event-stream",
        headers = {
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
