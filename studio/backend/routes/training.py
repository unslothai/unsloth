# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Training API routes
"""

import contextlib
import json
import os
import re
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Path as ApiPath,
    Request,
    UploadFile,
)
from fastapi.responses import StreamingResponse
from typing import Dict, Literal, Optional, Any
import structlog
from loggers import get_logger
import asyncio
from datetime import datetime
import uuid as _uuid

backend_path = Path(__file__).parent.parent.parent
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))

try:
    from core.training import get_training_backend
    from core.training.training import (
        TrainingStartCancellationCapacityError,
        TrainingStatusIdentitySnapshot,
    )
    from core.training.resume import (
        can_resume_run,
        get_resume_checkpoint_path,
        has_resume_state,
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
    from core.training.training import (
        TrainingStartCancellationCapacityError,
        TrainingStatusIdentitySnapshot,
    )
    from core.training.resume import (
        can_resume_run,
        get_resume_checkpoint_path,
        has_resume_state,
        normalize_resume_output_dir,
        training_run_config,
    )
    from storage.studio_db import get_resumable_run_by_output_dir
    from utils.models.model_config import detect_gguf_model, load_model_defaults
    from utils.paths import is_local_path, normalize_path, resolve_dataset_path

from auth.authentication import authenticated_via_api_key, get_current_subject

from utils.utils import (
    canonical_model_repo_id,
    hf_dns_dead,
    hf_env_offline,
    hf_reachability_memo,
    hf_unreachable,
    log_and_http_error,
)

from models import (
    TrainingStartRequest,
    TrainingStartRequestStatus,
    TrainingJobResponse,
    TrainingStatus,
    TrainingProgress,
)
from models.training import (
    DiffusionCaptionUpdateRequest,
    DiffusionDatasetExample,
    DiffusionDatasetExamplesResponse,
    DiffusionDatasetImageRecord,
    DiffusionDatasetImagesResponse,
    DiffusionDatasetImportRequest,
    DiffusionDatasetImportResponse,
    DiffusionDatasetSummary,
    DiffusionDatasetUploadResponse,
    DiffusionMetricHistory,
    DiffusionTrainableFamily,
    DiffusionTrainingInfoResponse,
    DiffusionTrainingRunDetail,
    DiffusionTrainingRunsResponse,
    DiffusionTrainingRunSummary,
    DiffusionTrainingStartRequest,
    DiffusionTrainingStartResponse,
    DiffusionTrainingStatusResponse,
    DiffusionTrainingStopRequest,
    TRAINING_REQUEST_ID_PATTERN,
)
from models.responses import TrainingStopResponse, TrainingMetricsResponse
from pydantic import (
    BaseModel as PydanticBaseModel,
    Field as PydanticField,
    ValidationError,
)


class TrainingStopRequest(PydanticBaseModel):
    save: bool = True
    expected_job_id: str = PydanticField(
        ...,
        min_length = 1,
        max_length = 128,
        pattern = TRAINING_REQUEST_ID_PATTERN,
    )


class TrainingResetRequest(PydanticBaseModel):
    # Stays optional: every Studio build before the train-page rework posts /reset with no
    # body, and those clients only ever reset a finished run. The backend refuses an
    # unscoped reset that would touch a LIVE run instead, so the guard costs no compat.
    expected_job_id: Optional[str] = PydanticField(
        default = None,
        min_length = 1,
        max_length = 128,
        pattern = TRAINING_REQUEST_ID_PATTERN,
    )


router = APIRouter()
logger = get_logger(__name__)

_TRAINING_START_ERROR_RESPONSES = {
    400: {"description": "The requested training configuration or resource is not trainable"},
    409: {"description": "The requested start conflicts with offline, resume, or runtime state"},
    429: {"description": "Remote resource verification was rate-limited"},
    503: {"description": "Remote resource metadata is temporarily unavailable"},
}


def _hub_unreachable() -> bool:
    """Bounded, memoised Hub reachability check.

    hf_env_offline() only reads env vars, so a link that is merely dead burns the full
    5s + 10s metadata budget per leg -- once per resolved address, so 30s at best and
    minutes on a multi-homed resolver -- before the cached fallbacks are consulted. The
    training worker subprocess already guards itself this way (core/training/worker.py);
    the route that spawns it did not. Fails open, so an online start is unchanged.
    """
    memo = hf_reachability_memo()
    if memo is not None:
        return memo
    return hf_dns_dead() or hf_unreachable()


_LOCAL_MODEL_PROBE_LIMIT = 2000
_REMOTE_MODEL_METADATA_TIMEOUT_SECONDS = 5.0
_REMOTE_MODEL_METADATA_RETRY_TIMEOUT_SECONDS = 10.0
_REMOTE_DATASET_METADATA_TIMEOUT_SECONDS = 5.0
_REMOTE_DATASET_METADATA_RETRY_TIMEOUT_SECONDS = 10.0


class _LocalModelProbeIncomplete(RuntimeError):
    pass


def _training_start_error(status_code: int, code: str, message: str) -> HTTPException:
    return HTTPException(
        status_code = status_code,
        detail = {"code": code, "message": message},
    )


def _hf_preflight_error(status_code: int, code: str, message: str) -> HTTPException:
    return _training_start_error(status_code, code, message)


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


# Consecutive 1s polls without a step update that count as a stall. Applied only once
# stepping: the model-load + tokenization phase can take far longer without being stuck.
_PROGRESS_STALL_TIMEOUT_POLLS = 1800  # ~30 min at 1 poll/sec


def _stop_training_if_active(
    backend, *, save: bool, expected_job_id: str
) -> Literal["idle", "stopped", "superseded"]:
    from core.training.lifecycle import training_lifecycle_guard
    with training_lifecycle_guard():
        if not _run_active(backend):
            return "idle"
        current_job_id = getattr(backend, "current_job_id", None)
        if current_job_id is not None and current_job_id != expected_job_id:
            return "superseded"
        stopped = backend.stop_training(
            save = save,
            expected_job_id = expected_job_id,
        )
        return "stopped" if stopped else "idle"


def _is_finalizing(progress, msg_lower: str) -> bool:
    """Worker alive past the last step, `complete` not yet drained.

    The save emits no step updates, so the bar sat at 100% labelled "training", which is
    indistinguishable from a hang. Non-terminal by design: the last step means the optimizer
    loop ended, not that the save succeeded, so completion still comes solely from
    is_completed.
    """
    if any(k in msg_lower for k in ("saving", "merging")):
        return True
    total = getattr(progress, "total_steps", 0) or 0
    step = getattr(progress, "step", 0) or 0
    return total > 0 and step >= total


def _run_finished(backend) -> bool:
    """Whether the run reported terminal (see TrainingBackend.is_run_finished), so status and
    progress stop waiting on the worker to exit. getattr-guarded like the other backend reads
    here: a stand-in without it keeps the old liveness-only behaviour."""
    check = getattr(backend, "is_run_finished", None)
    return bool(check()) if callable(check) else False


def _run_active(backend) -> bool:
    """Liveness minus terminal: a run that reported terminal is done even while its worker
    tears down. The GPU admission guards deliberately keep using is_training_active(), since
    a worker still winding down is still holding its VRAM."""
    return backend.is_training_active() and not _run_finished(backend)


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
        status = {
            "pending": "pending",
            "accepted": "queued",
            "rejected": "error",
        }[record.state],
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


def _trainable_local_roots(path: Path, model_name: Optional[str] = None) -> list[Path]:
    """The snapshot root plus any subdirectory a load reads from.

    Spark-TTS / BiCodec keep config.json and the weights under <snapshot>/LLM, so probing
    only the root reports a perfectly good cached model as having no trainable weights.
    """
    roots = [path]
    if not model_name:
        return roots
    from hub.utils.hf_cache_state import with_load_subdirs

    for name in with_load_subdirs(model_name, ("config.json",)):
        if "/" in name:
            candidate = path / name.rsplit("/", 1)[0]
            if candidate not in roots:
                roots.append(candidate)
    return roots


def _has_trainable_local_weights(path: Path, model_name: Optional[str] = None) -> bool:
    if model_name:
        return any(
            _has_trainable_local_weights(root) for root in _trainable_local_roots(path, model_name)
        )
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
    from utils.security import load_scan_target

    # Registry aliases such as "Spark-TTS-0.5B/LLM" are not repos; probe the repo the trainer
    # really downloads and treat its load subdir as the weight root. Registry lookup only.
    repo_id, load_subdirs = load_scan_target(canonical_model_repo_id(model_name), ())
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
                raise _hf_preflight_error(
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
                    raise _hf_preflight_error(
                        429,
                        "hf_model_verification_rate_limited",
                        "Hugging Face model verification is rate-limited. Retry shortly.",
                    ) from error
            elif status_code is not None:
                raise _hf_preflight_error(
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
            raise _hf_preflight_error(
                503,
                "hf_model_metadata_unavailable",
                (
                    "Hugging Face model metadata is temporarily unavailable. "
                    "Retry before starting training."
                ),
            ) from error

    load_roots = ("", *(f"{subdir.strip('/')}/" for subdir in load_subdirs if subdir))
    root_files: set[str] = set()
    has_gguf = False
    for sibling in getattr(info, "siblings", None) or ():
        name = getattr(sibling, "rfilename", None)
        if not isinstance(name, str):
            continue
        normalized = name.replace("\\", "/")
        if normalized.casefold().endswith(".gguf"):
            has_gguf = True
        for root in load_roots:
            if normalized.startswith(root) and "/" not in normalized[len(root) :]:
                root_files.add(normalized[len(root) :])
    if "adapter_config.json" in root_files:
        return "adapter"
    has_trainable_weights = any(name in root_files for name, _ in _MODEL_WEIGHT_CANDIDATES)
    if has_gguf and not has_trainable_weights:
        return "gguf"
    return None


def _preflight_hf_dataset_request(request: TrainingStartRequest) -> None:
    dataset_id = request.hf_dataset
    if not dataset_id:
        return

    from huggingface_hub.utils import HFValidationError, validate_repo_id

    try:
        validate_repo_id(dataset_id)
    except HFValidationError as error:
        raise _hf_preflight_error(
            400,
            "hf_dataset_verification_failed",
            (
                "The Hugging Face dataset could not be verified. "
                "Check the repository ID and your access token."
            ),
        ) from error

    cached_path = None
    if not request.dataset_streaming:
        from hub.utils.dataset_cache import training_dataset_cache_pin

        cached_path, _ = training_dataset_cache_pin(
            dataset_id,
            request.dataset_snapshot_path or request.dataset_local_path,
        )
        # Only a start that pins the cache may skip Hub verification; an unpinned start still downloads.
        pins_cache = bool(
            request.dataset_known_cached
            or request.dataset_local_path
            or request.dataset_snapshot_path
        )
        if cached_path is not None and pins_cache:
            return

    if hf_env_offline() or _hub_unreachable():
        if request.dataset_streaming:
            raise _hf_preflight_error(
                409,
                "hf_dataset_streaming_offline",
                (
                    "Streaming requires access to the Hugging Face Hub, which cannot be "
                    "reached while offline. Disable streaming to use a cached dataset."
                ),
            )
        if cached_path is not None:
            return
        raise _hf_preflight_error(
            409,
            "hf_dataset_not_cached_offline",
            (
                "The selected Hugging Face dataset is not available in the local cache, "
                "and Hugging Face cannot be reached while offline."
            ),
        )

    from hub.utils.hf_errors import hf_error_status
    from huggingface_hub import HfApi

    api = HfApi(token = request.hf_token or False)
    timeouts = (
        _REMOTE_DATASET_METADATA_TIMEOUT_SECONDS,
        _REMOTE_DATASET_METADATA_RETRY_TIMEOUT_SECONDS,
    )
    for attempt, timeout in enumerate(timeouts):
        try:
            api.dataset_info(dataset_id, timeout = timeout)
            return
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
                raise _hf_preflight_error(
                    422,
                    "hf_dataset_access_denied",
                    (
                        "Hugging Face denied access to this dataset. Add a valid Hugging "
                        "Face token with repository access and accept any required access "
                        "terms, then try again."
                    ),
                ) from error
            retry_available = attempt + 1 < len(timeouts)
            if transient_status:
                if retry_available:
                    continue
                if status_code == 429:
                    raise _hf_preflight_error(
                        429,
                        "hf_dataset_verification_rate_limited",
                        "Hugging Face dataset verification is rate-limited. Retry shortly.",
                    ) from error
            elif status_code is not None:
                raise _hf_preflight_error(
                    status_code,
                    "hf_dataset_verification_failed",
                    (
                        "The Hugging Face dataset could not be verified. "
                        "Check the repository ID and your access token."
                    ),
                ) from error
            elif retry_available:
                continue
            logger.warning(
                "Could not verify Hugging Face dataset %s after retry (%s)",
                dataset_id,
                type(error).__name__,
            )
            raise _hf_preflight_error(
                503,
                "hf_dataset_metadata_unavailable",
                (
                    "Hugging Face dataset metadata is temporarily unavailable. "
                    "Retry before starting training."
                ),
            ) from error


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
        raise _training_start_error(
            400,
            "training_model_gguf_not_trainable",
            "GGUF models are inference-only and cannot be trained.",
        )
    if model_format == "adapter":
        raise _training_start_error(
            400,
            "training_model_adapter_not_trainable",
            "Adapter models are inference-only and cannot be trained as base models.",
        )
    path: Optional[Path] = None
    model_name = request.model_name
    model_local_path: Optional[str] = None
    cached_model_pin: Optional[tuple[str, str]] = None
    offline_mode = False
    if is_local_path(request.model_name):
        try:
            path = Path(normalize_path(request.model_name)).expanduser().resolve(strict = True)
        except (OSError, RuntimeError, ValueError) as error:
            raise _training_start_error(
                400,
                "training_local_model_unavailable",
                "Local model path was not found or could not be accessed.",
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
        from hub.utils.hf_cache_state import (
            latest_snapshot_from_cache_path,
            with_load_subdirs,
        )

        snapshot = None
        offline_mode = hf_env_offline()
        if request.resume_from_checkpoint and request.model_snapshot_path:
            snapshot = latest_snapshot_from_cache_path(
                request.model_snapshot_path,
                "model",
                canonical_model_repo_id(actual_model_repo_id or request.model_name),
                with_load_subdirs(request.model_name, ("config.json", "adapter_config.json")),
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
    if path is None and offline_mode:
        raise _hf_preflight_error(
            409,
            "hf_model_not_cached_offline",
            (
                "Offline mode is enabled, but the selected model is not available in the "
                "local cache. Disable offline mode to download it, or select an on-device "
                "model before starting training."
            ),
        )
    metadata_error: Optional[HTTPException] = None
    if path is None:
        try:
            # Raised inside the try on purpose: the except HTTPException handler still runs
            # _resolve_model_snapshot, so a cached snapshot pins as before without the remote budget.
            if _hub_unreachable():
                raise _hf_preflight_error(
                    503,
                    "hf_model_metadata_unavailable",
                    (
                        "Hugging Face model metadata is temporarily unavailable. "
                        "Retry before starting training."
                    ),
                )
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
                raise _training_start_error(
                    400,
                    "training_remote_model_gguf_only",
                    "GGUF-only remote models are inference-only and cannot be trained.",
                )
            raise _training_start_error(
                400,
                "training_remote_model_adapter_only",
                "Adapter models are inference-only and cannot be trained as base models.",
            )
    has_trainable_weights = _has_trainable_local_weights(path, request.model_name)
    if has_trainable_weights:
        return _ModelPreflightResult(model_name, model_local_path, cached_model_pin)
    if _has_adapter_metadata(path):
        raise _training_start_error(
            400,
            "training_local_model_adapter_only",
            "Adapter-only local models are inference-only and cannot be trained as base models.",
        )
    try:
        has_gguf = _detect_local_gguf(path) is not None
    except _LocalModelProbeIncomplete:
        raise _training_start_error(
            400,
            "training_local_model_scan_incomplete",
            (
                "The local model directory is too large or could not be read safely. "
                "Select its snapshot directory containing config.json and trainable weights."
            ),
        )
    if has_gguf:
        raise _training_start_error(
            400,
            "training_local_model_gguf_only",
            "GGUF-only local models are inference-only and cannot be trained.",
        )
    if metadata_error is not None:
        raise metadata_error
    raise _training_start_error(
        400,
        "training_local_model_weights_missing",
        "The selected local model does not contain trainable weights.",
    )


def _validate_training_platform(request: TrainingStartRequest) -> None:
    from utils.hardware import hardware

    if hardware.DEVICE != hardware.DeviceType.MLX:
        return
    if request.training_type == "Continued Pretraining":
        raise HTTPException(
            status_code = 400,
            detail = "Continued Pretraining is not supported for MLX training yet.",
        )
    if request.is_embedding:
        raise HTTPException(
            status_code = 400,
            detail = "Embedding model training is not supported for MLX training yet.",
        )
    if request.is_dataset_audio:
        raise HTTPException(
            status_code = 400,
            detail = "Audio dataset training is not yet supported on Apple Silicon.",
        )
    if request.use_loftq:
        raise HTTPException(
            status_code = 400,
            detail = "LoftQ is not supported for MLX training yet.",
        )
    if request.use_dora:
        raise HTTPException(
            status_code = 400,
            detail = "DoRA is not supported for MLX training yet.",
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
    except ValidationError as error:
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

    # Off-loop: the first call blocks on detection while the warm is importing torch.
    return await asyncio.to_thread(get_gpu_utilization)


@router.get("/hardware/visible")
async def get_visible_hardware_utilization(current_subject: str = Depends(get_current_subject)):
    from utils.hardware import get_visible_gpu_utilization

    # Off the event loop: the ROCm fallbacks shell out (Windows perf counters, sysfs) and the System view polls this route.
    return await asyncio.to_thread(get_visible_gpu_utilization)


@router.get("/start-requests/{start_request_id}", response_model = TrainingStartRequestStatus)
async def get_training_start_request(
    start_request_id: str = ApiPath(
        ...,
        min_length = 1,
        max_length = 128,
        pattern = TRAINING_REQUEST_ID_PATTERN,
    ),
    current_subject: str = Depends(get_current_subject),
):
    backend = get_training_backend()
    record = backend.get_start_request(start_request_id)
    if record is None:
        raise HTTPException(status_code = 404, detail = "Training start request not found")
    return _start_request_status_response(record)


def _start_request_status_response(record) -> TrainingStartRequestStatus:
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
    start_request_id: str = ApiPath(
        ...,
        min_length = 1,
        max_length = 128,
        pattern = TRAINING_REQUEST_ID_PATTERN,
    ),
    current_subject: str = Depends(get_current_subject),
):
    backend = get_training_backend()
    if not backend.acknowledge_start_request(start_request_id):
        raise HTTPException(
            status_code = 409,
            detail = "Training start request is not ready to acknowledge",
        )
    return {"status": "ok"}


@router.post(
    "/start-requests/{start_request_id}/cancel",
    response_model = TrainingStartRequestStatus,
)
async def cancel_training_start_request(
    start_request_id: str = ApiPath(
        ...,
        min_length = 1,
        max_length = 128,
        pattern = TRAINING_REQUEST_ID_PATTERN,
    ),
    current_subject: str = Depends(get_current_subject),
):
    backend = get_training_backend()
    try:
        outcome, record = await asyncio.to_thread(
            backend.cancel_start_request,
            start_request_id,
        )
    except TrainingStartCancellationCapacityError as exc:
        raise HTTPException(status_code = 429, detail = str(exc)) from exc
    if outcome == "superseded":
        raise HTTPException(
            status_code = 409,
            detail = "Training start request no longer owns the current job",
        )
    return _start_request_status_response(record)


def _background_video_generation_active() -> bool:
    """Whether a video clip is generating on the video backend's worker thread.

    POST /video/generate returns at once and generates in the background, so an
    in-flight clip is invisible to the keep-warm in-flight request count the
    API-key training guards consult; ask the backend directly. Best-effort: a
    probe failure must never block a training start."""
    try:
        from core.inference.video import get_video_backend
        return bool(get_video_backend().generate_progress().get("active"))
    except Exception as e:  # noqa: BLE001
        logger.warning("Could not check video generation state for training guard: %s", e)
        return False


@router.post("/start", responses = _TRAINING_START_ERROR_RESPONSES)
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

        # When Unsloth is driven as an inference API (API-key auth), refuse to start training while
        # a request is in flight: training frees VRAM by unloading the chat model, killing the
        # stream. The UI (session auth) still starts and coexists. Mixed UI+API is not special-cased.
        if via_api_key is True:
            from core.inference.llama_keepwarm import other_inference_request_count
            if (
                other_inference_request_count(current_request_counted = False) > 0
                or _background_video_generation_active()
            ):
                raise HTTPException(
                    status_code = 409,
                    detail = (
                        "Cannot start training over the API while an inference request is in "
                        "progress. Wait for it to finish, or start training from the Unsloth UI."
                    ),
                )

        # No in-process ensure_transformers_version(): worker.py activates it before ML imports.

        # A consented latest-transformers install stage-and-swaps .venv_t5_latest mid-spawn.
        from utils.transformers_latest import is_install_in_progress

        if is_install_in_progress():
            raise HTTPException(
                status_code = 409,
                detail = ("A transformers installation is in progress. Retry when it completes."),
            )

        # S3 dataset loading needs the optional boto3 dependency. Reject early so credentials are
        # never accepted and then silently dropped on a host without boto3.
        if request.s3_config is not None and not request.resume_from_checkpoint:
            from core.training.s3_dataset import boto3_available
            if not boto3_available():
                raise HTTPException(
                    status_code = 501,
                    detail = "S3 dataset loading requires boto3. Install it with: pip install boto3",
                )

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

        # A diffusion LoRA job runs in its own subprocess on the same GPU, so refuse while one is active.
        if _diffusion_training_active():
            message = (
                "A diffusion (Images) LoRA training job is already running. "
                "Stop it before starting an LLM training run."
            )
            _reject_start_request(
                backend,
                reserved_start_request_id,
                message,
            )
            return TrainingJobResponse(
                job_id = "",
                status = "error",
                message = message,
                error = "Diffusion training already active",
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
                detail = "Resume checkpoint must belong to a stopped or errored run with complete saved trainer state."
                # Only when the checkpoint itself is intact. can_resume_run refuses for several reasons and
                # the blocker is computed independently of which one fired, so asking unconditionally would
                # answer a provenance sentence even when the checkpoint is what is missing. has_resume_state
                # is the discriminator can_resume_run itself short-circuits on.
                if resume_run and await asyncio.to_thread(
                    has_resume_state, resume_run.get("output_dir")
                ):
                    from core.training.provenance import (
                        resource_provenance_resume_blocker,
                    )
                    blocker = await asyncio.to_thread(
                        resource_provenance_resume_blocker,
                        training_run_config(resume_run),
                    )
                    if blocker:
                        detail = blocker
                raise HTTPException(status_code = 400, detail = detail)
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

        from utils.hardware import hardware as _hw
        from utils.hardware import ensure_hardware_detected

        await asyncio.to_thread(ensure_hardware_detected)
        _validate_training_platform(request)

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
            # Streaming is HF-only: reject when the request also carries a local dataset path or an
            # S3 config, since those sources cannot be streamed via HF's loader.
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

        if request.hf_dataset:
            await asyncio.to_thread(_preflight_hf_dataset_request, request)

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

        # Latest-sidecar models size and train 16-bit (same flip as chat load): 4-bit is disabled for
        # brand-new architectures, so VRAM checks must not underestimate a load the worker refuses.
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

        # Training page has no trust_remote_code toggle, so honor the YAML default -- but only for
        # genuine first-party (unsloth/nvidia) Hub repos, never a local path or a lookalike name.
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

        # Free VRAM for training: stop export, unload chat unless it can coexist. A before_spawn
        # hook, so it runs only after start_training's guards pass and never for a refused start.
        def _free_vram_for_training() -> None:
            try:
                from core.export import get_export_backend
                exp_backend = get_export_backend()
                # Tear down the export subprocess whenever an export is in flight, not just once a
                # checkpoint is loaded: current_checkpoint is still unset while the worker allocates GPU.
                if exp_backend.current_checkpoint or exp_backend.is_export_active():
                    logger.info("Shutting down export subprocess to free GPU memory for training")
                    exp_backend._shutdown_subprocess()
                    exp_backend.current_checkpoint = None
                    exp_backend.is_vision = False
                    exp_backend.is_peft = False
            except Exception as e:
                logger.warning("Could not shut down export subprocess: %s", e)

            try:
                # A resident or in-flight Images pipeline holds GPU memory the run needs and cannot be
                # cheaply sized, so tear it down unconditionally and release the arbiter. Before the chat block.
                from core.inference import gpu_arbiter
                from core.inference.diffusion_engine_router import (
                    get_active_diffusion_engine,
                )

                # The ACTIVE engine, not the diffusers singleton: on a native (sd_cpp) selection the diffusers backend reports unloaded while the native engine still holds state.
                diffusion = get_active_diffusion_engine()
                if diffusion.is_loaded:
                    logger.info(
                        "Unloading diffusion (Images) model to free GPU memory for training"
                    )
                diffusion.unload()
                gpu_arbiter.release(gpu_arbiter.DIFFUSION)
            except Exception as e:
                logger.warning("Could not unload diffusion model for training: %s", e)

            try:
                # A resident Video pipeline holds GPU memory too and loads under the VIDEO arbiter owner the teardown above never touches. Tear it down the same way and release VIDEO. Must precede the chat block.
                from core.inference import gpu_arbiter
                from core.inference.video import get_video_backend

                video = get_video_backend()
                if video.status().get("loaded"):
                    logger.info("Unloading Video model to free GPU memory for training")
                video.unload()
                gpu_arbiter.release(gpu_arbiter.VIDEO)
            except Exception as e:
                logger.warning("Could not unload video model for training: %s", e)

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

        def _run_backend_start_without_admission() -> bool:
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

        def _run_backend_start() -> bool:
            try:
                # Keep the diffusion admission in the worker thread across the whole spawn: a disconnected
                # request must not release it while the shielded start is still freeing VRAM.
                with _diffusion_gpu_admission():
                    return _run_backend_start_without_admission()
            except _DiffusionStartInFlight as exc:
                _reject_start_request(
                    backend,
                    reserved_start_request_id,
                    str(exc),
                )
                raise

        try:
            start_task = asyncio.create_task(asyncio.to_thread(_run_backend_start))
            success = await asyncio.shield(start_task)
        except _DiffusionStartInFlight as exc:
            return TrainingJobResponse(
                job_id = "",
                status = "error",
                message = str(exc),
                error = "Diffusion training already active",
            )
        except SidecarSwapInProgress as exc:
            # Expected loss of the race against a sidecar install: a retryable 409, not an internal error.
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
        # Deliberate rejections (S3 not implemented, resume validation) keep their original status.
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
    body: TrainingStopRequest, current_subject: str = Depends(get_current_subject)
):
    """
    Stop the currently running training job.

    Body:
        save (bool): If True (default), save the model at the current checkpoint.
        expected_job_id (str): Identifier of the job the caller intends to stop.
    """
    try:
        backend = get_training_backend()
        outcome = await asyncio.to_thread(
            _stop_training_if_active,
            backend,
            save = body.save,
            expected_job_id = body.expected_job_id,
        )
        logger.info("Stop requested: save=%s outcome=%s", body.save, outcome)
        if outcome == "superseded":
            raise HTTPException(
                status_code = 409,
                detail = "The requested training job is no longer active.",
            )
        if outcome == "idle":
            return TrainingStopResponse(
                status = "idle", message = "No training job is currently running"
            )

        return TrainingStopResponse(
            status = "stopped",
            message = "Stop requested. Training will stop at the next safe step.",
        )

    except HTTPException:
        raise
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


def _training_status_identity(backend) -> TrainingStatusIdentitySnapshot:
    snapshot = getattr(backend, "training_status_identity", None)
    if callable(snapshot):
        return snapshot()
    status_start_request = getattr(backend, "status_start_request", None)
    current_start_request_id = getattr(backend, "current_start_request_id", None)
    current_start_request = (
        backend.get_start_request(current_start_request_id)
        if current_start_request_id is not None
        else None
    )
    status_start_request = getattr(backend, "status_start_request", None)
    return TrainingStatusIdentitySnapshot(
        current_job_id = getattr(backend, "current_job_id", "") or "",
        current_start_request_id = current_start_request_id,
        current_start_request = current_start_request,
        status_start_request = status_start_request() if callable(status_start_request) else None,
        new_job_spawn_id = getattr(backend, "_new_job_spawn_id", None),
        spawn_in_progress = getattr(backend, "_spawn_in_progress", False),
    )


def _build_training_status(
    backend, identity: TrainingStatusIdentitySnapshot, is_active: bool
) -> TrainingStatus:
    owner_job_id = identity.current_job_id
    job_id = owner_job_id
    start_request_id = identity.current_start_request_id
    start_request = identity.current_start_request
    status_start_request = identity.status_start_request
    new_job_spawn_id = identity.new_job_spawn_id

    if new_job_spawn_id is not None:
        job_id = new_job_spawn_id
        start_request = next(
            (
                request
                for request in (status_start_request, identity.current_start_request)
                if request is not None and request.job_id == new_job_spawn_id
            ),
            None,
        )
        start_request_id = start_request.start_request_id if start_request is not None else None
    elif is_active:
        start_request = identity.current_start_request
        start_request_id = identity.current_start_request_id
    elif status_start_request is not None and status_start_request.state in {
        "pending",
        "rejected",
    }:
        start_request = status_start_request
        job_id = "" if start_request.state == "rejected" else start_request.job_id
        start_request_id = start_request.start_request_id
    elif start_request is None and (
        status_start_request is not None and status_start_request.job_id == owner_job_id
    ):
        start_request = status_start_request
        start_request_id = status_start_request.start_request_id

    start_request_state = start_request.state if start_request is not None else None
    exposes_owner_state = (
        job_id == owner_job_id
        and start_request_state not in {"pending", "rejected"}
        and new_job_spawn_id is None
    )
    progress = None
    if exposes_owner_state:
        try:
            progress = backend.trainer.get_training_progress()
        except Exception:
            progress = None

    status_message = (
        getattr(progress, "status_message", None) if progress else None
    ) or "Ready to train"
    error_message = getattr(progress, "error", None) if progress else None
    warnings = list(getattr(progress, "warnings", ()) or ()) if progress else []
    if start_request is not None and start_request.state == "pending":
        status_message = start_request.message
        error_message = None
        warnings = []
    elif start_request is not None and start_request.state == "rejected":
        status_message = start_request.message
        error_message = start_request.error or start_request.message
        warnings = []
    elif new_job_spawn_id is not None:
        status_message = "Training job is starting"

    trainer_stopped = getattr(backend, "_should_stop", False)
    if start_request_state == "pending":
        phase = "configuring"
    elif start_request_state == "rejected":
        phase = "error"
    elif new_job_spawn_id is not None:
        phase = "configuring"
    elif error_message:
        phase = "error"
    elif is_active:
        msg_lower = status_message.lower()
        if "loading" in msg_lower or "importing" in msg_lower:
            phase = "loading_model"
        elif any(k in msg_lower for k in ["preparing", "initializing", "configuring"]):
            phase = "configuring"
        elif _is_finalizing(progress, msg_lower):
            phase = "finalizing"
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
            "output_dir": getattr(backend, "_output_dir", None) or None,
        }

    metric_history = None
    if exposes_owner_state and backend.step_history:
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
        eval_enabled = backend.eval_enabled if exposes_owner_state else False,
        message = status_message,
        error = error_message,
        warnings = warnings,
        details = details,
        metric_history = metric_history,
    )


@router.get("/status")
async def get_training_status(current_subject: str = Depends(get_current_subject)):
    """
    Get the current training status.
    """
    try:
        backend = get_training_backend()
        for _ in range(3):
            identity_before = _training_status_identity(backend)
            is_active = await asyncio.to_thread(_run_active, backend)
            identity = _training_status_identity(backend)
            if identity != identity_before:
                continue
            status = _build_training_status(backend, identity, is_active)
            if _training_status_identity(backend) == identity:
                return status
        raise HTTPException(status_code = 409, detail = "Training state changed during status read")
    except HTTPException:
        raise
    except Exception as e:
        raise log_and_http_error(
            e,
            500,
            "Failed to get training status",
            event = "training.status_failed",
            log = logger,
        )


@router.get("/metrics", response_model = TrainingMetricsResponse)
async def get_training_metrics(
    expected_job_id: Optional[str] = None, current_subject: str = Depends(get_current_subject)
):
    """
    Get training metrics (loss, learning rate, steps).
    """
    try:
        backend = get_training_backend()
        job_id = getattr(backend, "current_job_id", "") or ""
        if getattr(backend, "_new_job_spawn_id", None) is not None or (
            expected_job_id is not None and expected_job_id != job_id
        ):
            raise HTTPException(status_code = 409, detail = "Training job was superseded")

        loss_history = list(backend.loss_history)
        lr_history = list(backend.lr_history)
        step_history = list(backend.step_history)
        grad_norm_history = list(getattr(backend, "grad_norm_history", []))
        grad_norm_step_history = list(getattr(backend, "grad_norm_step_history", []))

        if (
            getattr(backend, "_new_job_spawn_id", None) is not None
            or (getattr(backend, "current_job_id", "") or "") != job_id
        ):
            raise HTTPException(status_code = 409, detail = "Training job was superseded")

        current_loss = loss_history[-1] if loss_history else None
        current_lr = lr_history[-1] if lr_history else None
        current_step = step_history[-1] if step_history else None

        return TrainingMetricsResponse(
            job_id = job_id,
            loss_history = loss_history,
            lr_history = lr_history,
            step_history = step_history,
            grad_norm_history = grad_norm_history,
            grad_norm_step_history = grad_norm_step_history,
            current_loss = current_loss,
            current_lr = current_lr,
            current_step = current_step,
        )

    except HTTPException:
        raise
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
    request: Request,
    expected_job_id: Optional[str] = None,
    current_subject: str = Depends(get_current_subject),
):
    """
    Stream training progress via Server-Sent Events (SSE).

    Real-time progress with reconnection support per the SSE spec:
      - `id:` per event so the browser tracks position.
      - `retry:` to control reconnection interval.
      - Named `event:` types (progress, heartbeat, complete, error).
      - Reads `Last-Event-ID` on reconnect to replay missed steps.
    """
    last_event_id = request.headers.get("last-event-id")
    resume_from_step: Optional[int] = None
    if last_event_id is not None:
        try:
            resume_from_step = int(last_event_id)
            # Fires on every reconnect; the meaningful signal is the "replayed N missed steps" line.
            logger.debug(f"SSE reconnect: resuming from step {resume_from_step}")
        except ValueError:
            logger.warning(f"Invalid Last-Event-ID: {last_event_id}")

    async def event_generator():
        backend = get_training_backend()
        backend_job_id = getattr(backend, "current_job_id", "") or ""
        job_id = expected_job_id if expected_job_id is not None else backend_job_id

        def is_current_job() -> bool:
            return (
                getattr(backend, "_new_job_spawn_id", None) is None
                and (getattr(backend, "current_job_id", "") or "") == job_id
            )

        # ── Helpers ──────────────────────────────────────────────
        def run_active() -> bool:
            """A run that reported terminal is done, even while its worker tears down."""
            return _run_active(backend)

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

        if not is_current_job():
            return

        # ── Retry directive ──────────────────────────────────────
        yield "retry: 3000\n\n"

        # ── Replay missed steps on reconnect ─────────────────────
        if not is_current_job():
            return
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
                if not is_current_job():
                    return
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
                    if not is_current_job():
                        return
                    yield format_sse(payload.model_dump_json(), event = "progress", event_id = step_val)
                    replayed += 1
            if replayed:
                logger.info(f"SSE reconnect: replayed {replayed} missed steps")

        # ── Initial status (only on fresh connections) ───────────
        if resume_from_step is None:
            if not is_current_job():
                return
            is_active = await asyncio.to_thread(run_active)
            if not is_current_job():
                return
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
            if not is_current_job():
                return
            yield format_sse(initial_progress.model_dump_json(), event = "progress", event_id = 0)

            if not is_active:
                _live = (getattr(tp, "step", 0) or 0) if tp else 0
                if backend.step_history or _live > 0:
                    final_step = backend.step_history[-1] if backend.step_history else 0
                    final_loss = backend.loss_history[-1] if backend.loss_history else None
                    final_lr = backend.lr_history[-1] if backend.lr_history else None
                    # Histories skip non-finite steps; report the live step with loss=None.
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
                    if not is_current_job():
                        return
                    yield format_sse(
                        payload.model_dump_json(), event = "complete", event_id = final_step
                    )
                else:
                    payload = build_progress(-1, None, None, 0, progress = tp)
                    if not is_current_job():
                        return
                    yield format_sse(
                        payload.model_dump_json(),
                        event = "complete",
                        event_id = 0,
                    )
                return

        # ── Live polling loop ────────────────────────────────────
        last_step = resume_from_step if resume_from_step is not None else -1
        no_update_count = 0
        # The stall timeout applies only once the run is stepping (pre-step prep may legitimately
        # emit no step for a long time). On reconnect to an already-stepping run, seed from the
        # resume point / history, else a worker that hangs after step N never times out.
        seen_live_step = (resume_from_step is not None and resume_from_step > 0) or bool(
            backend.step_history
        )

        while True:
            if not is_current_job():
                return
            is_active = await asyncio.to_thread(run_active)
            if not is_current_job():
                return
            if not is_active:
                break
            # Client gone: end the generator without the final "complete" frame, which a buffered
            # consumer could otherwise read as a finished run while training is still active.
            if await request.is_disconnected():
                return
            if not is_current_job():
                return
            try:
                tp_inner = getattr(getattr(backend, "trainer", None), "training_progress", None)
                live_step = (getattr(tp_inner, "step", 0) or 0) if tp_inner else 0
                if backend.step_history or live_step > 0:
                    current_step = backend.step_history[-1] if backend.step_history else 0
                    current_loss = backend.loss_history[-1] if backend.loss_history else None
                    current_lr = backend.lr_history[-1] if backend.lr_history else None
                    # Histories skip non-finite steps; follow the live step and report its loss.
                    if live_step > current_step:
                        current_step = live_step
                        current_loss = getattr(tp_inner, "loss", None)
                        current_lr = getattr(tp_inner, "learning_rate", current_lr)
                    current_total_steps = (
                        getattr(tp_inner, "total_steps", current_step) if tp_inner else current_step
                    )
                    current_epoch = getattr(tp_inner, "epoch", None) if tp_inner else None

                    if current_step != last_step:
                        progress_payload = build_progress(
                            current_step,
                            current_loss,
                            current_lr,
                            current_total_steps,
                            current_epoch,
                            progress = tp_inner,
                        )
                        if not is_current_job():
                            return
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
                        if no_update_count % 10 == 0:
                            heartbeat_payload = build_progress(
                                current_step,
                                current_loss,
                                current_lr,
                                current_total_steps,
                                current_epoch,
                                progress = tp_inner,
                            )
                            if not is_current_job():
                                return
                            yield format_sse(
                                heartbeat_payload.model_dump_json(),
                                event = "heartbeat",
                                event_id = current_step,
                            )
                else:
                    # No steps yet, but training is active (model loading, etc.).
                    no_update_count += 1
                    if no_update_count % 5 == 0:
                        # Pull total_steps + status so the frontend can show "Tokenizing..." etc.
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
                        if not is_current_job():
                            return
                        yield format_sse(
                            preparing_payload.model_dump_json(),
                            event = "heartbeat",
                            event_id = 0,
                        )

                # Fires only once stepping: a long pre-first-step prep phase is not a stall.
                if seen_live_step and no_update_count > _PROGRESS_STALL_TIMEOUT_POLLS:
                    logger.warning("Progress stream timeout - no updates received")
                    tp_timeout = getattr(
                        getattr(backend, "trainer", None), "training_progress", None
                    )
                    timeout_payload = build_progress(last_step, None, None, 0, progress = tp_timeout)
                    if not is_current_job():
                        return
                    yield format_sse(
                        timeout_payload.model_dump_json(),
                        event = "error",
                        event_id = last_step if last_step >= 0 else 0,
                    )
                    return

                await asyncio.sleep(1)  # Poll every second

            except Exception as e:
                if not is_current_job():
                    return
                logger.error(f"Error in progress stream: {e}", exc_info = True)
                tp_error = getattr(getattr(backend, "trainer", None), "training_progress", None)
                error_payload = build_progress(0, None, None, 0, progress = tp_error)
                if not is_current_job():
                    return
                yield format_sse(
                    error_payload.model_dump_json(),
                    event = "error",
                    event_id = last_step if last_step >= 0 else 0,
                )
                return

        # ── Final "complete" event ───────────────────────────────
        if not is_current_job():
            return
        final_step = backend.step_history[-1] if backend.step_history else last_step
        final_loss = backend.loss_history[-1] if backend.loss_history else None
        final_lr = backend.lr_history[-1] if backend.lr_history else None
        final_tp = getattr(getattr(backend, "trainer", None), "training_progress", None)
        # If the run ended on a non-finite stretch, report the live step with loss=None.
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
        if not is_current_job():
            return
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


# ── Diffusion (SDXL) LoRA training ────────────────────────────────────────────
# A separate, lightweight job path: diffusion runs use DiffusionTrainingService (its own subprocess + event pump), not the LLM TrainingBackend.


def _diffusion_training_active() -> bool:
    """Whether a diffusion (SDXL) LoRA job is currently running. Best-effort so the
    interlock never blocks a start just because the service could not be imported."""
    try:
        from core.training.diffusion_training_service import get_diffusion_training_service
        return get_diffusion_training_service().is_active()
    except Exception:  # noqa: BLE001
        return False


class _DiffusionStartInFlight(RuntimeError):
    """An LLM start lost the race to a diffusion start (route: refuse, don't spawn)."""


@contextlib.contextmanager
def _diffusion_gpu_admission():
    """Hold the diffusion service's GPU admission across the LLM spawn.

    Makes the cross-trainer admission atomic: entering re-tests the diffusion state under the
    service's own lock and raises if a diffusion run is reserved or active, and while it is held
    the diffusion ``reserve()`` refuses. So of two near-simultaneous starts of different types,
    exactly one proceeds. Fails OPEN on an import/health failure, like every other guard here: a
    chat-only install has no diffusion service and must still be able to train."""
    try:
        from core.training.diffusion_training_service import (
            TrainingActiveError,
            get_diffusion_training_service,
        )
        service = get_diffusion_training_service()
    except Exception:  # noqa: BLE001 -- no diffusion stack: nothing to coordinate with
        yield
        return
    try:
        cm = service.gpu_load_admission()
    except Exception:  # noqa: BLE001
        yield
        return
    try:
        cm.__enter__()
    except TrainingActiveError as exc:
        raise _DiffusionStartInFlight(
            "A diffusion (Images) LoRA training job is already running. "
            "Stop it before starting an LLM training run."
        ) from exc
    try:
        yield
    finally:
        cm.__exit__(None, None, None)


def _require_diffusion_dataset_mutable() -> None:
    """Reject a dataset mutation while a diffusion run is active.

    The trainer re-opens dataset images during the loop, so mutating underneath it makes the run
    nondeterministic or raises a FileNotFoundError mid-step. Fails open (a service-import failure
    never blocks a mutation on an unknowable state), matching the start interlock."""
    if _diffusion_training_active():
        raise HTTPException(
            status_code = 409,
            detail = (
                "Training images cannot be changed while diffusion training is active. "
                "Stop the run before uploading, importing, editing captions, or deleting images."
            ),
        )


def diffusion_dataset_interlock():
    """Dependency holding the dataset interlock for a whole mutating request.

    The check above only covers the instant it runs: every one of these endpoints then hands its
    filesystem work to a thread, and a ``/diffusion/start`` reserving in that gap would move
    captions or images underneath the preflight or the running trainer. As a yield dependency the
    registration spans the endpoint, so ``reserve()`` sees it and refuses instead. Fails open on an
    import error, like the check it replaces."""
    try:
        from core.training.diffusion_training_service import (
            TrainingActiveError,
            get_diffusion_training_service,
        )
        service = get_diffusion_training_service()
    except Exception:  # noqa: BLE001 -- unknowable state never blocks a mutation
        yield
        return
    try:
        with service.dataset_mutation():
            yield
    except TrainingActiveError as exc:
        raise HTTPException(status_code = 409, detail = str(exc)) from exc


def _free_gpu_for_diffusion_training() -> None:
    """Free GPU residents before the diffusion trainer spawns its own SDXL pipeline.

    The trainer subprocess loads a full SDXL pipeline; an export worker, a resident
    Images pipeline, or loaded chat models would otherwise keep their VRAM allocated and
    OOM the run. Mirrors the LLM start path's pre-spawn cleanup (export + diffusion
    pipeline + chat). Best-effort: a failure to free one resident never blocks the start."""
    try:
        from core.export import get_export_backend
        exp_backend = get_export_backend()
        if exp_backend.current_checkpoint or exp_backend.is_export_active():
            logger.info("Shutting down export subprocess to free GPU memory for diffusion training")
            exp_backend._shutdown_subprocess()
            exp_backend.current_checkpoint = None
            exp_backend.is_vision = False
            exp_backend.is_peft = False
    except Exception as e:  # noqa: BLE001
        logger.warning("Could not shut down export subprocess: %s", e)

    try:
        from core.inference import gpu_arbiter
        from core.inference.diffusion_engine_router import get_active_diffusion_engine

        # The ACTIVE engine, not the diffusers singleton: on a native (sd_cpp) selection the resident sd-server still holds the GPU, so unloading only the singleton is a no-op.
        diffusion = get_active_diffusion_engine()
        if diffusion.is_loaded:
            logger.info("Unloading resident Images pipeline to free GPU memory for training")
        diffusion.unload()  # no-op when nothing is loaded; also preempts an in-flight load
        gpu_arbiter.release(gpu_arbiter.DIFFUSION)
    except Exception as e:  # noqa: BLE001
        logger.warning("Could not unload Images pipeline for diffusion training: %s", e)

    try:
        # A resident Video pipeline loads under the VIDEO arbiter owner the Images teardown does not free; unload it too and release VIDEO.
        from core.inference import gpu_arbiter
        from core.inference.video import get_video_backend

        video = get_video_backend()
        if video.status().get("loaded"):
            logger.info("Unloading resident Video pipeline to free GPU memory for training")
        video.unload()  # no-op when nothing is loaded; also preempts an in-flight load
        gpu_arbiter.release(gpu_arbiter.VIDEO)
    except Exception as e:  # noqa: BLE001
        logger.warning("Could not unload Video pipeline for diffusion training: %s", e)

    try:
        # The SDXL trainer footprint cannot be cheaply sized against a resident chat model, so free chat unconditionally rather than risk an OOM.
        from routes.training_vram import free_chat_models_for_training, summarize_resident_chat
        if summarize_resident_chat()["any"]:
            freed = free_chat_models_for_training(reason = "diffusion training starting")
            logger.info("Freed chat model(s) for diffusion training: %s", freed)
    except Exception as e:  # noqa: BLE001
        logger.warning("Could not free chat models for diffusion training: %s", e)


def _preflight_gated_base(base_model: str, hf_token: Optional[str]) -> None:
    """HEAD a remote base repo's model_index.json with the caller's token; raise HTTP 400 on
    401/403 (gated / unauthorized) with an actionable message. Best-effort: a local path,
    a non-repo string, or a network hiccup passes through so the trainer can surface any real
    load error itself. Runs before GPU teardown so a doomed start never evicts a loaded model."""
    import urllib.error
    import urllib.request

    repo = (base_model or "").strip()
    # Only remote 'org/name' repos are gated; skip local paths and single-file names.
    if (
        not repo
        or repo.count("/") != 1
        or repo.startswith((".", "/", "~"))
        or repo.endswith(".gguf")
    ):
        return
    url = f"https://huggingface.co/{repo}/resolve/main/model_index.json"
    headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}
    req = urllib.request.Request(url, method = "HEAD", headers = headers)
    try:
        urllib.request.urlopen(req, timeout = 5)
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            raise HTTPException(
                status_code = 400,
                detail = (
                    f"Access to '{repo}' is gated or unauthorized. Accept the model's license "
                    f"on its Hugging Face page and add your HF token in Studio settings, then "
                    f"try again."
                ),
            )
        # 404 (e.g. a repo without a root model_index.json) and other codes are not an access problem; let the trainer surface any genuine load error.
    except Exception:  # noqa: BLE001 -- network/DNS hiccup must not block a start
        return


def _resolve_diffusion_data_dir(raw: str) -> Path:
    """Resolve a diffusion-training ``data_dir``. The upload/labeling routes create and
    manage image datasets directly under ``datasets_root()`` and the UI passes the bare
    folder name back as ``data_dir``, but the generic :func:`resolve_dataset_path`
    searches the LLM uploads and recipe dataset roots FIRST -- so an unrelated upload
    file or recipe folder sharing that name would shadow the just-uploaded image
    dataset (preflight 400 "not a directory", or training the wrong data). Prefer the
    image dataset root for a bare single-component name that exists there; everything
    else (explicit "uploads/..." / "recipes/..." prefixes, absolute paths, missing
    names) resolves exactly as before."""
    from utils.paths import datasets_root

    value = str(raw or "").strip()
    if value and "\x00" not in value:
        p = Path(value)
        # A single component that is not "..", so joining under datasets_root() cannot escape it.
        if not p.is_absolute() and len(p.parts) == 1 and p.parts[0] != "..":
            direct = datasets_root() / value
            # Route a bare name through the same protected resolver the CRUD routes use, so a symlink to an external directory is rejected here too. A broken symlink is included so it is rejected, not passed on.
            if direct.is_dir() or direct.is_symlink():
                return _resolve_dataset_folder(value)
    return resolve_dataset_path(raw)


@router.post("/diffusion/start", response_model = DiffusionTrainingStartResponse)
async def start_diffusion_training(
    body: DiffusionTrainingStartRequest,
    current_subject: str = Depends(get_current_subject),
    via_api_key: bool = Depends(authenticated_via_api_key),
):
    """Start an SDXL LoRA training job from an image + caption dataset."""
    from core.training.diffusion_training_service import get_diffusion_training_service

    # Under API-key auth, refuse to start training while a request is in flight: _free_gpu_for_diffusion_training() below unloads the chat backends, killing the stream. Mirrors start_training.
    if via_api_key is True:
        from core.inference.llama_keepwarm import other_inference_request_count
        if (
            other_inference_request_count(current_request_counted = False) > 0
            or _background_video_generation_active()
        ):
            raise HTTPException(
                status_code = 409,
                detail = (
                    "Cannot start diffusion (Images) training over the API while an inference "
                    "request is in progress. Wait for it to finish, or start training from the "
                    "Studio UI."
                ),
            )

    # Interlock: refuse while an LLM training run holds the GPU (symmetric with the diffusion check in start_training), so the two trainers never contend for VRAM.
    try:
        if get_training_backend().is_training_active():
            raise HTTPException(
                status_code = 409,
                detail = (
                    "An LLM training job is already running. "
                    "Stop it before starting diffusion (Images) training."
                ),
            )
    except HTTPException:
        raise
    except Exception:  # noqa: BLE001 -- backend import/health issue must not block a start
        pass

    # Resolve + contain the dataset and output paths BEFORE spawning: the trainer subprocess would otherwise resolve them relative to its own cwd.
    config = body.model_dump()
    try:
        from utils.paths import outputs_root, resolve_output_dir

        config["data_dir"] = str(_resolve_diffusion_data_dir(config["data_dir"]))
        # A name that cleans away to nothing ("." / "outputs" / "./.") resolves to the outputs ROOT, where the trainer would write the adapter flat and the is_dir()-filtered listings could never see it.
        root = outputs_root().resolve()
        out_dir = resolve_output_dir(config["output_dir"])
        if Path(out_dir).resolve() == root:
            raise HTTPException(
                status_code = 400,
                detail = (
                    f"'{config['output_dir']}' is the outputs folder itself, not a run inside it. "
                    "Pick a name for this run."
                ),
            )
        config["output_dir"] = str(out_dir)
        # The persistent conditioning cache is another trainer-written directory, so it gets the same containment. Blank/None means the in-memory cache, so it must not resolve to the outputs root.
        cond_cache = str(config.get("cond_cache_dir") or "").strip()
        cond_cache_dir = resolve_output_dir(cond_cache) if cond_cache else None
        # Same collapse, but the cache has an honest "off" to fall back to: one flat safetensors per cached latent in the trained-models directory is never what was meant.
        if cond_cache_dir is not None and Path(cond_cache_dir).resolve() == root:
            cond_cache_dir = None
        config["cond_cache_dir"] = str(cond_cache_dir) if cond_cache_dir is not None else None
    except ValueError as e:
        raise HTTPException(status_code = 400, detail = str(e))

    # Validate the config BEFORE freeing resident GPU workloads, so a refused start never tears down the user's chat/Images model. service.start() re-runs this before spawn.
    from core.training.diffusion_lora_trainer import _config_from_dict

    try:
        normalized_cfg = _config_from_dict(config).normalized()
    except ValueError as e:
        raise HTTPException(status_code = 400, detail = str(e))

    # Only the DiT trainer reads cond_cache_dir; the SDXL trainer's latent cache is per-process
    # and in-memory. Checked against the RESOLVED family, not the request field.
    if cond_cache and normalized_cfg.resolved_family == "sdxl":
        raise HTTPException(
            status_code = 400,
            detail = (
                "cond_cache_dir is not supported for the sdxl family: its trainer uses a "
                "per-run in-memory latent cache and would ignore the persistent one. Omit it, "
                "or train a DiT family (flux.1, flux.2-klein, flux.2-dev, qwen-image, "
                "z-image, krea-2), which reuses conditioning across runs."
            ),
        )

    # Preflight the requested DiT precision BEFORE freeing GPU residents: the trainer's own
    # checks fire only in the child, AFTER eviction. Fail fast (400).
    from core.training.diffusion_train_common import training_precision_preflight_error

    _precision_reason = training_precision_preflight_error(
        normalized_cfg.resolved_family, normalized_cfg.base_precision
    )
    if _precision_reason:
        raise HTTPException(status_code = 400, detail = _precision_reason)

    # Run the trainers' trust gate here too, so an untrusted/typoed base 400s BEFORE freeing GPU residents rather than failing in the child.
    from core.training.diffusion_train_common import _assert_trusted_base_model

    try:
        _assert_trusted_base_model(config.get("base_model", ""))
    except ValueError as e:
        raise HTTPException(status_code = 400, detail = str(e))

    # Preflight access to a gated base repo with the user's token BEFORE freeing GPU residents,
    # so a missing token fails fast (400). In a worker thread: blocking urlopen HEAD (5s).
    await asyncio.to_thread(
        _preflight_gated_base, config.get("base_model", ""), config.get("hf_token")
    )

    from core.training import diffusion_train_common as _dtc

    service = get_diffusion_training_service()
    # Reserve the training slot BEFORE the dataset preflight: is_active() otherwise flips true
    # only at service.start(), so a concurrent upload/delete could mutate the dataset meanwhile.
    reserved = False
    try:
        service.reserve()
        reserved = True
        # Preflight the dataset: a missing/empty/uncaptionable data_dir otherwise fails inside the trainer AFTER eviction. Same discovery the trainer runs, so the two cannot disagree.
        try:
            await asyncio.to_thread(
                _dtc.discover_image_caption_pairs,
                config["data_dir"],
                instance_prompt = config.get("instance_prompt") or None,
                caption_column = config.get("caption_column") or "text",
                # Decode-probe every image now (cheap PIL header check) so a corrupt/zero-byte upload 400s BEFORE the GPU teardown.
                verify_images = True,
            )
        except (FileNotFoundError, ValueError) as e:
            raise HTTPException(status_code = 400, detail = str(e))
        # Free resident GPU workloads before the trainer loads its own pipeline. Offloaded: the teardown blocks on generation locks and a subprocess join.
        await asyncio.to_thread(_free_gpu_for_diffusion_training)
        job_id = service.start(config)
    except ValueError as e:
        raise HTTPException(status_code = 400, detail = str(e))
    except RuntimeError as e:
        # A job is already running (or a start is already reserved), or a dataset mutation is open (DatasetMutationInFlight) -- the same interlock from the other side, so also a 409.
        raise HTTPException(status_code = 409, detail = str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise log_and_http_error(
            e,
            500,
            "Failed to start diffusion training",
            event = "diffusion_training.start_failed",
            log = logger,
        )
    finally:
        # On success the live proc keeps is_active() true; on failure this clears the reservation. Only the request that reserved clears it.
        if reserved:
            service.unreserve()
    return DiffusionTrainingStartResponse(job_id = job_id, status = "running")


@router.post("/diffusion/stop")
async def stop_diffusion_training(
    body: Optional[DiffusionTrainingStopRequest] = None,
    current_subject: str = Depends(get_current_subject),
):
    """Request a clean stop of the running diffusion training job. The optional body's
    ``save`` mirrors the LLM /stop: true (default, also for an empty POST) exports the
    partial adapter, false cancels without saving one."""
    from core.training.diffusion_training_service import get_diffusion_training_service

    save = body.save if body is not None else True
    stopped = get_diffusion_training_service().stop(save = save)
    return {"status": "stopping" if stopped else "idle"}


@router.get("/diffusion/status", response_model = DiffusionTrainingStatusResponse)
async def diffusion_training_status(current_subject: str = Depends(get_current_subject)):
    """Poll the current diffusion training job's status/progress (JSON)."""
    from core.training.diffusion_training_service import get_diffusion_training_service

    snap = get_diffusion_training_service().status()
    # Fold the service's flat history arrays into the nested metric_history the UI charts.
    metric_history = DiffusionMetricHistory(
        steps = snap.pop("metric_steps", []),
        loss = snap.pop("metric_loss", []),
        lr = snap.pop("metric_lr", []),
        grad_norm = snap.pop("metric_grad_norm", []),
    )
    return DiffusionTrainingStatusResponse(**snap, metric_history = metric_history)


@router.get("/diffusion/runs", response_model = DiffusionTrainingRunsResponse)
async def list_diffusion_training_runs(
    limit: int = 20, current_subject: str = Depends(get_current_subject)
):
    """Previous diffusion training runs (terminal), newest first, from the persisted
    per-run records. Summaries only; fetch one run for its config + metric logs."""
    from core.training.diffusion_training_service import list_diffusion_runs

    summaries: list[DiffusionTrainingRunSummary] = []
    for r in list_diffusion_runs(limit = limit):
        # list_diffusion_runs skips non-dict / missing-id records, but a wrong-typed field would still raise here; catch per record so one bad file never breaks the panel.
        try:
            summaries.append(DiffusionTrainingRunSummary(**r))
        except ValidationError:
            continue
    return DiffusionTrainingRunsResponse(runs = summaries)


@router.get("/diffusion/runs/{job_id}", response_model = DiffusionTrainingRunDetail)
async def get_diffusion_training_run(
    job_id: str, current_subject: str = Depends(get_current_subject)
):
    """One persisted diffusion run's full record: summary + scrubbed start config + the
    step/loss/grad-norm logs (for re-plotting a past run's charts)."""
    from core.training.diffusion_training_service import get_diffusion_run

    rec = get_diffusion_run(job_id)
    # A valid-JSON file that is not an object makes DiffusionTrainingRunDetail(**rec) raise TypeError, not the ValidationError caught below. Treat any non-dict record as absent.
    if not isinstance(rec, dict):
        raise HTTPException(status_code = 404, detail = "No such training run.")
    try:
        return DiffusionTrainingRunDetail(**rec)
    except ValidationError:
        # A malformed on-disk record (hand-edited / older shape) reads as absent rather than 500 the endpoint, like the list route skips bad records.
        raise HTTPException(status_code = 404, detail = "No such training run.")


# Extensions accepted into an image-training dataset folder: images the trainer reads, plus its caption sources (per-image sidecars and metadata/captions jsonl).
_DIFFUSION_DATASET_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
_DIFFUSION_DATASET_TEXT_EXTS = {".txt", ".caption", ".jsonl"}


def _resolve_dataset_caption(
    folder: Path, image_path: Path, meta_captions: dict[str, str]
) -> Optional[str]:
    """Resolve an image's caption using the same sidecar > metadata precedence the trainer
    applies in ``discover_image_caption_pairs``. A per-image .txt/.caption sidecar wins and
    is stripped, so an empty (tombstone) sidecar shadows metadata and yields "" -- the
    trainer then skips that image (``if caption:``), so it must not count as captioned."""
    caption: Optional[str] = None
    sidecar_present = False
    for ext in (".txt", ".caption"):
        sidecar = image_path.with_suffix(ext)
        if sidecar.is_file():
            sidecar_present = True
            try:
                caption = sidecar.read_text(encoding = "utf-8").strip()
            except (OSError, UnicodeError):
                # Unreadable / invalid UTF-8 sidecar: the EMPTY TOMBSTONE, not "no sidecar", matching the
                # trainer. Uploads accept raw bytes, so reading it as absent would show a replaced caption.
                caption = ""
            break
    if not sidecar_present:
        try:
            rel = image_path.relative_to(folder).as_posix()
        except ValueError:
            rel = None
        caption = meta_captions.get(image_path.name) or (
            meta_captions.get(rel) if rel is not None else None
        )
    return caption


_DATASET_IMPORT_LOCKS: Dict[str, "threading.Lock"] = {}
_DATASET_IMPORT_LOCKS_GUARD = threading.Lock()


def _dataset_import_lock(folder: Path) -> "threading.Lock":
    """One lock per dataset folder, so two imports cannot fill the same empty name at once.

    Keyed by the resolved path (the same folder can be reached by different names), and kept for
    the process lifetime: there are a handful of dataset folders and a Lock is tiny, while
    dropping one while another thread holds it would defeat the point."""
    key = str(folder.resolve(strict = False))
    with _DATASET_IMPORT_LOCKS_GUARD:
        lock = _DATASET_IMPORT_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _DATASET_IMPORT_LOCKS[key] = lock
        return lock


def _import_response(
    entry: dict, folder: Path, *, imported: int
) -> "DiffusionDatasetImportResponse":
    summary = _diffusion_dataset_summary(folder)
    return DiffusionDatasetImportResponse(
        name = folder.name,
        path = str(folder),
        image_count = summary.image_count,
        caption_count = summary.caption_count,
        imported = imported,
        license = entry["license"],
        source_repo = entry["repo"],
    )


def _diffusion_dataset_summary(folder: Path) -> DiffusionDatasetSummary:
    # Count an image as captioned only when it resolves to a NON-EMPTY caption via the trainer's sidecar-over-metadata precedence: an empty tombstone makes the trainer skip the image.
    meta_captions = _load_metadata_captions(folder)
    images = captions = 0
    for f in folder.iterdir():
        if not f.is_file() or f.suffix.lower() not in _DIFFUSION_DATASET_IMAGE_EXTS:
            continue
        images += 1
        if _resolve_dataset_caption(folder, f, meta_captions):
            captions += 1
    return DiffusionDatasetSummary(
        name = folder.name, path = str(folder), image_count = images, caption_count = captions
    )


@router.get("/diffusion/info", response_model = DiffusionTrainingInfoResponse)
async def diffusion_training_info(current_subject: str = Depends(get_current_subject)):
    """Describe where diffusion training reads/writes, and list usable dataset folders.

    A dataset folder is any direct child of the datasets root that contains at least one
    image. The UI uses this to offer a picker instead of a blind free-text path."""
    from utils.paths import datasets_root, outputs_root

    def scan() -> DiffusionTrainingInfoResponse:
        root = datasets_root()
        found: list[DiffusionDatasetSummary] = []
        try:
            # Skip hidden dirs: never user datasets, and an in-progress example import stages into a dot-prefixed sibling.
            children = sorted(
                p
                for p in root.iterdir()
                # Skip symlinked dirs: the CRUD resolver rejects them, so discovery must not advertise one as selectable.
                if p.is_dir() and not p.is_symlink() and not p.name.startswith(".")
            )
        except OSError:
            children = []
        for child in children:
            try:
                summary = _diffusion_dataset_summary(child)
            except OSError:
                continue
            if summary.image_count > 0:
                found.append(summary)
        from core.training.diffusion_train_common import family_train_infos

        families = [DiffusionTrainableFamily(**info) for info in family_train_infos()]
        return DiffusionTrainingInfoResponse(
            datasets_root = str(root),
            outputs_root = str(outputs_root()),
            datasets = found,
            families = families,
        )

    return await asyncio.to_thread(scan)


_DATASET_NAME_RE = None  # compiled lazily; module keeps its import block torch-free


# Reserved in EVERY directory on Windows, with or without an extension (NUL.txt is NUL). The superscript COM/LPT digits count as digits to Win32 and are reserved too.
_WINDOWS_RESERVED_NAMES = frozenset(
    {"con", "prn", "aux", "nul"}
    | {f"com{d}" for d in "123456789¹²³"}
    | {f"lpt{d}" for d in "123456789¹²³"}
)


_DATASETS_CASE_INSENSITIVE: Optional[bool] = None


def _dataset_folder_is_case_insensitive(folder: Path) -> bool:
    """True when ``folder`` cannot hold two names differing only by case.

    Probed once per process against the real filesystem rather than keyed off ``sys.platform``:
    NTFS and the default APFS fold case, but macOS also ships case-SENSITIVE APFS volumes and a
    Linux host can keep its Studio home on an exFAT/NTFS mount. A failed probe answers False,
    which keeps the case-sensitive behaviour (two names, two files) unchanged.
    """
    global _DATASETS_CASE_INSENSITIVE
    if _DATASETS_CASE_INSENSITIVE is None:
        import tempfile
        try:
            with tempfile.NamedTemporaryFile(prefix = ".case-probe-", dir = folder) as probe:
                probe_name = Path(probe.name).name
                _DATASETS_CASE_INSENSITIVE = (folder / probe_name.upper()).exists()
        except OSError:
            return False
    return _DATASETS_CASE_INSENSITIVE


def _clean_diffusion_dataset_name(name: str) -> str:
    """Validate a dataset folder name: a single path component, no traversal, printable.

    Windows path rules are applied on EVERY platform, not just Windows: a dataset created on one
    machine is opened on another, and both failures are silent or confusing. A reserved device name
    dies in mkdir with an unhandled OSError, and a trailing period is stripped by Win32
    normalization, so an upload to the "new" dataset 'photos.' would quietly write into the
    existing 'photos'."""
    import re

    global _DATASET_NAME_RE
    if _DATASET_NAME_RE is None:
        _DATASET_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._ -]{0,127}$")
    cleaned = (name or "").strip()
    if not _DATASET_NAME_RE.fullmatch(cleaned) or ".." in cleaned:
        raise HTTPException(
            status_code = 400,
            detail = (
                "Dataset name must be a plain folder name (letters, numbers, dots, "
                "dashes, spaces; no slashes), e.g. 'my-style-photos'."
            ),
        )
    if cleaned.endswith("."):
        raise HTTPException(
            status_code = 400,
            detail = (
                "Dataset name cannot end with a period: Windows strips it, so this name would "
                f"open the existing '{cleaned.rstrip('.')}' dataset instead of a new one."
            ),
        )
    # The stem alone is checked, since NUL.txt is the NUL device too.
    if cleaned.split(".", 1)[0].casefold() in _WINDOWS_RESERVED_NAMES:
        raise HTTPException(
            status_code = 400,
            detail = (
                f"'{cleaned}' is a reserved device name on Windows and cannot be a folder. "
                "Pick another dataset name."
            ),
        )
    return cleaned


@router.post("/diffusion/dataset", response_model = DiffusionDatasetUploadResponse)
async def upload_diffusion_dataset(
    name: str = Form(...),
    files: list[UploadFile] = File(...),
    current_subject: str = Depends(get_current_subject),
    _interlock: None = Depends(diffusion_dataset_interlock),
):
    """Upload training images (and optional caption .txt / metadata.jsonl files) into a
    named folder under the Studio datasets root, creating it if needed. Repeat uploads
    into the same name accumulate, so large datasets can arrive in batches. The returned
    name can be passed directly as ``data_dir`` to /diffusion/start."""
    import os
    import tempfile

    from utils.upload_limits import get_upload_limit_bytes, get_upload_limit_label

    _require_diffusion_dataset_mutable()
    cleaned = _clean_diffusion_dataset_name(name)
    # Run the same symlink + root-containment check as the read/caption/delete endpoints before any write, so a symlinked name cannot make the upload write outside root.
    folder = _resolve_dataset_folder(name, must_exist = False)
    folder.mkdir(parents = True, exist_ok = True)
    # Serialize against a concurrent import into the SAME folder: the training interlock counts
    # mutations rather than excluding them. The duplicate-stem check below is inside the lock.
    _lock = _dataset_import_lock(folder)
    if not _lock.acquire(blocking = False):
        raise HTTPException(
            status_code = 409,
            detail = (
                f"An import into '{folder.name}' is already running. Wait for it to finish, "
                "then upload again."
            ),
        )
    try:
        limit_bytes = get_upload_limit_bytes()
        total_bytes = 0
        uploaded = 0
        allowed = _DIFFUSION_DATASET_IMAGE_EXTS | _DIFFUSION_DATASET_TEXT_EXTS
        # Validate every filename up front so a valid image ahead of a bad one is not left on disk when the 400 fires; the upload is all-or-nothing.
        names: list[str] = []
        for f in files:
            # Normalise to a safe basename. Path.name does not split on a backslash on POSIX, so fold
            # backslashes first or a Windows client's path is stored verbatim, ".." and all.
            filename = Path((f.filename or "").replace("\\", "/")).name.strip().replace("\x00", "")
            ext = Path(filename).suffix.lower()
            if not filename or ".." in filename or ext not in allowed:
                exts = ", ".join(sorted(allowed))
                raise HTTPException(
                    status_code = 400,
                    detail = f"Unsupported file '{f.filename}'. Allowed: {exts}",
                )
            # Reject an EXACT duplicate name within THIS batch: the same-name exemption below is for
            # SEPARATE repeat uploads, while inside one batch the later replace would discard the earlier.
            fname_cf = filename.casefold()
            if filename in names:
                raise HTTPException(
                    status_code = 400,
                    detail = (
                        f"Duplicate file '{filename}' appears more than once in this upload. "
                        "Files sharing a name would overwrite each other; rename one before "
                        "uploading."
                    ),
                )
            # Reject a second IMAGE sharing this stem but differing by extension (sample.png vs .jpg):
            # both resolve to the same <stem>.txt sidecar. Caption files are exempt.
            if ext in _DIFFUSION_DATASET_IMAGE_EXTS:
                stem = Path(filename).stem
                # Compare stems case-insensitively: on case-insensitive filesystems two stems differing only
                # by case resolve to the SAME sidecar. cat.PNG vs cat.png is not exempt.
                stem_cf = stem.casefold()

                def _shares_sidecar(other_name: str) -> bool:
                    other = Path(other_name)
                    if (
                        other_name == filename
                        or other.suffix.lower() not in _DIFFUSION_DATASET_IMAGE_EXTS
                        or other.stem.casefold() != stem_cf
                    ):
                        return False
                    # A casefold-equal full name is exempt unless the stems match EXACTLY (extension-case variants collide on one sidecar on case-sensitive filesystems).
                    return other.stem == stem or other_name.casefold() != fname_cf

                clash = next(
                    (p.name for p in folder.iterdir() if p.is_file() and _shares_sidecar(p.name)),
                    None,
                )
                if clash is None:
                    clash = next((n for n in names if _shares_sidecar(n)), None)
                if clash is not None:
                    raise HTTPException(
                        status_code = 400,
                        detail = (
                            f"Duplicate image name '{stem}'. '{clash}' is already in this "
                            f"dataset; two images sharing a name would share one '{stem}.txt' "
                            f"caption. Rename one before uploading."
                        ),
                    )
            # Reject a CASE variant of a name already in this batch, but only where the filesystem folds
            # case: there 'Cat.png' and 'cat.png' are ONE destination, so the commit would replace the
            # first staged part with the second while `uploaded` counted both. The same-name exemption
            # above is for a SEPARATE repeat upload; on a case-sensitive filesystem both stay allowed.
            if any(n.casefold() == fname_cf for n in names) and _dataset_folder_is_case_insensitive(
                folder
            ):
                clash_cf = next(n for n in names if n.casefold() == fname_cf)
                raise HTTPException(
                    status_code = 400,
                    detail = (
                        f"Duplicate file '{filename}' differs from '{clash_cf}' only by letter "
                        "case, so on this filesystem they are one file and would overwrite each "
                        "other. Rename one before uploading."
                    ),
                )
            names.append(filename)
        # Stage each file to a temp name and move it in only once the whole batch is written, so a mid-batch failure leaves the dataset untouched, including any same-name file a direct write would truncate.
        staged: list[tuple[Path, Path]] = []  # (temp, final)
        committed = False
        try:
            for f, filename in zip(files, names):
                dest = folder / filename
                # A filename-independent temp name so a long (but valid) filename cannot overflow NAME_MAX with the staging suffix.
                tmp = folder / f".upload-{_uuid.uuid4().hex}.part"
                staged.append((tmp, dest))
                with open(tmp, "wb") as out:
                    while chunk := await f.read(1024 * 1024):
                        total_bytes += len(chunk)
                        if total_bytes > limit_bytes:
                            raise HTTPException(
                                status_code = 413,
                                detail = (
                                    "Dataset upload too large. "
                                    f"Maximum is {get_upload_limit_label()} per upload; "
                                    "add the remaining images in another batch."
                                ),
                            )
                        out.write(chunk)
                # Reject a decompression bomb before commit: a small PNG can pass the byte limit yet decode to huge pixels and OOM the trainer's latent cache.
                if Path(filename).suffix.lower() in _DIFFUSION_DATASET_IMAGE_EXTS:
                    _validate_uploaded_training_image(tmp, filename)
                uploaded += 1
            # Re-check the interlock immediately before the commit: the entry guard only saw the
            # pre-upload state, so a /diffusion/start could have reserved the slot while we streamed.
            _require_diffusion_dataset_mutable()
            # Commit every staged file as one transaction: a plain replace loop is not atomic. Back up
            # each pre-existing destination first and restore them all on any failure.
            backups: list[tuple[Path, Optional[Path]]] = []  # (dest, backup path or None)
            installed: list[Path] = []
            try:
                for tmp, dest in staged:
                    backup: Optional[Path] = None
                    if dest.exists():
                        backup = folder / f".upload-backup-{_uuid.uuid4().hex}.part"
                        dest.replace(backup)
                    backups.append((dest, backup))
                    tmp.replace(dest)  # atomic on the same filesystem
                    installed.append(dest)
                committed = True
            except BaseException:
                # Roll back: drop every new version, then restore every displaced original.
                for dest in reversed(installed):
                    try:
                        dest.unlink(missing_ok = True)
                    except OSError:
                        pass
                for dest, backup in reversed(backups):
                    if backup is not None and backup.exists():
                        try:
                            backup.replace(dest)
                        except OSError:
                            pass
                raise
            else:
                for _, backup in backups:
                    if backup is not None:
                        try:
                            backup.unlink(missing_ok = True)
                        except OSError:
                            pass
        finally:
            if not committed:
                for tmp, _ in staged:
                    try:
                        tmp.unlink(missing_ok = True)
                    except OSError:
                        pass

        summary = _diffusion_dataset_summary(folder)
        return DiffusionDatasetUploadResponse(
            name = cleaned,
            path = str(folder),
            image_count = summary.image_count,
            caption_count = summary.caption_count,
            uploaded = uploaded,
        )
    finally:
        _lock.release()


# ── Dataset labeling (per-image caption editing) + one-click example imports ──
# Thumbnails live in a hidden subdir so they never appear in dataset listings or the trainer's image discovery.
_THUMBS_DIRNAME = ".thumbs"
_MAX_CAPTION_CHARS = 2000


def _resolve_dataset_folder(name: str, *, must_exist: bool = True) -> Path:
    """Validate ``name`` (single component, no traversal) and resolve it under the Studio
    datasets root. 404 when a read target is missing."""
    from utils.paths import datasets_root

    cleaned = _clean_diffusion_dataset_name(name)
    root = datasets_root().resolve()
    folder = root / cleaned
    # Reject a symlinked dataset directory and prove the resolved folder stays under root: _safe_dataset_image_path only checks each image path, not the folder.
    if folder.is_symlink():
        raise HTTPException(
            status_code = 400,
            detail = f"Dataset '{cleaned}' must not be a symbolic link.",
        )
    if must_exist and not folder.is_dir():
        raise HTTPException(status_code = 404, detail = f"Dataset '{cleaned}' not found.")
    try:
        folder.resolve(strict = must_exist).relative_to(root)
    except (OSError, ValueError):
        raise HTTPException(
            status_code = 400,
            detail = f"Dataset '{cleaned}' escapes the Studio datasets directory.",
        )
    return folder


# Per-side dimension bound for uploaded training images, matching diffusion._decode_b64_image's 4096px guard, so a compressible PNG cannot smuggle huge pixels past the byte limit.
_MAX_TRAINING_IMAGE_SIDE = 4096


def _validate_uploaded_training_image(path: Path, original_name: str) -> None:
    """Reject an uploaded training image whose decoded dimensions exceed the per-side limit.

    Reads only the header (never img.load()), so a small-payload / huge-dimension file is caught
    before it spikes memory. Bytes PIL cannot identify are left as-is (the upload contract accepts
    arbitrary bytes under an image extension), so only oversized real images change behaviour."""
    from PIL import Image, UnidentifiedImageError

    try:
        with Image.open(path) as image:
            width, height = image.size
    except Image.DecompressionBombError:
        # Past Pillow's ~179 MP limit Image.open() raises before .size can be read, with an error deriving straight from Exception, so letting it escape would 500 the upload.
        raise HTTPException(
            status_code = 400,
            detail = (
                f"Image '{original_name}' is too large; maximum is "
                f"{_MAX_TRAINING_IMAGE_SIDE}px per side."
            ),
        )
    except (OSError, UnidentifiedImageError, ValueError):
        return  # not a decodable image -> not a bomb; leave the existing contract
    if width > _MAX_TRAINING_IMAGE_SIDE or height > _MAX_TRAINING_IMAGE_SIDE:
        raise HTTPException(
            status_code = 400,
            detail = (
                f"Image '{original_name}' is too large ({width}x{height}); maximum is "
                f"{_MAX_TRAINING_IMAGE_SIDE}px per side."
            ),
        )


def _safe_dataset_image_path(folder: Path, filename: str) -> Path:
    """Resolve ``filename`` to an image path strictly inside ``folder``. Rejects any path
    separators / traversal / null bytes and non-image extensions."""
    raw = filename or ""
    if "/" in raw or "\\" in raw or ".." in raw or "\x00" in raw or raw != Path(raw).name:
        raise HTTPException(status_code = 400, detail = "Invalid image filename.")
    if Path(raw).suffix.lower() not in _DIFFUSION_DATASET_IMAGE_EXTS:
        exts = ", ".join(sorted(_DIFFUSION_DATASET_IMAGE_EXTS))
        raise HTTPException(status_code = 400, detail = f"Not an image file. Allowed: {exts}")
    path = folder / raw
    # Defense in depth: the real path must stay under the dataset folder.
    try:
        path.resolve().relative_to(folder.resolve())
    except ValueError:
        raise HTTPException(status_code = 400, detail = "Invalid image filename.")
    return path


def _load_metadata_captions(folder: Path) -> dict[str, str]:
    """Read metadata.jsonl / captions.jsonl into {file_name: caption}, mirroring the
    trainer's discovery (keys file_name/image/file; caption in the ``text`` column)."""
    import json

    out: dict[str, str] = {}
    for meta_name in ("metadata.jsonl", "captions.jsonl"):
        meta_path = folder / meta_name
        if not meta_path.is_file():
            continue
        # Tolerate a bad upload (invalid UTF-8, or non-object JSON): skip the record so these endpoints do not 500.
        try:
            lines = meta_path.read_text(encoding = "utf-8").splitlines()
        except (OSError, UnicodeError):
            continue
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except (json.JSONDecodeError, TypeError):
                continue
            if not isinstance(row, dict):
                continue
            key = row.get("file_name") or row.get("image") or row.get("file")
            value = row.get("text")
            # A JSON null is "no caption", not the string "None".
            if key and value is not None:
                out[str(key)] = str(value)
    return out


def _image_record(
    folder: Path, image_path: Path, meta_captions: dict[str, str]
) -> DiffusionDatasetImageRecord:
    """Build one image record, resolving its caption with sidecar > metadata precedence
    (the same order the trainer uses). A per-image .txt / .caption sidecar wins because
    it is the user's explicit edit from the labeling grid, which must override a
    metadata.jsonl / captions.jsonl row for the image."""
    caption: Optional[str] = None
    source = "none"
    sidecar_present = False
    for ext in (".txt", ".caption"):
        sidecar = image_path.with_suffix(ext)
        if sidecar.is_file():
            sidecar_present = True
            try:
                caption = sidecar.read_text(encoding = "utf-8").strip()
                source = "sidecar"
            except (OSError, UnicodeError):
                # Unreadable / invalid UTF-8 sidecar: UnicodeDecodeError is a ValueError, so an OSError-only
                # guard 500'd the labeling grid. The trainer treats any existing sidecar as a tombstone.
                caption = None
                source = "sidecar"
            break
    if caption is None and not sidecar_present:
        # Basename first, then the relative path as written in the jsonl (as_posix so Windows paths match): discover_image_caption_pairs order.
        meta = meta_captions.get(image_path.name)
        if meta is None:
            try:
                meta = meta_captions.get(image_path.relative_to(folder).as_posix())
            except ValueError:
                meta = None
        if meta is not None:
            caption = meta
            source = "metadata"
    try:
        size_bytes = image_path.stat().st_size
    except OSError:
        size_bytes = 0
    width = height = 0
    try:
        from PIL import Image
        with Image.open(image_path) as im:
            width, height = im.size
    except Exception:  # noqa: BLE001 -- an unreadable image still lists (0x0) rather than 500
        pass
    return DiffusionDatasetImageRecord(
        filename = image_path.name,
        caption = caption,
        caption_source = source,  # type: ignore[arg-type]
        width = width,
        height = height,
        size_bytes = size_bytes,
    )


@router.get("/diffusion/dataset/{name}/images", response_model = DiffusionDatasetImagesResponse)
async def list_diffusion_dataset_images(
    name: str, current_subject: str = Depends(get_current_subject)
):
    """List every image in a dataset folder with its resolved caption (including
    uncaptioned images), for the labeling grid."""
    folder = _resolve_dataset_folder(name)

    def scan() -> DiffusionDatasetImagesResponse:
        meta = _load_metadata_captions(folder)
        records: list[DiffusionDatasetImageRecord] = []
        for p in sorted(folder.iterdir()):
            if p.is_file() and p.suffix.lower() in _DIFFUSION_DATASET_IMAGE_EXTS:
                records.append(_image_record(folder, p, meta))
        return DiffusionDatasetImagesResponse(name = folder.name, path = str(folder), images = records)

    return await asyncio.to_thread(scan)


@router.get("/diffusion/dataset/{name}/image/{filename}")
async def get_diffusion_dataset_image(
    name: str,
    filename: str,
    thumb: Optional[int] = None,
    current_subject: str = Depends(get_current_subject),
):
    """Serve a dataset image. ``?thumb=<px>`` returns a cached downscaled JPEG (regenerated
    when the source is newer), used by the labeling grid to stay light."""
    from fastapi.responses import FileResponse

    folder = _resolve_dataset_folder(name)
    image_path = _safe_dataset_image_path(folder, filename)
    if not image_path.is_file():
        raise HTTPException(status_code = 404, detail = "Image not found.")
    if not thumb:
        return FileResponse(str(image_path))

    size = max(32, min(1024, int(thumb)))

    def make_thumb() -> Path:
        from PIL import Image

        thumbs_dir = folder / _THUMBS_DIRNAME
        thumbs_dir.mkdir(exist_ok = True)
        # Key on the full filename, not the stem: two images sharing a stem would collide on one cache file and the mtime-newer entry would be served for both.
        thumb_path = thumbs_dir / f"{image_path.name}_{size}.jpg"
        src_mtime = image_path.stat().st_mtime
        if thumb_path.is_file() and thumb_path.stat().st_mtime >= src_mtime:
            return thumb_path
        with Image.open(image_path) as im:
            im = im.convert("RGB")
            im.thumbnail((size, size), Image.LANCZOS)
            im.save(thumb_path, format = "JPEG", quality = 85)
        return thumb_path

    try:
        thumb_path = await asyncio.to_thread(make_thumb)
    except Exception as e:  # noqa: BLE001 -- fall back to the original on any decode failure
        logger.warning("Thumbnail generation failed for %s: %s", image_path, e)
        return FileResponse(str(image_path))
    return FileResponse(str(thumb_path), media_type = "image/jpeg")


@router.put(
    "/diffusion/dataset/{name}/caption/{filename}",
    response_model = DiffusionDatasetImageRecord,
)
async def set_diffusion_dataset_caption(
    name: str,
    filename: str,
    body: DiffusionCaptionUpdateRequest,
    current_subject: str = Depends(get_current_subject),
    _interlock: None = Depends(diffusion_dataset_interlock),
):
    """Write (or, when blank, clear) an image's ``.txt`` caption sidecar. Returns the
    updated image record."""
    _require_diffusion_dataset_mutable()
    folder = _resolve_dataset_folder(name)
    image_path = _safe_dataset_image_path(folder, filename)
    if not image_path.is_file():
        raise HTTPException(status_code = 404, detail = "Image not found.")
    caption = (body.caption or "").strip()
    if len(caption) > _MAX_CAPTION_CHARS:
        raise HTTPException(
            status_code = 400,
            detail = f"Caption too long (max {_MAX_CAPTION_CHARS} characters).",
        )

    def write() -> DiffusionDatasetImageRecord:
        sidecar = image_path.with_suffix(".txt")
        if caption:
            sidecar.write_text(caption, encoding = "utf-8")
            image_path.with_suffix(".caption").unlink(missing_ok = True)
            return _image_record(folder, image_path, _load_metadata_captions(folder))
        # Blank must actually clear. Unlinking alone would resurface this image's metadata caption,
        # so write an EMPTY sidecar: reader and trainer treat it as an authoritative tombstone.
        meta = _load_metadata_captions(folder)
        try:
            rel = image_path.relative_to(folder).as_posix()
        except ValueError:
            rel = image_path.name
        if image_path.name in meta or rel in meta:
            sidecar.write_text("", encoding = "utf-8")
        else:
            sidecar.unlink(missing_ok = True)
        image_path.with_suffix(".caption").unlink(missing_ok = True)
        return _image_record(folder, image_path, meta)

    return await asyncio.to_thread(write)


@router.delete("/diffusion/dataset/{name}/image/{filename}")
async def delete_diffusion_dataset_image(
    name: str,
    filename: str,
    current_subject: str = Depends(get_current_subject),
    _interlock: None = Depends(diffusion_dataset_interlock),
):
    """Remove an image, its caption sidecars, and any cached thumbnails."""
    _require_diffusion_dataset_mutable()
    folder = _resolve_dataset_folder(name)
    image_path = _safe_dataset_image_path(folder, filename)
    if not image_path.is_file():
        raise HTTPException(status_code = 404, detail = "Image not found.")

    def remove() -> dict:
        import glob as _glob

        image_path.unlink(missing_ok = True)
        # Sidecars are keyed on the STEM, so cat.jpg and cat.png share cat.txt: deleting it with one
        # would strip the survivor's caption. New collisions are refused at upload; legacy ones exist.
        stem_still_used = any(
            p.is_file()
            and p != image_path
            and p.stem == image_path.stem
            and p.suffix.lower() in _DIFFUSION_DATASET_IMAGE_EXTS
            for p in folder.iterdir()
        )
        if not stem_still_used:
            for ext in (".txt", ".caption"):
                image_path.with_suffix(ext).unlink(missing_ok = True)
        thumbs_dir = folder / _THUMBS_DIRNAME
        if thumbs_dir.is_dir():
            # Thumbs are keyed on the full filename, so match that here; a stem-only glob would strand this image's thumbs or delete a sibling's. Escape the name so a glob metacharacter cannot match siblings.
            for t in thumbs_dir.glob(f"{_glob.escape(image_path.name)}_*.jpg"):
                t.unlink(missing_ok = True)
        return {"deleted": image_path.name}

    return await asyncio.to_thread(remove)


# Curated, license-labelled example datasets. ``loader`` picks the materialization strategy:
# "hf_dataset" streams rows; "imagefolder_jsonl" snapshot-downloads a repo with *.jsonl captions.
_DATASET_EXAMPLES: list[dict] = [
    {
        "id": "dreambooth-dog",
        "label": "Dog (DreamBooth subject)",
        "repo": "diffusers/dog-example",
        "description": "5 photos of one dog. Teach a subject, then call it with the trigger.",
        "license": "Google, research and demos",
        "image_cap": 10,
        "suggested_trigger": "a photo of sks dog",
        "loader": "hf_dataset",
        "caption_column": None,
        "no_checks": False,
    },
    {
        "id": "tuxemon",
        "label": "Tuxemon (captioned style set)",
        "repo": "linoyts/Tuxemon",
        "description": "Captioned cartoon monster art. A style set, no trigger needed.",
        "license": "cc-by-sa-3.0",
        "image_cap": 60,
        "suggested_trigger": None,
        "loader": "hf_dataset",
        "caption_column": "prompt",
        "no_checks": True,
    },
    {
        "id": "tarot-1920",
        "label": "1920 Tarot (public domain style set)",
        "repo": "multimodalart/1920-raider-waite-tarot-public-domain",
        "description": "Captioned 1920 Raider-Waite tarot art. A permissive style set.",
        "license": "public domain",
        "image_cap": 60,
        "suggested_trigger": None,
        "loader": "imagefolder_jsonl",
        "caption_column": "text",
        "no_checks": True,
    },
    {
        "id": "smithsonian-butterflies",
        "label": "Smithsonian Butterflies",
        "repo": "huggan/smithsonian_butterflies_subset",
        "description": "100 butterfly photos. No captions, so use the trigger prompt.",
        "license": "CC0",
        "image_cap": 100,
        # The metadata columns are species names / boilerplate alt-text, not captions, so train it as a subject set with the trigger prompt instead.
        "suggested_trigger": "a photo of a sks butterfly",
        "loader": "hf_dataset",
        "caption_column": None,
        "no_checks": False,
    },
    {
        "id": "pixel-nouns",
        "label": "Nouns (pixel avatars)",
        "repo": "m1guelpf/nouns",
        "description": "100 captioned pixel-art avatars. A style set, no trigger needed.",
        "license": "cc0-1.0",
        "image_cap": 100,
        "suggested_trigger": None,
        "loader": "hf_dataset",
        "caption_column": "text",
        "no_checks": False,
    },
]


def _example_by_id(example_id: str) -> dict:
    for entry in _DATASET_EXAMPLES:
        if entry["id"] == example_id:
            return entry
    raise HTTPException(status_code = 404, detail = f"Unknown example dataset '{example_id}'.")


@router.get("/diffusion/dataset-examples", response_model = DiffusionDatasetExamplesResponse)
async def list_diffusion_dataset_examples(current_subject: str = Depends(get_current_subject)):
    """List the curated example datasets available for one-click import."""
    return DiffusionDatasetExamplesResponse(
        examples = [
            DiffusionDatasetExample(
                id = e["id"],
                label = e["label"],
                repo = e["repo"],
                description = e["description"],
                license = e["license"],
                image_cap = e["image_cap"],
                suggested_trigger = e["suggested_trigger"],
            )
            for e in _DATASET_EXAMPLES
        ]
    )


def _detect_image_column(features) -> Optional[str]:
    """Return the first datasets Image-feature column name, else None."""
    try:
        from datasets import Image as HFImage
    except Exception:  # noqa: BLE001
        HFImage = None  # type: ignore[assignment]
    for col, feat in features.items():
        if HFImage is not None and isinstance(feat, HFImage):
            return col
        if type(feat).__name__ == "Image":
            return col
    return None


def _detect_image_column_from_row(row: dict) -> Optional[str]:
    """Image column picked from one materialized row, for a streamed dataset that arrives with no
    feature metadata to inspect."""
    try:
        from PIL.Image import Image as PILImage
    except Exception:  # noqa: BLE001 -- no Pillow -> the caller reports "no image column"
        return None
    for col, value in row.items():
        if isinstance(value, PILImage):
            return col
    return None


def _detect_caption_column(entry: dict, columns: list[str]) -> Optional[str]:
    """Pick the caption column: the entry's declared one if present, else a common name."""
    declared = entry.get("caption_column")
    if declared and declared in columns:
        return declared
    for cand in ("text", "prompt", "caption", "captions"):
        if cand in columns:
            return cand
    return None


def _materialize_hf_dataset(entry: dict, dest: Path, cap: int) -> int:
    """Stream rows from datasets.load_dataset into ``dest`` as numbered images + optional
    .txt sidecars. Returns the number of images written."""
    from datasets import load_dataset

    kwargs = {"split": "train"}
    if entry.get("no_checks"):
        kwargs["verification_mode"] = "no_checks"
    # Stream rather than prepare the whole split: the loop keeps at most `cap` rows while these
    # curated repos run to tens of thousands. A repo that cannot stream falls back to prepared.
    try:
        ds = load_dataset(entry["repo"], streaming = True, **kwargs)
        features = ds.features
    except Exception:  # noqa: BLE001 -- not streamable; the prepared load is the fallback
        ds = load_dataset(entry["repo"], **kwargs)
        features = ds.features
    # Streaming can hand back a dataset whose features are only known once a row is read, so resolve the columns from the first row then.
    image_col = _detect_image_column(features) if features else None
    if image_col is None and features:
        raise HTTPException(
            status_code = 502,
            detail = f"'{entry['repo']}' has no image column to import.",
        )
    caption_col = _detect_caption_column(entry, list(features.keys())) if features else None
    written = 0
    for row in ds:
        if written >= cap:
            break
        if image_col is None:
            image_col = _detect_image_column_from_row(row)
            if image_col is None:
                raise HTTPException(
                    status_code = 502,
                    detail = f"'{entry['repo']}' has no image column to import.",
                )
            caption_col = _detect_caption_column(entry, list(row.keys()))
        img = row[image_col]
        if img is None:
            continue
        img = img.convert("RGB")
        stem = f"img_{written:04d}"
        img.save(dest / f"{stem}.png", format = "PNG")
        if caption_col:
            cap_text = row.get(caption_col)
            if cap_text:
                (dest / f"{stem}.txt").write_text(str(cap_text).strip(), encoding = "utf-8")
        written += 1
    return written


def _materialize_imagefolder_jsonl(entry: dict, dest: Path, cap: int) -> int:
    """Snapshot-download a dataset repo whose captions live in *.jsonl (file_name/text),
    then copy referenced images + write .txt sidecars. Returns images written."""
    import json
    import shutil

    from huggingface_hub import snapshot_download

    caption_col = entry.get("caption_column") or "text"
    snap = Path(
        snapshot_download(
            entry["repo"],
            repo_type = "dataset",
            allow_patterns = [
                "*.jsonl",
                "*.jpg",
                "*.jpeg",
                "*.png",
                "*.webp",
                "*.bmp",
                "**/*.jpg",
                "**/*.jpeg",
                "**/*.png",
                "**/*.webp",
                "**/*.bmp",
            ],
        )
    )
    # Map basename -> caption from every jsonl carrying file_name + caption column.
    captions: dict[str, str] = {}
    for jf in sorted(snap.rglob("*.jsonl")):
        for line in jf.read_text(encoding = "utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            fn = row.get("file_name") or row.get("image") or row.get("file")
            value = row.get(caption_col)
            # A JSON null is "no caption", not the string "None".
            if fn and value is not None:
                # First writer wins over sorted manifests, for deterministic results.
                captions.setdefault(Path(str(fn)).name, str(value))
    # Copy images (those with a caption first, so a cap keeps captioned pairs).
    images = sorted(
        p
        for p in snap.rglob("*")
        if p.is_file() and p.suffix.lower() in _DIFFUSION_DATASET_IMAGE_EXTS
    )
    images.sort(key = lambda p: (p.name not in captions, p.name))
    written = 0
    for src in images:
        if written >= cap:
            break
        stem = f"img_{written:04d}"
        shutil.copyfile(src, dest / f"{stem}{src.suffix.lower()}")
        cap_text = captions.get(src.name)
        if cap_text:
            (dest / f"{stem}.txt").write_text(cap_text.strip(), encoding = "utf-8")
        written += 1
    return written


@router.post("/diffusion/dataset/import-example", response_model = DiffusionDatasetImportResponse)
async def import_diffusion_dataset_example(
    body: DiffusionDatasetImportRequest,
    current_subject: str = Depends(get_current_subject),
    _interlock: None = Depends(diffusion_dataset_interlock),
):
    """Materialize a curated example dataset into a Studio dataset folder (images + .txt
    captions), ready to train. Idempotent: a folder that already holds images is returned
    as-is rather than re-downloaded."""
    _require_diffusion_dataset_mutable()
    entry = _example_by_id(body.id)
    folder = _resolve_dataset_folder(body.name or entry["id"], must_exist = False)

    def do_import() -> DiffusionDatasetImportResponse:
        folder.mkdir(parents = True, exist_ok = True)
        if _diffusion_dataset_summary(folder).image_count > 0:
            return _import_response(entry, folder, imported = 0)
        # One import at a time per dataset folder: the training interlock COUNTS mutations rather than excluding them, so two
        # imports into the same empty name both passed the emptiness check and merged. Refusing the second is honest.
        lock = _dataset_import_lock(folder)
        if not lock.acquire(blocking = False):
            raise HTTPException(
                status_code = 409,
                detail = (
                    f"An import into '{folder.name}' is already running. Wait for it to finish, "
                    "then reload the dataset list."
                ),
            )
        try:
            return _do_import_locked(entry, folder)
        finally:
            lock.release()

    def _do_import_locked(entry: dict, folder: Path) -> DiffusionDatasetImportResponse:
        import os
        import shutil
        import tempfile

        imported = 0
        # Re-read under the lock: a winner may have promoted its staging dir while this request was checking, so returning the folder as-is matches the idempotent path.
        existing = _diffusion_dataset_summary(folder)
        if existing.image_count == 0:
            cap = int(entry["image_cap"])
            # Materialize into a private staging dir and promote only after the whole import succeeds, so a partial materialize
            # leaves only that dir. Staged as a hidden same-filesystem sibling so promotion is an atomic rename.
            staging = Path(tempfile.mkdtemp(dir = folder.parent, prefix = f".{folder.name}.import-"))
            # Superseded same-name entries are parked here rather than deleted, so a failed promotion can put them back too.
            rescue = Path(tempfile.mkdtemp(dir = folder.parent, prefix = f".{folder.name}.rescue-"))
            # (entry's new home, where it came from) for every pre-existing entry moved out of the folder.
            folded: list[tuple[Path, Path]] = []

            def restore_folded() -> None:
                """Undo the fold-in so a failed promotion leaves the dataset as it was."""
                folder.mkdir(parents = True, exist_ok = True)
                for moved, original in folded:
                    try:
                        if moved.exists() and not original.exists():
                            shutil.move(str(moved), str(original))
                    except OSError:
                        # Best effort: one unrestorable entry must not mask the original failure.
                        pass

            try:
                try:
                    if entry["loader"] == "imagefolder_jsonl":
                        imported = _materialize_imagefolder_jsonl(entry, staging, cap)
                    else:
                        imported = _materialize_hf_dataset(entry, staging, cap)
                except HTTPException:
                    raise
                except Exception as e:  # noqa: BLE001 -- surface a readable fetch/parse failure
                    raise HTTPException(
                        status_code = 502,
                        detail = f"Could not import '{entry['repo']}': {e}",
                    )
                if imported == 0:
                    raise HTTPException(
                        status_code = 502,
                        detail = f"No images found in '{entry['repo']}'.",
                    )
                # Promote the fully-materialized staging dir as a UNIT: a same-filesystem rename is atomic, so a hard process death
                # leaves either the old folder or the finished import. rmdir needs an empty target, so fold any pre-existing files (a .thumbs cache, an older metadata.jsonl) INTO staging first and keep one atomic promotion.
                try:
                    for p in sorted(folder.iterdir()):
                        # Same name in both: the imported file wins, as the previous per-file move did. Park the old one in the rescue dir so the folder can be emptied for the rename without destroying it.
                        dest = (rescue if (staging / p.name).exists() else staging) / p.name
                        shutil.move(str(p), str(dest))
                        folded.append((dest, p))
                    os.rmdir(folder)
                    os.replace(str(staging), str(folder))
                except (OSError, shutil.Error) as e:
                    # Every step here can fail (an unmovable entry, a folder that gained a file, a rename held by antivirus), and by then the folder's entries live only in the staging/rescue dirs the finally deletes.
                    restore_folded()
                    raise HTTPException(
                        status_code = 409,
                        detail = (
                            f"Could not update '{folder.name}' with the imported example "
                            f"({getattr(e, 'strerror', None) or e}). Nothing was written; try again."
                        ),
                    )
            finally:
                shutil.rmtree(staging, ignore_errors = True)
                shutil.rmtree(rescue, ignore_errors = True)
        return _import_response(entry, folder, imported = imported)

    return await asyncio.to_thread(do_import)
