# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Training backend — subprocess orchestrator.

Each job runs in a fresh spawn subprocess (solving transformers version-switching);
the in-process UnslothTrainer singleton is only used inside the worker. This file
orchestrates the subprocess lifecycle, pumps events from the worker's mp.Queue, and
exposes the same API to routes/training.py. Pattern follows data_recipe/jobs/manager.py.
"""

import json as _json
import math
import multiprocessing as mp
import os
import platform
import queue
import re
import shutil
import threading
import time
import traceback
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
from loggers import get_logger
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Optional, Tuple, Any, Callable, Union, TYPE_CHECKING, Literal, Iterator

if TYPE_CHECKING:
    import matplotlib.pyplot as plt
from utils.hardware import get_device, prepare_gpu_selection
from utils.native_path_leases import (
    native_path_secret_removed_for_child_start,
    run_without_native_path_secret,
)
from utils.paths import is_local_path, outputs_root
from utils.utils import canonical_model_repo_id

logger = get_logger(__name__)


def _env_int(name: str, default: int) -> int:
    try:
        raw = (os.environ.get(name) or "").strip()
        return int(raw) if raw else default
    except ValueError:
        return default


# Primary trigger is a short grace once "complete" (save done); the absolute cap is a backstop, long
# for save=True and shorter for a cancel.
_STOP_GRACE_S = _env_int("UNSLOTH_STUDIO_TRAINING_STOP_GRACE_S", 15)
_STOP_TIMEOUT_S = _env_int("UNSLOTH_STUDIO_TRAINING_STOP_TIMEOUT_S", 600)
_CANCEL_TIMEOUT_S = _env_int("UNSLOTH_STUDIO_TRAINING_CANCEL_TIMEOUT_S", 120)
# Generous: is_run_finished already unwedges the UI, and a post-run wandb sync can legitimately take a while.
_COMPLETE_EXIT_GRACE_S = _env_int("UNSLOTH_STUDIO_TRAINING_COMPLETE_EXIT_GRACE_S", 120)

# A few short retries so a transient SQLite lock doesn't lose the terminal state.
_DB_FINALIZE_RETRIES = 3
_DB_FINALIZE_RETRY_S = 0.5
_MAX_TRACKED_START_REQUESTS = 64
_MAX_START_CANCEL_TOMBSTONES = 1024
_START_CANCEL_TOMBSTONE_TTL_S = 300.0
_START_CANCELLED_ERROR_CODE = "training_start_cancelled"

_pyplot = None
_pyplot_failed = False


@dataclass(frozen = True)
class TrainingStartRequestRecord:
    start_request_id: str
    job_id: str
    state: Literal["pending", "accepted", "rejected"]
    message: str
    error: Optional[str] = None
    error_code: Optional[str] = None


class TrainingStartCancellationCapacityError(RuntimeError):
    pass


@dataclass(frozen = True)
class TrainingStatusIdentitySnapshot:
    current_job_id: str
    current_start_request_id: Optional[str]
    current_start_request: Optional[TrainingStartRequestRecord]
    status_start_request: Optional[TrainingStartRequestRecord]
    new_job_spawn_id: Optional[str]
    spawn_in_progress: bool


def _load_pyplot():
    """Lazily import matplotlib.pyplot (headless Agg); return it, or None if
    matplotlib is unavailable. Deferred so a blocked native wheel (e.g. Windows
    Smart App Control) never breaks server startup, only loss plotting.
    """
    global _pyplot, _pyplot_failed
    if _pyplot is not None or _pyplot_failed:
        return _pyplot
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        _pyplot = plt
    except Exception as e:
        _pyplot_failed = True
        logger.warning("matplotlib unavailable; loss plots disabled", error = str(e))
    return _pyplot


def _coerce_seed(value, default = 3407) -> int:
    """Normalize None / non-int to `default` (transformers.set_seed(None) raises)."""
    if value is None:
        return int(default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _coerce_optional_bool(value, default: bool) -> bool:
    """Treat explicit None as `default` instead of `bool(None) == False`."""
    if value is None:
        return bool(default)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ("true", "1", "yes", "on"):
            return True
        if normalized in ("false", "0", "no", "off", ""):
            return False
    return bool(value)


def _coerce_optional_nonneg_float(name: str, value):
    """Reject negatives and non-finite; `ge=0` misses raw callers, and inf never binds."""
    if value is None:
        return None
    try:
        coerced = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"Unsloth: {name}={value!r} must be a non-negative float or None.")
    if coerced < 0 or not math.isfinite(coerced):
        raise ValueError(f"Unsloth: {name}={coerced} must be finite and >= 0.")
    return coerced


def is_apple_silicon_training_platform() -> bool:
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def is_mlx_training_device(device: Any) -> bool:
    return (
        str(device).lower() == "mlx"
        or str(device).lower().endswith(".mlx")
        or getattr(device, "name", "").lower() == "mlx"
    )


def should_use_mlx_training_backend(*, device: Optional[Any] = None) -> bool:
    if device is not None:
        return is_mlx_training_device(device)
    return is_apple_silicon_training_platform()


def _build_training_worker_config(values: dict[str, Any]) -> dict[str, Any]:
    """Build the normalized worker config shared by Unsloth and the CLI adapter."""
    config = {
        "model_name": values["model_name"],
        "project_name": values.get("project_name"),
        "training_type": values.get("training_type", "LoRA/QLoRA"),
        "hf_token": values.get("hf_token", ""),
        "load_in_4bit": values.get("load_in_4bit", True),
        "max_seq_length": values.get("max_seq_length", 2048),
        "vision_image_size": values.get("vision_image_size"),
        "hf_dataset": values.get("hf_dataset", ""),
        "model_known_cached": values.get("model_known_cached", False),
        "model_local_path": values.get("model_local_path"),
        "model_format": values.get("model_format"),
        "model_snapshot_path": values.get("model_snapshot_path"),
        "model_revision": values.get("model_revision"),
        "actual_model_repo_id": values.get("actual_model_repo_id"),
        "resume_model_load_mode": values.get("resume_model_load_mode"),
        "dataset_known_cached": values.get("dataset_known_cached", False),
        "dataset_local_path": values.get("dataset_local_path"),
        "dataset_snapshot_path": values.get("dataset_snapshot_path"),
        "dataset_revision": values.get("dataset_revision"),
        "local_datasets": values.get("local_datasets"),
        "local_eval_datasets": values.get("local_eval_datasets"),
        "format_type": values.get("format_type", ""),
        "subset": values.get("subset"),
        "train_split": values.get("train_split", "train"),
        "eval_split": values.get("eval_split"),
        "eval_steps": values.get("eval_steps", 0.00),
        "dataset_streaming": values.get("dataset_streaming", False),
        "dataset_slice_start": values.get("dataset_slice_start"),
        "dataset_slice_end": values.get("dataset_slice_end"),
        "custom_format_mapping": values.get("custom_format_mapping"),
        "is_dataset_image": values.get("is_dataset_image", False),
        "is_dataset_audio": values.get("is_dataset_audio", False),
        "is_embedding": values.get("is_embedding", False),
        "num_epochs": values.get("num_epochs", 3),
        "learning_rate": values.get("learning_rate", "2e-4"),
        "embedding_learning_rate": values.get("embedding_learning_rate"),
        "batch_size": values.get("batch_size", 2),
        "gradient_accumulation_steps": values.get("gradient_accumulation_steps", 4),
        "warmup_steps": values.get("warmup_steps"),
        "warmup_ratio": values.get("warmup_ratio"),
        "max_steps": values.get("max_steps", 0),
        "save_steps": values.get("save_steps", 0),
        "weight_decay": values.get("weight_decay", 0.001),
        "max_grad_norm": _coerce_optional_nonneg_float(
            "max_grad_norm", values.get("max_grad_norm")
        ),
        "max_grad_value": _coerce_optional_nonneg_float(
            "max_grad_value", values.get("max_grad_value")
        ),
        "max_grad_leaf_norm": _coerce_optional_nonneg_float(
            "max_grad_leaf_norm", values.get("max_grad_leaf_norm")
        ),
        "cast_norm_output_to_input_dtype": _coerce_optional_bool(
            values.get("cast_norm_output_to_input_dtype"), True
        ),
        "random_seed": _coerce_seed(values.get("random_seed")),
        "packing": values.get("packing", False),
        "optim": values.get("optim", "adamw_8bit"),
        "lr_scheduler_type": values.get("lr_scheduler_type", "linear"),
        "use_lora": values.get("use_lora", True),
        "lora_r": values.get("lora_r", 16),
        "lora_alpha": values.get("lora_alpha", 16),
        "lora_dropout": values.get("lora_dropout", 0.0),
        "target_modules": values.get("target_modules"),
        "gradient_checkpointing": values.get("gradient_checkpointing", "unsloth"),
        "use_rslora": values.get("use_rslora", False),
        "use_loftq": values.get("use_loftq", False),
        "use_dora": values.get("use_dora", False),
        "train_on_completions": values.get("train_on_completions", False),
        "finetune_vision_layers": values.get("finetune_vision_layers", True),
        "finetune_language_layers": values.get("finetune_language_layers", True),
        "finetune_attention_modules": values.get("finetune_attention_modules", True),
        "finetune_mlp_modules": values.get("finetune_mlp_modules", True),
        "enable_wandb": values.get("enable_wandb", False),
        "wandb_token": values.get("wandb_token"),
        "wandb_project": values.get("wandb_project", "unsloth-training"),
        "enable_tensorboard": values.get("enable_tensorboard", False),
        "tensorboard_dir": values.get("tensorboard_dir", "runs"),
        "resume_from_checkpoint": values.get("resume_from_checkpoint"),
        "require_exact_resume_resources": values.get("require_exact_resume_resources", False),
        "require_exact_model_resource": values.get("require_exact_model_resource", False),
        "require_exact_dataset_resource": values.get("require_exact_dataset_resource", False),
        "require_validated_model_snapshot": values.get("require_validated_model_snapshot", False),
        "trust_remote_code": values.get("trust_remote_code", False),
        "approved_remote_code_fingerprint": values.get("approved_remote_code_fingerprint"),
        "subject": values.get("subject"),
        "gpu_ids": values.get("gpu_ids"),
        "s3_config": values.get("s3_config"),
        "disable_xet": values.get("disable_xet", False),
    }
    for key in ("output_dir", "allow_external_output_dir"):
        if key in values:
            config[key] = values.get(key)
    if config["training_type"] == "Full Finetuning":
        config["load_in_4bit"] = False
    # The parent's detected backend: the worker's apply_gpu_ids() uses it without probing torch.
    config["device_backend"] = get_device().value
    return config


_HF_TMP_CHECKPOINT_RE = re.compile(r"^tmp-checkpoint-\d+$")


def _sanitize_db_config(config: dict[str, Any]) -> dict[str, Any]:
    # ``subject`` is worker-only metadata; never persist it to config_json, which run-history returns.
    db_config = {
        k: v
        for k, v in config.items()
        if k
        not in {
            "hf_token",
            "wandb_token",
            "s3_config",
            "subject",
            "cache_pin_warnings",
            "require_exact_resume_resources",
            "require_exact_model_resource",
            "require_exact_dataset_resource",
            "require_validated_model_snapshot",
            "resume_model_load_mode",
        }
    }
    s3_config = config.get("s3_config")
    if hasattr(s3_config, "model_dump"):
        s3_config = s3_config.model_dump()
    if isinstance(s3_config, dict) and s3_config:
        db_config["dataset_source"] = "s3"
        db_config["s3_dataset"] = {
            "bucket": s3_config.get("bucket"),
            "region": s3_config.get("region"),
            "prefix": s3_config.get("prefix"),
            "use_iam_role": bool(s3_config.get("use_iam_role")),
        }
    return db_config


_MODEL_SNAPSHOT_METADATA = ("config.json", "adapter_config.json")
# refs/main can point at a revision that only ever fetched metadata, so prefer a snapshot that
# carries weights. Keep in step with _MODEL_WEIGHT_CANDIDATES in routes/training.py: selecting a
# snapshot the start route rejects reproduces the 400.
_MODEL_SNAPSHOT_WEIGHTS = (
    "model.safetensors",
    "model.safetensors.index.json",
    "pytorch_model.bin",
    "pytorch_model.bin.index.json",
    "adapter_model.safetensors",
    "adapter_model.bin",
)


def _with_load_subdirs(model_name: str, names: tuple[str, ...]) -> tuple[str, ...]:
    """Extend snapshot filenames with the subdirectories a load actually reads.

    Spark-TTS / BiCodec keep the trainable model under ``<snapshot>/LLM``, so such a
    snapshot carries no root-level ``config.json`` and no root-level weights. The remote
    preflight already expands those load roots via ``load_scan_target``; mirroring it
    here keeps the cached path in agreement, otherwise a perfectly good cache resolves
    to None and the start route rejects it as hf_model_not_cached_offline.
    """
    from hub.utils.hf_cache_state import with_load_subdirs
    return with_load_subdirs(model_name, names)


def _resolve_model_snapshot(model_name: str, local_path: Optional[str]) -> Optional[str]:
    from hub.utils.hf_cache_state import (
        iter_repo_cache_dirs,
        latest_snapshot_from_cache_path,
    )

    repo_id = canonical_model_repo_id(model_name)
    metadata_names = _with_load_subdirs(model_name, _MODEL_SNAPSHOT_METADATA)
    weight_names = _with_load_subdirs(model_name, _MODEL_SNAPSHOT_WEIGHTS)
    # Pass 1 demands metadata AND weights, so neither a metadata-only refs/main nor a weights-only fetch
    # displaces a complete sibling; pass 2 keeps the old metadata-only path.
    passes: tuple[dict[str, Any], ...] = (
        {"required_groups": (metadata_names, weight_names)},
        {"metadata_filenames": metadata_names},
    )
    if local_path:
        for kwargs in passes:
            snapshot = latest_snapshot_from_cache_path(local_path, "model", repo_id, **kwargs)
            if snapshot:
                return snapshot
        return None
    for kwargs in passes:
        for repo_dir in iter_repo_cache_dirs("model", repo_id):
            snapshot = latest_snapshot_from_cache_path(
                str(repo_dir),
                "model",
                repo_id,
                **kwargs,
            )
            if snapshot:
                return snapshot
    return None


def _apply_model_cache_pin(config: dict[str, Any], warnings: list[str]) -> None:
    resume = bool(config.get("resume_from_checkpoint"))
    model_name = config["model_name"]
    if is_local_path(model_name):
        config["actual_model_repo_id"] = None
        config["model_snapshot_path"] = None
        config["model_revision"] = None
        return
    requested_pin = config.get("model_snapshot_path")
    require_validated_snapshot = bool(config.get("require_validated_model_snapshot"))
    if require_validated_snapshot and not (requested_pin and config.get("actual_model_repo_id")):
        from .provenance import ExactResumeResourcesUnavailable
        raise ExactResumeResourcesUnavailable(
            "The cached model snapshot selected during preflight is no longer available."
        )
    model_claimed = bool(config.get("model_known_cached") or config.get("model_local_path"))
    if resume and requested_pin:
        from hub.utils.hf_cache_state import latest_snapshot_from_cache_path

        pinned_repo_id = config.get("actual_model_repo_id") or canonical_model_repo_id(model_name)
        pin = latest_snapshot_from_cache_path(
            requested_pin,
            "model",
            pinned_repo_id,
            _with_load_subdirs(model_name, _MODEL_SNAPSHOT_METADATA),
        )
        if pin is None:
            if config.get("require_exact_resume_resources") or config.get(
                "require_exact_model_resource"
            ):
                from .provenance import ExactResumeResourcesUnavailable
                raise ExactResumeResourcesUnavailable(
                    "The exact model snapshot for this run is no longer available."
                )
            warnings.append(
                f"The cached model snapshot this run was trained from is no longer on "
                f"disk; resuming by downloading {model_name} from Hugging Face — base "
                f"weights may differ from the original run."
            )
        config["model_snapshot_path"] = pin
        if pin is None:
            config["actual_model_repo_id"] = None
            config["model_revision"] = None
        else:
            config["actual_model_repo_id"] = pinned_repo_id
            config["model_revision"] = Path(pin).name
    elif requested_pin and config.get("actual_model_repo_id"):
        from hub.utils.hf_cache_state import latest_snapshot_from_cache_path

        pinned_repo_id = config["actual_model_repo_id"]
        pin = latest_snapshot_from_cache_path(
            requested_pin,
            "model",
            pinned_repo_id,
            _with_load_subdirs(model_name, _MODEL_SNAPSHOT_METADATA),
        )
        config["model_snapshot_path"] = pin
        if pin is None:
            if require_validated_snapshot:
                from .provenance import ExactResumeResourcesUnavailable
                raise ExactResumeResourcesUnavailable(
                    "The cached model snapshot selected during preflight is no longer available."
                )
            config["actual_model_repo_id"] = None
            config["model_revision"] = None
        else:
            config["model_revision"] = Path(pin).name
    elif model_claimed:
        pinned_repo_id = canonical_model_repo_id(model_name)
        pin = _resolve_model_snapshot(model_name, config.get("model_local_path"))
        if pin is None:
            warnings.append(
                f"Cached copy of {model_name} not found on disk; downloading from Hugging Face."
            )
        config["model_snapshot_path"] = pin
        config["actual_model_repo_id"] = pinned_repo_id if pin is not None else None
        config["model_revision"] = Path(pin).name if pin is not None else None
    else:
        config["model_snapshot_path"] = None
        config["actual_model_repo_id"] = None
        config["model_revision"] = None


def resolve_training_model_load_target(values: dict[str, Any]) -> str:
    config = {
        "model_name": values["model_name"],
        "model_known_cached": values.get("model_known_cached", False),
        "model_local_path": values.get("model_local_path"),
        "model_snapshot_path": values.get("model_snapshot_path"),
        "model_revision": values.get("model_revision"),
        "actual_model_repo_id": values.get("actual_model_repo_id"),
        "resume_model_load_mode": values.get("resume_model_load_mode"),
        "resume_from_checkpoint": values.get("resume_from_checkpoint"),
        "require_exact_resume_resources": values.get("require_exact_resume_resources", False),
        "require_exact_model_resource": values.get("require_exact_model_resource", False),
        "require_validated_model_snapshot": values.get("require_validated_model_snapshot", False),
        "load_in_4bit": values.get("load_in_4bit", True),
    }
    _apply_model_cache_pin(config, [])
    return config.get("model_snapshot_path") or config["model_name"]


def _apply_cache_pins(config: dict[str, Any]) -> None:
    warnings: list[str] = []
    resume = bool(config.get("resume_from_checkpoint"))
    if resume:
        from .provenance import (
            validate_exact_dataset_pin,
            validate_exact_model_pin,
            validate_exact_resource_pins,
        )
        if config.get("require_exact_resume_resources"):
            model_snapshot, dataset_snapshot = validate_exact_resource_pins(config)
            config["model_snapshot_path"] = model_snapshot
            config["dataset_snapshot_path"] = dataset_snapshot
        else:
            if config.get("require_exact_model_resource"):
                config["model_snapshot_path"] = validate_exact_model_pin(config)
            if config.get("require_exact_dataset_resource"):
                config["dataset_snapshot_path"] = validate_exact_dataset_pin(config)
    _apply_model_cache_pin(config, warnings)

    hf_dataset = config.get("hf_dataset") or ""
    requested_ds_pin = config.get("dataset_snapshot_path")
    ds_claimed = bool(config.get("dataset_known_cached") or config.get("dataset_local_path"))
    config["dataset_revision"] = None
    if not hf_dataset or config.get("dataset_streaming"):
        config["dataset_snapshot_path"] = None
    elif resume and requested_ds_pin:
        from hub.utils.dataset_cache import (
            dataset_cache_path_from_cache_path,
            dataset_snapshot_from_cache_path,
        )

        snap = dataset_cache_path_from_cache_path(requested_ds_pin, hf_dataset)
        if snap is None:
            if config.get("require_exact_resume_resources") or config.get(
                "require_exact_dataset_resource"
            ):
                from .provenance import ExactResumeResourcesUnavailable
                raise ExactResumeResourcesUnavailable(
                    "The exact dataset snapshot for this run is no longer available."
                )
            warnings.append(
                f"The cached dataset data this run was trained from is no longer on "
                f"disk; resuming by downloading {hf_dataset} from Hugging Face."
            )
        config["dataset_snapshot_path"] = str(snap) if snap else None
        snapshot = (
            dataset_snapshot_from_cache_path(str(snap), hf_dataset) if snap is not None else None
        )
        if snapshot is not None:
            config["dataset_revision"] = snapshot.name
    elif ds_claimed:
        from hub.utils.dataset_cache import training_dataset_cache_pin

        snap, revision = training_dataset_cache_pin(
            hf_dataset,
            config.get("dataset_local_path"),
        )
        config["dataset_revision"] = revision
        if snap is None:
            if revision:
                warnings.append(
                    f"The cached snapshot of dataset {hf_dataset} is incomplete; "
                    f"downloading its exact revision from Hugging Face."
                )
            else:
                warnings.append(
                    f"Cached copy of dataset {hf_dataset} not found on disk; downloading from "
                    f"Hugging Face."
                )
        config["dataset_snapshot_path"] = str(snap) if snap else None
    else:
        config["dataset_snapshot_path"] = None

    config["cache_pin_warnings"] = warnings


def _s3_dataset_name(s3_dataset: Any) -> Optional[str]:
    if not isinstance(s3_dataset, dict):
        return None
    bucket = s3_dataset.get("bucket")
    if not bucket:
        return None
    prefix = s3_dataset.get("prefix")
    return f"s3://{bucket}/{prefix}" if prefix else f"s3://{bucket}"


def _cleanup_cancelled_checkpoints(output_dir: Union[str, os.PathLike]) -> None:
    """Remove only HF Trainer ``tmp-checkpoint-<step>/`` partials after a cancel.

    Completed ``checkpoint-<int>/`` dirs survive. Symlinked output_dir / children
    are skipped so containment can't be bypassed.
    """
    out = Path(output_dir)
    if not out.exists() or not out.is_dir() or out.is_symlink():
        return
    try:
        out_real = out.resolve()
        out_root_real = Path(outputs_root()).resolve()
    except OSError:
        return
    try:
        out_real.relative_to(out_root_real)
    except ValueError:
        logger.warning(
            "Skipping checkpoint cleanup - %s is not under outputs_root %s",
            out_real,
            out_root_real,
        )
        return
    removed = 0
    for entry in out.iterdir():
        if not entry.is_dir() or entry.is_symlink():
            continue
        if not _HF_TMP_CHECKPOINT_RE.match(entry.name):
            continue
        try:
            shutil.rmtree(entry, ignore_errors = False)
            removed += 1
        except OSError as exc:
            logger.warning("Could not remove %s: %s", entry, exc)
    logger.info(
        "Cancelled-run cleanup removed %d in-flight tmp-checkpoint dir(s) under %s",
        removed,
        out,
    )


_CTX = mp.get_context("spawn")

PLOT_WIDTH = 8
PLOT_HEIGHT = 3.5


@dataclass
class TrainingProgress:
    """Shared training progress payload for Unsloth and backend-aware trainers."""

    epoch: float = 0
    step: int = 0
    total_steps: int = 0
    loss: Optional[float] = None
    learning_rate: Optional[float] = None
    is_training: bool = False
    is_completed: bool = False
    error: Optional[str] = None
    warnings: list[str] = field(default_factory = list)
    status_message: str = "Ready to train"
    elapsed_seconds: Optional[float] = None
    eta_seconds: Optional[float] = None
    grad_norm: Optional[float] = None
    num_tokens: Optional[int] = None
    eval_loss: Optional[float] = None
    peak_memory_gb: Optional[float] = None
    output_dir: Optional[str] = None
    # The end-of-run record has no step loss, so the progress filter would drop it, and with it the only
    # elapsed time that includes the final evaluation, checkpoint save and best-model reload.
    is_run_summary: bool = False


class _MLXTrainerAdapter:
    """Adapts the legacy UnslothTrainer API to the shared Unsloth MLX worker path."""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.training_thread = None
        self.training_progress = TrainingProgress()
        self.progress_callbacks: list[Callable[[TrainingProgress], None]] = []
        self.is_training = False
        self.should_stop = False
        self.save_on_stop = True
        self.load_in_4bit = True
        self.output_dir = None

        self.is_cpt = False
        self.is_vlm = False
        self.is_audio = False
        self.is_audio_vlm = False
        self.model_name = None
        self.max_seq_length = None

        self._model_config: dict[str, Any] = {}
        self._peft_config: dict[str, Any] = {}
        self._dataset_config: dict[str, Any] = {}
        self._event_queue: Optional[queue.Queue] = None
        self._stop_queue: Optional[queue.Queue] = None
        self._pump_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def _activate_transformers_for_model(self, model_name: str, hf_token: Optional[str]) -> None:
        try:
            from utils.transformers_version import activate_transformers_for_subprocess
            activate_transformers_for_subprocess(model_name, hf_token)
        except Exception as exc:
            logger.warning("MLX trainer adapter Transformers activation failed", error = str(exc))

    def add_progress_callback(self, callback: Callable[[TrainingProgress], None]):
        self.progress_callbacks.append(callback)

    def _update_progress(self, **kwargs):
        with self._lock:
            for key, value in kwargs.items():
                if hasattr(self.training_progress, key):
                    setattr(self.training_progress, key, value)
            progress = self.training_progress
        for callback in self.progress_callbacks:
            try:
                callback(progress)
            except Exception:
                pass

    def load_model(
        self,
        model_name: str,
        max_seq_length: int = 2048,
        load_in_4bit: bool = True,
        hf_token: Optional[str] = None,
        is_dataset_image: bool = False,
        is_dataset_audio: bool = False,
        trust_remote_code: bool = False,
        full_finetuning: bool = False,
        gpu_ids: Optional[list[int]] = None,
    ) -> bool:
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.load_in_4bit = load_in_4bit
        self._audio_type = None
        self._activate_transformers_for_model(model_name, hf_token)
        try:
            from utils.models import detect_audio_type, is_vision_model

            self._audio_type = detect_audio_type(model_name, hf_token)
            if self._audio_type == "audio_vlm":
                self.is_audio = False
                self.is_audio_vlm = bool(is_dataset_audio)
                self._audio_type = None
            else:
                self.is_audio = self._audio_type is not None
                self.is_audio_vlm = False
            vision = is_vision_model(model_name, hf_token = hf_token) if not self.is_audio else False
            self.is_vlm = not self.is_audio_vlm and vision and bool(is_dataset_image)
        except Exception as exc:
            logger.warning("MLX trainer adapter model type detection failed", error = str(exc))
            self.is_vlm = False
            self.is_audio = False
            self.is_audio_vlm = False
        self.model = object()
        self.tokenizer = object()
        self._model_config = {
            "model_name": model_name,
            "max_seq_length": max_seq_length,
            "load_in_4bit": load_in_4bit,
            "hf_token": hf_token or "",
            "is_dataset_image": bool(is_dataset_image),
            "is_dataset_audio": bool(is_dataset_audio),
            "trust_remote_code": bool(trust_remote_code),
            "gpu_ids": gpu_ids,
        }
        self._update_progress(
            is_training = False,
            is_completed = False,
            error = None,
            step = 0,
            loss = 0.0,
            epoch = 0,
            status_message = f"Queued MLX model load: {model_name}",
        )
        return True

    def prepare_model_for_training(
        self,
        use_lora: bool = True,
        finetune_vision_layers: bool = True,
        finetune_language_layers: bool = True,
        finetune_attention_modules: bool = True,
        finetune_mlp_modules: bool = True,
        target_modules: Optional[Union[list, str]] = None,
        lora_r: int = 16,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
        use_gradient_checkpointing: Union[str, bool] = "unsloth",
        use_rslora: bool = False,
        use_loftq: bool = False,
        use_dora: bool = False,
    ) -> bool:
        self._peft_config = {
            "use_lora": bool(use_lora),
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "target_modules": target_modules,
            "gradient_checkpointing": use_gradient_checkpointing,
            "use_rslora": bool(use_rslora),
            "use_loftq": bool(use_loftq),
            "use_dora": bool(use_dora),
            "finetune_vision_layers": bool(finetune_vision_layers),
            "finetune_language_layers": bool(finetune_language_layers),
            "finetune_attention_modules": bool(finetune_attention_modules),
            "finetune_mlp_modules": bool(finetune_mlp_modules),
        }
        self._update_progress(status_message = "Queued MLX training setup")
        return True

    def load_and_format_dataset(
        self,
        dataset_source: Optional[str],
        format_type: str = "auto",
        local_datasets: Optional[list[str]] = None,
        local_eval_datasets: Optional[list[str]] = None,
        custom_format_mapping: Optional[dict[str, Any]] = None,
        subset: Optional[str] = None,
        train_split: str = "train",
        eval_split: Optional[str] = None,
        dataset_streaming: bool = False,
        eval_steps: float = 0.00,
        dataset_slice_start: Optional[int] = None,
        dataset_slice_end: Optional[int] = None,
        is_cpt: bool = False,
        s3_config: dict = None,
        dataset_local_files_only: bool = False,
        dataset_local_path: Optional[str] = None,
        dataset_revision: Optional[str] = None,
        require_exact_resume_resources: bool = False,
        max_train_rows: Optional[int] = None,
        max_train_rows_seed: int = 3407,
    ) -> Optional[tuple]:
        # Signature must match UnslothTrainer: the MLX worker loads its own data and derives the row bound
        # from its config, so the two bound arguments are accepted and deliberately not forwarded.
        self._dataset_config = {
            "hf_dataset": dataset_source or "",
            "local_datasets": local_datasets,
            "local_eval_datasets": local_eval_datasets,
            "format_type": format_type or "",
            "custom_format_mapping": custom_format_mapping,
            "subset": subset,
            "train_split": train_split or "train",
            "eval_split": eval_split,
            "dataset_streaming": bool(dataset_streaming),
            "eval_steps": eval_steps or 0.0,
            "dataset_slice_start": dataset_slice_start,
            "dataset_slice_end": dataset_slice_end,
            "s3_config": s3_config,
            "dataset_known_cached": bool(dataset_local_files_only),
            "dataset_snapshot_path": dataset_local_path,
            "dataset_revision": dataset_revision,
            "require_exact_dataset_resource": bool(require_exact_resume_resources),
        }
        self.is_cpt = bool(is_cpt)
        self._update_progress(status_message = "Queued MLX dataset load")
        return ({"dataset": [], "final_format": "deferred_mlx_cli", "success": True}, None)

    def start_training(
        self,
        dataset = None,
        eval_dataset = None,
        **training_args,
    ) -> bool:
        if self.is_training and self.training_thread and self.training_thread.is_alive():
            return False
        if self._pump_thread and self._pump_thread.is_alive():
            self._pump_thread.join(timeout = 2.0)
            if self._pump_thread.is_alive():
                self._update_progress(error = "Previous training event pump is still finalizing")
                return False
        if not self._model_config:
            self._update_progress(error = "Model not loaded")
            return False
        if not self._dataset_config:
            self._update_progress(error = "Dataset not loaded")
            return False
        if self.is_cpt:
            self._update_progress(
                error = "Continued Pretraining is not supported for MLX training yet.",
                is_training = False,
                is_completed = False,
            )
            return False

        config = self._build_worker_config(training_args)
        event_queue = queue.Queue()
        stop_queue = queue.Queue()
        self._event_queue = event_queue
        self._stop_queue = stop_queue
        self.should_stop = False
        self.is_training = True
        self.training_progress = TrainingProgress(
            is_training = True,
            status_message = "Initializing MLX training...",
        )

        self.training_thread = threading.Thread(
            target = self._run_training_thread,
            args = (config, event_queue, stop_queue),
            daemon = True,
        )
        self._pump_thread = threading.Thread(
            target = self._pump_events,
            args = (event_queue, self.training_thread),
            daemon = True,
        )
        self.training_thread.start()
        self._pump_thread.start()
        return True

    def _build_worker_config(self, training_args: dict[str, Any]) -> dict[str, Any]:
        peft = {
            "use_lora": True,
            "lora_r": 16,
            "lora_alpha": 16,
            "lora_dropout": 0.0,
            "target_modules": None,
            "gradient_checkpointing": "unsloth",
            "use_rslora": False,
            "use_loftq": False,
            "use_dora": False,
            "finetune_vision_layers": True,
            "finetune_language_layers": True,
            "finetune_attention_modules": True,
            "finetune_mlp_modules": True,
            **self._peft_config,
        }
        output_dir = training_args.get("output_dir")
        if output_dir:
            output_dir = os.path.abspath(os.path.expanduser(str(output_dir)))
        values = {
            **self._model_config,
            **self._dataset_config,
            **training_args,
            "training_type": (
                "Continued Pretraining"
                if self.is_cpt
                else "LoRA/QLoRA"
                if peft["use_lora"]
                else "Full Finetuning"
            ),
            **peft,
            "output_dir": output_dir,
            "allow_external_output_dir": bool(output_dir),
        }
        config = _build_training_worker_config(values)
        config["resolved_gpu_ids"] = None
        config["gpu_selection"] = None
        return config

    def _run_training_thread(
        self, config: dict[str, Any], event_queue: queue.Queue, stop_queue: queue.Queue
    ):
        try:
            self._run_mlx_worker(config, event_queue, stop_queue)
        except Exception as exc:
            if event_queue is not None:
                event_queue.put(
                    {
                        "type": "error",
                        "error": str(exc),
                        "stack": traceback.format_exc(limit = 20),
                        "ts": time.time(),
                    }
                )

    def _run_mlx_worker(
        self, config: dict[str, Any], event_queue: queue.Queue, stop_queue: queue.Queue
    ):
        from .worker import run_mlx_training_process
        run_mlx_training_process(
            event_queue = event_queue,
            stop_queue = stop_queue,
            config = config,
        )

    def _pump_events(self, event_queue: queue.Queue, training_thread: threading.Thread):
        while True:
            event = None
            try:
                event = event_queue.get(timeout = 0.25)
            except queue.Empty:
                pass
            if event is not None:
                self._handle_event(event)
                continue
            if not training_thread.is_alive():
                self._drain_events(event_queue)
                with self._lock:
                    if self.training_progress.is_training:
                        self.training_progress.is_training = False
                        if self.should_stop:
                            self.training_progress.status_message = "Training stopped."
                        elif (
                            not self.training_progress.error
                            and not self.training_progress.is_completed
                        ):
                            self.training_progress.error = "Training process exited unexpectedly"
                    self.is_training = False
                    self._event_queue = None
                    self._stop_queue = None
                return

    def _drain_events(self, event_queue: Optional[queue.Queue] = None):
        event_queue = event_queue or self._event_queue
        if event_queue is None:
            return
        while True:
            try:
                self._handle_event(event_queue.get_nowait())
            except queue.Empty:
                return

    def _handle_event(self, event: dict[str, Any]):
        etype = event.get("type")
        if etype == "status":
            self._update_progress(
                status_message = event.get("status_message") or event.get("message") or ""
            )
            return
        if etype == "warning":
            message = event.get("message")
            if isinstance(message, str):
                message = message.strip()
                if message:
                    with self._lock:
                        if message not in self.training_progress.warnings:
                            self.training_progress.warnings.append(message)
                            logger.warning(message)
            return
        if etype == "progress":
            self._update_progress(
                step = event.get("step", self.training_progress.step),
                epoch = event.get("epoch", self.training_progress.epoch),
                loss = event.get("loss", self.training_progress.loss),
                learning_rate = event.get("learning_rate", self.training_progress.learning_rate),
                total_steps = event.get("total_steps", self.training_progress.total_steps),
                elapsed_seconds = event.get(
                    "elapsed_seconds",
                    self.training_progress.elapsed_seconds,
                ),
                eta_seconds = event.get("eta_seconds", self.training_progress.eta_seconds),
                grad_norm = event.get("grad_norm", self.training_progress.grad_norm),
                num_tokens = event.get("num_tokens", self.training_progress.num_tokens),
                eval_loss = event.get("eval_loss", self.training_progress.eval_loss),
                peak_memory_gb = event.get("peak_memory_gb", self.training_progress.peak_memory_gb),
            )
            return
        if etype == "complete":
            status_message = event.get("status_message") or "Training completed"
            output_dir = event.get("output_dir")
            was_cancelled = self.should_stop or status_message.strip().lower() in {
                "training cancelled",
                "training stopped",
            }
            self.output_dir = output_dir
            self._update_progress(
                is_training = False,
                is_completed = not was_cancelled,
                error = None,
                status_message = status_message,
                output_dir = output_dir,
            )
            self.is_training = False
            return
        if etype == "error":
            self._update_progress(
                is_training = False,
                is_completed = False,
                error = event.get("error") or event.get("message") or "Training failed",
            )
            self.is_training = False
            return

    def stop_training(self, save: bool = True):
        self.should_stop = True
        self.save_on_stop = bool(save)
        if self._stop_queue is not None:
            self._stop_queue.put({"type": "stop", "save": save})
        status_message = (
            "Stopping training and saving checkpoint..." if save else "Cancelling training..."
        )
        self._update_progress(status_message = status_message)
        return True

    def get_training_progress(self) -> TrainingProgress:
        pump_thread = self._pump_thread
        training_thread = self.training_thread
        if (
            pump_thread is not None
            and pump_thread.is_alive()
            and (training_thread is None or not training_thread.is_alive())
            and threading.current_thread() is not pump_thread
        ):
            pump_thread.join(timeout = 5.0)
        if pump_thread is None or not pump_thread.is_alive():
            self._drain_events()
        with self._lock:
            return replace(self.training_progress)


def create_mlx_trainer_adapter(*args, **kwargs):
    return _MLXTrainerAdapter(*args, **kwargs)


class TrainingBackend:
    """
    Training orchestration backend — subprocess-based.
    Launches a fresh subprocess per job, communicates via mp.Queue.
    """

    FLUSH_THRESHOLD: int = 10

    def __init__(self):
        self._proc: Optional[mp.Process] = None
        # True from the sidecar-swap handshake until the worker is recorded (startup counts as active).
        self._spawn_in_progress: bool = False
        self._new_job_spawn_id: Optional[str] = None
        self._event_queue: Any = None
        self._stop_queue: Any = None
        self._pump_thread: Optional[threading.Thread] = None
        # True while a pump thread should run; left True after an abnormal death so a crash is spotted.
        self._pump_running: bool = False
        self._lock = threading.Lock()
        self._provenance_lock = threading.Lock()
        self._run_intent_lock = threading.RLock()

        # The watched proc is tracked so a new run always gets its own watchdog.
        self._stop_watchdog: Optional[threading.Thread] = None
        self._stop_watchdog_proc: Optional[mp.Process] = None
        self._complete_seen = threading.Event()

        # Progress state (updated by pump thread from subprocess events)
        self._progress = TrainingProgress()
        self._should_stop = False
        self._cancel_requested = False  # True only for stop(save=False)
        self._cancel_cleanup_output_dir: Optional[str] = None

        # Throttled training-status logging to the server log (not one line/step).
        self._last_progress_log_ts: float = 0.0
        self._last_progress_log_step: int = -1
        # (elapsed_seconds, num_tokens) at the previous logged line, so the next one reports throughput over
        # the interval between them.
        self._last_progress_log_elapsed: Optional[float] = None
        self._last_progress_log_tokens: Optional[int] = None

        # Training metrics (consumed by routes for SSE and /metrics)
        self.loss_history: list = []
        self.lr_history: list = []
        self.step_history: list = []
        self.grad_norm_history: list = []
        self.grad_norm_step_history: list = []
        self.eval_loss_history: list = []
        self.eval_step_history: list = []
        self.eval_enabled: bool = False
        self.current_theme: str = "light"

        self.current_job_id: Optional[str] = None
        self.current_start_request_id: Optional[str] = None
        self._start_requests: dict[str, TrainingStartRequestRecord] = {}
        self._start_cancel_tombstones: dict[str, tuple[float, TrainingStartRequestRecord]] = {}
        self._start_cancel_tombstone_reservations: dict[str, int] = {}
        self._pending_start_request_id: Optional[str] = None
        self._status_start_request_id: Optional[str] = None
        self._output_dir: Optional[str] = None
        self._resume_source_run_id: Optional[str] = None
        self._terminal_finalize_payload: Optional[dict] = None

        self._metric_buffer: list[dict] = []
        self._run_finalized: bool = False
        self._db_run_created: bool = False
        self._db_create_in_progress: bool = False
        self._db_total_steps_set: bool = False
        self._db_config: Optional[dict] = None
        self._db_started_at: Optional[str] = None

        # Xet -> HTTP model-load fallback state (config kept for the respawn).
        self._last_full_config: Optional[dict] = None
        self._in_model_load: bool = False
        self._xet_fallback_used: bool = False
        self._needs_xet_respawn: bool = False

        logger.info("TrainingBackend initialized (subprocess mode)")


    def reserve_start_request(
        self, start_request_id: str, job_id: str
    ) -> tuple[str, TrainingStartRequestRecord]:
        with self._lock:
            self._prune_start_cancel_tombstones_locked()
            existing = self._start_requests.get(start_request_id)
            if existing is not None:
                return "existing", existing
            cancelled = self._start_cancel_tombstones.get(start_request_id)
            if cancelled is not None:
                record = cancelled[1]
                self._start_cancel_tombstones[start_request_id] = (
                    time.monotonic() + _START_CANCEL_TOMBSTONE_TTL_S,
                    record,
                )
                return "existing", record
            if self._pending_start_request_id is not None:
                record = TrainingStartRequestRecord(
                    start_request_id = start_request_id,
                    job_id = job_id,
                    state = "rejected",
                    message = (
                        "Another training start is still being processed. "
                        "Wait for it to finish before starting a new one."
                    ),
                    error = "Training start already pending",
                )
                self._start_requests[start_request_id] = record
                self._prune_start_requests_locked()
                return "conflict", record

            record = TrainingStartRequestRecord(
                start_request_id = start_request_id,
                job_id = job_id,
                state = "pending",
                message = "Training start is being validated",
            )
            self._start_requests[start_request_id] = record
            self._pending_start_request_id = start_request_id
            self._status_start_request_id = start_request_id
            self._prune_start_requests_locked()
            return "reserved", record

    def peek_start_request(self, start_request_id: str) -> Optional[TrainingStartRequestRecord]:
        """The lookup half of reserve_start_request(), with no reservation.

        Returns the record a retry would replay (live or cancellation-tombstoned), refreshing
        the tombstone TTL as the reserve path does so a retry keeps a cancellation alive, or
        None when the id is unknown and the caller is free to reserve it."""
        with self._lock:
            self._prune_start_cancel_tombstones_locked()
            existing = self._start_requests.get(start_request_id)
            if existing is not None:
                return existing
            cancelled = self._start_cancel_tombstones.get(start_request_id)
            if cancelled is None:
                return None
            record = cancelled[1]
            self._start_cancel_tombstones[start_request_id] = (
                time.monotonic() + _START_CANCEL_TOMBSTONE_TTL_S,
                record,
            )
            return record

    def resolve_start_request(
        self,
        start_request_id: str,
        *,
        state: Literal["accepted", "rejected"],
        message: str,
        error: Optional[str] = None,
        error_code: Optional[str] = None,
    ) -> Optional[TrainingStartRequestRecord]:
        if state not in {"accepted", "rejected"}:
            raise ValueError(f"Invalid training start request state: {state}")
        with self._lock:
            existing = self._start_requests.get(start_request_id)
            if existing is None:
                return None
            if existing.state != "pending":
                return existing
            record = replace(
                existing,
                state = state,
                message = message,
                error = error,
                error_code = error_code,
            )
            self._start_requests[start_request_id] = record
            if self._pending_start_request_id == start_request_id:
                self._pending_start_request_id = None
            return record

    def cancel_start_request(
        self, start_request_id: str
    ) -> tuple[Literal["cancelled", "superseded"], TrainingStartRequestRecord]:
        from .lifecycle import training_lifecycle_guard

        reserved_cancel_tombstone = False
        with self._lock:
            self._prune_start_cancel_tombstones_locked()
            existing = self._start_requests.get(start_request_id)
            if existing is None:
                cancelled_tombstone = self._start_cancel_tombstones.get(start_request_id)
                if cancelled_tombstone is not None:
                    record = cancelled_tombstone[1]
                    self._start_cancel_tombstones[start_request_id] = (
                        time.monotonic() + _START_CANCEL_TOMBSTONE_TTL_S,
                        record,
                    )
                    return "cancelled", record
                self._reserve_start_cancel_tombstone_locked(start_request_id)
                cancelled = TrainingStartRequestRecord(
                    start_request_id = start_request_id,
                    job_id = "",
                    state = "rejected",
                    message = "Training start was cancelled",
                    error = "Training start was cancelled",
                    error_code = _START_CANCELLED_ERROR_CODE,
                )
                self._commit_start_cancel_tombstone_locked(start_request_id, cancelled)
                return "cancelled", cancelled

            owns_current = (
                self.current_start_request_id == start_request_id
                and self.current_job_id == existing.job_id
            )
            if existing.state == "rejected" and (
                not owns_current or existing.error_code == _START_CANCELLED_ERROR_CODE
            ):
                if existing.error_code == _START_CANCELLED_ERROR_CODE:
                    self._reserve_start_cancel_tombstone_locked(
                        start_request_id,
                        reclaim_capacity = True,
                    )
                    self._start_requests.pop(start_request_id, None)
                    self._commit_start_cancel_tombstone_locked(start_request_id, existing)
                if self._status_start_request_id == start_request_id:
                    self._status_start_request_id = None
                return "cancelled", existing

            if existing.state == "pending" and not owns_current:
                # Not the active run, so it does not get the owner's over-cap slot: start plus cancel could
                # otherwise be repeated to grow the table without bound.
                self._reserve_start_cancel_tombstone_locked(start_request_id)
                cancelled = replace(
                    existing,
                    state = "rejected",
                    message = "Training start was cancelled",
                    error = "Training start was cancelled",
                    error_code = _START_CANCELLED_ERROR_CODE,
                )
                self._start_requests.pop(start_request_id, None)
                self._commit_start_cancel_tombstone_locked(start_request_id, cancelled)
                if self._pending_start_request_id == start_request_id:
                    self._pending_start_request_id = None
                if self._status_start_request_id == start_request_id:
                    self._status_start_request_id = None
                return "cancelled", cancelled

            reserved_cancel_tombstone = self._reserve_start_cancel_tombstone_locked(
                start_request_id,
                reclaim_capacity = owns_current,
            )

        try:
            with training_lifecycle_guard():
                with self._lock:
                    cancelled_tombstone = self._start_cancel_tombstones.get(start_request_id)
                    if cancelled_tombstone is not None:
                        record = cancelled_tombstone[1]
                        self._start_cancel_tombstones[start_request_id] = (
                            time.monotonic() + _START_CANCEL_TOMBSTONE_TTL_S,
                            record,
                        )
                        return "cancelled", record
                    latest = self._start_requests.get(start_request_id)
                    if latest is None:
                        return "superseded", existing
                    existing = latest
                    owns_current = (
                        self.current_start_request_id == start_request_id
                        and self.current_job_id == existing.job_id
                    )
                    if not reserved_cancel_tombstone:
                        reserved_cancel_tombstone = self._reserve_start_cancel_tombstone_locked(
                            start_request_id,
                            reclaim_capacity = owns_current,
                        )
                    if existing.error_code == _START_CANCELLED_ERROR_CODE:
                        if self._status_start_request_id == start_request_id:
                            self._status_start_request_id = None
                        return "cancelled", existing
                    if not owns_current or self._run_finished_locked():
                        return "superseded", existing
                    expected_job_id = existing.job_id
                if not self._stop_training_with_lifecycle_reserved(
                    save = False,
                    expected_job_id = expected_job_id,
                ):
                    return "superseded", existing

            if self.reset_training_state(expected_job_id = expected_job_id) != "reset":
                return "superseded", existing

            with self._lock:
                cancelled_tombstone = self._start_cancel_tombstones.get(start_request_id)
                if cancelled_tombstone is not None:
                    record = cancelled_tombstone[1]
                    self._start_cancel_tombstones[start_request_id] = (
                        time.monotonic() + _START_CANCEL_TOMBSTONE_TTL_S,
                        record,
                    )
                    return "cancelled", record
                latest = self._start_requests.get(start_request_id)
                if latest is None:
                    return "superseded", existing
                if (
                    self.current_start_request_id != start_request_id
                    or self.current_job_id != expected_job_id
                ):
                    return "superseded", latest
                cancelled = replace(
                    latest,
                    state = "rejected",
                    message = "Training start was cancelled",
                    error = "Training start was cancelled",
                    error_code = _START_CANCELLED_ERROR_CODE,
                )
                self._start_requests.pop(start_request_id, None)
                self._commit_start_cancel_tombstone_locked(start_request_id, cancelled)
                reserved_cancel_tombstone = False
                self.current_start_request_id = None
                if self._pending_start_request_id == start_request_id:
                    self._pending_start_request_id = None
                if self._status_start_request_id == start_request_id:
                    self._status_start_request_id = None
                return "cancelled", cancelled
        finally:
            if reserved_cancel_tombstone:
                with self._lock:
                    self._release_start_cancel_tombstone_locked(start_request_id)

    def _start_request_allows_spawn_locked(
        self, start_request_id: Optional[str], job_id: str
    ) -> bool:
        if start_request_id is None:
            return True
        record = self._start_requests.get(start_request_id)
        return bool(record is not None and record.state == "pending" and record.job_id == job_id)

    def get_start_request(self, start_request_id: str) -> Optional[TrainingStartRequestRecord]:
        with self._lock:
            self._prune_start_cancel_tombstones_locked()
            record = self._start_requests.get(start_request_id)
            if record is not None:
                return record
            cancelled = self._start_cancel_tombstones.get(start_request_id)
            return cancelled[1] if cancelled is not None else None

    def status_start_request(self) -> Optional[TrainingStartRequestRecord]:
        with self._lock:
            if self._status_start_request_id is None:
                return None
            return self._start_requests.get(self._status_start_request_id)

    def training_status_identity(self) -> TrainingStatusIdentitySnapshot:
        with self._lock:
            current_start_request = (
                self._start_requests.get(self.current_start_request_id)
                if self.current_start_request_id is not None
                else None
            )
            status_start_request = (
                self._start_requests.get(self._status_start_request_id)
                if self._status_start_request_id is not None
                else None
            )
            return TrainingStatusIdentitySnapshot(
                current_job_id = self.current_job_id or "",
                current_start_request_id = self.current_start_request_id,
                current_start_request = current_start_request,
                status_start_request = status_start_request,
                new_job_spawn_id = self._new_job_spawn_id,
                spawn_in_progress = self._spawn_in_progress,
            )

    @contextmanager
    def _new_job_spawn_reservation(self, job_id: str) -> Iterator[bool]:
        with self._lock:
            reserved = not self._spawn_in_progress and self._new_job_spawn_id is None
            if reserved:
                self._new_job_spawn_id = job_id
                self._spawn_in_progress = True
        try:
            yield reserved
        finally:
            if reserved:
                with self._lock:
                    if self._new_job_spawn_id == job_id:
                        self._spawn_in_progress = False
                        self._new_job_spawn_id = None

    def acknowledge_start_request(self, start_request_id: str) -> bool:
        with self._lock:
            self._prune_start_cancel_tombstones_locked()
            record = self._start_requests.get(start_request_id)
            if record is None and start_request_id in self._start_cancel_tombstones:
                return True
            if record is None or record.state == "pending":
                return False
            if self._status_start_request_id == start_request_id:
                self._status_start_request_id = None
            return True

    def _prune_start_cancel_tombstones_locked(self) -> None:
        now = time.monotonic()
        for request_id, (expires_at, _) in tuple(self._start_cancel_tombstones.items()):
            if expires_at <= now:
                del self._start_cancel_tombstones[request_id]

    def _reserve_start_cancel_tombstone_locked(
        self,
        start_request_id: str,
        *,
        reclaim_capacity: bool = False,
    ) -> bool:
        self._prune_start_cancel_tombstones_locked()
        if start_request_id in self._start_cancel_tombstones:
            return False
        reservation_count = self._start_cancel_tombstone_reservations.get(start_request_id, 0)
        if reservation_count:
            self._start_cancel_tombstone_reservations[start_request_id] = reservation_count + 1
            return True
        if (
            len(self._start_cancel_tombstones) + len(self._start_cancel_tombstone_reservations)
            >= _MAX_START_CANCEL_TOMBSTONES
        ):
            if not reclaim_capacity:
                raise TrainingStartCancellationCapacityError(
                    "Too many training start cancellations are pending"
                )
            # Everything left is unexpired, so evicting one would forget a live cancellation and let its delayed
            # /start spawn the job just cancelled.
        self._start_cancel_tombstone_reservations[start_request_id] = 1
        return True

    def _commit_start_cancel_tombstone_locked(
        self, start_request_id: str, record: TrainingStartRequestRecord
    ) -> None:
        self._start_cancel_tombstone_reservations.pop(start_request_id, None)
        self._start_cancel_tombstones[start_request_id] = (
            time.monotonic() + _START_CANCEL_TOMBSTONE_TTL_S,
            record,
        )

    def _release_start_cancel_tombstone_locked(self, start_request_id: str) -> None:
        reservation_count = self._start_cancel_tombstone_reservations.get(start_request_id, 0)
        if reservation_count <= 1:
            self._start_cancel_tombstone_reservations.pop(start_request_id, None)
        else:
            self._start_cancel_tombstone_reservations[start_request_id] = reservation_count - 1

    def _prune_start_requests_locked(self) -> None:
        overflow = len(self._start_requests) - _MAX_TRACKED_START_REQUESTS
        if overflow <= 0:
            return
        for request_id, record in tuple(self._start_requests.items()):
            if overflow <= 0:
                break
            if record.state == "pending" or request_id == self.current_start_request_id:
                continue
            del self._start_requests[request_id]
            overflow -= 1

    def start_training(
        self,
        job_id: str,
        *,
        before_spawn = None,
        resume_source_run_id: Optional[str] = None,
        start_request_id: Optional[str] = None,
        **kwargs,
    ) -> bool:
        # Reserve before lifecycle locking and validation: routes call start_training from worker threads,
        # so this compare-and-set stops two requests reaching the spawn.
        with self._new_job_spawn_reservation(job_id) as spawn_reserved:
            if not spawn_reserved:
                logger.warning("Training subprocess already running")
                return False

            from .lifecycle import training_lifecycle_guard
            with training_lifecycle_guard():
                resume_checkpoint = kwargs.get("resume_from_checkpoint")
                if resume_checkpoint:
                    from .resume import get_resume_checkpoint_path
                    if get_resume_checkpoint_path(resume_checkpoint) is None:
                        message = "Resume checkpoint is no longer available."
                        with self._lock:
                            self._progress.is_training = False
                            self._progress.error = message
                            self._progress.status_message = message
                        return False
                return self._start_training_with_lifecycle_reserved(
                    job_id,
                    before_spawn = before_spawn,
                    resume_source_run_id = resume_source_run_id,
                    start_request_id = start_request_id,
                    spawn_already_reserved = True,
                    **kwargs,
                )

    def _start_training_with_lifecycle_reserved(
        self,
        job_id: str,
        *,
        before_spawn = None,
        resume_source_run_id: Optional[str] = None,
        start_request_id: Optional[str] = None,
        spawn_already_reserved: bool = False,
        **kwargs,
    ) -> bool:
        """Spawn a subprocess to run the full training pipeline.

        All kwargs are serialized into a config dict and sent to the worker.
        Returns True if the subprocess started successfully.

        ``before_spawn`` is an optional no-arg callable run after synchronous
        validation (start guards, config build, explicit gpu_ids) passes but
        before VRAM-dependent auto GPU-selection and the spawn -- used to free
        VRAM (e.g. unload chat) without tearing it down on a refused start, while
        still letting auto-selection place training against the freed memory.
        Hook failures never block the start.
        """
        with self._lock:
            if not self._start_request_allows_spawn_locked(start_request_id, job_id):
                logger.info(
                    "Training start request %s was resolved before worker spawn",
                    start_request_id,
                )
                return False
            if (self._spawn_in_progress and not spawn_already_reserved) or (
                self._proc is not None and self._proc.is_alive()
            ):
                logger.warning("Training subprocess already running")
                return False

        # Wait for pump thread to finish DB finalization (8s covers SQLite's 5s lock timeout).
        if self._pump_thread is not None and self._pump_thread.is_alive():
            self._pump_thread.join(timeout = 5.0)
            if self._pump_thread.is_alive():
                logger.warning("Previous pump thread did not exit within 5s — refusing to start")
                return False
        self._pump_thread = None
        # Clear a stale crash flag so the watchdog can't treat this fresh setup as a death.
        self._pump_running = False

        config = _build_training_worker_config(kwargs)

        _apply_cache_pins(config)
        from .provenance import initialize_resource_provenance

        initialize_resource_provenance(config)

        # Explicit gpu_ids are validated here, so the route 400s before any teardown and their placement
        # survives the VRAM hook; auto-selection ranks GPUs by FREE VRAM, so it is deferred until after
        # the hook, else it pins training onto a GPU the hook is about to clear.
        from utils.hardware import hardware as _hw

        gpu_ids = kwargs.get("gpu_ids")
        gpu_selection_kwargs = dict(
            model_name = config["model_name"],
            hf_token = config["hf_token"] or None,
            training_type = config["training_type"],
            load_in_4bit = config["load_in_4bit"],
            batch_size = config.get("batch_size", 4),
            max_seq_length = config.get("max_seq_length", 2048),
            lora_rank = config.get("lora_r", 16),
            target_modules = config.get("target_modules"),
            gradient_checkpointing = config.get("gradient_checkpointing", "unsloth"),
            optimizer = config.get("optim", "adamw_8bit"),
        )

        defer_auto_selection = False
        if should_use_mlx_training_backend(device = _hw.DEVICE):
            config["resolved_gpu_ids"] = None
            config["gpu_selection"] = None
        elif gpu_ids:
            resolved_gpu_ids, gpu_selection = prepare_gpu_selection(gpu_ids, **gpu_selection_kwargs)
            config["resolved_gpu_ids"] = resolved_gpu_ids
            config["gpu_selection"] = gpu_selection
        else:
            defer_auto_selection = True

        # Handshake with the sidecar install route: mark the spawn in progress BEFORE rechecking the
        # reservation, so either this recheck aborts or the install sees the flag and refuses.
        from utils.transformers_version import sidecar_swap_in_progress

        spawn_reservation = (
            nullcontext(True) if spawn_already_reserved else self._new_job_spawn_reservation(job_id)
        )
        with spawn_reservation as spawn_reserved:
            if not spawn_reserved:
                logger.warning("Training subprocess already running")
                return False
            if sidecar_swap_in_progress():
                from utils.transformers_version import SidecarSwapInProgress
                raise SidecarSwapInProgress(
                    "A transformers installation is replacing the latest sidecar; "
                    "retry when it completes."
                )

            if (
                config.get("require_exact_resume_resources")
                or config.get("require_exact_model_resource")
            ) and config.get("load_in_4bit"):
                from .provenance import effective_training_load_in_4bit
                effective_training_load_in_4bit(
                    config,
                    config.get("model_snapshot_path") or config["model_name"],
                    config.get("hf_token") or None,
                )
            with self._lock:
                if not self._start_request_allows_spawn_locked(start_request_id, job_id):
                    logger.info(
                        "Training start request %s was cancelled during validation",
                        start_request_id,
                    )
                    return False
            # Free VRAM after the handshake, so a lost race cannot tear down chat for a run that never spawns.
            if before_spawn is not None:
                try:
                    before_spawn()
                except Exception:
                    logger.warning("before_spawn hook failed; continuing", exc_info = True)

            if defer_auto_selection:
                resolved_gpu_ids, gpu_selection = prepare_gpu_selection(
                    None, **gpu_selection_kwargs
                )
                config["resolved_gpu_ids"] = resolved_gpu_ids
                config["gpu_selection"] = gpu_selection

            from utils.hf_cache_settings import child_environment_for_spawn, get_hf_cache_paths

            cache_env = get_hf_cache_paths().child_env({})

            try:
                with (
                    child_environment_for_spawn(cache_env),
                    native_path_secret_removed_for_child_start(),
                ):
                    event_queue = _CTX.Queue()
                    stop_queue = _CTX.Queue()

                    proc = _CTX.Process(
                        target = run_without_native_path_secret,
                        args = ("core.training.worker", "run_training_process", cache_env),
                        kwargs = {
                            "event_queue": event_queue,
                            "stop_queue": stop_queue,
                            "config": config,
                        },
                        daemon = True,
                    )
                    from utils.process_lifetime import adopt_pid

                    previous_job_id = None
                    previous_start_request_id = None
                    with self._lock:
                        if not self._start_request_allows_spawn_locked(
                            start_request_id,
                            job_id,
                        ):
                            logger.info(
                                "Training start request %s was cancelled before worker spawn",
                                start_request_id,
                            )
                            return False
                        previous_job_id = self.current_job_id
                        previous_start_request_id = self.current_start_request_id
                        proc.start()
                        self.current_job_id = job_id
                        self.current_start_request_id = start_request_id
                    try:
                        adopt_pid(proc.pid)
                    except Exception:
                        logger.error(
                            "Failed to adopt training subprocess; terminating it",
                            exc_info = True,
                        )
                        try:
                            if proc.is_alive():
                                proc.terminate()
                            proc.join(timeout = 5.0)
                            if proc.is_alive():
                                proc.kill()
                                proc.join(timeout = 2.0)
                        finally:
                            with self._lock:
                                if (
                                    self.current_job_id == job_id
                                    and self.current_start_request_id == start_request_id
                                ):
                                    self.current_job_id = previous_job_id
                                    self.current_start_request_id = previous_start_request_id
                                if start_request_id is not None:
                                    record = self._start_requests.get(start_request_id)
                                    if record is not None and record.state == "pending":
                                        self._start_requests[start_request_id] = replace(
                                            record,
                                            state = "rejected",
                                            message = "Failed to start training subprocess",
                                            error = "Failed to adopt training subprocess",
                                        )
                                        if self._pending_start_request_id == start_request_id:
                                            self._pending_start_request_id = None
                        return False
            except Exception:
                logger.error("Failed to start training subprocess", exc_info = True)
                return False

            logger.info("Training subprocess started (pid=%s)", proc.pid)

            self._should_stop = False
            self._cancel_requested = False
            self._cancel_cleanup_output_dir = None
            self._complete_seen.clear()
            self._progress = TrainingProgress(
                is_training = True, status_message = "Initializing training..."
            )
            # Reset the throttle so the new run logs its first step even within 30s of a prior run.
            self._last_progress_log_ts = 0.0
            self._last_progress_log_step = -1
            self._last_progress_log_elapsed = None
            self._last_progress_log_tokens = None
            self.loss_history.clear()
            self.lr_history.clear()
            self.step_history.clear()
            self.grad_norm_history.clear()
            self.grad_norm_step_history.clear()
            self.eval_loss_history.clear()
            self.eval_step_history.clear()
            self.eval_enabled = False
            self._output_dir = config.get("output_dir") if resume_source_run_id else None
            self._progress.output_dir = self._output_dir
            self._resume_source_run_id = resume_source_run_id
            self._terminal_finalize_payload = None
            self._metric_buffer.clear()
            self._run_finalized = False
            self._db_run_created = False
            self._db_create_in_progress = False
            self._db_total_steps_set = False
            self._db_config = _sanitize_db_config(config)
            self._db_started_at = datetime.now(timezone.utc).isoformat()
            # Start each job Xet-first; keep config so a stall can respawn over HTTP.
            self._last_full_config = config
            self._last_hf_cache_env = cache_env
            self._in_model_load = False
            self._xet_fallback_used = False
            self._needs_xet_respawn = False

            # Create the DB run row before the pump consumes events, so it appears in history during model
            # loading and a fast terminal worker cannot race the pump.
            self._ensure_db_run_created()
            if resume_source_run_id and not self._db_run_created:
                if proc.is_alive():
                    proc.terminate()
                proc.join(timeout = 5.0)
                if proc.is_alive():
                    proc.kill()
                    proc.join(timeout = 2.0)
                self._progress.is_training = False
                self._progress.error = "Resume checkpoint is no longer available."
                return False

            # Assign handles and start the pump under the lock, else a poll sees a live _proc with no pump.
            new_pump = threading.Thread(target = self._pump_loop, daemon = True)
            with self._lock:
                self._pump_running = False
                self._event_queue = event_queue
                self._stop_queue = stop_queue
                self._proc = proc
                self._pump_thread = new_pump
                # Start under the lock so a concurrent _ensure_pump_alive can't spawn yet another pump.
                new_pump.start()
                if self._new_job_spawn_id == job_id:
                    self._spawn_in_progress = False
                    self._new_job_spawn_id = None

            if start_request_id is not None:
                self.resolve_start_request(
                    start_request_id,
                    state = "accepted",
                    message = "Training job queued and starting in subprocess",
                )
            return True

    def stop_training(
        self,
        save: bool = True,
        *,
        expected_job_id: str,
    ) -> bool:
        """Send stop signal to the training subprocess."""
        from .lifecycle import training_lifecycle_guard
        with training_lifecycle_guard():
            return self._stop_training_with_lifecycle_reserved(
                save = save,
                expected_job_id = expected_job_id,
            )

    def _stop_training_with_lifecycle_reserved(self, save: bool, expected_job_id: str) -> bool:
        with self._run_intent_lock:
            with self._lock:
                if not expected_job_id or self.current_job_id != expected_job_id:
                    return False
                run_id = self.current_job_id
            if not save and run_id:
                persist_error: Optional[Exception] = None
                for attempt in range(_DB_FINALIZE_RETRIES):
                    try:
                        from storage.studio_db import mark_run_cancel_requested

                        self._ensure_db_run_created()
                        with self._lock:
                            terminal_payload = self._terminal_finalize_payload
                            if (
                                terminal_payload
                                and terminal_payload.get("expected_job_id") == run_id
                            ):
                                return False
                            if not mark_run_cancel_requested(run_id):
                                if self._db_run_created:
                                    return False
                                raise RuntimeError(
                                    "Training run disappeared before cancellation persisted"
                                )
                            if self.current_job_id != run_id:
                                return False
                            self._should_stop = self._cancel_requested = True
                            self._cancel_cleanup_output_dir = self._output_dir
                            self._output_dir = self._progress.output_dir = None
                        persist_error = None
                        break
                    except Exception as exc:
                        persist_error = exc
                        if attempt + 1 < _DB_FINALIZE_RETRIES:
                            time.sleep(_DB_FINALIZE_RETRY_S)
                if persist_error is not None:
                    raise RuntimeError("Failed to persist Stop-without-Save") from persist_error
            with self._lock:
                if self.current_job_id != run_id:
                    return False
                # The pump can finish the run between the route's terminal check and this lock, so re-test: latching
                # _should_stop after the fact would report a saved run as stopped for good.
                if save and self._run_finished_locked():
                    return False
                if save or not run_id:
                    self._should_stop = True
                if not save and not run_id:
                    self._cancel_requested = True
                    self._cancel_cleanup_output_dir = self._output_dir
                    self._output_dir = self._progress.output_dir = None
                self._needs_xet_respawn = False
                if self._stop_queue is not None:
                    try:
                        self._stop_queue.put({"type": "stop", "save": save})
                    except (OSError, ValueError):
                        pass
                self._progress.status_message = (
                    "Stopping training and saving checkpoint..."
                    if save
                    else "Cancelling training..."
                )
        self._start_stop_watchdog(cancel = not save, expected_job_id = run_id)
        return True

    def reset_training_state(self, expected_job_id: Optional[str] = None) -> str:
        from .lifecycle import training_lifecycle_guard

        with training_lifecycle_guard():
            with self._lock:
                if expected_job_id is not None and self.current_job_id != expected_job_id:
                    return "superseded"
                target_job_id = self.current_job_id

            is_active = self.is_training_active()
            with self._lock:
                if self.current_job_id != target_job_id or (
                    expected_job_id is not None and self.current_job_id != expected_job_id
                ):
                    return "superseded"
                # An unscoped reset cannot prove it means THIS run, so it never force-terminates: _cancel_requested
                # is cleared after current_job_id is set, so a bodyless reset landing in that window would kill the
                # run that just started. Otherwise this is a stale reset of a live run (409).
                if expected_job_id is None and is_active:
                    return "superseded" if self._cancel_requested else "active"
                cancel_requested = self._cancel_requested
                proc = self._proc

            if is_active:
                if not cancel_requested:
                    return "active"

        if is_active:
            self.force_terminate(target_proc = proc)

        with training_lifecycle_guard():
            with self._lock:
                if self.current_job_id != target_job_id or (
                    expected_job_id is not None and self.current_job_id != expected_job_id
                ):
                    return "superseded"
                self._should_stop = False
                self._progress.is_training = False
                self._progress.is_completed = False
                self._progress.error = None
                self._progress.status_message = "Ready to train"
                self._progress.step = 0
                self._progress.loss = None
                self._progress.epoch = 0
                self._progress.total_steps = 0
                self.loss_history.clear()
                self.lr_history.clear()
                self.step_history.clear()
                self.grad_norm_history.clear()
                self.grad_norm_step_history.clear()
                self._needs_xet_respawn = False
                self._status_start_request_id = None
            return "reset"

    def _start_stop_watchdog(
        self,
        cancel: bool,
        expected_job_id: Optional[str] = None,
        grace_s: Optional[float] = None,
        terminal_seen: bool = False,
    ) -> None:
        """Start a daemon that force-terminates a worker that will not exit. Armed by a stop
        and by a run's own terminal event, since a wedged worker strands the UI either way.
        No-op if no worker is alive or a live watchdog already watches this proc (a stale one
        never blocks a new run). ``grace_s`` overrides the post-terminal grace;
        ``terminal_seen`` starts it now, for an ending that never sets ``_complete_seen``."""
        with self._lock:
            if expected_job_id is not None and self.current_job_id != expected_job_id:
                return
            proc = self._proc
            if proc is None or not proc.is_alive():
                return
            if (
                self._stop_watchdog is not None
                and self._stop_watchdog.is_alive()
                and self._stop_watchdog_proc is proc
            ):
                return
            watchdog = threading.Thread(
                target = self._stop_watchdog_loop,
                args = (proc, cancel, self.current_job_id),
                kwargs = {"grace_s": grace_s, "terminal_seen": terminal_seen},
                name = f"stop-watchdog-{self.current_job_id or 'unknown'}",
                daemon = True,
            )
            self._stop_watchdog = watchdog
            self._stop_watchdog_proc = proc
            watchdog.start()

    def _stop_watchdog_loop(
        self,
        target_proc: "mp.Process",
        cancel: bool,
        watched_job_id: Optional[str] = None,
        grace_s: Optional[float] = None,
        terminal_seen: bool = False,
    ) -> None:
        """Escalate a worker that will not exit to force_terminate(): grace after "complete",
        else the absolute backstop (module timeouts). No-ops on a clean exit or once a new run
        replaces the worker. ``grace_s`` overrides ``_STOP_GRACE_S``; ``terminal_seen`` starts
        the grace at entry, so an ending that never sets ``_complete_seen`` (an error) does
        not sit out the whole backstop."""
        started = time.monotonic()
        complete_at: Optional[float] = started if terminal_seen else None
        reason = ""
        while True:
            with self._lock:
                superseded = self._proc is not target_proc
                # A later cancel has nothing to save, so tighten an in-flight save watchdog to the cancel cap.
                cancelling = cancel or self._cancel_requested
            if superseded or not target_proc.is_alive():
                return
            now = time.monotonic()
            abs_timeout = _CANCEL_TIMEOUT_S if cancelling else _STOP_TIMEOUT_S
            grace = _STOP_GRACE_S if grace_s is None else grace_s
            if complete_at is None and self._complete_seen.is_set():
                complete_at = now
            if complete_at is not None and now - complete_at >= grace:
                reason = "worker still alive after save"
                break
            if now - started >= abs_timeout:
                reason = "worker did not exit within the absolute timeout"
                break
            time.sleep(0.5)

        with self._lock:
            superseded = self._proc is not target_proc
        if superseded or not target_proc.is_alive():
            return
        if complete_at is None:
            # Backstop fired pre-completion: a save may still be in progress.
            logger.warning(
                "Stop watchdog: absolute timeout with no completion signal; "
                "force-terminating a possibly-mid-save worker: %s",
                reason,
            )
        else:
            logger.warning(
                "Training watchdog force-terminating a worker that will not exit: %s", reason
            )
        # force_terminate can raise on a wedged child; finalize regardless.
        try:
            self.force_terminate(target_proc = target_proc)
        except Exception:
            logger.exception("Stop watchdog: force_terminate failed; finalizing anyway")
        finally:
            self._finalize_stopped_after_escalation(
                target_proc = target_proc, watched_job_id = watched_job_id
            )

    def _finalize_stopped_after_escalation(
        self,
        target_proc: "Optional[mp.Process]" = None,
        watched_job_id: Optional[str] = None,
    ) -> None:
        """Finalize parent state after a force-terminate so the UI leaves "Stopping..."
        even if the worker is wedged in driver teardown; preserves output_dir on a save so
        the checkpoint is kept, and clears it on a cancel (Stop without saving must not
        offer resume/export). No-ops if a new run already replaced the watched worker, so a
        stale watchdog never marks a fresh run stopped or drops its handle.

        Supersession is checked on both the watched proc and job id: start_training sets
        current_job_id before it installs the new _proc, so a stale watchdog entering that
        startup window still sees the old (dead) handle and is caught by the job-id guard.

        The run's terminal DB state is recorded (create-if-needed + finish by captured id)
        BEFORE _proc is dropped: a wedged worker still reports alive, so the pump never
        reaches its own finalize and would bail on its _proc-is-None guard once the handle
        is gone. While the handle is held is_training_active() stays true, so no new run can
        start and current_job_id stays the watched run for the write. _proc is dropped last,
        re-guarded on target_proc so a run that did replace the worker keeps its handle."""
        with self._lock:
            if target_proc is not None and self._proc is not target_proc:
                return
            if watched_job_id is not None and self.current_job_id != watched_job_id:
                return
            run_id = self.current_job_id
            self._progress.is_training = False
        terminal_payload = self._terminal_finalize_kwargs()
        status = terminal_payload["status"]
        error_message = terminal_payload.get("error_message")
        output_dir = terminal_payload["output_dir"]
        clear_output_dir = terminal_payload["clear_output_dir"]
        resume_blocked = bool(terminal_payload.get("resume_blocked"))
        with self._lock:
            if self.current_job_id != run_id:
                return
            if error_message:
                self._progress.status_message = self._progress.error = error_message
            elif status != "completed":
                self._progress.status_message = "Training stopped."
            # A completed run keeps its message; reaping a wedged worker must not relabel it.
        # Create the row if a start-time create failed; skipped while the pump is mid-create, whose create-
        # then-finalize records the run instead.
        self._ensure_db_run_created()
        with self._provenance_lock:
            with self._lock:
                claim = (
                    bool(run_id)
                    and self.current_job_id == run_id
                    and self._db_run_created
                    and not self._run_finalized
                )
                batch: list = []
                final_step = final_loss = duration = None
                loss_history: list = []
                if clear_output_dir:
                    self._output_dir = self._progress.output_dir = None
                if claim:
                    self._run_finalized = True
                    batch = list(self._metric_buffer)
                    del self._metric_buffer[: len(batch)]
                    final_step = self._progress.step
                    final_loss = self._progress.loss
                    if final_loss is not None and not math.isfinite(final_loss):
                        final_loss = None
                    duration = self._progress.elapsed_seconds
                    loss_history = list(self.loss_history)
                    config_json = (
                        _json.dumps(_sanitize_db_config(self._db_config))
                        if self._db_config is not None
                        else None
                    )
                else:
                    config_json = None
            if claim:
                self._finish_stopped_run(
                    run_id,
                    output_dir,
                    batch,
                    final_step,
                    final_loss,
                    duration,
                    loss_history,
                    status = status,
                    error_message = error_message,
                    clear_output_dir = clear_output_dir,
                    resume_blocked = resume_blocked,
                    config_json = config_json,
                )
        with self._lock:
            if target_proc is None or self._proc is target_proc:
                self._proc = None  # drop only our handle, never a run that replaced it

    def _finish_stopped_run(
        self,
        run_id: str,
        output_dir: Optional[str],
        batch: list,
        final_step: Optional[int],
        final_loss: Optional[float],
        duration: Optional[float],
        loss_history: list,
        status: str = "stopped",
        error_message: Optional[str] = None,
        clear_output_dir: bool = False,
        resume_blocked: bool = False,
        config_json: Optional[str] = None,
    ) -> None:
        """Record a force-stopped run finished by its captured id, from state snapshotted
        under the lock. insert_metrics_batch upserts and finish_run is an idempotent UPDATE,
        so a concurrent pump finalize of the same run is harmless and a different current run
        is never touched. The watchdog is the sole finalizer once _proc is dropped, so a
        transient DB error (e.g. a SQLite lock) is retried a few times; on final failure the
        finalize is unclaimed (only if the run is still current) so the row is not left
        claimed-but-unfinalized."""
        for attempt in range(_DB_FINALIZE_RETRIES):
            try:
                from storage.studio_db import finish_run, insert_metrics_batch
                from utils.downsample import downsample

                if batch:
                    insert_metrics_batch(run_id, batch)
                sparkline = downsample(loss_history, 50)
                finish_run(
                    id = run_id,
                    status = status,
                    ended_at = datetime.now(timezone.utc).isoformat(),
                    final_step = final_step,
                    final_loss = final_loss,
                    duration_seconds = duration,
                    loss_sparkline = _json.dumps(sparkline),
                    output_dir = output_dir,
                    error_message = error_message,
                    clear_output_dir = clear_output_dir,
                    resume_blocked = resume_blocked,
                    config_json = config_json,
                )
                return
            except Exception:
                if attempt + 1 < _DB_FINALIZE_RETRIES:
                    time.sleep(_DB_FINALIZE_RETRY_S)
                    continue
                logger.warning(
                    "Failed to finalize stopped run %s in DB after %d attempts",
                    run_id,
                    _DB_FINALIZE_RETRIES,
                    exc_info = True,
                )
                with self._lock:
                    # Only if still current; a new run's finalize state is never touched.
                    if self.current_job_id == run_id:
                        self._run_finalized = False

    def force_terminate(self, target_proc: "Optional[mp.Process]" = None) -> None:
        """Force-kill the training subprocess so state can be reset immediately. With
        ``target_proc``, terminate only that handle and no-op if a new run has replaced
        it, so the watchdog can never kill a fresh worker."""
        with self._lock:
            proc = self._proc
            if target_proc is not None and proc is not target_proc:
                return
            if proc is not None and proc.is_alive():
                logger.info("Force-terminating training subprocess (pid=%s)", proc.pid)
                proc.terminate()
            cancelled = self._cancel_requested
            output_dir = self._cancel_cleanup_output_dir or self._output_dir

        if proc is not None:
            proc.join(timeout = 5.0)
            if proc.is_alive():
                proc.kill()
                proc.join(timeout = 2.0)

        # Wait for pump thread to finish DB finalization (8s covers SQLite's 5s lock timeout).
        if self._pump_thread is not None and self._pump_thread.is_alive():
            self._pump_thread.join(timeout = 8.0)

        if cancelled and output_dir:
            try:
                _cleanup_cancelled_checkpoints(output_dir)
            except Exception:
                logger.exception(
                    "Failed to clean up cancelled-run checkpoints under %s",
                    output_dir,
                )

    def _handle_stall_event(self, event: dict) -> None:
        """A worker reported a no-progress download stall.

        On the first model-load, terminate the worker so the pump loop respawns it
        over HTTP. A later stall (already on HTTP, or outside model-load) surfaces
        as an error instead.
        """
        msg = event.get("message", "Download stalled")
        with self._lock:
            recover = self._in_model_load and not self._xet_fallback_used
            proc = self._proc
            run_id = self.current_job_id
            if recover:
                self._xet_fallback_used = True
                self._needs_xet_respawn = True
                self._progress.status_message = (
                    "Model download stalled on Xet; retrying over HTTP..."
                )
            else:
                self._progress.error = self._progress.error or (
                    "Model download stalled even over HTTP -- check your network connection"
                )
        if recover:
            logger.warning("Training model-load stalled on Xet; respawning over HTTP: %s", msg)
        else:
            logger.error("Training download stalled with no further fallback: %s", msg)
        # Terminate either way so the pump loop proceeds (respawn or finalize).
        if proc is not None and proc.is_alive():
            proc.terminate()
        if not recover:
            # terminate() is only a request, so arm the same backstop as the other terminal paths. Signal first,
            # since arming no-ops when a watchdog already watches this proc.
            self._complete_seen.set()
            self._start_stop_watchdog(
                cancel = False,
                expected_job_id = run_id,
                grace_s = _COMPLETE_EXIT_GRACE_S,
                terminal_seen = True,
            )

    def _respawn_worker_disable_xet(self, expected_job_id: Optional[str] = None) -> bool:
        """Respawn the worker once with HF_HUB_DISABLE_XET=1 after a model-load
        stall. Runs on the exiting pump thread, reaps the terminated worker, and
        starts a fresh worker + pump. DB/progress run-state is preserved so the
        history row is not duplicated; the new worker re-formats and loads over HTTP.
        """
        from .lifecycle import training_lifecycle_guard

        with training_lifecycle_guard():
            with self._lock:
                if expected_job_id is not None and self.current_job_id != expected_job_id:
                    return False
                if self._should_stop or self._cancel_requested:
                    return False
                reservation_job_id = self.current_job_id
                config = self._last_full_config
                old_proc = self._proc
                self._spawn_in_progress = True

        def release_spawn_reservation() -> None:
            with self._lock:
                if self.current_job_id == reservation_job_id:
                    self._spawn_in_progress = False

        try:
            if config is None:
                logger.error("Cannot respawn training worker: no stored config")
                release_spawn_reservation()
                return False

            if old_proc is not None:
                old_proc.join(timeout = 5.0)
                if old_proc.is_alive():
                    old_proc.kill()
                    old_proc.join(timeout = 2.0)

            config = {**config, "disable_xet": True}
            logger.warning("Respawning training worker with HF_HUB_DISABLE_XET=1 after Xet stall")

            cache_env = getattr(self, "_last_hf_cache_env", None)
            if not cache_env:
                from utils.hf_cache_settings import get_hf_cache_paths
                cache_env = get_hf_cache_paths().child_env({})
            from utils.hf_cache_settings import child_environment_for_spawn
            from utils.transformers_version import sidecar_swap_in_progress

            swap_wait_deadline = time.time() + 120
            while time.time() < swap_wait_deadline:
                with self._lock:
                    superseded = self.current_job_id != reservation_job_id
                    stopping = self._should_stop or self._cancel_requested
                if superseded or stopping or not sidecar_swap_in_progress():
                    break
                time.sleep(0.25)
        except Exception:
            release_spawn_reservation()
            raise

        with training_lifecycle_guard():
            with self._lock:
                if self.current_job_id != reservation_job_id:
                    return False
                if self._should_stop or self._cancel_requested:
                    self._spawn_in_progress = False
                    if self._cancel_requested:
                        self._should_stop = True
                    return False

            try:
                swap_in_progress = sidecar_swap_in_progress()
            except Exception:
                release_spawn_reservation()
                raise
            if swap_in_progress:
                release_spawn_reservation()
                msg = (
                    "A transformers installation is replacing the latest sidecar; "
                    "cannot respawn the training worker."
                )
                logger.error(msg)
                with self._lock:
                    self._progress.is_training = False
                    self._progress.error = msg
                self._ensure_db_run_created()
                self._finalize_run_in_db(status = "error", error_message = msg)
                return False

            with self._lock:
                self._last_full_config = config

            # Reset the handshake flag on any failure past this point, else is_training_active wedges.
            try:
                try:
                    with (
                        child_environment_for_spawn(cache_env),
                        native_path_secret_removed_for_child_start(),
                    ):
                        event_queue = _CTX.Queue()
                        stop_queue = _CTX.Queue()
                        new_proc = _CTX.Process(
                            target = run_without_native_path_secret,
                            args = ("core.training.worker", "run_training_process", cache_env),
                            kwargs = {
                                "event_queue": event_queue,
                                "stop_queue": stop_queue,
                                "config": config,
                            },
                            daemon = True,
                        )
                        new_proc.start()
                        from utils.process_lifetime import adopt_pid

                        adopt_pid(new_proc.pid)
                except Exception:
                    logger.error("Failed to respawn training subprocess", exc_info = True)
                    self._spawn_in_progress = False
                    with self._lock:
                        # No replacement pump will run; clear the flag so a later run can't inherit it.
                        self._pump_running = False
                        self._progress.is_training = False
                        self._progress.error = "Failed to recover stalled model download"
                    self._ensure_db_run_created()
                    self._finalize_run_in_db(
                        status = "error",
                        error_message = "Failed to recover stalled model download",
                    )
                    return False

                logger.info(
                    "Training subprocess respawned with Xet disabled (pid=%s)", new_proc.pid
                )
                new_pump = threading.Thread(target = self._pump_loop, daemon = True)
                with self._lock:
                    self._in_model_load = False
                    self._event_queue = event_queue
                    self._stop_queue = stop_queue
                    self._proc = new_proc
                    self._spawn_in_progress = False
                    self._pump_thread = new_pump
                    # Start under the lock so _ensure_pump_alive can't see the new pump as dead and duplicate it.
                    new_pump.start()
                return True
            except Exception:
                release_spawn_reservation()
                raise

    def _ensure_pump_alive(self) -> bool:
        """Restart the event pump if it crashed, even after the worker exited.

        Defence in depth behind _pump_loop's guards. _pump_running stays True only
        after an abnormal exit (the loop clears it on intended exits), so a True
        flag plus a dead thread is an unambiguous crash. Restarts even after worker
        exit so a fresh pump can drain the terminal events and finalize; otherwise
        the run looks stuck "running" forever. Returns True if restarted.
        """
        with self._lock:
            if not self._pump_running:
                return False
            # A restarted pump needs the worker handle and queue; their absence means nothing to recover.
            if self._proc is None or self._event_queue is None:
                return False
            if self._pump_thread is not None and self._pump_thread.is_alive():
                return False
            logger.error(
                "Training event pump thread died while the worker is still running; "
                "restarting it so progress updates resume."
            )
            new_pump = threading.Thread(target = self._pump_loop, daemon = True)
            self._pump_thread = new_pump
            new_pump.start()
        return True

    def _run_finished_locked(self) -> bool:
        """is_run_finished()'s terminal test, for callers already holding _lock."""
        if self._spawn_in_progress or self._new_job_spawn_id is not None:
            return False
        p = self._progress
        return bool(self._complete_seen.is_set() or p.is_completed or p.error)

    def is_run_finished(self) -> bool:
        """Whether the current run reached its own terminal state (saved and finalized).

        is_training_active() is liveness-based, so it stays true until the worker exits, which
        can lag minutes behind a slow teardown or never happen at all, leaving the UI at 100%.
        Status and progress read this so a finished run reports terminal at once; the GPU
        admission guards keep using is_training_active(), since a lingering worker holds VRAM."""
        if getattr(self, "_spawn_in_progress", False):
            return False
        with self._lock:
            return self._run_finished_locked()

    def is_training_active(self) -> bool:
        """Check if training is currently active."""
        # A spawn past its sidecar-swap recheck counts as active even before _proc is recorded.
        if getattr(self, "_new_job_spawn_id", None) is not None or getattr(
            self, "_spawn_in_progress", False
        ):
            return True
        # Self-heal a crashed pump first: a dead pump would leave the worker training invisibly.
        self._ensure_pump_alive()
        with self._lock:
            if self._proc is not None and self._proc.is_alive():
                return True

            if self._should_stop:
                return False

            p = self._progress
            if p.is_training:
                return True
            if p.is_completed or p.error:
                return False

            # Infer activity from the status message.
            status_lower = (p.status_message or "").lower()
            if any(
                k in status_lower
                for k in [
                    "cancelled",
                    "canceled",
                    "stopped",
                    "completed",
                    "ready to train",
                ]
            ):
                return False
            if any(
                k in status_lower
                for k in [
                    "loading",
                    "preparing",
                    "training",
                    "configuring",
                    "tokenizing",
                    "starting",
                    "importing",
                ]
            ):
                return True

            return False

    def active_output_dir(self) -> Optional[str]:
        if not self.is_training_active():
            return None
        with self._lock:
            config = self._db_config or {}
            output_dir = (
                self._output_dir or self._cancel_cleanup_output_dir or config.get("output_dir")
            )
            resume_from_checkpoint = config.get("resume_from_checkpoint")
        if not output_dir:
            from .worker import _output_dir_from_resume_checkpoint
            output_dir = _output_dir_from_resume_checkpoint(resume_from_checkpoint)
        return str(output_dir) if output_dir else None

    def get_training_status(self, theme: str = "light") -> Tuple:
        """Get current training status and loss plot."""
        with self._lock:
            progress = self._progress

        if not (progress.is_training or progress.is_completed or progress.error):
            return (None, progress)

        plot = self._create_loss_plot(progress, theme)
        return (plot, progress)

    def refresh_plot_for_theme(self, theme: str) -> "Optional[plt.Figure]":
        """Refresh plot with new theme."""
        if theme and isinstance(theme, str) and theme in ["light", "dark"]:
            self.current_theme = theme
        if self.loss_history:
            with self._lock:
                progress = self._progress
            return self._create_loss_plot(progress, self.current_theme)
        return None


    class _TrainerShim:
        """Minimal shim so routes that access backend.trainer.* still work."""

        def __init__(self, backend: "TrainingBackend"):
            self._backend = backend
            self.should_stop = False

        @property
        def training_progress(self):
            return self._backend._progress

        @training_progress.setter
        def training_progress(self, value):
            self._backend._progress = value

        def get_training_progress(self):
            return self._backend._progress

        def _update_progress(self, **kwargs):
            with self._backend._lock:
                for key, value in kwargs.items():
                    if hasattr(self._backend._progress, key):
                        setattr(self._backend._progress, key, value)

    @property
    def trainer(self):
        """Compatibility shim for routes that access backend.trainer.*"""
        return self._TrainerShim(self)


    def _safe_handle_event(self, event: dict) -> None:
        """Apply one event, swallowing any handler error.

        The pump is the only writer of the progress state every status surface
        reads, so a malformed event must never propagate and kill it.
        """
        try:
            self._handle_event(event)
        except Exception:
            etype = event.get("type") if isinstance(event, dict) else type(event).__name__
            logger.exception("Training event pump: failed to handle %s event; skipping", etype)

    def _pump_loop(self) -> None:
        """Background thread: consume subprocess events and update state.

        Sole writer of the in-memory progress state that /progress, /status,
        /metrics and DB history read. If it exited while the worker still ran, the
        run would burn GPU with events piling up while every surface froze. So no
        single bad event or transient queue/DB error may end it; it returns only
        through intended exits (worker gone, respawn handed off, finalized).
        """
        self._pump_running = True
        while True:
            if self._proc is None or self._event_queue is None:
                self._pump_running = False
                return

            try:
                event = self._read_queue(self._event_queue, timeout_sec = 0.25)
            except Exception:
                # If a read keeps raising after the worker died, finalize instead of spinning.
                logger.exception("Training event pump: queue read failed; continuing")
                if self._proc is not None and self._proc.is_alive():
                    time.sleep(0.1)
                    continue
                event = None

            if event is not None:
                self._safe_handle_event(event)
                continue

            # Snapshot: the watchdog drops _proc last, so a re-read can hit None and kill this thread; a dropped
            # handle means it already finalized.
            proc = self._proc
            if proc is None:
                self._pump_running = False
                return
            if proc.is_alive():
                continue

            # Worker exited. Drain the backlog and finalize, guarded so a failing DB write can't strand.
            try:
                for e in self._drain_queue(self._event_queue):
                    self._safe_handle_event(e)

                # Model-load stall: respawn over HTTP instead of finalizing as failure. The fresh pump takes over
                # _pump_running, so this exit leaves it set.
                with self._lock:
                    needs_xet_respawn = self._needs_xet_respawn
                    self._needs_xet_respawn = False
                    respawn_job_id = self.current_job_id
                if needs_xet_respawn and self._respawn_worker_disable_xet(
                    expected_job_id = respawn_job_id
                ):
                    return

                with self._lock:
                    if self._progress.is_training:
                        if self._should_stop:
                            self._progress.is_training = False
                            self._progress.status_message = "Training stopped."
                        else:
                            self._progress.is_training = False
                            self._progress.error = (
                                self._progress.error or "Training process exited unexpectedly"
                            )

                self._ensure_db_run_created()
                terminal_payload = self._terminal_finalize_kwargs()
                with self._lock:
                    if terminal_payload["clear_output_dir"]:
                        self._output_dir = self._progress.output_dir = None
                    if terminal_payload.get("error_message"):
                        self._progress.error = terminal_payload["error_message"]
                        self._progress.status_message = terminal_payload["error_message"]
                self._finalize_run_in_db(**terminal_payload)
            except Exception:
                logger.exception("Training event pump: finalization after worker exit failed")
            self._pump_running = False
            return

    def _has_current_resume_checkpoint(self, output_dir, step) -> bool:
        # A valid checkpoint at the current step means the stop-and-save landed on disk even if the worker
        # died before confirming it.
        if not output_dir or not isinstance(step, int) or step <= 0:
            return False
        from core.training.resume import get_resume_checkpoint_path
        return get_resume_checkpoint_path(output_dir, expected_step = step) is not None

    def _terminal_finalize_kwargs(self) -> dict:
        with self._lock:
            job_id = self.current_job_id
            payload = self._terminal_finalize_payload
            if payload and payload.get("expected_job_id") == job_id:
                return dict(payload)
            cancel, stopped = self._cancel_requested, self._should_stop
            output_dir = None if cancel else self._output_dir
            step = self._progress.step
            existing_error = self._progress.error
        status, error, blocked = (
            ("stopped", None, cancel)
            if stopped
            else (
                "error",
                existing_error or "Training process terminated unexpectedly",
                False,
            )
        )
        # Block only when no valid current-step checkpoint actually landed.
        if stopped and not cancel and not self._has_current_resume_checkpoint(output_dir, step):
            status = "error"
            error = "Stop and Save ended before a valid current-step checkpoint was written."
            blocked = True
        return {
            "status": status,
            "error_message": error,
            "output_dir": output_dir,
            "clear_output_dir": cancel,
            "resume_blocked": blocked,
            "expected_job_id": job_id,
        }

    def _handle_resource_provenance_event(self, event: dict[str, Any]) -> None:
        from .provenance import normalize_worker_provenance_event
        with self._provenance_lock:
            with self._lock:
                if not self.current_job_id or self._db_config is None or self._run_finalized:
                    return
                run_id = self.current_job_id
                current_config = dict(self._db_config)

            updates = normalize_worker_provenance_event(event, current_config)
            with self._lock:
                if self.current_job_id != run_id or self._db_config is None or self._run_finalized:
                    return
                self._db_config.update(updates)
                if self._last_full_config is not None:
                    self._last_full_config.update(updates)
                config_json = _json.dumps(_sanitize_db_config(self._db_config))
                db_run_created = self._db_run_created

            if not db_run_created:
                return
            for attempt in range(_DB_FINALIZE_RETRIES):
                try:
                    from storage.studio_db import update_run_config_json
                    if not update_run_config_json(run_id, config_json):
                        logger.warning(
                            "Training provenance was not persisted because run %s is no longer active",
                            run_id,
                        )
                    return
                except Exception:
                    if attempt + 1 < _DB_FINALIZE_RETRIES:
                        time.sleep(_DB_FINALIZE_RETRY_S)
                        continue
                    logger.warning(
                        "Failed to persist training resource provenance for run %s",
                        run_id,
                        exc_info = True,
                    )

    def _handle_event(self, event: dict) -> None:
        """Apply a subprocess event to local state.

        State updates happen inside self._lock; DB I/O happens after releasing
        it so status-polling endpoints aren't blocked by slow SQLite writes.
        """
        etype = event.get("type")
        db_action: Optional[str] = None
        db_action_kwargs: dict = {}

        if etype == "resource_provenance":
            self._handle_resource_provenance_event(event)
            return

        # Model-load lifecycle + stall recovery (no DB metrics); handled first.
        if etype == "model_load_started":
            with self._lock:
                self._in_model_load = True
            return
        if etype == "model_load_completed":
            with self._lock:
                self._in_model_load = False
            return
        if etype == "stall":
            self._handle_stall_event(event)
            return

        with self._lock:
            if etype == "progress":
                self._progress.step = event.get("step", self._progress.step)
                self._progress.epoch = event.get("epoch", self._progress.epoch)
                # loss/lr sanitized below.
                _raw_loss = event.get("loss")
                _raw_lr = event.get("learning_rate")
                try:
                    _safe_loss = float(_raw_loss) if _raw_loss is not None else None
                except (TypeError, ValueError):
                    logger.debug("Could not convert loss to float: %s", _raw_loss)
                    _safe_loss = None
                _loss_is_nonfinite = _safe_loss is not None and not math.isfinite(_safe_loss)
                if _loss_is_nonfinite:
                    # Drop the value rather than laundering it back to the last finite loss: clients see loss=None at
                    # this step so the NaN is not hidden.
                    _safe_loss = None
                    if not getattr(self._progress, "_nonfinite_loss_warned", False):
                        self._progress._nonfinite_loss_warned = True
                        logger.warning(
                            "Training produced non-finite loss at step %s; "
                            "loss field will report null until it recovers.",
                            event.get("step", "?"),
                        )
                try:
                    _safe_lr = float(_raw_lr) if _raw_lr is not None else None
                except (TypeError, ValueError):
                    logger.debug("Could not convert learning_rate to float: %s", _raw_lr)
                    _safe_lr = None
                if _safe_lr is not None and not math.isfinite(_safe_lr):
                    _safe_lr = None
                if _safe_loss is not None:
                    self._progress.loss = _safe_loss
                elif _loss_is_nonfinite:
                    # Clear stale finite loss so the API doesn't keep reporting the last good value during NaN.
                    self._progress.loss = None
                if _safe_lr is not None:
                    self._progress.learning_rate = _safe_lr
                self._progress.total_steps = event.get("total_steps", self._progress.total_steps)
                self._progress.elapsed_seconds = event.get("elapsed_seconds")
                self._progress.eta_seconds = event.get("eta_seconds")
                self._progress.grad_norm = event.get("grad_norm")
                self._progress.num_tokens = event.get("num_tokens")
                self._progress.eval_loss = event.get("eval_loss")
                _peak = event.get("peak_memory_gb")
                if _peak is not None:
                    try:
                        self._progress.peak_memory_gb = float(_peak)
                    except (TypeError, ValueError):
                        pass
                self._progress.is_training = True
                status = event.get("status_message", "")
                if status:
                    self._progress.status_message = status

                step = event.get("step", 0)
                loss = _safe_loss
                lr = _safe_lr
                # Only ever move forward: HF can log more than one record at the same global_step around the end of
                # a run, so a 30-step run charted 33 points.
                _last_step = self.step_history[-1] if self.step_history else None
                if step > 0 and loss is not None and (_last_step is None or step > _last_step):
                    self.loss_history.append(loss)
                    self.lr_history.append(lr if lr is not None else 0.0)
                    self.step_history.append(step)

                grad_norm = event.get("grad_norm")
                gn = None
                if grad_norm is not None:
                    try:
                        gn = float(grad_norm)
                    except (TypeError, ValueError):
                        gn = None
                    if step > 0 and gn is not None and math.isfinite(gn):
                        self.grad_norm_history.append(gn)
                        self.grad_norm_step_history.append(step)
                    else:
                        gn = None

                eval_loss = event.get("eval_loss")
                if eval_loss is not None:
                    try:
                        eval_loss = float(eval_loss)
                    except (TypeError, ValueError):
                        logger.debug("Could not convert eval_loss to float: %s", eval_loss)
                        eval_loss = None
                    if step > 0 and eval_loss is not None and math.isfinite(eval_loss):
                        self.eval_loss_history.append(eval_loss)
                        self.eval_step_history.append(step)
                        self.eval_enabled = True
                    else:
                        eval_loss = None

                self._metric_buffer.append(
                    {
                        "step": step,
                        "loss": loss,
                        "learning_rate": lr,
                        "grad_norm": gn,
                        "eval_loss": eval_loss,
                        "epoch": event.get("epoch"),
                        "num_tokens": event.get("num_tokens"),
                        "elapsed_seconds": event.get("elapsed_seconds"),
                    }
                )

                # Pick the DB action to run after releasing the lock.
                if not self._db_run_created and self.current_job_id and self._db_config:
                    db_action = "create_run"
                    db_action_kwargs = {
                        "job_id": self.current_job_id,
                        "model_name": self._db_config["model_name"],
                        "dataset_name": self._db_config.get("hf_dataset")
                        or next(iter(self._db_config.get("local_datasets") or []), "unknown"),
                        "config_json": _json.dumps(self._db_config),
                        "started_at": self._db_started_at or datetime.now(timezone.utc).isoformat(),
                        "total_steps": event.get("total_steps"),
                    }
                elif (
                    event.get("total_steps")
                    and self._db_run_created
                    and not self._db_total_steps_set
                ):
                    db_action = "update_total_steps"
                    db_action_kwargs = {
                        "job_id": self.current_job_id,
                        "total_steps": event["total_steps"],
                    }
                elif len(self._metric_buffer) >= self.FLUSH_THRESHOLD:
                    db_action = "flush"

            elif etype == "eval_configured":
                self.eval_enabled = True

            elif etype == "output_dir":
                event_output_dir = event.get("output_dir")
                if self._cancel_requested:
                    self._cancel_cleanup_output_dir = event_output_dir
                    self._output_dir = self._progress.output_dir = None
                else:
                    self._output_dir = event_output_dir
                    db_action = "persist_output_dir"

            elif etype == "status":
                self._progress.status_message = event.get("message", "")
                self._progress.is_training = True

            elif etype == "warning":
                message = event.get("message")
                if isinstance(message, str):
                    message = message.strip()
                    if message and message not in self._progress.warnings:
                        self._progress.warnings.append(message)
                        logger.warning("Training warning: %s", message)

            elif etype == "complete":
                msg = event.get("status_message", "Training completed")
                stopped = self._should_stop or msg.strip().lower() in {
                    "training cancelled",
                    "training stopped",
                }
                # Nothing left to save: drop an in-flight watchdog to its grace, not the save backstop.
                self._complete_seen.set()
                self._progress.is_training = False
                self._progress.is_completed = not stopped
                event_output_dir = event.get("output_dir")
                if self._cancel_requested:
                    self._cancel_cleanup_output_dir = event_output_dir
                    self._output_dir = None
                else:
                    self._output_dir = event_output_dir
                self._progress.output_dir = self._output_dir
                self._progress.status_message = msg
                if not self._db_run_created and self.current_job_id and self._db_config:
                    db_action = "create_and_finalize"
                else:
                    db_action = "finalize"
                db_action_kwargs = {
                    "status": "stopped" if stopped else "completed",
                    "output_dir": self._output_dir,
                    "clear_output_dir": self._cancel_requested,
                    "expected_job_id": self.current_job_id,
                }
                self._terminal_finalize_payload = dict(db_action_kwargs)

            elif etype == "error":
                self._progress.is_training = False
                self._progress.error = event.get("error", "Unknown error")
                # Nothing left to save: drop an in-flight watchdog to its grace, not the save backstop.
                self._complete_seen.set()
                if self._cancel_requested:
                    self._output_dir = self._progress.output_dir = None
                logger.error("Training error: %s", event.get("error"))
                stack = event.get("stack", "")
                if stack:
                    logger.error("Stack trace:\n%s", stack)
                if not self._db_run_created and self.current_job_id and self._db_config:
                    db_action = "create_and_finalize"
                else:
                    db_action = "finalize"
                stop_save_failed = (
                    self._should_stop
                    and not self._cancel_requested
                    and not self._has_current_resume_checkpoint(
                        self._output_dir, self._progress.step
                    )
                )
                db_action_kwargs = {
                    "status": "stopped"
                    if self._should_stop
                    and not stop_save_failed
                    and not event.get("keep_error_status")
                    else "error",
                    "error_message": event.get("error", "Unknown error"),
                    "output_dir": self._output_dir,
                    "clear_output_dir": self._cancel_requested,
                    "resume_blocked": stop_save_failed or bool(event.get("resume_blocked")),
                    "expected_job_id": self.current_job_id,
                }
                self._terminal_finalize_payload = dict(db_action_kwargs)

        if db_action == "create_run":
            self._ensure_db_run_created()
            if self._db_run_created:
                if db_action_kwargs["total_steps"]:
                    self._db_total_steps_set = True
                self._persist_output_dir()
        elif db_action == "persist_output_dir":
            self._persist_output_dir()
        elif db_action == "create_and_finalize":
            self._ensure_db_run_created()
            self._finalize_run_in_db(**db_action_kwargs)
        elif db_action == "update_total_steps":
            try:
                from storage.studio_db import update_run_total_steps
                update_run_total_steps(db_action_kwargs["job_id"], db_action_kwargs["total_steps"])
                self._db_total_steps_set = True
            except Exception:
                logger.warning("Failed to update total_steps in DB", exc_info = True)
        elif db_action == "flush":
            self._flush_metrics_to_db()
        elif db_action == "finalize":
            self._finalize_run_in_db(**db_action_kwargs)

        # Bound how long a worker that will not exit can hold the UI at 100%. Outside the lock:
        # _start_stop_watchdog takes it and it is not reentrant.
        if etype in ("complete", "error"):
            self._start_stop_watchdog(
                cancel = False,
                expected_job_id = db_action_kwargs.get("expected_job_id"),
                grace_s = _COMPLETE_EXIT_GRACE_S,
                terminal_seen = True,
            )

        if etype == "progress":
            self._log_training_progress()

    def _persist_output_dir(self) -> None:
        # Re-queue the claimed batch at the front so it retries on the next flush.
        with self._lock:
            if (
                not self._output_dir
                or not self.current_job_id
                or not self._db_run_created
                or self._cancel_requested
            ):
                return
            run_id, output_dir = self.current_job_id, self._output_dir
        try:
            from storage.studio_db import update_run_output_dir
            update_run_output_dir(run_id, output_dir)
        except Exception:
            logger.warning("Failed to persist output_dir", exc_info = True)

    def _log_training_progress(self) -> None:
        """One throttled training-status line to the server log (the per-step stream
        still goes to the UI via SSE): first step, then at most every 30s, plus the
        final step; resyncs on a new run. Runs on the pump thread."""
        p = self._progress
        step = int(p.step or 0)
        if step <= 0:
            return
        total = int(p.total_steps or 0)
        is_final = total > 0 and step >= total
        prev = self._last_progress_log_step
        if step == prev:
            return
        now = time.monotonic()
        if prev >= 0 and step > prev and not is_final and (now - self._last_progress_log_ts) < 30.0:
            return
        # Throughput over the interval since the previous logged line: it used to appear only in HF's own
        # tqdm bar ("1.84s/it") and its per-step train_tokens_per_second print, both raw stdout rather
        # than structured.
        elapsed = p.elapsed_seconds
        tokens = p.num_tokens
        s_per_step = tok_per_s = None
        prev_elapsed = self._last_progress_log_elapsed
        prev_tokens = self._last_progress_log_tokens
        if elapsed is not None and prev_elapsed is not None and prev >= 0:
            d_time = elapsed - prev_elapsed
            d_steps = step - prev
            if d_time > 0 and d_steps > 0:
                s_per_step = round(d_time / d_steps, 3)
                if tokens is not None and prev_tokens is not None and tokens > prev_tokens:
                    tok_per_s = round((tokens - prev_tokens) / d_time, 1)
        # The first logged line reports no throughput on purpose: elapsed_seconds is wall time since the
        # worker started (imports, download, load, dataset build), and a resumed run's counters are older.

        self._last_progress_log_ts = now
        self._last_progress_log_step = step
        self._last_progress_log_elapsed = elapsed
        self._last_progress_log_tokens = tokens
        logger.info(
            "training_progress",
            step = step,
            total_steps = total or None,
            percent = int(step * 100 / total) if total > 0 else None,
            loss = round(p.loss, 4) if p.loss is not None else None,
            epoch = round(p.epoch, 2) if p.epoch is not None else None,
            eta_s = int(p.eta_seconds) if p.eta_seconds else None,
            s_per_step = s_per_step,
            tok_per_s = tok_per_s,
        )

    def _ensure_db_run_created(self) -> None:
        """Create the DB row if it doesn't exist yet. An in-progress flag lets only one
        caller create at a time, and ``_db_run_created`` is published only after
        ``create_run`` commits, so a concurrent finalize never runs ``finish_run`` against a
        not-yet-inserted row (a zero-row UPDATE that would leave the run stuck as running)."""
        self._run_intent_lock.acquire()
        with self._lock:
            if (
                self._db_run_created
                or self._db_create_in_progress
                or not self.current_job_id
                or not self._db_config
            ):
                self._run_intent_lock.release()
                return
            self._db_create_in_progress = True
            job_id = self.current_job_id
            db_config = self._db_config
            started_at = self._db_started_at or datetime.now(timezone.utc).isoformat()
            total_steps = self._progress.total_steps or None
        created = False
        try:
            from storage.studio_db import create_run

            dataset_name = (
                db_config.get("hf_dataset")
                or next(iter(db_config.get("local_datasets") or []), None)
                or _s3_dataset_name(db_config.get("s3_dataset"))
                or "unknown"
            )
            with self._lock:
                if self.current_job_id != job_id:
                    return
                output_dir = self._output_dir
                cancel_requested = self._cancel_requested
                resumed_from_run_id = self._resume_source_run_id
            create_run(
                id = job_id,
                model_name = db_config["model_name"],
                dataset_name = dataset_name,
                config_json = _json.dumps(db_config),
                started_at = started_at,
                total_steps = total_steps,
                output_dir = output_dir,
                cancel_requested = cancel_requested,
                resumed_from_run_id = resumed_from_run_id,
            )
            created = True
        except Exception:
            logger.warning("Failed to create DB run record for early failure", exc_info = True)
        finally:
            with self._lock:
                # Publish the flags only if this is still the current run: they are backend-wide, and a killed
                # worker lets a new /start proceed mid-create.
                if self.current_job_id == job_id:
                    if created:
                        self._db_run_created = True  # publish only after the insert commits
                    self._db_create_in_progress = False
            self._run_intent_lock.release()

    def _finalize_run_in_db(
        self,
        status: str,
        error_message: Optional[str] = None,
        output_dir: Optional[str] = None,
        clear_output_dir: bool = False,
        resume_blocked: bool = False,
        expected_job_id: Optional[str] = None,
    ) -> None:
        """Flush remaining metrics and mark a run finished in the DB. Claims the finalize
        under the lock so the watchdog and pump can't double-finalize, and no-ops when
        ``expected_job_id`` no longer matches (a new run took over). The run id and final
        progress are snapshotted under the lock and threaded through the flush/finish calls,
        so a new run racing between this claim and the DB writes can't be flushed or marked
        stopped under the old run's finalize."""
        with self._provenance_lock:
            with self._lock:
                if expected_job_id is not None and self.current_job_id != expected_job_id:
                    return
                if not self.current_job_id or not self._db_run_created or self._run_finalized:
                    return
                self._run_finalized = True
                run_id = self.current_job_id
                final_step = self._progress.step
                final_loss = self._progress.loss
                if final_loss is not None and not math.isfinite(final_loss):
                    final_loss = None
                duration = self._progress.elapsed_seconds
                loss_history = list(self.loss_history)
                config_json = (
                    _json.dumps(_sanitize_db_config(self._db_config))
                    if self._db_config is not None
                    else None
                )
            self._flush_metrics_to_db(run_id = run_id)
            for attempt in range(_DB_FINALIZE_RETRIES):
                try:
                    from storage.studio_db import finish_run
                    from utils.downsample import downsample

                    finish_run(
                        id = run_id,
                        status = status,
                        ended_at = datetime.now(timezone.utc).isoformat(),
                        final_step = final_step,
                        final_loss = final_loss,
                        duration_seconds = duration,
                        loss_sparkline = _json.dumps(downsample(loss_history, 50)),
                        output_dir = output_dir,
                        error_message = error_message,
                        clear_output_dir = clear_output_dir,
                        resume_blocked = resume_blocked,
                        config_json = config_json,
                    )
                    return
                except Exception:
                    if attempt + 1 < _DB_FINALIZE_RETRIES:
                        time.sleep(_DB_FINALIZE_RETRY_S)
                        continue
                    with self._lock:
                        if self.current_job_id == run_id:
                            self._run_finalized = False
                    logger.warning(
                        "Failed to finalize run in DB (status=%s)",
                        status,
                        exc_info = True,
                    )

    def _flush_metrics_to_db(self, run_id: Optional[str] = None) -> None:
        """Flush buffered metrics to the DB and update live progress. The target run id,
        metric batch, and progress snapshot are all taken under the lock, so a concurrent
        flush can't double-remove metrics and a racing new run can't redirect the write to
        a different job. A finalizer passes ``run_id`` to pin the target to its captured run."""
        with self._lock:
            target = run_id if run_id is not None else self.current_job_id
            if not self._metric_buffer or not target or not self._db_run_created:
                return
            # Cap buffer to bound memory growth.
            if len(self._metric_buffer) > 500:
                logger.warning(
                    "Metric buffer exceeded 500 entries (%d) — trimming oldest",
                    len(self._metric_buffer),
                )
                del self._metric_buffer[:-500]
            # Claim the batch under the lock so a concurrent flush can't re-remove it.
            batch = list(self._metric_buffer)
            del self._metric_buffer[: len(batch)]
            step = self._progress.step
            loss = self._progress.loss
            if loss is not None and not math.isfinite(loss):
                loss = None
            duration = self._progress.elapsed_seconds
        try:
            from storage.studio_db import insert_metrics_batch, update_run_progress
            insert_metrics_batch(target, batch)
            update_run_progress(id = target, step = step, loss = loss, duration_seconds = duration)
        except Exception:
            # Re-queue the claimed batch at the front so it retries on the next flush.
            with self._lock:
                self._metric_buffer[:0] = batch
            logger.warning("Failed to flush metrics to DB", exc_info = True)

    @staticmethod
    def _read_queue(q: Any, timeout_sec: float) -> Optional[dict]:
        try:
            return q.get(timeout = timeout_sec)
        except queue.Empty:
            return None
        except (EOFError, OSError, ValueError):
            # A closed/broken queue reads as "no event"; other errors go to _pump_loop's guarded block.
            return None

    @staticmethod
    def _drain_queue(q: Any) -> list:
        events = []
        while True:
            try:
                events.append(q.get_nowait())
            except queue.Empty:
                return events
            except Exception:
                # A drain error must not abort finalization: return what we have so the run finalizes.
                logger.exception(
                    "Training event pump: queue drain failed; finalizing with drained events"
                )
                return events


    def _create_loss_plot(
        self,
        progress: TrainingProgress,
        theme: str = "light",
    ) -> "Optional[plt.Figure]":
        """Create training loss plot with theme-aware styling.

        matplotlib is loaded lazily; returns None if it is unavailable.
        """
        plt = _load_pyplot()
        if plt is None:
            return None
        plt.close("all")

        LIGHT_STYLE = {
            "facecolor": "#ffffff",
            "grid_color": "#d1d5db",
            "line": "#16b88a",
            "text": "#1f2937",
            "empty_text": "#6b7280",
        }
        DARK_STYLE = {
            "facecolor": "#292929",
            "grid_color": "#404040",
            "line": "#4ade80",
            "text": "#e5e7eb",
            "empty_text": "#9ca3af",
        }

        style = LIGHT_STYLE if theme == "light" else DARK_STYLE

        fig, ax = plt.subplots(figsize = (PLOT_WIDTH, PLOT_HEIGHT))
        fig.patch.set_facecolor(style["facecolor"])
        ax.set_facecolor(style["facecolor"])

        if self.loss_history:
            steps = self.step_history
            losses = self.loss_history
            scatter_color = "#60a5fa"
            ax.scatter(
                steps,
                losses,
                s = 16,
                alpha = 0.6,
                color = scatter_color,
                linewidths = 0,
                label = "Training Loss (raw)",
            )

            MA_WINDOW = 20
            window = min(MA_WINDOW, len(losses))

            if window >= 2:
                cumsum = [0.0]
                for v in losses:
                    cumsum.append(cumsum[-1] + float(v))

                ma = []
                for i in range(len(losses)):
                    start = max(0, i - window + 1)
                    denom = i - start + 1
                    ma.append((cumsum[i + 1] - cumsum[start]) / denom)

                ax.plot(
                    steps,
                    ma,
                    color = style["line"],
                    linewidth = 2.5,
                    alpha = 0.95,
                    label = f"Moving Avg ({ma[-1]:.4f})",
                )

                leg = ax.legend(frameon = False, fontsize = 9)
                for t in leg.get_texts():
                    t.set_color(style["text"])

            ax.set_xlabel("Steps", fontsize = 10, color = style["text"])
            ax.set_ylabel("Loss", fontsize = 10, color = style["text"])

            if progress.error:
                title = f"Error: {progress.error}"
            elif progress.is_completed:
                loss_str = f"{progress.loss:.4f}" if progress.loss is not None else "--"
                title = f"Training completed! Final loss: {loss_str}"
            elif progress.status_message:
                title = progress.status_message
            elif progress.step > 0:
                loss_str = f"{progress.loss:.4f}" if progress.loss is not None else "--"
                title = f"Epoch: {progress.epoch} | Step: {progress.step}/{progress.total_steps} | Loss: {loss_str}"
            else:
                title = "Training Loss"

            ax.set_title(title, fontsize = 11, fontweight = "bold", pad = 10, color = style["text"])
            ax.grid(True, alpha = 0.4, linestyle = "--", color = style["grid_color"])
            ax.tick_params(colors = style["text"], which = "both")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["bottom"].set_color(style["text"])
            ax.spines["left"].set_color(style["text"])
        else:
            display_msg = (
                progress.status_message
                if progress.status_message
                else "Waiting for training data..."
            )
            ax.text(
                0.5,
                0.5,
                display_msg,
                ha = "center",
                va = "center",
                fontsize = 16,
                color = style["empty_text"],
                transform = ax.transAxes,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

        fig.tight_layout()
        return fig


_training_backend = None


def get_training_backend() -> TrainingBackend:
    """Get global training backend instance"""
    global _training_backend
    if _training_backend is None:
        _training_backend = TrainingBackend()
    return _training_backend
