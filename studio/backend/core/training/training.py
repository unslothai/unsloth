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
import structlog
from datetime import datetime, timezone
from loggers import get_logger
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Optional, Tuple, Any, Callable, Union, TYPE_CHECKING

if TYPE_CHECKING:
    import matplotlib.pyplot as plt
from utils.hardware import prepare_gpu_selection
from utils.native_path_leases import (
    native_path_secret_removed_for_child_start,
    run_without_native_path_secret,
)
from utils.paths import outputs_root

logger = get_logger(__name__)


def _env_int(name: str, default: int) -> int:
    try:
        raw = (os.environ.get(name) or "").strip()
        return int(raw) if raw else default
    except ValueError:
        return default


# Stop-watchdog escalation timeouts. Primary trigger: a short grace once "complete"
# (save done). Absolute cap is a backstop: long for save=True so a slow save is never
# killed mid-write, shorter for a cancel that has nothing to save.
_STOP_GRACE_S = _env_int("UNSLOTH_STUDIO_TRAINING_STOP_GRACE_S", 15)
_STOP_TIMEOUT_S = _env_int("UNSLOTH_STUDIO_TRAINING_STOP_TIMEOUT_S", 600)
_CANCEL_TIMEOUT_S = _env_int("UNSLOTH_STUDIO_TRAINING_CANCEL_TIMEOUT_S", 120)

# Watchdog DB finalize: a few short retries so a transient SQLite lock doesn't lose the
# terminal state, since the watchdog is the sole finalizer once _proc is dropped.
_DB_FINALIZE_RETRIES = 3
_DB_FINALIZE_RETRY_S = 0.5

_pyplot = None
_pyplot_failed = False


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

        matplotlib.use("Agg")  # headless backend
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
    """Reject negatives; HTTP `ge=0` doesn't cover raw `**kwargs` callers."""
    if value is None:
        return None
    try:
        coerced = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"Unsloth: {name}={value!r} must be a non-negative float or None.")
    if coerced < 0:
        raise ValueError(f"Unsloth: {name}={coerced} must be >= 0 (use 0 or None to disable).")
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
        "max_grad_norm": values.get("max_grad_norm", 0.0),
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
    return config


_HF_TMP_CHECKPOINT_RE = re.compile(r"^tmp-checkpoint-\d+$")


def _sanitize_db_config(config: dict[str, Any]) -> dict[str, Any]:
    # ``subject`` (the run owner's username / API-key id) is worker-only metadata; never
    # persist it to config_json, which run-history GET returns to any authenticated user.
    db_config = {
        k: v
        for k, v in config.items()
        if k not in {"hf_token", "wandb_token", "s3_config", "subject"}
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

# Plot styling constants
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
    status_message: str = "Ready to train"
    elapsed_seconds: Optional[float] = None
    eta_seconds: Optional[float] = None
    grad_norm: Optional[float] = None
    num_tokens: Optional[int] = None
    eval_loss: Optional[float] = None
    peak_memory_gb: Optional[float] = None
    output_dir: Optional[str] = None


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
    ) -> Optional[tuple]:
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
        # Subprocess state
        self._proc: Optional[mp.Process] = None
        self._event_queue: Any = None
        self._stop_queue: Any = None
        self._pump_thread: Optional[threading.Thread] = None
        # True while a pump thread should be running; cleared on intended exits.
        # Left True after an abnormal death so _ensure_pump_alive spots a crash.
        self._pump_running: bool = False
        self._lock = threading.Lock()

        # Stop watchdog: after a stop is requested, escalates to force_terminate()
        # if the worker does not exit on its own within a bounded time. The watched
        # proc is tracked so a new run always gets its own watcher.
        self._stop_watchdog: Optional[threading.Thread] = None
        self._stop_watchdog_proc: Optional[mp.Process] = None
        self._complete_seen = threading.Event()

        # Progress state (updated by pump thread from subprocess events)
        self._progress = TrainingProgress()
        self._should_stop = False
        self._cancel_requested = False  # True only for stop(save=False)

        # Throttled training-status logging to the server log (not one line/step).
        self._last_progress_log_ts: float = 0.0
        self._last_progress_log_step: int = -1

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

        # Job metadata
        self.current_job_id: Optional[str] = None
        self._output_dir: Optional[str] = None

        # DB persistence
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

    # ------------------------------------------------------------------
    # Public API (called by routes/training.py)
    # ------------------------------------------------------------------

    def start_training(
        self,
        job_id: str,
        *,
        before_spawn = None,
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
            if self._proc is not None and self._proc.is_alive():
                logger.warning("Training subprocess already running")
                return False

        # Join prior pump thread — refuse to start if it won't die
        if self._pump_thread is not None and self._pump_thread.is_alive():
            self._pump_thread.join(timeout = 5.0)
            if self._pump_thread.is_alive():
                logger.warning("Previous pump thread did not exit within 5s — refusing to start")
                return False
        self._pump_thread = None
        # Clear a stale crash flag from a prior died pump so the watchdog can't
        # treat this fresh setup as a recoverable death.
        self._pump_running = False

        config = _build_training_worker_config(kwargs)

        # Split GPU validation from placement around the VRAM hook:
        #   * Explicit gpu_ids are validated here (raises -> the route returns 400
        #     before any teardown) and their placement is VRAM-independent, so it
        #     stays correct after the hook frees memory.
        #   * Auto-selection ranks GPUs by *free* VRAM, so it is deferred until
        #     after the hook frees export/chat -- otherwise it could pin training
        #     onto a GPU the hook is about to clear (and onto a kept chat model).
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

        # Handshake with the sidecar install route: mark the spawn in progress BEFORE rechecking
        # the reservation, so either this recheck aborts, or the install's is_training_active()
        # sees this flag (or the recorded proc) and refuses.
        from utils.transformers_version import sidecar_swap_in_progress

        self._spawn_in_progress = True
        if sidecar_swap_in_progress():
            self._spawn_in_progress = False
            from utils.transformers_version import SidecarSwapInProgress
            raise SidecarSwapInProgress(
                "A transformers installation is replacing the latest sidecar; "
                "retry when it completes."
            )

        # Any exception between the handshake above and the flag reset below would
        # otherwise leave _spawn_in_progress latched, wedging is_training_active
        # (and the install route) until restart.
        try:
            # Synchronous validation passed -> free VRAM (export + chat) now, before
            # auto-selection and the spawn, so placement sees the freed memory. Runs AFTER the handshake
            # so a lost race to an install can't tear down chat/export for a training run that never spawns.
            if before_spawn is not None:
                try:
                    before_spawn()
                except Exception:
                    logger.warning("before_spawn hook failed; continuing", exc_info = True)

            if defer_auto_selection:
                try:
                    resolved_gpu_ids, gpu_selection = prepare_gpu_selection(
                        None, **gpu_selection_kwargs
                    )
                except Exception:
                    # Flag is already set; a failed GPU selection must not leave is_training_active stuck True.
                    self._spawn_in_progress = False
                    raise
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
                    proc.start()
                    from utils.process_lifetime import adopt_pid

                    adopt_pid(proc.pid)  # bind to parent lifetime (Windows job / sweep)
            except Exception:
                logger.error("Failed to start training subprocess", exc_info = True)
                self._spawn_in_progress = False
                return False

            logger.info("Training subprocess started (pid=%s)", proc.pid)

            # Reset state (old pump thread dead, proc.start() succeeded).
            self.current_job_id = job_id
            self._should_stop = False
            self._cancel_requested = False
            self._complete_seen.clear()
            self._progress = TrainingProgress(
                is_training = True, status_message = "Initializing training..."
            )
            # Reset the progress-log throttle so the new run always logs its first step,
            # even if it starts within 30s of a prior run whose last logged step matches.
            self._last_progress_log_ts = 0.0
            self._last_progress_log_step = -1
            self.loss_history.clear()
            self.lr_history.clear()
            self.step_history.clear()
            self.grad_norm_history.clear()
            self.grad_norm_step_history.clear()
            self.eval_loss_history.clear()
            self.eval_step_history.clear()
            self.eval_enabled = False
            self._output_dir = None
            self._metric_buffer.clear()
            self._run_finalized = False
            self._db_run_created = False
            self._db_create_in_progress = False  # a stale watchdog create can't block this run
            self._db_total_steps_set = False
            self._db_config = _sanitize_db_config(config)
            self._db_started_at = datetime.now(timezone.utc).isoformat()
            # Start each job Xet-first; keep config so a stall can respawn over HTTP.
            self._last_full_config = config
            self._last_hf_cache_env = cache_env
            self._in_model_load = False
            self._xet_fallback_used = False
            self._needs_xet_respawn = False

            # Create the DB run row before the pump can consume events, so it appears
            # in history during model loading and a fast terminal worker can't race the
            # pump into a duplicate create/finalize. From here the pump only finalizes.
            self._ensure_db_run_created()

            # Assign handles and start the pump together under the lock so a concurrent
            # poll can't see a live _proc with no pump and spawn a duplicate.
            new_pump = threading.Thread(target = self._pump_loop, daemon = True)
            with self._lock:
                self._pump_running = False
                self._event_queue = event_queue
                self._stop_queue = stop_queue
                self._proc = proc
                self._pump_thread = new_pump
                new_pump.start()
                self._spawn_in_progress = False

            return True

        except Exception:
            self._spawn_in_progress = False
            raise

    def stop_training(self, save: bool = True) -> bool:
        """Send stop signal to the training subprocess."""
        self._should_stop = True
        if not save:
            self._cancel_requested = True
        with self._lock:
            if self._stop_queue is not None:
                try:
                    self._stop_queue.put({"type": "stop", "save": save})
                except (OSError, ValueError):
                    pass
            # Update progress immediately for responsive UI.
            self._progress.status_message = (
                "Stopping training and saving checkpoint..." if save else "Cancelling training..."
            )
        # Guarantee the run finalizes even if the worker wedges after saving.
        self._start_stop_watchdog(cancel = not save)
        return True

    def _start_stop_watchdog(self, cancel: bool) -> None:
        """Start a daemon that force-terminates the worker if a requested stop does not
        exit on its own. No-op if no worker is alive or a live watchdog already watches
        this proc (a stale watchdog on an old proc never blocks a new run's watcher)."""
        with self._lock:
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
    ) -> None:
        """Escalate a stuck stop to force_terminate(): grace after "complete", else the
        absolute backstop (see the module timeouts). No-ops on a clean exit; exits
        silently if a new run replaces the worker."""
        started = time.monotonic()
        complete_at: Optional[float] = None
        reason = ""
        while True:
            with self._lock:
                superseded = self._proc is not target_proc
                # A later cancel has nothing to save, so tighten an in-flight save
                # watchdog to the shorter cancel cap.
                cancelling = cancel or self._cancel_requested
            if superseded or not target_proc.is_alive():
                return
            now = time.monotonic()
            abs_timeout = _CANCEL_TIMEOUT_S if cancelling else _STOP_TIMEOUT_S
            if complete_at is None and self._complete_seen.is_set():
                complete_at = now
            if complete_at is not None and now - complete_at >= _STOP_GRACE_S:
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
            logger.warning("Stop watchdog force-terminating stuck training worker: %s", reason)
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
        even if the worker is wedged in driver teardown; preserves output_dir so a saved
        checkpoint is kept. No-ops if a new run already replaced the watched worker, so a
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
                return  # a new run replaced the worker; never touch its state
            if watched_job_id is not None and self.current_job_id != watched_job_id:
                return  # a new run is already starting up; leave its state alone
            run_id = self.current_job_id  # == watched_job_id
            self._progress.is_training = False
            self._progress.status_message = "Training stopped."
        # Create the row if a start-time create failed (no-op otherwise; skips when the pump
        # is mid-create, in which case its create-then-finalize records the run instead).
        self._ensure_db_run_created()
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
            output_dir = self._output_dir
            if claim:
                self._run_finalized = True  # claim this run's finalize
                batch = list(self._metric_buffer)
                del self._metric_buffer[: len(batch)]
                final_step = self._progress.step
                final_loss = self._progress.loss
                if final_loss is not None and not math.isfinite(final_loss):
                    final_loss = None
                duration = self._progress.elapsed_seconds
                loss_history = list(self.loss_history)
        if claim:
            self._finish_stopped_run(
                run_id, output_dir, batch, final_step, final_loss, duration, loss_history
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
                    status = "stopped",
                    ended_at = datetime.now(timezone.utc).isoformat(),
                    final_step = final_step,
                    final_loss = final_loss,
                    duration_seconds = duration,
                    loss_sparkline = _json.dumps(sparkline),
                    output_dir = output_dir,
                    error_message = None,
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
                return  # superseded by a new run; do not touch the new worker
            if proc is not None and proc.is_alive():
                logger.info("Force-terminating training subprocess (pid=%s)", proc.pid)
                proc.terminate()
            cancelled = self._cancel_requested
            output_dir = self._output_dir

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

    def _respawn_worker_disable_xet(self) -> None:
        """Respawn the worker once with HF_HUB_DISABLE_XET=1 after a model-load
        stall. Runs on the exiting pump thread, reaps the terminated worker, and
        starts a fresh worker + pump. DB/progress run-state is preserved so the
        history row is not duplicated; the new worker re-formats and loads over HTTP.
        """
        config = self._last_full_config
        if config is None:
            logger.error("Cannot respawn training worker: no stored config")
            return

        with self._lock:
            old_proc = self._proc
        if old_proc is not None:
            old_proc.join(timeout = 5.0)
            if old_proc.is_alive():
                old_proc.kill()
                old_proc.join(timeout = 2.0)

        config = {**config, "disable_xet": True}
        self._last_full_config = config
        logger.warning("Respawning training worker with HF_HUB_DISABLE_XET=1 after Xet stall")

        cache_env = getattr(self, "_last_hf_cache_env", None)
        if not cache_env:
            from utils.hf_cache_settings import get_hf_cache_paths
            cache_env = get_hf_cache_paths().child_env({})
        from utils.hf_cache_settings import child_environment_for_spawn

        # This run is active, so an install request 409s rather than proceeds: a reservation seen here
        # is transient (an aborting install or short lazy repair). Wait it out instead of stranding the
        # stalled run; only a wedged reservation fails the respawn.
        from utils.transformers_version import sidecar_swap_in_progress

        self._spawn_in_progress = True
        _swap_wait_deadline = time.time() + 120
        while sidecar_swap_in_progress() and time.time() < _swap_wait_deadline:
            time.sleep(1)
        if sidecar_swap_in_progress():
            # Raising here would land in the pump's broad finalization catch and
            # strand the run in a training state with no worker: finalize it as a
            # failure explicitly instead.
            self._spawn_in_progress = False
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
            return

        # Reset the handshake flag on any unexpected failure past this point, so a
        # crashed respawn cannot wedge is_training_active until restart.
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

                    adopt_pid(new_proc.pid)  # bind to parent lifetime (Windows job / sweep)
            except Exception:
                logger.error("Failed to respawn training subprocess", exc_info = True)
                self._spawn_in_progress = False
                with self._lock:
                    # No replacement pump will run; clear the flag so a later run can't
                    # inherit a stale _pump_running=True and spawn a duplicate.
                    self._pump_running = False
                    self._progress.is_training = False
                    self._progress.error = "Failed to recover stalled model download"
                self._ensure_db_run_created()
                self._finalize_run_in_db(
                    status = "error",
                    error_message = "Failed to recover stalled model download",
                )
                return

            logger.info("Training subprocess respawned with Xet disabled (pid=%s)", new_proc.pid)
            new_pump = threading.Thread(target = self._pump_loop, daemon = True)
            with self._lock:
                self._in_model_load = False
                self._event_queue = event_queue
                self._stop_queue = stop_queue
                self._proc = new_proc
                self._spawn_in_progress = False
                self._pump_thread = new_pump
                # Start under the lock so _ensure_pump_alive can never observe the
                # new pump as a not-yet-started (dead) thread and spawn a duplicate.
                new_pump.start()
        except Exception:
            self._spawn_in_progress = False
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
            # A restarted pump needs the worker handle and queue to drain/finalize;
            # their absence means nothing is left to recover.
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
            # Start under the lock so a concurrent _ensure_pump_alive can't see
            # this thread as not-yet-started and spawn yet another pump.
            new_pump.start()
        return True

    def is_training_active(self) -> bool:
        """Check if training is currently active."""
        # A spawn past its sidecar-swap recheck counts as active even before _proc is recorded,
        # so an install cannot slip in mid-spawn.
        if getattr(self, "_spawn_in_progress", False):
            return True
        # Self-heal a crashed pump first: a dead pump must never leave the worker
        # training invisibly behind a frozen UI. Cheap enough for per-second polls.
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

    # ------------------------------------------------------------------
    # Compatibility shims — routes/training.py accesses these
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Event pump (background thread)
    # ------------------------------------------------------------------

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
                # If a read keeps raising after the worker died, fall through to
                # finalize instead of spinning; only retry while the worker lives.
                logger.exception("Training event pump: queue read failed; continuing")
                if self._proc is not None and self._proc.is_alive():
                    time.sleep(0.1)
                    continue
                event = None

            if event is not None:
                self._safe_handle_event(event)
                continue

            if self._proc.is_alive():
                continue

            # Worker exited. Drain the backlog and finalize, guarded so a slow or
            # failing DB write can't strand the thread; we return either way.
            try:
                for e in self._drain_queue(self._event_queue):
                    self._safe_handle_event(e)

                # Model-load stall: respawn over HTTP instead of finalizing as failure.
                # Starts a fresh pump on this thread (no self-join); it takes over
                # _pump_running, so this exit leaves the flag set.
                if self._needs_xet_respawn:
                    self._needs_xet_respawn = False
                    self._respawn_worker_disable_xet()
                    return

                # Mark done if no explicit complete/error was received.
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
                self._finalize_run_in_db(
                    status = "stopped" if self._should_stop else "error",
                    error_message = None
                    if self._should_stop
                    else "Training process terminated unexpectedly",
                )
            except Exception:
                logger.exception("Training event pump: finalization after worker exit failed")
            self._pump_running = False
            return

    def _handle_event(self, event: dict) -> None:
        """Apply a subprocess event to local state.

        State updates happen inside self._lock; DB I/O happens after releasing
        it so status-polling endpoints aren't blocked by slow SQLite writes.
        """
        etype = event.get("type")
        db_action: Optional[str] = None
        db_action_kwargs: dict = {}

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
                    # Drop the value rather than laundering it back to the last
                    # finite loss; clients see loss=None at this step so the NaN
                    # is not hidden behind a stale value. Training continues.
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
                    # Clear stale finite loss so the API doesn't keep
                    # reporting the last good value while NaN is happening.
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

                # Update metric histories using sanitized values.
                step = event.get("step", 0)
                loss = _safe_loss
                lr = _safe_lr
                if step > 0 and loss is not None:
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

                # Buffer metric for DB flush.
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

            elif etype == "status":
                self._progress.status_message = event.get("message", "")
                self._progress.is_training = True

            elif etype == "complete":
                msg = event.get("status_message", "Training completed")
                stopped = self._should_stop or msg.strip().lower() in {
                    "training cancelled",
                    "training stopped",
                }
                # Save is done by now; let the stop watchdog start its grace timer.
                self._complete_seen.set()
                self._progress.is_training = False
                self._progress.is_completed = not stopped
                self._output_dir = event.get("output_dir")
                self._progress.output_dir = self._output_dir
                self._progress.status_message = msg
                if not self._db_run_created and self.current_job_id and self._db_config:
                    db_action = "create_and_finalize"
                else:
                    db_action = "finalize"
                db_action_kwargs = {
                    "status": "stopped" if stopped else "completed",
                    "output_dir": self._output_dir,
                }

            elif etype == "error":
                self._progress.is_training = False
                self._progress.error = event.get("error", "Unknown error")
                logger.error("Training error: %s", event.get("error"))
                stack = event.get("stack", "")
                if stack:
                    logger.error("Stack trace:\n%s", stack)
                if not self._db_run_created and self.current_job_id and self._db_config:
                    db_action = "create_and_finalize"
                else:
                    db_action = "finalize"
                db_action_kwargs = {
                    "status": "stopped" if self._should_stop else "error",
                    "error_message": event.get("error", "Unknown error"),
                }

        # --- DB I/O outside the lock ---
        if db_action == "create_run":
            try:
                from storage.studio_db import create_run

                create_run(
                    id = db_action_kwargs["job_id"],
                    model_name = db_action_kwargs["model_name"],
                    dataset_name = db_action_kwargs["dataset_name"],
                    config_json = db_action_kwargs["config_json"],
                    started_at = db_action_kwargs["started_at"],
                    total_steps = db_action_kwargs["total_steps"],
                )
                self._db_run_created = True
                if db_action_kwargs["total_steps"]:
                    self._db_total_steps_set = True
            except Exception:
                logger.warning("Failed to create DB run record", exc_info = True)
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

        if etype == "progress":
            self._log_training_progress()

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
        self._last_progress_log_ts = now
        self._last_progress_log_step = step
        logger.info(
            "training_progress",
            step = step,
            total_steps = total or None,
            percent = int(step * 100 / total) if total > 0 else None,
            loss = round(p.loss, 4) if p.loss is not None else None,
            epoch = round(p.epoch, 2) if p.epoch is not None else None,
            eta_s = int(p.eta_seconds) if p.eta_seconds else None,
        )

    def _ensure_db_run_created(self) -> None:
        """Create the DB row if it doesn't exist yet. An in-progress flag lets only one
        caller create at a time, and ``_db_run_created`` is published only after
        ``create_run`` commits, so a concurrent finalize never runs ``finish_run`` against a
        not-yet-inserted row (a zero-row UPDATE that would leave the run stuck as running)."""
        with self._lock:
            if (
                self._db_run_created
                or self._db_create_in_progress
                or not self.current_job_id
                or not self._db_config
            ):
                return
            self._db_create_in_progress = True  # only one caller creates
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
            create_run(
                id = job_id,
                model_name = db_config["model_name"],
                dataset_name = dataset_name,
                config_json = _json.dumps(db_config),
                started_at = started_at,
                total_steps = total_steps,
            )
            created = True
        except Exception:
            logger.warning("Failed to create DB run record for early failure", exc_info = True)
        finally:
            with self._lock:
                # Publish the flags only if this is still the current run. A killed worker
                # lets a new /start proceed mid-create, and these flags are backend-wide, so
                # a stale create for the captured job must not satisfy the new run's DB state
                # (the row was still created by id; the new run owns/creates its own row).
                if self.current_job_id == job_id:
                    if created:
                        self._db_run_created = True  # publish only after the insert commits
                    self._db_create_in_progress = False

    def _finalize_run_in_db(
        self,
        status: str,
        error_message: Optional[str] = None,
        output_dir: Optional[str] = None,
        expected_job_id: Optional[str] = None,
    ) -> None:
        """Flush remaining metrics and mark a run finished in the DB. Claims the finalize
        under the lock so the watchdog and pump can't double-finalize, and no-ops when
        ``expected_job_id`` no longer matches (a new run took over). The run id and final
        progress are snapshotted under the lock and threaded through the flush/finish calls,
        so a new run racing between this claim and the DB writes can't be flushed or marked
        stopped under the old run's finalize."""
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
        self._flush_metrics_to_db(run_id = run_id)
        try:
            from storage.studio_db import finish_run
            from utils.downsample import downsample

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
            )
        except Exception:
            with self._lock:
                self._run_finalized = False  # unclaim so a later flush can retry
            logger.warning("Failed to finalize run in DB (status=%s)", status, exc_info = True)

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
            # A closed/broken queue reads as "no event"; any other error is left to
            # _pump_loop's guarded block, which logs and backs off.
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
                # A drain error must not abort finalization: return what we have so
                # the run finalizes rather than wedging "active" behind a dead worker.
                logger.exception(
                    "Training event pump: queue drain failed; finalizing with drained events"
                )
                return events

    # ------------------------------------------------------------------
    # Plot generation
    # ------------------------------------------------------------------

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

    def _transfer_to_inference_backend(self) -> bool:
        """Transfer model to inference backend.

        No-op: with subprocess training the model is freed on exit, so inference
        must load from the saved checkpoint on disk.
        """
        logger.info(
            "_transfer_to_inference_backend: subprocess training — "
            "model must be loaded from disk (output_dir=%s)",
            self._output_dir,
        )
        return False


# ========== GLOBAL INSTANCE ==========
_training_backend = None


def get_training_backend() -> TrainingBackend:
    """Get global training backend instance"""
    global _training_backend
    if _training_backend is None:
        _training_backend = TrainingBackend()
    return _training_backend
