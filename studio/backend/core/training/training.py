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
import queue
import re
import shutil
import threading
import time
import structlog
from datetime import datetime, timezone
from loggers import get_logger
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, Any, TYPE_CHECKING

if TYPE_CHECKING:
    import matplotlib.pyplot as plt
from utils.hardware import prepare_gpu_selection
from utils.native_path_leases import (
    native_path_secret_removed_for_child_start,
    run_without_native_path_secret,
)
from utils.paths import outputs_root

logger = get_logger(__name__)

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


def _cleanup_cancelled_checkpoints(output_dir: str | os.PathLike) -> None:
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
    """Mirror of trainer.TrainingProgress so the parent never imports heavy ML modules."""

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

        # Progress state (updated by pump thread from subprocess events)
        self._progress = TrainingProgress()
        self._should_stop = False
        self._cancel_requested = False  # True only for stop(save=False)

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

        # Build config dict for the subprocess
        config = {
            "model_name": kwargs["model_name"],
            "project_name": kwargs.get("project_name"),
            "training_type": kwargs.get("training_type", "LoRA/QLoRA"),
            "hf_token": kwargs.get("hf_token", ""),
            "load_in_4bit": kwargs.get("load_in_4bit", True),
            "max_seq_length": kwargs.get("max_seq_length", 2048),
            "vision_image_size": kwargs.get("vision_image_size"),
            "hf_dataset": kwargs.get("hf_dataset", ""),
            "local_datasets": kwargs.get("local_datasets"),
            "local_eval_datasets": kwargs.get("local_eval_datasets"),
            "format_type": kwargs.get("format_type", ""),
            "subset": kwargs.get("subset"),
            "train_split": kwargs.get("train_split", "train"),
            "eval_split": kwargs.get("eval_split"),
            "eval_steps": kwargs.get("eval_steps", 0.00),
            "dataset_streaming": kwargs.get("dataset_streaming", False),
            "dataset_slice_start": kwargs.get("dataset_slice_start"),
            "dataset_slice_end": kwargs.get("dataset_slice_end"),
            "custom_format_mapping": kwargs.get("custom_format_mapping"),
            "is_dataset_image": kwargs.get("is_dataset_image", False),
            "is_dataset_audio": kwargs.get("is_dataset_audio", False),
            "is_embedding": kwargs.get("is_embedding", False),
            "num_epochs": kwargs.get("num_epochs", 3),
            "learning_rate": kwargs.get("learning_rate", "2e-4"),
            "embedding_learning_rate": kwargs.get("embedding_learning_rate"),
            "batch_size": kwargs.get("batch_size", 2),
            "gradient_accumulation_steps": kwargs.get("gradient_accumulation_steps", 4),
            "warmup_steps": kwargs.get("warmup_steps"),
            "warmup_ratio": kwargs.get("warmup_ratio"),
            "max_steps": kwargs.get("max_steps", 0),
            "save_steps": kwargs.get("save_steps", 0),
            "weight_decay": kwargs.get("weight_decay", 0.001),
            "max_grad_norm": kwargs.get("max_grad_norm", 0.0),
            "max_grad_value": _coerce_optional_nonneg_float(
                "max_grad_value", kwargs.get("max_grad_value")
            ),
            "max_grad_leaf_norm": _coerce_optional_nonneg_float(
                "max_grad_leaf_norm", kwargs.get("max_grad_leaf_norm")
            ),
            "cast_norm_output_to_input_dtype": _coerce_optional_bool(
                kwargs.get("cast_norm_output_to_input_dtype"), True
            ),
            # MLX/CUDA/embedding workers need an int (transformers.set_seed(None) raises).
            "random_seed": _coerce_seed(kwargs.get("random_seed")),
            "packing": kwargs.get("packing", False),
            "optim": kwargs.get("optim", "adamw_8bit"),
            "lr_scheduler_type": kwargs.get("lr_scheduler_type", "linear"),
            "use_lora": kwargs.get("use_lora", True),
            "lora_r": kwargs.get("lora_r", 16),
            "lora_alpha": kwargs.get("lora_alpha", 16),
            "lora_dropout": kwargs.get("lora_dropout", 0.0),
            "target_modules": kwargs.get("target_modules"),
            "gradient_checkpointing": kwargs.get("gradient_checkpointing", "unsloth"),
            "use_rslora": kwargs.get("use_rslora", False),
            "use_loftq": kwargs.get("use_loftq", False),
            "train_on_completions": kwargs.get("train_on_completions", False),
            "finetune_vision_layers": kwargs.get("finetune_vision_layers", True),
            "finetune_language_layers": kwargs.get("finetune_language_layers", True),
            "finetune_attention_modules": kwargs.get("finetune_attention_modules", True),
            "finetune_mlp_modules": kwargs.get("finetune_mlp_modules", True),
            "enable_wandb": kwargs.get("enable_wandb", False),
            "wandb_token": kwargs.get("wandb_token"),
            "wandb_project": kwargs.get("wandb_project", "unsloth-training"),
            "enable_tensorboard": kwargs.get("enable_tensorboard", False),
            "tensorboard_dir": kwargs.get("tensorboard_dir", "runs"),
            "output_dir": kwargs.get("output_dir"),
            "resume_from_checkpoint": kwargs.get("resume_from_checkpoint"),
            "trust_remote_code": kwargs.get("trust_remote_code", False),
            "approved_remote_code_fingerprint": kwargs.get("approved_remote_code_fingerprint"),
            "subject": kwargs.get("subject"),
            "gpu_ids": kwargs.get("gpu_ids"),
            "s3_config": kwargs.get("s3_config"),
            # Flipped to True only by the HTTP-fallback respawn after a stall.
            "disable_xet": kwargs.get("disable_xet", False),
        }

        # Full finetuning always runs in 16-bit; LoRA/QLoRA/CPT keep the request.
        if config["training_type"] == "Full Finetuning":
            config["load_in_4bit"] = False

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
        if _hw.DEVICE == _hw.DeviceType.MLX:
            config["resolved_gpu_ids"] = None
            config["gpu_selection"] = None
        elif gpu_ids:
            resolved_gpu_ids, gpu_selection = prepare_gpu_selection(gpu_ids, **gpu_selection_kwargs)
            config["resolved_gpu_ids"] = resolved_gpu_ids
            config["gpu_selection"] = gpu_selection
        else:
            defer_auto_selection = True

        # Synchronous validation passed -> free VRAM (export + chat) now, before
        # auto-selection and the spawn, so placement sees the freed memory.
        if before_spawn is not None:
            try:
                before_spawn()
            except Exception:
                logger.warning("before_spawn hook failed; continuing", exc_info = True)

        if defer_auto_selection:
            resolved_gpu_ids, gpu_selection = prepare_gpu_selection(None, **gpu_selection_kwargs)
            config["resolved_gpu_ids"] = resolved_gpu_ids
            config["gpu_selection"] = gpu_selection

        from .worker import run_training_process

        try:
            with native_path_secret_removed_for_child_start():
                event_queue = _CTX.Queue()
                stop_queue = _CTX.Queue()

                proc = _CTX.Process(
                    target = run_without_native_path_secret,
                    args = (run_training_process,),
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
            return False

        logger.info("Training subprocess started (pid=%s)", proc.pid)

        # Reset state (old pump thread dead, proc.start() succeeded).
        self.current_job_id = job_id
        self._should_stop = False
        self._cancel_requested = False
        self._progress = TrainingProgress(
            is_training = True, status_message = "Initializing training..."
        )
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
        self._db_total_steps_set = False
        self._db_config = _sanitize_db_config(config)
        self._db_started_at = datetime.now(timezone.utc).isoformat()
        # Start each job Xet-first; keep config so a stall can respawn over HTTP.
        self._last_full_config = config
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

        return True

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
        return True

    def shutdown_with_checkpoint(self, timeout: float = 60.0) -> None:
        """Best-effort stop-and-save before force-terminating (server shutdown path)."""
        with self._lock:
            proc = self._proc
            active = proc is not None and proc.is_alive()
            cancelled = self._cancel_requested
            stopping = self._should_stop
        if active and not cancelled and timeout > 0:
            logger.info(
                "Training active at shutdown -- saving a stop checkpoint "
                "(up to %.0fs, Ctrl+C again to skip)...",
                timeout,
            )
            if not stopping:
                self.stop_training(save = True)
            proc.join(timeout = timeout)
        self.force_terminate()

    def force_terminate(self) -> None:
        """Force-kill the training subprocess so state can be reset immediately."""
        with self._lock:
            if self._proc is not None and self._proc.is_alive():
                logger.info("Force-terminating training subprocess (pid=%s)", self._proc.pid)
                self._proc.terminate()
            proc = self._proc
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

        from .worker import run_training_process

        try:
            with native_path_secret_removed_for_child_start():
                event_queue = _CTX.Queue()
                stop_queue = _CTX.Queue()
                new_proc = _CTX.Process(
                    target = run_without_native_path_secret,
                    args = (run_training_process,),
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
            self._pump_thread = new_pump
            # Start under the lock so _ensure_pump_alive can never observe the
            # new pump as a not-yet-started (dead) thread and spawn a duplicate.
            new_pump.start()

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
                with self._lock:
                    interrupted_output_dir = (
                        self._output_dir
                        if self._should_stop and not self._cancel_requested
                        else None
                    )
                    interrupted_clear_output_dir = self._cancel_requested
                    if interrupted_clear_output_dir:
                        # /status serializes _output_dir; a cancelled run must not expose it.
                        self._output_dir = None
                self._finalize_run_in_db(
                    status = "stopped" if self._should_stop else "error",
                    error_message = None
                    if self._should_stop
                    else "Training process terminated unexpectedly",
                    output_dir = interrupted_output_dir,
                    clear_output_dir = interrupted_clear_output_dir,
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

            elif etype == "output_dir":
                self._output_dir = event.get("output_dir")
                db_action = "persist_output_dir"

            elif etype == "status":
                self._progress.status_message = event.get("message", "")
                self._progress.is_training = True

            elif etype == "complete":
                self._progress.is_training = False
                self._progress.is_completed = True
                self._output_dir = event.get("output_dir")
                msg = event.get("status_message", "Training completed")
                self._progress.status_message = msg
                if not self._db_run_created and self.current_job_id and self._db_config:
                    db_action = "create_and_finalize"
                else:
                    db_action = "finalize"
                db_action_kwargs = {
                    "status": "stopped" if self._should_stop else "completed",
                    "output_dir": self._output_dir,
                    "clear_output_dir": self._cancel_requested,
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
                # keep_error_status: a failed stop-and-save checkpoint stays an error.
                db_action_kwargs = {
                    "status": "stopped"
                    if self._should_stop and not event.get("keep_error_status")
                    else "error",
                    "error_message": event.get("error", "Unknown error"),
                    "output_dir": self._output_dir,
                    "clear_output_dir": self._cancel_requested,
                    "resume_blocked": bool(event.get("resume_blocked")),
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
                self._persist_output_dir()
            except Exception:
                logger.warning("Failed to create DB run record", exc_info = True)
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

    def _persist_output_dir(self) -> None:
        if not self._output_dir or not self.current_job_id or not self._db_run_created:
            return
        try:
            from storage.studio_db import update_run_output_dir
            update_run_output_dir(self.current_job_id, self._output_dir)
        except Exception:
            logger.warning("Failed to persist output_dir", exc_info = True)

    def _ensure_db_run_created(self) -> None:
        """Create the DB row if it doesn't exist yet. Called outside the lock."""
        if self._db_run_created or not self.current_job_id or not self._db_config:
            return
        try:
            from storage.studio_db import create_run

            dataset_name = (
                self._db_config.get("hf_dataset")
                or next(iter(self._db_config.get("local_datasets") or []), None)
                or _s3_dataset_name(self._db_config.get("s3_dataset"))
                or "unknown"
            )
            create_run(
                id = self.current_job_id,
                model_name = self._db_config["model_name"],
                dataset_name = dataset_name,
                config_json = _json.dumps(self._db_config),
                started_at = self._db_started_at or datetime.now(timezone.utc).isoformat(),
                total_steps = self._progress.total_steps or None,
            )
            self._db_run_created = True
        except Exception:
            logger.warning("Failed to create DB run record for early failure", exc_info = True)

    def _finalize_run_in_db(
        self,
        status: str,
        error_message: Optional[str] = None,
        output_dir: Optional[str] = None,
        clear_output_dir: bool = False,
        resume_blocked: bool = False,
    ) -> None:
        """Flush remaining metrics and mark a run as finished in the DB."""
        if not self.current_job_id or not self._db_run_created or self._run_finalized:
            return
        self._flush_metrics_to_db()
        try:
            from storage.studio_db import finish_run
            from utils.downsample import downsample

            sparkline = downsample(self.loss_history, 50)
            finish_run(
                id = self.current_job_id,
                status = status,
                ended_at = datetime.now(timezone.utc).isoformat(),
                final_step = self._progress.step,
                final_loss = self._progress.loss
                if (self._progress.loss is not None and math.isfinite(self._progress.loss))
                else None,
                duration_seconds = self._progress.elapsed_seconds,
                loss_sparkline = _json.dumps(sparkline),
                output_dir = output_dir,
                error_message = error_message,
                clear_output_dir = clear_output_dir,
                resume_blocked = resume_blocked,
            )
            self._run_finalized = True
        except Exception:
            logger.warning("Failed to finalize run in DB (status=%s)", status, exc_info = True)

    def _flush_metrics_to_db(self) -> None:
        """Flush buffered metrics to the database and update live progress."""
        if not self._metric_buffer or not self.current_job_id or not self._db_run_created:
            return
        # Cap buffer to bound memory growth.
        if len(self._metric_buffer) > 500:
            logger.warning(
                "Metric buffer exceeded 500 entries (%d) — trimming oldest",
                len(self._metric_buffer),
            )
            self._metric_buffer = self._metric_buffer[-500:]
        # Snapshot before insert so metrics arriving during the write survive.
        batch = list(self._metric_buffer)
        try:
            from storage.studio_db import insert_metrics_batch, update_run_progress

            insert_metrics_batch(self.current_job_id, batch)
            del self._metric_buffer[: len(batch)]
            update_run_progress(
                id = self.current_job_id,
                step = self._progress.step,
                loss = self._progress.loss
                if (self._progress.loss is not None and math.isfinite(self._progress.loss))
                else None,
                duration_seconds = self._progress.elapsed_seconds,
            )
        except Exception:
            # Leave buffer intact for retry on next flush
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
