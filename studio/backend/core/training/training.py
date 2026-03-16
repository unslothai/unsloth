# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Training backend — subprocess orchestrator.

Each training job runs in a fresh subprocess (mp.get_context("spawn")),
solving the transformers version-switching problem. The old in-process
UnslothTrainer singleton is only used inside the subprocess (worker.py).

This file orchestrates the subprocess lifecycle, pumps events from the
worker's mp.Queue, and exposes the same API surface to routes/training.py.

Pattern follows core/data_recipe/jobs/manager.py.
"""

import math
import multiprocessing as mp
import queue
import threading
import time
import structlog
from loggers import get_logger
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, Any

import matplotlib.pyplot as plt

logger = get_logger(__name__)

_CTX = mp.get_context("spawn")

# Plot styling constants
PLOT_WIDTH = 8
PLOT_HEIGHT = 3.5


@dataclass
class TrainingProgress:
    """Mirror of trainer.TrainingProgress — kept here so the parent process
    never needs to import the heavy ML modules."""

    epoch: float = 0
    step: int = 0
    total_steps: int = 0
    loss: float = 0.0
    learning_rate: float = 0.0
    is_training: bool = False
    is_completed: bool = False
    error: Optional[str] = None
    status_message: str = "Ready to train"
    elapsed_seconds: Optional[float] = None
    eta_seconds: Optional[float] = None
    grad_norm: Optional[float] = None
    num_tokens: Optional[int] = None
    eval_loss: Optional[float] = None


class TrainingBackend:
    """
    Training orchestration backend — subprocess-based.
    Launches a fresh subprocess per training job, communicates via mp.Queue.
    """

    def __init__(self):
        # Subprocess state
        self._proc: Optional[mp.Process] = None
        self._event_queue: Any = None
        self._stop_queue: Any = None
        self._pump_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Progress state (updated by pump thread from subprocess events)
        self._progress = TrainingProgress()
        self._should_stop = False
        self._cancel_requested = False  # True only for stop(save=False)

        # Training Metrics (consumed by routes for SSE and /metrics)
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

        logger.info("TrainingBackend initialized (subprocess mode)")

    # ------------------------------------------------------------------
    # Public API (called by routes/training.py)
    # ------------------------------------------------------------------

    def start_training(self, **kwargs) -> bool:
        """Spawn a subprocess to run the full training pipeline.

        All kwargs are serialized into a config dict and sent to the worker.
        Returns True if the subprocess was started successfully.
        """
        with self._lock:
            if self._proc is not None and self._proc.is_alive():
                logger.warning("Training subprocess already running")
                return False

        # Join prior pump thread to prevent it from consuming events
        # from the new job's queue (it reads self._event_queue dynamically).
        if self._pump_thread is not None and self._pump_thread.is_alive():
            self._pump_thread.join(timeout = 5.0)
            if self._pump_thread.is_alive():
                logger.warning("Previous pump thread did not exit within 5s")
        self._pump_thread = None

        # Reset state
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

        # Build config dict for the subprocess
        config = {
            "model_name": kwargs["model_name"],
            "training_type": kwargs.get("training_type", "LoRA/QLoRA"),
            "hf_token": kwargs.get("hf_token", ""),
            "load_in_4bit": kwargs.get("load_in_4bit", True),
            "max_seq_length": kwargs.get("max_seq_length", 2048),
            "hf_dataset": kwargs.get("hf_dataset", ""),
            "local_datasets": kwargs.get("local_datasets"),
            "local_eval_datasets": kwargs.get("local_eval_datasets"),
            "format_type": kwargs.get("format_type", ""),
            "subset": kwargs.get("subset"),
            "train_split": kwargs.get("train_split", "train"),
            "eval_split": kwargs.get("eval_split"),
            "eval_steps": kwargs.get("eval_steps", 0.00),
            "dataset_slice_start": kwargs.get("dataset_slice_start"),
            "dataset_slice_end": kwargs.get("dataset_slice_end"),
            "custom_format_mapping": kwargs.get("custom_format_mapping"),
            "is_dataset_image": kwargs.get("is_dataset_image", False),
            "is_dataset_audio": kwargs.get("is_dataset_audio", False),
            "is_embedding": kwargs.get("is_embedding", False),
            "num_epochs": kwargs.get("num_epochs", 3),
            "learning_rate": kwargs.get("learning_rate", "2e-4"),
            "batch_size": kwargs.get("batch_size", 2),
            "gradient_accumulation_steps": kwargs.get("gradient_accumulation_steps", 4),
            "warmup_steps": kwargs.get("warmup_steps"),
            "warmup_ratio": kwargs.get("warmup_ratio"),
            "max_steps": kwargs.get("max_steps", 0),
            "save_steps": kwargs.get("save_steps", 0),
            "weight_decay": kwargs.get("weight_decay", 0.01),
            "random_seed": kwargs.get("random_seed", 3407),
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
            "finetune_attention_modules": kwargs.get(
                "finetune_attention_modules", True
            ),
            "finetune_mlp_modules": kwargs.get("finetune_mlp_modules", True),
            "enable_wandb": kwargs.get("enable_wandb", False),
            "wandb_token": kwargs.get("wandb_token"),
            "wandb_project": kwargs.get("wandb_project", "unsloth-training"),
            "enable_tensorboard": kwargs.get("enable_tensorboard", False),
            "tensorboard_dir": kwargs.get("tensorboard_dir", "runs"),
            "trust_remote_code": kwargs.get("trust_remote_code", False),
        }

        # Derive load_in_4bit from training_type
        if config["training_type"] != "LoRA/QLoRA":
            config["load_in_4bit"] = False

        # Spawn subprocess
        from .worker import run_training_process

        self._event_queue = _CTX.Queue()
        self._stop_queue = _CTX.Queue()

        self._proc = _CTX.Process(
            target = run_training_process,
            kwargs = {
                "event_queue": self._event_queue,
                "stop_queue": self._stop_queue,
                "config": config,
            },
            daemon = True,
        )
        self._proc.start()
        logger.info("Training subprocess started (pid=%s)", self._proc.pid)

        # Start event pump thread
        self._pump_thread = threading.Thread(target = self._pump_loop, daemon = True)
        self._pump_thread.start()

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
            # Update progress immediately for responsive UI
            self._progress.status_message = (
                "Stopping training and saving checkpoint..."
                if save
                else "Cancelling training..."
            )
        return True

    def force_terminate(self) -> None:
        """Force-kill the training subprocess so state can be reset immediately."""
        with self._lock:
            if self._proc is not None and self._proc.is_alive():
                logger.info(
                    "Force-terminating training subprocess (pid=%s)", self._proc.pid
                )
                self._proc.terminate()
            proc = self._proc

        if proc is not None:
            proc.join(timeout = 5.0)
            if proc.is_alive():
                proc.kill()
                proc.join(timeout = 2.0)

    def is_training_active(self) -> bool:
        """Check if training is currently active."""
        with self._lock:
            # Subprocess alive = active
            if self._proc is not None and self._proc.is_alive():
                return True

            # Stop was requested and process exited → inactive
            if self._should_stop:
                return False

            # Check progress state
            p = self._progress
            if p.is_training:
                return True
            if p.is_completed or p.error:
                return False

            # Check status message for activity indicators
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

    def refresh_plot_for_theme(self, theme: str) -> Optional[plt.Figure]:
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

    def _pump_loop(self) -> None:
        """Background thread: consume events from subprocess → update state."""
        while True:
            if self._proc is None or self._event_queue is None:
                return

            # Try to read an event
            event = self._read_queue(self._event_queue, timeout_sec = 0.25)
            if event is not None:
                self._handle_event(event)
                continue

            # No event — check if process is still alive
            if self._proc.is_alive():
                continue

            # Process exited — drain remaining events
            for e in self._drain_queue(self._event_queue):
                self._handle_event(e)

            # Mark as done if no explicit complete/error was received
            with self._lock:
                if self._progress.is_training:
                    if self._should_stop:
                        self._progress.is_training = False
                        self._progress.status_message = "Training stopped."
                    else:
                        self._progress.is_training = False
                        self._progress.error = (
                            self._progress.error
                            or "Training process exited unexpectedly"
                        )
            return

    def _handle_event(self, event: dict) -> None:
        """Apply a subprocess event to local state."""
        etype = event.get("type")

        with self._lock:
            if etype == "progress":
                self._progress.step = event.get("step", self._progress.step)
                self._progress.epoch = event.get("epoch", self._progress.epoch)
                self._progress.loss = event.get("loss", self._progress.loss)
                self._progress.learning_rate = event.get(
                    "learning_rate", self._progress.learning_rate
                )
                self._progress.total_steps = event.get(
                    "total_steps", self._progress.total_steps
                )
                self._progress.elapsed_seconds = event.get("elapsed_seconds")
                self._progress.eta_seconds = event.get("eta_seconds")
                self._progress.grad_norm = event.get("grad_norm")
                self._progress.num_tokens = event.get("num_tokens")
                self._progress.eval_loss = event.get("eval_loss")
                self._progress.is_training = True
                status = event.get("status_message", "")
                if status:
                    self._progress.status_message = status

                # Update metric histories
                step = event.get("step", 0)
                loss = event.get("loss", 0.0)
                lr = event.get("learning_rate", 0.0)
                if step >= 0 and loss > 0:
                    self.loss_history.append(loss)
                    self.lr_history.append(lr)
                    self.step_history.append(step)

                grad_norm = event.get("grad_norm")
                if grad_norm is not None:
                    try:
                        gn = float(grad_norm)
                    except (TypeError, ValueError):
                        gn = None
                    if gn is not None and math.isfinite(gn):
                        self.grad_norm_history.append(gn)
                        self.grad_norm_step_history.append(step)

                eval_loss = event.get("eval_loss")
                if eval_loss is not None:
                    self.eval_loss_history.append(eval_loss)
                    self.eval_step_history.append(step)
                    self.eval_enabled = True

            elif etype == "eval_configured":
                self.eval_enabled = True

            elif etype == "status":
                self._progress.status_message = event.get("message", "")
                self._progress.is_training = True

            elif etype == "complete":
                self._progress.is_training = False
                self._progress.is_completed = True
                self._output_dir = event.get("output_dir")
                msg = event.get("status_message", "Training completed")
                self._progress.status_message = msg

            elif etype == "error":
                self._progress.is_training = False
                self._progress.error = event.get("error", "Unknown error")
                logger.error("Training error: %s", event.get("error"))
                stack = event.get("stack", "")
                if stack:
                    logger.error("Stack trace:\n%s", stack)

    @staticmethod
    def _read_queue(q: Any, timeout_sec: float) -> Optional[dict]:
        try:
            return q.get(timeout = timeout_sec)
        except queue.Empty:
            return None
        except (EOFError, OSError, ValueError):
            return None

    @staticmethod
    def _drain_queue(q: Any) -> list:
        events = []
        while True:
            try:
                events.append(q.get_nowait())
            except queue.Empty:
                return events
            except (EOFError, OSError, ValueError):
                return events

    # ------------------------------------------------------------------
    # Plot generation (unchanged from original)
    # ------------------------------------------------------------------

    def _create_loss_plot(
        self, progress: TrainingProgress, theme: str = "light"
    ) -> plt.Figure:
        """Create training loss plot with theme-aware styling."""
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
                title = f"Training completed! Final loss: {progress.loss:.4f}"
            elif progress.status_message:
                title = progress.status_message
            elif progress.step > 0:
                title = f"Epoch: {progress.epoch} | Step: {progress.step}/{progress.total_steps} | Loss: {progress.loss:.4f}"
            else:
                title = "Training Loss"

            ax.set_title(
                title, fontsize = 11, fontweight = "bold", pad = 10, color = style["text"]
            )
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

        With subprocess-based training, the model lives in the subprocess
        and is freed when it exits. Inference must load from the saved
        checkpoint on disk. This is a no-op placeholder.
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
