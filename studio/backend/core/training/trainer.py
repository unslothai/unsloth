# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Unsloth Training Backend
Integrates Unsloth training with the FastAPI backend.
"""

import gc
import os
import sys
import types

# Off on Linux so datasets' forked map() workers can't deadlock. On spawn platforms
# (Windows/macOS) map() runs in-process, so keep the fast tokenizer's Rust threads on
# (the only parallelism single-process tokenize gets; off makes prep run serially).
os.environ["TOKENIZERS_PARALLELISM"] = "true" if sys.platform in ("win32", "darwin") else "false"

# Make compiled cache modules importable by any subprocess. On spawn platforms
# (Windows/macOS) spawned dataset.map() workers re-import top-level modules, and
# the cache's trainer files import torch + unsloth_zoo (inits CUDA). Propagating
# UNSLOTH_COMPILE_LOCATION via PYTHONPATH lets any subprocess find them.
# Do NOT import unsloth_zoo.compiler here -- it triggers heavy torch/triton imports.
if sys.platform in ("win32", "darwin"):
    _compile_cache = os.environ.get("UNSLOTH_COMPILE_LOCATION", "unsloth_compiled_cache")
    if not os.path.isabs(_compile_cache):
        _compile_cache = os.path.abspath(_compile_cache)
        os.environ["UNSLOTH_COMPILE_LOCATION"] = _compile_cache
    _pp = os.environ.get("PYTHONPATH", "")
    if _compile_cache not in _pp.split(os.pathsep):
        os.environ["PYTHONPATH"] = _compile_cache + (os.pathsep + _pp if _pp else "")
    if _compile_cache not in sys.path:
        sys.path.insert(0, _compile_cache)

import torch
from utils.hardware import (
    clear_gpu_cache,
    safe_num_proc,
    dataset_map_num_proc,
    get_device_map,
    raise_if_offloaded,
    get_visible_gpu_count,
)

# recompile_limit was removed in some ROCm torch builds; guard so training
# doesn't crash on RDNA2/RDNA3 with older ROCm wheels.
if hasattr(torch._dynamo.config, "recompile_limit"):
    torch._dynamo.config.recompile_limit = 64


# Drop any unsloth/unsloth_zoo namespace-package shadow before importing them.
from core.import_guards import ensure_real_packages as _ensure_real_packages

_ensure_real_packages("unsloth_zoo", "unsloth")
from unsloth import FastLanguageModel, FastVisionModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template

import json
import threading
import math
import subprocess
import structlog
from loggers import get_logger
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable
import pandas as pd
from datasets import Dataset
from utils.datasets.cache_safe import load_dataset_cache_safe as load_dataset

from core.inference.llama_cpp import _hf_offline_if_dns_dead
from utils.models import is_vision_model, detect_audio_type
from utils.models.model_config import _env_offline
from utils.datasets import format_and_template_dataset
from utils.datasets.completion_masking import apply_completion_masking
from utils.datasets.iterable import is_streaming_dataset as detect_streaming_dataset
from utils.datasets.raw_text import prepare_raw_text_dataset, resolve_column_names
from utils.paths import (
    ensure_dir,
    resolve_dataset_path,
    resolve_output_dir,
    resolve_tensorboard_dir,
)
from trl import SFTTrainer, SFTConfig

from utils.native_path_leases import child_env_without_native_path_secret
from utils.subprocess_compat import (
    windows_hidden_subprocess_kwargs as _windows_hidden_subprocess_kwargs,
)
from .training import (
    TrainingProgress,
    create_mlx_trainer_adapter,
    should_use_mlx_training_backend,
)

logger = get_logger(__name__)

# A streaming eval dataset has no __len__, so a streaming evaluation would
# iterate the entire (potentially unbounded) source on every eval step. Cap it
# to a fixed sample count so each evaluation terminates predictably.
STREAMING_EVAL_MAX_SAMPLES = 500


def _build_report_targets(training_args) -> list[str] | str:
    report_to: list[str] = []
    if training_args.get("enable_wandb", False):
        report_to.append("wandb")
    if training_args.get("enable_tensorboard", False):
        report_to.append("tensorboard")
    return report_to or "none"


class UnslothTrainer:
    """
    Unsloth Training Backend
    """

    def __new__(cls, *args, **kwargs):
        if cls is UnslothTrainer and should_use_mlx_training_backend():
            return create_mlx_trainer_adapter(*args, **kwargs)
        return super().__new__(cls)

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.training_thread = None
        self.training_progress = TrainingProgress()
        self.progress_callbacks = []
        self.is_training = False
        self.should_stop = False
        self.save_on_stop = True
        self.load_in_4bit = True  # Quantization mode for metadata

        # Model state tracking
        self.is_cpt = False  # True for Continued Pretraining
        self.is_vlm = False
        self.is_audio = False
        self.is_audio_vlm = False  # Multimodal model (e.g. Gemma 3N) trained on audio data
        self._audio_type = None  # 'csm', 'whisper', 'snac', 'bicodec', 'dac'
        self._cuda_audio_used = False  # Set once after audio CUDA preprocessing; never cleared
        self._spark_tts_repo_dir = None  # Downloaded Spark-TTS repo path (for BiCodecTokenizer)
        self.model_name = None

        # Training metrics tracking
        self.training_start_time: Optional[float] = None
        self.batch_size: Optional[int] = None
        self.max_seq_length: Optional[int] = None
        self.gradient_accumulation_steps: Optional[int] = None

        # Thread safety
        self._lock = threading.Lock()

        # Training context for later transfer
        self.training_context = {
            "base_model_name": None,
            "output_dir": None,
            "is_lora": True,  # Default to LoRA
        }

    def pre_detect_and_load_tokenizer(
        self,
        model_name: str,
        max_seq_length: int = 2048,
        hf_token: Optional[str] = None,
        is_dataset_image: bool = False,
        is_dataset_audio: bool = False,
        trust_remote_code: bool = False,
    ) -> None:
        """Lightweight detection and tokenizer load — no model weights, no VRAM.

        Sets is_vlm, _audio_type, is_audio_vlm, model_name and loads a lightweight
        tokenizer for dataset formatting. Call before load_and_format_dataset() so
        the dataset is processed before the training model loads (avoids VRAM
        contention). load_model() later re-detects and loads the full model +
        tokenizer, overwriting the lightweight one set here.
        """
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.trust_remote_code = trust_remote_code

        if hf_token:
            os.environ["HF_TOKEN"] = hf_token

        # --- Detect audio type (reads config.json only, no VRAM) ---
        self._audio_type = detect_audio_type(model_name, hf_token)
        if self._audio_type == "audio_vlm":
            self.is_audio = False
            self.is_audio_vlm = is_dataset_audio
            self._audio_type = None
        else:
            self.is_audio = self._audio_type is not None
            self.is_audio_vlm = False

        if not self.is_audio and not self.is_audio_vlm:
            self._cuda_audio_used = False

        # --- Detect VLM ---
        vision = is_vision_model(model_name, hf_token = hf_token) if not self.is_audio else False
        self.is_vlm = not self.is_audio_vlm and vision and is_dataset_image

        logger.info(
            "pre_detect: audio_type=%s, is_audio=%s, is_audio_vlm=%s, is_vlm=%s",
            self._audio_type,
            self.is_audio,
            self.is_audio_vlm,
            self.is_vlm,
        )

        # --- Load lightweight tokenizer/processor (CPU only, no VRAM) ---
        # Whisper needs AutoProcessor; others use AutoTokenizer (CSM loads its
        # own processor inline).
        if self._audio_type == "whisper":
            from transformers import AutoProcessor
            self.tokenizer = AutoProcessor.from_pretrained(
                model_name,
                trust_remote_code = trust_remote_code,
                token = hf_token,
            )
        else:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code = trust_remote_code,
                token = hf_token,
            )

        logger.info("Pre-loaded tokenizer for %s", model_name)

    def add_progress_callback(self, callback: Callable[[TrainingProgress], None]):
        """Add callback for training progress updates"""
        self.progress_callbacks.append(callback)

    def _update_progress(self, **kwargs):
        """Update training progress and notify callbacks"""
        with self._lock:
            for key, value in kwargs.items():
                if hasattr(self.training_progress, key):
                    setattr(self.training_progress, key, value)

            for callback in self.progress_callbacks:
                try:
                    callback(self.training_progress)
                except Exception as e:
                    logger.error(f"Error in progress callback: {e}")

    def _create_progress_callback(self):
        """Create a TrainerCallback for progress tracking. Reused by all training branches."""
        from transformers import TrainerCallback

        trainer_ref = self

        class _ProgressCallback(TrainerCallback):
            def on_log(
                self,
                args,
                state,
                control,
                logs = None,
                **kwargs,
            ):
                if not logs:
                    return
                loss_value = logs.get("loss", logs.get("train_loss", None))
                current_step = state.global_step
                grad_norm = logs.get("grad_norm", None)

                elapsed_seconds = None
                if trainer_ref.training_start_time is not None:
                    elapsed_seconds = time.time() - trainer_ref.training_start_time

                eta_seconds = None
                if elapsed_seconds is not None and current_step > 0:
                    total_steps = trainer_ref.training_progress.total_steps
                    if total_steps > 0:
                        steps_remaining = total_steps - current_step
                        if steps_remaining > 0:
                            eta_seconds = (elapsed_seconds / current_step) * steps_remaining

                num_tokens = getattr(state, "num_input_tokens_seen", None)

                trainer_ref._update_progress(
                    step = current_step,
                    epoch = round(state.epoch, 2) if state.epoch else 0,
                    loss = loss_value,
                    learning_rate = logs.get("learning_rate", None),
                    elapsed_seconds = elapsed_seconds,
                    eta_seconds = eta_seconds,
                    grad_norm = grad_norm,
                    num_tokens = num_tokens,
                    eval_loss = logs.get("eval_loss", None),
                    status_message = "",
                )

            def on_epoch_end(self, args, state, control, **kwargs):
                trainer_ref._update_progress(epoch = state.epoch, step = state.global_step)

            def on_step_end(self, args, state, control, **kwargs):
                if trainer_ref.should_stop:
                    logger.info(f"Stop detected at step {state.global_step}\n")
                    control.should_training_stop = True
                    return control

        return _ProgressCallback()

    def _calculate_total_steps(self, num_samples, batch_size, grad_accum, num_epochs, max_steps):
        """Calculate total training steps from dataset size and training params."""
        if max_steps and max_steps > 0:
            return max_steps
        len_dataloader = math.ceil(num_samples / batch_size)
        steps_per_epoch = max(
            len_dataloader // grad_accum + int(len_dataloader % grad_accum > 0), 1
        )
        return steps_per_epoch * num_epochs

    def _build_audio_training_args(
        self,
        training_args,
        output_dir,
        *,
        extra_args = None,
    ):
        """Build the training args dict for audio branches: common config (batch
        size, lr, warmup, fp16/bf16, etc.) with per-branch overrides via extra_args.
        """
        batch_size = training_args.get("batch_size", 2)
        gradient_accumulation_steps = training_args.get("gradient_accumulation_steps", 4)
        warmup_steps_val = training_args.get("warmup_steps", 5)
        max_steps_val = training_args.get("max_steps", 0)
        learning_rate = training_args.get("learning_rate", 2e-4)
        weight_decay = training_args.get("weight_decay", 0.001)
        lr_scheduler_type = training_args.get("lr_scheduler_type", "linear")
        random_seed = training_args.get("random_seed", 3407)
        optim_value = training_args.get("optim", "adamw_8bit")

        config = {
            "per_device_train_batch_size": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "warmup_steps": warmup_steps_val if warmup_steps_val is not None else 5,
            "learning_rate": learning_rate,
            "fp16": not is_bfloat16_supported(),
            "bf16": is_bfloat16_supported(),
            "logging_steps": 1,
            "optim": optim_value,
            "weight_decay": weight_decay,
            "lr_scheduler_type": lr_scheduler_type,
            "seed": random_seed,
            "output_dir": output_dir,
            "report_to": _build_report_targets(training_args),
        }

        if training_args.get("enable_tensorboard", False):
            config["logging_dir"] = str(
                resolve_tensorboard_dir(training_args.get("tensorboard_dir"))
            )

        # max_steps vs epochs
        if max_steps_val and max_steps_val > 0:
            config["max_steps"] = max_steps_val
        else:
            config["num_train_epochs"] = training_args.get("num_epochs", 3)

        # save_steps
        save_steps_val = training_args.get("save_steps", 0)
        if save_steps_val and save_steps_val > 0:
            config["save_steps"] = save_steps_val
            config["save_strategy"] = "steps"

        # Apply per-branch overrides
        if extra_args:
            config.update(extra_args)

        return config

    def _finalize_training(
        self,
        output_dir,
        label = "",
    ):
        """Save model after training and update progress. Used by all training branches."""
        if self.should_stop and self.save_on_stop:
            self.trainer._save_checkpoint(self.trainer.model, trial = None)
            self.trainer.save_model()
            self.tokenizer.save_pretrained(output_dir)
            self._patch_adapter_config(output_dir)
            msg = f"{label} training stopped" if label else "Training stopped"
            logger.info(f"\n{msg}. Model saved to {output_dir}\n")
            self._update_progress(
                is_training = False,
                status_message = f"Training stopped. Model saved to {output_dir}",
            )
        elif self.should_stop:
            msg = f"{label} training cancelled" if label else "Training cancelled"
            logger.info(f"\n{msg}.\n")
            self._update_progress(is_training = False, status_message = "Training cancelled.")
        else:
            self.trainer.save_model()
            self.tokenizer.save_pretrained(output_dir)
            self._patch_adapter_config(output_dir)
            msg = f"{label} training completed" if label else "Training completed"
            logger.info(f"\n{msg}! Model saved to {output_dir}\n")
            self._update_progress(
                is_training = False,
                is_completed = True,
                status_message = f"Training completed! Model saved to {output_dir}",
            )

    def _cleanup_audio_artifacts(self):
        """Remove sys.path/sys.modules entries from previous audio preprocessing.

        After audio training, cloned repo dirs (OuteTTS, Spark-TTS) and heavy
        modules (snac, whisper, sparktts, outetts) linger; the next
        dataset.map(num_proc=N) forks children that inherit this stale state and
        deadlock.
        """
        # Remove cloned audio repo paths from sys.path
        base_dir = os.path.dirname(os.path.abspath(__file__))
        audio_paths = [
            os.path.join(base_dir, "inference", "OuteTTS"),  # DAC/OuteTTS
        ]
        # Spark-TTS path is relative to the downloaded repo
        if self._spark_tts_repo_dir:
            spark_code_dir = os.path.join(os.path.dirname(self._spark_tts_repo_dir), "Spark-TTS")
            audio_paths.append(spark_code_dir)

        removed_paths = []
        for path in audio_paths:
            if path in sys.path:
                sys.path.remove(path)
                removed_paths.append(path)

        # Remove stale audio modules from sys.modules
        prefixes = ("snac", "whisper", "sparktts", "outetts")
        removed_modules = [key for key in sys.modules if key.startswith(prefixes)]
        for key in removed_modules:
            del sys.modules[key]

        if removed_paths or removed_modules:
            logger.info(
                f"Cleaned up audio artifacts: {len(removed_paths)} paths, "
                f"{len(removed_modules)} modules\n"
            )

    def _resolve_audio_columns(
        self,
        dataset,
        custom_format_mapping: dict = None,
    ):
        """Resolve audio/text/speaker columns from user mapping or fallback.

        Returns dict with keys audio_col, text_col, speaker_col (may be None).
        """
        cols = dataset.column_names

        if custom_format_mapping:
            audio_col = None
            text_col = None
            speaker_col = None
            for col, role in custom_format_mapping.items():
                if role == "audio":
                    audio_col = col
                elif role == "text":
                    text_col = col
                elif role == "speaker_id":
                    speaker_col = col
            # Use mapping if both required columns exist
            if audio_col and audio_col in cols and text_col and text_col in cols:
                return {
                    "audio_col": audio_col,
                    "text_col": text_col,
                    "speaker_col": speaker_col,
                }

        # Hardcoded fallback
        audio_col = next((c for c in cols if c.lower() in ("audio", "speech")), None)
        text_col = next(
            (c for c in cols if c.lower() in ("text", "sentence", "transcript", "transcription")),
            None,
        )

        speaker_col = None
        if "source" in cols:
            speaker_col = "source"
        elif "speaker_id" in cols:
            speaker_col = "speaker_id"

        return {
            "audio_col": audio_col,
            "text_col": text_col,
            "speaker_col": speaker_col,
        }

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
        """Load model for training (supports both text and vision models)"""
        self.load_in_4bit = load_in_4bit  # For training_meta.json
        self.trust_remote_code = trust_remote_code  # For AutoProcessor etc. used during training
        try:
            if self.model is not None:
                del self.model
            if self.tokenizer is not None:
                del self.tokenizer

            if self.trainer is not None:
                del self.trainer

            logger.info("\nClearing GPU memory before training...")
            clear_gpu_cache()

            # Clean up sys.path/sys.modules from previous audio preprocessing
            # to prevent deadlocks when forking dataset.map() workers
            self._cleanup_audio_artifacts()

            # Reload Unsloth-patched modeling modules before clearing the cache.
            # __UNSLOTH_PATCHED__ blocks re-compilation, so clearing the disk
            # cache alone would leave files missing; reloading restores original
            # class defs so Unsloth re-compiles cleanly.
            import importlib

            for _key, _mod in list(sys.modules.items()):
                if "transformers.models." in _key and ".modeling_" in _key:
                    if hasattr(_mod, "__UNSLOTH_PATCHED__"):
                        try:
                            importlib.reload(_mod)
                        except Exception:
                            pass  # Non-critical — Unsloth handles stale modules

            # Remove stale compiled cache so the new model gets a fresh one
            from utils.cache_cleanup import clear_unsloth_compiled_cache

            _preserve = ["Unsloth*Trainer.py"] if sys.platform in ("win32", "darwin") else None
            clear_unsloth_compiled_cache(preserve_patterns = _preserve)
            # Detect audio model type dynamically (config.json + tokenizer)
            self._audio_type = detect_audio_type(model_name, hf_token)
            # audio_vlm is detected as an audio_type now; handle separately
            if self._audio_type == "audio_vlm":
                self.is_audio = False
                self.is_audio_vlm = is_dataset_audio  # Only use audio VLM path if dataset has audio
                self._audio_type = None
            else:
                self.is_audio = self._audio_type is not None
                self.is_audio_vlm = False

            if not self.is_audio and not self.is_audio_vlm:
                self._cuda_audio_used = False

            # VLM: vision model + image dataset (mutually exclusive with audio)
            vision = is_vision_model(model_name, hf_token = hf_token) if not self.is_audio else False
            self.is_vlm = not self.is_audio_vlm and vision and is_dataset_image
            self.model_name = model_name
            self.max_seq_length = max_seq_length

            logger.info(
                f"Audio type: {self._audio_type}, is_audio: {self.is_audio}, is_audio_vlm: {self.is_audio_vlm}"
            )
            logger.info(f"Dataset has images: {is_dataset_image}, audio: {is_dataset_audio}")
            logger.info(f"Using VLM path: {self.is_vlm}")

            # Reset training state for new run
            self._update_progress(
                is_training = True,
                is_completed = False,
                error = None,
                step = 0,
                loss = 0.0,
                epoch = 0,
            )

            # Update UI with loading message
            model_display = model_name.split("/")[-1] if "/" in model_name else model_name
            model_type_label = "audio" if self.is_audio else ("vision" if self.is_vlm else "text")
            self._update_progress(
                status_message = f"Loading {model_type_label} model... {model_display}"
            )

            logger.info(f"\nLoading {model_type_label} model: {model_name}")

            if hf_token:
                os.environ["HF_TOKEN"] = hf_token

            # Proactive gated-model check before from_pretrained (catches all
            # gated/private models). Skip when offline -- it uses the cache.
            if "/" in model_name and not _env_offline():
                try:
                    from huggingface_hub import model_info as hf_model_info
                    info = hf_model_info(model_name, token = hf_token or None)
                    # model_info succeeds for gated repos (metadata is public);
                    # info.gated tells us if files require acceptance/token.
                    if info.gated and not hf_token:
                        friendly = (
                            f"Access denied for '{model_name}'. This model is gated. "
                            f"Please add a Hugging Face token with access and try again."
                        )
                        logger.error(
                            f"Model '{model_name}' is gated (gated={info.gated}) and no HF token provided"
                        )
                        self._update_progress(error = friendly, is_training = False)
                        return False
                except Exception as gate_err:
                    from huggingface_hub.utils import (
                        GatedRepoError,
                        RepositoryNotFoundError,
                    )
                    if isinstance(gate_err, (GatedRepoError, RepositoryNotFoundError)):
                        friendly = (
                            f"Access denied for '{model_name}'. This model is gated or private. "
                            f"Please add a Hugging Face token with access and try again."
                        )
                        logger.error(f"Gated model check failed: {gate_err}")
                        self._update_progress(error = friendly, is_training = False)
                        return False

            device_map = get_device_map(gpu_ids)
            logger.info(
                f"Using device_map='{device_map}' ({get_visible_gpu_count()} GPU(s) visible)"
            )

            # AMD ROCm without native bf16 (e.g. RDNA2/gfx103x) crashes with an
            # LLVM error on the first bf16 kernel if dtype=None auto-picks bf16, so
            # force float16 there. NVIDIA keeps dtype=None so unsloth's auto-detect
            # (incl. FORCE_FLOAT32) is honored -- T4/V100 must NOT be coerced to
            # float16. Derive ROCm inline since hardware.IS_ROCM may be unset here.
            _is_rocm = (
                bool(getattr(torch.version, "hip", None)) or "rocm" in torch.__version__.lower()
            )
            _auto_dtype = torch.float16 if (_is_rocm and not is_bfloat16_supported()) else None

            # Branch based on model type
            if self._audio_type == "csm":
                # CSM: FastModel, auto_model=CsmForConditionalGeneration, load_in_4bit=False
                from unsloth import FastModel
                from transformers import CsmForConditionalGeneration

                self.model, self.tokenizer = FastModel.from_pretrained(
                    model_name = model_name,
                    max_seq_length = max_seq_length,
                    dtype = _auto_dtype,
                    auto_model = CsmForConditionalGeneration,
                    load_in_4bit = False,
                    device_map = device_map,
                    full_finetuning = full_finetuning,
                    token = hf_token,
                    trust_remote_code = trust_remote_code,
                )
                logger.info("Loaded CSM audio model")

            elif self._audio_type == "whisper":
                # Whisper: FastModel, auto_model=WhisperForConditionalGeneration, load_in_4bit=False
                from unsloth import FastModel
                from transformers import WhisperForConditionalGeneration

                self.model, self.tokenizer = FastModel.from_pretrained(
                    model_name = model_name,
                    dtype = _auto_dtype,
                    load_in_4bit = False,
                    device_map = device_map,
                    full_finetuning = full_finetuning,
                    auto_model = WhisperForConditionalGeneration,
                    whisper_language = "English",
                    whisper_task = "transcribe",
                    token = hf_token,
                    trust_remote_code = trust_remote_code,
                )
                # Generation settings (notebook lines 100-105)
                self.model.generation_config.language = "<|en|>"
                self.model.generation_config.task = "transcribe"
                self.model.config.suppress_tokens = []
                self.model.generation_config.forced_decoder_ids = None
                logger.info("Loaded Whisper audio model (FastModel)")

            elif self._audio_type == "snac":
                # Orpheus: language model with audio codec tokens
                self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                    model_name = model_name,
                    max_seq_length = max_seq_length,
                    dtype = _auto_dtype,
                    load_in_4bit = load_in_4bit,
                    device_map = device_map,
                    full_finetuning = full_finetuning,
                    token = hf_token,
                    trust_remote_code = trust_remote_code,
                )
                logger.info(f"Loaded {self._audio_type} audio model (FastLanguageModel)")

            elif self._audio_type == "bicodec":
                # Spark-TTS: download full repo (sparktts + BiCodec weights), then
                # load only the LLM subfolder. model_name may be
                # "Spark-TTS-0.5B/LLM" (YAML mapping) or "unsloth/Spark-TTS-0.5B".
                from unsloth import FastModel
                from huggingface_hub import snapshot_download

                if model_name.endswith("/LLM"):
                    # "Spark-TTS-0.5B/LLM" → parent="Spark-TTS-0.5B"
                    local_dir = model_name.rsplit("/", 1)[0]
                    hf_repo = f"unsloth/{local_dir}"
                    llm_path = model_name
                else:
                    # "unsloth/Spark-TTS-0.5B" → local_dir="Spark-TTS-0.5B"
                    hf_repo = model_name
                    local_dir = model_name.split("/")[-1]
                    llm_path = f"{local_dir}/LLM"

                repo_path = snapshot_download(hf_repo, local_dir = local_dir)
                self._spark_tts_repo_dir = os.path.abspath(repo_path)  # Absolute for sys.path
                llm_path = os.path.join(self._spark_tts_repo_dir, "LLM")

                self.model, self.tokenizer = FastModel.from_pretrained(
                    model_name = llm_path,
                    max_seq_length = max_seq_length,
                    dtype = torch.float32,  # Spark-TTS requires float32
                    load_in_4bit = False,
                    device_map = device_map,
                    full_finetuning = full_finetuning,
                    token = hf_token,
                    trust_remote_code = trust_remote_code,
                )
                logger.info("Loaded Spark-TTS (bicodec) model")

            elif self._audio_type == "dac":
                # OuteTTS: uses FastModel (not FastLanguageModel) with load_in_4bit=False
                from unsloth import FastModel
                self.model, self.tokenizer = FastModel.from_pretrained(
                    model_name,
                    max_seq_length = max_seq_length,
                    load_in_4bit = False,
                    device_map = device_map,
                    full_finetuning = full_finetuning,
                    token = hf_token,
                    trust_remote_code = trust_remote_code,
                )
                logger.info("Loaded OuteTTS (dac) model (FastModel)")

            elif self.is_audio_vlm:
                # Audio VLM: multimodal model trained on audio (e.g. Gemma 3N).
                # FastModel (general loader) returns (model, processor).
                from unsloth import FastModel
                self.model, self.tokenizer = FastModel.from_pretrained(
                    model_name = model_name,
                    max_seq_length = max_seq_length,
                    dtype = _auto_dtype,
                    load_in_4bit = load_in_4bit,
                    device_map = device_map,
                    full_finetuning = full_finetuning,
                    token = hf_token,
                    trust_remote_code = trust_remote_code,
                )
                logger.info("Loaded audio VLM model (FastModel)")

            elif self.is_vlm:
                # Load vision model - returns (model, tokenizer)
                self.model, self.tokenizer = FastVisionModel.from_pretrained(
                    model_name = model_name,
                    max_seq_length = max_seq_length,
                    dtype = _auto_dtype,
                    load_in_4bit = load_in_4bit,
                    device_map = device_map,
                    full_finetuning = full_finetuning,
                    token = hf_token,
                    trust_remote_code = trust_remote_code,
                )
                logger.info("Loaded vision model")

                # Did FastVisionModel return a Processor or a raw tokenizer?
                from transformers import ProcessorMixin

                tok = self.tokenizer
                has_image_proc = isinstance(tok, ProcessorMixin) or hasattr(tok, "image_processor")
                logger.info(f"\n[VLM Diagnostic] FastVisionModel returned: {type(tok).__name__}")
                logger.info(
                    f"[VLM Diagnostic] Is ProcessorMixin: {isinstance(tok, ProcessorMixin)}"
                )
                logger.info(
                    f"[VLM Diagnostic] Has image_processor: {hasattr(tok, 'image_processor')}"
                )
                logger.info(f"[VLM Diagnostic] Usable as vision processor: {has_image_proc}\n")
            else:
                # Load text model - returns (model, tokenizer)
                self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                    model_name = model_name,
                    max_seq_length = max_seq_length,
                    dtype = _auto_dtype,
                    load_in_4bit = load_in_4bit,
                    device_map = device_map,
                    full_finetuning = full_finetuning,
                    token = hf_token,
                    trust_remote_code = trust_remote_code,
                )
                logger.info("Loaded text model")

            raise_if_offloaded(self.model, device_map, "Unsloth training")

            if self.should_stop:
                return False

            if full_finetuning:
                # Enable training mode so all params are trainable (else frozen)
                self.model.for_training()

            self._update_progress(status_message = "Model loaded successfully")
            logger.info("Model loaded successfully")
            return True

        except OSError as e:
            if "could not get source code" in str(e) and not getattr(
                self, "_source_code_retried", False
            ):
                # Unsloth patching can leave stale state that breaks
                # inspect.getsource() when switching model families (e.g. gemma3 →
                # gemma3n); the first failure clears it, so a retry succeeds.
                self._source_code_retried = True
                logger.info(f"\n'could not get source code' — retrying once...\n")
                return self.load_model(
                    model_name = model_name,
                    max_seq_length = max_seq_length,
                    load_in_4bit = load_in_4bit,
                    hf_token = hf_token,
                    is_dataset_image = is_dataset_image,
                    is_dataset_audio = is_dataset_audio,
                    trust_remote_code = trust_remote_code,
                    full_finetuning = full_finetuning,
                    gpu_ids = gpu_ids,
                )
            error_msg = str(e)
            error_lower = error_msg.lower()
            if any(
                k in error_lower
                for k in (
                    "gated repo",
                    "access to it at",
                    "401",
                    "403",
                    "unauthorized",
                    "forbidden",
                )
            ):
                error_msg = (
                    f"Access denied for '{model_name}'. This model is gated or private. "
                    f"Please add a Hugging Face token with access and try again."
                )
            logger.error(f"Error loading model: {e}")
            self._update_progress(error = error_msg, is_training = False)
            return False
        except Exception as e:
            error_msg = str(e)
            # Surface a friendly message for gated/auth errors
            error_lower = error_msg.lower()
            if any(
                k in error_lower
                for k in (
                    "gated repo",
                    "access to it at",
                    "401",
                    "403",
                    "unauthorized",
                    "forbidden",
                )
            ):
                error_msg = (
                    f"Access denied for '{model_name}'. This model is gated or private. "
                    f"Please add a Hugging Face token with access and try again."
                )
            logger.error(f"Error loading model: {e}")
            self._update_progress(error = error_msg, is_training = False)
            return False
        finally:
            self._source_code_retried = False

    def prepare_model_for_training(
        self,
        use_lora: bool = True,
        # Vision-specific LoRA parameters (only used if is_vlm=True)
        finetune_vision_layers: bool = True,
        finetune_language_layers: bool = True,
        finetune_attention_modules: bool = True,
        finetune_mlp_modules: bool = True,
        # Standard LoRA parameters
        target_modules: list = None,
        lora_r: int = 16,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
        use_gradient_checkpointing: str = "unsloth",
        use_rslora: bool = False,
        use_loftq: bool = False,
        modules_to_save: list = None,
    ) -> bool:
        """
        Prepare model for training (with optional LoRA).
        """
        try:
            if self.model is None:
                raise ValueError("Model not loaded. Call load_model() first.")

            # Full finetuning - skip PEFT entirely
            if not use_lora:
                self._update_progress(status_message = "Full finetuning mode - no LoRA adapters")
                logger.info("Full finetuning mode - training all parameters\n")
                return True

            # LoRA/QLoRA - apply PEFT.
            # "all-linear" is a PEFT keyword targeting every linear layer.
            if isinstance(target_modules, list) and "all-linear" in target_modules:
                if len(target_modules) == 1:
                    target_modules = "all-linear"
                else:
                    target_modules = [m for m in target_modules if m != "all-linear"]
            elif target_modules is None or (
                isinstance(target_modules, list) and len(target_modules) == 0
            ):
                target_modules = [
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ]

            # Normalize gradient_checkpointing to True, False, or "unsloth"
            if isinstance(use_gradient_checkpointing, str):
                use_gradient_checkpointing = use_gradient_checkpointing.strip().lower()
                if use_gradient_checkpointing == "" or use_gradient_checkpointing == "unsloth":
                    use_gradient_checkpointing = "unsloth"
                elif use_gradient_checkpointing in ("true", "1", "yes"):
                    use_gradient_checkpointing = True
                elif use_gradient_checkpointing in ("false", "0", "no", "none", "off"):
                    use_gradient_checkpointing = False
                else:
                    # Invalid value -> "unsloth"
                    logger.warning(
                        f"Invalid gradient_checkpointing value: {use_gradient_checkpointing}, defaulting to 'unsloth'"
                    )
                    use_gradient_checkpointing = "unsloth"
            elif use_gradient_checkpointing not in (True, False, "unsloth"):
                # Invalid type/value -> "unsloth"
                logger.warning(
                    f"Invalid gradient_checkpointing type/value: {use_gradient_checkpointing}, defaulting to 'unsloth'"
                )
                use_gradient_checkpointing = "unsloth"

            # Verify model is loaded
            if self.model is None:
                error_msg = "Model is None - model was not loaded properly"
                logger.error(error_msg)
                self._update_progress(error = error_msg)
                return False

            # Check expected attributes
            if not hasattr(self.model, "config"):
                error_msg = (
                    "Model does not have config attribute - model may not be loaded correctly"
                )
                logger.error(error_msg)
                self._update_progress(error = error_msg)
                return False

            logger.info(f"Configuring LoRA adapters (r={lora_r}, alpha={lora_alpha})...\n")
            logger.info(
                f"Gradient checkpointing: {use_gradient_checkpointing} (type: {type(use_gradient_checkpointing).__name__})\n"
            )

            # Branch by model type: audio, audio_vlm, vision, or text
            if self._audio_type in ("csm", "bicodec", "dac") or self.is_audio_vlm:
                # Use FastModel.get_peft_model (codec audio + audio VLM)
                from unsloth import FastModel

                label = self._audio_type or "audio_vlm"
                logger.info(f"{label} LoRA configuration:")
                logger.info(f"  - Target modules: {target_modules}")
                if self.is_audio_vlm:
                    logger.info(f"  - Finetune vision layers: {finetune_vision_layers}")
                    logger.info(f"  - Finetune language layers: {finetune_language_layers}")
                    logger.info(f"  - Finetune attention modules: {finetune_attention_modules}")
                    logger.info(f"  - Finetune MLP modules: {finetune_mlp_modules}")
                logger.info()

                peft_kwargs = dict(
                    r = lora_r,
                    target_modules = target_modules,
                    lora_alpha = lora_alpha,
                    lora_dropout = lora_dropout,
                    bias = "none",
                    use_gradient_checkpointing = use_gradient_checkpointing,
                    random_state = 3407,
                    use_rslora = use_rslora,
                    loftq_config = {"loftq_bits": 4, "loftq_iter": 1} if use_loftq else None,
                )
                # Audio VLM models support VLM-style layer selection
                if self.is_audio_vlm:
                    peft_kwargs.update(
                        finetune_vision_layers = finetune_vision_layers,
                        finetune_language_layers = finetune_language_layers,
                        finetune_attention_modules = finetune_attention_modules,
                        finetune_mlp_modules = finetune_mlp_modules,
                    )

                self.model = FastModel.get_peft_model(self.model, **peft_kwargs)

            elif self._audio_type == "whisper":
                # Whisper: FastModel.get_peft_model with task_type=None
                from unsloth import FastModel

                logger.info(f"Audio model (whisper) LoRA configuration:")
                logger.info(f"  - Target modules: {target_modules}\n")

                self.model = FastModel.get_peft_model(
                    self.model,
                    r = lora_r,
                    target_modules = target_modules,
                    lora_alpha = lora_alpha,
                    lora_dropout = lora_dropout,
                    bias = "none",
                    use_gradient_checkpointing = use_gradient_checkpointing,
                    random_state = 3407,
                    use_rslora = use_rslora,
                    loftq_config = {"loftq_bits": 4, "loftq_iter": 1} if use_loftq else None,
                    task_type = None,
                )

            elif self._audio_type == "snac":
                # Orpheus: FastLanguageModel.get_peft_model
                logger.info(f"Audio model ({self._audio_type}) LoRA configuration:")
                logger.info(f"  - Target modules: {target_modules}\n")

                self.model = FastLanguageModel.get_peft_model(
                    self.model,
                    r = lora_r,
                    target_modules = target_modules,
                    lora_alpha = lora_alpha,
                    lora_dropout = lora_dropout,
                    bias = "none",
                    use_gradient_checkpointing = use_gradient_checkpointing,
                    random_state = 3407,
                    use_rslora = use_rslora,
                    loftq_config = {"loftq_bits": 4, "loftq_iter": 1} if use_loftq else None,
                )

            elif self.is_vlm:
                # Vision model LoRA
                logger.info(f"Vision model LoRA configuration:")
                logger.info(f"  - Finetune vision layers: {finetune_vision_layers}")
                logger.info(f"  - Finetune language layers: {finetune_language_layers}")
                logger.info(f"  - Finetune attention modules: {finetune_attention_modules}")
                logger.info(f"  - Finetune MLP modules: {finetune_mlp_modules}\n")

                self.model = FastVisionModel.get_peft_model(
                    self.model,
                    finetune_vision_layers = finetune_vision_layers,
                    finetune_language_layers = finetune_language_layers,
                    finetune_attention_modules = finetune_attention_modules,
                    finetune_mlp_modules = finetune_mlp_modules,
                    r = lora_r,
                    target_modules = target_modules,
                    lora_alpha = lora_alpha,
                    lora_dropout = lora_dropout,
                    bias = "none",
                    use_gradient_checkpointing = use_gradient_checkpointing,
                    random_state = 3407,
                    use_rslora = use_rslora,
                    loftq_config = {"loftq_bits": 4, "loftq_iter": 1} if use_loftq else None,
                    modules_to_save = modules_to_save,
                )
            else:
                # Text model LoRA
                logger.info(f"Text model LoRA configuration:")
                logger.info(f"  - Target modules: {target_modules}\n")
                if modules_to_save:
                    logger.info(f"  - Modules to save: {modules_to_save}\n")

                self.model = FastLanguageModel.get_peft_model(
                    self.model,
                    r = lora_r,
                    target_modules = target_modules,
                    lora_alpha = lora_alpha,
                    lora_dropout = lora_dropout,
                    bias = "none",
                    use_gradient_checkpointing = use_gradient_checkpointing,
                    random_state = 3407,
                    use_rslora = use_rslora,
                    loftq_config = {"loftq_bits": 4, "loftq_iter": 1} if use_loftq else None,
                    modules_to_save = modules_to_save,
                )

            # Check if stopped during LoRA preparation
            if self.should_stop:
                logger.info("Stopped during LoRA configuration\n")
                return False

            self._update_progress(status_message = "LoRA adapters configured")
            logger.info("LoRA adapters configured successfully\n")
            return True

        except Exception as e:
            import traceback
            import sys

            error_details = (
                f"{type(e).__name__}: {str(e)}" if str(e) else f"{type(e).__name__} (no message)"
            )
            full_traceback = traceback.format_exc()
            logger.error(f"Error preparing model: {error_details}")
            logger.error(f"Full traceback:\n{full_traceback}")
            logger.info(f"\n[ERROR] Error preparing model: {error_details}")
            logger.info(f"[ERROR] Full traceback:\n{full_traceback}")
            self._update_progress(error = error_details)
            return False

    def _apply_csm_forward_fix(self):
        """Monkey-patch CsmForConditionalGeneration.forward for depth decoder kwargs.

        The original forward leaks raw **kwargs (num_items_in_batch, causal_mask,
        etc.) from Trainer/PEFT into the depth decoder, causing
        depth_decoder_loss=None and a 'Tensor + NoneType' crash. Patch at both
        instance and class level and strip non-TransformersKwargs params.
        """
        import torch
        import torch.nn as nn
        from transformers.models.csm.modeling_csm import (
            CsmForConditionalGeneration,
            CsmOutputWithPast,
        )

        base_csm = self.model.base_model.model  # CsmForConditionalGeneration

        # Original forward (@can_return_tuple wrapped version)
        _original_forward = CsmForConditionalGeneration.forward

        # Keys the depth decoder and its sub-layers understand
        _TRANSFORMERS_KWARGS = {
            "num_items_in_batch",
            "output_hidden_states",
            "output_attentions",
            "output_router_logits",
            "cu_seq_lens_q",
            "cu_seq_lens_k",
            "max_length_q",
            "max_length_k",
        }

        def _fixed_csm_forward(
            self,
            input_ids = None,
            input_values = None,
            attention_mask = None,
            input_values_cutoffs = None,
            position_ids = None,
            past_key_values = None,
            inputs_embeds = None,
            labels = None,
            use_cache = None,
            cache_position = None,
            logits_to_keep = 0,
            **kwargs,
        ):
            # Strip non-standard kwargs from Unsloth/PEFT.
            output_attentions = kwargs.pop("output_attentions", None)
            output_hidden_states = kwargs.pop("output_hidden_states", None)
            kwargs.pop("return_dict", None)
            kwargs.pop("causal_mask", None)
            kwargs.pop("num_logits_to_keep", None)
            kwargs.pop("task_ids", None)

            # Only keep recognized TransformersKwargs
            clean_kwargs = {k: v for k, v in kwargs.items() if k in _TRANSFORMERS_KWARGS}

            if input_ids is not None and input_ids.ndim == 2:
                merged = self._merge_input_ids_with_input_values(
                    input_ids, input_values, input_values_cutoffs, labels
                )
                inputs_embeds = merged["inputs_embeds"]
                labels = merged["labels"]
                input_ids = None

            backbone_outputs = self.backbone_model(
                input_ids = input_ids,
                attention_mask = attention_mask,
                position_ids = position_ids,
                past_key_values = past_key_values,
                inputs_embeds = inputs_embeds,
                use_cache = use_cache,
                cache_position = cache_position,
                output_attentions = output_attentions,
                output_hidden_states = output_hidden_states,
                **clean_kwargs,
            )

            backbone_hidden_states = backbone_outputs[0]
            slice_indices = (
                slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
            )
            backbone_logits = self.lm_head(backbone_hidden_states[:, slice_indices, :])

            loss = None
            backbone_loss = None
            depth_decoder_loss = None
            depth_decoder_outputs = None
            if labels is not None:
                backbone_labels = labels[:, :, 0]
                backbone_loss = self.loss_function(
                    logits = backbone_logits,
                    labels = backbone_labels,
                    vocab_size = self.config.vocab_size,
                    **clean_kwargs,
                )

                train_mask = ~(labels[:, :, 1:] == -100).all(dim = -1)
                depth_decoder_input_ids = labels[train_mask][..., : self.config.num_codebooks - 1]
                depth_decoder_input_ids = nn.functional.pad(
                    depth_decoder_input_ids, (1, 0), value = 0
                )

                train_idxs = train_mask.nonzero(as_tuple = True)
                backbone_last_hidden_states = backbone_hidden_states[
                    train_idxs[0], train_idxs[1] - 1, :
                ]
                depth_decoder_labels = labels[train_mask]

                # Scale num_items_in_batch for the depth decoder's 31 codebooks.
                dd_kwargs = clean_kwargs.copy()
                if "num_items_in_batch" in dd_kwargs:
                    dd_kwargs["num_items_in_batch"] = dd_kwargs["num_items_in_batch"] * (
                        self.config.num_codebooks - 1
                    )

                depth_decoder_outputs = self.depth_decoder(
                    input_ids = depth_decoder_input_ids,
                    backbone_last_hidden_state = backbone_last_hidden_states,
                    use_cache = False,
                    return_dict = True,
                    labels = depth_decoder_labels,
                    output_attentions = output_attentions,
                    output_hidden_states = output_hidden_states,
                    **dd_kwargs,
                )

                depth_decoder_loss = depth_decoder_outputs.loss
                if depth_decoder_loss is None:
                    logger.warning(
                        "CSM depth_decoder_loss is None! "
                        f"labels shape={depth_decoder_labels.shape}, "
                        f"train_mask sum={train_mask.sum().item()}"
                    )
                    # Fallback: use only backbone loss to avoid crashing
                    loss = backbone_loss
                else:
                    loss = backbone_loss + depth_decoder_loss

            return CsmOutputWithPast(
                loss = loss,
                backbone_loss = backbone_loss,
                depth_decoder_loss = depth_decoder_loss,
                logits = backbone_logits,
                past_key_values = backbone_outputs.past_key_values,
                hidden_states = backbone_outputs.hidden_states,
                attentions = backbone_outputs.attentions,
                depth_decoder_logits = (
                    depth_decoder_outputs.logits if depth_decoder_outputs else None
                ),
                depth_decoder_past_key_values = (
                    depth_decoder_outputs.past_key_values if depth_decoder_outputs else None
                ),
                depth_decoder_hidden_states = (
                    depth_decoder_outputs.hidden_states if depth_decoder_outputs else None
                ),
                depth_decoder_attentions = (
                    depth_decoder_outputs.attentions if depth_decoder_outputs else None
                ),
            )

        # Instance level: catches BaseTuner.forward -> self.model.forward().
        base_csm.forward = types.MethodType(_fixed_csm_forward, base_csm)
        # Class level: catches paths resolving through the class dict.
        CsmForConditionalGeneration.forward = _fixed_csm_forward
        logger.info("Applied CSM forward fix (class + instance level)\n")

    def _preprocess_csm_dataset(
        self,
        dataset,
        custom_format_mapping = None,
    ):
        """Preprocess dataset for CSM TTS training (exact notebook copy)."""
        from transformers import AutoProcessor
        from datasets import Audio
        import torch

        processor = AutoProcessor.from_pretrained(
            self.model_name,
            trust_remote_code = getattr(self, "trust_remote_code", False),
        )

        # Strip pad_to_multiple_of from tokenizer init_kwargs — some fine-tuned
        # models save it in tokenizer_config.json, and _merge_kwargs leaks it into
        # audio_kwargs where EncodecFeatureExtractor rejects it.
        processor.tokenizer.init_kwargs.pop("pad_to_multiple_of", None)

        # Resolve columns from user mapping or hardcoded fallback
        resolved = self._resolve_audio_columns(dataset, custom_format_mapping)
        audio_col = resolved["audio_col"]
        text_col = resolved["text_col"]
        speaker_key = resolved["speaker_col"]

        if audio_col is None:
            raise ValueError(f"No audio column found in dataset. Columns: {dataset.column_names}")
        if text_col is None:
            raise ValueError(f"No text column found in dataset. Columns: {dataset.column_names}")
        if speaker_key is None:
            logger.info("No speaker found, adding default 'source' of 0 for all examples\n")
            dataset = dataset.add_column("source", ["0"] * len(dataset))
            speaker_key = "source"

        logger.info(
            f"CSM preprocessing: audio_col='{audio_col}', text_col='{text_col}', speaker_key='{speaker_key}'\n"
        )

        dataset = dataset.cast_column(audio_col, Audio(sampling_rate = 24000))

        required_keys = [
            "input_ids",
            "attention_mask",
            "labels",
            "input_values",
            "input_values_cutoffs",
        ]

        self._update_progress(status_message = "Preprocessing CSM dataset...")
        processed_examples = []
        skipped = 0
        for idx in range(len(dataset)):
            if self.should_stop:
                logger.info("Stopped during CSM preprocessing\n")
                break

            example = dataset[idx]
            try:
                conversation = [
                    {
                        "role": str(example[speaker_key]),
                        "content": [
                            {"type": "text", "text": example.get(text_col, "")},
                            {"type": "audio", "path": example[audio_col]["array"]},
                        ],
                    }
                ]
                # pad_to_multiple_of omitted: _merge_kwargs leaks it to
                # EncodecFeatureExtractor, which rejects it.
                model_inputs = processor.apply_chat_template(
                    conversation,
                    tokenize = True,
                    return_dict = True,
                    output_labels = True,
                    text_kwargs = {
                        "padding": "max_length",
                        "max_length": 256,
                        "padding_side": "right",
                    },
                    audio_kwargs = {
                        "sampling_rate": 24_000,
                        "max_length": 240001,
                        "padding": "max_length",
                    },
                    common_kwargs = {"return_tensors": "pt"},
                )

                out = {}
                for k in required_keys:
                    if k not in model_inputs:
                        raise KeyError(f"Missing required key '{k}' in model outputs")
                    out[k] = model_inputs[k][0]

                if not all(isinstance(out[k], torch.Tensor) for k in out):
                    skipped += 1
                    continue

                processed_examples.append(out)

            except Exception as e:
                logger.warning(f"Error processing CSM example {idx}: {e}")
                skipped += 1
                continue

            if (idx + 1) % 100 == 0:
                self._update_progress(
                    status_message = f"Preprocessing CSM... {idx + 1}/{len(dataset)}"
                )

        if not processed_examples:
            raise ValueError(f"No valid examples after CSM preprocessing (skipped {skipped})")

        result_dataset = Dataset.from_list(processed_examples)
        logger.info(
            f"CSM preprocessing complete: {len(result_dataset)} examples " f"({skipped} skipped)\n"
        )
        return result_dataset

    def _format_audio_vlm_dataset(
        self,
        dataset,
        custom_format_mapping = None,
    ):
        """Format dataset as audio chat messages for multimodal models (e.g. Gemma 3N).

        Expects columns audio (Audio), text (str). Produces a messages column
        with system/user/assistant chat format.
        """
        from datasets import Audio

        resolved = self._resolve_audio_columns(dataset, custom_format_mapping)
        audio_col = resolved["audio_col"]
        text_col = resolved["text_col"]
        if not audio_col or not text_col:
            raise ValueError(
                f"Audio VLM dataset needs 'audio' and 'text' columns, got: {dataset.column_names}"
            )

        # Store resolved audio column name for the collator closure
        self._audio_vlm_audio_col = audio_col

        # Cast audio to 16kHz (standard for speech models)
        dataset = dataset.cast_column(audio_col, Audio(sampling_rate = 16000))

        def format_messages(samples):
            formatted = {"messages": []}
            for idx in range(len(samples[audio_col])):
                audio = samples[audio_col][idx]["array"]
                label = str(samples[text_col][idx])
                message = [
                    {
                        "role": "system",
                        "content": [
                            {
                                "type": "text",
                                "text": "You are an assistant that transcribes speech accurately.",
                            }
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "audio", "audio": audio},
                            {"type": "text", "text": "Please transcribe this audio."},
                        ],
                    },
                    {"role": "assistant", "content": [{"type": "text", "text": label}]},
                ]
                formatted["messages"].append(message)
            return formatted

        self._update_progress(status_message = "Formatting audio VLM dataset...")
        dataset = dataset.map(
            format_messages,
            batched = True,
            batch_size = 4,
            num_proc = dataset_map_num_proc(4),
        )
        logger.info(f"Audio VLM dataset formatted: {len(dataset)} examples\n")
        return dataset

    def _preprocess_snac_dataset(
        self,
        dataset,
        custom_format_mapping = None,
    ):
        """Preprocess dataset for Orpheus TTS training with SNAC codec.

        Mirrors Orpheus_(3B)-TTS.ipynb: encode audio with SNAC (24kHz, 3
        hierarchical layers), interleave 7 codes per frame, wrap with Orpheus
        special tokens, train on full sequence (no label masking).
        """
        import torch
        import torchaudio.transforms as T

        SNAC_MODEL_NAME = "hubertsiuzdak/snac_24khz"
        SNAC_SAMPLE_RATE = 24000
        device = "cuda" if torch.cuda.is_available() else "cpu"
        max_length = self.max_seq_length or 2048
        tokenizer = self.tokenizer

        # Orpheus special token IDs (hardcoded in tokenizer vocabulary)
        START_OF_HUMAN = 128259
        END_OF_HUMAN = 128260
        START_OF_AI = 128261
        END_OF_AI = 128262
        START_OF_SPEECH = 128257
        END_OF_SPEECH = 128258
        END_OF_TEXT = 128009
        AUDIO_OFFSET = 128266

        resolved = self._resolve_audio_columns(dataset, custom_format_mapping)
        audio_col = resolved["audio_col"]
        text_col = resolved["text_col"]
        speaker_col = resolved["speaker_col"]
        has_source = speaker_col is not None
        if not audio_col or not text_col:
            raise ValueError(
                f"SNAC dataset needs 'audio' and 'text' columns, got: {dataset.column_names}"
            )

        # Cast audio so datasets 4.x AudioDecoder objects decode to dicts
        from datasets import Audio

        dataset = dataset.cast_column(audio_col, Audio(sampling_rate = SNAC_SAMPLE_RATE))

        # Sample rate from first example (after cast, always SNAC_SAMPLE_RATE)
        first_audio = dataset[0][audio_col]
        ds_sample_rate = (
            first_audio.get("sampling_rate", SNAC_SAMPLE_RATE)
            if isinstance(first_audio, dict)
            else SNAC_SAMPLE_RATE
        )

        # Load SNAC codec model
        self._update_progress(status_message = "Loading SNAC codec model...")
        logger.info("Loading SNAC codec model...\n")
        from snac import SNAC

        snac_model = SNAC.from_pretrained(SNAC_MODEL_NAME)
        snac_model = snac_model.to(device).eval()

        # Resample transform (created once)
        resample_transform = (
            T.Resample(orig_freq = ds_sample_rate, new_freq = SNAC_SAMPLE_RATE)
            if ds_sample_rate != SNAC_SAMPLE_RATE
            else None
        )

        self._update_progress(status_message = "Encoding audio with SNAC...")
        logger.info(
            f"SNAC preprocessing: audio_col='{audio_col}', text_col='{text_col}', "
            f"has_source={has_source}, ds_sample_rate={ds_sample_rate}\n"
        )

        processed_examples = []
        skipped = 0
        for idx in range(len(dataset)):
            if self.should_stop:
                logger.info("Stopped during SNAC preprocessing\n")
                break

            example = dataset[idx]
            try:
                text = example.get(text_col)
                if not text:
                    skipped += 1
                    continue

                audio_data = example.get(audio_col)
                if audio_data is None or audio_data.get("array") is None:
                    skipped += 1
                    continue

                # --- Encode audio with SNAC (notebook 122-142) ---
                waveform = (
                    torch.from_numpy(audio_data["array"]).unsqueeze(0).to(dtype = torch.float32)
                )
                if resample_transform is not None:
                    waveform = resample_transform(waveform)

                waveform = waveform.unsqueeze(0).to(device)
                with torch.inference_mode():
                    codes = snac_model.encode(waveform)

                # Interleave 7 codes per frame with layer offsets (notebook 134-142)
                all_codes = []
                for i in range(codes[0].shape[1]):
                    all_codes.append(codes[0][0][i].item() + AUDIO_OFFSET)
                    all_codes.append(codes[1][0][2 * i].item() + AUDIO_OFFSET + 4096)
                    all_codes.append(codes[2][0][4 * i].item() + AUDIO_OFFSET + (2 * 4096))
                    all_codes.append(codes[2][0][(4 * i) + 1].item() + AUDIO_OFFSET + (3 * 4096))
                    all_codes.append(codes[1][0][(2 * i) + 1].item() + AUDIO_OFFSET + (4 * 4096))
                    all_codes.append(codes[2][0][(4 * i) + 2].item() + AUDIO_OFFSET + (5 * 4096))
                    all_codes.append(codes[2][0][(4 * i) + 3].item() + AUDIO_OFFSET + (6 * 4096))

                if len(all_codes) == 0:
                    skipped += 1
                    continue

                # Dedup consecutive frames with same first code (notebook 185-207)
                deduped = all_codes[:7]
                for i in range(7, len(all_codes), 7):
                    if all_codes[i] != deduped[-7]:
                        deduped.extend(all_codes[i : i + 7])
                all_codes = deduped

                # --- Build text tokens (notebook 217-224) ---
                text_prompt = (
                    f"{example[speaker_col]}: {text}"
                    if has_source and example.get(speaker_col)
                    else text
                )
                text_ids = tokenizer.encode(text_prompt, add_special_tokens = True)
                text_ids.append(END_OF_TEXT)

                # --- Build full input_ids (notebook 225-234) ---
                input_ids = (
                    [START_OF_HUMAN]
                    + text_ids
                    + [END_OF_HUMAN]
                    + [START_OF_AI]
                    + [START_OF_SPEECH]
                    + all_codes
                    + [END_OF_SPEECH]
                    + [END_OF_AI]
                )

                # Truncate to max_length
                input_ids = input_ids[:max_length]

                # Labels = input_ids (no masking; Orpheus trains full sequence)
                labels = list(input_ids)
                attention_mask = [1] * len(input_ids)

                processed_examples.append(
                    {
                        "input_ids": input_ids,
                        "labels": labels,
                        "attention_mask": attention_mask,
                    }
                )

            except Exception as e:
                logger.warning(f"Error processing SNAC example {idx}: {e}")
                skipped += 1
                continue

            # Progress update every 100 examples
            if (idx + 1) % 100 == 0:
                self._update_progress(status_message = f"Encoding audio... {idx + 1}/{len(dataset)}")

        # Free SNAC model from GPU
        logger.info("Freeing SNAC codec model from GPU...\n")
        snac_model.to("cpu")
        del snac_model

        gc.collect()
        torch.cuda.empty_cache()
        self._cuda_audio_used = True

        if not processed_examples:
            raise ValueError(f"No valid examples after SNAC preprocessing (skipped {skipped})")

        result_dataset = Dataset.from_list(processed_examples)
        logger.info(
            f"SNAC preprocessing complete: {len(result_dataset)} examples " f"({skipped} skipped)\n"
        )
        return result_dataset

    def _preprocess_bicodec_dataset(
        self,
        dataset,
        custom_format_mapping = None,
    ):
        """Preprocess dataset for Spark-TTS training with BiCodec tokenizer.

        Mirrors Spark_TTS_(0_5B).ipynb: encode audio with BiCodec (semantic +
        global tokens), format as special-token text strings for SFTTrainer
        with dataset_text_field="text".
        """
        import torch
        import numpy as np
        import torchaudio.transforms as T

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # sparktts lives in the SparkAudio/Spark-TTS GitHub repo, not the HF model
        # repo. Clone if needed.
        spark_code_dir = os.path.join(os.path.dirname(self._spark_tts_repo_dir), "Spark-TTS")
        sparktts_pkg = os.path.join(spark_code_dir, "sparktts")
        if not os.path.isdir(sparktts_pkg):
            self._update_progress(status_message = "Cloning Spark-TTS code repo...")
            logger.info(f"Cloning SparkAudio/Spark-TTS to {spark_code_dir}...\n")
            subprocess.run(
                [
                    "git",
                    "clone",
                    "--depth",
                    "1",
                    "https://github.com/SparkAudio/Spark-TTS",
                    spark_code_dir,
                ],
                check = True,
                env = child_env_without_native_path_secret(),
                **_windows_hidden_subprocess_kwargs(),
            )

        if spark_code_dir not in sys.path:
            sys.path.insert(0, spark_code_dir)

        from sparktts.models.audio_tokenizer import BiCodecTokenizer
        from sparktts.utils.audio import audio_volume_normalize

        # Resolve audio and text columns
        resolved = self._resolve_audio_columns(dataset, custom_format_mapping)
        audio_col = resolved["audio_col"]
        text_col = resolved["text_col"]
        speaker_col = resolved["speaker_col"]
        has_source = speaker_col is not None
        if not audio_col or not text_col:
            raise ValueError(
                f"BiCodec dataset needs 'audio' and 'text' columns, got: {dataset.column_names}"
            )

        # Cast audio so datasets 4.x AudioDecoder objects decode to dicts.
        # Don't resample here — BiCodec's target_sr may differ; the loop does it.
        from datasets import Audio

        dataset = dataset.cast_column(audio_col, Audio())

        # Load BiCodec tokenizer
        self._update_progress(status_message = "Loading BiCodec tokenizer...")
        logger.info("Loading BiCodec tokenizer...\n")
        audio_tokenizer = BiCodecTokenizer(self._spark_tts_repo_dir, device)

        target_sr = audio_tokenizer.config["sample_rate"]

        self._update_progress(status_message = "Encoding audio with BiCodec...")
        logger.info(
            f"BiCodec preprocessing: audio_col='{audio_col}', text_col='{text_col}', "
            f"has_source={has_source}, target_sr={target_sr}\n"
        )

        def extract_wav2vec2_features(wavs: torch.Tensor) -> torch.Tensor:
            """Extract wav2vec2 features (average of layers 11, 14, 16)."""
            if wavs.shape[0] != 1:
                raise ValueError(f"Expected batch size 1, but got shape {wavs.shape}")
            wav_np = wavs.squeeze(0).cpu().numpy()

            processed = audio_tokenizer.processor(
                wav_np,
                sampling_rate = 16000,
                return_tensors = "pt",
                padding = True,
            )
            input_values = processed.input_values.to(audio_tokenizer.feature_extractor.device)
            model_output = audio_tokenizer.feature_extractor(input_values)

            if model_output.hidden_states is None:
                raise ValueError("Wav2Vec2Model did not return hidden states.")

            feats_mix = (
                model_output.hidden_states[11]
                + model_output.hidden_states[14]
                + model_output.hidden_states[16]
            ) / 3
            return feats_mix

        processed_examples = []
        skipped = 0
        for idx in range(len(dataset)):
            if self.should_stop:
                logger.info("Stopped during BiCodec preprocessing\n")
                break

            example = dataset[idx]
            try:
                text = example.get(text_col)
                if not text:
                    skipped += 1
                    continue

                audio_data = example.get(audio_col)
                if audio_data is None or audio_data.get("array") is None:
                    skipped += 1
                    continue

                audio_array = audio_data["array"]
                sampling_rate = audio_data.get("sampling_rate", target_sr)

                # Resample if needed
                if sampling_rate != target_sr:
                    resampler = T.Resample(orig_freq = sampling_rate, new_freq = target_sr)
                    audio_tensor_temp = torch.from_numpy(audio_array).float()
                    audio_array = resampler(audio_tensor_temp).numpy()

                # Volume normalize if configured
                if audio_tokenizer.config.get("volume_normalize", False):
                    audio_array = audio_volume_normalize(audio_array)

                # Get reference clip
                ref_wav_np = audio_tokenizer.get_ref_clip(audio_array)

                # Prepare tensors
                audio_tensor = torch.from_numpy(audio_array).unsqueeze(0).float().to(device)
                ref_wav_tensor = torch.from_numpy(ref_wav_np).unsqueeze(0).float().to(device)

                # Extract wav2vec2 features
                feat = extract_wav2vec2_features(audio_tensor)

                batch = {
                    "wav": audio_tensor,
                    "ref_wav": ref_wav_tensor,
                    "feat": feat.to(device),
                }

                # BiCodec tokenize
                semantic_token_ids, global_token_ids = audio_tokenizer.model.tokenize(batch)

                global_tokens = "".join(
                    [f"<|bicodec_global_{i}|>" for i in global_token_ids.squeeze().cpu().numpy()]
                )
                semantic_tokens = "".join(
                    [
                        f"<|bicodec_semantic_{i}|>"
                        for i in semantic_token_ids.squeeze().cpu().numpy()
                    ]
                )

                # Prefix text with source if available
                text_content = (
                    f"{example[speaker_col]}: {text}"
                    if has_source and example.get(speaker_col)
                    else text
                )

                formatted = "".join(
                    [
                        "<|task_tts|>",
                        "<|start_content|>",
                        text_content,
                        "<|end_content|>",
                        "<|start_global_token|>",
                        global_tokens,
                        "<|end_global_token|>",
                        "<|start_semantic_token|>",
                        semantic_tokens,
                        "<|end_semantic_token|>",
                        "<|im_end|>",
                    ]
                )

                processed_examples.append({"text": formatted})

            except Exception as e:
                logger.warning(f"Error processing BiCodec example {idx}: {e}")
                skipped += 1
                continue

            # Progress update every 100 examples
            if (idx + 1) % 100 == 0:
                self._update_progress(
                    status_message = f"Encoding audio with BiCodec... {idx + 1}/{len(dataset)}"
                )

        # Free BiCodec model from GPU
        logger.info("Freeing BiCodec tokenizer from GPU...\n")
        audio_tokenizer.model.cpu()
        audio_tokenizer.feature_extractor.cpu()
        del audio_tokenizer

        gc.collect()
        torch.cuda.empty_cache()
        self._cuda_audio_used = True

        if not processed_examples:
            raise ValueError(f"No valid examples after BiCodec preprocessing (skipped {skipped})")

        result_dataset = Dataset.from_list(processed_examples)
        logger.info(
            f"BiCodec preprocessing complete: {len(result_dataset)} examples "
            f"({skipped} skipped)\n"
        )
        # Debug: first example text (truncated)
        sample = result_dataset[0]["text"]
        logger.info(f"Sample text (first 200 chars): {sample[:200]}...\n")
        logger.info(f"Sample text length: {len(sample)} chars\n")
        return result_dataset

    def _preprocess_dac_dataset(
        self,
        dataset,
        custom_format_mapping = None,
    ):
        """Preprocess dataset for OuteTTS training with DAC codec.

        Mirrors Oute_TTS_(1B).ipynb DataCreationV3: Whisper for word timings,
        OuteTTS AudioProcessor for speaker representations, PromptProcessor for
        training prompts. Outputs text strings for SFTTrainer with
        dataset_text_field="text".
        """
        import io
        import tempfile
        import torch
        import numpy as np
        import soundfile as sf
        from datasets import Dataset as HFDataset
        from utils.paths import ensure_dir, tmp_root

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Clone OuteTTS repo (same as audio_codecs._load_dac)
        base_dir = os.path.dirname(os.path.abspath(__file__))
        outetts_code_dir = os.path.join(base_dir, "inference", "OuteTTS")
        outetts_pkg = os.path.join(outetts_code_dir, "outetts")
        if not os.path.isdir(outetts_pkg):
            self._update_progress(status_message = "Cloning OuteTTS code repo...")
            logger.info(f"Cloning edwko/OuteTTS to {outetts_code_dir}...\n")
            subprocess.run(
                [
                    "git",
                    "clone",
                    "--depth",
                    "1",
                    "https://github.com/edwko/OuteTTS",
                    outetts_code_dir,
                ],
                check = True,
                env = child_env_without_native_path_secret(),
                **_windows_hidden_subprocess_kwargs(),
            )
            for fpath in [
                os.path.join(outetts_pkg, "models", "gguf_model.py"),
                os.path.join(outetts_pkg, "interface.py"),
                os.path.join(outetts_pkg, "__init__.py"),
            ]:
                if os.path.exists(fpath):
                    os.remove(fpath)
                    logger.info(f"Removed {fpath}\n")

        if outetts_code_dir not in sys.path:
            sys.path.insert(0, outetts_code_dir)

        from outetts.version.v3.audio_processor import AudioProcessor
        from outetts.version.v3.prompt_processor import PromptProcessor
        from outetts.models.config import ModelConfig as OuteTTSModelConfig
        from outetts.utils.preprocessing import text_normalizations

        # Resolve audio and text columns
        resolved = self._resolve_audio_columns(dataset, custom_format_mapping)
        audio_col = resolved["audio_col"]
        text_col = resolved["text_col"]
        if not audio_col or not text_col:
            raise ValueError(
                f"DAC dataset needs 'audio' and 'text' columns, got: {dataset.column_names}"
            )

        # Cast audio to 24kHz (notebook: cast_column("audio", Audio(sampling_rate=24000)))
        from datasets import Audio

        dataset = dataset.cast_column(audio_col, Audio(sampling_rate = 24000))
        logger.info("Cast audio column to 24kHz\n")

        # Load Whisper for word timings
        self._update_progress(status_message = "Loading Whisper model for word timings...")
        logger.info("Loading Whisper model for word timings...\n")
        import whisper

        whisper_model = whisper.load_model("turbo", device = device)

        # Load OuteTTS AudioProcessor + PromptProcessor
        self._update_progress(status_message = "Loading OuteTTS AudioProcessor...")
        logger.info("Loading OuteTTS AudioProcessor...\n")
        model_tokenizer_path = "OuteAI/Llama-OuteTTS-1.0-1B"
        dummy_config = OuteTTSModelConfig(
            tokenizer_path = model_tokenizer_path,
            device = device,
            audio_codec_path = None,
        )
        audio_processor = AudioProcessor(config = dummy_config)
        prompt_processor = PromptProcessor(model_tokenizer_path)

        self._update_progress(status_message = "Preprocessing audio with OuteTTS...")
        logger.info(f"DAC preprocessing: audio_col='{audio_col}', text_col='{text_col}'\n")

        processed_examples = []
        skipped = 0
        for idx in range(len(dataset)):
            if self.should_stop:
                logger.info("Stopped during DAC preprocessing\n")
                break

            example = dataset[idx]
            try:
                text = example.get(text_col)
                if not text or not isinstance(text, str):
                    skipped += 1
                    continue

                audio_data = example.get(audio_col)
                if audio_data is None or audio_data.get("array") is None:
                    skipped += 1
                    continue

                audio_array = np.array(audio_data["array"], dtype = np.float32)
                sampling_rate = audio_data.get("sampling_rate", 24000)

                # Convert to WAV bytes (Whisper needs a file path)
                buf = io.BytesIO()
                sf.write(buf, audio_array, sampling_rate, format = "WAV", subtype = "FLOAT")
                buf.seek(0)
                audio_bytes = buf.getvalue()

                # 1. Get word timings from Whisper
                with tempfile.NamedTemporaryFile(
                    suffix = ".wav",
                    delete = False,
                    dir = str(ensure_dir(tmp_root())),
                ) as tmp:
                    tmp.write(audio_bytes)
                    tmp.flush()
                    tmp_path = tmp.name
                try:
                    whisper_result = whisper_model.transcribe(tmp_path, word_timestamps = True)
                finally:
                    Path(tmp_path).unlink(missing_ok = True)

                normalized_transcript = text_normalizations(text)
                words_with_timings = []
                if whisper_result and "segments" in whisper_result:
                    for segment in whisper_result["segments"]:
                        for word_info in segment.get("words", []):
                            cleaned = word_info["word"].strip()
                            if cleaned:
                                words_with_timings.append(
                                    {
                                        "word": cleaned,
                                        "start": float(word_info["start"]),
                                        "end": float(word_info["end"]),
                                    }
                                )

                if not words_with_timings:
                    skipped += 1
                    continue

                # 2. Create speaker representation with AudioProcessor
                speaker_data_dict = {
                    "audio": {"bytes": audio_bytes},
                    "text": normalized_transcript,
                    "words": words_with_timings,
                }
                speaker = audio_processor.create_speaker_from_dict(speaker_data_dict)
                if speaker is None:
                    skipped += 1
                    continue

                # 3. Get training prompt from PromptProcessor
                prompt = prompt_processor.get_training_prompt(speaker)
                if prompt:
                    processed_examples.append({"text": prompt})

            except Exception as e:
                logger.warning(f"Error processing DAC example {idx}: {e}")
                skipped += 1
                continue

            if (idx + 1) % 100 == 0:
                self._update_progress(
                    status_message = f"Preprocessing audio with OuteTTS... {idx + 1}/{len(dataset)}"
                )

        # Free Whisper from GPU (notebook: whisper_model.to('cpu'))
        logger.info("Moving Whisper model to CPU...\n")
        whisper_model.to("cpu")
        del whisper_model
        del audio_processor
        del prompt_processor

        gc.collect()
        torch.cuda.empty_cache()
        self._cuda_audio_used = True

        if not processed_examples:
            raise ValueError(f"No valid examples after DAC preprocessing (skipped {skipped})")

        result_dataset = HFDataset.from_list(processed_examples)
        logger.info(
            f"DAC preprocessing complete: {len(result_dataset)} examples " f"({skipped} skipped)\n"
        )
        sample = result_dataset[0]["text"]
        logger.info(f"Sample text (first 200 chars): {sample[:200]}...\n")
        return result_dataset

    def _preprocess_whisper_dataset(
        self,
        dataset,
        eval_split = None,
        custom_format_mapping = None,
    ):
        """Preprocess dataset for Whisper speech-to-text training.

        Mirrors Whisper.ipynb: extract audio features with Whisper's feature
        extractor, tokenize text labels. Returns (train_data, eval_data),
        each a list of dicts with 'input_features' and 'labels'.
        """
        from datasets import Audio

        WHISPER_SAMPLE_RATE = 16000

        resolved = self._resolve_audio_columns(dataset, custom_format_mapping)
        audio_col = resolved["audio_col"]
        text_col = resolved["text_col"]
        if not audio_col or not text_col:
            raise ValueError(
                f"Whisper dataset needs 'audio' and 'text' columns, got: {dataset.column_names}"
            )

        # Cast audio to 16kHz (Whisper's expected sample rate)
        dataset = dataset.cast_column(audio_col, Audio(sampling_rate = WHISPER_SAMPLE_RATE))

        # Train/eval split (notebook does dataset.train_test_split)
        eval_dataset_raw = None
        if eval_split:
            splits = dataset.train_test_split(test_size = 0.06, seed = 42)
            dataset = splits["train"]
            eval_dataset_raw = splits["test"]

        self._update_progress(status_message = "Processing audio for Whisper...")
        logger.info(
            f"Whisper preprocessing: audio_col='{audio_col}', text_col='{text_col}', "
            f"samples={len(dataset)}\n"
        )

        def process_split(ds, split_name = "train"):
            processed = []
            skipped = 0
            for idx in range(len(ds)):
                if self.should_stop:
                    logger.info(f"Stopped during Whisper {split_name} preprocessing\n")
                    break

                example = ds[idx]
                try:
                    audio_data = example.get(audio_col)
                    text = example.get(text_col)
                    if audio_data is None or audio_data.get("array") is None or not text:
                        skipped += 1
                        continue

                    # Extract audio features (notebook 112-115)
                    features = self.tokenizer.feature_extractor(
                        audio_data["array"], sampling_rate = audio_data["sampling_rate"]
                    )
                    # Tokenize text (notebook 116)
                    tokenized_text = self.tokenizer.tokenizer(text)

                    processed.append(
                        {
                            "input_features": features.input_features[0],
                            "labels": tokenized_text.input_ids,
                        }
                    )
                except Exception as e:
                    logger.warning(f"Error processing Whisper {split_name} example {idx}: {e}")
                    skipped += 1
                    continue

                if (idx + 1) % 100 == 0:
                    self._update_progress(
                        status_message = f"Processing {split_name} audio... {idx + 1}/{len(ds)}"
                    )

            logger.info(
                f"Whisper {split_name} preprocessing: {len(processed)} examples ({skipped} skipped)\n"
            )
            return processed

        train_data = process_split(dataset, "train")
        eval_data = process_split(eval_dataset_raw, "eval") if eval_dataset_raw else None

        if not train_data:
            raise ValueError("No valid examples after Whisper preprocessing")

        return (train_data, eval_data)

    @staticmethod
    def _resolve_local_files(file_paths: list) -> list[str]:
        """Resolve a list of local dataset paths to concrete file paths."""
        all_files: list[str] = []
        for dataset_file in file_paths:
            if os.path.isabs(dataset_file):
                file_path = dataset_file
            elif os.path.exists(dataset_file):
                # A path relative to the current working directory (CLI usage)
                file_path = os.path.abspath(dataset_file)
            else:
                file_path = str(resolve_dataset_path(dataset_file))

            file_path_obj = Path(file_path)

            if file_path_obj.is_dir():
                parquet_dir = (
                    file_path_obj / "parquet-files"
                    if (file_path_obj / "parquet-files").exists()
                    else file_path_obj
                )
                parquet_files = sorted(parquet_dir.glob("*.parquet"))
                if parquet_files:
                    all_files.extend(str(p) for p in parquet_files)
                    continue
                candidates: list[Path] = []
                for ext in (".json", ".jsonl", ".csv", ".parquet"):
                    candidates.extend(sorted(file_path_obj.glob(f"*{ext}")))
                if candidates:
                    all_files.extend(str(c) for c in candidates)
                    continue
                raise ValueError(f"No supported data files in directory: {file_path_obj}")
            else:
                all_files.append(str(file_path_obj))
        return all_files

    @staticmethod
    def _loader_for_files(files: list[str]) -> str:
        """Determine the HF datasets loader type from file extensions."""
        first_ext = Path(files[0]).suffix.lower()
        if first_ext in (".json", ".jsonl"):
            return "json"
        elif first_ext == ".csv":
            return "csv"
        elif first_ext == ".parquet":
            return "parquet"
        raise ValueError(f"Unsupported dataset format: {files[0]}")

    def load_and_format_dataset(
        self,
        dataset_source: Optional[str],
        format_type: str = "auto",
        local_datasets: Optional[List[str]] = None,
        local_eval_datasets: Optional[List[str]] = None,
        custom_format_mapping: Optional[Dict[str, Any]] = None,
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
        """
        Load and prepare a dataset for training.

        Strategy: format first, then split — ensures both train and eval
        portions are formatted and templated.

        Returns (dataset_info, eval_dataset) or None on error; eval_dataset
        may be None if no eval split is available.
        """
        from core.training.s3_dataset import S3DownloadCancelled

        s3_download = None
        try:
            dataset = None
            eval_dataset = None
            has_separate_eval_source = False  # True if eval comes from a separate HF split
            eval_enabled = eval_steps is not None and eval_steps > 0
            raw_text_mode = is_cpt or format_type == "raw"

            def _raw_mode_label() -> str:
                return "CPT" if is_cpt else "raw text"

            def _apply_raw_text_prep(ds: Dataset, split_name: str) -> Dataset:
                try:
                    result = prepare_raw_text_dataset(
                        ds,
                        mode_label = _raw_mode_label(),
                        split_name = split_name,
                        eos_token = getattr(self.tokenizer, "eos_token", None),
                        append_eos = True,
                    )
                except ValueError as exc:
                    error_msg = str(exc)
                    logger.error(error_msg)
                    self._update_progress(error = error_msg)
                    raise

                for notice in result.notices:
                    if notice.level == "warning":
                        logger.warning(notice.message)
                        if notice.update_status:
                            self._update_progress(status_message = notice.message)
                    else:
                        logger.info(f"{notice.message}\n")

                return result.dataset

            # S3 datasets are downloaded to a local temp dir and then consumed
            # through the same local-file path below.
            if s3_config and not local_datasets:
                from core.training.s3_dataset import prepare_s3_dataset_download

                self._update_progress(status_message = "Downloading dataset from S3...")
                s3_download = prepare_s3_dataset_download(
                    s3_config,
                    cancel_callback = lambda: self.should_stop,
                )
                local_datasets = s3_download.files
                if self.should_stop:
                    logger.info("Stopped during S3 download\n")
                    return None
                logger.info(f"Downloaded {len(local_datasets)} file(s) from S3\n")

            if local_datasets:
                # Use load_dataset() for an Arrow-backed result; in-memory
                # Dataset.from_list() has no cache and forces num_proc=1 during
                # tokenization/map (sharding needs Arrow files).
                all_files = self._resolve_local_files(local_datasets)

                if all_files:
                    loader = self._loader_for_files(all_files)
                    dataset = load_dataset(loader, data_files = all_files, split = "train")

                    # Check if stopped during dataset loading
                    if self.should_stop:
                        logger.info("Stopped during dataset loading\n")
                        return None

                    self._update_progress(
                        status_message = f"Loaded {len(dataset)} samples from local files"
                    )
                    logger.info(f"Loaded {len(dataset)} samples from local files\n")
                    logger.info(f"[DEBUG] Dataset cache_files: {dataset.cache_files}\n")

                # Load local eval datasets if provided
                if local_eval_datasets and eval_enabled:
                    eval_all_files = self._resolve_local_files(local_eval_datasets)
                    if eval_all_files:
                        eval_loader = self._loader_for_files(eval_all_files)
                        eval_dataset = load_dataset(
                            eval_loader, data_files = eval_all_files, split = "train"
                        )
                        has_separate_eval_source = True
                        logger.info(
                            f"Loaded {len(eval_dataset)} eval samples from local eval files\n"
                        )

            elif dataset_source:
                # Load from Hugging Face
                split_name = train_split or "train"
                load_kwargs = {"path": dataset_source, "split": split_name}
                if subset:
                    load_kwargs["name"] = subset

                if dataset_streaming:
                    self._update_progress(status_message = f"Streaming dataset: {dataset_source}...")
                    dataset = load_dataset(**load_kwargs, streaming = True)

                    # Optional iterable slicing
                    if dataset_slice_start is not None and dataset_slice_start > 0:
                        dataset = dataset.skip(dataset_slice_start)

                    if dataset_slice_end is not None:
                        slice_start = dataset_slice_start or 0
                        take_count = dataset_slice_end - slice_start + 1
                        if take_count <= 0:
                            raise ValueError(
                                "Train Split End must be greater than or equal to Train Split Start."
                            )
                        dataset = dataset.take(take_count)
                        # IterableDataset.take(N) yields *at most* N samples — if
                        # the source is shorter, the user silently gets fewer rows.
                        logger.warning(
                            f"Streaming slice requested up to {take_count} rows "
                            f"[{slice_start}, {dataset_slice_end}]; actual yield "
                            f"may be smaller if the dataset has fewer rows."
                        )
                        if take_count == 1:
                            # start == end is a valid slice but produces a single
                            # training row, which is almost always user error.
                            logger.warning(
                                "Dataset slice resolves to a single row "
                                f"(start == end == {slice_start}); training on 1 "
                                "sample is likely unintended."
                            )

                    logger.info(
                        f"Loaded Hugging Face dataset in streaming mode: {dataset_source}\n"
                    )
                    self._update_progress(status_message = f"Streaming {dataset_source}")
                else:
                    # Non-streaming: if a slice end is given, stream only the needed
                    # rows and materialize them (avoids downloading the whole dataset);
                    # the eager [start, end] trim happens further below.
                    _slice_start = dataset_slice_start or 0
                    # streaming=True rejects HF slice syntax (e.g. "train[:50%]")
                    # with "Bad split", so the streaming shortcut is unusable when
                    # train_split already carries a slice expression, so fall back to
                    # the regular download path, which handles HF slice syntax.
                    _split_has_slice = (train_split or "").find("[") != -1
                    if (
                        not _split_has_slice
                        and dataset_slice_end is not None
                        and dataset_slice_end >= 0
                        and dataset_slice_end >= _slice_start
                    ):
                        rows_to_stream = dataset_slice_end + 1
                        logger.info(
                            f"[dataset-slice] Manual slice specified "
                            f"(start={dataset_slice_start}, end={dataset_slice_end}), "
                            f"streaming {rows_to_stream} rows\n"
                        )
                        stream = load_dataset(**load_kwargs, streaming = True)
                        dataset = Dataset.from_list(list(stream.take(rows_to_stream)))
                        logger.info(
                            f"[dataset-slice] Downloaded {len(dataset)} rows "
                            f"(requested {rows_to_stream})\n"
                        )
                    else:
                        self._update_progress(
                            status_message = f"Downloading dataset: {dataset_source}..."
                        )
                        dataset = load_dataset(**load_kwargs)

                        n_rows = len(dataset) if hasattr(dataset, "__len__") else 0
                        self._update_progress(
                            status_message = f"Downloaded {dataset_source} ({n_rows:,} rows)"
                        )
                        logger.info(
                            f"Loaded dataset from Hugging Face: {dataset_source} ({n_rows:,} rows)\n"
                        )

                # Check if stopped during dataset loading
                if self.should_stop:
                    logger.info("Stopped during dataset loading\n")
                    return None

                # Resolve eval split from a separate HF split (explicit or auto)
                if eval_enabled:
                    effective_train = train_split or "train"
                    if eval_split and eval_split != effective_train:
                        # Explicit eval split — load directly
                        logger.info(f"Loading explicit eval split: '{eval_split}'\n")
                        eval_load_kwargs = {"path": dataset_source, "split": eval_split}
                        if subset:
                            eval_load_kwargs["name"] = subset

                        if dataset_streaming:
                            # Probe available splits before the streaming load.
                            # load_dataset(streaming=True) returns an IterableDataset
                            # without validating the split name — a typo would only
                            # surface on the first eval batch mid-training.
                            from datasets import get_dataset_split_names

                            probe_kwargs = {"path": dataset_source}
                            if subset:
                                probe_kwargs["config_name"] = subset
                            try:
                                available_splits = get_dataset_split_names(**probe_kwargs)
                            except Exception as probe_err:
                                raise ValueError(
                                    f"Could not list splits for '{dataset_source}' "
                                    f"to validate eval_split='{eval_split}': {probe_err}"
                                )
                            # Streaming rejects HF slice syntax, and the request
                            # validator already blocks bracketed streaming splits,
                            # so eval_split here is always a bare split name.
                            if eval_split not in available_splits:
                                raise ValueError(
                                    f"Requested eval split '{eval_split}' not found in "
                                    f"dataset '{dataset_source}'. Available splits: "
                                    f"{available_splits}"
                                )
                            eval_dataset = load_dataset(**eval_load_kwargs, streaming = True)
                            # A streaming eval dataset has no __len__; bound it so
                            # each evaluation terminates instead of consuming the
                            # whole stream. .take() stays lazy and survives the
                            # later format/raw-text .map() passes.
                            if not hasattr(eval_dataset, "__len__"):
                                eval_dataset = eval_dataset.take(STREAMING_EVAL_MAX_SAMPLES)
                                logger.info(
                                    f"Streaming eval split capped to "
                                    f"{STREAMING_EVAL_MAX_SAMPLES} samples\n"
                                )
                        else:
                            eval_dataset = load_dataset(**eval_load_kwargs)

                        has_separate_eval_source = True
                        if hasattr(eval_dataset, "__len__"):
                            logger.info(
                                f"Loaded eval split '{eval_split}' with {len(eval_dataset)} rows\n"
                            )
                        else:
                            logger.info(f"Loaded eval split '{eval_split}' in streaming mode\n")
                    elif eval_split and eval_split == effective_train:
                        if dataset_streaming:
                            raise ValueError(
                                "Streaming mode does not support using the same split for both train and eval. "
                                "Please provide a separate eval split or set eval_steps to 0."
                            )
                        # Same split as training — split 80/20 after formatting
                        logger.info(
                            f"Eval split '{eval_split}' is the same as train split — will split 80/20\n"
                        )
                    else:
                        if dataset_streaming:
                            raise ValueError(
                                "Streaming mode currently requires an explicit eval split when evaluation is enabled."
                            )
                        # Auto-detect eval split from HF (separate dataset or None)
                        eval_dataset = self._auto_detect_eval_split_from_hf(
                            dataset_source = dataset_source,
                            subset = subset,
                        )
                        if eval_dataset is not None:
                            has_separate_eval_source = True
                else:
                    logger.info("Eval disabled (eval_steps <= 0), skipping eval split detection\n")

            if dataset is None:
                raise ValueError("No dataset provided")

            # Apply eager-only index range slicing if requested (inclusive on both ends).
            # Streaming already sliced lazily via skip()/take() above; the non-streaming
            # manual-slice path fetched up to end+1 rows and is trimmed to [start, end] here.
            if (not dataset_streaming) and (
                dataset_slice_start is not None or dataset_slice_end is not None
            ):
                total_rows = len(dataset)
                start = dataset_slice_start if dataset_slice_start is not None else 0
                end = dataset_slice_end if dataset_slice_end is not None else total_rows - 1
                # Clamp to valid range
                start = max(0, min(start, total_rows - 1))
                end = max(start, min(end, total_rows - 1))
                dataset = dataset.select(range(start, end + 1))
                logger.info(
                    f"Sliced dataset to rows [{start}, {end}]: {len(dataset)} of {total_rows} rows\n"
                )
                self._update_progress(
                    status_message = f"Sliced dataset to {len(dataset)} rows (indices {start}-{end})"
                )

            # Check if stopped before applying template
            if self.should_stop:
                logger.info("Stopped before applying chat template\n")
                return None

            # ========== AUDIO MODELS: custom preprocessing ==========
            if self._audio_type == "csm":
                processed = self._preprocess_csm_dataset(dataset, custom_format_mapping)
                return (processed, None)

            elif self._audio_type == "whisper":
                train_data, eval_data = self._preprocess_whisper_dataset(
                    dataset,
                    eval_split = eval_split,
                    custom_format_mapping = custom_format_mapping,
                )
                return (train_data, eval_data)

            elif self._audio_type == "snac":
                processed = self._preprocess_snac_dataset(dataset, custom_format_mapping)
                return (processed, None)

            elif self._audio_type == "bicodec":
                processed = self._preprocess_bicodec_dataset(dataset, custom_format_mapping)
                return ({"dataset": processed, "final_format": "audio_bicodec"}, None)

            elif self._audio_type == "dac":
                processed = self._preprocess_dac_dataset(dataset, custom_format_mapping)
                return ({"dataset": processed, "final_format": "audio_dac"}, None)

            # ========== RAW TEXT BYPASS ==========
            if raw_text_mode:
                logger.info(
                    f"{_raw_mode_label().capitalize()} mode: bypassing chat template, "
                    "using raw text\n"
                )
                dataset = _apply_raw_text_prep(dataset, "train")
                if has_separate_eval_source and eval_dataset is not None:
                    eval_dataset = _apply_raw_text_prep(eval_dataset, "eval")

                dataset_info = {
                    "dataset": dataset,
                    "detected_format": "raw_text",
                    "final_format": "raw_text",
                    "success": True,
                }

                if has_separate_eval_source and eval_dataset is not None:
                    eval_rows = (
                        f"{len(eval_dataset):,} rows"
                        if hasattr(eval_dataset, "__len__")
                        else "streaming"
                    )
                    logger.info(
                        f"{_raw_mode_label().capitalize()}: eval dataset "
                        f"({eval_rows}) kept as raw text\n"
                    )
                elif eval_enabled and not has_separate_eval_source and not dataset_streaming:
                    # _resolve_eval_split_from_dataset does a train_test_split (needs
                    # len/random access). Streaming always provides a separate eval
                    # split (route-enforced), so this auto-split is non-streaming only.
                    split_result = self._resolve_eval_split_from_dataset(dataset)
                    if split_result is not None:
                        train_portion, eval_dataset = split_result
                        dataset_info["dataset"] = train_portion

                train_dataset = dataset_info["dataset"]
                n = len(train_dataset) if hasattr(train_dataset, "__len__") else None
                n_display = f"{n:,}" if isinstance(n, int) else "streaming"
                self._update_progress(
                    status_message = f"Dataset ready ({n_display} samples, raw text)"
                )
                logger.info(f"Raw-text dataset ready ({n_display} samples)\n")

                # Streaming datasets can report column_names as None, which would
                # make "text" not in None raise TypeError; resolve_column_names
                # falls back to features/first-row probing.
                train_columns = resolve_column_names(train_dataset)
                if "text" not in train_columns:
                    raise ValueError(f"Raw-text dataset missing 'text' column: {train_columns}")
                return (dataset_info, eval_dataset)

            elif self.is_audio_vlm:
                formatted = self._format_audio_vlm_dataset(dataset, custom_format_mapping)
                return (formatted, None)

            # ========== FORMAT FIRST ==========
            logger.info(f"Formatting dataset with format_type='{format_type}'...\n")

            dataset_info = format_and_template_dataset(
                dataset,
                model_name = self.model_name,
                tokenizer = self.tokenizer,
                is_vlm = self.is_vlm,
                format_type = format_type,
                dataset_name = dataset_source,
                custom_format_mapping = custom_format_mapping,
                progress_callback = self._update_progress,
            )

            # Check if stopped during formatting
            if self.should_stop:
                logger.info("Stopped during dataset formatting\n")
                return None

            # Abort if dataset formatting/conversion failed
            if not dataset_info.get("success", True):
                errors = dataset_info.get("errors", [])
                error_msg = "; ".join(errors) if errors else "Dataset formatting failed"
                logger.error(f"Dataset conversion failed: {error_msg}")
                self._update_progress(error = error_msg)
                return None

            detected = dataset_info.get("detected_format", "unknown")
            final_ds = dataset_info.get("dataset")
            final_n = len(final_ds) if hasattr(final_ds, "__len__") else "?"
            self._update_progress(
                status_message = f"Dataset ready ({final_n:,} samples, {detected} format)"
                if isinstance(final_n, int)
                else f"Dataset ready ({final_n} samples, {detected} format)"
            )
            logger.info(f"Dataset formatted successfully ({final_n} samples, {detected})\n")

            # ========== THEN SPLIT ==========
            if has_separate_eval_source and eval_dataset is not None:
                # Eval came from a separate HF split — format it too
                eval_n = len(eval_dataset) if hasattr(eval_dataset, "__len__") else "?"
                logger.info(f"Formatting eval dataset ({eval_n} rows)...\n")
                eval_info = format_and_template_dataset(
                    eval_dataset,
                    model_name = self.model_name,
                    tokenizer = self.tokenizer,
                    is_vlm = self.is_vlm,
                    format_type = format_type,
                    dataset_name = dataset_source,
                    custom_format_mapping = custom_format_mapping,
                )
                eval_dataset = eval_info["dataset"]
                logger.info("Eval dataset formatted successfully\n")
            elif eval_enabled and not has_separate_eval_source and not dataset_streaming:
                # No separate eval source — split the already-formatted dataset
                formatted_dataset = dataset_info["dataset"]
                split_result = self._resolve_eval_split_from_dataset(formatted_dataset)
                if split_result is not None:
                    train_portion, eval_dataset = split_result
                    dataset_info["dataset"] = train_portion

            return (dataset_info, eval_dataset)

        except S3DownloadCancelled:
            logger.info("Stopped during S3 download\n")
            return None
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            self._update_progress(error = str(e))
            return None
        finally:
            if s3_download is not None:
                s3_download.cleanup()

    def _auto_detect_eval_split_from_hf(
        self, dataset_source: str, subset: str
    ) -> Optional[Dataset]:
        """Auto-detect an eval split from an HF dataset (named split only)."""
        try:
            from datasets import get_dataset_split_names

            load_kwargs = {"path": dataset_source}
            if subset:
                load_kwargs["config_name"] = subset
            available_splits = get_dataset_split_names(**load_kwargs)
            logger.info(f"Available splits: {available_splits}\n")

            # Check for common eval split names
            for candidate in ["eval", "validation", "valid", "val", "test"]:
                if candidate in available_splits:
                    eval_load_kwargs = {"path": dataset_source, "split": candidate}
                    if subset:
                        eval_load_kwargs["name"] = subset
                    candidate_ds = load_dataset(**eval_load_kwargs)
                    if len(candidate_ds) >= 16:
                        logger.info(
                            f"Auto-detected eval split '{candidate}' with {len(candidate_ds)} rows\n"
                        )
                        return candidate_ds
                    else:
                        logger.info(
                            f"Found eval split '{candidate}' but only {len(candidate_ds)} rows (< 16), skipping\n"
                        )

        except Exception as e:
            logger.warning(f"Could not check dataset splits: {e}")

        # No separate HF eval split — caller handles programmatic splitting
        return None

    def _resolve_eval_split_from_dataset(self, dataset) -> Optional[tuple]:
        """Split a dataset into train and eval portions.

        Returns (train_dataset, eval_dataset), or None if too small.
        """
        MIN_EVAL_ROWS = 16
        MIN_TOTAL_ROWS = 32  # Need at least 16 train + 16 eval

        n = len(dataset)
        if n < MIN_TOTAL_ROWS:
            logger.info(f"Dataset too small ({n} rows) for eval split, skipping eval\n")
            return None

        eval_size = max(MIN_EVAL_ROWS, min(128, int(0.05 * n)))
        # Don't take more than half the dataset
        eval_size = min(eval_size, n // 2)

        logger.info(f"Auto-splitting: {eval_size} rows for eval from {n} total\n")
        split_result = dataset.train_test_split(test_size = eval_size, seed = 3407)
        logger.info(
            f"Split complete: {len(split_result['train'])} train, {len(split_result['test'])} eval\n"
        )
        return (split_result["train"], split_result["test"])

    def start_training(
        self,
        dataset: Dataset,
        eval_dataset: Dataset = None,
        eval_steps: float = 0.00,
        output_dir: str | None = None,
        num_epochs: int = 3,
        learning_rate: float = 2e-4,
        embedding_learning_rate: float | None = None,
        batch_size: int = 2,
        gradient_accumulation_steps: int = 4,
        warmup_steps: int = None,
        warmup_ratio: float = None,
        max_steps: int = 0,
        save_steps: int = 0,
        weight_decay: float = 0.001,
        random_seed: int = 3407,
        packing: bool = False,
        train_on_completions: bool = False,
        enable_wandb: bool = False,
        wandb_project: str = "unsloth-training",
        wandb_token: str = None,
        enable_tensorboard: bool = False,
        tensorboard_dir: str | None = None,
        **kwargs,
    ) -> bool:
        """Start training in a separate thread"""

        if self.is_training:
            logger.warning("Training already in progress")
            return False

        if self.model is None or self.tokenizer is None:
            self._update_progress(error = "Model not loaded")
            return False

        # Pre-import heavy transformers modules on the main thread. Unsloth's
        # patched_import hook isn't thread-safe with importlib's cache, causing
        # KeyError: 'size' if first imported in the worker thread.
        import transformers  # noqa: F401 – ensures submodules are cached
        from transformers import (  # noqa: F401
            Trainer as _HFTrainer,
            TrainingArguments as _TrainingArguments,
            TrainerCallback as _TrainerCallback,
        )

        if self._audio_type == "whisper":
            from transformers import (  # noqa: F401
                Seq2SeqTrainer as _Seq2SeqTrainer,
                Seq2SeqTrainingArguments as _Seq2SeqTrainingArguments,
            )

        # Start training in separate thread
        self.training_thread = threading.Thread(
            target = self._train_worker,
            args = (dataset,),
            kwargs = {
                "output_dir": output_dir,
                "num_epochs": num_epochs,
                "learning_rate": learning_rate,
                "embedding_learning_rate": embedding_learning_rate,
                "batch_size": batch_size,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "warmup_steps": warmup_steps,
                "warmup_ratio": warmup_ratio,
                "max_steps": max_steps,
                "save_steps": save_steps,
                "weight_decay": weight_decay,
                "random_seed": random_seed,
                "packing": packing,
                "train_on_completions": train_on_completions,
                "enable_wandb": enable_wandb,
                "wandb_project": wandb_project,
                "wandb_token": wandb_token,
                "enable_tensorboard": enable_tensorboard,
                "tensorboard_dir": tensorboard_dir,
                "eval_dataset": eval_dataset,
                "eval_steps": eval_steps,
                **kwargs,
            },
        )

        self.should_stop = False
        self.is_training = True
        try:
            self.training_thread.start()
            return True
        except Exception as e:
            self.is_training = False
            logger.error(f"Failed to start training thread: {e}")
            return False

    def _chat_template_renders_empty(self) -> bool:
        """True when the chat template renders a sample to empty text (base-model signature)."""
        try:
            ds = getattr(self.trainer, "train_dataset", None)
            if ds is None or len(ds) == 0:
                return False
            row = ds[0]
            messages = row.get("messages") if isinstance(row, dict) else None
            if not messages:
                return False
            tok = self.tokenizer
            if not hasattr(tok, "apply_chat_template"):
                return False
            rendered = tok.apply_chat_template(
                messages, tokenize = False, add_generation_prompt = False
            )
            return not (isinstance(rendered, str) and rendered.strip())
        except Exception:
            return False

    def _preflight_first_batch(self) -> Optional[str]:
        """Validate the first real batch before train(). A base model whose chat
        template renders empty yields empty float32 input_ids that crash the
        embedding on step 1; catch it here. Returns None for a valid batch."""
        try:
            loader = self.trainer.get_train_dataloader()
            batch = next(iter(loader))
        except StopIteration:
            return (
                "Cannot start training: the dataset produced no training rows. "
                "This usually means a split/slice or streaming filter removed every "
                "row. Check your train split, slice range, and dataset filters."
            )
        except Exception as e:
            model = self.model_name or "this model"
            return (
                f"Cannot start training: failed to build the first training batch "
                f"for '{model}': {e}"
            )

        try:
            input_ids = batch["input_ids"] if "input_ids" in batch else None
        except Exception:
            input_ids = getattr(batch, "input_ids", None)
        if input_ids is None:
            return None  # some collators omit input_ids

        seq_len = input_ids.shape[-1] if input_ids.ndim > 0 else 0
        if not (input_ids.is_floating_point() or input_ids.numel() == 0 or seq_len == 0):
            return None

        model = self.model_name or "this model"
        if self._chat_template_renders_empty():
            low = model.lower()
            suffix = (
                f" such as '{model}-Instruct'"
                if not any(t in low for t in ("instruct", "chat", "-it", "_it"))
                else ""
            )
            return (
                f"Cannot start training: the chat template for '{model}' produced "
                f"no text for your dataset, so the first batch had empty token IDs. "
                f"'{model}' looks like a base (pretrained) model without a chat "
                f"template suited to conversational fine-tuning. Use the "
                f"instruction-tuned variant{suffix} or provide a chat template."
            )
        return (
            f"Cannot start training: the first batch produced invalid token IDs "
            f"(dtype={input_ids.dtype}, length={seq_len}). Check that your dataset "
            f"columns are mapped correctly for '{model}'."
        )

    def _train_worker(self, dataset: Dataset | dict, **training_args):
        """Worker function for training (runs in separate thread).

        ``dataset`` is either a raw ``datasets.Dataset`` (audio preprocessing
        paths such as CSM / Whisper / SNAC / Audio-VLM) or a ``dict`` wrapper
        returned by ``format_and_template_dataset`` (text and image VLM paths).
        Streaming HF datasets arrive wrapped in the latter ``dict`` — they are
        never passed as a bare ``IterableDataset``.
        """
        try:
            # On spawn platforms, register compiled-cache dirs on sys.path/PYTHONPATH
            # before any dataset.map() so spawned workers can import compiled
            # modules such as UnslothSFTTrainer.
            if sys.platform in ("win32", "darwin"):
                from utils.cache_cleanup import register_compiled_cache_on_path
                register_compiled_cache_on_path()

            # Store training parameters for metrics calculation
            self.batch_size = training_args.get("batch_size", 2)
            self.max_seq_length = training_args.get("max_seq_length", 2048)
            self.gradient_accumulation_steps = training_args.get("gradient_accumulation_steps", 4)

            # Set training start time
            self.training_start_time = time.time()

            self._update_progress(is_training = True, error = None)

            # Setup logging
            if training_args.get("enable_wandb", False) and training_args.get("wandb_token"):
                os.environ["WANDB_API_KEY"] = training_args["wandb_token"]
                import wandb
                wandb.init(project = training_args.get("wandb_project", "unsloth-training"))

            # Create output directory
            output_dir = str(resolve_output_dir(training_args.get("output_dir")))
            ensure_dir(Path(output_dir))

            # ========== AUDIO TRAINER BRANCH ==========
            if self._audio_type == "csm":
                # CSM uses plain HF Trainer (not SFTTrainer) with
                # remove_unused_columns=False for the depth decoder.
                from transformers import Trainer as HFTrainer, TrainingArguments

                self._apply_csm_forward_fix()

                config = self._build_audio_training_args(
                    training_args,
                    output_dir,
                    extra_args = {
                        "remove_unused_columns": False,
                    },
                )
                self.trainer = HFTrainer(
                    model = self.model,
                    train_dataset = dataset,
                    args = TrainingArguments(**config),
                )
                self.trainer.add_callback(self._create_progress_callback())

                batch_size = training_args.get("batch_size", 2)
                total = self._calculate_total_steps(
                    len(dataset),
                    batch_size,
                    training_args.get("gradient_accumulation_steps", 4),
                    training_args.get("num_epochs", 3),
                    training_args.get("max_steps", 0),
                )
                self._update_progress(total_steps = total, status_message = "Starting CSM training...")
                logger.info(f"CSM training config: {config}\n")
                self.trainer.train(
                    resume_from_checkpoint = training_args.get("resume_from_checkpoint")
                )
                self._finalize_training(output_dir, "CSM")
                return

            elif self._audio_type == "snac":
                # Orpheus: LM with SNAC codec tokens — plain HF Trainer.
                # DataCollatorForSeq2Seq pads variable-length sequences per batch
                # and pads labels with -100.
                from transformers import (
                    Trainer as HFTrainer,
                    TrainingArguments,
                    DataCollatorForSeq2Seq,
                )

                config = self._build_audio_training_args(training_args, output_dir)
                self.trainer = HFTrainer(
                    model = self.model,
                    train_dataset = dataset,
                    args = TrainingArguments(**config),
                    data_collator = DataCollatorForSeq2Seq(
                        tokenizer = self.tokenizer,
                        padding = True,
                        pad_to_multiple_of = 8,
                    ),
                )
                self.trainer.add_callback(self._create_progress_callback())

                batch_size = training_args.get("batch_size", 2)
                total = self._calculate_total_steps(
                    len(dataset),
                    batch_size,
                    training_args.get("gradient_accumulation_steps", 4),
                    training_args.get("num_epochs", 3),
                    training_args.get("max_steps", 0),
                )
                self._update_progress(total_steps = total, status_message = "Starting SNAC training...")
                logger.info(f"SNAC training config: {config}\n")
                self.trainer.train(
                    resume_from_checkpoint = training_args.get("resume_from_checkpoint")
                )
                self._finalize_training(output_dir, "SNAC")
                return

            elif self._audio_type == "whisper":
                # Whisper: Seq2SeqTrainer with custom speech collator
                from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
                from utils.datasets import DataCollatorSpeechSeq2SeqWithPadding

                eval_dataset = training_args.get("eval_dataset", None)
                extra = {"remove_unused_columns": False, "label_names": ["labels"]}
                if eval_dataset:
                    extra["eval_strategy"] = "steps"
                    extra["eval_steps"] = training_args.get("eval_steps", 5)

                config = self._build_audio_training_args(
                    training_args, output_dir, extra_args = extra
                )

                trainer_kwargs = {
                    "model": self.model,
                    "train_dataset": dataset,
                    "data_collator": DataCollatorSpeechSeq2SeqWithPadding(processor = self.tokenizer),
                    "processing_class": self.tokenizer.feature_extractor,
                    "args": Seq2SeqTrainingArguments(**config),
                }
                if eval_dataset:
                    trainer_kwargs["eval_dataset"] = eval_dataset

                self.trainer = Seq2SeqTrainer(**trainer_kwargs)
                self.trainer.add_callback(self._create_progress_callback())

                batch_size = training_args.get("batch_size", 2)
                total = self._calculate_total_steps(
                    len(dataset),
                    batch_size,
                    training_args.get("gradient_accumulation_steps", 4),
                    training_args.get("num_epochs", 3),
                    training_args.get("max_steps", 0),
                )
                self._update_progress(
                    total_steps = total, status_message = "Starting Whisper training..."
                )
                logger.info(f"Whisper training config: {config}\n")
                self.trainer.train(
                    resume_from_checkpoint = training_args.get("resume_from_checkpoint")
                )
                self._finalize_training(output_dir, "Whisper")
                return

            elif self._audio_type is not None and self._audio_type not in (
                "bicodec",
                "dac",
            ):
                # bicodec/dac use the standard SFTTrainer text path below
                raise NotImplementedError(
                    f"Audio training for '{self._audio_type}' not yet implemented"
                )

            # ========== DATA COLLATOR SELECTION ==========
            model_name_lower = self.model_name.lower()
            is_deepseek_ocr = "deepseek" in model_name_lower and "ocr" in model_name_lower

            logger.info("Configuring data collator...\n")

            dataset_final_format = (
                str(dataset.get("final_format", "")).lower() if isinstance(dataset, dict) else ""
            )
            raw_text_mode = dataset_final_format == "raw_text"

            data_collator = None  # Default: built-in collator
            if is_deepseek_ocr:
                # DeepSeek OCR collator - auto-install if needed
                logger.info("Detected DeepSeek OCR model\n")
                if not _ensure_deepseek_ocr_installed():
                    error_msg = (
                        "Failed to install DeepSeek OCR module. "
                        "Please install manually: "
                        "from huggingface_hub import snapshot_download; "
                        "snapshot_download('unsloth/DeepSeek-OCR', local_dir='deepseek_ocr')"
                    )
                    logger.error(error_msg)
                    self._update_progress(error = error_msg, is_training = False)
                    return

                try:
                    from backend.data_utils import DeepSeekOCRDataCollator

                    logger.info("Configuring DeepSeek OCR data collator...\n")
                    FastVisionModel.for_training(self.model)
                    # (image_size, base_size, crop_mode) is a coupled preset;
                    # changing image_size alone desyncs the per-crop grid from
                    # num_queries. Use Gundam preset.
                    if training_args.get("vision_image_size") is not None:
                        logger.info(
                            "Vision image resize ignored for DeepSeek OCR "
                            "(uses fixed Gundam preset).\n"
                        )
                    data_collator = DeepSeekOCRDataCollator(
                        tokenizer = self.tokenizer,
                        model = self.model,
                        image_size = 640,
                        base_size = 1024,
                        crop_mode = True,
                        train_on_responses_only = training_args.get("train_on_completions", False),
                    )
                    logger.info("DeepSeek OCR data collator configured successfully\n")

                except Exception as e:
                    logger.error(f"Failed to configure DeepSeek OCR collator: {e}")
                    error_msg = f"Error configuring DeepSeek OCR: {str(e)}"
                    self._update_progress(error = error_msg, is_training = False)
                    return

            elif self.is_audio_vlm and not raw_text_mode:
                # Audio VLM collator (e.g. Gemma 3N), mirrors the
                # Gemma3N_(4B)-Audio notebook collate_fn.
                logger.info("Configuring audio VLM data collator...\n")
                processor = self.tokenizer  # FastModel returns processor as tokenizer

                audio_col_name = getattr(self, "_audio_vlm_audio_col", "audio")

                def audio_vlm_collate_fn(examples):
                    texts = []
                    audios = []
                    for example in examples:
                        text = processor.apply_chat_template(
                            example["messages"],
                            tokenize = False,
                            add_generation_prompt = False,
                        ).strip()
                        texts.append(text)
                        audios.append(example[audio_col_name]["array"])

                    batch = processor(text = texts, audio = audios, return_tensors = "pt", padding = True)

                    # Labels = input_ids with special tokens masked
                    labels = batch["input_ids"].clone()
                    labels[labels == processor.tokenizer.pad_token_id] = -100
                    for attr in (
                        "audio_token_id",
                        "image_token_id",
                        "boi_token_id",
                        "eoi_token_id",
                    ):
                        token_id = getattr(processor.tokenizer, attr, None)
                        if token_id is not None:
                            labels[labels == token_id] = -100
                    batch["labels"] = labels
                    return batch

                data_collator = audio_vlm_collate_fn
                logger.info("Audio VLM data collator configured\n")

            elif self.is_vlm and not raw_text_mode:
                # Standard VLM collator (images)
                logger.info("Using UnslothVisionDataCollator for vision model\n")
                from unsloth.trainer import UnslothVisionDataCollator

                FastVisionModel.for_training(self.model)
                vision_image_size = training_args.get("vision_image_size")
                if vision_image_size is None:
                    data_collator = UnslothVisionDataCollator(self.model, self.tokenizer)
                else:
                    logger.info(f"Vision image resize: {vision_image_size} (max dimension)\n")
                    data_collator = UnslothVisionDataCollator(
                        self.model,
                        self.tokenizer,
                        resize = vision_image_size,
                        resize_dimension = "max",
                    )
                logger.info("Vision data collator configured\n")

            # ========== TRAINING CONFIGURATION ==========
            warmup_steps_val = training_args.get("warmup_steps", None)
            warmup_ratio_val = training_args.get("warmup_ratio", None)

            lr_value = training_args.get("learning_rate", 2e-4)
            logger.info(
                f"[DEBUG] learning_rate from training_args: {lr_value} (type: {type(lr_value).__name__})\n"
            )

            config_args = {
                "per_device_train_batch_size": training_args.get("batch_size", 2),
                "gradient_accumulation_steps": training_args.get("gradient_accumulation_steps", 4),
                "num_train_epochs": training_args.get("num_epochs", 3),  # Default to epochs
                "learning_rate": lr_value,
                "fp16": not is_bfloat16_supported(),
                "bf16": is_bfloat16_supported(),
                "logging_steps": 1,
                "weight_decay": training_args.get("weight_decay", 0.001),
                "seed": training_args.get("random_seed", 3407),
                "output_dir": output_dir,
                "report_to": _build_report_targets(training_args),
                "include_num_input_tokens_seen": True,  # Enable token counting
                "dataset_num_proc": dataset_map_num_proc(
                    1
                    if (self.is_audio or self.is_audio_vlm or self._cuda_audio_used)
                    else max(1, (os.cpu_count() or 1) // 4)
                ),
                "max_seq_length": training_args.get("max_seq_length", 2048),
            }
            if training_args.get("enable_tensorboard", False):
                config_args["logging_dir"] = str(
                    resolve_tensorboard_dir(training_args.get("tensorboard_dir"))
                )
            logger.info(
                f"[DEBUG] dataset_num_proc={config_args['dataset_num_proc']} (is_audio={self.is_audio}, is_audio_vlm={self.is_audio_vlm}, _cuda_audio_used={self._cuda_audio_used})"
            )

            # On spawn platforms with transformers 5.x, disable DataLoader
            # multiprocessing to avoid modified sys.path issues in spawned workers.
            if sys.platform in ("win32", "darwin"):
                import transformers as _tf
                if _tf.__version__.startswith("5."):
                    config_args["dataloader_num_workers"] = 0

            # Use warmup_ratio if provided, else warmup_steps
            if warmup_ratio_val is not None:
                config_args["warmup_ratio"] = warmup_ratio_val
                logger.info(f"Using warmup_ratio: {warmup_ratio_val}\n")
            elif warmup_steps_val is not None:
                config_args["warmup_steps"] = warmup_steps_val
                logger.info(f"Using warmup_steps: {warmup_steps_val}\n")
            else:
                # Default if neither provided
                config_args["warmup_steps"] = 5
                logger.info("Using default warmup_steps: 5\n")

            # Add save_steps if specified
            save_steps_val = training_args.get("save_steps", 0)
            if save_steps_val and save_steps_val > 0:
                config_args["save_steps"] = save_steps_val
                config_args["save_strategy"] = "steps"

            # If max_steps is specified, use it instead of epochs
            max_steps_val = training_args.get("max_steps", 0)
            if max_steps_val and max_steps_val > 0:
                del config_args["num_train_epochs"]
                config_args["max_steps"] = max_steps_val
                logger.info(f"Training for {max_steps_val} steps\n")
            else:
                logger.info(f"Training for {config_args['num_train_epochs']} epochs\n")

            # ========== EVAL CONFIGURATION ==========
            eval_dataset = training_args.get("eval_dataset", None)
            eval_steps_val = training_args.get("eval_steps", 0.00)
            if eval_dataset is not None:
                if eval_steps_val > 0:
                    config_args["eval_strategy"] = "steps"
                    config_args["eval_steps"] = eval_steps_val
                    config_args["per_device_eval_batch_size"] = config_args[
                        "per_device_train_batch_size"
                    ]
                    logger.info(
                        f"✅ Evaluation enabled: eval_steps={eval_steps_val} (fraction of total steps)\n"
                    )
                    if hasattr(eval_dataset, "__len__"):
                        logger.info(f"Eval dataset: {len(eval_dataset)} rows\n")
                    else:
                        logger.info("Eval dataset is streaming / length unknown\n")
                else:
                    logger.info(
                        f"⚠️  Eval dataset provided but eval_steps={eval_steps_val} (disabled)\n"
                    )
                    logger.info("To enable evaluation, set eval_steps > 0.0\n")
            else:
                logger.info("No eval dataset — evaluation disabled\n")

            # Model-specific params: use training_args optim/lr_scheduler_type if given, else defaults
            optim_value = training_args.get("optim", "adamw_8bit")
            lr_scheduler_type_value = training_args.get("lr_scheduler_type", "linear")

            if (self.is_vlm or self.is_audio_vlm) and not raw_text_mode:
                # Vision / audio VLM config (both need skip_prepare_dataset +
                # remove_unused_columns); raw-text VLM goes to the text path below.
                label = "audio VLM" if self.is_audio_vlm else "vision"
                logger.info(f"Configuring {label} model training parameters\n")
                # Provided values or vision defaults
                optim_value = training_args.get("optim", "adamw_torch_fused")
                lr_scheduler_type_value = training_args.get("lr_scheduler_type", "cosine")
                config_args.update(
                    {
                        "optim": optim_value,
                        "lr_scheduler_type": lr_scheduler_type_value,
                        "gradient_checkpointing": True,
                        "gradient_checkpointing_kwargs": {"use_reentrant": False},
                        "max_grad_norm": 0.3,
                        "remove_unused_columns": False,
                        "dataset_text_field": "",
                        "dataset_kwargs": {"skip_prepare_dataset": True},
                        "max_length": training_args.get("max_seq_length", 2048),
                    }
                )
            else:
                is_cpt = training_args.get("is_cpt", False)
                self.is_cpt = is_cpt
                if is_cpt:
                    logger.info("Configuring Continued Pretraining (CPT) parameters\n")
                elif raw_text_mode:
                    logger.info("Configuring raw-text training parameters\n")
                else:
                    logger.info("Configuring text model training parameters\n")
                config_args.update(
                    {
                        "optim": optim_value,
                        "lr_scheduler_type": lr_scheduler_type_value,
                        "dataset_text_field": "text",
                    }
                )

                # Packing for text models only (DeepSeek OCR is VLM)
                if not is_deepseek_ocr:
                    packing_enabled = training_args.get("packing", False)
                    if packing_enabled and training_args.get("dataset_streaming", False):
                        logger.warning(
                            "Sequence packing is enabled with dataset streaming: "
                            "max_steps governs training length and packed-sample "
                            "counts are approximate since the stream length is unknown.\n"
                        )
                    config_args["packing"] = packing_enabled
                    logger.info(
                        f"Sequence packing: {'enabled' if packing_enabled else 'disabled'}\n"
                    )

            # Audio codec overrides — BiCodec/DAC use the text SFTTrainer path
            if self._audio_type == "bicodec":
                config_args["packing"] = False
                logger.info("Applied BiCodec overrides: packing=False\n")
            elif self._audio_type == "dac":
                config_args["packing"] = False
                logger.info("Applied DAC overrides: packing=False\n")

            logger.info(f"The configuration is: {config_args}")

            logger.info("Training configuration prepared\n")
            # ========== TRAINER INITIALIZATION ==========
            if self.is_audio_vlm and not raw_text_mode:
                # Audio VLM (e.g. Gemma 3N + audio): raw Dataset from _format_audio_vlm_dataset
                # Notebook uses processing_class=processor.tokenizer (text tokenizer only)
                # Raw-text runs are routed to the text path below.
                train_dataset = dataset["dataset"] if isinstance(dataset, dict) else dataset
                processing_class = (
                    self.tokenizer.tokenizer
                    if hasattr(self.tokenizer, "tokenizer")
                    else self.tokenizer
                )
                trainer_kwargs = {
                    "model": self.model,
                    "train_dataset": train_dataset,
                    "processing_class": processing_class,
                    "data_collator": data_collator,
                    "args": SFTConfig(**config_args),
                }
                if eval_dataset is not None:
                    trainer_kwargs["eval_dataset"] = eval_dataset
                self.trainer = SFTTrainer(**trainer_kwargs)
            elif self.is_vlm and not raw_text_mode:
                # Image VLM: dataset is a dict wrapper from
                # format_and_template_dataset. Raw-text runs go to the text path below.
                train_dataset = dataset["dataset"] if isinstance(dataset, dict) else dataset
                trainer_kwargs = {
                    "model": self.model,
                    "train_dataset": train_dataset,
                    "processing_class": self.tokenizer,
                    "data_collator": data_collator,
                    "args": SFTConfig(**config_args),
                }
                if eval_dataset is not None:
                    trainer_kwargs["eval_dataset"] = eval_dataset
                self.trainer = SFTTrainer(**trainer_kwargs)
            else:
                # For text-only training, unwrap a Processor (e.g. Gemma-3 returns
                # ProcessorMixin even for text) to the raw tokenizer; else SFTTrainer
                # sets _is_vlm=True, skips _prepare_dataset, and 'text' never gets
                # tokenized to 'input_ids'.
                from transformers import ProcessorMixin

                sft_tokenizer = self.tokenizer
                if isinstance(self.tokenizer, ProcessorMixin) and hasattr(
                    self.tokenizer, "tokenizer"
                ):
                    logger.info("Unwrapping Processor → raw tokenizer for text-only SFTTrainer")
                    sft_tokenizer = self.tokenizer.tokenizer

                if is_cpt:
                    try:
                        from unsloth import (
                            UnslothTrainer as _UnslothCPTTrainer,
                            UnslothTrainingArguments as _UnslothTrainingArguments,
                        )
                    except ImportError as exc:
                        raise RuntimeError(
                            "CPT requires a newer Unsloth install that exports "
                            "`UnslothTrainer` and `UnslothTrainingArguments` "
                            "(for embedding_learning_rate support). "
                            "Upgrade with: `pip install -U unsloth unsloth_zoo`."
                        ) from exc

                    embedding_lr = training_args.get("embedding_learning_rate")
                    logger.info(
                        f"CPT: using UnslothTrainer with embedding_learning_rate={embedding_lr}\n"
                    )
                    trainer_kwargs = {
                        "model": self.model,
                        "tokenizer": sft_tokenizer,
                        "train_dataset": dataset["dataset"],
                        "data_collator": data_collator,
                        "args": _UnslothTrainingArguments(
                            embedding_learning_rate = embedding_lr,
                            **config_args,
                        ),
                    }
                    if eval_dataset is not None:
                        trainer_kwargs["eval_dataset"] = eval_dataset
                    self.trainer = _UnslothCPTTrainer(**trainer_kwargs)
                else:
                    trainer_kwargs = {
                        "model": self.model,
                        "tokenizer": sft_tokenizer,
                        "train_dataset": dataset["dataset"],
                        "data_collator": data_collator,
                        "args": SFTConfig(**config_args),
                    }
                    if eval_dataset is not None:
                        trainer_kwargs["eval_dataset"] = eval_dataset
                    self.trainer = SFTTrainer(**trainer_kwargs)
                # Restore full processor so checkpoints include
                # preprocessor_config.json (needed for GGUF export).
                if sft_tokenizer is not self.tokenizer:
                    self.trainer.processing_class = self.tokenizer
            logger.info("Trainer initialized\n")

            # ========== TRAIN ON RESPONSES ONLY ==========
            # Raw-text datasets always train on all tokens.
            is_cpt = training_args.get("is_cpt", False)
            train_on_responses_enabled = (
                False
                if (is_cpt or raw_text_mode)
                else training_args.get("train_on_completions", False)
            )

            if is_cpt:
                logger.info("CPT mode: skipping train_on_responses_only — training on all tokens\n")
            elif raw_text_mode:
                logger.info(
                    "Raw-text mode: skipping train_on_responses_only — training on all tokens\n"
                )

            # DeepSeek OCR handles this internally in its collator, so skip
            # Audio VLM handles label masking in its collator, so skip
            # Markers auto-detected from the chat template first, manual table
            # as fallback; gpt-oss stays on its manual markers. See
            # apply_completion_masking.
            if (
                train_on_responses_enabled
                and not self.is_audio_vlm
                and not self.is_audio
                and not (is_deepseek_ocr or dataset_final_format == "alpaca")
            ):
                from unsloth.chat_templates import train_on_responses_only

                logger.info("Configuring train on responses only...\n")

                def _notify(level, message):
                    if level == "warning":
                        logger.warning(message)
                    else:
                        logger.info(f"{message}\n")

                # No try/except: the helper handles detection failures and
                # double misses itself, so an exception here is a real masking
                # failure that must fail the run, not silently train on full
                # sequences.
                self.trainer, masking_applied = apply_completion_masking(
                    self.trainer,
                    self.model_name,
                    train_on_responses_only,
                    num_proc = config_args["dataset_num_proc"],
                    notify = _notify,
                )

                if not masking_applied:
                    train_on_responses_enabled = False

                if masking_applied:
                    try:
                        # ── Safety net: check if all samples were filtered out ──
                        # train_on_responses_only masks non-response tokens with -100; a
                        # row becomes all -100 (Unsloth drops it) when the response
                        # template is not found in the formatted text. Usually a
                        # dataset/template mismatch (already-formatted data, or 'Train on
                        # completions' on data that doesn't match the model's chat
                        # template); only sometimes max_seq_length truncating the response
                        # away. Skip this len()-based check for streaming.
                        if detect_streaming_dataset(self.trainer.train_dataset):
                            logger.info("Skipping post-filter length check for streaming dataset\n")
                        else:
                            filtered_len = len(self.trainer.train_dataset)
                            original_dataset_obj = (
                                dataset["dataset"] if isinstance(dataset, dict) else dataset
                            )
                            original_len = len(original_dataset_obj)
                            dropped = original_len - filtered_len
                            drop_pct = (
                                round(100 * dropped / original_len, 1) if original_len > 0 else 0
                            )

                            if filtered_len == 0 or drop_pct > 30:
                                max_seq = training_args.get("max_seq_length", 2048)
                                error_msg = (
                                    f"{dropped}/{original_len} samples ({drop_pct}%) were "
                                    f"dropped after applying 'Train on completions': after "
                                    f"masking, those rows had no trainable response tokens "
                                    f"left. The usual cause is that this model's response "
                                    f"template was not found in the formatted samples, so "
                                    f"every token was masked out. That typically means the "
                                    f"dataset is already formatted, or its structure does "
                                    f"not match the model's chat template, so 'Train on "
                                    f"completions' should be turned off for this dataset. "
                                    f"Less commonly, a max_seq_length ({max_seq}) shorter "
                                    f"than the prompt can truncate the response away; only "
                                    f"raise it if your samples are actually longer than that."
                                )
                                logger.error(error_msg)
                                self._update_progress(error = error_msg, is_training = False)
                                return

                            if dropped > 0:
                                logger.info(
                                    f"⚠️ {dropped}/{original_len} samples "
                                    f"({drop_pct}%) were dropped (all labels "
                                    f"masked). {filtered_len} samples remain.\n"
                                )
                            logger.info(f"Post-filter dataset size: {filtered_len} samples\n")

                    except Exception as e:
                        logger.warning(f"Post-masking dataset size check failed: {e}")
            else:
                if train_on_responses_enabled and is_deepseek_ocr:
                    logger.info("Train on responses handled by DeepSeek OCR collator\n")
                else:
                    logger.info("Training on full sequences (including prompts)\n")

            # ========== PROGRESS TRACKING ==========
            self.trainer.add_callback(self._create_progress_callback())

            train_dataset_obj = dataset["dataset"] if isinstance(dataset, dict) else dataset
            is_streaming_dataset = detect_streaming_dataset(train_dataset_obj)

            max_steps_value = training_args.get("max_steps")
            max_steps = 0 if max_steps_value is None else int(max_steps_value)

            if is_streaming_dataset and max_steps <= 0:
                raise ValueError(
                    "Streaming mode requires max_steps > 0 because the training dataset has no length."
                )

            if is_streaming_dataset:
                total_steps = max_steps
            else:
                # Prefer the trainer's processed dataset length (post
                # train-on-responses filtering); fall back to the raw dataset
                # if it has no len().
                num_samples = None
                if getattr(self.trainer, "train_dataset", None) is not None:
                    try:
                        num_samples = len(self.trainer.train_dataset)
                    except TypeError:
                        num_samples = None
                if num_samples is None:
                    num_samples = len(train_dataset_obj)
                batch_size = training_args.get("batch_size", 2)
                total_steps = self._calculate_total_steps(
                    num_samples,
                    batch_size,
                    training_args.get("gradient_accumulation_steps", 4),
                    training_args.get("num_epochs", 3),
                    max_steps,
                )

            self._update_progress(total_steps = total_steps)
            # ========== START TRAINING ==========
            # Fail fast on an invalid first batch (empty/float input_ids) vs a step-1 crash.
            preflight_error = self._preflight_first_batch()
            if preflight_error:
                logger.error(preflight_error)
                self._update_progress(error = preflight_error, is_training = False)
                return

            self._update_progress(total_steps = total_steps, status_message = "Starting training...")
            logger.info("Starting training...\n")
            self.trainer.train(resume_from_checkpoint = training_args.get("resume_from_checkpoint"))

            # ========== SAVE MODEL ==========
            self._finalize_training(output_dir)

        except Exception as e:
            import traceback

            logger.error(f"Training error: {e}")
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            self._update_progress(is_training = False, error = str(e))

        finally:
            self.is_training = False

    def _patch_adapter_config(self, output_dir: str) -> None:
        """Patch adapter_config.json with unsloth_training_method.

        Values: 'qlora', 'lora', 'FT', 'CPT', 'DPO', 'GRPO', etc.
        For LoRA/QLoRA, the distinction comes from load_in_4bit.
        """
        config_path = os.path.join(output_dir, "adapter_config.json")
        if not os.path.exists(config_path):
            logger.info("No adapter_config.json found — skipping training method patch")
            return

        try:
            with open(config_path, "r", encoding = "utf-8") as f:
                config = json.load(f)

            # Determine training method
            if self.is_cpt:
                method = "CPT"
            elif self.load_in_4bit:
                method = "qlora"
            else:
                method = "lora"

            config["unsloth_training_method"] = method
            logger.info(f"Patching adapter_config.json with unsloth_training_method='{method}'")

            with open(config_path, "w", encoding = "utf-8") as f:
                json.dump(config, f, indent = 2)

        except Exception as e:
            logger.warning(f"Failed to patch adapter_config.json: {e}")

    def stop_training(self, save: bool = True):
        """Stop ongoing training"""
        logger.info(f"\nStopping training (save={save})...")
        self.should_stop = True
        self.save_on_stop = save
        stop_msg = (
            "Stopping training and saving checkpoint..." if save else "Cancelling training..."
        )
        self._update_progress(status_message = stop_msg)

        # If trainer exists, try to stop it gracefully
        if self.trainer:
            try:
                # The callback catches should_stop and stops the loop
                logger.info("Training will stop at next step...\n")
            except Exception as e:
                logger.error(f"Error stopping trainer: {e}")

    def get_training_progress(self) -> TrainingProgress:
        """Get current training progress"""
        with self._lock:
            return self.training_progress

    def cleanup(self):
        """Cleanup resources"""
        if self.trainer:
            self.trainer = None
        if self.model:
            self.model = None
        if self.tokenizer:
            self.tokenizer = None

        # Clear GPU memory
        clear_gpu_cache()


def _ensure_deepseek_ocr_installed():
    """Auto-install the DeepSeek OCR module from HF hub if missing.

    Returns True if available (already installed or just installed).
    """
    try:
        from deepseek_ocr.modeling_deepseekocr import format_messages
        logger.info("DeepSeek OCR module already available")
        return True
    except ImportError:
        pass

    try:
        logger.info("DeepSeek OCR module not found. Auto-installing from HuggingFace...")
        logger.info("\n Downloading DeepSeek OCR module from HuggingFace...\n")

        from huggingface_hub import snapshot_download
        import sys
        import os

        script_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(script_dir)  # project root

        # Download to project root as 'deepseek_ocr' folder
        local_dir = os.path.join(parent_dir, "deepseek_ocr")

        snapshot_download("unsloth/DeepSeek-OCR", local_dir = local_dir, local_dir_use_symlinks = False)

        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)

        from deepseek_ocr.modeling_deepseekocr import format_messages

        logger.info("DeepSeek OCR module installed successfully")
        logger.info("DeepSeek OCR module installed successfully!\n")
        return True

    except Exception as e:
        logger.error(f"Failed to install DeepSeek OCR module: {e}")
        logger.info(f"\n❌ Failed to install DeepSeek OCR module: {e}\n")
        return False


# Global trainer instance
_trainer_instance = None


def get_trainer() -> UnslothTrainer:
    """Get global trainer instance"""
    global _trainer_instance
    if _trainer_instance is None:
        _trainer_instance = UnslothTrainer()
    return _trainer_instance
