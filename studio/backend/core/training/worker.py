# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Training subprocess entry point.

Each training job runs in a fresh subprocess (mp.get_context("spawn")).
This gives us a clean Python interpreter with no stale module state —
solving the transformers version-switching problem completely.

Pattern follows core/data_recipe/jobs/worker.py.
"""

from __future__ import annotations

import structlog
from loggers import get_logger
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

logger = get_logger(__name__)


def _activate_transformers_version(model_name: str) -> None:
    """Activate the correct transformers version BEFORE any ML imports.

    If the model needs transformers 5.x, prepend the pre-installed .venv_t5/
    directory to sys.path. Otherwise do nothing (default 4.57.x in .venv/).
    """
    # Ensure backend is on path for utils imports
    backend_path = str(Path(__file__).resolve().parent.parent.parent)
    if backend_path not in sys.path:
        sys.path.insert(0, backend_path)

    from utils.transformers_version import needs_transformers_5, _resolve_base_model

    resolved = _resolve_base_model(model_name)
    if needs_transformers_5(resolved):
        venv_t5 = os.path.join(
            os.path.expanduser("~"), ".unsloth", "studio", ".venv_t5"
        )
        if os.path.isdir(venv_t5):
            sys.path.insert(0, venv_t5)
            logger.info("Activated transformers 5.x from %s", venv_t5)
        else:
            # Fallback: pip install at runtime (slower, ~10-15s)
            logger.warning(".venv_t5 not found at %s — installing at runtime", venv_t5)
            import subprocess as sp

            os.makedirs(venv_t5, exist_ok = True)
            r1 = sp.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--target",
                    venv_t5,
                    "--no-deps",
                    "transformers==5.3.0",
                ],
                stdout = sp.PIPE,
                stderr = sp.STDOUT,
            )
            r2 = sp.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--target",
                    venv_t5,
                    "--no-deps",
                    "huggingface_hub==1.3.0",
                ],
                stdout = sp.PIPE,
                stderr = sp.STDOUT,
            )
            if r1.returncode != 0 or r2.returncode != 0:
                raise RuntimeError(
                    f"Failed to install transformers 5.x into {venv_t5}. "
                    f"pip returncode: transformers={r1.returncode}, huggingface_hub={r2.returncode}"
                )
            sys.path.insert(0, venv_t5)
        # Propagate to child subprocesses (e.g. GGUF converter)
        _pp = os.environ.get("PYTHONPATH", "")
        os.environ["PYTHONPATH"] = venv_t5 + (os.pathsep + _pp if _pp else "")
    else:
        logger.info("Using default transformers (4.57.x) for %s", model_name)


def run_training_process(
    *,
    event_queue: Any,
    stop_queue: Any,
    config: dict,
) -> None:
    """Subprocess entrypoint. Fresh Python — no stale module state.

    Args:
        event_queue: mp.Queue for sending progress/status/error events to parent.
        stop_queue: mp.Queue for receiving stop commands from parent.
        config: Training configuration dict with all parameters.
    """
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PYTHONWARNINGS"] = (
        "ignore"  # Suppress warnings at C-level before imports
    )

    import warnings
    from loggers.config import LogConfig

    if os.getenv("ENVIRONMENT_TYPE", "production") == "production":
        warnings.filterwarnings("ignore")

    LogConfig.setup_logging(
        service_name = "unsloth-studio-training-worker",
        env = os.getenv("ENVIRONMENT_TYPE", "production"),
    )

    model_name = config["model_name"]

    # ── 1. Activate correct transformers version BEFORE any ML imports ──
    try:
        _activate_transformers_version(model_name)
    except Exception as exc:
        event_queue.put(
            {
                "type": "error",
                "error": f"Failed to activate transformers version: {exc}",
                "stack": traceback.format_exc(limit = 20),
                "ts": time.time(),
            }
        )
        return

    # ── 1a. Auto-enable trust_remote_code for unsloth/* transformers 5.x models ──
    # Some newer architectures (e.g. NemotronH) have config parsing bugs in
    # transformers that require trust_remote_code=True as a workaround.
    # Only auto-enable for unsloth/* prefixed models (trusted source).
    from utils.transformers_version import needs_transformers_5

    if (
        needs_transformers_5(model_name)
        and model_name.lower().startswith("unsloth/")
        and not config.get("trust_remote_code", False)
    ):
        config["trust_remote_code"] = True
        logger.info(
            "Auto-enabled trust_remote_code for unsloth/* transformers 5.x model: %s",
            model_name,
        )

    # ── 1b. Auto-install mamba-ssm for SSM/hybrid models (NemotronH, Falcon-H1) ──
    _SSM_MODEL_SUBSTRINGS = ("nemotron_h", "nemotron-3-nano", "falcon_h1", "falcon-h1")
    if any(sub in model_name.lower() for sub in _SSM_MODEL_SUBSTRINGS):
        try:
            import mamba_ssm  # noqa: F401

            logger.info("mamba-ssm already installed")
        except ImportError:
            logger.info(
                "SSM model detected — installing mamba-ssm and causal-conv1d (this may take several minutes)..."
            )
            _send_status(
                event_queue, "Installing mamba-ssm (first time only, ~7 min)..."
            )
            import subprocess as _sp

            # --no-build-isolation: compile against current torch (no version conflicts)
            # --no-deps: don't pull in torch/transformers/triton (already installed)
            for _pkg in ["causal_conv1d", "mamba_ssm"]:
                _r = _sp.run(
                    [
                        sys.executable,
                        "-m",
                        "pip",
                        "install",
                        "--no-build-isolation",
                        "--no-deps",
                        "--no-cache-dir",
                        _pkg,
                    ],
                    stdout = _sp.PIPE,
                    stderr = _sp.STDOUT,
                    text = True,
                )
                if _r.returncode != 0:
                    logger.error("Failed to install %s:\n%s", _pkg, _r.stdout)
                else:
                    logger.info("Installed %s successfully", _pkg)
            logger.info("mamba-ssm installation complete")

    # ── 1c. Set fork start method so dataset.map() can multiprocess ──
    # The parent launched us via spawn (clean process), but the compiled
    # SFTTrainer checks get_start_method() and disables num_proc if not "fork".
    # Linux only: fork is the default start method and is safe here (no CUDA
    # context exists yet). macOS defaults to spawn since Python 3.8 because
    # fork is unsafe with macOS frameworks (Metal/MPS, CoreFoundation) --
    # do NOT override on macOS. Windows has no fork at all.
    if sys.platform == "linux":
        import multiprocessing as _mp

        try:
            _mp.set_start_method("fork", force = True)
        except RuntimeError:
            pass  # Already set

    # ── 1c. On Windows, check Triton availability (must be before import torch) ──
    if sys.platform == "win32":
        try:
            import triton  # noqa: F401

            logger.info("Triton available — torch.compile enabled")
        except ImportError:
            os.environ["TORCHDYNAMO_DISABLE"] = "1"
            logger.warning(
                "Triton not found on Windows — torch.compile disabled. "
                'Install for better performance: pip install "triton-windows<3.7"'
            )

    # ── 2. Now import ML libraries (fresh in this clean process) ──
    try:
        _send_status(event_queue, "Importing Unsloth...")

        backend_path = str(Path(__file__).resolve().parent.parent.parent)
        if backend_path not in sys.path:
            sys.path.insert(0, backend_path)

        from core.training.trainer import UnslothTrainer, TrainingProgress
        from utils.paths import (
            ensure_dir,
            resolve_output_dir,
            resolve_tensorboard_dir,
            datasets_root,
        )

        import transformers

        logger.info("Subprocess loaded transformers %s", transformers.__version__)
    except Exception as exc:
        event_queue.put(
            {
                "type": "error",
                "error": f"Failed to import ML libraries: {exc}",
                "stack": traceback.format_exc(limit = 20),
                "ts": time.time(),
            }
        )
        return

    # ── 2b. EMBEDDING MODEL FAST-PATH ──
    # Embedding models use a completely different pipeline (FastSentenceTransformer
    # + SentenceTransformerTrainer + MultipleNegativesRankingLoss) so we branch
    # early and handle the entire flow in a self-contained function.
    if config.get("is_embedding", False):
        try:
            _run_embedding_training(event_queue, stop_queue, config)
        except Exception as exc:
            event_queue.put(
                {
                    "type": "error",
                    "error": str(exc),
                    "stack": traceback.format_exc(limit = 20),
                    "ts": time.time(),
                }
            )
        return

    # ── 3. Create a fresh trainer instance ──
    trainer = UnslothTrainer()

    # Wire up progress callback → event_queue
    def _on_progress(progress: TrainingProgress):
        has_train_loss = progress.step >= 0 and progress.loss > 0
        has_eval_loss = progress.eval_loss is not None
        if has_train_loss or has_eval_loss:
            event_queue.put(
                {
                    "type": "progress",
                    "step": progress.step,
                    "epoch": progress.epoch,
                    "loss": progress.loss,
                    "learning_rate": progress.learning_rate,
                    "total_steps": progress.total_steps,
                    "elapsed_seconds": progress.elapsed_seconds,
                    "eta_seconds": progress.eta_seconds,
                    "grad_norm": progress.grad_norm,
                    "num_tokens": progress.num_tokens,
                    "eval_loss": progress.eval_loss,
                    "status_message": progress.status_message,
                    "ts": time.time(),
                }
            )
        if progress.status_message:
            _send_status(event_queue, progress.status_message)

    trainer.add_progress_callback(_on_progress)

    # Wire up stop_queue polling to trainer.should_stop
    import threading
    import queue as _queue

    def _poll_stop():
        while True:
            try:
                msg = stop_queue.get(timeout = 1.0)
                if msg and msg.get("type") == "stop":
                    save = msg.get("save", True)
                    trainer.should_stop = True
                    trainer.save_on_stop = save
                    logger.info("Stop signal received (save=%s)", save)
                    return
            except _queue.Empty:
                continue
            except (EOFError, OSError):
                return

    stop_thread = threading.Thread(target = _poll_stop, daemon = True)
    stop_thread.start()

    # ── 4. Execute the training pipeline ──
    # Order: detect → dataset → model → prepare → train
    # Dataset processing (including LLM-assisted detection) runs BEFORE model
    # loading so both never occupy VRAM at the same time.
    try:
        hf_token = config.get("hf_token", "")
        hf_token = hf_token if hf_token and hf_token.strip() else None

        # ── 4a. Lightweight detection + tokenizer (no VRAM) ──
        _send_status(event_queue, "Detecting model type...")
        trainer.pre_detect_and_load_tokenizer(
            model_name = model_name,
            max_seq_length = config["max_seq_length"],
            hf_token = hf_token,
            is_dataset_image = config.get("is_dataset_image", False),
            is_dataset_audio = config.get("is_dataset_audio", False),
            trust_remote_code = config.get("trust_remote_code", False),
        )
        if trainer.should_stop:
            event_queue.put({"type": "complete", "output_dir": None, "ts": time.time()})
            return

        # ── 4b. Load and format dataset (LLM helper may use VRAM briefly) ──
        _send_status(event_queue, "Loading and formatting dataset...")
        hf_dataset = config.get("hf_dataset", "")
        dataset_result = trainer.load_and_format_dataset(
            dataset_source = hf_dataset if hf_dataset and hf_dataset.strip() else None,
            format_type = config.get("format_type", ""),
            local_datasets = config.get("local_datasets") or None,
            local_eval_datasets = config.get("local_eval_datasets") or None,
            custom_format_mapping = config.get("custom_format_mapping"),
            subset = config.get("subset"),
            train_split = config.get("train_split", "train"),
            eval_split = config.get("eval_split"),
            eval_steps = config.get("eval_steps", 0.00),
            dataset_slice_start = config.get("dataset_slice_start"),
            dataset_slice_end = config.get("dataset_slice_end"),
        )

        if isinstance(dataset_result, tuple):
            dataset, eval_dataset = dataset_result
        else:
            dataset = dataset_result
            eval_dataset = None

        # [DEBUG] Print first sample before model is loaded
        # dataset is a dict {"dataset": <Dataset>, "detected_format": ..., ...}
        # or a raw Dataset for audio paths
        # try:
        #     ds = dataset["dataset"] if isinstance(dataset, dict) else dataset
        #     print(
        #         f"\n[DEBUG] Dataset loaded BEFORE model. type={type(ds).__name__}, len={len(ds)}",
        #         flush = True,
        #     )
        #     print(f"[DEBUG] Columns: {ds.column_names}", flush = True)
        #     sample = ds[0]
        #     preview = {k: str(v)[:300] for k, v in sample.items()}
        #     print(f"[DEBUG] First sample: {preview}\n", flush = True)
        # except Exception as e:
        #     print(
        #         f"[DEBUG] Could not preview first sample: {type(e).__name__}: {e}",
        #         flush = True,
        #     )

        # Disable eval if eval_steps <= 0
        eval_steps = config.get("eval_steps", 0.00)
        if eval_steps is not None and float(eval_steps) <= 0:
            eval_dataset = None

        # Tell the parent process that eval is configured so the frontend
        # shows "Waiting for first evaluation step..." instead of "not configured"
        if eval_dataset is not None:
            event_queue.put(
                {
                    "type": "eval_configured",
                    "ts": time.time(),
                }
            )

        if dataset is None or trainer.should_stop:
            if trainer.should_stop:
                event_queue.put(
                    {"type": "complete", "output_dir": None, "ts": time.time()}
                )
            else:
                event_queue.put(
                    {
                        "type": "error",
                        "error": trainer.training_progress.error
                        or "Failed to load dataset",
                        "stack": "",
                        "ts": time.time(),
                    }
                )
            return

        # ── Start tqdm monitor early so it captures download + tokenization bars ──
        import threading as _th

        _tqdm_stop = _th.Event()

        def _monitor_tqdm():
            from tqdm.auto import tqdm as _tqdm_cls

            while not _tqdm_stop.is_set():
                for bar in list(getattr(_tqdm_cls, "_instances", set())):
                    try:
                        n, total = bar.n or 0, bar.total or 0
                        desc = getattr(bar, "desc", "") or ""
                        if total > 0 and n > 0 and desc:
                            pct = min(int(n * 100 / total), 100)
                            _send_status(
                                event_queue, f"{desc.strip()} {pct}% ({n:,}/{total:,})"
                            )
                    except (AttributeError, ReferenceError):
                        pass
                _tqdm_stop.wait(3)

        _tqdm_thread = _th.Thread(target = _monitor_tqdm, daemon = True)
        _tqdm_thread.start()

        training_type = config.get("training_type", "LoRA/QLoRA")
        use_lora = training_type == "LoRA/QLoRA"

        # ── 4c. Load training model (uses VRAM — dataset already formatted) ──
        _send_status(event_queue, "Loading model...")
        success = trainer.load_model(
            model_name = model_name,
            max_seq_length = config["max_seq_length"],
            load_in_4bit = config["load_in_4bit"],
            full_finetuning = not use_lora,
            hf_token = hf_token,
            is_dataset_image = config.get("is_dataset_image", False),
            is_dataset_audio = config.get("is_dataset_audio", False),
            trust_remote_code = config.get("trust_remote_code", False),
        )
        if not success or trainer.should_stop:
            if trainer.should_stop:
                event_queue.put(
                    {"type": "complete", "output_dir": None, "ts": time.time()}
                )
            else:
                error_msg = trainer.training_progress.error or "Failed to load model"
                event_queue.put(
                    {
                        "type": "error",
                        "error": error_msg,
                        "stack": "",
                        "ts": time.time(),
                    }
                )
            return

        # ── 4d. Prepare model (LoRA or full finetuning) ──
        if use_lora:
            _send_status(event_queue, "Configuring LoRA adapters...")
            success = trainer.prepare_model_for_training(
                use_lora = True,
                finetune_vision_layers = config.get("finetune_vision_layers", True),
                finetune_language_layers = config.get("finetune_language_layers", True),
                finetune_attention_modules = config.get(
                    "finetune_attention_modules", True
                ),
                finetune_mlp_modules = config.get("finetune_mlp_modules", True),
                target_modules = config.get("target_modules"),
                lora_r = config.get("lora_r", 16),
                lora_alpha = config.get("lora_alpha", 16),
                lora_dropout = config.get("lora_dropout", 0.0),
                use_gradient_checkpointing = config.get(
                    "gradient_checkpointing", "unsloth"
                ),
                use_rslora = config.get("use_rslora", False),
                use_loftq = config.get("use_loftq", False),
            )
        else:
            _send_status(event_queue, "Preparing model for full finetuning...")
            success = trainer.prepare_model_for_training(use_lora = False)

        if not success or trainer.should_stop:
            if trainer.should_stop:
                event_queue.put(
                    {"type": "complete", "output_dir": None, "ts": time.time()}
                )
            else:
                event_queue.put(
                    {
                        "type": "error",
                        "error": trainer.training_progress.error
                        or "Failed to prepare model",
                        "stack": "",
                        "ts": time.time(),
                    }
                )
            return

        # Convert learning rate
        try:
            lr_value = float(config.get("learning_rate", "2e-4"))
        except ValueError:
            event_queue.put(
                {
                    "type": "error",
                    "error": f"Invalid learning rate: {config.get('learning_rate')}",
                    "stack": "",
                    "ts": time.time(),
                }
            )
            return

        # Generate output dir
        output_dir = config.get("output_dir")
        if not output_dir:
            output_dir = f"{model_name.replace('/', '_')}_{int(time.time())}"
        output_dir = str(resolve_output_dir(output_dir))
        ensure_dir(Path(output_dir))

        tensorboard_dir = config.get("tensorboard_dir")
        if config.get("enable_tensorboard", False):
            tensorboard_dir = str(resolve_tensorboard_dir(tensorboard_dir))
            ensure_dir(Path(tensorboard_dir))

        # Start training (directly — no inner thread, we ARE the subprocess)
        dataset_display = (
            config.get("hf_dataset", "") or config.get("uploaded_file", "") or ""
        )
        _send_status(
            event_queue,
            f'Training "{model_name}"'
            + (f"\nDataset = {dataset_display}" if dataset_display else ""),
        )
        max_steps = config.get("max_steps", 0)
        save_steps = config.get("save_steps", 0)

        trainer._train_worker(
            dataset,
            output_dir = output_dir,
            num_epochs = config.get("num_epochs", 3),
            learning_rate = lr_value,
            batch_size = config.get("batch_size", 2),
            gradient_accumulation_steps = config.get("gradient_accumulation_steps", 4),
            warmup_steps = config.get("warmup_steps"),
            warmup_ratio = config.get("warmup_ratio"),
            max_steps = max_steps if max_steps and max_steps > 0 else 0,
            save_steps = save_steps if save_steps and save_steps > 0 else 0,
            weight_decay = config.get("weight_decay", 0.01),
            random_seed = config.get("random_seed", 3407),
            packing = config.get("packing", False),
            train_on_completions = config.get("train_on_completions", False),
            enable_wandb = config.get("enable_wandb", False),
            wandb_project = config.get("wandb_project", "unsloth-training"),
            wandb_token = config.get("wandb_token"),
            enable_tensorboard = config.get("enable_tensorboard", False),
            tensorboard_dir = tensorboard_dir,
            eval_dataset = eval_dataset,
            eval_steps = eval_steps,
            max_seq_length = config.get("max_seq_length", 2048),
            optim = config.get("optim", "adamw_8bit"),
            lr_scheduler_type = config.get("lr_scheduler_type", "linear"),
        )

        _tqdm_stop.set()

        # Check final state
        progress = trainer.get_training_progress()
        if progress.error:
            event_queue.put(
                {
                    "type": "error",
                    "error": progress.error,
                    "stack": "",
                    "ts": time.time(),
                }
            )
        else:
            event_queue.put(
                {
                    "type": "complete",
                    "output_dir": output_dir,
                    "status_message": progress.status_message or "Training completed",
                    "ts": time.time(),
                }
            )

    except Exception as exc:
        event_queue.put(
            {
                "type": "error",
                "error": str(exc),
                "stack": traceback.format_exc(limit = 20),
                "ts": time.time(),
            }
        )


def _send_status(event_queue: Any, message: str) -> None:
    """Send a status update to the parent process."""
    event_queue.put(
        {
            "type": "status",
            "message": message,
            "ts": time.time(),
        }
    )


def _run_embedding_training(event_queue: Any, stop_queue: Any, config: dict) -> None:
    """Self-contained embedding model training pipeline.

    Uses FastSentenceTransformer + SentenceTransformerTrainer +
    MultipleNegativesRankingLoss — completely separate from the
    LLM/VLM/audio paths in UnslothTrainer.

    Mirrors the pattern from the reference embedding notebooks:
      All_MiniLM_L6_v2.py, BGE_M3.py, EmbeddingGemma_300M.py,
      ModernBert.py, Qwen3_Embedding_0_6B.py
    """
    import math
    import queue as _queue
    import threading

    model_name = config["model_name"]
    training_start_time = time.time()

    # ── 1. Import embedding-specific libraries ──
    _send_status(event_queue, "Importing embedding libraries...")
    try:
        from unsloth import FastSentenceTransformer, is_bfloat16_supported
        from sentence_transformers import (
            SentenceTransformerTrainer,
            SentenceTransformerTrainingArguments,
        )
        from sentence_transformers.losses import MultipleNegativesRankingLoss
        from sentence_transformers.training_args import BatchSamplers
        from datasets import load_dataset, Dataset
        from transformers import TrainerCallback
        from utils.paths import datasets_root, resolve_output_dir
    except ImportError as e:
        event_queue.put(
            {
                "type": "error",
                "error": f"Failed to import embedding libraries: {e}. "
                "Ensure 'sentence_transformers' and 'unsloth' are installed.",
                "stack": traceback.format_exc(limit = 20),
                "ts": time.time(),
            }
        )
        return

    # ── Stop signal handling ──
    _should_stop = False
    _save_on_stop = True

    def _poll_stop():
        nonlocal _should_stop, _save_on_stop
        while True:
            try:
                msg = stop_queue.get(timeout = 1.0)
                if msg and msg.get("type") == "stop":
                    _save_on_stop = msg.get("save", True)
                    _should_stop = True
                    logger.info(
                        "Embedding training: stop signal received (save=%s)",
                        _save_on_stop,
                    )
                    return
            except _queue.Empty:
                continue
            except (EOFError, OSError):
                return

    stop_thread = threading.Thread(target = _poll_stop, daemon = True)
    stop_thread.start()

    # ── 2. Load model ──
    _send_status(event_queue, "Loading embedding model...")
    try:
        hf_token = config.get("hf_token", "")
        hf_token = hf_token if hf_token and hf_token.strip() else None
        max_seq_length = config.get("max_seq_length", 512)
        training_type = config.get("training_type", "LoRA/QLoRA")
        use_lora = training_type == "LoRA/QLoRA"

        model = FastSentenceTransformer.from_pretrained(
            model_name = model_name,
            max_seq_length = max_seq_length,
            full_finetuning = not use_lora,
            token = hf_token,
        )
    except Exception as e:
        event_queue.put(
            {
                "type": "error",
                "error": f"Failed to load embedding model '{model_name}': {e}",
                "stack": traceback.format_exc(limit = 20),
                "ts": time.time(),
            }
        )
        return

    if _should_stop:
        event_queue.put({"type": "complete", "output_dir": None, "ts": time.time()})
        return

    # ── 3. Apply LoRA ──
    if use_lora:
        _send_status(event_queue, "Configuring LoRA adapters (FEATURE_EXTRACTION)...")
        try:
            gradient_checkpointing = config.get("gradient_checkpointing", False)
            # Normalize: "none" or empty → False
            if gradient_checkpointing in ("none", "", None):
                gradient_checkpointing = False

            model = FastSentenceTransformer.get_peft_model(
                model,
                r = config.get("lora_r", 32),
                target_modules = config.get("target_modules")
                or ["q_proj", "k_proj", "v_proj", "o_proj"],
                lora_alpha = config.get("lora_alpha", 64),
                lora_dropout = config.get("lora_dropout", 0.0),
                bias = "none",
                use_gradient_checkpointing = gradient_checkpointing,
                random_state = config.get("random_seed", 3407),
                use_rslora = config.get("use_rslora", False),
                loftq_config = {"loftq_bits": 4, "loftq_iter": 1}
                if config.get("use_loftq")
                else None,
                task_type = "FEATURE_EXTRACTION",
            )
        except Exception as e:
            event_queue.put(
                {
                    "type": "error",
                    "error": f"Failed to configure LoRA for embedding model: {e}",
                    "stack": traceback.format_exc(limit = 20),
                    "ts": time.time(),
                }
            )
            return

    if _should_stop:
        event_queue.put({"type": "complete", "output_dir": None, "ts": time.time()})
        return

    # ── 4. Load dataset ──
    _send_status(event_queue, "Loading dataset...")
    try:
        hf_dataset = config.get("hf_dataset", "")
        local_datasets = config.get("local_datasets") or []
        subset = config.get("subset") or None
        train_split = config.get("train_split", "train") or "train"

        if hf_dataset and hf_dataset.strip():
            hf_token = config.get("hf_token", "")
            hf_token = hf_token if hf_token and hf_token.strip() else None
            dataset = load_dataset(
                hf_dataset.strip(),
                subset,
                split = train_split,
                token = hf_token,
            )
        elif local_datasets:
            # Load from local file(s) — mirrors the non-embedding pipeline's
            # directory handling so recipe outputs (parquet-files/) work.
            all_files: list[str] = []
            for dataset_file in local_datasets:
                file_path = (
                    dataset_file
                    if os.path.isabs(dataset_file)
                    else os.path.join(
                        str(datasets_root()),
                        dataset_file,
                    )
                )
                if os.path.isdir(file_path):
                    file_path_obj = Path(file_path)
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
                    raise ValueError(
                        f"No supported data files in directory: {file_path_obj}"
                    )
                else:
                    all_files.append(file_path)

            if all_files:
                first_ext = Path(all_files[0]).suffix.lower()
                if first_ext in (".json", ".jsonl"):
                    loader = "json"
                elif first_ext == ".csv":
                    loader = "csv"
                elif first_ext == ".parquet":
                    loader = "parquet"
                else:
                    raise ValueError(
                        f"Unsupported local dataset format: {all_files[0]}"
                    )
                dataset = load_dataset(loader, data_files = all_files, split = "train")
        else:
            event_queue.put(
                {
                    "type": "error",
                    "error": "No dataset specified for embedding training.",
                    "stack": "",
                    "ts": time.time(),
                }
            )
            return

        # Apply dataset slicing if specified
        slice_start = config.get("dataset_slice_start")
        slice_end = config.get("dataset_slice_end")
        if slice_start is not None or slice_end is not None:
            start = slice_start if slice_start is not None else 0
            end = slice_end if slice_end is not None else len(dataset)
            dataset = dataset.select(range(start, min(end + 1, len(dataset))))

        logger.info(f"Embedding dataset loaded: {len(dataset)} samples")
    except Exception as e:
        event_queue.put(
            {
                "type": "error",
                "error": f"Failed to load dataset: {e}",
                "stack": traceback.format_exc(limit = 20),
                "ts": time.time(),
            }
        )
        return

    if _should_stop:
        event_queue.put({"type": "complete", "output_dir": None, "ts": time.time()})
        return

    # ── 5. Create loss function ──
    loss = MultipleNegativesRankingLoss(model)

    # ── 6. Build training arguments ──
    _send_status(event_queue, "Configuring training...")
    try:
        lr_value = float(config.get("learning_rate", "2e-4"))
    except ValueError:
        event_queue.put(
            {
                "type": "error",
                "error": f"Invalid learning rate: {config.get('learning_rate')}",
                "stack": "",
                "ts": time.time(),
            }
        )
        return

    output_dir = config.get("output_dir")
    if not output_dir:
        output_dir = str(
            resolve_output_dir(f"{model_name.replace('/', '_')}_{int(time.time())}")
        )

    num_epochs = config.get("num_epochs", 2)
    batch_size = config.get("batch_size", 256)
    gradient_accumulation_steps = config.get("gradient_accumulation_steps", 1)
    max_steps_val = config.get("max_steps", 0)
    save_steps_val = config.get("save_steps", 0)
    warmup_ratio = config.get("warmup_ratio", 0.03)
    warmup_steps_val = config.get("warmup_steps")
    log_frequency = config.get("log_frequency", 50)

    # Build args dict
    training_args_kwargs = {
        "output_dir": output_dir,
        "per_device_train_batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "learning_rate": lr_value,
        "fp16": not is_bfloat16_supported(),
        "bf16": is_bfloat16_supported(),
        "logging_steps": 1,
        "report_to": ["wandb"] if config.get("enable_wandb") else "none",
        "lr_scheduler_type": config.get("lr_scheduler_type", "linear"),
        "batch_sampler": BatchSamplers.NO_DUPLICATES,
        "optim": config.get("optim", "adamw_8bit"),
        "weight_decay": config.get("weight_decay", 0.01),
        "seed": config.get("random_seed", 3407),
    }

    # max_steps vs epochs
    if max_steps_val and max_steps_val > 0:
        training_args_kwargs["max_steps"] = max_steps_val
    else:
        training_args_kwargs["num_train_epochs"] = num_epochs if num_epochs > 0 else 2

    # warmup: prefer warmup_ratio (standard for embedding scripts), fallback to steps
    if warmup_ratio is not None and warmup_ratio > 0:
        training_args_kwargs["warmup_ratio"] = warmup_ratio
    elif warmup_steps_val is not None and warmup_steps_val > 0:
        training_args_kwargs["warmup_steps"] = warmup_steps_val

    # save_steps
    if save_steps_val and save_steps_val > 0:
        training_args_kwargs["save_steps"] = save_steps_val
        training_args_kwargs["save_strategy"] = "steps"

    args = SentenceTransformerTrainingArguments(**training_args_kwargs)

    # ── 7. Calculate total steps for progress tracking ──
    if max_steps_val and max_steps_val > 0:
        total_steps = max_steps_val
    else:
        effective_epochs = num_epochs if num_epochs > 0 else 2
        len_dataloader = math.ceil(len(dataset) / batch_size)
        steps_per_epoch = max(len_dataloader // gradient_accumulation_steps, 1)
        total_steps = steps_per_epoch * effective_epochs

    # ── 8. Create progress callback ──
    class _EmbeddingProgressCallback(TrainerCallback):
        """Sends training progress events to the parent process via event_queue."""

        def on_log(self, args, state, control, logs = None, **kwargs):
            if not logs:
                return
            loss_value = logs.get("loss", logs.get("train_loss", 0.0))
            current_step = state.global_step

            elapsed = time.time() - training_start_time
            eta = None
            if current_step > 0 and total_steps > 0:
                remaining = total_steps - current_step
                if remaining > 0:
                    eta = (elapsed / current_step) * remaining

            event_queue.put(
                {
                    "type": "progress",
                    "step": current_step,
                    "epoch": round(state.epoch, 2) if state.epoch else 0,
                    "loss": loss_value,
                    "learning_rate": logs.get("learning_rate", 0.0),
                    "total_steps": total_steps,
                    "elapsed_seconds": elapsed,
                    "eta_seconds": eta,
                    "grad_norm": logs.get("grad_norm"),
                    "num_tokens": getattr(state, "num_input_tokens_seen", None),
                    "eval_loss": logs.get("eval_loss"),
                    "status_message": "",
                    "ts": time.time(),
                }
            )

        def on_step_end(self, args, state, control, **kwargs):
            if _should_stop:
                logger.info("Embedding training: stop at step %d", state.global_step)
                control.should_training_stop = True
                return control

    # ── 9. Create trainer and train ──
    _send_status(event_queue, "Starting embedding training...")
    try:
        trainer = SentenceTransformerTrainer(
            model = model,
            train_dataset = dataset,
            loss = loss,
            args = args,
            callbacks = [_EmbeddingProgressCallback()],
        )

        trainer.train()
    except Exception as e:
        event_queue.put(
            {
                "type": "error",
                "error": f"Embedding training failed: {e}",
                "stack": traceback.format_exc(limit = 20),
                "ts": time.time(),
            }
        )
        return

    # ── 10. Save model ──
    if _should_stop and not _save_on_stop:
        event_queue.put(
            {
                "type": "complete",
                "output_dir": None,
                "status_message": "Training cancelled",
                "ts": time.time(),
            }
        )
        return

    _send_status(event_queue, "Saving model...")
    try:
        model.save_pretrained(output_dir)
        model.tokenizer.save_pretrained(output_dir)
        logger.info("Embedding model saved to %s", output_dir)
    except Exception as e:
        logger.error("Failed to save embedding model: %s", e)
        event_queue.put(
            {
                "type": "error",
                "error": f"Training completed but failed to save: {e}",
                "stack": traceback.format_exc(limit = 20),
                "ts": time.time(),
            }
        )
        return

    # ── 11. Done ──
    event_queue.put(
        {
            "type": "complete",
            "output_dir": output_dir,
            "status_message": "Embedding training completed",
            "ts": time.time(),
        }
    )
