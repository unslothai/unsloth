"""
Training subprocess entry point.

Each training job runs in a fresh subprocess (mp.get_context("spawn")).
This gives us a clean Python interpreter with no stale module state —
solving the transformers version-switching problem completely.

Pattern follows core/data_recipe/jobs/worker.py.
"""
from __future__ import annotations

import logging
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _activate_transformers_version(model_name: str, project_root: str) -> None:
    """Activate the correct transformers version BEFORE any ML imports.

    If the model needs transformers 5.x, prepend the pre-installed .venv_t5/
    directory to sys.path. Otherwise do nothing (default 4.57.x in .venv/).
    """
    # Ensure backend is on path for utils imports
    backend_path = os.path.join(project_root, "studio", "backend")
    if backend_path not in sys.path:
        sys.path.insert(0, backend_path)

    from utils.transformers_version import needs_transformers_5, _resolve_base_model

    resolved = _resolve_base_model(model_name)
    if needs_transformers_5(resolved):
        venv_t5 = os.path.join(project_root, ".venv_t5")
        if os.path.isdir(venv_t5):
            sys.path.insert(0, venv_t5)
            logger.info("Activated transformers 5.x from %s", venv_t5)
        else:
            # Fallback: pip install at runtime (slower, ~10-15s)
            logger.warning(".venv_t5 not found at %s — installing at runtime", venv_t5)
            import subprocess as sp
            os.makedirs(venv_t5, exist_ok=True)
            r1 = sp.run(
                [sys.executable, "-m", "pip", "install", "--target", venv_t5,
                 "--no-deps", "transformers==5.2.0"],
                stdout=sp.PIPE, stderr=sp.STDOUT,
            )
            r2 = sp.run(
                [sys.executable, "-m", "pip", "install", "--target", venv_t5,
                 "--no-deps", "huggingface_hub==1.3.0"],
                stdout=sp.PIPE, stderr=sp.STDOUT,
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

    project_root = config["project_root"]
    model_name = config["model_name"]

    # ── 1. Activate correct transformers version BEFORE any ML imports ──
    try:
        _activate_transformers_version(model_name, project_root)
    except Exception as exc:
        event_queue.put({
            "type": "error",
            "error": f"Failed to activate transformers version: {exc}",
            "stack": traceback.format_exc(limit=20),
            "ts": time.time(),
        })
        return

    # ── 2. Now import ML libraries (fresh in this clean process) ──
    try:
        _send_status(event_queue, "Importing ML libraries...")

        backend_path = os.path.join(project_root, "studio", "backend")
        if backend_path not in sys.path:
            sys.path.insert(0, backend_path)

        from core.training.trainer import UnslothTrainer, TrainingProgress

        import transformers
        logger.info("Subprocess loaded transformers %s", transformers.__version__)
    except Exception as exc:
        event_queue.put({
            "type": "error",
            "error": f"Failed to import ML libraries: {exc}",
            "stack": traceback.format_exc(limit=20),
            "ts": time.time(),
        })
        return

    # ── 3. Create a fresh trainer instance ──
    trainer = UnslothTrainer()

    # Wire up progress callback → event_queue
    def _on_progress(progress: TrainingProgress):
        if progress.step >= 0 and progress.loss > 0:
            event_queue.put({
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
            })
        if progress.status_message:
            _send_status(event_queue, progress.status_message)

    trainer.add_progress_callback(_on_progress)

    # Wire up stop_queue polling to trainer.should_stop
    import threading
    import queue as _queue

    def _poll_stop():
        while True:
            try:
                msg = stop_queue.get(timeout=1.0)
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

    stop_thread = threading.Thread(target=_poll_stop, daemon=True)
    stop_thread.start()

    # ── 4. Execute the training pipeline ──
    try:
        hf_token = config.get("hf_token", "")
        hf_token = hf_token if hf_token and hf_token.strip() else None

        # Load model
        _send_status(event_queue, "Loading model...")
        success = trainer.load_model(
            model_name=model_name,
            max_seq_length=config["max_seq_length"],
            load_in_4bit=config["load_in_4bit"],
            hf_token=hf_token,
            is_dataset_multimodal=config.get("is_dataset_multimodal", False),
        )
        if not success or trainer.should_stop:
            if trainer.should_stop:
                event_queue.put({"type": "complete", "output_dir": None, "ts": time.time()})
            else:
                event_queue.put({
                    "type": "error",
                    "error": trainer.training_progress.error or "Failed to load model",
                    "stack": "", "ts": time.time(),
                })
            return

        # Prepare model (LoRA or full finetuning)
        training_type = config.get("training_type", "LoRA/QLoRA")
        use_lora = (training_type == "LoRA/QLoRA")
        if use_lora:
            _send_status(event_queue, "Configuring LoRA adapters...")
            success = trainer.prepare_model_for_training(
                use_lora=True,
                finetune_vision_layers=config.get("finetune_vision_layers", True),
                finetune_language_layers=config.get("finetune_language_layers", True),
                finetune_attention_modules=config.get("finetune_attention_modules", True),
                finetune_mlp_modules=config.get("finetune_mlp_modules", True),
                target_modules=config.get("target_modules"),
                lora_r=config.get("lora_r", 16),
                lora_alpha=config.get("lora_alpha", 16),
                lora_dropout=config.get("lora_dropout", 0.0),
                use_gradient_checkpointing=config.get("gradient_checkpointing", "unsloth"),
                use_rslora=config.get("use_rslora", False),
                use_loftq=config.get("use_loftq", False),
            )
        else:
            _send_status(event_queue, "Preparing model for full finetuning...")
            success = trainer.prepare_model_for_training(use_lora=False)

        if not success or trainer.should_stop:
            if trainer.should_stop:
                event_queue.put({"type": "complete", "output_dir": None, "ts": time.time()})
            else:
                event_queue.put({
                    "type": "error",
                    "error": trainer.training_progress.error or "Failed to prepare model",
                    "stack": "", "ts": time.time(),
                })
            return

        # Load dataset
        _send_status(event_queue, "Loading and formatting dataset...")
        hf_dataset = config.get("hf_dataset", "")
        dataset_result = trainer.load_and_format_dataset(
            dataset_source=hf_dataset if hf_dataset and hf_dataset.strip() else None,
            format_type=config.get("format_type", ""),
            local_datasets=config.get("local_datasets") or None,
            custom_format_mapping=config.get("custom_format_mapping"),
            subset=config.get("subset"),
            train_split=config.get("train_split", "train"),
            eval_split=config.get("eval_split"),
            eval_steps=config.get("eval_steps", 0.00),
            dataset_slice_start=config.get("dataset_slice_start"),
            dataset_slice_end=config.get("dataset_slice_end"),
        )

        if isinstance(dataset_result, tuple):
            dataset, eval_dataset = dataset_result
        else:
            dataset = dataset_result
            eval_dataset = None

        # Disable eval if eval_steps <= 0
        eval_steps = config.get("eval_steps", 0.00)
        if eval_steps is not None and float(eval_steps) <= 0:
            eval_dataset = None

        if dataset is None or trainer.should_stop:
            if trainer.should_stop:
                event_queue.put({"type": "complete", "output_dir": None, "ts": time.time()})
            else:
                event_queue.put({
                    "type": "error",
                    "error": trainer.training_progress.error or "Failed to load dataset",
                    "stack": "", "ts": time.time(),
                })
            return

        # Convert learning rate
        try:
            lr_value = float(config.get("learning_rate", "2e-4"))
        except ValueError:
            event_queue.put({
                "type": "error",
                "error": f"Invalid learning rate: {config.get('learning_rate')}",
                "stack": "", "ts": time.time(),
            })
            return

        # Generate output dir
        output_dir = config.get("output_dir")
        if not output_dir:
            output_dir = f"./outputs/{model_name.replace('/', '_')}_{int(time.time())}"

        # Start training (directly — no inner thread, we ARE the subprocess)
        _send_status(event_queue, "Starting training...")
        max_steps = config.get("max_steps", 0)
        save_steps = config.get("save_steps", 0)

        trainer._train_worker(
            dataset,
            output_dir=output_dir,
            num_epochs=config.get("num_epochs", 3),
            learning_rate=lr_value,
            batch_size=config.get("batch_size", 2),
            gradient_accumulation_steps=config.get("gradient_accumulation_steps", 4),
            warmup_steps=config.get("warmup_steps"),
            warmup_ratio=config.get("warmup_ratio"),
            max_steps=max_steps if max_steps and max_steps > 0 else 0,
            save_steps=save_steps if save_steps and save_steps > 0 else 0,
            weight_decay=config.get("weight_decay", 0.01),
            random_seed=config.get("random_seed", 3407),
            packing=config.get("packing", False),
            train_on_completions=config.get("train_on_completions", False),
            enable_wandb=config.get("enable_wandb", False),
            wandb_project=config.get("wandb_project", "unsloth-training"),
            wandb_token=config.get("wandb_token"),
            enable_tensorboard=config.get("enable_tensorboard", False),
            tensorboard_dir=config.get("tensorboard_dir", "runs"),
            eval_dataset=eval_dataset,
            eval_steps=eval_steps,
            max_seq_length=config.get("max_seq_length", 2048),
            optim=config.get("optim", "adamw_8bit"),
            lr_scheduler_type=config.get("lr_scheduler_type", "linear"),
        )

        # Check final state
        progress = trainer.get_training_progress()
        if progress.error:
            event_queue.put({
                "type": "error",
                "error": progress.error,
                "stack": "",
                "ts": time.time(),
            })
        else:
            event_queue.put({
                "type": "complete",
                "output_dir": output_dir,
                "status_message": progress.status_message or "Training completed",
                "ts": time.time(),
            })

    except Exception as exc:
        event_queue.put({
            "type": "error",
            "error": str(exc),
            "stack": traceback.format_exc(limit=20),
            "ts": time.time(),
        })


def _send_status(event_queue: Any, message: str) -> None:
    """Send a status update to the parent process."""
    event_queue.put({
        "type": "status",
        "message": message,
        "ts": time.time(),
    })
