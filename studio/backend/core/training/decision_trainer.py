# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import contextlib
import json
import math
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from loggers import get_logger

logger = get_logger(__name__)

EVAL_MAX = 2000
MIN_REPORTED_ITEMS = 50
STRUCT_COLUMNS_WARNING = (
    "The state, questions or gold columns are stored as objects rather than JSON strings, so "
    "missing fields come back as nulls and choice options can change order. Store them as JSON "
    "strings to train on what the Decision API sees."
)


class _Stopped(Exception):
    pass


def _studio_validate(name: str, question) -> None:
    from fastapi import HTTPException
    from pydantic import ValidationError

    from routes.systemone import QuestionIn, _validate

    # The Decision API's own checks, so every trained question is one it will serve.
    try:
        _validate(name, QuestionIn.model_validate(question))
    except ValidationError:
        raise ValueError("is not a valid question") from None
    except HTTPException as exc:
        raise ValueError(exc.detail["message"]) from None


def _without_nulls(value):
    if isinstance(value, dict):
        return {k: _without_nulls(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [_without_nulls(v) for v in value]
    return value


def _has_struct(feature) -> bool:
    if isinstance(feature, dict):
        return True
    inner = getattr(feature, "feature", None)
    return inner is not None and _has_struct(inner)


def _from_arrow(rows, warn: Callable[[str], None]) -> list[dict]:
    features = getattr(rows, "features", None) or {}
    if any(
        _has_struct(features[key])
        for key in ("state", "questions", "gold", "answers")
        if key in features
    ):
        warn(STRUCT_COLUMNS_WARNING)
    # Arrow pads each row with every row's keys as None; drop those, but keep "" descriptions.
    return [
        {
            **row,
            **{
                key: _without_nulls(row[key])
                for key in ("questions", "gold", "answers")
                if isinstance(row.get(key), dict)
            },
        }
        for row in rows
    ]


def _read_local_rows(paths: list[str], load_dataset, warn: Callable[[str], None]) -> list[dict]:
    from utils.datasets.cells import csv_as_text_kwargs
    from utils.paths import dataset_files_in_dir, datasets_root

    files: list[Path] = []
    for entry in paths:
        path = Path(entry if os.path.isabs(entry) else os.path.join(str(datasets_root()), entry))
        files.extend(dataset_files_in_dir(path) if path.is_dir() else [path])
    if not files:
        raise ValueError("No local dataset files found")
    rows: list[dict] = []
    for path in files:
        suffix = path.suffix.lower()
        if suffix in (".json", ".jsonl"):
            # Read directly: Arrow would merge every question's criteria keys into one struct.
            text = path.read_text(encoding = "utf-8-sig")
            try:
                data = json.loads(text)
                rows.extend(data if isinstance(data, list) else [data])
            except ValueError:
                rows.extend(json.loads(line) for line in text.splitlines() if line.strip())
        elif suffix in (".csv", ".parquet"):
            dataset = load_dataset(
                suffix[1:],
                data_files = [str(path)],
                split = "train",
                **csv_as_text_kwargs([str(path)]),
            )
            rows.extend(_from_arrow(dataset, warn))
        else:
            raise ValueError(f"Unsupported local dataset format: {path.name}")
    return [row for row in rows if isinstance(row, dict)]


def _eval_sample(rows, seed: int):
    # Every row holds at least one decision, so EVAL_MAX rows can fill the eval set.
    if len(rows) <= EVAL_MAX:
        return rows
    return [rows[i] for i in sorted(random.Random(seed).sample(range(len(rows)), EVAL_MAX))]


def _load_rows(
    config: dict,
    should_stop: Callable[[], bool],
    status: Callable[[str], None],
    warn: Callable[[str], None],
) -> tuple[list, list | None]:
    from core.training.eval_dataset import evaluation_enabled
    from core.training.s3_dataset import S3DownloadCancelled, prepare_s3_dataset_download
    from core.training.worker import _load_embedding_hf_dataset
    from utils.datasets.cache_safe import load_dataset_cache_safe as load_dataset

    evaluate = evaluation_enabled(config.get("eval_steps"))
    seed = config["random_seed"]
    eval_rows = None
    # Local files, then S3, then the Hub: the order the route's cached-dataset check relies on.
    if config.get("local_datasets"):
        rows = _read_local_rows(config["local_datasets"], load_dataset, warn)
    elif config.get("s3_config"):
        status("Downloading dataset from S3...")
        try:
            download = prepare_s3_dataset_download(config["s3_config"], cancel_callback = should_stop)
        except S3DownloadCancelled:
            raise _Stopped("Training cancelled") from None
        try:
            rows = _read_local_rows(download.files, load_dataset, warn)
        finally:
            download.cleanup()
    elif config.get("hf_dataset"):
        rows = _from_arrow(_load_embedding_hf_dataset(config, load_dataset, status), warn)
        eval_split = config.get("eval_split") if evaluate else None
        if eval_split and eval_split == (config.get("train_split") or "train"):
            warn(
                "The evaluation split is the training split, so part of it is held out "
                "of training for evaluation instead."
            )
        elif eval_split:
            # Loaded like the train split, from the same cached snapshot, without the row slice.
            held = {
                **config,
                "train_split": eval_split,
                "dataset_slice_start": None,
                "dataset_slice_end": None,
            }
            eval_rows = _from_arrow(
                _eval_sample(_load_embedding_hf_dataset(held, load_dataset, status), seed), warn
            )
    else:
        raise ValueError("No dataset specified for decision training.")
    if evaluate and config.get("local_eval_datasets"):
        eval_rows = _eval_sample(
            _read_local_rows(config["local_eval_datasets"], load_dataset, warn), seed
        )

    start, end = config.get("dataset_slice_start"), config.get("dataset_slice_end")
    if start is not None or end is not None:
        rows = rows[start or 0 : (len(rows) if end is None else end + 1)]
    return rows, eval_rows


def run_decision_training(event_queue: Any, stop_queue: Any, config: dict) -> None:
    import torch

    from utils.paths import is_local_path, resolve_output_dir
    from utils.training_runs import build_default_output_dir_name

    model_name, subfolder = config["model_name"], config.get("model_subfolder")
    output_dir = str(
        resolve_output_dir(
            config.get("output_dir")
            or build_default_output_dir_name(
                model_name
                if is_local_path(model_name) or not subfolder
                else f"{model_name}-{subfolder}",
                config.get("project_name"),
            )
        )
    )
    try:
        _run(event_queue, stop_queue, config, output_dir)
    except _Stopped as stopped:
        event_queue.put(
            {
                "type": "complete",
                "output_dir": None,
                "status_message": str(stopped),
                "ts": time.time(),
            }
        )
    except torch.cuda.OutOfMemoryError:
        event_queue.put(
            {
                "type": "error",
                "error": (
                    "Out of GPU memory while training the decision model. Lower the batch size "
                    "and raise gradient accumulation to keep the same effective batch."
                ),
                "stack": "",
                "ts": time.time(),
            }
        )
    finally:
        # The trainer creates the folder at init; a run that saved nothing leaves none behind.
        with contextlib.suppress(OSError):
            os.rmdir(output_dir)


def _run(event_queue: Any, stop_queue: Any, config: dict, output_dir: str) -> None:
    from core.import_guards import ensure_real_packages

    ensure_real_packages("unsloth_zoo", "unsloth")
    from unsloth import DecisionTrainer, FastDecisionModel, is_bfloat16_supported

    import torch
    from huggingface_hub.errors import LocalEntryNotFoundError
    from transformers import TrainingArguments

    from core.systemone import laya_runtime
    from core.systemone.catalog import Checkpoint
    from core.training.eval_dataset import evaluation_enabled
    from core.training.trainer import (
        _drop_hf_stdout_callbacks,
        _hf_stdout_progress_disabled,
        normalize_gradient_checkpointing,
    )
    from core.training.worker import (
        _create_embedding_progress_callback,
        _emit_output_dir,
        _send_status,
        _start_worker_stop_poller,
        _worker_hf_token,
    )

    def send(kind: str, **payload) -> None:
        event_queue.put({"type": kind, **payload, "ts": time.time()})

    def status(message: str) -> None:
        _send_status(event_queue, message)

    def warn(message: str) -> None:
        send("warning", message = message)

    stop = {"requested": False, "save": True}

    def on_stop(save: bool) -> None:
        stop["requested"], stop["save"] = True, save
        logger.info("Decision training: stop signal received (save=%s)", save)

    def check_stop() -> None:
        if stop["requested"]:
            raise _Stopped("Training stopped" if stop["save"] else "Training cancelled")

    def validate(name: str, question) -> None:
        # build_dataset calls this for every decision, so a stop does not wait for the whole dataset.
        check_stop()
        _studio_validate(name, question)

    _start_worker_stop_poller(stop_queue, on_stop)

    model_name = config["model_name"]
    subfolder = config.get("model_subfolder") or None
    hf_token = _worker_hf_token(config)
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
    seed = config["random_seed"]
    use_lora = config["training_type"] == "LoRA/QLoRA"
    gradient_checkpointing = normalize_gradient_checkpointing(config["gradient_checkpointing"])

    status("Loading decision model...")
    try:
        root = laya_runtime._checkpoint_dir(Checkpoint("base", model_name, subfolder, ""))
    except LocalEntryNotFoundError as exc:
        send("error", error = f"Could not download {model_name}: {exc}", stack = "")
        return
    except FileNotFoundError as exc:
        send("error", error = f"Not a Laya decision checkpoint: {exc}", stack = "")
        return
    model, tokenizer = FastDecisionModel.from_pretrained(
        str(root),
        subfolder = subfolder,
        full_finetuning = not use_lora,
        use_gradient_checkpointing = gradient_checkpointing,
    )
    if use_lora:
        model = FastDecisionModel.get_peft_model(
            model,
            r = config["lora_r"],
            lora_alpha = config["lora_alpha"],
            lora_dropout = config["lora_dropout"],
            use_gradient_checkpointing = gradient_checkpointing,
            random_state = seed,
            use_rslora = config["use_rslora"],
        )
    check_stop()

    status("Loading dataset...")
    rows, eval_rows = _load_rows(config, lambda: stop["requested"], status, warn)
    check_stop()
    status("Preparing decisions...")
    items, report = FastDecisionModel.build_dataset(rows, tokenizer, model, validate)
    eval_items = []
    if eval_rows is not None:
        eval_items, eval_report = FastDecisionModel.build_dataset(
            eval_rows, tokenizer, model, validate
        )
        report = {
            "total": report["total"] + eval_report["total"],
            "skipped": report["skipped"] + eval_report["skipped"],
            "reason": report["reason"] or eval_report["reason"],
        }
    if not items:
        reason = f" ({report['reason']})" if report["reason"] else ""
        send(
            "error",
            error = f"No usable decisions in the dataset{reason}. Each row needs state, questions and gold.",
            stack = "",
        )
        return
    if report["skipped"]:
        warn(f"Skipped {report['skipped']:,} of {report['total']:,} decisions: {report['reason']}.")
    if not eval_items:
        items, eval_items = FastDecisionModel.split_holdout(items, seed)
        if eval_items:
            status(f"Holding out {len(eval_items):,} decisions to calibrate confidence...")
    elif len(eval_items) > EVAL_MAX:
        eval_items = random.Random(seed).sample(eval_items, EVAL_MAX)

    batch_size, accumulation = config["batch_size"], config["gradient_accumulation_steps"]
    max_steps, epochs = config["max_steps"] or 0, max(1, config["num_epochs"])
    steps_per_epoch = max(1, math.ceil(math.ceil(len(items) / batch_size) / accumulation))
    total_steps = max_steps if max_steps > 0 else steps_per_epoch * epochs
    bf16 = is_bfloat16_supported()
    arguments = {
        "output_dir": output_dir,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": 16,
        "gradient_accumulation_steps": accumulation,
        "learning_rate": float(config["learning_rate"]),
        "weight_decay": config["weight_decay"],
        "lr_scheduler_type": config["lr_scheduler_type"],
        "optim": config["optim"],
        "seed": seed,
        "bf16": bf16,
        "fp16": torch.cuda.is_available() and not bf16,
        "logging_steps": 1,
        "report_to": "none",
        "disable_tqdm": _hf_stdout_progress_disabled(),
        "warmup_steps": config["warmup_steps"]
        or round((config["warmup_ratio"] or 0.0) * total_steps),
        "save_strategy": "no",
    }
    if max_steps > 0:
        arguments["max_steps"] = max_steps
    else:
        arguments["num_train_epochs"] = epochs
    if config["max_grad_norm"] is not None:
        arguments["max_grad_norm"] = config["max_grad_norm"]
    if eval_items:
        if evaluation_enabled(config["eval_steps"]):
            arguments.update(eval_strategy = "steps", eval_steps = float(config["eval_steps"]))
        else:
            arguments["eval_strategy"] = "epoch"
        send("eval_configured")
    report_to = []
    if config.get("enable_tensorboard"):
        from utils.paths import resolve_tensorboard_dir
        report_to.append("tensorboard")
        # transformers 5 ignores logging_dir; its TensorBoard callback reads this instead.
        os.environ["TENSORBOARD_LOGGING_DIR"] = str(
            resolve_tensorboard_dir(config.get("tensorboard_dir"))
        )
    if config["enable_wandb"]:
        try:
            import wandb

            if config.get("wandb_token"):
                os.environ["WANDB_API_KEY"] = config["wandb_token"]
            wandb.init(project = config.get("wandb_project") or "unsloth-training")
            report_to.append("wandb")
        except Exception as exc:
            warn(f"Weights & Biases logging is off: {exc}")
    arguments["report_to"] = report_to or "none"

    base_metrics = None
    if eval_items:
        status(f"Evaluating the base model on {len(eval_items):,} held-out decisions...")
        base_metrics = FastDecisionModel.evaluate(model, tokenizer, eval_items)
        logger.info("Base held-out metrics: %s", base_metrics)
    check_stop()

    _emit_output_dir(event_queue, output_dir)
    trainer = DecisionTrainer(
        model = model,
        args = TrainingArguments(**arguments),
        train_dataset = items,
        eval_dataset = eval_items or None,
        processing_class = tokenizer,
        callbacks = [
            _create_embedding_progress_callback(
                event_queue,
                total_steps = total_steps,
                training_start_time = time.time(),
                should_stop = lambda: stop["requested"],
            )
        ],
    )
    _drop_hf_stdout_callbacks(trainer)
    start = time.time()
    trainer.train()
    if stop["requested"] and not stop["save"]:
        raise _Stopped("Training cancelled")

    tuned_metrics = None
    if eval_items:
        status("Calibrating confidence...")
        tuned_metrics = FastDecisionModel.calibrate(model, tokenizer, eval_items)
        logger.info("Fine-tuned held-out metrics: %s", tuned_metrics)

    status("Saving model...")
    peak_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else None
    model.decision_config["training"] = {
        "base": model_name,
        "subfolder": subfolder,
        "method": "lora" if use_lora else "full",
        "objective": "soft_cross_entropy",
        "dataset": [Path(path).name for path in config.get("local_datasets") or []]
        or config.get("hf_dataset")
        or None,
        "steps": trainer.state.global_step,
        "epochs": round(trainer.state.epoch or 0, 2),
        "heldout_decisions": len(eval_items),
        "heldout_accuracy_base": base_metrics and round(base_metrics["accuracy"], 4),
        "heldout_accuracy": tuned_metrics and round(tuned_metrics["accuracy"], 4),
        "heldout_ece_base": base_metrics and round(base_metrics["ece"], 4),
        "heldout_ece": tuned_metrics and round(tuned_metrics["ece"], 4),
        "seconds": round(time.time() - start, 1),
        "peak_memory_gb": peak_gb and round(peak_gb, 2),
        "date": datetime.now(timezone.utc).isoformat(timespec = "seconds"),
    }
    model.save_pretrained_merged(output_dir, tokenizer)
    logger.info("Decision model saved to %s: %s", output_dir, model.decision_config["training"])

    message = "Decision training completed"
    if base_metrics and tuned_metrics and len(eval_items) >= MIN_REPORTED_ITEMS:
        message = (
            f"Held-out accuracy {base_metrics['accuracy']:.2f} -> {tuned_metrics['accuracy']:.2f}, "
            f"calibration error {tuned_metrics['ece']:.2f}"
        )
    send("complete", output_dir = output_dir, status_message = message)
