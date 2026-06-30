# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import time
from pathlib import Path
from typing import Optional

import typer

from unsloth_cli.config import Config, load_config
from unsloth_cli.options import add_options_from_config


class _TrainingFailed(Exception):
    pass


def _send_cli_notification(
    webhook_url,
    model,
    status,
    *,
    total_steps = None,
    final_loss = None,
    duration_s = None,
    error = None,
):
    if not webhook_url:
        return
    try:
        from studio.backend.core.training.notifications import (
            TrainingTerminalEvent,
            WebhookSink,
        )
        event = TrainingTerminalEvent(
            job_id = "cli",
            status = status,
            model = model or "",
            total_steps = total_steps or None,
            final_loss = final_loss,
            duration_s = duration_s,
            error = error if status == "error" else None,
        )
        WebhookSink(webhook_url).deliver(event)
    except Exception:
        pass  # best-effort


@add_options_from_config(Config)
def train(
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help = "Path to YAML/JSON config file. CLI flags override config values.",
    ),
    hf_token: Optional[str] = typer.Option(
        None, "--hf-token", envvar = "HF_TOKEN", help = "Hugging Face token if needed."
    ),
    wandb_token: Optional[str] = typer.Option(
        None, "--wandb-token", envvar = "WANDB_API_KEY", help = "Weights & Biases API key."
    ),
    notify_webhook: Optional[str] = typer.Option(
        None,
        "--notify-webhook",
        envvar = "UNSLOTH_NOTIFY_WEBHOOK",
        help = "Webhook URL to POST a summary when training finishes or fails "
        "(auto-formats for Slack/Discord, generic JSON otherwise).",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help = "Show resolved config and exit without training.",
    ),
    config_overrides: dict = None,
):
    """Launch training using the existing Unsloth training backend."""
    try:
        cfg = load_config(config)
    except FileNotFoundError as e:
        typer.echo(f"Error: {e}", err = True)
        raise typer.Exit(code = 2)

    cfg.apply_overrides(**config_overrides)

    # CLI/env tokens take precedence; guard against unresolved typer.Option
    # (decorator interaction)
    from typer.models import OptionInfo

    if isinstance(hf_token, OptionInfo):
        hf_token = None
    if isinstance(wandb_token, OptionInfo):
        wandb_token = None
    if isinstance(notify_webhook, OptionInfo):
        notify_webhook = None
    hf_token = hf_token or cfg.logging.hf_token
    wandb_token = wandb_token or cfg.logging.wandb_token

    if dry_run:
        import yaml

        data = cfg.model_dump()
        data["training"]["output_dir"] = str(data["training"]["output_dir"])
        typer.echo(yaml.dump(data, default_flow_style = False, sort_keys = False))
        raise typer.Exit(code = 0)

    if not cfg.model:
        typer.echo("Error: provide --model or set model in --config", err = True)
        raise typer.Exit(code = 2)

    if not cfg.data.dataset and not cfg.data.local_dataset:
        typer.echo("Error: provide --dataset or --local-dataset (or via --config)", err = True)
        raise typer.Exit(code = 2)

    # A LoRA adapter dir has adapter_config.json
    model_path = Path(cfg.model) if cfg.model else None
    model_is_lora = (
        model_path and model_path.is_dir() and (model_path / "adapter_config.json").exists()
    )
    use_lora = cfg.training.training_type.lower() == "lora"

    if model_is_lora and not use_lora:
        typer.echo(
            "Error: Cannot do full finetuning on a LoRA adapter. "
            "Use --training-type lora or provide a base model.",
            err = True,
        )
        raise typer.Exit(code = 2)

    # Single try so one notification covers any terminal state.
    interrupted = False
    status = "error"
    error_message = "Training did not complete"
    final = None
    try:
        from studio.backend.core.training.trainer import UnslothTrainer

        trainer = UnslothTrainer()

        # Load model (trainer.is_vlm is set after this)
        if not trainer.load_model(
            model_name = cfg.model,
            max_seq_length = cfg.training.max_seq_length,
            load_in_4bit = cfg.training.load_in_4bit if use_lora else False,
            hf_token = hf_token,
        ):
            raise _TrainingFailed("Model load failed")

        is_vision = trainer.is_vlm

        if not trainer.prepare_model_for_training(**cfg.model_kwargs(use_lora, is_vision)):
            raise _TrainingFailed("Model preparation failed")

        result = trainer.load_and_format_dataset(
            dataset_source = cfg.data.dataset or "",
            format_type = cfg.data.format_type,
            local_datasets = cfg.data.local_dataset,
        )
        if result is None:
            raise _TrainingFailed("Dataset load failed")

        ds, eval_ds = result

        training_kwargs = cfg.training_kwargs()
        training_kwargs["wandb_token"] = wandb_token  # CLI/env takes precedence
        if not trainer.start_training(dataset = ds, eval_dataset = eval_ds, **training_kwargs):
            raise _TrainingFailed("Training failed to start")

        try:
            while trainer.training_thread and trainer.training_thread.is_alive():
                time.sleep(1)
        except KeyboardInterrupt:
            interrupted = True
            typer.echo("Stopping training (Ctrl+C detected)...")
            trainer.stop_training()
        finally:
            if trainer.training_thread:
                trainer.training_thread.join()

        final = trainer.get_training_progress()
        if getattr(final, "error", None):
            error_message = final.error
        else:
            status = "completed"
            error_message = None
    except _TrainingFailed as exc:
        error_message = str(exc)
    except KeyboardInterrupt:
        interrupted = True  # user cancel: stay silent, then propagate
        raise
    except Exception as exc:
        error_message = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if not interrupted:
            _send_cli_notification(
                notify_webhook,
                cfg.model,
                status,
                total_steps = getattr(final, "total_steps", None),
                final_loss = getattr(final, "loss", None),
                duration_s = getattr(final, "elapsed_seconds", None),
                error = error_message,
            )

    if status == "error":
        typer.echo(error_message or "Training failed", err = True)
        raise typer.Exit(code = 1)
