# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import time
from pathlib import Path
from typing import Optional

import typer

from unsloth_cli.config import Config, load_config
from unsloth_cli.options import add_options_from_config


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

    # CLI/env tokens take precedence over config
    # Handle case where typer.Option isn't resolved (decorator interaction)
    from typer.models import OptionInfo

    if isinstance(hf_token, OptionInfo):
        hf_token = None
    if isinstance(wandb_token, OptionInfo):
        wandb_token = None
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
        typer.echo(
            "Error: provide --dataset or --local-dataset (or via --config)", err = True
        )
        raise typer.Exit(code = 2)

    # Check if the model path is a LoRA adapter (has adapter_config.json)
    model_path = Path(cfg.model) if cfg.model else None
    model_is_lora = (
        model_path
        and model_path.is_dir()
        and (model_path / "adapter_config.json").exists()
    )
    use_lora = cfg.training.training_type.lower() == "lora"

    if model_is_lora and not use_lora:
        typer.echo(
            "Error: Cannot do full finetuning on a LoRA adapter. "
            "Use --training-type lora or provide a base model.",
            err = True,
        )
        raise typer.Exit(code = 2)

    from studio.backend.core.training.trainer import UnslothTrainer

    trainer = UnslothTrainer()

    # Load model (trainer.is_vlm is set after this)
    if not trainer.load_model(
        model_name = cfg.model,
        max_seq_length = cfg.training.max_seq_length,
        load_in_4bit = cfg.training.load_in_4bit if use_lora else False,
        hf_token = hf_token,
    ):
        typer.echo("Model load failed", err = True)
        raise typer.Exit(code = 1)

    is_vision = trainer.is_vlm

    if not trainer.prepare_model_for_training(**cfg.model_kwargs(use_lora, is_vision)):
        typer.echo("Model preparation failed", err = True)
        raise typer.Exit(code = 1)

    result = trainer.load_and_format_dataset(
        dataset_source = cfg.data.dataset or "",
        format_type = cfg.data.format_type,
        local_datasets = cfg.data.local_dataset,
    )
    if result is None:
        typer.echo("Dataset load failed", err = True)
        raise typer.Exit(code = 1)

    ds, eval_ds = result

    training_kwargs = cfg.training_kwargs()
    training_kwargs["wandb_token"] = wandb_token  # CLI/env takes precedence
    started = trainer.start_training(
        dataset = ds, eval_dataset = eval_ds, **training_kwargs
    )

    if not started:
        typer.echo("Training failed to start", err = True)
        raise typer.Exit(code = 1)

    try:
        while trainer.training_thread and trainer.training_thread.is_alive():
            time.sleep(1)
    except KeyboardInterrupt:
        typer.echo("Stopping training (Ctrl+C detected)...")
        trainer.stop_training()
    finally:
        if trainer.training_thread:
            trainer.training_thread.join()

    final = trainer.get_training_progress()
    if getattr(final, "error", None):
        typer.echo(f"Training error: {final.error}", err = True)
        raise typer.Exit(code = 1)
