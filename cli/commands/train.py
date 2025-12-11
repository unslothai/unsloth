import time
from pathlib import Path
from typing import Optional

import typer

from cli.config import Config, load_config
from cli.options import add_options_from_config


@add_options_from_config(Config)
def train(
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to YAML/JSON config file. CLI flags override config values.",
    ),
    hf_token: Optional[str] = typer.Option(
        None, "--hf-token", envvar="HF_TOKEN", help="Hugging Face token if needed."
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Show resolved config and exit without training.",
    ),
    config_overrides: dict = None,
):
    """Launch training using the existing Unsloth training backend."""
    try:
        cfg = load_config(config)
    except FileNotFoundError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=2)

    cfg.apply_overrides(**config_overrides)

    if dry_run:
        import yaml
        data = cfg.model_dump()
        data["training"]["output_dir"] = str(data["training"]["output_dir"])
        typer.echo(yaml.dump(data, default_flow_style=False, sort_keys=False))
        raise typer.Exit(code=0)

    if not cfg.model:
        typer.echo("Error: provide --model or set model in --config", err=True)
        raise typer.Exit(code=2)

    if not cfg.data.dataset and not cfg.data.local_dataset:
        typer.echo(
            "Error: provide --dataset or --local-dataset (or via --config)", err=True
        )
        raise typer.Exit(code=2)

    from backend.trainer import UnslothTrainer
    from backend.model_config import ModelConfig

    trainer = UnslothTrainer()

    model_config = ModelConfig.from_ui_selection(
        dropdown_value=cfg.model, search_value=None, hf_token=hf_token, is_lora=False
    )
    if not model_config:
        typer.echo("Could not resolve model config", err=True)
        raise typer.Exit(code=1)

    is_vision = model_config.is_vision
    use_lora = cfg.training.training_type.lower() == "lora"

    if not trainer.load_model(
        model_name=model_config.identifier,
        max_seq_length=cfg.training.max_seq_length,
        load_in_4bit=cfg.training.load_in_4bit if use_lora else False,
        hf_token=hf_token,
    ):
        typer.echo("Model load failed", err=True)
        raise typer.Exit(code=1)

    if not trainer.prepare_model_for_training(**cfg.model_kwargs(use_lora, is_vision)):
        typer.echo("Model preparation failed", err=True)
        raise typer.Exit(code=1)

    ds = trainer.load_and_format_dataset(
        dataset_source=cfg.data.dataset or "",
        format_type=cfg.data.format_type,
        local_datasets=cfg.data.local_dataset,
    )
    if ds is None:
        typer.echo("Dataset load failed", err=True)
        raise typer.Exit(code=1)

    started = trainer.start_training(dataset=ds, **cfg.training_kwargs())

    if not started:
        typer.echo("Training failed to start", err=True)
        raise typer.Exit(code=1)

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
        typer.echo(f"Training error: {final.error}", err=True)
        raise typer.Exit(code=1)
