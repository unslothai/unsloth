import logging
import sys
import time
from pathlib import Path
from typing import Optional

import typer

from cli.config import Config, load_config
from cli.options import add_options_from_config

app = typer.Typer(
    help="Command-line interface for Unsloth training, chat, and export.",
    context_settings={"help_option_names": ["-h", "--help"]},
)


def configure_logging(verbose: bool):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%H:%M:%S",
    )


@app.command()
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
    verbose: bool = typer.Option(False, "--verbose/--quiet"),
    config_overrides: dict = None,  # Injected by decorator
):
    """
    Launch training using the existing Unsloth training backend.
    """
    try:
        cfg = load_config(config)
    except FileNotFoundError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=2)

    # Apply CLI overrides
    cfg.apply_overrides(**config_overrides)

    # Dry run: show resolved config and exit
    if dry_run:
        import yaml
        data = cfg.model_dump()
        data["training"]["output_dir"] = str(data["training"]["output_dir"])
        typer.echo(yaml.dump(data, default_flow_style=False, sort_keys=False))
        raise typer.Exit(code=0)

    # Validate required fields
    if not cfg.model:
        typer.echo("Error: provide --model or set model in --config", err=True)
        raise typer.Exit(code=2)

    if not cfg.data.dataset and not cfg.data.local_dataset:
        typer.echo(
            "Error: provide --dataset or --local-dataset (or via --config)", err=True
        )
        raise typer.Exit(code=2)

    # Lazy imports to avoid triggering Unsloth patches on --help
    from backend.trainer import UnslothTrainer
    from backend.model_config import ModelConfig

    configure_logging(verbose)
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


@app.command()
def inference(
    model: str = typer.Argument(..., help="HF model id or local path."),
    prompt: str = typer.Argument(..., help="Prompt to send to the model."),
    hf_token: Optional[str] = typer.Option(
        None, "--hf-token", envvar="HF_TOKEN", help="Hugging Face token if needed."
    ),
    temperature: float = typer.Option(0.7, "--temperature"),
    top_p: float = typer.Option(0.9, "--top-p"),
    top_k: int = typer.Option(40, "--top-k"),
    max_new_tokens: int = typer.Option(256, "--max-new-tokens"),
    repetition_penalty: float = typer.Option(1.1, "--repetition-penalty"),
    system_prompt: str = typer.Option(
        "",
        "--system-prompt",
        help="Optional system prompt to prepend.",
    ),
    max_seq_length: int = typer.Option(2048, "--max-seq-length"),
    load_in_4bit: bool = typer.Option(True, "--load-in-4bit/--no-load-in-4bit"),
    verbose: bool = typer.Option(False, "--verbose/--quiet"),
):
    """
    Run a single inference using the specified model.
    """
    # Lazy imports to avoid triggering Unsloth patches on --help
    from backend.model_config import ModelConfig
    from backend.inference import get_inference_backend

    configure_logging(verbose)
    inference_backend = get_inference_backend()
    model_config = ModelConfig.from_ui_selection(
        dropdown_value=model, search_value=None, hf_token=hf_token, is_lora=False
    )
    if not model_config:
        typer.echo("Could not resolve model config", err=True)
        raise typer.Exit(code=1)

    if not inference_backend.load_model(
        config=model_config,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        hf_token=hf_token,
    ):
        typer.echo("Model load failed", err=True)
        raise typer.Exit(code=1)

    messages = [{"role": "user", "content": prompt}]
    stream = inference_backend.generate_chat_response(
        messages=messages,
        system_prompt=system_prompt,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_new_tokens=max_new_tokens,
        repetition_penalty=repetition_penalty,
    )

    typer.echo("Assistant:", nl=True)
    previous = ""
    for chunk in stream:
        # Backend yields cumulative text; print only the delta
        delta = chunk[len(previous):]
        if delta:
            sys.stdout.write(delta)
            sys.stdout.flush()
        previous = chunk
    sys.stdout.write("\n")
    sys.stdout.flush()


@app.command("list-checkpoints")
def list_checkpoints(
    outputs_dir: Path = typer.Option(
        Path("./outputs"), "--outputs-dir", help="Directory that holds training runs."
    ),
):
    """
    List checkpoints detected in the outputs directory.
    """
    from backend.export import ExportBackend

    backend = ExportBackend()
    checkpoints = backend.scan_checkpoints(outputs_dir=str(outputs_dir))
    if not checkpoints:
        typer.echo("No checkpoints found.")
        raise typer.Exit()

    for display, path in checkpoints:
        typer.echo(f"{display}: {path}")


if __name__ == "__main__":
    app()
