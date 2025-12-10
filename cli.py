import logging
import sys
import time
from pathlib import Path
from typing import Optional, List, TYPE_CHECKING

import typer

if TYPE_CHECKING:
    # Only import for type hints to avoid triggering heavy backend initialization on CLI --help
    from backend.trainer import TrainingProgress

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


def _print_progress(progress):
    parts = []
    if progress.step:
        if progress.total_steps:
            parts.append(f"step {progress.step}/{progress.total_steps}")
        else:
            parts.append(f"step {progress.step}")
    if progress.epoch:
        parts.append(f"epoch {progress.epoch}")
    if progress.loss:
        parts.append(f"loss {progress.loss:.4f}")
    if progress.learning_rate:
        parts.append(f"lr {progress.learning_rate:.2e}")
    status = progress.status_message or ""
    if not parts and status:
        line = status
    else:
        line = " | ".join(parts)
        if status:
            line = f"{line} | {status}"
    if line:
        typer.echo(line)


@app.command()
def train(
    model: str = typer.Argument(..., help="HF model id or local path."),
    dataset: Optional[str] = typer.Option(
        None,
        "--dataset",
        "-d",
        help="HF dataset to train on (e.g. 'tatsu-lab/alpaca').",
    ),
    local_dataset: Optional[List[str]] = typer.Option(
        None,
        "--local-dataset",
        help="Filename(s) under datasets/ to use (e.g. 'alpaca_unsloth.json').",
    ),
    output_dir: Path = typer.Option(
        Path("./outputs"),
        "--output-dir",
        help="Where to store checkpoints.",
    ),
    training_type: str = typer.Option(
        "lora",
        "--training-type",
        help="Training mode: 'lora' (LoRA/QLoRA) or 'full'.",
    ),
    hf_token: Optional[str] = typer.Option(
        None, "--hf-token", envvar="HF_TOKEN", help="Hugging Face token if needed."
    ),
    max_seq_length: int = typer.Option(2048, "--max-seq-length"),
    load_in_4bit: bool = typer.Option(True, "--load-in-4bit/--no-load-in-4bit"),
    num_epochs: int = typer.Option(3, "--epochs"),
    learning_rate: float = typer.Option(2e-4, "--lr"),
    batch_size: int = typer.Option(2, "--batch-size"),
    gradient_accumulation_steps: int = typer.Option(4, "--grad-accum"),
    warmup_steps: int = typer.Option(5, "--warmup-steps"),
    max_steps: int = typer.Option(0, "--max-steps", help="Overrides epochs if >0."),
    save_steps: int = typer.Option(0, "--save-steps", help="0 uses trainer defaults."),
    weight_decay: float = typer.Option(0.01, "--weight-decay"),
    random_seed: int = typer.Option(3407, "--seed"),
    packing: bool = typer.Option(False, "--packing/--no-packing"),
    train_on_completions: bool = typer.Option(
        False, "--train-on-completions", help="Train on responses only when supported."
    ),
    lora_r: int = typer.Option(64, "--lora-r"),
    lora_alpha: int = typer.Option(16, "--lora-alpha"),
    lora_dropout: float = typer.Option(0.0, "--lora-dropout"),
    gradient_checkpointing: bool = typer.Option(
        True, "--gradient-checkpointing/--no-gradient-checkpointing"
    ),
    target_modules: str = typer.Option(
        "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        "--target-modules",
        help="Comma-separated target modules for LoRA.",
    ),
    use_rslora: bool = typer.Option(False, "--rslora/--no-rslora"),
    use_loftq: bool = typer.Option(False, "--loftq/--no-loftq"),
    enable_wandb: bool = typer.Option(False, "--wandb/--no-wandb"),
    wandb_project: str = typer.Option("unsloth-training", "--wandb-project"),
    wandb_token: Optional[str] = typer.Option(
        None, "--wandb-token", envvar="WANDB_API_KEY"
    ),
    enable_tensorboard: bool = typer.Option(
        False, "--tensorboard/--no-tensorboard", help="Enable TensorBoard logging."
    ),
    tensorboard_dir: str = typer.Option("runs", "--tensorboard-dir"),
    format_type: str = typer.Option(
        "auto",
        "--format-type",
        help="Dataset formatting: auto|alpaca|chatml|sharegpt.",
    ),
    verbose: bool = typer.Option(False, "--verbose/--quiet"),
):
    """
    Launch training using the existing Unsloth training backend.
    """
    if not dataset and not local_dataset:
        typer.echo("Error: provide --dataset or --local-dataset", err=True)
        raise typer.Exit(code=2)

    # Lazy imports to avoid triggering Unsloth patches on --help
    from backend.trainer import UnslothTrainer
    from backend.model_config import ModelConfig

    configure_logging(verbose)
    trainer = UnslothTrainer()

    def progress_cb(progress: "TrainingProgress"):
        _print_progress(progress)

    trainer.add_progress_callback(progress_cb)

    model_config = ModelConfig.from_ui_selection(
        dropdown_value=model, search_value=None, hf_token=hf_token, is_lora=False
    )
    if not model_config:
        typer.echo("Could not resolve model config", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"Loading model: {model_config.identifier}")
    if not trainer.load_model(
        model_name=model_config.identifier,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit if training_type.lower() == "lora" else False,
        hf_token=hf_token,
    ):
        typer.echo("Model load failed", err=True)
        raise typer.Exit(code=1)

    use_lora = training_type.lower() == "lora"
    typer.echo(f"Preparing model for {'LoRA' if use_lora else 'full'} finetuning...")
    if not trainer.prepare_model_for_training(
        use_lora=use_lora,
        finetune_vision_layers=True,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        target_modules=[m.strip() for m in target_modules.split(",") if m.strip()],
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        use_gradient_checkpointing=gradient_checkpointing,
        use_rslora=use_rslora,
        use_loftq=use_loftq,
    ):
        typer.echo("Model preparation failed", err=True)
        raise typer.Exit(code=1)

    if not dataset and not local_dataset:
        typer.echo("Provide --dataset or --local-dataset", err=True)
        raise typer.Exit(code=2)

    typer.echo("Loading dataset...")
    ds = trainer.load_and_format_dataset(
        dataset_source=dataset or "",
        format_type=format_type,
        local_datasets=local_dataset,
    )
    if ds is None:
        typer.echo("Dataset load failed", err=True)
        raise typer.Exit(code=1)

    typer.echo("Starting training...")
    started = trainer.start_training(
        dataset=ds,
        output_dir=str(output_dir),
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=warmup_steps,
        max_steps=max_steps,
        save_steps=save_steps,
        weight_decay=weight_decay,
        random_seed=random_seed,
        packing=packing,
        train_on_completions=train_on_completions,
        enable_wandb=enable_wandb,
        wandb_project=wandb_project,
        wandb_token=wandb_token,
        enable_tensorboard=enable_tensorboard,
        tensorboard_dir=tensorboard_dir,
        max_seq_length=max_seq_length,
    )

    if not started:
        typer.echo("Training failed to start", err=True)
        raise typer.Exit(code=1)

    try:
        while trainer.training_thread and trainer.training_thread.is_alive():
            progress = trainer.get_training_progress()
            _print_progress(progress)
            time.sleep(5)
    except KeyboardInterrupt:
        typer.echo("Stopping training (Ctrl+C detected)...")
        trainer.stop_training()
    finally:
        if trainer.training_thread:
            trainer.training_thread.join()

    final = trainer.get_training_progress()
    if final.error:
        typer.echo(f"Training error: {final.error}", err=True)
        raise typer.Exit(code=1)
    typer.echo(final.status_message or "Training complete")


@app.command()
def chat(
    model: str = typer.Argument(..., help="HF model id or local path."),
    prompt: str = typer.Argument(..., help="User prompt to send."),
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
    Run a single chat turn using the inference backend.
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

    typer.echo(f"Loading model: {model_config.identifier}")
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
    for chunk in stream:
        sys.stdout.write(chunk)
        sys.stdout.flush()
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
