import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional, List

import typer
import yaml

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


def _load_config(config_path: Optional[Path]) -> dict:
    if not config_path:
        return {}
    path = Path(config_path)
    if not path.exists():
        raise typer.BadParameter(f"Config file not found: {config_path}")
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        return yaml.safe_load(text) or {}
    else:
        return json.loads(text or "{}")


def _flatten_config(cfg: dict) -> dict:
    """
    Flatten nested config sections into a single dict.

    Expected sections:
        data: dataset, local_dataset, format_type
        training: training_type, max_seq_length, load_in_4bit, output_dir, etc.
        lora: lora_r, lora_alpha, lora_dropout, target_modules, etc.
        vision: finetune_vision_layers, finetune_language_layers, etc.
        logging: enable_wandb, wandb_project, wandb_token, enable_tensorboard, etc.
    """
    if not isinstance(cfg, dict):
        return {}

    flattened = {}

    # Handle top-level 'model' key
    if "model" in cfg:
        flattened["model"] = cfg["model"]

    sections = ["data", "training", "lora", "vision", "logging"]

    for section in sections:
        if section in cfg and isinstance(cfg[section], dict):
            flattened.update(cfg[section])

    return flattened


def _merge_config(cfg: dict, defaults: dict, overrides: dict) -> dict:
    """
    Merge CLI overrides with config and defaults.
    CLI override wins, then config value, then default.
    """
    merged = {}
    for key, default in defaults.items():
        cli_val = overrides.get(key, None)
        if cli_val is not None:
            merged[key] = cli_val
        elif key in cfg and cfg[key] is not None:
            merged[key] = cfg[key]
        else:
            merged[key] = default
    return merged


@app.command()
def train(
    model: Optional[str] = typer.Option(
        None,
        "--model",
        "-m",
        help="HF model id or local path. Required unless provided in --config.",
    ),
    training_type: Optional[str] = typer.Option(
        None,
        "--training-type",
        help="Training mode: 'lora' (LoRA/QLoRA) or 'full'. Defaults to 'lora'.",
    ),
    hf_token: Optional[str] = typer.Option(
        None, "--hf-token", envvar="HF_TOKEN", help="Hugging Face token if needed."
    ),
    max_seq_length: Optional[int] = typer.Option(None, "--max-seq-length"),
    load_in_4bit: Optional[bool] = typer.Option(
        None, "--load-in-4bit/--no-load-in-4bit"
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir",
        help="Where to store checkpoints. Defaults to ./outputs",
    ),
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
    format_type: Optional[str] = typer.Option(
        None,
        "--format-type",
        help="Dataset formatting: auto|alpaca|chatml|sharegpt. Defaults to auto.",
    ),
    num_epochs: Optional[int] = typer.Option(None, "--epochs"),
    learning_rate: Optional[float] = typer.Option(None, "--lr"),
    batch_size: Optional[int] = typer.Option(None, "--batch-size"),
    gradient_accumulation_steps: Optional[int] = typer.Option(None, "--grad-accum"),
    warmup_steps: Optional[int] = typer.Option(None, "--warmup-steps"),
    max_steps: Optional[int] = typer.Option(
        None, "--max-steps", help="Overrides epochs if >0."
    ),
    save_steps: Optional[int] = typer.Option(
        None, "--save-steps", help="0 uses trainer defaults."
    ),
    weight_decay: Optional[float] = typer.Option(None, "--weight-decay"),
    random_seed: Optional[int] = typer.Option(None, "--seed"),
    packing: Optional[bool] = typer.Option(None, "--packing/--no-packing"),
    train_on_completions: Optional[bool] = typer.Option(
        None, "--train-on-completions", help="Train on responses only when supported."
    ),
    lora_r: Optional[int] = typer.Option(None, "--lora-r"),
    lora_alpha: Optional[int] = typer.Option(None, "--lora-alpha"),
    lora_dropout: Optional[float] = typer.Option(None, "--lora-dropout"),
    gradient_checkpointing: Optional[bool] = typer.Option(
        None, "--gradient-checkpointing/--no-gradient-checkpointing"
    ),
    target_modules: Optional[str] = typer.Option(
        None,
        "--target-modules",
        help="Comma-separated target modules for LoRA.",
    ),
    vision_all_linear: Optional[bool] = typer.Option(
        None,
        "--vision-all-linear/--no-vision-all-linear",
        help="For vision models, finetune all linear layers (mirrors UI toggle).",
    ),
    finetune_vision_layers: Optional[bool] = typer.Option(
        None,
        "--finetune-vision-layers/--no-finetune-vision-layers",
        help="For vision LoRA: train vision layers.",
    ),
    finetune_language_layers: Optional[bool] = typer.Option(
        None,
        "--finetune-language-layers/--no-finetune-language-layers",
        help="For vision LoRA: train language layers.",
    ),
    finetune_attention_modules: Optional[bool] = typer.Option(
        None,
        "--finetune-attention-modules/--no-finetune-attention-modules",
        help="For vision LoRA: train attention modules.",
    ),
    finetune_mlp_modules: Optional[bool] = typer.Option(
        None,
        "--finetune-mlp-modules/--no-finetune-mlp-modules",
        help="For vision LoRA: train MLP modules.",
    ),
    use_rslora: Optional[bool] = typer.Option(None, "--rslora/--no-rslora"),
    use_loftq: Optional[bool] = typer.Option(None, "--loftq/--no-loftq"),
    enable_wandb: Optional[bool] = typer.Option(None, "--wandb/--no-wandb"),
    wandb_project: Optional[str] = typer.Option(None, "--wandb-project"),
    wandb_token: Optional[str] = typer.Option(
        None, "--wandb-token", envvar="WANDB_API_KEY"
    ),
    enable_tensorboard: Optional[bool] = typer.Option(
        None, "--tensorboard/--no-tensorboard", help="Enable TensorBoard logging."
    ),
    tensorboard_dir: Optional[str] = typer.Option(None, "--tensorboard-dir"),
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to YAML/JSON config file. CLI flags override config values.",
    ),
    verbose: bool = typer.Option(False, "--verbose/--quiet"),
):
    """
    Launch training using the existing Unsloth training backend.
    """
    cfg = _load_config(config)
    cfg = _flatten_config(cfg)

    # Defaults (match previous behavior)
    defaults = {
        "model": None,
        "training_type": "lora",
        "max_seq_length": 2048,
        "load_in_4bit": True,
        "output_dir": Path("./outputs"),
        "dataset": None,
        "local_dataset": None,
        "format_type": "auto",
        "num_epochs": 3,
        "learning_rate": 2e-4,
        "batch_size": 2,
        "gradient_accumulation_steps": 4,
        "warmup_steps": 5,
        "max_steps": 0,
        "save_steps": 0,
        "weight_decay": 0.01,
        "random_seed": 3407,
        "packing": False,
        "train_on_completions": False,
        "lora_r": 64,
        "lora_alpha": 16,
        "lora_dropout": 0.0,
        "gradient_checkpointing": True,
        "target_modules": "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        "vision_all_linear": False,
        "finetune_vision_layers": True,
        "finetune_language_layers": True,
        "finetune_attention_modules": True,
        "finetune_mlp_modules": True,
        "use_rslora": False,
        "use_loftq": False,
        "enable_wandb": False,
        "wandb_project": "unsloth-training",
        "enable_tensorboard": False,
        "tensorboard_dir": "runs",
    }

    overrides = {
        "training_type": training_type,
        "max_seq_length": max_seq_length,
        "load_in_4bit": load_in_4bit,
        "output_dir": output_dir,
        "dataset": dataset,
        "local_dataset": local_dataset,
        "format_type": format_type,
        "num_epochs": num_epochs,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "warmup_steps": warmup_steps,
        "max_steps": max_steps,
        "save_steps": save_steps,
        "weight_decay": weight_decay,
        "random_seed": random_seed,
        "packing": packing,
        "train_on_completions": train_on_completions,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "gradient_checkpointing": gradient_checkpointing,
        "target_modules": target_modules,
        "vision_all_linear": vision_all_linear,
        "finetune_vision_layers": finetune_vision_layers,
        "finetune_language_layers": finetune_language_layers,
        "finetune_attention_modules": finetune_attention_modules,
        "finetune_mlp_modules": finetune_mlp_modules,
        "use_rslora": use_rslora,
        "use_loftq": use_loftq,
        "enable_wandb": enable_wandb,
        "wandb_project": wandb_project,
        "wandb_token": wandb_token,
        "enable_tensorboard": enable_tensorboard,
        "tensorboard_dir": tensorboard_dir,
    }

    merged = _merge_config(cfg, defaults, overrides)

    model_val = merged.get("model")
    if not model_val:
        typer.echo("Error: provide --model or set model in --config", err=True)
        raise typer.Exit(code=2)

    # Convert specific types
    output_dir_val = Path(merged["output_dir"])
    dataset_val = merged.get("dataset")
    local_dataset_val = merged.get("local_dataset")

    if not dataset_val and not local_dataset_val:
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
        dropdown_value=model_val, search_value=None, hf_token=hf_token, is_lora=False
    )
    if not model_config:
        typer.echo("Could not resolve model config", err=True)
        raise typer.Exit(code=1)

    is_vision = model_config.is_vision

    if not trainer.load_model(
        model_name=model_config.identifier,
        max_seq_length=merged["max_seq_length"],
        load_in_4bit=merged["load_in_4bit"]
        if merged["training_type"].lower() == "lora"
        else False,
        hf_token=hf_token,
    ):
        typer.echo("Model load failed", err=True)
        raise typer.Exit(code=1)

    use_lora = merged["training_type"].lower() == "lora"

    # Match UI behavior for target modules:
    # - Text: use parsed target modules list
    # - Vision: if vision_all_linear, use ["all-linear"]; otherwise empty list
    target_modules_list = [
        m.strip() for m in merged["target_modules"].split(",") if m.strip()
    ]
    if use_lora and is_vision:
        if merged["vision_all_linear"]:
            target_modules_list = ["all-linear"]
        else:
            target_modules_list = []

    if not trainer.prepare_model_for_training(
        use_lora=use_lora,
        finetune_vision_layers=merged["finetune_vision_layers"],
        finetune_language_layers=merged["finetune_language_layers"],
        finetune_attention_modules=merged["finetune_attention_modules"],
        finetune_mlp_modules=merged["finetune_mlp_modules"],
        target_modules=target_modules_list,
        lora_r=merged["lora_r"],
        lora_alpha=merged["lora_alpha"],
        lora_dropout=merged["lora_dropout"],
        use_gradient_checkpointing=merged["gradient_checkpointing"],
        use_rslora=merged["use_rslora"],
        use_loftq=merged["use_loftq"],
    ):
        typer.echo("Model preparation failed", err=True)
        raise typer.Exit(code=1)

    ds = trainer.load_and_format_dataset(
        dataset_source=dataset_val or "",
        format_type=merged["format_type"],
        local_datasets=local_dataset_val,
    )
    if ds is None:
        typer.echo("Dataset load failed", err=True)
        raise typer.Exit(code=1)

    started = trainer.start_training(
        dataset=ds,
        output_dir=str(output_dir_val),
        num_epochs=merged["num_epochs"],
        learning_rate=merged["learning_rate"],
        batch_size=merged["batch_size"],
        gradient_accumulation_steps=merged["gradient_accumulation_steps"],
        warmup_steps=merged["warmup_steps"],
        max_steps=merged["max_steps"],
        save_steps=merged["save_steps"],
        weight_decay=merged["weight_decay"],
        random_seed=merged["random_seed"],
        packing=merged["packing"],
        train_on_completions=merged["train_on_completions"],
        enable_wandb=merged["enable_wandb"],
        wandb_project=merged["wandb_project"],
        wandb_token=merged.get("wandb_token"),
        enable_tensorboard=merged["enable_tensorboard"],
        tensorboard_dir=merged["tensorboard_dir"],
        max_seq_length=merged["max_seq_length"],
    )

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
