# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import contextlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import typer
import yaml


@contextlib.contextmanager
def _silence():
    from rich.console import Console

    sys.stdout.flush()
    sys.stderr.flush()
    real = os.fdopen(os.dup(1), "w", closefd = True)
    saved_out, saved_err = os.dup(1), os.dup(2)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull_fd, 1)
        os.dup2(devnull_fd, 2)
        yield Console(file = real)
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved_out, 1)
        os.dup2(saved_err, 2)
        os.close(saved_out)
        os.close(saved_err)
        os.close(devnull_fd)
        real.close()


def resolve_base_model(model: str) -> Optional[str]:
    path = Path(model)
    config = path / "adapter_config.json"
    if not (path.is_dir() and config.exists()):
        return None
    try:
        data = json.loads(config.read_text(encoding = "utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return data.get("base_model_name_or_path")


def make_jsonl_task(
    data_file: Path,
    input_key: str,
    target_key: str,
    out_dir: Path,
) -> str:
    data_file = Path(data_file).resolve()
    task_name = data_file.stem
    builder = "json" if data_file.suffix.lower() in {".json", ".jsonl"} else "csv"
    task_spec = {
        "task": task_name,
        "dataset_path": builder,
        "dataset_kwargs": {"data_files": str(data_file)},
        "test_split": "train",
        "output_type": "generate_until",
        "doc_to_text": "{{" + input_key + "}}",
        "doc_to_target": "{{" + target_key + "}}",
        "generation_kwargs": {"until": ["\n"]},
        "metric_list": [
            {"metric": "exact_match", "aggregation": "mean", "higher_is_better": True},
        ],
    }
    out_dir = Path(out_dir)
    out_dir.mkdir(parents = True, exist_ok = True)
    (out_dir / f"{task_name}.yaml").write_text(
        yaml.safe_dump(task_spec, sort_keys = False), encoding = "utf-8"
    )
    return task_name


def resolve_tasks(
    tasks: str,
    input_key: str,
    target_key: str,
    tmp_dir: Path,
) -> Tuple[List[str], List[str]]:
    names: List[str] = []
    include_paths: List[str] = []

    def _add_include(directory: str) -> None:
        if directory not in include_paths:
            include_paths.append(directory)

    for raw in tasks.split(","):
        entry = raw.strip()
        if not entry:
            continue
        suffix = Path(entry).suffix.lower()

        if suffix in {".yaml", ".yml"}:
            path = Path(entry)
            if not path.exists():
                raise FileNotFoundError(f"Custom task file not found: {entry}")
            spec = yaml.safe_load(path.read_text(encoding = "utf-8")) or {}
            name = spec.get("task")
            if not name:
                raise ValueError(f"Custom task file '{entry}' is missing a 'task:' name.")
            names.append(str(name))
            _add_include(str(path.resolve().parent))

        elif suffix in {".jsonl", ".json", ".csv"}:
            path = Path(entry)
            if not path.exists():
                raise FileNotFoundError(f"Dataset file not found: {entry}")
            names.append(make_jsonl_task(path, input_key, target_key, tmp_dir))
            _add_include(str(Path(tmp_dir).resolve()))

        else:
            names.append(entry)

    if not names:
        raise ValueError("No tasks provided. Pass --tasks with at least one task.")
    return names, include_paths


def _render_results(results: dict) -> None:
    from rich.console import Console
    from rich.table import Table

    table = Table(title = "Evaluation results")
    table.add_column("Task", style = "cyan")
    table.add_column("Metric")
    table.add_column("Value", justify = "right")
    table.add_column("± stderr", justify = "right")

    for task, metrics in results.get("results", {}).items():
        for key, value in metrics.items():
            if key == "alias" or "_stderr" in key or not isinstance(value, (int, float)):
                continue
            metric, _, flt = key.partition(",")
            stderr_key = f"{metric}_stderr,{flt}" if flt else f"{metric}_stderr"
            stderr = metrics.get(stderr_key)
            stderr_str = f"{stderr:.4f}" if isinstance(stderr, (int, float)) else "—"
            table.add_row(task, key, f"{value:.4f}", stderr_str)

    Console().print(table)


def evaluate(
    model: str = typer.Argument(
        ..., help = "Path to a checkpoint/adapter directory or a HuggingFace model id."
    ),
    tasks: str = typer.Option(
        ...,
        "--tasks",
        "-t",
        help = "Comma-separated built-in task names (e.g. mmlu,gsm8k), or a path to a "
               "custom .yaml task or a .jsonl/.csv dataset.",
    ),
    base_model: Optional[str] = typer.Option(
        None,
        "--base-model",
        help = "Base model for a LoRA adapter. Auto-detected from adapter_config.json; "
               "set this to override a moved/renamed base.",
    ),
    num_fewshot: Optional[int] = typer.Option(
        None, "--num-fewshot", "-n", help = "Few-shot examples (default: per-task)."
    ),
    limit: Optional[int] = typer.Option(
        None, "--limit", help = "Cap examples per task (for quick smoke tests)."
    ),
    batch_size: str = typer.Option(
        "auto", "--batch-size", "-b", help = "Batch size, or 'auto'."
    ),
    max_seq_length: int = typer.Option(
        2048, "--max-seq-length", help = "Max sequence length for the model."
    ),
    load_in_4bit: bool = typer.Option(
        True, "--load-in-4bit/--no-load-in-4bit", help = "Load the model in 4-bit."
    ),
    backend: str = typer.Option(
        "unsloth",
        "--backend",
        help = "Model backend: 'unsloth' (fast kernels; needs an NVIDIA/AMD/Intel "
               "GPU) or 'hf' (plain transformers; works on CPU/MPS/Mac). "
               "Auto-falls back to 'hf' on Apple Silicon.",
    ),
    device: Optional[str] = typer.Option(
        None,
        "--device",
        help = "Device for the hf backend (e.g. cpu, mps, cuda). Default: auto.",
    ),
    input_key: str = typer.Option(
        "question", "--input-key", help = "Prompt field for a .jsonl/.csv dataset task."
    ),
    target_key: str = typer.Option(
        "answer", "--target-key", help = "Answer field for a .jsonl/.csv dataset task."
    ),
    output_dir: Path = typer.Option(
        Path("./eval_results"), "--output-dir", "-o", help = "Directory for results.json."
    ),
    hf_token: Optional[str] = typer.Option(
        None, "--hf-token", envvar = "HF_TOKEN", help = "HuggingFace token if needed."
    ),
):
    """Evaluate a checkpoint or LoRA adapter using lm-eval-harness."""
    try:
        import lm_eval
        from lm_eval.models.huggingface import HFLM
        from lm_eval.tasks import TaskManager
    except ImportError as e:
        typer.echo(
            "Error: evaluation requires lm-eval. Install it with "
            "`pip install unsloth[eval]`.",
            err = True,
        )
        raise typer.Exit(code = 1) from e

    tmp_dir = Path(tempfile.mkdtemp(prefix = "unsloth_eval_"))
    try:
        task_names, include_paths = resolve_tasks(tasks, input_key, target_key, tmp_dir)
    except (FileNotFoundError, ValueError) as e:
        typer.echo(f"Error: {e}", err = True)
        raise typer.Exit(code = 2) from e

    bs = int(batch_size) if batch_size.isdigit() else batch_size
    detected_base = resolve_base_model(model)
    task_manager = TaskManager(include_path = include_paths) if include_paths else None

    if backend == "unsloth":
        with _silence():
            import unsloth

        if getattr(unsloth, "DEVICE_TYPE", None) == "mlx":
            typer.echo(
                "Note: Apple Silicon (MLX) detected — Unsloth's backend can't feed "
                "lm-eval's torch wrapper, so falling back to --backend hf "
                "(plain transformers)."
            )
            backend = "hf"

    typer.echo(f"Running tasks: {', '.join(task_names)} (backend: {backend})")

    if backend == "hf":
        if device is None:
            import torch

            if torch.cuda.is_available():
                device = "cuda"
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        if bs == "auto" and device != "cuda":
            typer.echo("Note: batch_size 'auto' is slow on CPU/MPS — using 1 (override with --batch-size).")
            bs = 1
        if detected_base:
            pretrained = base_model or detected_base
            args = [f"pretrained={pretrained}", f"peft={model}"]
            typer.echo(f"Evaluating adapter '{model}' on base '{pretrained}'.")
        else:
            args = [f"pretrained={model}"]
        if load_in_4bit and device and device.startswith("cuda"):
            args.append("load_in_4bit=True")
        with _silence() as ui, ui.status(f"[cyan]Evaluating {', '.join(task_names)}…"):
            results = lm_eval.simple_evaluate(
                model = "hf",
                model_args = ",".join(args),
                tasks = task_names,
                num_fewshot = num_fewshot,
                limit = limit,
                batch_size = bs,
                device = device,
                task_manager = task_manager,
            )
    else:
        from unsloth import FastLanguageModel

        load_kwargs = dict(
            max_seq_length = max_seq_length,
            load_in_4bit = load_in_4bit,
            token = hf_token or None,
        )
        if base_model and detected_base:
            typer.echo(f"Loading base model '{base_model}' with adapter '{model}'...")
            with _silence():
                lmodel, tokenizer = FastLanguageModel.from_pretrained(
                    model_name = base_model, **load_kwargs
                )
                from peft import PeftModel

                lmodel = PeftModel.from_pretrained(lmodel, model)
        else:
            if detected_base:
                typer.echo(f"Detected LoRA adapter (base: {detected_base}).")
            typer.echo(f"Loading model: {model}")
            with _silence():
                lmodel, tokenizer = FastLanguageModel.from_pretrained(
                    model_name = model, **load_kwargs
                )

        with _silence() as ui, ui.status(f"[cyan]Evaluating {', '.join(task_names)}…"):
            FastLanguageModel.for_inference(lmodel)
            lm = HFLM(pretrained = lmodel, tokenizer = tokenizer, batch_size = bs)
            results = lm_eval.simple_evaluate(
                model = lm,
                tasks = task_names,
                num_fewshot = num_fewshot,
                limit = limit,
                task_manager = task_manager,
            )

    if results is None:
        typer.echo("Error: evaluation returned no results.", err = True)
        raise typer.Exit(code = 1)

    _render_results(results)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents = True, exist_ok = True)
    results_path = output_dir / "results.json"
    results_path.write_text(
        json.dumps(results, indent = 2, default = str), encoding = "utf-8"
    )
    typer.echo(f"Saved results to: {results_path}")
