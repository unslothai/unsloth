# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Dataset diagnostics without loading model weights."""

import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Optional

import typer

from unsloth_cli.config import ConfigError, load_config

dataset_app = typer.Typer(help = "Check training datasets before loading model weights.")


def _run_audit(request):
    # Same accelerator isolation as Studio's token preview, before torch imports in the child.
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "",
        "HIP_VISIBLE_DEVICES": "",
        "ONEAPI_DEVICE_SELECTOR": "opencl:cpu",
        "UNSLOTH_ALLOW_CPU": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTHONIOENCODING": "utf-8",
    }
    root = str(Path(__file__).resolve().parents[2])
    env["PYTHONPATH"] = root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    result = subprocess.run(
        [sys.executable, str(Path(root) / "studio/backend/core/training/dataset_audit.py")],
        input = json.dumps(request),
        capture_output = True,
        text = True,
        encoding = "utf-8",
        env = env,
    )
    try:
        report = json.loads(result.stdout)
    except ValueError:
        raise RuntimeError("The dataset audit worker stopped without a report.") from None
    if result.returncode not in (0, 1, 2) or not isinstance(report, dict):
        raise RuntimeError("The dataset audit worker failed.")
    return report, result.returncode


def _print_report(report):
    coverage = report["coverage"]
    total = coverage["total_rows"]
    typer.echo(
        f"Rows checked: {coverage['rows_scanned']:,}"
        + (f" / {total:,}" if total is not None else " (total unknown)")
    )
    if not coverage["complete"]:
        typer.echo("Coverage: bounded prefix only; later rows have not been checked.")
    else:
        typer.echo("Coverage: complete for the selected dataset.")
    typer.echo(f"Format: {report['format']} | Packing: off")
    template = report["template"]
    typer.echo(
        f"Template: {template['source']} (SHA256: {template['sha256']}; --json includes its text)"
        if template["sha256"]
        else "Template: not applied to this format"
    )
    typer.echo(
        f"Prepared rows: {report['rows_prepared']:,} | Dropped by formatting: {report['rows_dropped_by_formatter']:,}"
    )
    lengths = report["lengths"]
    typer.echo(
        f"Token lengths: min {lengths['min']}, median {lengths['p50']}, p95 {lengths['p95']}, max {lengths['max']}"
    )
    typer.echo(f"Over context length: {report['rows_over_context']:,}")
    typer.echo(
        f"Rows with no loss targets: {report['rows_without_loss_after_truncation']:,} ({report['rows_losing_all_loss_to_truncation']:,} lost all targets to truncation)"
    )
    typer.echo(
        f"Causal loss targets retained: {report['loss_targets_after_truncation']:,} / {report['loss_targets_before_truncation']:,}"
    )
    for warning in report["warnings"]:
        typer.echo(f"Note: {warning}")


@dataset_app.command("check")
def check(
    source: Optional[str] = typer.Argument(
        None, help = "Local dataset path or Hugging Face dataset ID; overrides the config's dataset."
    ),
    model: Optional[str] = typer.Option(
        None, "--model", help = "Model ID or local tokenizer directory."
    ),
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", help = "The same YAML/JSON config used by unsloth train."
    ),
    max_rows: int = typer.Option(
        1000, "--max-rows", min = 1, help = "Check at most this many head rows (default: 1000)."
    ),
    full: bool = typer.Option(
        False,
        "--full",
        help = "Load a finite dataset and check every row; may download the full dataset.",
    ),
    json_output: bool = typer.Option(
        False, "--json", help = "Print the structured report, including the effective template."
    ),
    max_seq_length: Optional[int] = typer.Option(None, "--max-seq-length", min = 1),
    train_on_completions: Optional[bool] = typer.Option(
        None, "--train-on-completions/--no-train-on-completions"
    ),
    format_type: Optional[str] = typer.Option(None, "--format-type"),
    hf_token: Optional[str] = typer.Option(None, "--hf-token", envvar = "HF_TOKEN"),
):
    """Audit unpacked text SFT/CPT or pre-tokenized rows with the training formatter and masker.

    Exit 0: no dropped or zero-loss rows in the checked portion. Exit 1: such rows found.
    Exit 2: invalid input, unsupported configuration or audit failure. A bounded scan never
    certifies unseen rows. Malformed files fail at the reader instead of being silently skipped.
    """
    try:
        # CLI training selects MLX on Apple Silicon; these statistics use PyTorch/TRL loss semantics.
        if platform.system() == "Darwin" and platform.machine() == "arm64":
            raise ConfigError(
                "Dataset audit currently supports PyTorch training; MLX is not supported yet."
            )
        cfg = load_config(config)
        cfg.apply_overrides(
            model = model,
            max_seq_length = max_seq_length,
            train_on_completions = train_on_completions,
            format_type = format_type,
        )
        if source:
            path = Path(source).expanduser()
            if (
                path.exists()
                or path.is_absolute()
                or source.startswith((".", "~"))
                or path.suffix.lower() in {".json", ".jsonl", ".csv", ".parquet"}
            ):
                cfg.data.local_dataset = [str(path)]
                cfg.data.dataset = None
            else:
                cfg.data.dataset = source
                cfg.data.local_dataset = None
        if not cfg.model or not (cfg.data.dataset or cfg.data.local_dataset):
            raise ConfigError("Provide a model and dataset, either as arguments or in --config.")
        if cfg.training.packing:
            raise ConfigError(
                "Dataset audit currently supports packing=False; packed loss boundaries are not simulated."
            )
        request = {
            "full": full,
            "max_rows": max_rows,
            "config": {
                "model_name": cfg.model,
                "dataset_name": cfg.data.dataset,
                "local_datasets": cfg.data.local_dataset,
                "format_type": cfg.data.format_type,
                "max_seq_length": cfg.training.max_seq_length,
                "train_on_completions": cfg.training.train_on_completions,
                "packing": cfg.training.packing,
                "load_in_4bit": cfg.training.training_type == "lora" and cfg.training.load_in_4bit,
                "hf_token": hf_token or cfg.logging.hf_token,
            },
        }
        report, code = _run_audit(request)
    except KeyboardInterrupt:
        raise typer.Exit(code = 130)
    except (ConfigError, FileNotFoundError, RuntimeError, OSError) as exc:
        report, code = {"error": str(exc)}, 2
    if json_output:
        typer.echo(json.dumps(report, indent = 2, ensure_ascii = False))
    elif "error" in report:
        typer.echo(f"Error: {report['error']}", err = True)
    else:
        _print_report(report)
    raise typer.Exit(code = code)
