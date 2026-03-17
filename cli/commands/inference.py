# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import sys
from typing import Optional

import typer


def inference(
    model: str = typer.Argument(..., help = "HF model id or local path."),
    prompt: str = typer.Argument(..., help = "Prompt to send to the model."),
    hf_token: Optional[str] = typer.Option(
        None, "--hf-token", envvar = "HF_TOKEN", help = "Hugging Face token if needed."
    ),
    temperature: float = typer.Option(0.7, "--temperature"),
    top_p: float = typer.Option(0.9, "--top-p"),
    top_k: int = typer.Option(40, "--top-k"),
    max_new_tokens: int = typer.Option(256, "--max-new-tokens"),
    repetition_penalty: float = typer.Option(1.1, "--repetition-penalty"),
    system_prompt: str = typer.Option(
        "",
        "--system-prompt",
        help = "Optional system prompt to prepend.",
    ),
    max_seq_length: int = typer.Option(2048, "--max-seq-length"),
    load_in_4bit: bool = typer.Option(True, "--load-in-4bit/--no-load-in-4bit"),
):
    """Run a single inference using the specified model."""
    from studio.backend.core import ModelConfig, get_inference_backend

    inference_backend = get_inference_backend()
    model_config = ModelConfig.from_ui_selection(
        dropdown_value = model, search_value = None, hf_token = hf_token, is_lora = False
    )
    if not model_config:
        typer.echo("Could not resolve model config", err = True)
        raise typer.Exit(code = 1)

    if not inference_backend.load_model(
        config = model_config,
        max_seq_length = max_seq_length,
        load_in_4bit = load_in_4bit,
        hf_token = hf_token,
    ):
        typer.echo("Model load failed", err = True)
        raise typer.Exit(code = 1)

    messages = [{"role": "user", "content": prompt}]
    stream = inference_backend.generate_chat_response(
        messages = messages,
        system_prompt = system_prompt,
        temperature = temperature,
        top_p = top_p,
        top_k = top_k,
        max_new_tokens = max_new_tokens,
        repetition_penalty = repetition_penalty,
    )

    typer.echo("Assistant:", nl = True)
    previous = ""
    for chunk in stream:
        delta = chunk[len(previous) :]
        if delta:
            sys.stdout.write(delta)
            sys.stdout.flush()
        previous = chunk
    sys.stdout.write("\n")
    sys.stdout.flush()
