# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from typing import List, Optional

import typer

from unsloth_cli._inference import (
    configure_quiet_logging,
    connect_studio_server,
    load_chat_backend,
    stream_to_stdout,
)


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
    tensor_parallel: bool = typer.Option(
        False,
        "--tensor-parallel/--no-tensor-parallel",
        help = (
            "Split a GGUF across GPUs by tensor (--split-mode tensor) instead "
            "of by layer. Ignored for non-GGUF models."
        ),
    ),
    llama_extra_args: Optional[List[str]] = typer.Option(
        None,
        "--llama-extra-arg",
        help = (
            "Extra llama-server arg for GGUF models. Repeat for multiple "
            "tokens, e.g. --llama-extra-arg=--top-k --llama-extra-arg 20."
        ),
    ),
    think: bool = typer.Option(
        False,
        "--think/--no-think",
        help = "Show the model's <think> reasoning. Off by default so reasoning "
        "models answer directly instead of spending the token budget thinking.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help = "Show backend and llama-server logs (otherwise only the answer).",
    ),
    no_server: bool = typer.Option(
        False,
        "--no-server",
        help = "Load the model in-process even if a Studio server is running.",
    ),
):
    """Run a single inference using the specified model."""
    if not verbose:
        configure_quiet_logging()

    # A running Studio server keeps the model warm between runs, which is
    # exactly what a one-shot command wants.
    load_opts = dict(
        hf_token = hf_token,
        max_seq_length = max_seq_length,
        load_in_4bit = load_in_4bit,
        tensor_parallel = tensor_parallel,
        llama_extra_args = llama_extra_args,
    )
    chat_backend = None if no_server else connect_studio_server(model, **load_opts)
    if chat_backend is None:
        chat_backend = load_chat_backend(model, **load_opts)
    try:
        stream = chat_backend.stream(
            [{"role": "user", "content": prompt}],
            system_prompt = system_prompt,
            temperature = temperature,
            top_p = top_p,
            top_k = top_k,
            max_new_tokens = max_new_tokens,
            repetition_penalty = repetition_penalty,
            enable_thinking = think,
        )
        typer.echo("Assistant:")
        stream_to_stdout(stream, show_thinking = think)
    finally:
        chat_backend.close()
