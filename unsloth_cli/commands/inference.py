# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from typing import List, Optional

import typer

from unsloth_cli._inference import (
    collect_stream,
    configure_quiet_logging,
    connect_studio_server,
    load_chat_backend,
    mlx_distributed_info,
    mlx_distributed_uses_mpi,
    raise_on_streamed_error,
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
            "of by layer. Under non-MPI mlx.launch, select MLX tensor "
            "parallel mode instead of pipeline mode."
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

    is_mlx_distributed, rank, _world_size = mlx_distributed_info()
    if is_mlx_distributed and mlx_distributed_uses_mpi():
        if rank == 0:
            typer.echo(
                "Distributed `unsloth inference` with MPI is not supported by "
                "the current subprocess backend. Use a non-MPI MLX launcher "
                "backend such as ring/JACCL for now.",
                err = True,
            )
        raise typer.Exit(code = 1)

    # A running Studio server keeps the model warm between runs. Under
    # mlx.launch, every rank must enter the local MLX path instead of rank 0
    # alone talking to a server.
    load_opts = dict(
        hf_token = hf_token,
        max_seq_length = max_seq_length,
        load_in_4bit = load_in_4bit,
        tensor_parallel = tensor_parallel,
        llama_extra_args = llama_extra_args,
    )
    chat_backend = (
        None if (no_server or is_mlx_distributed) else connect_studio_server(model, **load_opts)
    )
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
        if is_mlx_distributed:
            stream = raise_on_streamed_error(stream)
        if rank == 0:
            typer.echo("Assistant:")
            try:
                stream_to_stdout(stream, show_thinking = think)
            except RuntimeError as exc:
                if not is_mlx_distributed:
                    raise
                typer.echo(f"Error: {exc}", err = True)
                raise typer.Exit(code = 1)
        else:
            try:
                collect_stream(stream, show_thinking = think)
            except RuntimeError:
                if not is_mlx_distributed:
                    raise
                raise typer.Exit(code = 1)
    finally:
        chat_backend.close()
