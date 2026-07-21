# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import sys
from typing import List, Optional

import typer
from rich.console import Console

from unsloth_cli._inference import (
    collect_stream,
    configure_quiet_logging,
    connect_studio_server,
    ensure_studio_backend_path,
    load_chat_backend,
    mlx_distributed_info,
    mlx_distributed_uses_mpi,
    quiet_if_nonzero_mlx_rank,
    raise_on_streamed_error,
    render_columns,
    resolve_model_config,
    stream_markdown,
    visible_text,
)

_HELP = (
    "Commands: /exit (quit), /reset (clear history), "
    "/think (toggle reasoning), /compare (base vs tuned), /help"
)


def _you_prompt(colors: bool) -> str:
    # The prompt must go through input(), not a separate print — readline
    # redraws erase anything they didn't draw, eating the label. GNU readline
    # wants colors wrapped in \001/\002; libedit (macOS) prints those
    # literally, so it gets raw ANSI.
    try:
        import readline
    except ImportError:
        return "\n\x1b[1;36mYou: \x1b[0m" if colors else "\nYou: "
    libedit = (
        "libedit" in (readline.__doc__ or "")
        or getattr(readline, "backend", "") == "editline"
    )
    if not colors:
        return "\nYou: "
    if libedit:
        return "\n\x1b[1;36mYou: \x1b[0m"
    return "\n\001\x1b[1;36m\002You: \001\x1b[0m\002"


def _compare_blocked_reason(model_config) -> Optional[str]:
    if model_config.is_gguf:
        return (
            "GGUF models can't toggle adapters — load a LoRA fine-tune "
            "(transformers backend) to compare base vs tuned."
        )
    if not model_config.is_lora:
        return (
            "this isn't a LoRA adapter — compare turns the adapter off for the "
            "'base' column, so there's nothing to compare against."
        )
    return None


def _get_base_load_in_4bit(model_config) -> bool:
    """Determine load_in_4bit for base model based on tuned adapter precision."""
    if not model_config.is_lora or not model_config.path:
        # Fallback to default if not a LoRA or no path
        return True

    try:
        import json
        from pathlib import Path

        adapter_cfg_path = Path(model_config.path) / "adapter_config.json"
        if not adapter_cfg_path.exists():
            return True

        with open(adapter_cfg_path, encoding = "utf-8") as f:
            adapter_cfg = json.load(f)

        training_method = adapter_cfg.get("unsloth_training_method")
        if training_method == "lora":
            return False
        elif training_method == "qlora":
            return True
        elif not training_method:
            # Fallback: check base model name for -bnb-4bit suffix
            if (
                model_config.base_model
                and "-bnb-4bit" not in model_config.base_model.lower()
            ):
                return False
            return True
        return True
    except Exception:
        return True


def _compare_needs_second_model() -> bool:
    # MLX can't toggle the adapter off, so compare loads the base separately.
    # detect_hardware() would print into the chat (and import torch), so
    # probe its MLX condition quietly: Apple Silicon with mlx installed.
    try:
        from studio.backend.utils.hardware import hardware as hw

        if hw.DEVICE is not None:
            return hw.DEVICE == hw.DeviceType.MLX
        if not hw.is_apple_silicon():
            return False
        import mlx.core  # noqa: F401

        return True
    except Exception:
        return False


def _drain_available_stdin() -> None:
    """Drain already-buffered launcher stdin on nonzero distributed ranks."""
    try:
        import os
        from select import select

        fd = sys.stdin.fileno()
        while select([fd], [], [], 0)[0]:
            if not os.read(fd, 8192):
                break
    except Exception:
        return


def _pick_trained_model(console) -> str:
    ensure_studio_backend_path()
    from utils.models import scan_trained_models

    trained = scan_trained_models()
    if not trained:
        typer.echo(
            "No trained models found in your outputs folder. "
            "Pass a model id or path: `unsloth chat <model>`.",
            err = True,
        )
        raise typer.Exit(code = 1)

    console.print("Your trained models (newest first):", style = "bold")
    for i, (display_name, _, model_type) in enumerate(trained, 1):
        console.print(f"  {i}. {display_name}  ({model_type})", markup = False)

    while True:
        try:
            raw = input(f"Chat with [1-{len(trained)}, Enter = 1]: ").strip()
        except (EOFError, KeyboardInterrupt):
            raise typer.Exit(code = 1)
        if not raw:
            return trained[0][1]
        if raw.isdigit() and 1 <= int(raw) <= len(trained):
            return trained[int(raw) - 1][1]
        console.print(f"Pick a number between 1 and {len(trained)}.", style = "yellow")


def chat(
    model: Optional[str] = typer.Argument(
        None, help = "HF model id or local path. Omit to pick one of your trained models."
    ),
    hf_token: Optional[str] = typer.Option(
        None, "--hf-token", envvar = "HF_TOKEN", help = "Hugging Face token if needed."
    ),
    temperature: float = typer.Option(0.7, "--temperature"),
    top_p: float = typer.Option(0.9, "--top-p"),
    top_k: int = typer.Option(40, "--top-k"),
    max_new_tokens: int = typer.Option(512, "--max-new-tokens"),
    repetition_penalty: float = typer.Option(1.1, "--repetition-penalty"),
    system_prompt: str = typer.Option(
        "", "--system-prompt", help = "Optional system prompt for the conversation."
    ),
    max_seq_length: int = typer.Option(4096, "--max-seq-length"),
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
        help = "Start with the model's <think> reasoning shown. Toggle live with /think.",
    ),
    compare: bool = typer.Option(
        False,
        "--compare/--no-compare",
        help = "Answer each prompt twice — base vs fine-tuned — side by side. "
        "Needs a LoRA adapter. Toggle live with /compare.",
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help = "Show backend and llama-server logs."
    ),
    no_server: bool = typer.Option(
        False,
        "--no-server",
        help = "Load the model in-process even if an Unsloth server is running.",
    ),
):
    """Start an interactive chat with a model (loads once, stays warm)."""
    if not verbose:
        configure_quiet_logging()

    console = Console()
    err = Console(stderr = True)
    is_mlx_distributed, rank, _world_size = mlx_distributed_info()
    should_print = rank == 0

    if is_mlx_distributed and mlx_distributed_uses_mpi():
        if should_print:
            err.print(
                "Distributed `unsloth chat` with MPI needs rank-0 prompt broadcast, "
                "which is not enabled yet. Use a non-MPI MLX launcher backend "
                "such as ring/JACCL for now.",
                style = "red",
                markup = False,
            )
        raise typer.Exit(code = 1)

    if model is None:
        if is_mlx_distributed:
            if should_print:
                err.print(
                    "Distributed `unsloth chat` requires an explicit model id or path.",
                    style = "red",
                    markup = False,
                )
            raise typer.Exit(code = 1)
        model = _pick_trained_model(console)

    # Resolve first so --compare can be rejected before the slow load.
    with quiet_if_nonzero_mlx_rank():
        model_config = resolve_model_config(model, hf_token = hf_token)
    compare_blocked = _compare_blocked_reason(model_config)
    if is_mlx_distributed:
        compare_blocked = (
            "distributed MLX chat does not support compare mode yet because it "
            "would need a second distributed worker group on the same ranks"
        )
    if compare and compare_blocked:
        if should_print:
            err.print(
                f"--compare unavailable: {compare_blocked}", style = "red", markup = False
            )
        raise typer.Exit(code = 1)

    load_opts = dict(
        hf_token = hf_token,
        max_seq_length = max_seq_length,
        load_in_4bit = load_in_4bit,
        tensor_parallel = tensor_parallel,
        llama_extra_args = llama_extra_args,
    )

    # Prefer a running Unsloth server: instant starts, model shared with the UI.
    chat_backend = (
        None
        if (no_server or is_mlx_distributed)
        else connect_studio_server(model, **load_opts)
    )
    server_mode = chat_backend is not None
    if server_mode and should_print:
        console.print(
            "(Unsloth server connected — model stays warm after /exit)",
            style = "bright_black",
        )
    else:
        chat_backend = load_chat_backend(model, model_config = model_config, **load_opts)

    name = model_config.display_name or model
    show_thinking = think
    compare_mode = compare
    messages = []

    # Compare's base column: server mode keeps the tuned model remote and
    # loads the base locally; local MLX (no adapter toggle) does the same;
    # local CUDA just toggles the adapter on the one loaded model.
    dual_compare = compare_blocked is None and (
        server_mode or _compare_needs_second_model()
    )
    base_backend = None

    def load_base_for_compare():
        nonlocal base_backend
        if base_backend is not None:
            return True
        base_id = model_config.base_model
        if not base_id:
            if should_print:
                console.print(
                    "(compare unavailable: this adapter doesn't record its base model)",
                    style = "yellow",
                )
            return False
        if should_print:
            console.print(
                f"(loading base model {base_id} for compare — keeps two models in memory)",
                style = "bright_black",
                markup = False,
            )
        try:
            # Use the same precision as the tuned model for fair comparison
            base_load_opts = dict(load_opts)  # Copy original options
            base_load_opts["load_in_4bit"] = _get_base_load_in_4bit(model_config)
            base_backend = load_chat_backend(
                base_id, fresh_backend = True, **base_load_opts
            )
        except Exception as exc:
            if should_print:
                err.print(f"(base model load failed: {exc})", style = "red", markup = False)
            return False
        return True

    if compare and dual_compare and not load_base_for_compare():
        raise typer.Exit(code = 1)

    def generate(backend = None, use_adapter = None):
        # Reads messages and show_thinking live, so /reset and /think apply.
        stream = (backend or chat_backend).stream(
            messages,
            system_prompt = system_prompt,
            temperature = temperature,
            top_p = top_p,
            top_k = top_k,
            max_new_tokens = max_new_tokens,
            repetition_penalty = repetition_penalty,
            enable_thinking = show_thinking,
            use_adapter = use_adapter,
        )
        return raise_on_streamed_error(stream) if is_mlx_distributed else stream

    if should_print:
        console.print()
        console.print(f"Chatting with {name}", style = "bold green", markup = False)
        console.print(_HELP, style = "bright_black")

    # legacy_windows: pre-VT consoles print raw ANSI as ←[1;36m garbage.
    you_prompt = (
        _you_prompt(console.is_terminal and not console.legacy_windows)
        if should_print
        else ""
    )
    assistant_label = "[bold magenta]Assistant:[/bold magenta]"

    try:
        while True:
            if should_print:
                try:
                    user = input(you_prompt).strip()
                except (EOFError, KeyboardInterrupt):
                    if should_print:
                        console.print()
                    user = "/exit"
                turn = {"type": "turn", "text": user}
            else:
                turn = None

            if is_mlx_distributed:
                try:
                    turn = chat_backend.share_distributed_object(turn, timeout = None)
                    if not should_print:
                        _drain_available_stdin()
                except Exception as exc:
                    if should_print:
                        err.print(
                            f"\n(error sharing chat turn: {exc})",
                            style = "red",
                            markup = False,
                        )
                    raise typer.Exit(code = 1)
            if not turn:
                continue
            user = str(turn.get("text", "")).strip()

            if not user:
                continue
            if user in ("/exit", "/quit"):
                break
            if user == "/reset":
                messages = []
                if should_print:
                    console.print("(history cleared)", style = "bright_black")
                continue
            if user == "/think":
                show_thinking = not show_thinking
                if should_print:
                    state = "on" if show_thinking else "off"
                    console.print(f"(thinking {state})", style = "bright_black")
                continue
            if user == "/compare":
                if compare_blocked:
                    if should_print:
                        console.print(
                            f"(compare unavailable: {compare_blocked})", style = "yellow"
                        )
                    continue
                if not compare_mode and dual_compare and not load_base_for_compare():
                    continue
                compare_mode = not compare_mode
                if should_print:
                    state = "on" if compare_mode else "off"
                    console.print(f"(compare {state})", style = "bright_black")
                continue
            if user in ("/help", "/?"):
                if should_print:
                    console.print(_HELP, style = "bright_black")
                continue

            messages.append({"role": "user", "content": user})

            try:
                if compare_mode:
                    if should_print:
                        console.print(
                            "(comparing base vs tuned…)", style = "bright_black"
                        )
                    if dual_compare:
                        base_text = collect_stream(
                            generate(backend = base_backend), show_thinking
                        )
                        tuned_text = collect_stream(generate(), show_thinking)
                    else:
                        base_text = collect_stream(
                            generate(use_adapter = False), show_thinking
                        )
                        tuned_text = collect_stream(
                            generate(use_adapter = True), show_thinking
                        )
                    if should_print:
                        console.print()
                        render_columns(
                            "base",
                            base_text,
                            f"{name} (tuned)",
                            tuned_text,
                            console = console,
                        )
                    # History continues as the tuned model; base is just the reference.
                    answer = tuned_text
                else:
                    if should_print:
                        console.print(assistant_label)
                        answer = stream_markdown(
                            generate(), show_thinking, console = console
                        )
                    else:
                        answer = collect_stream(generate(), show_thinking)
            except KeyboardInterrupt:
                # Ctrl-C aborts this answer only; drop the unanswered turn.
                if should_print:
                    console.print("\n(interrupted)", style = "bright_black")
                messages.pop()
                continue
            except Exception as exc:
                if should_print:
                    err.print(f"\n(error: {exc})", style = "red", markup = False)
                messages.pop()
                if is_mlx_distributed:
                    raise typer.Exit(code = 1)
                continue

            messages.append(
                {
                    "role": "assistant",
                    "content": visible_text(answer, show_thinking = False),
                }
            )
    finally:
        chat_backend.close()
        if base_backend is not None:
            base_backend.close()
        if should_print:
            err.print("\nBye.", style = "bright_black")
