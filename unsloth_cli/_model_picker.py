# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from pathlib import Path
from typing import Callable, List, Optional

import typer

from unsloth_cli._model_catalog import ModelEntry


def _is_gguf_entry(entry) -> bool:
    if entry.detail.lower() == "gguf":
        return True
    # A training directory can be named *-GGUF or *.gguf without containing GGUF weights.
    try:
        if Path(entry.model).is_dir():
            return False
    except OSError:
        return False
    model = entry.model.lower().rstrip("/")
    if model.endswith(".gguf"):
        return True
    if not model.endswith("-gguf"):
        return False
    from huggingface_hub.utils import HFValidationError, validate_repo_id

    try:
        validate_repo_id(model)
    except HFValidationError:
        return False
    return True


def quiet_inventory_logs() -> None:
    # LOG_LEVEL is read, not exported: a server `unsloth start` spawns keeps its own logging.
    import logging
    import os

    try:
        import structlog
    except ModuleNotFoundError:
        return
    level = getattr(logging, os.environ.get("LOG_LEVEL", "WARNING").upper(), logging.WARNING)
    structlog.configure(wrapper_class = structlog.make_filtering_bound_logger(level))


def pick_model_entry(
    console,
    *,
    prompt: str = "Chat with",
    empty_hint: str = "No local models found. Pass a model id or path: `unsloth chat <model>`.",
    keep: Optional[Callable[[object], bool]] = None,
    entries: Optional[List] = None,
) -> ModelEntry:
    from unsloth_cli._model_catalog import list_chat_models

    if entries is None:
        entries = list_chat_models()
    if keep is not None:
        entries = [entry for entry in entries if keep(entry)]
    if not entries:
        typer.echo(empty_hint, err = True)
        raise typer.Exit(code = 1)

    console.print("Your models", style = "bold")
    width = max(len(e.name) for e in entries)
    group = None
    for i, entry in enumerate(entries, 1):
        if entry.group != group:
            group = entry.group
            console.print(f"\n  {group}", style = "bright_black")
        line = f"  {i:>2}. {entry.name:<{width}}   {entry.detail}".rstrip()
        console.print(line, markup = False, highlight = False, soft_wrap = True)
    console.print()

    while True:
        try:
            raw = input(f"{prompt} [1-{len(entries)}, Enter = 1]: ").strip()
        except (EOFError, KeyboardInterrupt):
            raise typer.Exit(code = 1)
        if not raw:
            return entries[0]
        if raw.isdecimal():
            try:
                index = int(raw)
            except ValueError:
                pass  # Python limits the length of integer strings.
            else:
                if 1 <= index <= len(entries):
                    return entries[index - 1]
        console.print(f"Pick a number between 1 and {len(entries)}.", style = "yellow")
