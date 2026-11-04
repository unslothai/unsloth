# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Numbered picker over the models on this device, shared by `unsloth chat` and `unsloth start`."""

from typing import Callable, List, Optional

import typer


def _is_gguf_entry(entry) -> bool:
    # Downloaded GGUFs resolve to a .gguf file (or keep a *-GGUF repo id when that fails); GGUF
    # exports and local folders carry exactly "gguf" as detail. Not a substring match: a path or
    # dataset title containing "gguf" would list a model the agent then loads and refuses.
    model = entry.model.lower().rstrip("/")
    return model.endswith((".gguf", "-gguf")) or entry.detail.lower() == "gguf"


def quiet_inventory_logs() -> None:
    """Hide the inventory scan's info lines ("Found 0 trained models in ...") above the list.

    Unlike configure_quiet_logging, LOG_LEVEL is read, not exported: `unsloth start` may go on
    to spawn a server, which must keep its own logging.
    """
    import logging
    import os

    try:
        import structlog
    except ModuleNotFoundError:
        return
    level = getattr(logging, os.environ.get("LOG_LEVEL", "WARNING").upper(), logging.WARNING)
    structlog.configure(wrapper_class = structlog.make_filtering_bound_logger(level))


def pick_local_model(
    console,
    *,
    prompt: str = "Chat with",
    empty_hint: str = "No local models found. Pass a model id or path: `unsloth chat <model>`.",
    keep: Optional[Callable[[object], bool]] = None,
) -> str:
    """Print the device's chat models with numbers and return the chosen one's load target.

    Callers put the Studio backend on sys.path first (ensure_studio_backend_path): the catalog
    reads the Studio inventory layer, imported here so `unsloth --help` never pays for it.
    """
    from unsloth_cli._model_catalog import list_chat_models

    entries: List = list_chat_models()
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
            return entries[0].model
        if raw.isdigit() and 1 <= int(raw) <= len(entries):
            return entries[int(raw) - 1].model
        console.print(f"Pick a number between 1 and {len(entries)}.", style = "yellow")


def pick_gguf_model(console, *, prompt: str, empty_hint: str) -> str:
    return pick_local_model(console, prompt = prompt, empty_hint = empty_hint, keep = _is_gguf_entry)
