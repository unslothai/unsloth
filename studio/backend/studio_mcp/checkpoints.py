# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Checkpoints by name. The checkpoints route lists each training output with its host path, and the export routes want that path back; agents only ever see ``<run folder>`` (the final weights) or ``<run folder>/<checkpoint>``, and the path is looked up here and never leaves this module."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from fastmcp.exceptions import ToolError

from studio_mcp.caller import Caller
from studio_mcp.tools import integer, number, opt_text, route_json


@dataclass(frozen = True)
class Checkpoint:
    run: str
    name: str
    path: str = field(repr = False)
    loss: Optional[float] = None
    base_model: Optional[str] = None
    peft_type: Optional[str] = None
    lora_rank: Optional[int] = None
    is_quantized: bool = False


def _leaf(path: str) -> str:
    return re.split(r"[\\/]", path.rstrip("\\/"))[-1]


def run_folder_path(checkpoints: list[dict], folder: str) -> Optional[str]:
    """The run folder's own path: its final checkpoint, else the parent of an intermediate one."""
    paths = [
        c.get("path") for c in checkpoints if isinstance(c, dict) and isinstance(c.get("path"), str)
    ]
    for path in paths:
        if _leaf(path) == folder:
            return path
    for path in paths:
        parent = re.split(r"[\\/]", path.rstrip("\\/"))
        if len(parent) > 1:
            return path[: len(path.rstrip("\\/")) - len(parent[-1])].rstrip("\\/")
    return None


async def list_checkpoints(caller: Caller) -> tuple[list[Checkpoint], dict[str, str]]:
    """Every checkpoint by name, plus each run folder's path for matching training runs."""
    listing = await route_json("GET", "/api/models/checkpoints", caller = caller)
    models = listing.get("models") if isinstance(listing, dict) else None
    found: list[Checkpoint] = []
    folders: dict[str, str] = {}
    for model in models or []:
        if not isinstance(model, dict) or not opt_text(model.get("name")):
            continue
        folder = model["name"]
        entries = [c for c in model.get("checkpoints") or [] if isinstance(c, dict)]
        folder_path = run_folder_path(entries, folder)
        if folder_path:
            folders[folder] = folder_path
        for entry in entries:
            path, label = opt_text(entry.get("path")), opt_text(entry.get("display_name"))
            if path is None or label is None:
                continue
            found.append(
                Checkpoint(
                    run = folder,
                    name = folder
                    if label == folder or _leaf(path) == folder
                    else f"{folder}/{label}",
                    path = path,
                    loss = number(entry.get("loss")),
                    base_model = opt_text(model.get("base_model")),
                    peft_type = opt_text(model.get("peft_type")),
                    lora_rank = integer(model.get("lora_rank")),
                    is_quantized = model.get("is_quantized") is True,
                )
            )
    return found, folders


async def resolve(caller: Caller, name: str) -> Checkpoint:
    checkpoints, _folders = await list_checkpoints(caller)
    for checkpoint in checkpoints:
        if checkpoint.name == name:
            return checkpoint
    available = ", ".join(sorted(c.name for c in checkpoints)[:20]) or "none"
    raise ToolError(f"No checkpoint named {name}. Available: {available}")
