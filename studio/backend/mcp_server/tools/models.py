# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""MCP tools for Studio's model catalog and checkpoints (read-only)."""

from __future__ import annotations

from typing import Any, Optional

from fastmcp import FastMCP

from loggers import get_logger
from mcp_server.auth import resolve_hf_token

logger = get_logger(__name__)

GROUP = "models"

# Structured hints surfaced to MCP clients so they can flag read-only calls as
# safe-to-auto-approve. See MCP ``ToolAnnotations``.
_READ_ONLY = {"readOnlyHint": True}


def models_list() -> dict[str, Any]:
    """List every model in Studio's curated catalog.

    Returns the curated model registry (the same set the Studio UI's model
    picker shows), each with its HuggingFace aliases and the YAML config file
    that backs it. Use ``models_get_config`` to retrieve the recommended
    training defaults for a specific model.
    """
    from utils.models.model_config import MODEL_NAME_MAPPING

    entries: list[dict[str, Any]] = []
    for config_file, aliases in MODEL_NAME_MAPPING.items():
        if not aliases:
            continue
        primary = aliases[0]
        entries.append(
            {
                "id": primary,
                "name": primary.split("/")[-1] if "/" in primary else primary,
                "aliases": list(aliases),
                "config_file": config_file,
            }
        )
    entries.sort(key = lambda e: e["id"].lower())
    return {"count": len(entries), "models": entries}


def models_get_config(model_name: str) -> dict[str, Any]:
    """Get the recommended training defaults for a model.

    Loads the per-model YAML defaults (LoRA rank, learning rate, max sequence
    length, trust_remote_code, etc.) that Studio applies when that model is
    selected. Returns an empty ``config`` object for unknown models (the
    trainer then falls back to sensible built-in defaults).
    """
    from utils.models.model_config import load_model_defaults

    config = load_model_defaults(model_name)
    return {"model_name": model_name, "config": config}


async def models_list_cached(hf_token: Optional[str] = None) -> dict[str, Any]:
    """List models already downloaded to the local HuggingFace cache.

    These can be used for training/inference without an additional download.
    Set ``hf_token`` (or the ``HF_TOKEN`` env var) to see gated repos.
    """
    from hub.services.models.cache_inventory import list_cached_models_response

    token = resolve_hf_token(hf_token)
    result = await list_cached_models_response(hf_token = token)
    if isinstance(result, dict):
        return result
    return {"cached": result}


def models_checkpoints(outputs_dir: Optional[str] = None) -> dict[str, Any]:
    """Scan a training-output directory for checkpoints.

    Returns discovered runs and their checkpoints (step path and best loss),
    matching what the Studio UI and ``unsloth list-checkpoints`` show. With no
    ``outputs_dir``, Studio's default outputs root is scanned.
    """
    from utils.models.checkpoints import scan_checkpoints
    from utils.paths import outputs_root

    root = outputs_dir if outputs_dir else str(outputs_root())
    scanned = scan_checkpoints(outputs_dir = root)

    runs: list[dict[str, Any]] = []
    for model_name, checkpoints, metadata in scanned:
        runs.append(
            {
                "model_name": model_name,
                "checkpoints": [
                    {"name": name, "path": path, "loss": loss} for name, path, loss in checkpoints
                ],
                "metadata": metadata,
            }
        )
    return {"outputs_dir": root, "runs": runs}


def register(mcp: FastMCP) -> list[str]:
    """Register the models tools onto ``mcp``; return the tool names added."""
    names: list[str] = []
    for fn in (models_list, models_get_config, models_list_cached, models_checkpoints):
        mcp.tool(fn, annotations = _READ_ONLY)
        names.append(fn.__name__)
    return names
