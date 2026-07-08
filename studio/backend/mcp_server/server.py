# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""FastMCP server factory and stdio entry point for the Studio MCP server."""

from __future__ import annotations

from typing import Optional, Sequence

from fastmcp import FastMCP

from loggers import get_logger
from .tools import FEATURE_GROUPS, register_all, resolve_groups

logger = get_logger(__name__)

# Re-exported so ``from mcp_server import FEATURE_GROUPS`` works.
__all__ = ["FEATURE_GROUPS", "build_server", "run_stdio", "list_tools"]

_INSTRUCTIONS = (
    "Unsloth Studio: fine-tune, export and synthesize datasets for open LLMs. "
    "Use models_list / models_get_config to pick a model and its recommended "
    "training defaults, data_list_local / data_check_format to prepare a "
    "dataset, train_start (then train_status / train_stop) to fine-tune, "
    "export_* to save a checkpoint to disk (merged / GGUF / LoRA), and "
    "recipe_* to generate synthetic datasets with Data Designer. "
    "Training and recipe jobs are asynchronous: start them, then poll the "
    "matching *_status tool."
)


def build_server(
    enabled: Optional[Sequence[str]] = None, disabled: Optional[Sequence[str]] = None
) -> FastMCP:
    """Build a FastMCP server exposing the selected Studio feature groups.

    ``enabled`` defaults to all groups; ``disabled`` removes from that set.
    Unknown names raise ``ValueError``.
    """
    selected = resolve_groups(enabled, disabled)
    mcp = FastMCP(name = "unsloth-studio", instructions = _INSTRUCTIONS)
    names = register_all(mcp, enabled = selected)
    logger.info(
        "Studio MCP server built: %d tools across groups [%s]",
        len(names),
        ", ".join(selected),
    )
    return mcp


def run_stdio(
    enabled: Optional[Sequence[str]] = None, disabled: Optional[Sequence[str]] = None
) -> None:
    """Run the Studio MCP server over stdio (blocks until the client disconnects)."""
    server = build_server(enabled = enabled, disabled = disabled)
    server.run(transport = "stdio")


async def list_tools(
    enabled: Optional[Sequence[str]] = None, disabled: Optional[Sequence[str]] = None
) -> list[dict[str, str]]:
    """Return ``[{name, description, group}]`` for each registered tool.

    Powers the ``unsloth studio mcp list-tools`` dry-run. Uses an in-memory
    FastMCP client so the reported set always reflects real registration.
    """
    from fastmcp import Client

    server = build_server(enabled = enabled, disabled = disabled)
    from .tools import _REGISTRARS  # intentional internal use (same package)

    group_for_name: dict[str, str] = {}
    for group in resolve_groups(enabled, disabled):
        # Register this group into a throwaway server solely to learn which
        # tool names it owns, then map each back to the group.
        probe = FastMCP(name = "probe")
        for name in _REGISTRARS[group](probe):
            group_for_name[name] = group

    async with Client(server) as client:
        tools = await client.list_tools()
    return [
        {
            "name": t.name,
            "description": next(iter((t.description or "").strip().splitlines()), ""),
            "group": group_for_name.get(t.name, "?"),
        }
        for t in tools
    ]
