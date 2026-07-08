# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unsloth Studio as a standalone MCP server.

Exposes Studio's core features -- models, datasets, training, export and
data-recipe (Data Designer) -- as MCP tools so any MCP client (Claude Desktop,
Cursor, Cline, ...) can drive Unsloth Studio directly.

The server reuses Studio's in-process core layer (the same singletons the
FastAPI routes use), so it speaks the exact same backend a Studio UI session
would. Long-running work (training, export, recipe jobs) runs in the existing
subprocess backends and is surfaced as start / status / stop tools.

Launch via the CLI::

    unsloth studio mcp                       # stdio, all features
    unsloth studio mcp --enable models data  # read-only subset

See ``studio/backend/mcp_server/server.py`` for the entry points and
``tools/`` for the per-feature tool implementations.
"""

from __future__ import annotations

from .server import FEATURE_GROUPS, build_server, run_stdio

__all__ = ["FEATURE_GROUPS", "build_server", "run_stdio"]
