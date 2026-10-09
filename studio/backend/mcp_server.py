# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Curated MCP tools for driving an Unsloth Studio instance. The MCP surface deliberately wraps the existing
Unsloth services instead of duplicating training or export logic. It is opt-in because several tools can start
GPU work or write model artifacts.
"""

from __future__ import annotations

import asyncio
from typing import Any

from fastmcp import FastMCP

from studio_mcp.tools.audio import register_audio
from studio_mcp.tools.cancel import register_cancel
from studio_mcp.tools.data import register_data
from studio_mcp.tools.export import register_export
from studio_mcp.tools.images import register_images
from studio_mcp.tools.jobs import register_jobs
from studio_mcp.tools.models import register_models
from studio_mcp.tools.status import register_status
from studio_mcp.tools.text import register_text
from studio_mcp.tools.training import register_training
from studio_mcp.tools.video import register_video


def create_studio_mcp() -> FastMCP:
    """Create the Unsloth MCP server and register the high-value tools."""
    mcp = FastMCP(
        "Unsloth Studio",
        instructions = (
            "Use read tools to inspect the local Unsloth state before starting GPU work. "
            "Training and export tools can consume substantial VRAM and write files. "
            "Never expose tokens or local paths from tool results unless the user asks."
        ),
    )

    register_status(mcp)

    register_models(mcp)
    register_text(mcp)
    register_images(mcp)
    register_audio(mcp)
    register_video(mcp)
    register_jobs(mcp)
    register_data(mcp)
    register_training(mcp)
    register_export(mcp)
    register_cancel(mcp)

    return mcp
