# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Studio's MCP server at /mcp: curated tools that reach Studio through its own routes as the agent's API key. Off until the owner turns it on, because several tools start GPU work or write files."""

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


INSTRUCTIONS = (
    "Unsloth Studio runs models, generates media, trains and exports. "
    "Call studio_status first to see what is loaded and running, and list_models to find models. "
    "Load a model with load_model before chat, embed, generate_image, generate_audio, transcribe or "
    "generate_video. "
    "Long work returns at once: start it, then poll get_job (video, recipe, export) or studio_status "
    "(training), and stop it with cancel. "
    "File paths in inputs work only when you run on the Studio computer; otherwise send data or a "
    "Studio id. "
    "Generated images, audio and videos are saved to the Studio galleries and returned with their id "
    "and URL. "
    "Training, generation and exports use the GPU and can take minutes."
)


def create_studio_mcp() -> FastMCP:
    """Create the Unsloth MCP server and register the high-value tools."""
    mcp = FastMCP("Unsloth Studio", instructions = INSTRUCTIONS)

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
