# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unsloth Studio's MCP server at /mcp: curated tools that reach Unsloth Studio through its own routes as the agent's API key. Off until the owner turns it on, because several tools start GPU work or write files."""

from __future__ import annotations

from fastmcp import FastMCP

from studio_mcp.tools import (
    audio,
    cancel,
    data,
    export,
    images,
    jobs,
    models,
    status,
    text,
    training,
    video,
)

INSTRUCTIONS = (
    "Unsloth Studio runs models, generates media, trains and exports. "
    "Call studio_status first to see what is loaded and running, and list_models to find models. "
    "Load a model with load_model before chat, generate_image, generate_audio or generate_video; embed "
    "and transcribe fall back to Unsloth Studio's own embedding and speech-to-text models. "
    "Unsloth Studio may unload one model to fit another on the GPU (load_model lists what it unloaded), "
    "so check studio_status before reusing a model you loaded earlier. "
    "Long work returns at once: start it, then poll get_job (video, recipe, export) or studio_status "
    "(training), and stop it with cancel. "
    "File paths in inputs work only when you run on the Unsloth Studio computer; otherwise send data or an "
    "Unsloth Studio id. "
    "Generated images, audio and videos are saved to the Unsloth Studio galleries and returned with their id "
    "and URL. "
    "Training, generation and exports use the GPU and can take minutes."
)


def create_studio_mcp() -> FastMCP:
    """Create the Unsloth MCP server and register the high-value tools."""
    mcp = FastMCP("Unsloth Studio", instructions = INSTRUCTIONS)

    for module in (
        status,
        models,
        text,
        images,
        audio,
        video,
        jobs,
        data,
        training,
        export,
        cancel,
    ):
        for tool, annotations, *schema in module.TOOLS:
            mcp.tool(
                tool, annotations = annotations, **({"output_schema": schema[0]} if schema else {})
            )

    return mcp
