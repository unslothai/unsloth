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

INSTRUCTIONS = " ".join(
    (
        "Unsloth Studio runs local AI models for chat, embed, images, audio, transcribe, video, datasets, fine-tune and export.",
        "Call studio_status first to see what is loaded and running, and list_models to find models.",
        # Load line: later PRs add load placement options after it.
        "Load a model with load_model before chat, generate_image, generate_audio or generate_video; embed and transcribe fall back to Unsloth Studio's own embedding and speech-to-text models.",
        # One-GPU rule line: later PRs add the multi-model modes after it.
        "Unsloth Studio may unload one model to fit another on the GPU, so load a model, generate, then unload_model before you switch kinds; speech-to-text models stay loaded beside them.",
        "load_model lists what it unloaded in evicted, so check studio_status before reusing a model you loaded earlier.",
        # Audio line: later PRs add audio analysis after it.
        "generate_audio can speak, clone a voice, edit or convert speech, make music (a song or sfx) and separate stems, with a tts or audio model loaded.",
        "chat, generate_audio and transcribe wait for their result and report progress every 2 s.",
        # Polling line: later PRs add waiting on long jobs after it.
        "Video, recipe, export and training jobs return at once: poll get_job (video, recipe, export) or studio_status (training) every 15 to 30 s.",
        "Stop work with cancel: a chat by the cancel_id you passed when starting it, a video by its id.",
        # Training line: later PRs add model prefetch after it.
        "To fine-tune, find a dataset with datasets, check the config with start_training(validate_only=true), start it, then export_model the finished run.",
        "Errors start with a tag such as [gpu_busy retry_after=12] or [not_loaded kind=image]; the hint after the message says what to do.",
        "File paths in inputs work only when you run on the Unsloth Studio computer; otherwise send data or an Unsloth Studio id.",
        # Galleries line: later PRs add download links after it.
        "Generated images, audio and videos are saved to the Unsloth Studio galleries and returned with their id and URL.",
    )
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
