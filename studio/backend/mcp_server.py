# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unsloth Studio's MCP server at /mcp: curated tools that reach Unsloth Studio through its own routes as the agent's API key. Off until the owner turns it on, because several tools start GPU work or write files."""

from __future__ import annotations

from fastmcp import FastMCP
from fastmcp.tools import Tool

from studio_mcp import prompts
from studio_mcp.profile import AGENT, CHAT, CallerProfile
from studio_mcp.surface import render, render_schema
from studio_mcp.tool_access import StudioToolAccess
from studio_mcp.tools import (
    audio,
    cancel,
    data,
    export,
    images,
    jobs,
    library,
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
        "Load a model with load_model before chat, generate_image, generate_audio or generate_video; embed and transcribe fall back to Unsloth Studio's own embedding and speech-to-text models.",
        "Unsloth Studio may unload one model to fit another on the GPU, so load a model, generate, then unload_model before you switch kinds; speech-to-text models stay loaded beside them.",
        "load_model lists what it unloaded in evicted, so check studio_status before reusing a model you loaded earlier.",
        "generate_audio can speak, clone a voice, edit or convert speech, make music (a song or sfx) and separate stems, with a tts or audio model loaded.",
        "chat, generate_audio and transcribe wait for their result and report progress every 2 s.",
        "Video, recipe, export and training jobs return at once: poll get_job (video, recipe, export) or studio_status (training) every 15 to 30 s.",
        "Stop work with cancel: a chat by the cancel_id you passed when starting it, a video by its id.",
        "To fine-tune, find a dataset with datasets, check the config with start_training(validate_only=true), start it, then export_model the finished run.",
        "Errors start with a tag such as [gpu_busy retry_after=12] or [not_loaded kind=image]; the hint after the message says what to do.",
        "File paths in inputs work only when you run on the Unsloth Studio computer; otherwise send data or an Unsloth Studio id.",
        "Generated images, audio and videos are saved to the Unsloth Studio galleries and returned with their id and URL.",
    )
)


TOOL_MODULES = (
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
    library,
    cancel,
)


def tool_catalog() -> list[dict[str, str]]:
    """Every tool as ``{name, area}`` in registration order; the area is its module's short name."""
    return [
        {"name": tool.__name__, "area": module.__name__.rsplit(".", 1)[-1]}
        for module in TOOL_MODULES
        for tool, *_ in module.TOOLS
    ]


def tool_names() -> tuple[str, ...]:
    return tuple(entry["name"] for entry in tool_catalog())


def tool_facts() -> dict[str, frozenset[str]]:
    """What each tool declares about itself (studio_mcp.profile), for the profiles to choose by."""
    return {
        name: frozenset(facts)
        for module in TOOL_MODULES
        for name, facts in getattr(module, "FACTS", {}).items()
    }


def withheld(profile: CallerProfile = CHAT) -> dict[str, str]:
    """The tools ``profile`` is never offered, each with the reason."""
    return profile.withheld(tool_facts())


def surface(profile: CallerProfile) -> frozenset[str]:
    return profile.tools(tool_facts(), tool_names())


def create_studio_mcp(profile: CallerProfile = AGENT, exclude: frozenset = frozenset()) -> FastMCP:
    """The Unsloth MCP server as ``profile`` sees it: its tools minus ``exclude``, their text rendered for that surface."""
    mcp = FastMCP("Unsloth Studio", instructions = INSTRUCTIONS)
    mcp.add_middleware(StudioToolAccess())
    tools = surface(profile) - exclude

    for module in TOOL_MODULES:
        for tool, annotations, *schema in module.TOOLS:
            if tool.__name__ not in tools:
                continue
            built = Tool.from_function(
                tool, annotations = annotations, **({"output_schema": schema[0]} if schema else {})
            )
            mcp.add_tool(
                built.model_copy(
                    update = {
                        "description": render(built.description, tools),
                        "parameters": render_schema(built.parameters, tools),
                        "output_schema": render_schema(built.output_schema, tools),
                    }
                )
            )

    if profile.prompts:
        for prompt, title in prompts.PROMPTS:
            mcp.prompt(prompt, title = title)

    return mcp
