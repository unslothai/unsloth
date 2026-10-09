# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
import sys
import types

import pytest

from mcp_server import create_studio_mcp


def _get_tool(name):
    tools = asyncio.run(create_studio_mcp().list_tools())
    return {tool.name: tool for tool in tools}[name]


def test_studio_mcp_registers_control_plane_tools():
    tools = asyncio.run(create_studio_mcp().list_tools())

    assert {tool.name for tool in tools} == {
        "studio_status",
        "list_models",
        "chat",
        "embed",
        "system_one",
        "generate_image",
        "generate_audio",
        "transcribe",
        "generate_video",
        "get_job",
        "run_recipe",
        "datasets",
        "load_model",
        "unload_model",
        "start_training",
        "cancel",
        "list_training_runs",
        "export_model",
    }
