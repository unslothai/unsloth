# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
import sys
import types

import pytest

from mcp_server import _clamp, _dump, create_studio_mcp


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
        "stop_training",
        "list_training_runs",
        "export_model",
    }


def test_dump_serializes_pydantic_values():
    class Response:
        def model_dump(self, *, mode):
            assert mode == "json"
            return {"ok": True}

    assert _dump(Response()) == {"ok": True}
    assert _dump({"already": "json"}) == {"already": "json"}


def test_clamp_restricts_to_inclusive_bounds():
    assert _clamp(5, 1, 200) == 5
    assert _clamp(-10, 1, 200) == 1
    assert _clamp(10_000, 1, 200) == 200
    assert _clamp(0, 1, 500) == 1
    assert _clamp(1_000, 1, 500) == 500


def test_export_and_checkpoint_tools_expose_forwarded_fields():
    stop_schema = _get_tool("stop_training").parameters
    assert "expected_job_id" in stop_schema["required"]


def _stub_module(monkeypatch, name, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    if "." in name:
        module.__path__ = []  # mark package-like so submodule imports resolve
    monkeypatch.setitem(sys.modules, name, module)
    return module


def test_stop_training_forwards_job_scope(monkeypatch):
    captured = {}

    class FakeTrainingStopRequest:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    async def fake_stop(request, current_subject):
        assert isinstance(request, FakeTrainingStopRequest)
        captured["current_subject"] = current_subject
        return {"status": "stopped"}

    _stub_module(monkeypatch, "routes")
    _stub_module(
        monkeypatch,
        "routes.training",
        TrainingStopRequest = FakeTrainingStopRequest,
        stop_training = fake_stop,
    )

    tool = _get_tool("stop_training")
    result = asyncio.run(tool.fn(expected_job_id = "job-A", save = False))

    assert captured["expected_job_id"] == "job-A"
    assert captured["save"] is False
    assert captured["current_subject"] == "mcp"
    assert result == {"status": "stopped"}
