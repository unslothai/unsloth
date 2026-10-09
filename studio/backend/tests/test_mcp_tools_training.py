# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
import json

from fastapi.testclient import TestClient

from mcp_server import create_studio_mcp

from .mcp_harness import call_tool, fake_studio, served

CONFIG = {
    "model_name": "unsloth/Qwen3-0.6B",
    "dataset": "mlabonne/FineTome-100k",
    "max_steps": 30,
    "hf_token": "hf_cfg",
}
STARTED = {"job_id": "job-7", "status": "queued", "message": "Training job queued", "error": None}

PAYLOADS = {("POST", "/api/train/start"): STARTED}


def _studio(overrides = None):
    routes = {
        key: (lambda payload: lambda request, body: payload)(value)
        for key, value in PAYLOADS.items()
    }
    routes.update(overrides or {})
    return fake_studio(routes)


def _call(monkeypatch, studio, name, args):
    with TestClient(served(create_studio_mcp(), studio, monkeypatch = monkeypatch)) as http:
        return call_tool(http, name, args)


def _bodies(studio, path):
    return [json.loads(b) for _m, p, _h, b in studio.state.calls if p == path]


def test_start_forwards_the_config_as_sent(monkeypatch):
    studio = _studio()
    result = _call(monkeypatch, studio, "start_training", {"config": CONFIG})
    assert result["structuredContent"] == {
        "job_id": "job-7",
        "status": "queued",
        "message": "Training job queued",
    }
    assert _bodies(studio, "/api/train/start") == [CONFIG]
    headers = studio.state.calls[0][2]
    assert "x-unsloth-hf-token" not in headers


def test_an_already_active_200_is_a_tool_error(monkeypatch):
    refused = {
        "job_id": "",
        "status": "error",
        "message": "Training is already in progress. Stop current training before starting a new one.",
        "error": "Training already active",
    }
    studio = _studio({("POST", "/api/train/start"): lambda request, body: refused})
    result = _call(monkeypatch, studio, "start_training", {"config": CONFIG})
    assert result["isError"] is True
    assert result["content"][0]["text"] == refused["message"]


def test_start_training_annotations():
    tool = {t.name: t for t in asyncio.run(create_studio_mcp().list_tools())}["start_training"]
    assert tool.annotations.readOnlyHint is False
    assert tool.annotations.destructiveHint is False
    assert tool.annotations.openWorldHint is False
    assert tool.output_schema is not None
