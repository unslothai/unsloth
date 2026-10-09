# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio

import pytest
from fastapi.testclient import TestClient

from mcp_server import create_studio_mcp

from .mcp_harness import call_tool, fake_studio, served

CATALOG = {
    "object": "list",
    "data": [
        {
            "id": "unsloth/Llama-3.2-1B-Instruct-GGUF",
            "object": "model",
            "loaded": True,
            "quant": "Q4_K_M",
            "context_length": 8192,
            "display_name": "Llama 3.2 1B",
        },
        {"id": "unsloth/Qwen3-0.6B", "object": "model", "loaded": False},
        {
            "id": "black-forest-labs/FLUX.1-schnell",
            "task": "text-to-image",
            "loaded": False,
            "quant": "Q8_0",
        },
        {"id": "Lightricks/LTX-Video", "task": "text-to-video", "loaded": True},
        {"id": "openai/whisper-small", "task": "automatic-speech-recognition", "loaded": False},
        {
            "id": "unsloth/orpheus-3b",
            "task": "text-to-speech",
            "loaded": False,
            "audio_workflows": ["speak", "clone"],
        },
        {
            "id": "audio-separator/htdemucs",
            "task": "audio-to-audio",
            "loaded": False,
            "audio_workflows": ["separate"],
        },
    ],
}
LOADED = {
    "object": "list",
    "data": [{"id": "unsloth/Llama-3.2-1B-Instruct-GGUF", "context_length": 8192, "loaded": True}],
}
CONFIG = {
    "id": "unsloth/Qwen3-0.6B",
    "display_name": "Qwen3 0.6B",
    "config": {
        "training": {
            "max_seq_length": 2048,
            "learning_rate": "2e-4",
            "packing": False,
            "output_dir": "/srv/out",
        },
        "lora": {"lora_r": 16, "target_modules": ["q_proj"]},
        "inference": {"temperature": 0.7},
    },
}

PAYLOADS = {
    ("GET", "/v1/models"): CATALOG,
    ("GET", "/api/inference/loaded-models"): LOADED,
    ("GET", "/api/models/config/unsloth/Qwen3-0.6B"): CONFIG,
}


def _list(
    monkeypatch,
    args = None,
    headers = None,
):
    studio = fake_studio(
        {
            key: (lambda payload: lambda request, body: payload)(value)
            for key, value in PAYLOADS.items()
        }
    )
    with TestClient(served(create_studio_mcp(), studio, monkeypatch = monkeypatch)) as http:
        result = call_tool(http, "list_models", args or {}, headers = headers)
    return result, studio


@pytest.mark.parametrize(
    "kind,expected",
    [
        ("llm", {"unsloth/Llama-3.2-1B-Instruct-GGUF", "unsloth/Qwen3-0.6B"}),
        ("image", {"black-forest-labs/FLUX.1-schnell"}),
        ("video", {"Lightricks/LTX-Video"}),
        ("stt", {"openai/whisper-small"}),
        ("tts", {"unsloth/orpheus-3b"}),
        ("audio", {"audio-separator/htdemucs"}),
    ],
)
def test_kind_comes_from_the_task(monkeypatch, kind, expected):
    result, _studio = _list(monkeypatch, {"kind": kind})
    models = result["structuredContent"]["models"]
    assert {model["id"] for model in models} == expected
    assert {model["kind"] for model in models} == {kind}


def test_entries_carry_only_the_allowlisted_fields(monkeypatch):
    result, studio = _list(monkeypatch)
    models = {model["id"]: model for model in result["structuredContent"]["models"]}
    assert models["unsloth/Llama-3.2-1B-Instruct-GGUF"] == {
        "id": "unsloth/Llama-3.2-1B-Instruct-GGUF",
        "kind": "llm",
        "loaded": True,
        "display_name": "Llama 3.2 1B",
        "quant": "Q4_K_M",
        "context_length": 8192,
        "audio_workflows": None,
    }
    assert models["unsloth/orpheus-3b"]["audio_workflows"] == ["speak", "clone"]
    assert result["structuredContent"]["training_defaults"] is None
    (_method, path, _headers, _body) = studio.state.calls[-1]
    assert path == "/v1/models"


def test_output_modalities_is_never_sent(monkeypatch):
    seen = []
    studio = fake_studio(
        {
            ("GET", "/v1/models"): lambda request, body: seen.append(str(request.url.query))
            or CATALOG
        }
    )
    with TestClient(served(create_studio_mcp(), studio, monkeypatch = monkeypatch)) as http:
        for args in ({}, {"kind": "image"}, {"kind": "stt", "loaded_only": True}):
            call_tool(http, "list_models", args)
    assert seen == ["", "", ""]


def test_loaded_only_takes_the_fast_path_for_chat(monkeypatch):
    result, studio = _list(monkeypatch, {"loaded_only": True})
    assert [model["id"] for model in result["structuredContent"]["models"]] == [
        "unsloth/Llama-3.2-1B-Instruct-GGUF"
    ]
    assert [call[1] for call in studio.state.calls] == ["/api/inference/loaded-models"]

    result, studio = _list(monkeypatch, {"loaded_only": True, "kind": "video"})
    assert [model["id"] for model in result["structuredContent"]["models"]] == [
        "Lightricks/LTX-Video"
    ]
    assert [call[1] for call in studio.state.calls] == ["/v1/models"]


def test_model_adds_scalar_training_defaults(monkeypatch):
    result, _studio = _list(monkeypatch, {"model": "unsloth/Qwen3-0.6B"})
    assert result["structuredContent"]["training_defaults"] == {
        "max_seq_length": 2048,
        "learning_rate": "2e-4",
        "packing": False,
        "lora_r": 16,
    }


def test_the_config_request_carries_the_hub_token_only_when_given(monkeypatch):
    _result, studio = _list(monkeypatch, {"model": "unsloth/Qwen3-0.6B"})
    config_headers = [
        h for _m, path, h, _b in studio.state.calls if path.startswith("/api/models/config/")
    ]
    assert "x-unsloth-hf-token" not in config_headers[0]

    _result, studio = _list(
        monkeypatch, {"model": "unsloth/Qwen3-0.6B"}, headers = {"X-Unsloth-HF-Token": "hf_secret"}
    )
    calls = {path: h for _m, path, h, _b in studio.state.calls}
    assert calls["/api/models/config/unsloth/Qwen3-0.6B"]["x-unsloth-hf-token"] == "hf_secret"
    assert "x-unsloth-hf-token" not in calls["/v1/models"]


def test_list_models_is_read_only():
    tool = {t.name: t for t in asyncio.run(create_studio_mcp().list_tools())}["list_models"]
    assert tool.annotations.readOnlyHint is True
    assert tool.annotations.openWorldHint is False
    assert tool.output_schema is not None
    assert set(tool.parameters["properties"]) == {"kind", "loaded_only", "model"}
