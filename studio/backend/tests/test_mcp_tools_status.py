# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio

from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from mcp_server import create_studio_mcp

from .mcp_harness import call_tool, fake_studio, served

CHAT = {
    "active_model": "Llama 3.2 1B",
    "model_identifier": "unsloth/Llama-3.2-1B-Instruct-GGUF",
    "is_gguf": True,
    "loaded": ["unsloth/Llama-3.2-1B-Instruct-GGUF", "unsloth/Qwen3-0.6B"],
    "loading": ["unsloth/gemma-3-1b-it"],
    "serving_checkpoints": ["/srv/models/llama"],
}
IMAGE = {
    "loaded": True,
    "repo_id": "/srv/hf/flux",
    "display_repo_id": "black-forest-labs/FLUX.1-schnell",
    "family": "flux",
}
VIDEO = {"loaded": True, "repo_id": "Lightricks/LTX-Video", "display_repo_id": None}
STT = {
    "available": True,
    "transformers": {
        "loaded_model": None,
        "loading": False,
        "download": {
            "downloading": True,
            "model": "openai/whisper-small",
            "bytes_done": 25,
            "bytes_total": 100,
        },
    },
    "gguf": {
        "loaded_model": "whisper-large-v3-turbo-q5",
        "loading": False,
        "download": {"downloading": False},
    },
    "mtmd": {"loaded_model": None, "loading": True},
}
EMBEDDER = {"embedding_model": "/srv/custom-embedder", "is_custom": True, "loaded": True}
TRAINING = {
    "job_id": "job-1",
    "phase": "training",
    "is_training_running": True,
    "message": "Training step 25",
    "error": None,
    "details": {"step": 25, "total_steps": 100, "loss": 1.25, "output_dir": "/srv/outputs/run"},
}
EXPORT = {
    "is_export_active": False,
    "active_op_kind": None,
    "last_op_status": "succeeded",
    "last_op_output_path": "/srv/exports/my-model-gguf",
}
HARDWARE = {
    "available": True,
    "backend": "cuda",
    "devices": [
        {
            "index": 0,
            "gpu_utilization_pct": 40,
            "vram_used_gb": 3.5,
            "vram_total_gb": 24.0,
            "temperature_c": 51,
            "name": "RTX",
        }
    ],
}

PAYLOADS = {
    ("GET", "/api/inference/status"): CHAT,
    ("GET", "/api/inference/images/status"): IMAGE,
    ("GET", "/api/inference/video/status"): VIDEO,
    ("GET", "/api/inference/audio/stt/status"): STT,
    ("GET", "/api/settings/embedding-model"): EMBEDDER,
    ("GET", "/api/train/status"): TRAINING,
    ("GET", "/api/export/status"): EXPORT,
    ("GET", "/api/train/hardware"): HARDWARE,
}


def _answer(payload):
    return lambda request, body: payload


def _status(monkeypatch, overrides = None):
    routes = {key: _answer(payload) for key, payload in PAYLOADS.items()}
    routes.update(overrides or {})
    studio = fake_studio(routes)
    with TestClient(served(create_studio_mcp(), studio, monkeypatch = monkeypatch)) as http:
        result = call_tool(http, "studio_status")
    return result, studio


def test_every_slot_is_reported(monkeypatch):
    result, studio = _status(monkeypatch)
    status = result["structuredContent"]
    assert status["chat"] == {
        "loaded": [
            {
                "id": "unsloth/Llama-3.2-1B-Instruct-GGUF",
                "display_name": "Llama 3.2 1B",
                "is_gguf": True,
            },
            {"id": "unsloth/Qwen3-0.6B", "display_name": None, "is_gguf": None},
        ],
        "loading": ["unsloth/gemma-3-1b-it"],
    }
    assert status["image"] == {
        "loaded": True,
        "model": "black-forest-labs/FLUX.1-schnell",
        "family": "flux",
        "generating": False,
    }
    assert status["video"] == {"loaded": True, "model": "Lightricks/LTX-Video", "generating": False}
    assert status["stt"] == {
        "engine": "gguf",
        "model": "whisper-large-v3-turbo-q5",
        "loading": True,
        "downloading": [{"model": "openai/whisper-small", "fraction": 0.25}],
    }
    assert status["embedder"] == {"available": True, "loaded": True, "model": "custom"}
    assert status["training"] == {
        "job_id": "job-1",
        "phase": "training",
        "is_training_running": True,
        "message": "Training step 25",
        "error": None,
        "step": 25,
        "total_steps": 100,
        "loss": 1.25,
        "progress_percent": 25.0,
        "eta_seconds": None,
    }
    assert status["export"] == {
        "active": False,
        "op_kind": None,
        "last_op_status": "succeeded",
        "last_output": "my-model-gguf",
    }
    assert status["hardware"] == {
        "available": True,
        "backend": "cuda",
        "devices": [
            {
                "index": 0,
                "gpu_utilization_pct": 40.0,
                "vram_used_gb": 3.5,
                "vram_total_gb": 24.0,
                "temperature_c": 51.0,
            }
        ],
    }
    assert status["unavailable"] == {}
    called = {(method, path) for method, path, _headers, _body in studio.state.calls}
    assert called == set(PAYLOADS)


def test_a_failing_slot_does_not_fail_the_tool(monkeypatch):
    def broken(request, body):
        return JSONResponse({"detail": "Video backend crashed at /srv/hf/ltx"}, status_code = 500)

    result, _studio = _status(monkeypatch, {("GET", "/api/inference/video/status"): broken})
    assert result["isError"] is False
    status = result["structuredContent"]
    assert set(status["unavailable"]) == {"video"}
    assert "Video backend crashed" in status["unavailable"]["video"]
    assert "/srv/hf" not in status["unavailable"]["video"]
    assert status["video"] == {"loaded": False, "model": None, "generating": False}
    assert status["chat"]["loaded"]


def test_a_managed_key_sees_the_embedder_as_unavailable(monkeypatch):
    def owner_only(request, body):
        return JSONResponse({"detail": "Only the installation owner can do this"}, status_code = 403)

    result, _studio = _status(monkeypatch, {("GET", "/api/settings/embedding-model"): owner_only})
    status = result["structuredContent"]
    assert status["embedder"] == {"available": False, "loaded": False, "model": None}
    assert "embedder" not in status["unavailable"]


def test_unloaded_slots_name_no_model(monkeypatch):
    overrides = {
        ("GET", "/api/inference/images/status"): _answer(
            {"loaded": False, "repo_id": "stale/repo"}
        ),
        ("GET", "/api/inference/video/status"): _answer({"loaded": False}),
    }
    result, _studio = _status(monkeypatch, overrides)
    status = result["structuredContent"]
    assert status["image"] == {"loaded": False, "model": None, "family": None, "generating": False}
    assert status["video"] == {"loaded": False, "model": None, "generating": False}


def test_a_running_generation_is_reported(monkeypatch):
    result, _studio = _status(
        monkeypatch,
        {
            ("GET", "/api/inference/images/generate-progress"): _answer(
                {"active": True, "step": 3}
            ),
            ("GET", "/api/inference/video/generate-progress"): _answer({"active": False}),
        },
    )
    status = result["structuredContent"]
    assert status["image"]["generating"] is True
    assert status["video"]["generating"] is False


def test_an_unreadable_generation_check_reads_as_not_generating(monkeypatch):
    result, _studio = _status(monkeypatch)
    status = result["structuredContent"]
    assert status["image"]["generating"] is False
    assert status["video"]["generating"] is False
    assert "image" not in status["unavailable"]


def test_studio_status_is_read_only_with_an_output_schema():
    tools = {tool.name: tool for tool in asyncio.run(create_studio_mcp().list_tools())}
    tool = tools["studio_status"]
    assert tool.annotations.readOnlyHint is True
    assert tool.annotations.openWorldHint is False
    assert tool.output_schema is not None
    assert tool.output_schema["additionalProperties"] is False
    assert tool.parameters.get("properties", {}) == {}
