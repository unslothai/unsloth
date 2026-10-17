# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from fastapi.responses import JSONResponse

from .mcp_harness import run_tool

CHAT = {
    "active_model": "Llama 3.2 1B",
    "model_identifier": "unsloth/Llama-3.2-1B-Instruct-GGUF",
    "is_gguf": True,
    "loaded": ["unsloth/Llama-3.2-1B-Instruct-GGUF", "unsloth/Qwen3-0.6B"],
    "loading": ["unsloth/gemma-3-1b-it"],
    "serving_checkpoints": ["/srv/models/llama"],
}
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
TRAINING = {
    "job_id": "job-1",
    "phase": "training",
    "is_training_running": True,
    "message": "Training step 25",
    "error": None,
    "details": {"step": 25, "total_steps": 100, "loss": 1.25, "output_dir": "/srv/outputs/run"},
}
DIFFUSION_TRAINING = {
    "active": True,
    "job_id": "diff-1",
    "status": "training",
    "message": "Step 12/500",
    "step": 12,
    "total_steps": 500,
    "loss": 0.5,
    "output_dir": "/srv/outputs/diff",
    "lora_path": "/srv/outputs/diff/lora.safetensors",
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
    ("GET", "/api/inference/images/status"): {
        "loaded": True,
        "repo_id": "/srv/hf/flux",
        "display_repo_id": "black-forest-labs/FLUX.1-schnell",
        "family": "flux",
    },
    ("GET", "/api/inference/video/status"): {
        "loaded": True,
        "repo_id": "Lightricks/LTX-Video",
        "display_repo_id": None,
    },
    ("GET", "/api/inference/audio/stt/status"): STT,
    ("GET", "/api/settings/embedding-model"): {
        "embedding_model": "/srv/custom-embedder",
        "is_custom": True,
        "loaded": True,
    },
    ("GET", "/api/train/status"): TRAINING,
    ("GET", "/api/train/diffusion/status"): DIFFUSION_TRAINING,
    ("GET", "/api/export/status"): {
        "is_export_active": False,
        "active_op_kind": None,
        "last_op_status": "succeeded",
        "last_op_output_path": "/srv/exports/my-model-gguf",
    },
    ("GET", "/api/train/hardware"): HARDWARE,
}


def _status(monkeypatch, overrides = None):
    return run_tool(monkeypatch, {**PAYLOADS, **(overrides or {})}, "studio_status")


def test_every_slot_is_reported(monkeypatch):
    # No generate-progress route answers here: an unreadable generation check reads as not generating.
    result, studio = _status(monkeypatch)
    status = result["structuredContent"]
    assert status["chat"] == {
        "loaded": [
            {
                "id": "unsloth/Llama-3.2-1B-Instruct-GGUF",
                "display_name": "Llama 3.2 1B",
                "is_gguf": True,
                "audio_options": None,
            },
            {
                "id": "unsloth/Qwen3-0.6B",
                "display_name": None,
                "is_gguf": None,
                "audio_options": None,
            },
        ],
        "loading": ["unsloth/gemma-3-1b-it"],
    }
    assert status["image"] == {
        "loaded": True,
        "model": "black-forest-labs/FLUX.1-schnell",
        "family": "flux",
        "generating": False,
        "placement": None,
    }
    assert status["video"] == {
        "loaded": True,
        "model": "Lightricks/LTX-Video",
        "generating": False,
        "placement": None,
    }
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
    # Image LoRA training runs on its own trainer: start_training(kind="diffusion") is followed here.
    assert status["diffusion_training"] == {
        "job_id": "diff-1",
        "phase": "training",
        "is_training_running": True,
        "message": "Step 12/500",
        "error": None,
        "step": 12,
        "total_steps": 500,
        "loss": 0.5,
        "progress_percent": 2.4,
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


def test_failing_and_owner_only_slots_do_not_fail_the_tool(monkeypatch):
    broken = JSONResponse({"detail": "Video backend crashed at /srv/hf/ltx"}, status_code = 500)
    owner_only = JSONResponse(
        {"detail": "Only the installation owner can do this"}, status_code = 403
    )
    overrides = {
        ("GET", "/api/inference/video/status"): broken,
        # A managed key sees the embedder as unavailable, not as a failure.
        ("GET", "/api/settings/embedding-model"): owner_only,
        ("GET", "/api/inference/images/generate-progress"): {"active": True, "step": 3},
        ("GET", "/api/inference/video/generate-progress"): {"active": False},
    }
    result, _studio = _status(monkeypatch, overrides)
    assert result["isError"] is False
    status = result["structuredContent"]
    assert set(status["unavailable"]) == {"video"}
    assert "Video backend crashed" in status["unavailable"]["video"]
    assert "/srv/hf" not in status["unavailable"]["video"]
    assert status["video"] == {
        "loaded": False,
        "model": None,
        "generating": False,
        "placement": None,
    }
    assert status["embedder"] == {"available": False, "loaded": False, "model": None}
    assert status["image"]["generating"] is True
    assert status["chat"]["loaded"]


def test_the_active_audio_model_lists_its_options(monkeypatch):
    tts = "unsloth/Qwen3-TTS-12Hz-1.7B-CustomVoice-GGUF"
    voice = {
        "name": "voice",
        "type": "enum",
        "description": "Built-in speaker.",
        "required": False,
        "default": "vivian",
        "min": None,
        "max": None,
        "values": ["vivian", "ryan"],
    }
    chat = {
        "model_identifier": tts,
        "loaded": [tts],
        "audio_options": [voice, {"name": "temperature", "type": "float", "default": 0.9}, "x"],
    }
    overrides = {("GET", "/api/inference/status"): chat}
    (model,) = _status(monkeypatch, overrides)[0]["structuredContent"]["chat"]["loaded"]
    assert model["audio_options"] == [
        {
            "name": "voice",
            "type": "enum",
            "description": "Built-in speaker.",
            "default": "vivian",
            "min": None,
            "max": None,
            "values": ["vivian", "ryan"],
        },
        {
            "name": "temperature",
            "type": "float",
            "description": None,
            "default": 0.9,
            "min": None,
            "max": None,
            "values": [],
        },
    ]


def test_unloaded_slots_name_no_model(monkeypatch):
    overrides = {
        ("GET", "/api/inference/images/status"): {"loaded": False, "repo_id": "stale/repo"},
        ("GET", "/api/inference/video/status"): {"loaded": False},
    }
    status = _status(monkeypatch, overrides)[0]["structuredContent"]
    unplaced = {"generating": False, "placement": None}
    assert status["image"] == {"loaded": False, "model": None, "family": None, **unplaced}
    assert status["video"] == {"loaded": False, "model": None, **unplaced}
