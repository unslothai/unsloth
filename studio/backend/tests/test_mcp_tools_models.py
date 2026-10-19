# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio

import pytest
from fastapi.responses import JSONResponse

from .mcp_harness import run_tool

CATALOG = {
    "object": "list",
    "data": [
        {
            "id": "unsloth/Llama-3.2-1B-Instruct-GGUF",
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

MEDIA_ROUTE = "/api/inference/media/catalog"
AUDIO_ROUTE = "/api/inference/audio/catalog"
HARDWARE_ROUTE = "/api/train/hardware"
TI2V = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
TI2V_GGUF = "unsloth/Wan2.2-TI2V-5B-GGUF"
MEDIA = {
    "models": [
        {"id": "unsloth/Z-Image-Turbo", "kind": "image", "size_bytes": 20500000000},
        {"id": "unsloth/stable-diffusion-xl-base-1.0", "kind": "image", "family": "sdxl"},
        {"id": TI2V, "kind": "video", "display_name": "TI2V", "size_bytes": 24200000000},
        {"id": TI2V_GGUF, "kind": "video", "family": "wan2.2-ti2v-5b", "low_vram": True},
    ]
}
# Rows as C2's /audio/catalog serves them, one of each task.
AUDIO = {
    "source": "hub",
    "models": [
        {"id": "audio-cpp/audio.cpp-gguf/Kokoro-82M-GGUF", "task": "tts", "workflows": ["speak"]},
        {"id": "Jmica/Drama-3-GGUF", "task": "tts", "workflows": ["speak", "clone"]},
        {"id": "audio-cpp/audio.cpp-gguf/ACE-Step-GGUF", "task": "music", "workflows": ["music"]},
        {"id": "audio-cpp/audio.cpp-gguf/Whisper-GGUF", "task": "asr", "size_bytes": 900000000},
        {"id": "audio-cpp/audio.cpp-gguf/Experimental-GGUF", "task": "", "workflows": []},
    ],
}


def gpus(*sizes_gib, used = 0.0):
    devices = [
        {"index": i, "vram_total_gb": gib, "vram_used_gb": used} for i, gib in enumerate(sizes_gib)
    ]
    return {"available": True, "backend": "cuda", "devices": devices}


PAYLOADS = {
    ("GET", "/v1/models"): CATALOG,
    ("GET", "/api/inference/loaded-models"): LOADED,
    ("GET", "/api/models/config/unsloth/Qwen3-0.6B"): CONFIG,
    ("GET", MEDIA_ROUTE): MEDIA,
    ("GET", AUDIO_ROUTE): AUDIO,
    ("GET", HARDWARE_ROUTE): gpus(24),
}


def _list(
    monkeypatch,
    args = None,
    headers = None,
    routes = PAYLOADS,
):
    return run_tool(monkeypatch, routes, "list_models", args or {}, headers = headers)


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
    result, studio = _list(monkeypatch, {"kind": kind})
    models = result["structuredContent"]["models"]
    assert {model["id"] for model in models} == expected
    assert {model["kind"] for model in models} == {kind}
    assert [query for _m, _p, query in studio.state.queries] == [""]


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
        "size_bytes": None,
        "low_vram": None,
        "fits_now": None,
        "source": None,
        "audio_workflows": None,
    }
    assert models["unsloth/orpheus-3b"]["audio_workflows"] == ["speak", "clone"]
    assert result["structuredContent"]["training_defaults"] is None
    (_method, path, _headers, _body) = studio.state.calls[-1]
    assert path == "/v1/models"


LLAMA = "unsloth/Llama-3.2-1B-Instruct-GGUF"


# The fast path knows only chat models; without a kind, resident media models are listed too.
@pytest.mark.parametrize(
    "kind,expected,route",
    [
        ("llm", [LLAMA], "/api/inference/loaded-models"),
        (None, [LLAMA, "Lightricks/LTX-Video"], "/v1/models"),
        ("video", ["Lightricks/LTX-Video"], "/v1/models"),
    ],
)
def test_loaded_only_takes_the_fast_path_for_chat(monkeypatch, kind, expected, route):
    args = {"loaded_only": True, **({"kind": kind} if kind else {})}
    result, studio = _list(monkeypatch, args)
    assert [model["id"] for model in result["structuredContent"]["models"]] == expected
    assert [(path, query) for _m, path, query in studio.state.queries] == [(route, "")]


@pytest.mark.parametrize(
    "workflow,expected",
    [
        ("speak", ["unsloth/orpheus-3b"]),
        ("clone", ["unsloth/orpheus-3b"]),
        ("separate", ["audio-separator/htdemucs"]),
        ("music", []),
    ],
)
def test_workflow_keeps_the_models_that_run_that_audio_page(monkeypatch, workflow, expected):
    result, _studio = _list(monkeypatch, {"workflow": workflow})
    assert [model["id"] for model in result["structuredContent"]["models"]] == expected


# Resident chat models carry no audio_workflows, so a workflow filter always reads /v1/models.
def test_workflow_with_loaded_only_reads_every_model(monkeypatch):
    args = {"workflow": "speak", "loaded_only": True, "kind": "llm"}
    result, studio = _list(monkeypatch, args)
    assert result["structuredContent"]["models"] == []
    assert [path for _m, path, _q in studio.state.queries] == ["/v1/models"]


# ``model`` narrows the list to that model and adds its scalar training defaults.
@pytest.mark.parametrize(
    "asked", ["unsloth/Qwen3-0.6B", "unsloth/qwen3-0.6b", "unsloth/Qwen3-0.6B:Q4_K_M"]
)
def test_model_matches_whatever_case_or_variant_is_asked_for(monkeypatch, asked):
    routes = {**PAYLOADS, ("GET", f"/api/models/config/{asked}"): CONFIG}
    result, _studio = _list(monkeypatch, {"model": asked}, routes = routes)
    assert [m["id"] for m in result["structuredContent"]["models"]] == ["unsloth/Qwen3-0.6B"]
    assert result["structuredContent"]["training_defaults"] == {
        "max_seq_length": 2048,
        "learning_rate": "2e-4",
        "packing": False,
        "lora_r": 16,
    }


@pytest.mark.parametrize("token", [None, "hf_secret"])
def test_the_config_request_carries_the_hub_token_only_when_given(monkeypatch, token):
    headers = {"X-Unsloth-HF-Token": token} if token else None
    _result, studio = _list(monkeypatch, {"model": "unsloth/Qwen3-0.6B"}, headers = headers)
    calls = {path: h for _m, path, h, _b in studio.state.calls}
    assert calls["/api/models/config/unsloth/Qwen3-0.6B"].get("x-unsloth-hf-token") == token
    assert "x-unsloth-hf-token" not in calls["/v1/models"]


def _catalog(
    monkeypatch,
    args = None,
    routes = PAYLOADS,
):
    result, studio = _list(monkeypatch, {"source": "catalog", **(args or {})}, routes = routes)
    assert result["isError"] is False, result
    return result["structuredContent"]["models"], [path for _m, path, _q in studio.state.queries]


def test_catalog_lists_media_rows_then_audio_rows_one_to_one(monkeypatch):
    models, paths = _catalog(monkeypatch)
    media, audio = models[: len(MEDIA["models"])], models[len(MEDIA["models"]) :]
    assert [m["id"] for m in media] == [row["id"] for row in MEDIA["models"]]
    assert [m["kind"] for m in media] == ["image", "image", "video", "video"]
    assert [m["low_vram"] for m in media] == [None, None, None, True]
    assert [(m["id"], m["audio_workflows"]) for m in audio] == [
        (row["id"], row.get("workflows", [])) for row in AUDIO["models"]
    ]
    assert [m["kind"] for m in audio] == ["tts", "tts", "audio", "stt", "audio"]
    assert {m["source"] for m in models} == {"catalog"}
    assert models[2]["display_name"] == "TI2V" and models[2]["size_bytes"] == 24200000000
    assert sorted(paths) == sorted([MEDIA_ROUTE, AUDIO_ROUTE, HARDWARE_ROUTE])


# A 24 GiB card holds 25.77 decimal GB; one load lands on one GPU, so two cards never add up.
@pytest.mark.parametrize(
    "hardware,size,fits",
    [
        (gpus(24), 24200000000, True),
        (gpus(12), 24200000000, False),
        (gpus(24), 25500000000, True),
        (gpus(24), 25900000000, False),
        (gpus(12, 12), 20000000000, False),
        (gpus(12, 24), 20000000000, True),
        (gpus(80, used = 70.0), 20000000000, False),
        ({"available": False, "devices": []}, 20000000000, None),
        ({"devices": [{"index": 0, "vram_total_gb": 24}]}, 20000000000, None),
        (gpus(24), None, None),
    ],
)
def test_fits_now_reads_the_most_free_memory_on_one_gpu(monkeypatch, hardware, size, fits):
    row = {"id": TI2V, "kind": "video", "size_bytes": size}
    routes = {**PAYLOADS, ("GET", MEDIA_ROUTE): {"models": [row]}}
    routes[("GET", HARDWARE_ROUTE)] = hardware
    models, _paths = _catalog(monkeypatch, {"kind": "video"}, routes = routes)
    assert [m["fits_now"] for m in models] == [fits]


def test_fits_now_is_unknown_when_the_hardware_route_fails(monkeypatch):
    routes = {**PAYLOADS, ("GET", HARDWARE_ROUTE): JSONResponse({"detail": "x"}, status_code = 500)}
    models, _paths = _catalog(monkeypatch, routes = routes)
    assert {m["fits_now"] for m in models} == {None}


@pytest.mark.parametrize(
    "args,ids,paths",
    [
        ({"kind": "video"}, [TI2V, TI2V_GGUF], [MEDIA_ROUTE]),
        ({"kind": "stt"}, ["audio-cpp/audio.cpp-gguf/Whisper-GGUF"], [AUDIO_ROUTE]),
        ({"workflow": "clone"}, ["Jmica/Drama-3-GGUF"], [AUDIO_ROUTE]),
        ({"kind": "llm"}, [], []),
    ],
)
def test_catalog_fetches_only_what_the_filters_need(monkeypatch, args, ids, paths):
    models, called = _catalog(monkeypatch, args)
    assert [m["id"] for m in models] == ids
    assert [p for p in called if p != HARDWARE_ROUTE] == paths
    assert (HARDWARE_ROUTE in called) == bool(paths)


def test_catalog_model_filter_keeps_training_defaults(monkeypatch):
    routes = {**PAYLOADS, ("GET", f"/api/models/config/{TI2V_GGUF.lower()}"): CONFIG}
    args = {"source": "catalog", "model": TI2V_GGUF.lower()}
    result, _studio = _list(monkeypatch, args, routes = routes)
    assert [m["id"] for m in result["structuredContent"]["models"]] == [TI2V_GGUF]
    assert result["structuredContent"]["training_defaults"]["lora_r"] == 16


def test_catalog_refuses_loaded_only(monkeypatch):
    result, studio = _list(monkeypatch, {"source": "catalog", "loaded_only": True})
    assert result["isError"] is True
    assert "use the default source" in result["content"][0]["text"]
    assert studio.state.queries == []


def test_a_failed_catalog_part_names_itself_instead_of_a_partial_list(monkeypatch):
    routes = {**PAYLOADS, ("GET", AUDIO_ROUTE): JSONResponse({"detail": "down"}, status_code = 503)}
    result, _studio = _list(monkeypatch, {"source": "catalog"}, routes = routes)
    assert result["isError"] is True
    text = result["content"][0]["text"]
    assert "could not list the audio catalog" in text and 'Pass kind="image"' in text


# Negative control: the default source reads only /v1/models, and the new fields stay null.
def test_the_default_source_is_unchanged(monkeypatch):
    result, studio = _list(monkeypatch)
    explicit, _studio = _list(monkeypatch, {"source": "local"})
    models = result["structuredContent"]["models"]
    assert explicit["structuredContent"] == result["structuredContent"]
    assert [m["id"] for m in models] == [row["id"] for row in CATALOG["data"]]
    assert [path for _m, path, _q in studio.state.queries] == ["/v1/models"]
    new = ("size_bytes", "low_vram", "fits_now", "source")
    assert {m[key] for m in models for key in new} == {None}


def test_source_defaults_to_local_in_the_schema():
    from mcp_server import create_studio_mcp

    tools = {tool.name: tool for tool in asyncio.run(create_studio_mcp().list_tools())}
    source = tools["list_models"].parameters["properties"]["source"]
    assert source["default"] == "local" and source["enum"] == ["local", "catalog"]


def test_catalog_skips_audio_rows_the_catalog_refuses(monkeypatch):
    refused = {
        "id": "audio-cpp/audio.cpp-gguf/Unknown-GGUF",
        "task": "",
        "workflows": [],
        "size_bytes": 1000,
        "unsupported_reason": "Unsloth Studio does not support this model family yet.",
    }
    routes = {**PAYLOADS, ("GET", AUDIO_ROUTE): {"models": [*AUDIO["models"], refused]}}
    models, _paths = _catalog(monkeypatch, {"kind": "audio"}, routes = routes)
    assert refused["id"] not in [m["id"] for m in models]
    assert "audio-cpp/audio.cpp-gguf/ACE-Step-GGUF" in [m["id"] for m in models]
