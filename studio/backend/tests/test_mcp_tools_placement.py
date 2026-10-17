# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""load_model's media knobs and device/alongside placement, the placement echo, the partial-variant refusal and unload_model(kind="all")."""

import pytest
from fastapi.responses import JSONResponse

from studio_mcp import forward, model_downloads

from .mcp_harness import bodies, fake_studio, fast_polls, run_tool, sequence, stt_state  # noqa: F401
from .test_mcp_tools_loading import (
    EMPTY_STATUS,
    FLUX_GGUF,
    LLM,
    LOAD,
    LOADED,
    PAYLOADS,
    READY,
    STATUS,
    STATUS_ROUTE,
    _media_studio,
)

KOKORO = "audio-cpp/audio.cpp-gguf/Kokoro-82M-GGUF"
OUTETTS = "OuteAI/OuteTTS-1.0-0.6B-GGUF"
QWEN = "unsloth/Qwen3-0.6B-GGUF"
PLACED = {
    "loaded": True,
    "repo_id": FLUX_GGUF,
    "device": "cuda",
    "memory_mode": "low_vram",
    "offload_policy": "group",
    "transformer_quant": None,
    "text_encoder_quant": "fp8",
}
CPU = {"gpu_memory_mode": "manual", "gpu_layers": 0, "speculative_type": "off"}


def _load(
    monkeypatch,
    args,
    routes = PAYLOADS,
):
    result, studio = run_tool(monkeypatch, routes, "load_model", args)
    return result, studio, bodies(studio, "/api/inference/load")


def test_media_knobs_reach_the_plan_and_the_load(monkeypatch, fast_polls):
    studio = _media_studio("image", progress = [READY], statuses = [PLACED])
    knobs = {"memory_mode": "low_vram", "transformer_quant": "int8", "speed_mode": "off"}
    result, _studio = run_tool(
        monkeypatch, studio, "load_model", {"model": FLUX_GGUF, "kind": "image", **knobs}
    )
    for path in ("download-plan", "load"):
        (sent,) = bodies(studio, f"/api/inference/images/{path}")
        assert {key: sent[key] for key in knobs} == knobs
    assert result["structuredContent"]["placement"] == {
        "device": "gpu",
        "memory_mode": "low_vram",
        "offload_policy": "group",
        "transformer_quant": None,
        "text_encoder_quant": "fp8",
    }


def test_a_plan_that_refuses_the_knobs_is_asked_without_them(monkeypatch, fast_polls):
    refused = JSONResponse({"detail": "extra fields not permitted"}, status_code = 422)
    studio = _media_studio(
        "video",
        statuses = [{"loaded": True, "repo_id": "a/v"}],
        plan = sequence(refused, {"entries": []}),
    )
    run_tool(
        monkeypatch, studio, "load_model", {"model": "a/v", "kind": "video", "h3_task": "fl2va"}
    )
    plans = bodies(studio, "/api/inference/video/download-plan")
    assert plans == [{"model_path": "a/v", "h3_task": "fl2va"}, {"model_path": "a/v"}]
    assert bodies(studio, "/api/inference/video/load") == [
        {"model_path": "a/v", "h3_task": "fl2va"}
    ]


@pytest.mark.parametrize(
    "kind,args",
    [
        ("llm", {"memory_mode": "low_vram"}),
        ("tts", {"h3_task": "fl2va"}),
        ("video", {"transformer_quant": "int8"}),
        ("image", {"device": "cpu"}),
        ("stt", {"alongside": True}),
        ("llm", {"device": "cpu", "model": "unsloth/Qwen3-0.6B"}),
        ("llm", {"device": "gpu", "gpu_layers": 0}),
    ],
)
def test_a_knob_the_kind_does_not_take_is_refused_before_any_call(monkeypatch, kind, args):
    studio = fake_studio(fallback = {})
    result, _studio = run_tool(
        monkeypatch, studio, "load_model", {"model": LLM, "kind": kind, **args}
    )
    assert result["isError"] is True
    assert not [call for call in studio.state.calls if call[0] == "POST"]


@pytest.mark.parametrize(
    "model,kind,device,expected",
    [
        # audio.cpp honours audio_device; it is a GGUF repo that llama-server never sees.
        (KOKORO, "tts", "cpu", {"audio_device": "cpu"}),
        # llama-server TTS (OuteTTS, snac/bicodec/dac) ignores audio_device: no GPU layers either.
        (OUTETTS, "tts", "cpu", {"audio_device": "cpu", **CPU}),
        # A chat GGUF must never be sent audio_device, which the route refuses for it.
        (QWEN, "llm", "cpu", CPU),
        (OUTETTS, "audio", "gpu", {"audio_device": "gpu"}),
        (QWEN, "llm", "gpu", {}),
    ],
)
def test_device_maps_to_the_load_fields(monkeypatch, model, kind, device, expected):
    _result, _studio, (body,) = _load(monkeypatch, {"model": model, "kind": kind, "device": device})
    base = {"model_path": model, "load_in_4bit": True, "max_seq_length": 0}
    assert body == {**base, **expected}


def test_kv_cache_and_gpu_layers(monkeypatch):
    args = {"model": QWEN, "kv_cache_type": "q8_0", "gpu_layers": 20, "alongside": False}
    _result, _studio, (body,) = _load(monkeypatch, args)
    assert body["cache_type_kv"] == "q8_0"
    assert (body["gpu_memory_mode"], body["gpu_layers"], body["alongside"]) == ("manual", 20, False)


@pytest.mark.parametrize(
    "answer,device,placed",
    [
        # An explicit gpu that audio.cpp could not honour reports where the model really is.
        ({"audio_device_fallback": "cpu"}, "gpu", "cpu"),
        ({"offloaded_layers": 0}, None, "cpu"),
        ({"offloaded_layers": 28}, None, "gpu"),
        ({}, None, None),
    ],
)
def test_the_resolved_device_is_echoed(monkeypatch, answer, device, placed):
    routes = {**PAYLOADS, LOAD: {**LOADED, **answer}}
    args = {"model": OUTETTS, "kind": "tts", **({"device": device} if device else {})}
    result, _studio, _bodies = _load(monkeypatch, args, routes)
    placement = result["structuredContent"]["placement"]
    assert (placement or {}).get("device") == placed


@pytest.mark.parametrize(
    "mode,after,applied",
    [
        # Off replaces the primary; When they fit kept it serving next to the new model.
        ("off", {**STATUS, "serving": [OUTETTS]}, False),
        ("fit", {**STATUS, "serving": [LLM, OUTETTS]}, True),
    ],
)
def test_alongside_applied_reports_whether_the_chat_model_still_serves(
    monkeypatch, mode, after, applied
):
    before = {**STATUS, "multi_model_mode": mode}
    routes = {
        **PAYLOADS,
        LOAD: {**LOADED, "model": OUTETTS, "evicted": []},
        STATUS_ROUTE: sequence(before, after),
    }
    args = {"model": OUTETTS, "kind": "tts", "alongside": True}
    result, _studio, _bodies = _load(monkeypatch, args, routes)
    assert result["structuredContent"]["alongside_applied"] is applied


def test_always_with_cpu_joins_the_serving_chat_model(monkeypatch):
    always = {**STATUS, "multi_model_mode": "always"}
    routes = {
        **PAYLOADS,
        LOAD: {**LOADED, "model": KOKORO, "evicted": []},
        STATUS_ROUTE: sequence(always, always, {**always, "serving": [LLM, KOKORO]}),
    }
    result, _studio, (body,) = _load(
        monkeypatch, {"model": KOKORO, "kind": "tts", "device": "cpu"}, routes
    )
    assert body["alongside"] is True
    assert result["structuredContent"]["evicted"] == []
    assert result["structuredContent"]["alongside_applied"] is True


def _partial_routes(state):
    variants = {
        "default_variant": "Q4_K_M",
        "variants": [
            {"quant": "Q4_K_M", "filename": "q.gguf", "downloaded": False, "partial": True}
        ],
    }
    return {
        **PAYLOADS,
        ("GET", "/api/hub/gguf-variants"): variants,
        ("GET", "/api/hub/download-status"): {"state": state},
    }


def test_a_partial_variant_is_refused(monkeypatch):
    result, studio, sent = _load(monkeypatch, {"model": QWEN}, _partial_routes("idle"))
    assert result["isError"] is True
    assert (
        "still downloading or incomplete; wait for it or redownload it"
        in result["content"][0]["text"]
    )
    assert sent == []


def test_a_variant_unsloth_studio_is_downloading_is_not_refused(monkeypatch):
    # Reachable once the download routes are on the MCP allowlist; until then the refusal applies.
    monkeypatch.setattr(
        forward, "ALLOWED_PREFIXES", (*forward.ALLOWED_PREFIXES, "/api/hub/download")
    )
    # load_model waits for that download first (K16 guard 2), so let it finish.
    monkeypatch.setattr(model_downloads, "WAIT_INTERVAL_S", 0.01)
    monkeypatch.setattr(model_downloads, "_remembered", model_downloads.OrderedDict())
    routes = _partial_routes("running")
    routes[("GET", "/api/hub/download-status")] = sequence(
        {"state": "running"}, {"state": "complete"}
    )
    # Once it finishes, the variant is on disk.
    partial = routes[("GET", "/api/hub/gguf-variants")]
    done = {
        **partial,
        "variants": [{**partial["variants"][0], "downloaded": True, "partial": False}],
    }
    routes[("GET", "/api/hub/gguf-variants")] = sequence(partial, done)
    result, _studio, sent = _load(monkeypatch, {"model": QWEN}, routes)
    assert result.get("isError") is not True
    assert len(sent) == 1


def test_unload_all_frees_every_slot_but_the_embedder(monkeypatch):
    serving = {
        "loaded": [LLM, QWEN],
        "serving": [LLM, QWEN],
        "serving_checkpoints": ["/m/llm.gguf", "/m/qwen.gguf"],
    }
    image = "/api/inference/images"
    video = "/api/inference/video"
    stt = "/api/inference/audio/stt/status"
    routes = {
        STATUS_ROUTE: sequence(serving, EMPTY_STATUS),
        ("POST", "/api/inference/unload"): {"status": "unloaded"},
        ("GET", f"{image}/status"): {"loaded": True, "repo_id": FLUX_GGUF},
        ("POST", f"{image}/unload"): {"loaded": False},
        ("GET", f"{video}/status"): {"loaded": True, "repo_id": "a/v"},
        ("POST", f"{video}/unload"): {"loaded": False},
        ("GET", stt): sequence({"transformers": stt_state(loaded = "openai/whisper-small")}, {}),
        ("POST", "/api/inference/audio/stt/unload"): {"status": "unloaded"},
    }
    result, studio = run_tool(monkeypatch, routes, "unload_model", {"kind": "all"}, fallback = {})
    assert result["structuredContent"] == {
        "kind": "all",
        "unloaded": [
            {"kind": "llm", "model": LLM},
            {"kind": "llm", "model": QWEN},
            {"kind": "image", "model": FLUX_GGUF},
            {"kind": "video", "model": "a/v"},
            {"kind": "stt", "model": "openai/whisper-small"},
        ],
    }
    # The kept model goes first, so the primary is never replaced by it mid-way.
    assert bodies(studio, "/api/inference/unload") == [
        {"model_path": "/m/qwen.gguf"},
        {"model_path": "/m/llm.gguf"},
    ]
    assert not [call for call in studio.state.calls if "embedding-model" in call[1]]
