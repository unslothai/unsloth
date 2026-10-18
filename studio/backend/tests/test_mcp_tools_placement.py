# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""load_model's media knobs and device/alongside placement, the placement echo, the partial-variant refusal and unload_model(kind="all")."""

import pytest
from fastapi.responses import JSONResponse

from studio_mcp import forward

from .mcp_harness import (  # noqa: F401
    bodies,
    fake_studio,
    fast_polls,
    run_tool,
    sequence,
    stt_state,
)
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
GGUF_FILES = {"default_variant": "Q4_K_M", "variants": [{"quant": "Q4_K_M", "filename": "q.gguf"}]}


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
    "listing,args,refusal",
    [
        # "gguf" in a repo name proves nothing: only listed GGUF files do.
        ({"variants": []}, {"model": "someone/not-a-gguf-repo"}, "needs a GGUF model"),
        (None, {"model": QWEN}, "needs a GGUF model"),
        (GGUF_FILES, {"model": QWEN, "variant": "Q8_0"}, "has no GGUF variant Q8_0"),
    ],
)
def test_cpu_needs_a_confirmed_gguf(monkeypatch, listing, args, refusal):
    routes = {**PAYLOADS}
    if listing is not None:
        routes[("GET", "/api/hub/gguf-variants")] = listing
    result, studio, sent = _load(monkeypatch, {"kind": "llm", "device": "cpu", **args}, routes)
    assert result["isError"] is True and refusal in str(result["content"])
    assert sent == []


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
    routes = {**PAYLOADS, ("GET", "/api/hub/gguf-variants"): GGUF_FILES}
    args = {"model": model, "kind": kind, "device": device}
    _result, _studio, (body,) = _load(monkeypatch, args, routes)
    base = {"model_path": model, "load_in_4bit": True, "max_seq_length": 0}
    assert body == {**base, **expected}


@pytest.mark.parametrize("model,kind", [("/models/qwen3", "llm"), ("someone/outetts", "tts")])
def test_cpu_fields_follow_the_listing_not_the_name(monkeypatch, model, kind):
    routes = {**PAYLOADS, ("GET", "/api/hub/gguf-variants"): GGUF_FILES}
    _result, _studio, (body,) = _load(
        monkeypatch, {"model": model, "kind": kind, "device": "cpu"}, routes
    )
    assert {key: body.get(key) for key in CPU} == CPU


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
        # A load that kept its layers on the GPU is not reported as on the CPU because cpu was asked.
        ({"offloaded_layers": 28}, "cpu", "gpu"),
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
    # The real allowlist: download-status is forwarded, so a managed download in flight is seen.
    result, _studio, sent = _load(monkeypatch, {"model": QWEN}, _partial_routes("running"))
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


def test_a_media_load_already_downloading_the_variant_is_not_refused(monkeypatch, fast_polls):
    studio = _media_studio(
        "image",
        progress = [{"phase": "downloading"}, READY],
        statuses = [{"loaded": True, "repo_id": FLUX_GGUF}],
    )
    partial = {
        "default_variant": "Q4_K_S",
        "variants": [{"quant": "Q4_K_S", "filename": "f.gguf", "partial": True}],
    }
    studio.router.routes = [
        r for r in studio.router.routes if getattr(r, "path", "") != "/api/hub/gguf-variants"
    ]
    studio.add_api_route("/api/hub/gguf-variants", lambda: partial, methods = ["GET"])
    result, _studio = run_tool(
        monkeypatch, studio, "load_model", {"model": FLUX_GGUF, "kind": "image"}
    )
    assert result.get("isError") is not True, result


def test_unload_all_also_frees_models_held_behind_the_active_one(monkeypatch):
    held = {"loaded": [LLM, QWEN], "serving": [LLM], "serving_checkpoints": ["/m/llm.gguf"]}
    routes = {
        STATUS_ROUTE: sequence(held, {**EMPTY_STATUS, "loaded": [QWEN]}, EMPTY_STATUS),
        ("POST", "/api/inference/unload"): {"status": "unloaded"},
    }
    result, studio = run_tool(monkeypatch, routes, "unload_model", {"kind": "all"}, fallback = {})
    assert bodies(studio, "/api/inference/unload") == [
        {"model_path": "/m/llm.gguf"},
        {"model_path": QWEN},
    ]
    assert [row["model"] for row in result["structuredContent"]["unloaded"]] == [LLM, QWEN]


def test_unload_all_keeps_going_past_a_failed_unload(monkeypatch):
    serving = {
        "loaded": [LLM, QWEN],
        "serving": [LLM, QWEN],
        "serving_checkpoints": ["/m/llm.gguf", "/m/qwen.gguf"],
    }
    busy = JSONResponse({"detail": "a generation is running"}, status_code = 409)

    def unload(request, body):
        return busy if b"qwen" in body else {"status": "unloaded"}

    routes = {
        STATUS_ROUTE: sequence(
            serving, {"loaded": [QWEN], "serving": [QWEN], "serving_checkpoints": ["/m/qwen.gguf"]}
        ),
        ("POST", "/api/inference/unload"): unload,
        ("GET", "/api/inference/images/status"): {"loaded": True, "repo_id": FLUX_GGUF},
        ("POST", "/api/inference/images/unload"): {"loaded": False},
    }
    result, studio = run_tool(monkeypatch, routes, "unload_model", {"kind": "all"}, fallback = {})
    assert result["isError"] is True
    text = str(result["content"])
    assert f"llm {LLM}" in text and f"image {FLUX_GGUF}" in text and QWEN in text
    # The failed kept model did not stop the primary or the image model from unloading.
    assert bodies(studio, "/api/inference/unload")[-1] == {"model_path": "/m/llm.gguf"}
    assert ("POST", "/api/inference/images/unload") in [call[:2] for call in studio.state.calls]


def test_the_cpu_check_reads_a_gated_listing_with_the_calls_hub_token(monkeypatch):
    def listing(request, body):
        token = request.headers.get("X-Unsloth-HF-Token")
        return GGUF_FILES if token == "hf_arg" else {"variants": []}

    routes = {**PAYLOADS, ("GET", "/api/hub/gguf-variants"): listing}
    args = {"model": "someone/gated-qwen", "device": "cpu", "hf_token": "hf_arg"}
    result, _studio, (body,) = _load(monkeypatch, args, routes)
    assert result.get("isError") is not True
    assert {key: body.get(key) for key in CPU} == CPU


def test_cpu_tts_falls_back_to_the_name_when_the_listing_cannot_be_read(monkeypatch):
    # No gguf-variants route: the listing is unreadable (an offline host, say).
    _result, _studio, (body,) = _load(
        monkeypatch, {"model": OUTETTS, "kind": "tts", "device": "cpu"}
    )
    assert body == {
        "model_path": OUTETTS,
        "load_in_4bit": True,
        "max_seq_length": 0,
        "audio_device": "cpu",
        **CPU,
    }


def test_unload_all_still_frees_media_when_chat_status_cannot_be_read(monkeypatch):
    routes = {
        STATUS_ROUTE: JSONResponse({"detail": "restarting"}, status_code = 503),
        ("GET", "/api/inference/images/status"): {"loaded": True, "repo_id": FLUX_GGUF},
        ("POST", "/api/inference/images/unload"): {"loaded": False},
    }
    result, studio = run_tool(monkeypatch, routes, "unload_model", {"kind": "all"}, fallback = {})
    assert result["isError"] is True
    assert f"image {FLUX_GGUF}" in str(result["content"]) and "llm:" in str(result["content"])
    assert ("POST", "/api/inference/images/unload") in [call[:2] for call in studio.state.calls]


def test_only_the_download_status_read_joins_the_allowlist():
    assert "/api/hub/download-status" in forward.ALLOWED_PREFIXES
    for path in ("/api/hub/download", "/api/hub/download/cancel", "/api/hub/download-progress"):
        assert not path.startswith(forward.ALLOWED_PREFIXES)
