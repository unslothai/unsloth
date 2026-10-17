# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""plan_load: what a load would download and unload, checked against what the same load then does."""

import pytest

from .mcp_harness import fast_polls, run_tool, stt_state  # noqa: F401
from .test_mcp_tools_loading import FLUX_GGUF, LOADED, LTX, READY, _media_studio

QWEN = "unsloth/Qwen3-0.6B-GGUF"
KEPT = "unsloth/gemma-3-1b-it-GGUF"
OUTETTS = "OuteAI/OuteTTS-1.0-0.6B-GGUF"
IMAGE, VIDEO = "/api/inference/images/status", "/api/inference/video/status"
STATUS = ("GET", "/api/inference/status")
VARIANTS = {
    "default_variant": "Q4_K_M",
    "has_vision": False,
    "variants": [
        {"quant": "Q4_K_M", "filename": "a.gguf", "size_bytes": 400, "downloaded": True},
        {
            "quant": "Q8_0",
            "filename": "b.gguf",
            "size_bytes": 800,
            "download_remaining_bytes": 300,
            "partial": True,
        },
    ],
}


def _chat(mode, serving, **fields):
    return {
        "serving": serving,
        "serving_checkpoints": [f"/m/{n}" for n in serving],
        "loaded": serving,
        **({"multi_model_mode": mode} if mode else {}),
        **fields,
    }


def _world(
    mode,
    serving,
    image = None,
    video = None,
    **chat_fields,
):
    return {
        STATUS: _chat(mode, serving, **chat_fields),
        ("GET", IMAGE): {"loaded": True, "repo_id": image} if image else {"loaded": False},
        ("GET", VIDEO): {"loaded": True, "repo_id": video} if video else {"loaded": False},
        ("GET", "/api/inference/audio/stt/status"): {
            "transformers": stt_state(loaded = "openai/whisper-small")
        },
        ("GET", "/api/hub/gguf-variants"): VARIANTS,
        ("POST", "/api/inference/estimate-memory"): {"available": True, "gpu_bytes": 1234},
        ("POST", "/api/inference/images/download-plan"): {"entries": [], "total_bytes": 9000},
    }


def _plan(monkeypatch, routes, args):
    result, _studio = run_tool(monkeypatch, routes, "plan_load", args, fallback = {})
    return result["structuredContent"]


def _names(slots):
    return {(s["kind"], s["model"]) for s in slots}


def _unloaded(monkeypatch, before, after_routes, args):
    # What the load really unloaded: the residents before minus the residents after.
    after = _plan(monkeypatch, after_routes, {"model": "x/y"})
    return _names(before["residents"]) - _names(after["residents"])


@pytest.mark.parametrize(
    "args,after_serving,after_image",
    [
        # A CPU TTS load still replaces the chat primary (TTS loads in the chat slot), and the kept
        # model goes with it; the image model stays because nothing touched the GPU.
        ({"model": OUTETTS, "kind": "tts", "device": "cpu"}, [OUTETTS], FLUX_GGUF),
        # A GPU chat load also takes the GPU from the image model.
        ({"model": OUTETTS, "kind": "tts"}, [OUTETTS], None),
        # Off ignores alongside.
        (
            {"model": OUTETTS, "kind": "tts", "device": "cpu", "alongside": True},
            [OUTETTS],
            FLUX_GGUF,
        ),
    ],
)
def test_off_is_exact_and_matches_the_load(monkeypatch, args, after_serving, after_image):
    before = _world("off", [QWEN, KEPT], image = FLUX_GGUF)
    plan = _plan(monkeypatch, before, args)
    assert plan["multi_model_mode"] == "off" and plan["would_evict_exact"] is True
    assert plan["would_refuse"] is False
    after = _world("off", after_serving, image = after_image)
    loaded = []

    def load_route(request, body):
        loaded.append(body)
        return {**LOADED, "model": OUTETTS, "evicted": []}

    # Status reads give the world before the load until the load POST lands, then the world after.
    load = {
        **before,
        STATUS: lambda request, body: after[STATUS] if loaded else before[STATUS],
        ("POST", "/api/inference/load"): load_route,
    }
    result, _studio = run_tool(monkeypatch, load, "load_model", args, fallback = {})
    assert result["structuredContent"]["model"] == OUTETTS
    assert _names(plan["would_evict"]) == _unloaded(monkeypatch, plan, after, args)


def test_off_media_load_evicts_chat_on_the_gpu_and_the_other_media_model(monkeypatch, fast_polls):
    before = _world("off", [QWEN], video = LTX)
    plan = _plan(monkeypatch, before, {"model": FLUX_GGUF, "kind": "image"})
    assert _names(plan["would_evict"]) == {("llm", QWEN), ("video", LTX)}
    assert plan["would_evict_exact"] is True and plan["download_bytes"] == 9000
    # A primary on the CPU holds no GPU claim, so the image load leaves it: no longer exact.
    cpu = _world("off", [QWEN], video = LTX, gpu_memory_mode = "manual", gpu_layers = 0)
    plan = _plan(monkeypatch, cpu, {"model": FLUX_GGUF, "kind": "image"})
    assert _names(plan["would_evict"]) == {("video", LTX)} and plan["would_evict_exact"] is False
    studio = _media_studio(
        "image",
        progress = [READY],
        statuses = [{"loaded": True, "repo_id": FLUX_GGUF}],
        chat_statuses = [_chat("off", [QWEN]), _chat("off", [])],
    )
    result, _studio = run_tool(
        monkeypatch, studio, "load_model", {"model": FLUX_GGUF, "kind": "image"}
    )
    assert result["structuredContent"]["evicted"] == [QWEN]


@pytest.mark.parametrize("mode", ["fit", None])
def test_fit_and_an_unknown_mode_give_an_upper_bound(monkeypatch, mode):
    before = _world(mode, [QWEN, KEPT], image = FLUX_GGUF)
    plan = _plan(monkeypatch, before, {"model": OUTETTS, "kind": "tts", "alongside": True})
    assert plan["would_evict_exact"] is False and plan["would_refuse"] is False
    # When they fit kept both chat models and the image model: a subset of the bound.
    actual = _unloaded(monkeypatch, plan, _world(mode, [QWEN, KEPT, OUTETTS], image = FLUX_GGUF), {})
    assert actual <= _names(plan["would_evict"])
    assert _names(plan["would_evict"]) >= {("llm", QWEN), ("llm", KEPT), ("image", FLUX_GGUF)}


@pytest.mark.parametrize(
    "args,refuse",
    [
        ({"model": OUTETTS, "kind": "tts"}, True),
        # Under Always a CPU load joins the serving chat model, so nothing needs unloading.
        ({"model": OUTETTS, "kind": "tts", "device": "cpu"}, False),
        ({"model": LTX, "kind": "video"}, True),
        ({"model": "openai/whisper-large-v3", "kind": "stt"}, False),
    ],
)
def test_always_never_evicts_and_bounds_the_refusal(monkeypatch, args, refuse):
    plan = _plan(monkeypatch, _world("always", [QWEN], image = FLUX_GGUF), args)
    assert plan["would_evict"] == [] and plan["would_refuse"] is refuse


def test_variants_sizes_and_the_gpu_estimate(monkeypatch):
    plan = _plan(monkeypatch, _world("off", []), {"model": QWEN})
    assert plan["variants"] == [
        {"variant": "Q4_K_M", "size_bytes": 400, "downloaded": True, "partial": False},
        {"variant": "Q8_0", "size_bytes": 800, "downloaded": False, "partial": True},
    ]
    assert (plan["download_bytes"], plan["est_gpu_bytes"]) == (0, 1234)
    plan = _plan(monkeypatch, _world("off", []), {"model": QWEN, "variant": "Q8_0"})
    assert (plan["download_bytes"], plan["est_gpu_bytes"]) == (300, None)
    assert _names(plan["residents"]) == {("stt", "openai/whisper-small")}
    missing, _studio = run_tool(
        monkeypatch, _world("off", []), "plan_load", {"model": QWEN, "variant": "Q2_K"}
    )
    assert missing["isError"] is True and "no GGUF variant Q2_K" in missing["content"][0]["text"]


def test_another_variant_of_a_resident_repo_is_a_new_load(monkeypatch):
    world = _world("off", [f"{QWEN}:Q4_K_M"], image = FLUX_GGUF)
    plan = _plan(monkeypatch, world, {"model": QWEN, "variant": "Q8_0"})
    assert _names(plan["would_evict"]) == {("llm", f"{QWEN}:Q4_K_M"), ("image", FLUX_GGUF)}
    assert _plan(monkeypatch, world, {"model": QWEN, "variant": "Q4_K_M"})["would_evict"] == []
    # Status naming only the bare repo cannot tell which variant serves.
    plan = _plan(monkeypatch, _world("off", [QWEN]), {"model": QWEN, "variant": "Q8_0"})
    assert plan["would_evict_exact"] is False


def test_nothing_left_to_download_is_zero(monkeypatch):
    rows = [
        {
            "quant": "Q8_0",
            "filename": "b.gguf",
            "download_remaining_bytes": 0,
            "download_size_bytes": 800,
        }
    ]
    world = {**_world("off", []), ("GET", "/api/hub/gguf-variants"): {"variants": rows}}
    assert _plan(monkeypatch, world, {"model": QWEN, "variant": "Q8_0"})["download_bytes"] == 0
