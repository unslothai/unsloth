# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""plan_load: what a load would download and unload, checked against what the same load then does."""

import pytest
from fastapi.responses import JSONResponse

from .mcp_harness import bodies, fast_polls, run_tool, stt_state  # noqa: F401
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
        # The partial Q8_0 is one Unsloth Studio is still downloading, so a load waits on it.
        ("GET", "/api/hub/download-status"): {"state": "running"},
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
        ({"model": OUTETTS, "kind": "tts", "device": "gpu"}, [OUTETTS], None),
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
    # Unloading the image model is a bound: the load reaches the GPU only on a GPU host.
    assert plan["multi_model_mode"] == "off" and plan["would_evict_exact"] is (
        after_image is not None
    )
    assert plan["would_refuse"] is False
    after = _world("off", after_serving, image = after_image)
    loaded = []

    def load_route(request, body):
        loaded.append(body)
        # No evicted list: load_model reports what the load unloaded from the slots before and after.
        return {key: value for key, value in LOADED.items() if key != "evicted"} | {
            "model": OUTETTS
        }

    def switch(route):
        return lambda request, body: (after if loaded else before)[route]

    # Status reads give the world before the load until the load POST lands, then the world after.
    load = {
        **before,
        **{route: switch(route) for route in (STATUS, ("GET", IMAGE), ("GET", VIDEO))},
        ("POST", "/api/inference/load"): load_route,
    }
    result, _studio = run_tool(monkeypatch, load, "load_model", args, fallback = {})
    reported = result["structuredContent"]
    assert reported["model"] == OUTETTS
    assert _names(plan["would_evict"]) == _names(reported["evicted_slots"])
    assert _names(plan["would_evict"]) == _unloaded(monkeypatch, plan, after, args)


def test_off_media_load_evicts_chat_on_the_gpu_and_the_other_media_model(monkeypatch, fast_polls):
    before = _world("off", [QWEN], video = LTX)
    plan = _plan(monkeypatch, before, {"model": FLUX_GGUF, "kind": "image"})
    assert _names(plan["would_evict"]) == {("llm", QWEN), ("video", LTX)}
    # An image load may run on the CPU and leave the chat model, so this is an upper bound.
    assert plan["would_evict_exact"] is False and plan["download_bytes"] == 9000
    alone = _plan(monkeypatch, _world("off", [], video = LTX), {"model": FLUX_GGUF, "kind": "image"})
    assert _names(alone["would_evict"]) == {("video", LTX)}
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
        ({"model": "openai/whisper-small", "kind": "stt"}, False),
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


@pytest.mark.parametrize(
    "routes,args",
    [
        # Each of these load_model refuses before sending anything, so plan_load refuses too.
        ({}, {"model": FLUX_GGUF, "kind": "image", "device": "cpu"}),
        ({}, {"model": QWEN, "kind": "stt", "alongside": True}),
        (
            {("GET", "/api/hub/gguf-variants"): {"variants": []}},
            {"model": "someone/not-a-gguf-repo", "device": "cpu"},
        ),
    ],
)
def test_a_load_that_load_model_refuses_gets_no_plan(monkeypatch, routes, args):
    world = {**_world("off", [QWEN]), **routes}
    result, _studio = run_tool(monkeypatch, world, "plan_load", args, fallback = {})
    assert result["isError"] is True


def test_the_plan_of_a_gated_media_repo_carries_the_hub_token(monkeypatch):
    result, studio = run_tool(
        monkeypatch,
        _world("off", []),
        "plan_load",
        {"model": FLUX_GGUF, "kind": "image"},
        headers = {"X-Unsloth-HF-Token": "hf_header"},
        fallback = {},
    )
    assert result["structuredContent"]["download_bytes"] == 9000
    (body,) = bodies(studio, "/api/inference/images/download-plan")
    assert body["hf_token"] == "hf_header"


def test_the_media_options_reach_the_download_plan(monkeypatch):
    world = {
        **_world("off", []),
        ("POST", "/api/inference/video/download-plan"): {"total_bytes": 7},
    }
    args = {"model": LTX, "kind": "video", "h3_task": "ref2va"}
    result, studio = run_tool(monkeypatch, world, "plan_load", args, fallback = {})
    assert result["structuredContent"]["download_bytes"] == 7
    assert bodies(studio, "/api/inference/video/download-plan") == [
        {"model_path": LTX, "h3_task": "ref2va"}
    ]


def test_a_download_plan_refusal_is_the_plans_error(monkeypatch):
    refused = JSONResponse({"detail": "no such precision"}, status_code = 400)
    world = {**_world("off", []), ("POST", "/api/inference/images/download-plan"): refused}
    args = {"model": FLUX_GGUF, "kind": "image", "transformer_quant": "int8"}
    result, _studio = run_tool(monkeypatch, world, "plan_load", args, fallback = {})
    assert result["isError"] is True and "no such precision" in str(result["content"])


def test_a_model_held_behind_the_primary_is_a_resident_off_may_unload(monkeypatch):
    world = _world("off", [QWEN])
    world[STATUS] = {**world[STATUS], "loaded": [QWEN, KEPT]}
    plan = _plan(monkeypatch, world, {"model": OUTETTS, "kind": "tts", "device": "cpu"})
    assert ("llm", KEPT) in _names(plan["residents"])
    # A held standard model can outlive a GGUF load, so it is listed as an upper bound.
    assert _names(plan["would_evict"]) == {("llm", QWEN), ("llm", KEPT)}
    assert plan["would_evict_exact"] is False


def test_a_pick_the_plan_calls_incompatible_gets_no_plan(monkeypatch):
    plan = {"entries": [], "total_bytes": 9, "incompatible_reason": "base and GGUF sizes differ"}
    world = {**_world("off", []), ("POST", "/api/inference/images/download-plan"): plan}
    args = {"model": FLUX_GGUF, "kind": "image"}
    result, _studio = run_tool(monkeypatch, world, "plan_load", args, fallback = {})
    assert result["isError"] is True and "sizes differ" in str(result["content"])


def test_a_partial_variant_load_model_refuses_gets_no_plan(monkeypatch):
    partial = {"variants": [{"quant": "Q8_0", "filename": "b.gguf", "partial": True}]}
    world = {
        **_world("off", []),
        ("GET", "/api/hub/gguf-variants"): partial,
        ("GET", "/api/hub/download-status"): {"state": "idle"},
    }
    args = {"model": QWEN, "variant": "Q8_0"}
    result, _studio = run_tool(monkeypatch, world, "plan_load", args, fallback = {})
    assert result["isError"] is True and "still downloading or incomplete" in str(result["content"])


def test_an_unreadable_media_status_makes_the_plan_an_upper_bound(monkeypatch):
    world = _world("off", [QWEN])
    world[("GET", VIDEO)] = JSONResponse({"detail": "busy"}, status_code = 503)
    plan = _plan(monkeypatch, world, {"model": FLUX_GGUF, "kind": "image"})
    assert plan["multi_model_mode"] == "off" and plan["would_evict_exact"] is False


def test_the_hf_token_argument_reaches_the_listing_and_the_plan(monkeypatch):
    args = {"model": FLUX_GGUF, "kind": "image", "hf_token": "hf_arg"}
    result, studio = run_tool(monkeypatch, _world("off", []), "plan_load", args, fallback = {})
    assert result.get("isError") is not True
    (body,) = bodies(studio, "/api/inference/images/download-plan")
    assert body["hf_token"] == "hf_arg"


def test_always_refuses_another_model_in_an_occupied_media_slot(monkeypatch):
    world = _world("always", [], video = "other/video-model")
    plan = _plan(monkeypatch, world, {"model": LTX, "kind": "video"})
    assert plan["would_refuse"] is True and plan["would_evict"] == []


def test_a_gguf_repo_is_known_by_its_listing_not_its_name(monkeypatch):
    plan = _plan(monkeypatch, _world("off", []), {"model": "someone/qwen3-quants"})
    assert [row["variant"] for row in plan["variants"]] == ["Q4_K_M", "Q8_0"]


def test_a_native_tts_primary_makes_a_media_plan_an_upper_bound(monkeypatch):
    # It may run on the CPU and keep its place, and status does not say which.
    world = _world("off", [OUTETTS], is_audio = True)
    plan = _plan(monkeypatch, world, {"model": FLUX_GGUF, "kind": "image"})
    assert _names(plan["would_evict"]) == {("llm", OUTETTS)}
    assert plan["would_evict_exact"] is False


@pytest.mark.parametrize(
    "resident,args",
    [
        # An omitted task loads H3's keyframe partition, not the resident reference one.
        ({"h3_task": "ref2va"}, {"model": LTX, "kind": "video"}),
        ({"h3_task": "fl2va"}, {"model": LTX, "kind": "video", "h3_task": "ref2va"}),
    ],
)
def test_a_load_of_the_resident_media_repo_may_load_it_again(monkeypatch, resident, args):
    world = _world("off", [QWEN], video = LTX)
    world[("GET", VIDEO)] = {**world[("GET", VIDEO)], **resident}
    plan = _plan(monkeypatch, world, args)
    assert _names(plan["would_evict"]) == {("llm", QWEN), ("video", LTX)}
    assert plan["would_evict_exact"] is False


def test_a_local_gguf_file_may_be_a_vision_model_on_the_gpu(monkeypatch):
    plan = _plan(
        monkeypatch,
        _world("off", [QWEN], image = FLUX_GGUF),
        {"model": "/models/vision.gguf", "device": "cpu"},
    )
    assert ("image", FLUX_GGUF) in _names(plan["would_evict"])
    assert plan["would_evict_exact"] is False


def test_unloading_another_kind_is_always_an_upper_bound(monkeypatch):
    # A TTS may run on the CPU (UNSLOTH_AUDIO_DEVICE=cpu, or no GPU at all) and leave the image model.
    world = _world("off", [QWEN], image = FLUX_GGUF)
    for args in (
        {"model": OUTETTS, "kind": "tts"},
        {"model": OUTETTS, "kind": "tts", "device": "gpu"},
    ):
        plan = _plan(monkeypatch, world, args)
        assert _names(plan["would_evict"]) == {("llm", QWEN), ("image", FLUX_GGUF)}
        assert plan["would_evict_exact"] is False


def test_a_speech_to_text_load_may_release_the_other_engines_model(monkeypatch):
    plan = _plan(
        monkeypatch, _world("off", []), {"model": "openai/whisper-large-v3", "kind": "stt"}
    )
    assert _names(plan["would_evict"]) == {("stt", "openai/whisper-small")}
    assert plan["would_evict_exact"] is False


@pytest.mark.parametrize(
    "args",
    [
        # The same variant may still load again (another context, device or cache type).
        {"model": QWEN, "variant": "Q4_K_M"},
        {"model": QWEN, "variant": "Q4_K_M", "kv_cache_type": "q8_0"},
        {"model": QWEN},
    ],
)
def test_a_load_of_a_resident_chat_repo_is_a_possible_reload(monkeypatch, args):
    plan = _plan(monkeypatch, _world("off", [f"{QWEN}:Q4_K_M"]), args)
    assert _names(plan["would_evict"]) == {("llm", f"{QWEN}:Q4_K_M")}
    assert plan["would_evict_exact"] is False


def test_local_paths_keep_their_case(monkeypatch):
    plan = _plan(monkeypatch, _world("always", ["/models/Foo.gguf"]), {"model": "/models/foo.gguf"})
    assert plan["would_refuse"] is True


def test_an_unreadable_stt_status_makes_an_stt_plan_an_upper_bound(monkeypatch):
    world = _world("off", [])
    world[("GET", "/api/inference/audio/stt/status")] = JSONResponse({}, status_code = 503)
    plan = _plan(monkeypatch, world, {"model": "openai/whisper-large-v3", "kind": "stt"})
    assert plan["would_evict"] == [] and plan["would_evict_exact"] is False


def test_always_does_not_protect_a_chat_model_holding_no_vram(monkeypatch):
    world = _world("always", [QWEN], gpu_memory_mode = "manual", gpu_layers = 0)
    plan = _plan(monkeypatch, world, {"model": FLUX_GGUF, "kind": "image"})
    assert plan["would_refuse"] is False


def test_an_unreadable_chat_status_makes_the_plan_an_upper_bound(monkeypatch):
    world = _world("off", [QWEN], image = FLUX_GGUF)
    world[STATUS] = JSONResponse({}, status_code = 503)
    plan = _plan(monkeypatch, world, {"model": QWEN, "variant": "Q4_K_M", "device": "gpu"})
    assert plan["would_evict_exact"] is False


def on_gpu_estimate(monkeypatch, world):
    return _plan(monkeypatch, world, {"model": QWEN, "gpu_layers": 20})["est_gpu_bytes"]


def test_gpu_layers_places_a_gguf_plan_like_device(monkeypatch):
    world = _world("off", [], image = FLUX_GGUF)
    off_gpu = _plan(monkeypatch, world, {"model": QWEN, "gpu_layers": 0})
    on_gpu = _plan(monkeypatch, world, {"model": QWEN, "gpu_layers": 20})
    auto = _plan(monkeypatch, world, {"model": QWEN, "gpu_layers": 0, "device": "auto"})
    assert auto["would_evict"] == []
    assert off_gpu["would_evict"] == [] and off_gpu["would_evict_exact"] is True
    # The memory estimate prices a GPU load, so it is left out for one kept off the GPU.
    assert off_gpu["est_gpu_bytes"] is None and on_gpu_estimate(monkeypatch, world) == 1234
    assert (
        _names(on_gpu["would_evict"]) == {("image", FLUX_GGUF)}
        and on_gpu["would_evict_exact"] is False
    )


def test_a_failed_listing_leaves_vision_unknown(monkeypatch):
    calls = []

    def listing(request, body):
        # The CPU check reads it; the plan's own read then fails.
        calls.append(1)
        return VARIANTS if len(calls) == 1 else JSONResponse({}, status_code = 503)

    world = {**_world("off", [], image = FLUX_GGUF), ("GET", "/api/hub/gguf-variants"): listing}
    plan = _plan(monkeypatch, world, {"model": QWEN, "device": "cpu"})
    assert plan["would_evict_exact"] is False


def test_a_chat_model_still_loading_makes_the_plan_an_upper_bound(monkeypatch):
    world = _world("off", [], loading = [KEPT])
    plan = _plan(monkeypatch, world, {"model": FLUX_GGUF, "kind": "image"})
    assert plan["would_evict_exact"] is False


def test_an_stt_model_status_repeats_is_listed_once(monkeypatch):
    world = _world("off", [])
    whisper = stt_state(loaded = "openai/whisper-small")
    world[("GET", "/api/inference/audio/stt/status")] = {"transformers": whisper, "active": whisper}
    plan = _plan(monkeypatch, world, {"model": "openai/whisper-large-v3", "kind": "stt"})
    assert plan["would_evict"] == [{"kind": "stt", "model": "openai/whisper-small"}]


def test_the_memory_estimate_carries_the_hub_token(monkeypatch):
    args = {"model": QWEN, "hf_token": "hf_arg"}
    result, studio = run_tool(monkeypatch, _world("off", []), "plan_load", args, fallback = {})
    assert result["structuredContent"]["est_gpu_bytes"] == 1234
    (body,) = bodies(studio, "/api/inference/estimate-memory")
    assert body["hf_token"] == "hf_arg"
