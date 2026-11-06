# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
import json
import time

import pytest
from fastapi.responses import JSONResponse, Response
from fastmcp.exceptions import ToolError

from studio_mcp import loading

from .mcp_harness import (
    DEFERRED,
    WHISPER,
    bodies,
    media_load_bodies,
    call_to,
    call_with_progress,
    fake_studio,
    fast_polls,  # noqa: F401  (fixture)
    make_caller,
    queries,
    run_tool,
    sequence,
    session,
    slow,
    stt_state,
)

LLM = "unsloth/Llama-3.2-1B-Instruct-GGUF"
LOADED = {
    "status": "loaded",
    "model": LLM,
    "display_name": "Llama 3.2 1B",
    "is_gguf": True,
    "inference": {"temperature": 0.7},
    "evicted": ["unsloth/Qwen3-0.6B"],
}
STATUS = {"loaded": [LLM], "serving": [LLM], "serving_checkpoints": ["/srv/models/llama.gguf"]}
EMPTY_STATUS = {"loaded": [], "serving": [], "serving_checkpoints": []}
LOAD = ("POST", "/api/inference/load")
# A load that finished inside its wait.
SETTLED = {"status": "loaded", "fraction": None, "job_id": None, "poll_after_s": None}
STT = "/api/inference/audio/stt"
STATUS_ROUTE = ("GET", "/api/inference/status")

PAYLOADS = {
    LOAD: LOADED,
    ("GET", "/api/inference/load-progress"): {"phase": "mmap", "fraction": 0.5},
    ("GET", "/api/models/download-progress"): {
        "progress": 0.25,
        "downloaded_bytes": 1,
        "expected_bytes": 4,
    },
    ("POST", "/api/inference/unload"): {"status": "unloaded", "model": LLM},
    STATUS_ROUTE: STATUS,
}


@pytest.fixture(autouse = True)
def _no_loads_left():
    # A job left running by one test would be followed by the next one loading the same model.
    loading.LOADS.reset()
    getattr(loading, "_MEDIA_TOKENS", {}).clear()
    yield
    loading.LOADS.reset()
    getattr(loading, "_MEDIA_TOKENS", {}).clear()


# tts / audio load through the LLM route but report the kind asked for. STATUS serves LLM
# before and after, so only the route's own evicted list counts.
@pytest.mark.parametrize("kind", [None, "tts", "audio"])
def test_load_sends_the_body_and_reports_the_result(monkeypatch, kind):
    args = {"model": LLM, "variant": "Q4_K_M", **({"kind": kind} if kind else {})}
    result, studio = run_tool(monkeypatch, PAYLOADS, "load_model", args)
    assert result["structuredContent"] == {
        "kind": kind or "llm",
        "model": LLM,
        "loaded": True,
        **SETTLED,
        "display_name": "Llama 3.2 1B",
        "placement": None,
        "alongside_applied": None,
        "evicted": ["unsloth/Qwen3-0.6B"],
        "evicted_slots": [{"kind": "llm", "model": "unsloth/Qwen3-0.6B"}],
    }
    assert bodies(studio, "/api/inference/load") == [
        {"model_path": LLM, "load_in_4bit": True, "max_seq_length": 0, "gguf_variant": "Q4_K_M"}
    ]


@pytest.mark.parametrize(
    "kind,route_evicted,before,after,expected",
    [
        # An audio engine switch unloads the chat model without the route saying so.
        (
            "tts",
            [],
            {**STATUS, "serving": ["unsloth/Qwen3-0.6B-GGUF"]},
            EMPTY_STATUS,
            ["unsloth/Qwen3-0.6B-GGUF"],
        ),
        # The route keys a dropped GGUF as repo:variant while status serves the bare repo id.
        (
            "llm",
            ["unsloth/Qwen3-8B-GGUF:Q4_K_M"],
            {**STATUS, "serving": [LLM, "unsloth/Qwen3-8B-GGUF"]},
            STATUS,
            ["unsloth/Qwen3-8B-GGUF:Q4_K_M"],
        ),
    ],
)
def test_evicted_merges_the_route_list_and_status(
    monkeypatch, kind, route_evicted, before, after, expected
):
    routes = {
        **PAYLOADS,
        LOAD: {**LOADED, "evicted": route_evicted},
        STATUS_ROUTE: sequence(before, after),
    }
    result, _studio = run_tool(monkeypatch, routes, "load_model", {"model": LLM, "kind": kind})
    assert result["structuredContent"]["evicted"] == expected


@pytest.mark.parametrize(
    "args,headers,token",
    [
        ({"hf_token": "hf_param", "load_in_4bit": False}, None, "hf_param"),
        ({}, {"X-Unsloth-HF-Token": "hf_header"}, "hf_header"),
    ],
)
def test_the_hub_token_goes_in_the_body_not_the_header(monkeypatch, args, headers, token):
    _result, studio = run_tool(
        monkeypatch, PAYLOADS, "load_model", {"model": LLM, **args}, headers = headers
    )
    (_m, _p, sent_headers, body) = call_to(studio, "/api/inference/load")
    assert json.loads(body)["hf_token"] == token
    assert json.loads(body)["load_in_4bit"] is ("load_in_4bit" not in args)
    assert "x-unsloth-hf-token" not in sent_headers


@pytest.mark.parametrize("deferred", DEFERRED)
def test_a_padded_deferred_error_is_a_tool_error(monkeypatch, deferred):
    padded = b" " * 40 + json.dumps({"_deferred_error": deferred}).encode()
    routes = {**PAYLOADS, LOAD: Response(padded, media_type = "application/json")}
    result, _studio = run_tool(monkeypatch, routes, "load_model", {"model": LLM})
    assert result["isError"] is True
    assert deferred["detail"] in result["content"][0]["text"]
    assert f"(HTTP {deferred['status_code']})" in result["content"][0]["text"]


def test_progress_is_reported_while_the_load_runs(monkeypatch, fast_polls):
    studio = fake_studio({**PAYLOADS, LOAD: slow(LOADED, 0.3)})
    progress, result = call_with_progress(monkeypatch, studio, "load_model", {"model": LLM}, "p1")
    assert progress, result
    assert progress[0]["progressToken"] == "p1"
    assert progress[0]["progress"] == 0.5
    assert "mmap" in progress[0]["message"]
    assert result["structuredContent"]["loaded"] is True


def test_download_progress_is_used_before_the_load_starts(monkeypatch, fast_polls):
    routes = {
        **PAYLOADS,
        LOAD: slow(LOADED, 0.2),
        ("GET", "/api/inference/load-progress"): {"phase": None, "fraction": 0.0},
    }
    _result, studio = run_tool(monkeypatch, routes, "load_model", {"model": LLM})
    downloads = [c for c in studio.state.calls if c[1] == "/api/models/download-progress"]
    assert downloads
    assert all(c[2].get("x-unsloth-hf-token") is None for c in downloads)


# Without a model the serving one is unloaded; a public id maps to its checkpoint. An unload
# that matched nothing, or left the model resident (STATUS still serves LLM), is not success.
@pytest.mark.parametrize(
    "args,statuses,model,sent",
    [
        ({}, (STATUS, EMPTY_STATUS), LLM, [{"model_path": "/srv/models/llama.gguf"}]),
        ({"model": LLM}, (STATUS, EMPTY_STATUS), LLM, [{"model_path": "/srv/models/llama.gguf"}]),
        ({"model": "unsloth/not-loaded"}, (STATUS,), "unsloth/not-loaded", None),
        ({"model": LLM}, (STATUS,), LLM, None),
        ({}, (EMPTY_STATUS,), None, []),
    ],
)
def test_unload_picks_the_serving_checkpoint_and_checks_the_result(
    monkeypatch, args, statuses, model, sent
):
    routes = {**PAYLOADS, STATUS_ROUTE: sequence(*statuses)}
    result, studio = run_tool(monkeypatch, routes, "unload_model", args)
    assert result["structuredContent"] == {"kind": "llm", "model": model, "unloaded": bool(sent)}
    if sent is not None:
        assert bodies(studio, "/api/inference/unload") == sent


FLUX_GGUF = "city96/FLUX.1-schnell-gguf"
LTX = "Lightricks/LTX-Video"
QUANTS = ("Q4_K_S", "Q8_0")
READY = {"phase": "ready"}


def _media_studio(
    kind,
    *,
    progress = ({"phase": "ready"},),
    statuses = ({"loaded": False},),
    load_answer = None,
    plan = None,
    chat_statuses = None,
):
    """``progress`` and ``statuses`` are consumed in order; the last one repeats."""
    prefix = "/api/inference/images" if kind == "image" else "/api/inference/video"
    return fake_studio(
        {
            **({STATUS_ROUTE: sequence(*chat_statuses)} if chat_statuses else {}),
            ("GET", "/api/hub/gguf-variants"): {
                "repo_id": FLUX_GGUF,
                "default_variant": "Q4_K_S",
                "variants": [{"filename": f"flux1-schnell-{q}.gguf", "quant": q} for q in QUANTS],
            },
            ("POST", f"{prefix}/download-plan"): plan or {"entries": [], "plan_failed": False},
            ("POST", f"{prefix}/load"): load_answer or {"loaded": False},
            ("GET", f"{prefix}/load-progress"): sequence(*progress),
            ("GET", f"{prefix}/status"): sequence(*statuses),
            ("POST", f"{prefix}/unload"): {"loaded": False},
        }
    )


# No variant named: the default is used, and the hf_token argument reaches the (gated) lookup.
@pytest.mark.parametrize(
    "extra,filename",
    [
        ({"variant": "Q8_0"}, "flux1-schnell-Q8_0.gguf"),
        ({"hf_token": "hf_arg"}, "flux1-schnell-Q4_K_S.gguf"),
    ],
)
def test_a_gguf_image_repo_sends_its_gguf_filename(monkeypatch, fast_polls, extra, filename):
    studio = _media_studio(
        "image",
        progress = [{"phase": "downloading", "bytes_downloaded": 1, "bytes_total": 2}, READY],
        statuses = [{"loaded": True, "repo_id": FLUX_GGUF, "gguf_filename": filename}],
    )
    result, _studio = run_tool(
        monkeypatch, studio, "load_model", {"model": FLUX_GGUF, "kind": "image", **extra}
    )
    loaded = {"kind": "image", "model": FLUX_GGUF, "loaded": True, **SETTLED, "display_name": None}
    loaded.update(placement = None, alongside_applied = None)
    assert result["structuredContent"] == {**loaded, "evicted": [], "evicted_slots": []}
    expected = {"model_path": FLUX_GGUF, "gguf_filename": filename, "model_kind": "gguf"}
    expected.update({"hf_token": extra["hf_token"]} if "hf_token" in extra else {})
    assert bodies(studio, "/api/inference/images/download-plan") == [expected]
    assert media_load_bodies(studio, "/api/inference/images/load") == [expected]
    lookup_token = call_to(studio, "/api/hub/gguf-variants")[2].get("x-unsloth-hf-token")
    assert lookup_token == extra.get("hf_token")


def test_a_plan_entry_supplies_the_checkpoint_file(monkeypatch, fast_polls):
    entry = {"repo_id": "a/b", "files": [], "bytes": 1, "checkpoint": True}
    plan = {"entries": [{**entry, "gguf_filename": "model.safetensors"}]}
    studio = _media_studio("image", statuses = [{"loaded": True, "repo_id": "a/b"}], plan = plan)
    run_tool(monkeypatch, studio, "load_model", {"model": "a/b", "kind": "image"})
    assert media_load_bodies(studio, "/api/inference/images/load") == [
        {"model_path": "a/b", "gguf_filename": "model.safetensors"}
    ]
    assert [c for c in studio.state.calls if c[1] == "/api/hub/gguf-variants"] == []


def test_a_video_checkpoint_with_a_dtype_in_its_name_still_loads(monkeypatch, fast_polls):
    # Video status publishes no file and no quant for a plain checkpoint.
    entry = {"checkpoint": True, "gguf_filename": "ltx-video-2b-v0.9.5-bf16.safetensors"}
    studio = _media_studio(
        "video",
        statuses = [{"loaded": True, "repo_id": LTX, "model_kind": "single_file"}],
        plan = {"entries": [entry]},
    )
    result, _studio = run_tool(monkeypatch, studio, "load_model", {"model": LTX, "kind": "video"})
    assert result["structuredContent"]["loaded"] is True


PREVIOUS = {"loaded": True, "repo_id": "old/model"}
AB = {"model": "a/b", "kind": "image"}
DISK_FULL = [{"phase": "downloading"}, {"phase": "error", "error": "Out of disk space"}]


@pytest.mark.parametrize(
    "args,studio_kwargs,message",
    [
        ({**AB, "model": FLUX_GGUF, "variant": "Q2_K"}, {}, "Q2_K"),
        (
            AB,
            {"plan": {"entries": [], "incompatible_reason": "Needs a CUDA GPU"}},
            "Needs a CUDA GPU",
        ),
        (AB, {"progress": DISK_FULL}, "Out of disk space"),
        # The previous model in the load answer is not success.
        (
            {**AB, "model": "new/m"},
            {"statuses": [PREVIOUS], "load_answer": PREVIOUS},
            "loading new/m",
        ),
        ({"model": LTX, "kind": "video"}, {"progress": [{}], "statuses": [PREVIOUS]}, "stopped"),
    ],
)
def test_a_media_load_failure_is_a_tool_error(
    monkeypatch, fast_polls, args, studio_kwargs, message
):
    studio = _media_studio(args["kind"], **studio_kwargs)
    result, _studio = run_tool(monkeypatch, studio, "load_model", args)
    assert result["isError"] is True
    assert message in result["content"][0]["text"]
    if "variant" in args:
        assert media_load_bodies(studio, "/api/inference/images/load") == []


def test_a_previous_model_answer_then_the_new_one_loads(monkeypatch, fast_polls):
    studio = _media_studio(
        "image",
        progress = [{"phase": None}, {"phase": "downloading"}, READY],
        statuses = [PREVIOUS, {"loaded": True, "repo_id": "new/model"}],
        load_answer = PREVIOUS,
    )
    result, _studio = run_tool(
        monkeypatch, studio, "load_model", {"model": "new/model", "kind": "image"}
    )
    assert result["structuredContent"]["model"] == "new/model"


# Image status names the file, video only its quant: either way the old variant is not the new one.
@pytest.mark.parametrize(
    "kind,old,new,filename",
    [
        (
            "image",
            {"gguf_filename": "flux1-schnell-Q4_K_S.gguf"},
            {"gguf_filename": "flux1-schnell-Q8_0.gguf"},
            None,
        ),
        ("video", {"gguf_variant": "Q4_K_S"}, {"gguf_variant": "Q8_0"}, None),
        # Same quant, other MiniMax-H3 partition: only h3_task tells them apart.
        (
            "video",
            {"gguf_variant": "Q8_0", "h3_task": "fl2va"},
            {"gguf_variant": "Q8_0", "h3_task": "ref2va"},
            "minimax_h3_ref2va-Q8_0.gguf",
        ),
        # Same quant, other bpw build: status cannot tell them apart before the load is ready.
        ("video", {"gguf_variant": "IQ4_XS"}, {"gguf_variant": "IQ4_XS"}, "m-IQ4_XS-3.97bpw.gguf"),
    ],
)
def test_another_variant_of_the_resident_repo_waits_for_its_own_load(
    monkeypatch, fast_polls, kind, old, new, filename
):
    if filename:

        async def pick(*_args):
            # The variant row, as _gguf_filename returns it here.
            return {"filename": filename}

        monkeypatch.setattr(loading, "_gguf_filename", pick)
    resident = {"loaded": True, "repo_id": FLUX_GGUF, "model_kind": "gguf"}
    old, unnamed, new = {**resident, **old}, {**resident, "gguf_variant": None}, {**resident, **new}
    studio = _media_studio(
        kind,
        progress = [{"phase": None}, {"phase": None}, READY],
        # Read before the load, at each idle poll, then once it is ready.
        statuses = [old, old, unnamed, new],
    )
    result, _studio = run_tool(
        monkeypatch, studio, "load_model", {"model": FLUX_GGUF, "kind": kind, "variant": "Q8_0"}
    )
    assert result["structuredContent"]["loaded"] is True
    # The old variant and a status naming no variant are both "not loaded yet".
    assert len([c for c in studio.state.calls if c[1].endswith("/load-progress")]) == 3


@pytest.mark.parametrize("kind,model", [("image", FLUX_GGUF), ("video", LTX)])
def test_a_media_load_reports_the_chat_model_the_gpu_arbiter_unloaded(
    monkeypatch, fast_polls, kind, model
):
    studio = _media_studio(
        kind,
        # The default variant, as the Hub listing names it.
        statuses = [{"loaded": True, "repo_id": model, "gguf_variant": "Q4_K_S"}],
        chat_statuses = [STATUS, EMPTY_STATUS],
    )
    result, _studio = run_tool(monkeypatch, studio, "load_model", {"model": model, "kind": kind})
    assert result["structuredContent"]["evicted"] == [LLM]


def test_video_progress_uses_its_own_field_names(monkeypatch, fast_polls):
    studio = _media_studio(
        "video",
        progress = [
            {"phase": "downloading", "downloaded_bytes": 1, "expected_bytes": 4},
            {"phase": "finalizing", "downloaded_bytes": 4, "expected_bytes": 4},
            {"phase": "ready"},
        ],
        statuses = [{"loaded": True, "repo_id": LTX}],
    )
    progress, result = call_with_progress(
        monkeypatch, studio, "load_model", {"model": LTX, "kind": "video"}
    )
    assert [p["progress"] for p in progress] == [0.25, 1.0]
    assert progress[0]["message"] == "Downloading"
    assert result["structuredContent"]["model"] == LTX
    assert media_load_bodies(studio, "/api/inference/video/load") == [{"model_path": LTX}]


@pytest.mark.parametrize(
    "kind,statuses,model",
    [
        ("image", [{"loaded": True, "repo_id": LTX}], LTX),
        ("video", [{"loaded": True, "repo_id": LTX}], LTX),
        ("image", [{"loaded": False}], None),
    ],
)
def test_media_unload_takes_no_body(monkeypatch, kind, statuses, model):
    studio = _media_studio(kind, statuses = statuses)
    result, _studio = run_tool(monkeypatch, studio, "unload_model", {"kind": kind})
    assert result["structuredContent"] == {
        "kind": kind,
        "model": model,
        "unloaded": model is not None,
    }
    unloads = [c[3] for c in studio.state.calls if c[1].endswith("/unload")]
    assert unloads == ([b""] if model else [])


def _stt_studio(
    statuses,
    *,
    download = None,
    extra = None,
):
    return fake_studio(
        {
            ("GET", f"{STT}/status"): sequence(*statuses),
            ("POST", f"{STT}/download"): download
            or {"downloading": True, "model": WHISPER, "download_id": "d1"},
            ("POST", f"{STT}/load"): {"loaded_model": WHISPER, "device": "cuda"},
            ("POST", f"{STT}/unload"): {"loaded_model": None, "device": None},
            **(extra or {}),
        }
    )


# The token rides the download header (never the body), from the header or the argument.
@pytest.mark.parametrize(
    "headers,extra", [({"X-Unsloth-HF-Token": "hf_h"}, {}), (None, {"hf_token": "hf_h"})]
)
def test_stt_downloads_then_loads_in_order(monkeypatch, fast_polls, headers, extra):
    progress = {"downloading": True, "model": WHISPER, "bytes_done": 1, "bytes_total": 2}
    done = {"downloading": False, "completed_download_ids": ["d1"]}
    studio = _stt_studio(
        [
            {"transformers": stt_state()},
            {"transformers": stt_state(download = progress)},
            {"transformers": stt_state(downloaded = [WHISPER], download = done)},
        ]
    )
    result, _studio = run_tool(
        monkeypatch,
        studio,
        "load_model",
        {"model": WHISPER, "kind": "stt", **extra},
        headers = headers,
    )
    assert result["structuredContent"]["model"] == WHISPER
    assert [(m, p) for m, p, _h, _b in studio.state.calls] == [
        ("GET", f"{STT}/status"),
        ("POST", f"{STT}/download"),
        ("GET", f"{STT}/status"),
        ("GET", f"{STT}/status"),
        ("POST", f"{STT}/load"),
    ]
    expected = [{"model": WHISPER, "engine": "transformers"}]
    assert bodies(studio, f"{STT}/download") == expected
    assert call_to(studio, f"{STT}/download")[2]["x-unsloth-hf-token"] == "hf_h"
    assert bodies(studio, f"{STT}/load") == expected


def test_a_downloaded_stt_model_loads_straight_away(monkeypatch):
    q5 = "whisper-small-q5"
    studio = _stt_studio([{"gguf": {**stt_state(downloaded = [q5]), "models": [q5]}}])
    result, _studio = run_tool(monkeypatch, studio, "load_model", {"model": q5, "kind": "stt"})
    assert result["isError"] is False
    assert [c[1] for c in studio.state.calls if c[0] == "POST"] == [f"{STT}/load"]
    assert bodies(studio, f"{STT}/load") == [{"model": q5, "engine": "gguf"}]


def test_a_requested_variant_is_downloaded_even_when_another_is_cached(monkeypatch, fast_polls):
    moon = "audio-cpp/audio.cpp-gguf/Moonshine-Streaming-GGUF"
    done = {"downloading": False, "completed_download_ids": ["d1"]}
    studio = _stt_studio(
        [
            {"audiocpp": {**stt_state(downloaded = [moon]), "models": [moon]}},
            {"audiocpp": {**stt_state(downloaded = [moon, f"{moon}:small/Q8_0"]), "download": done}},
        ]
    )
    args = {"model": moon, "kind": "stt", "variant": "small/Q8_0"}
    result, _studio = run_tool(monkeypatch, studio, "load_model", args)
    assert result["isError"] is False
    expected = [{"model": moon, "engine": "audiocpp", "gguf_variant": "small/Q8_0"}]
    assert bodies(studio, f"{STT}/download") == expected
    assert bodies(studio, f"{STT}/load") == expected


def test_a_catalog_asr_row_downloads_and_loads_on_audiocpp(monkeypatch, fast_polls):
    # list_models(source="catalog") offers it; no status list names it before the download.
    citrinet = "audio-cpp/audio.cpp-gguf/Citrinet-ASR-GGUF"
    idle = {**stt_state(), "models": []}
    done = {"downloading": False, "completed_download_ids": ["d1"]}
    studio = _stt_studio(
        [
            {"transformers": stt_state(), "audiocpp": idle},
            {"transformers": stt_state(), "audiocpp": {**idle, "download": done}},
        ],
        extra = {
            ("GET", "/api/inference/audio/catalog"): {"models": [{"id": citrinet, "task": "asr"}]}
        },
    )
    result, _studio = run_tool(
        monkeypatch, studio, "load_model", {"model": citrinet, "kind": "stt"}
    )
    assert result["isError"] is False
    expected = [{"model": citrinet, "engine": "audiocpp"}]
    assert bodies(studio, f"{STT}/download") == expected
    assert bodies(studio, f"{STT}/load") == expected


def test_a_failed_download_never_loads(monkeypatch, fast_polls):
    failed = {"downloading": False, "error": "401 gated repo"}
    studio = _stt_studio(
        [{"transformers": stt_state()}, {"transformers": stt_state(download = failed)}]
    )
    result, _studio = run_tool(monkeypatch, studio, "load_model", {"model": WHISPER, "kind": "stt"})
    assert result["isError"] is True
    assert "401 gated repo" in result["content"][0]["text"]
    assert [c for c in studio.state.calls if c[1].endswith("/load")] == []


@pytest.mark.parametrize(
    "statuses,args,model,query",
    [
        (
            [
                {"transformers": stt_state(), "mtmd": stt_state(loaded = "qwen3-asr")},
                {"transformers": stt_state(), "mtmd": stt_state()},
            ],
            {},
            "qwen3-asr",
            {"engine": "mtmd", "model": "qwen3-asr", "wait": "true"},
        ),
        ([{"transformers": stt_state(loaded = WHISPER)}], {"model": "other"}, "other", None),
    ],
)
def test_stt_unload_names_the_engine_and_model_in_the_query(
    monkeypatch, statuses, args, model, query
):
    studio = _stt_studio(statuses)
    result, _studio = run_tool(monkeypatch, studio, "unload_model", {"kind": "stt", **args})
    unloaded = query is not None
    assert result["structuredContent"] == {"kind": "stt", "model": model, "unloaded": unloaded}
    assert queries(studio, f"{STT}/unload") == ([query] if query else [])
    unloads = [c[3] for c in studio.state.calls if c[1].endswith("/unload")]
    assert unloads == ([b""] if unloaded else [])


IMAGE_STATUS = ("GET", "/api/inference/images/status")
VIDEO_STATUS = ("GET", "/api/inference/video/status")
FLUX_LOADED = {"loaded": True, "repo_id": FLUX_GGUF}
LTX_LOADED = {"loaded": True, "repo_id": LTX}
UNLOADED = {"loaded": False}


def _video_over_image(load_answer, *, delay = 0.0):
    """A video load while an image model is resident; the arbiter unloads the image model."""
    return fake_studio(
        {
            ("POST", "/api/inference/video/download-plan"): {"entries": []},
            ("POST", "/api/inference/video/load"): slow(load_answer, delay),
            ("GET", "/api/inference/video/load-progress"): READY,
            # Read before the load, then by the load itself, then after it.
            VIDEO_STATUS: sequence(UNLOADED, LTX_LOADED),
            IMAGE_STATUS: sequence(FLUX_LOADED, UNLOADED),
        }
    )


@pytest.mark.parametrize("wait_s,expected", [(None, 240), (300, 240), (50, 50), (0, 0), (-5, 0)])
def test_wait_s_defaults_to_the_cap_and_is_clamped(wait_s, expected):
    assert round(loading.deadline_after(wait_s) - time.monotonic()) == expected


def test_a_slow_load_returns_early_and_get_job_returns_its_result(monkeypatch, fast_polls):
    studio = _video_over_image(UNLOADED, delay = 0.5)
    with session(monkeypatch, studio) as call:
        early = call("load_model", {"model": LTX, "kind": "video", "wait_s": 0})[
            "structuredContent"
        ]
        assert early["loaded"] is False
        assert early["status"] == "loading"
        assert early["poll_after_s"] == loading.POLL_AFTER_S
        # A repeat load_model follows the load in flight instead of starting another.
        again = call("load_model", {"model": LTX, "kind": "video", "wait_s": 0})[
            "structuredContent"
        ]
        assert again["job_id"] == early["job_id"]
        job = call("get_job", {"kind": "load", "id": early["job_id"], "wait_s": 30})
    assert job["structuredContent"]["status"] == "completed"
    assert job["structuredContent"]["progress_percent"] == 100.0
    assert job["structuredContent"]["poll_after_s"] is None
    load = job["structuredContent"]["load"]
    assert (load["loaded"], load["status"], load["model"]) == (True, "loaded", LTX)
    assert load["evicted_slots"] == [{"kind": "image", "model": FLUX_GGUF}]
    assert len(bodies(studio, "/api/inference/video/load")) == 1


def test_a_client_that_gives_up_mid_post_does_not_lose_the_load(monkeypatch, fast_polls):
    studio = fake_studio({**PAYLOADS, LOAD: slow(LOADED, 0.3)})
    caller = make_caller(studio_app = studio)

    async def give_up() -> loading.LoadJob:
        call = asyncio.ensure_future(
            loading.load_llm(
                caller,
                None,
                model = LLM,
                variant = None,
                max_seq_length = None,
                load_in_4bit = True,
                hf_token = None,
            )
        )
        await asyncio.sleep(0.1)
        call.cancel()
        (job,) = loading.LOADS.jobs_of("owner")
        await asyncio.wait({job.task}, timeout = 5)
        return job

    job = asyncio.run(give_up())
    assert job.status == "completed"
    assert (job.result.model, job.result.loaded) == (LLM, True)


def test_a_chat_load_already_in_flight_is_followed_not_started_again(monkeypatch, fast_polls):
    loading_now = {"loading": [LLM], "serving": [], "loaded": []}
    routes = {**PAYLOADS, STATUS_ROUTE: sequence(loading_now, loading_now, STATUS)}
    progress, result = call_with_progress(
        monkeypatch, fake_studio(routes), "load_model", {"model": LLM}
    )
    assert result["structuredContent"]["loaded"] is True
    assert result["structuredContent"]["model"] == LLM
    assert progress and "mmap" in progress[0]["message"]


# The load answer's evicted_slots come first, else its evicted placed by what was resident; a
# before/after diff adds what the answer leaves out (here the image model) or, on load routes from
# before evicted_slots, finds everything.
@pytest.mark.parametrize(
    "answer,expected",
    [
        (
            {
                "evicted": [LLM],
                "evicted_slots": [
                    {"kind": "chat", "model": LLM},
                    {"kind": "audio", "model": "a/tts"},
                ],
            },
            [
                {"kind": "llm", "model": LLM},
                {"kind": "audio", "model": "a/tts"},
                {"kind": "image", "model": FLUX_GGUF},
            ],
        ),
        ({"evicted": [FLUX_GGUF]}, [{"kind": "image", "model": FLUX_GGUF}]),
        ({}, [{"kind": "image", "model": FLUX_GGUF}]),
    ],
)
def test_evicted_slots_read_the_answer_then_fall_back(monkeypatch, fast_polls, answer, expected):
    studio = _video_over_image({**UNLOADED, **answer})
    result, _studio = run_tool(monkeypatch, studio, "load_model", {"model": LTX, "kind": "video"})
    assert result["structuredContent"]["evicted_slots"] == expected
    assert result["structuredContent"]["evicted"] == [slot["model"] for slot in expected]


def test_a_tts_load_reports_the_video_model_it_unloaded(monkeypatch):
    routes = {
        **PAYLOADS,
        LOAD: {**LOADED, "evicted": []},
        STATUS_ROUTE: EMPTY_STATUS,
        VIDEO_STATUS: sequence(LTX_LOADED, UNLOADED),
    }
    result, _studio = run_tool(monkeypatch, routes, "load_model", {"model": LLM, "kind": "tts"})
    assert result["structuredContent"]["kind"] == "tts"
    assert result["structuredContent"]["evicted_slots"] == [{"kind": "video", "model": LTX}]


@pytest.mark.parametrize("kind", ["tts", "audio"])
def test_unload_echoes_the_kind_asked_for(monkeypatch, kind):
    routes = {**PAYLOADS, STATUS_ROUTE: sequence(STATUS, EMPTY_STATUS)}
    result, _studio = run_tool(monkeypatch, routes, "unload_model", {"kind": kind})
    assert result["structuredContent"] == {"kind": kind, "model": LLM, "unloaded": True}


def test_saving_the_chat_context_counts_as_progress(monkeypatch, fast_polls):
    posted = []

    async def load(request, body):
        await asyncio.sleep(0.2)
        posted.append(True)
        return UNLOADED

    studio = fake_studio(
        {
            ("POST", "/api/inference/video/download-plan"): {"entries": []},
            ("POST", "/api/inference/video/load"): load,
            ("GET", "/api/inference/video/load-progress"): lambda request, body: (
                READY if posted else {"phase": "saving_chat_context"}
            ),
            VIDEO_STATUS: LTX_LOADED,
        }
    )
    progress, result = call_with_progress(
        monkeypatch, studio, "load_model", {"model": LTX, "kind": "video"}
    )
    assert result["structuredContent"]["loaded"] is True
    assert [p["message"] for p in progress] == ["saving_chat_context"]


def test_a_chat_load_asking_for_a_variant_never_follows_another_load(monkeypatch, fast_polls):
    # The status names only the repo, so another variant of it loading elsewhere is not this load.
    loading_now = {"loading": [LLM], "serving": [], "loaded": []}
    routes = {**PAYLOADS, STATUS_ROUTE: sequence(loading_now, STATUS)}
    _result, studio = run_tool(monkeypatch, routes, "load_model", {"model": LLM, "variant": "Q8_0"})
    assert len(bodies(studio, "/api/inference/load")) == 1


def test_one_failed_status_read_does_not_end_a_followed_load(monkeypatch, fast_polls):
    loading_now = {"loading": [LLM], "serving": [], "loaded": []}
    busy = Response(status_code = 503)
    routes = {**PAYLOADS, STATUS_ROUTE: sequence(loading_now, busy, loading_now, STATUS)}
    result, studio = run_tool(monkeypatch, routes, "load_model", {"model": LLM})
    assert result["structuredContent"]["loaded"] is True
    assert bodies(studio, "/api/inference/load") == []


def test_a_listed_chat_drop_does_not_hide_a_media_eviction(monkeypatch):
    routes = {
        **PAYLOADS,
        STATUS_ROUTE: sequence({**STATUS, "serving": [LLM, "unsloth/Qwen3-0.6B"]}, STATUS),
        VIDEO_STATUS: sequence(LTX_LOADED, UNLOADED),
    }
    result, _studio = run_tool(monkeypatch, routes, "load_model", {"model": LLM})
    assert result["structuredContent"]["evicted_slots"] == [
        {"kind": "llm", "model": "unsloth/Qwen3-0.6B"},
        {"kind": "video", "model": LTX},
    ]


def test_progress_never_drops_back():
    job = loading.LoadJob(job_id = "j", account_id = "owner")
    job.note(0.5, "Downloading")
    job.note(None, "saving_chat_context")
    assert (job.fraction, job.phase, job.marks) == (
        0.5,
        "saving_chat_context",
        [(0.5, "Downloading")],
    )


def test_the_model_a_load_replaced_is_reported_beside_what_the_route_lists(monkeypatch):
    # The route lists the image model the arbiter unloaded, not the chat model this load replaced.
    answer = {
        **LOADED,
        "evicted": [FLUX_GGUF],
        "evicted_slots": [{"kind": "image", "model": FLUX_GGUF}],
    }
    routes = {
        **PAYLOADS,
        LOAD: answer,
        STATUS_ROUTE: sequence({**STATUS, "serving": ["unsloth/Qwen3-0.6B"], "loaded": []}, STATUS),
        IMAGE_STATUS: sequence(FLUX_LOADED, UNLOADED),
    }
    result, _studio = run_tool(monkeypatch, routes, "load_model", {"model": LLM})
    assert result["structuredContent"]["evicted_slots"] == [
        {"kind": "image", "model": FLUX_GGUF},
        {"kind": "llm", "model": "unsloth/Qwen3-0.6B"},
    ]


def test_a_speech_to_text_load_never_follows_a_chat_load(monkeypatch, fast_polls):
    chat = {"loading": [WHISPER], "serving": [], "loaded": []}
    studio = fake_studio(
        {
            STATUS_ROUTE: chat,
            ("GET", f"{STT}/status"): {"transformers": stt_state(downloaded = [WHISPER])},
            ("POST", f"{STT}/load"): {"loaded_model": WHISPER},
        }
    )
    result, studio = run_tool(monkeypatch, studio, "load_model", {"model": WHISPER, "kind": "stt"})
    assert result["structuredContent"]["kind"] == "stt"
    assert len(bodies(studio, f"{STT}/load")) == 1


def test_unloading_stt_stops_a_load_still_downloading(monkeypatch, fast_polls):
    # cancel has no load kind, so unload_model is how an agent stops a load it left running.
    downloading = {"downloading": True, "bytes_done": 1, "bytes_total": 10}
    studio = fake_studio(
        {
            ("GET", f"{STT}/status"): {"transformers": stt_state(download = downloading)},
            ("POST", f"{STT}/download"): {"downloading": True, "download_id": "d1"},
            ("POST", f"{STT}/download/cancel"): {"downloading": False, "cancelled": True},
            ("POST", f"{STT}/load"): {"loaded_model": WHISPER},
        }
    )
    with session(monkeypatch, studio) as call:
        early = call("load_model", {"model": WHISPER, "kind": "stt", "wait_s": 0})
        assert early["structuredContent"]["status"] == "loading"
        job_id = early["structuredContent"]["job_id"]
        unloaded = call("unload_model", {"kind": "stt"})["structuredContent"]
        job = call("get_job", {"kind": "load", "id": job_id})
    assert unloaded == {"kind": "stt", "model": WHISPER, "unloaded": True}
    assert job["structuredContent"]["status"] == "cancelled"
    assert bodies(studio, f"{STT}/download/cancel") == [
        {"model": WHISPER, "engine": "transformers", "download_id": "d1"}
    ]
    assert bodies(studio, f"{STT}/load") == []


class _TokenVideo:
    """The video routes, keeping the load-token contract of core/inference/media_load_ids.py.

    A load registers ``admit_after`` seconds into its POST (the arbiter freeing the GPU) and stays in
    flight until ``finish()``. An unload naming a token ends that load only while it is the last one
    admitted, answers ``superseded`` for one admitted earlier, and refuses one not admitted yet. A
    load sent without a token gets ``studio-N``; status names the last admitted load as ``load_id``.
    A load posted while another is in flight is refused with a 409, as Studio does.
    """

    def __init__(
        self,
        *,
        admit_after: float = 0.0,
        unload_s: float = 0.0,
        answer_status: int = 200,
    ):
        self.admit_after, self.unload_s, self.answer_status = admit_after, unload_s, answer_status
        self.current = None  # the last admitted load: [model, token, phase]
        self.resident = None
        self.seen: set = set()
        self.cancelled: set = set()
        self.events: list = []
        # Called while an unload is out, before it answers (another client acting meanwhile).
        self.during_unload = None
        self.stall_status = False
        self.status_s = 0.0
        self.made = 0

    def admit(self, model, token) -> bool:
        if token in self.cancelled:
            return False
        if token is None:
            self.made += 1
            token = f"studio-{self.made}"
        self.current = [model, token, "loading"]
        self.seen.add(token)
        self.events.append(("admitted", token))
        return True

    def finish(self) -> None:
        self.current[2] = "ready"
        self.resident = self.current[0]

    async def load(self, request, body):
        sent = json.loads(body)
        await asyncio.sleep(self.admit_after)
        if self.current is not None and self.current[2] == "loading":
            self.events.append(("busy", sent.get("load_token")))
            return JSONResponse({"detail": "A video load is already in progress."}, status_code = 409)
        if not self.admit(sent["model_path"], sent.get("load_token")):
            self.events.append(("refused", sent.get("load_token")))
            return JSONResponse(
                {"detail": "This load was cancelled by an unload before it started."},
                status_code = 409,
            )
        if self.answer_status != 200:
            return JSONResponse({"detail": "lost"}, status_code = self.answer_status)
        return UNLOADED

    def progress(self, request, body):
        if self.current is None:
            return {"phase": None}
        return {"phase": self.current[2]}

    async def status(self, request, body):
        if self.stall_status:
            await asyncio.sleep(30)
        delay, self.status_s = self.status_s, 0.0
        await asyncio.sleep(delay)
        if not self.resident:
            return UNLOADED
        return {"loaded": True, "repo_id": self.resident, "load_id": self.current[1]}

    async def unload(self, request, body):
        expected = json.loads(body).get("expected_load_id") if body else None
        self.events.append(("unload", expected))
        await asyncio.sleep(self.unload_s)
        if self.during_unload is not None:
            self.during_unload()
        self.events.append(("unload answered", expected))
        if expected is None or (self.current is not None and self.current[1] == expected):
            self.current = self.resident = None
            return {**UNLOADED, **({"unloaded": True, "outcome": "unloaded"} if expected else {})}
        if expected in self.seen:
            return {"unloaded": False, "outcome": "superseded"}
        self.cancelled.add(expected)
        return {"unloaded": False, "outcome": "cancelled"}

    def app(self):
        return fake_studio(
            {
                ("POST", "/api/inference/video/download-plan"): {"entries": []},
                ("POST", "/api/inference/video/load"): self.load,
                ("GET", "/api/inference/video/load-progress"): self.progress,
                VIDEO_STATUS: self.status,
                IMAGE_STATUS: UNLOADED,
                ("POST", "/api/inference/video/unload"): self.unload,
            }
        )


async def _video_load(caller):
    # Returns at once (no wait), leaving the load running in the job store, once its POST is out.
    result = await loading.load_media(
        caller,
        None,
        kind = "video",
        model = LTX,
        variant = None,
        hf_token = None,
        deadline = time.monotonic(),
    )
    job = [job for job in loading.LOADS.jobs_of(caller.account_id) if job.slot == "video"][-1]
    while job.post is None and not job.task.done():
        await asyncio.sleep(0.01)
    return result


async def _late_unloads_done():
    loop = asyncio.get_running_loop()
    while held := {t for t in loading._LATE_UNLOADS if t.get_loop() is loop and not t.done()}:
        await asyncio.wait(held, timeout = 5)


def _tokens_sent(backend):
    return [token for event, token in backend.events if event == "admitted" and token]


def test_unloading_media_stops_a_load_still_running(monkeypatch, fast_polls):
    # Nothing is resident yet: the unload names the load, which ends it in flight.
    backend = _TokenVideo()
    studio = backend.app()
    with session(monkeypatch, studio) as call:
        early = call("load_model", {"model": LTX, "kind": "video", "wait_s": 0})
        job_id = early["structuredContent"]["job_id"]
        unloaded = call("unload_model", {"kind": "video"})["structuredContent"]
        job = call("get_job", {"kind": "load", "id": job_id})
    assert unloaded == {"kind": "video", "model": LTX, "unloaded": True}
    assert job["structuredContent"]["status"] == "cancelled"
    (token,) = _tokens_sent(backend)
    assert bodies(studio, "/api/inference/video/unload") == [{"expected_load_id": token}]
    assert backend.current is None


def test_a_media_load_post_that_outlasts_the_unload_never_registers(monkeypatch, fast_polls):
    # The unload reaches Studio before the load POST registers its load: the load is refused on
    # arrival, and the unload never waits for the POST.
    backend = _TokenVideo(admit_after = 0.6)
    caller = make_caller(studio_app = backend.app())

    async def scenario():
        result = await _video_load(caller)
        began = time.monotonic()
        unloaded = await loading.unload_media(caller, "video")
        took = time.monotonic() - began
        (job,) = loading.LOADS.jobs_of("owner")
        await asyncio.wait({job.post}, timeout = 5)
        await _late_unloads_done()
        return result, unloaded, took

    result, unloaded, took = asyncio.run(scenario())
    assert result.status == "loading"
    assert took < 0.5
    assert unloaded.unloaded is False
    assert [e for e, _t in backend.events] == ["unload", "unload answered", "refused"]
    assert backend.current is None and backend.resident is None


def test_a_stopped_load_never_cancels_a_newer_load_from_the_ui(monkeypatch, fast_polls):
    backend = _TokenVideo(admit_after = 0.6)
    caller = make_caller(studio_app = backend.app())

    async def scenario():
        await _video_load(caller)
        await loading.unload_media(caller, "video")
        backend.admit(LTX, None)
        backend.finish()
        (job,) = loading.LOADS.jobs_of("owner")
        await asyncio.wait({job.post}, timeout = 5)
        await _late_unloads_done()

    asyncio.run(scenario())
    assert backend.events[-1][0] == "refused"
    assert backend.current == [LTX, "studio-1", "ready"] and backend.resident == LTX


def test_a_cancelled_media_unload_still_ends_the_load_it_stopped(monkeypatch, fast_polls):
    backend = _TokenVideo(unload_s = 0.4)
    caller = make_caller(studio_app = backend.app())

    async def scenario():
        await _video_load(caller)
        (job,) = loading.LOADS.jobs_of("owner")
        await asyncio.wait({job.post}, timeout = 5)
        unload = asyncio.ensure_future(loading.unload_media(caller, "video"))
        await asyncio.sleep(0.1)
        unload.cancel()
        await asyncio.wait({unload}, timeout = 5)
        await _late_unloads_done()

    asyncio.run(scenario())
    (token,) = _tokens_sent(backend)
    assert ("unload answered", token) in backend.events
    assert backend.current is None


def test_a_lost_load_answer_still_lets_the_token_end_the_load(monkeypatch, fast_polls):
    # The load registered but its answer failed: the job ended, the load runs on, and a later
    # unload still ends it by its token (nothing is resident yet, so status alone would not).
    backend = _TokenVideo(answer_status = 500)
    studio = backend.app()
    with session(monkeypatch, studio) as call:
        failed = call("load_model", {"model": LTX, "kind": "video", "wait_s": 30})
        assert failed["isError"] is True
        assert backend.current is not None
        unloaded = call("unload_model", {"kind": "video"})["structuredContent"]
    assert unloaded["unloaded"] is True
    (token,) = _tokens_sent(backend)
    assert bodies(studio, "/api/inference/video/unload") == [{"expected_load_id": token}]
    assert backend.current is None
    assert loading._MEDIA_TOKENS == {}


def test_a_superseded_token_leaves_the_newer_load_alone(monkeypatch, fast_polls):
    # This caller's load finished, then the UI started another; the old token ends nothing.
    backend = _TokenVideo()
    caller = make_caller(studio_app = backend.app())

    async def scenario():
        await _video_load(caller)
        (job,) = loading.LOADS.jobs_of("owner")
        await asyncio.wait({job.post}, timeout = 5)
        backend.finish()
        await asyncio.wait({job.task}, timeout = 5)
        backend.resident = None
        backend.admit(LTX, None)
        return await loading.unload_media(caller, "video")

    unloaded = asyncio.run(scenario())
    assert unloaded.unloaded is False
    assert backend.events[-1] == ("unload answered", _tokens_sent(backend)[0])
    assert backend.current == [LTX, "studio-1", "loading"]


def test_another_accounts_unload_sends_none_of_this_accounts_tokens(monkeypatch, fast_polls):
    backend = _TokenVideo()
    studio = backend.app()
    mine = make_caller(studio_app = studio)
    theirs = make_caller(studio_app = studio, account_id = "acct-b")

    async def scenario():
        await _video_load(mine)
        (job,) = loading.LOADS.jobs_of("owner")
        await asyncio.wait({job.post}, timeout = 5)
        return await loading.unload_media(theirs, "video")

    unloaded = asyncio.run(scenario())
    assert unloaded.unloaded is False
    assert bodies(studio, "/api/inference/video/unload") == []
    assert backend.current is not None


def test_a_media_load_posts_after_an_unload_already_sent(monkeypatch, fast_polls):
    # An unload of the slot still out would cancel a load registered before it ran.
    backend = _TokenVideo(unload_s = 0.4)
    backend.admit(LTX, None)
    backend.finish()
    caller = make_caller(studio_app = backend.app())

    async def scenario():
        unload = asyncio.ensure_future(loading.unload_media(caller, "video"))
        await asyncio.sleep(0.1)
        await _video_load(caller)
        await asyncio.wait({unload}, timeout = 5)
        (job,) = loading.LOADS.jobs_of("owner")
        await asyncio.wait({job.post}, timeout = 5)

    asyncio.run(scenario())
    events = [e for e, _t in backend.events]
    assert events == ["admitted", "unload", "unload answered", "admitted"]
    assert backend.current is not None and backend.current[2] == "loading"


def test_the_resident_unload_names_the_load_status_showed(monkeypatch, fast_polls):
    # While this caller's token unload is out, the UI starts another load over the resident one.
    # The unload of the resident names the load status showed, so it leaves that newer load alone.
    backend = _TokenVideo(unload_s = 0.2)
    caller = make_caller(studio_app = backend.app())

    async def scenario():
        await _video_load(caller)
        (job,) = loading.LOADS.jobs_of("owner")
        await asyncio.wait({job.post}, timeout = 5)
        backend.finish()
        await asyncio.wait({job.task}, timeout = 5)
        # The UI's load replaces it and is resident; this caller's token is now stale.
        backend.admit(LTX, None)
        backend.finish()

        def newer_load():
            if backend.current[1] == "studio-1":
                backend.admit(LTX, None)

        backend.during_unload = newer_load
        return await loading.unload_media(caller, "video")

    unloaded = asyncio.run(scenario())
    assert unloaded.unloaded is False
    token = _tokens_sent(backend)[0]
    assert [e for e in backend.events if e[0] == "unload"] == [
        ("unload", token),
        ("unload", "studio-1"),
    ]
    assert backend.current == [LTX, "studio-2", "loading"]


def test_the_resident_unload_unloads_what_status_showed(monkeypatch, fast_polls):
    backend = _TokenVideo()
    backend.admit(LTX, None)
    backend.finish()
    studio = backend.app()
    with session(monkeypatch, studio) as call:
        unloaded = call("unload_model", {"kind": "video"})["structuredContent"]
    assert unloaded == {"kind": "video", "model": LTX, "unloaded": True}
    assert bodies(studio, "/api/inference/video/unload") == [{"expected_load_id": "studio-1"}]
    assert backend.current is None


def test_a_live_token_outlasts_any_number_of_refused_loads(monkeypatch, fast_polls):
    # The first load registered but its answer failed, so it runs on; the retries are refused while it
    # does. Its token must still be named by the unload.
    backend = _TokenVideo(answer_status = 500)
    caller = make_caller(studio_app = backend.app())

    async def scenario():
        for _ in range(10):
            await _video_load(caller)
            (job,) = [j for j in loading.LOADS.jobs_of("owner") if not j.task.done()] or [None]
            if job is not None:
                await asyncio.wait({job.task}, timeout = 5)
        return await loading.unload_media(caller, "video")

    unloaded = asyncio.run(scenario())
    first = _tokens_sent(backend)[0]
    assert [e for e, _t in backend.events].count("busy") == 9
    assert unloaded.unloaded is True
    assert ("unload answered", first) in backend.events
    assert backend.current is None
    assert loading._MEDIA_TOKENS == {}


def test_a_load_shown_ready_retires_the_tokens_before_it(monkeypatch, fast_polls):
    backend = _TokenVideo()
    caller = make_caller(studio_app = backend.app())

    async def scenario():
        for _ in range(2):
            await _video_load(caller)
            (job,) = [j for j in loading.LOADS.jobs_of("owner") if not j.task.done()]
            await asyncio.wait({job.post}, timeout = 5)
            backend.finish()
            await asyncio.wait({job.task}, timeout = 5)

    asyncio.run(scenario())
    first, second = _tokens_sent(backend)
    assert list(loading._MEDIA_TOKENS[("owner", "video")]) == [second]


def test_a_stalled_status_read_still_ends_the_stopped_load(monkeypatch, fast_polls):
    monkeypatch.setattr(loading, "READ_S", 0.2)
    backend = _TokenVideo()
    caller = make_caller(studio_app = backend.app())

    async def scenario():
        await _video_load(caller)
        (job,) = loading.LOADS.jobs_of("owner")
        await asyncio.wait({job.post}, timeout = 5)
        backend.stall_status = True
        began = time.monotonic()
        with pytest.raises(ToolError, match = "did not answer the video status read"):
            await loading.unload_media(caller, "video")
        return time.monotonic() - began

    took = asyncio.run(scenario())
    assert took < 5
    (token,) = _tokens_sent(backend)
    assert ("unload answered", token) in backend.events
    assert backend.current is None


def test_a_load_started_during_the_unload_status_read_waits_for_the_unload(monkeypatch, fast_polls):
    # The status read is slow and this process starts a load of the slot meanwhile: that load posts
    # only once the unload answered, so status never names it and the unload never ends it.
    backend = _TokenVideo()
    backend.admit(LTX, None)
    backend.finish()
    caller = make_caller(studio_app = backend.app())

    async def scenario():
        backend.status_s = 0.3
        unload = asyncio.ensure_future(loading.unload_media(caller, "video"))
        await asyncio.sleep(0.05)
        load = asyncio.ensure_future(_video_load(caller))
        result = await unload
        await load
        (job,) = loading.LOADS.jobs_of("owner")
        await asyncio.wait({job.post}, timeout = 5)
        return result

    result = asyncio.run(scenario())
    assert result.unloaded is True
    (token,) = [t for t in _tokens_sent(backend) if not t.startswith("studio-")]
    events = [e for e in backend.events if e[0] in ("unload answered", "admitted")]
    assert events == [
        ("admitted", "studio-1"),
        ("unload answered", "studio-1"),
        ("admitted", token),
    ]
    assert backend.current == [LTX, token, "loading"]


def test_unloading_stt_lets_a_load_past_its_download_finish_then_unloads_it(
    monkeypatch, fast_polls
):
    # Cancelling the in-process load route would not stop the sidecar load, so it is waited for.
    studio = fake_studio(
        {
            ("GET", f"{STT}/status"): sequence(
                {"transformers": stt_state(downloaded = [WHISPER])},
                {"transformers": stt_state(downloaded = [WHISPER], loaded = WHISPER)},
                {"transformers": stt_state(downloaded = [WHISPER])},
            ),
            ("POST", f"{STT}/load"): slow({"loaded_model": WHISPER}, 0.5),
            ("POST", f"{STT}/unload"): {"loaded_model": None},
        }
    )
    with session(monkeypatch, studio) as call:
        early = call("load_model", {"model": WHISPER, "kind": "stt", "wait_s": 0})
        job_id = early["structuredContent"]["job_id"]
        unloaded = call("unload_model", {"kind": "stt"})["structuredContent"]
        job = call("get_job", {"kind": "load", "id": job_id})
    assert unloaded == {"kind": "stt", "model": WHISPER, "unloaded": True}
    assert job["structuredContent"]["status"] == "completed"
    assert bodies(studio, f"{STT}/download/cancel") == []
    assert len(bodies(studio, f"{STT}/load")) == 1


def test_an_stt_load_that_outlasts_the_unload_is_unloaded_once_it_finishes(monkeypatch, fast_polls):
    monkeypatch.setattr(loading, "MAX_WAIT_S", 0.05)
    on_disk = {"transformers": stt_state(downloaded = [WHISPER])}
    studio = fake_studio(
        {
            ("GET", f"{STT}/status"): sequence(
                on_disk,
                on_disk,
                {"transformers": stt_state(downloaded = [WHISPER], loaded = WHISPER)},
                on_disk,
            ),
            ("POST", f"{STT}/load"): slow({"loaded_model": WHISPER}, 0.5),
            ("POST", f"{STT}/unload"): {"loaded_model": None},
        }
    )
    with session(monkeypatch, studio) as call:
        early = call("load_model", {"model": WHISPER, "kind": "stt", "wait_s": 0})
        unloaded = call("unload_model", {"kind": "stt"})["structuredContent"]
        assert unloaded["unloaded"] is False
        deadline = time.monotonic() + 5
        while not queries(studio, f"{STT}/unload") and time.monotonic() < deadline:
            time.sleep(0.05)
        job = call("get_job", {"kind": "load", "id": early["structuredContent"]["job_id"]})
    assert job["structuredContent"]["status"] == "completed"
    assert queries(studio, f"{STT}/unload") == [
        {"engine": "transformers", "model": WHISPER, "wait": "true"}
    ]


# The resident-model snapshot and the media Hub and plan reads all run inside the job.
@pytest.mark.parametrize("slow_route", ["plan", "status"])
def test_a_slow_read_before_the_load_does_not_hold_the_call_past_wait_s(
    monkeypatch, fast_polls, slow_route
):
    studio = fake_studio(
        {
            ("POST", "/api/inference/video/download-plan"): (
                slow({"entries": []}, 1.0) if slow_route == "plan" else {"entries": []}
            ),
            ("POST", "/api/inference/video/load"): LTX_LOADED,
            ("GET", "/api/inference/video/load-progress"): READY,
            VIDEO_STATUS: sequence(UNLOADED, LTX_LOADED),
            IMAGE_STATUS: slow(UNLOADED, 1.0) if slow_route == "status" else UNLOADED,
        }
    )
    with session(monkeypatch, studio) as call:
        started = time.monotonic()
        early = call("load_model", {"model": LTX, "kind": "video", "wait_s": 0})
        assert time.monotonic() - started < 0.9
        assert early["structuredContent"]["status"] == "loading"
        job = call(
            "get_job", {"kind": "load", "id": early["structuredContent"]["job_id"], "wait_s": 30}
        )
    assert job["structuredContent"]["status"] == "completed"
    assert job["structuredContent"]["load"]["model"] == LTX


@pytest.mark.parametrize(
    "left,right,same",
    [
        ("unsloth/Q-GGUF:Q4_K_M", "unsloth/Q-GGUF", True),
        ("C:\\models\\old.gguf", "C:\\models\\new.gguf", False),
        ("C:/models/old.gguf", "C:/models/new.gguf", False),
    ],
)
def test_only_a_repo_id_drops_its_variant_when_matching(left, right, same):
    assert loading._same_id(left, right) is same


def test_a_retry_with_another_hub_token_starts_its_own_load(monkeypatch, fast_polls):
    studio = fake_studio({**PAYLOADS, LOAD: slow(LOADED, 1.0), STATUS_ROUTE: EMPTY_STATUS})
    with session(monkeypatch, studio) as call:
        first = call("load_model", {"model": LLM, "wait_s": 0, "hf_token": "hf_old"})
        same = call("load_model", {"model": LLM, "wait_s": 0, "hf_token": "hf_old"})
        retry = call("load_model", {"model": LLM, "wait_s": 0, "hf_token": "hf_new"})
    ids = [r["structuredContent"]["job_id"] for r in (first, same, retry)]
    assert ids[0] == ids[1] != ids[2]


@pytest.mark.parametrize(
    "left,right,same",
    [
        ("unsloth/Qwen3-0.6B", "Unsloth/qwen3-0.6b", True),
        ("/models/Foo.gguf", "/models/foo.gguf", False),
        ("C:\\models\\Foo.gguf", "c:\\models\\foo.gguf", False),
        ("models/Foo.gguf", "models/foo.gguf", False),
    ],
)
def test_hub_ids_match_in_any_case_and_local_paths_keep_theirs(
    tmp_path, monkeypatch, left, right, same
):
    # A relative path that exists on disk is a path, though it reads like a Hub id.
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "Foo.gguf").write_text("")
    monkeypatch.chdir(tmp_path)
    assert loading._same(left, right) is same


def test_a_status_route_that_stalls_never_holds_up_a_load(monkeypatch, fast_polls):
    # The snapshot only feeds the eviction report, so a slot that does not answer is left out.
    monkeypatch.setattr(loading, "READ_S", 0.2)
    routes = {**PAYLOADS, STATUS_ROUTE: EMPTY_STATUS, IMAGE_STATUS: slow(UNLOADED, 10)}
    started = time.monotonic()
    result, _studio = run_tool(monkeypatch, routes, "load_model", {"model": LLM, "wait_s": 30})
    assert time.monotonic() - started < 5
    assert result["structuredContent"]["loaded"] is True


def test_two_local_ref_handles_are_different_models():
    from studio_mcp import loading

    a, b = "ref:" + "1" * 32, "ref:" + "2" * 32
    assert not loading._same_id(a, b)
    assert loading._same_id(a, a)
    assert loading._same_id("org/repo:Q4_K_M", "org/repo")


def test_a_padded_keep_loaded_refusal_names_unload_model(monkeypatch):
    refusal = (
        "Loading this model would unload the chat model unsloth/Qwen3-0.6B, and Keep multiple"
        " models loaded is set to Always. Unload that model first, or choose When they fit."
    )
    padded = " " * 40 + json.dumps({"_deferred_error": {"status_code": 409, "detail": refusal}})
    routes = {**PAYLOADS, LOAD: Response(padded.encode(), media_type = "application/json")}
    result, _studio = run_tool(monkeypatch, routes, "load_model", {"model": LLM})
    message = result["content"][0]["text"]
    assert result["isError"] is True
    assert message.startswith("[multi_model_always kind=chat] Loading this model")
    assert message.endswith("Call unload_model(kind='llm') to free the GPU; retrying won't help.")
    assert "Retry after" not in message


def test_running_chats_block_a_load_with_a_wait_or_cancel_hint(monkeypatch):
    detail = {
        "error": "active_generations",
        "message": (
            "Load would stop 1 chat that is still generating. Stop them first, or retry with"
            " force_cancel_active."
        ),
        "running": 1,
        "thread_ids": ["thread-a"],
    }
    routes = {**PAYLOADS, LOAD: JSONResponse({"detail": detail}, status_code = 409)}
    result, _studio = run_tool(monkeypatch, routes, "load_model", {"model": LLM})
    message = result["content"][0]["text"]
    assert result["isError"] is True
    assert message.startswith("[active_generations running=1] Load would stop 1 chat")
    assert message.endswith(
        "Wait for the running chats to finish, or cancel a chat you started with"
        " cancel(kind='chat', id=<its cancel_id>)."
    )
    assert "thread" not in message
    assert "force_cancel_active" not in message


def _replacement_failed_studio(*, held):
    # The UI's model (token held) is resident; this caller's replacement t1 was in flight at the status
    # read and then failed, so its token unload answers superseded and the old model stays loaded.
    state = {"resident": LTX}

    def status(request, body):
        if state["resident"] is None:
            return UNLOADED
        return {"loaded": True, "repo_id": LTX, "load_id": "t1", "resident_load_id": held}

    def unload(request, body):
        expected = json.loads(body).get("expected_load_id") if body else None
        if expected is not None and expected == held and state["resident"] is not None:
            state["resident"] = None
            return {"loaded": False, "unloaded": True, "outcome": "unloaded"}
        return {"unloaded": False, "outcome": "superseded"}

    return fake_studio(
        {VIDEO_STATUS: status, ("POST", "/api/inference/video/unload"): unload}
    ), state


def test_an_unload_ends_the_model_a_failed_replacement_left_resident(monkeypatch):
    studio, state = _replacement_failed_studio(held = "ui-1")
    caller = make_caller(studio_app = studio)
    loading._remember_media_token(caller, "video", "t1")
    unloaded = asyncio.run(loading.unload_media(caller, "video"))
    assert bodies(studio, "/api/inference/video/unload") == [
        {"expected_load_id": "t1"},
        {"expected_load_id": "ui-1"},
    ]
    assert unloaded.unloaded is True and state["resident"] is None


def test_an_unload_names_no_resident_id_studio_does_not_report(monkeypatch):
    # An older Studio reports no resident_load_id: only the token unload goes out, as before.
    studio, state = _replacement_failed_studio(held = None)
    caller = make_caller(studio_app = studio)
    loading._remember_media_token(caller, "video", "t1")
    unloaded = asyncio.run(loading.unload_media(caller, "video"))
    assert bodies(studio, "/api/inference/video/unload") == [{"expected_load_id": "t1"}]
    assert unloaded.unloaded is False and state["resident"] == LTX


def _status_studio(status, answers = None):
    # Status answers `status`; each unload naming a token answers the next entry of answers[token].
    answers = {k: list(v) for k, v in (answers or {}).items()}

    def unload(request, body):
        expected = json.loads(body).get("expected_load_id") if body else None
        queue = answers.get(expected) or [{"unloaded": False, "outcome": "superseded"}]
        return queue.pop(0) if len(queue) > 1 else queue[0]

    return fake_studio({VIDEO_STATUS: status, ("POST", "/api/inference/video/unload"): unload})


def test_an_unload_sends_no_unconditional_unload_when_studio_cannot_name_the_load(monkeypatch):
    # Studio reports load ids, but this read could not name the state: an unconditional unload
    # could end a load admitted meanwhile, so none goes out.
    studio = _status_studio(
        {"loaded": True, "repo_id": LTX, "load_id": None, "resident_load_id": None}
    )
    caller = make_caller(studio_app = studio)
    with pytest.raises(ToolError, match = "could not identify"):
        asyncio.run(loading.unload_media(caller, "video"))
    assert bodies(studio, "/api/inference/video/unload") == []


def test_an_unload_of_a_model_studio_hides_keeps_the_plain_unload(monkeypatch):
    # Another account's resident (Studio hides its id and status fields): the plain unload as before,
    # which Studio refuses or performs for this caller.
    studio = _status_studio({"loaded": True, "yours": False})
    caller = make_caller(studio_app = studio)
    asyncio.run(loading.unload_media(caller, "video"))
    sent = [c[3] for c in studio.state.calls if c[1] == "/api/inference/video/unload"]
    assert len(sent) == 1 and "expected_load_id" not in (sent[0] or "")


def test_an_unload_retries_this_callers_resident_after_its_replacement_fails(monkeypatch):
    # This caller loaded A, then replacement B was in flight at the read: A's unload answers superseded
    # while B is loading, B fails and its own unload answers superseded, so A is named again after.
    studio = _status_studio(
        {"loaded": True, "repo_id": LTX, "load_id": "B", "resident_load_id": "A"},
        {
            "A": [
                {"unloaded": False, "outcome": "superseded"},
                {"loaded": False, "unloaded": True, "outcome": "unloaded"},
            ]
        },
    )
    caller = make_caller(studio_app = studio)
    loading._remember_media_token(caller, "video", "A")
    loading._remember_media_token(caller, "video", "B")
    unloaded = asyncio.run(loading.unload_media(caller, "video"))
    sent = bodies(studio, "/api/inference/video/unload")
    assert sent.count({"expected_load_id": "A"}) == 2 and {"expected_load_id": "B"} in sent
    assert unloaded.unloaded is True
