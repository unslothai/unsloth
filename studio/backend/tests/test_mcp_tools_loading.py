# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
import json
import time

import pytest
from fastapi.responses import Response

from studio_mcp import loading

from .mcp_harness import (
    DEFERRED,
    WHISPER,
    bodies,
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
    yield
    loading.LOADS.reset()


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
        statuses = [{"loaded": True, "repo_id": FLUX_GGUF, "family": "flux"}],
    )
    result, _studio = run_tool(
        monkeypatch, studio, "load_model", {"model": FLUX_GGUF, "kind": "image", **extra}
    )
    loaded = {"kind": "image", "model": FLUX_GGUF, "loaded": True, **SETTLED, "display_name": None}
    assert result["structuredContent"] == {**loaded, "evicted": [], "evicted_slots": []}
    expected = {"model_path": FLUX_GGUF, "gguf_filename": filename, "model_kind": "gguf"}
    expected.update({"hf_token": extra["hf_token"]} if "hf_token" in extra else {})
    assert bodies(studio, "/api/inference/images/download-plan") == [expected]
    assert bodies(studio, "/api/inference/images/load") == [expected]
    lookup_token = call_to(studio, "/api/hub/gguf-variants")[2].get("x-unsloth-hf-token")
    assert lookup_token == extra.get("hf_token")


def test_a_plan_entry_supplies_the_checkpoint_file(monkeypatch, fast_polls):
    entry = {"repo_id": "a/b", "files": [], "bytes": 1, "checkpoint": True}
    plan = {"entries": [{**entry, "gguf_filename": "model.safetensors"}]}
    studio = _media_studio("image", statuses = [{"loaded": True, "repo_id": "a/b"}], plan = plan)
    run_tool(monkeypatch, studio, "load_model", {"model": "a/b", "kind": "image"})
    assert bodies(studio, "/api/inference/images/load") == [
        {"model_path": "a/b", "gguf_filename": "model.safetensors"}
    ]
    assert [c for c in studio.state.calls if c[1] == "/api/hub/gguf-variants"] == []


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
        assert bodies(studio, "/api/inference/images/load") == []


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


@pytest.mark.parametrize("kind,model", [("image", FLUX_GGUF), ("video", LTX)])
def test_a_media_load_reports_the_chat_model_the_gpu_arbiter_unloaded(
    monkeypatch, fast_polls, kind, model
):
    studio = _media_studio(
        kind,
        statuses = [{"loaded": True, "repo_id": model}],
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
    assert bodies(studio, "/api/inference/video/load") == [{"model_path": LTX}]


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


def _stt_studio(statuses, *, download = None):
    return fake_studio(
        {
            ("GET", f"{STT}/status"): sequence(*statuses),
            ("POST", f"{STT}/download"): download
            or {"downloading": True, "model": WHISPER, "download_id": "d1"},
            ("POST", f"{STT}/load"): {"loaded_model": WHISPER, "device": "cuda"},
            ("POST", f"{STT}/unload"): {"loaded_model": None, "device": None},
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


def test_unloading_media_stops_a_load_still_running(monkeypatch, fast_polls):
    # Nothing is resident yet, so before this the unload answered "nothing loaded" and posted nothing.
    unload = "/api/inference/video/unload"
    studio = fake_studio(
        {
            ("POST", "/api/inference/video/download-plan"): {"entries": []},
            ("POST", "/api/inference/video/load"): slow(LTX_LOADED, 5.0),
            ("GET", "/api/inference/video/load-progress"): READY,
            VIDEO_STATUS: UNLOADED,
            IMAGE_STATUS: UNLOADED,
            ("POST", unload): UNLOADED,
        }
    )
    with session(monkeypatch, studio) as call:
        early = call("load_model", {"model": LTX, "kind": "video", "wait_s": 0})
        job_id = early["structuredContent"]["job_id"]
        unloaded = call("unload_model", {"kind": "video"})["structuredContent"]
        job = call("get_job", {"kind": "load", "id": job_id})
    assert unloaded == {"kind": "video", "model": LTX, "unloaded": True}
    assert job["structuredContent"]["status"] == "cancelled"
    assert len([c for c in studio.state.calls if c[1] == unload]) == 1


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
