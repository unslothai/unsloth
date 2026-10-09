# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Contracts every Unsloth Studio MCP tool keeps, one row per case: a refused input never reaches a route, a route's failure becomes the message the agent acts on, and only a local agent may name a host file."""

from typing import Optional

import pytest
from fastapi.responses import JSONResponse

from studio_mcp.inputs import PATH_REMOTE
from studio_mcp.outputs import RouteText, ToolOutput, Usage

from .mcp_harness import (
    CONFIG,
    DIFFUSION,
    LOCAL,
    PNG,
    PNG_URL,
    REMOTE,
    WAV,
    WAV_INPUT,
    WEBP,
    WHISPER,
    call_to,
    data_url,
    file_spy,  # noqa: F401  (fixture)
    form,
    openai_error,
    run_tool,
    stt_state,
)
from .test_mcp_tools_audio import TRANSCRIPT
from .test_mcp_tools_text import CHAT, URGENT

EXPORT = {"checkpoint": "qwen-lora", "format": "gguf"}


def _big_png(tmp_path):
    # Over the Decision API's 4 MiB per image; only a file path can carry it past the request limit.
    big = tmp_path / "big.png"
    big.write_bytes(PNG + b"\x00" * (4 * 1024 * 1024))
    return {"state": "x", "questions": URGENT, "images": [{"path": str(big)}]}


def _huge_wav(tmp_path):
    # Only a file path can carry this much: inline data stops at the 4 MiB request limit.
    huge = tmp_path / "huge.wav"
    with huge.open("wb") as handle:
        handle.truncate(200 * 1024 * 1024 + 1)
    return {"workflow": "clone", "text": "Hi", "reference": {"path": str(huge)}}


# (tool, args, a substring of the refusal or None). Args built from tmp_path name a host file,
# which only a loopback agent may send, so those rows call as one.
REFUSED = [
    ("chat", {}, None),
    ("chat", {"prompt": "a", "messages": [{"role": "user", "content": "b"}]}, None),
    ("chat", {"messages": []}, None),
    ("chat", {"messages": [{"role": "tool", "content": "x"}]}, None),
    ("chat", {"messages": [{"role": "assistant", "content": "x"}], "images": [PNG_URL]}, None),
    ("chat", {"prompt": "x", "images": [{"data_url": "data:image/gif;base64,R0lG"}]}, None),
    ("chat", {"prompt": "x", "images": [{"data_url": "data:image/png;base64,@@@"}]}, None),
    (
        "chat",
        {"prompt": "x", "images": [{"data_url": data_url(b"not an image", "image/png")}]},
        None,
    ),
    ("embed", {"texts": []}, None),
    ("embed", {"texts": ["x"] * 2049}, None),
    ("system_one", {"state": "x", "questions": {}}, None),
    (
        "system_one",
        {"state": "x", "questions": {f"q{i}": {"type": "noul"} for i in range(65)}},
        None,
    ),
    ("system_one", {"state": "x", "questions": {"q": {"type": "maybe"}}}, None),
    ("system_one", {"state": "x", "questions": {"q": {"type": "noul", "extra": 1}}}, None),
    (
        "system_one",
        {"state": "x", "questions": URGENT, "images": [{"data_url": data_url(WEBP, "image/webp")}]},
        None,
    ),
    ("system_one", {"state": "x", "questions": URGENT, "images": [PNG_URL] * 5}, None),
    ("system_one", _big_png, None),
    *[
        ("generate_audio", {"workflow": "clone", "text": "Hi", "reference": reference}, None)
        for reference in (
            {},
            {"clip_id": "a", "voice_id": "b"},
            {"data_base64": "AAAA"},
            {"clip_id": "../etc"},
        )
    ],
    ("generate_audio", _huge_wav, "larger than 200 MiB"),
    ("transcribe", {"audio": {"clip_id": "clip-7"}, "translate": True}, "under 25 MB"),
    ("generate_image", {"prompt": "x", "mask_image": PNG_URL}, "mask_image needs init_image"),
    ("generate_image", {"prompt": "x", "upscale": 2}, None),
    (
        "generate_image",
        {"prompt": "x", "init_image": PNG_URL, "reference_images": [PNG_URL] * 10},
        None,
    ),
    *[
        ("start_training", {"config": config, "validate_only": validate_only}, message)
        for config, message in (
            ({**CONFIG, "format_type": "bogus"}, "format_type must be one of"),
            (
                {"model_name": "unsloth/Qwen3-0.6B"},
                "Invalid training config: training_type: Field required",
            ),
            ({**CONFIG, "training_type": "Everything"}, "Invalid training config: training_type"),
        )
        for validate_only in (False, True)
    ],
    ("get_job", {"kind": "recipe", "id": "job-1", "rows": 0}, None),
    ("get_job", {"kind": "recipe", "id": "job-1", "rows": 501}, None),
    ("datasets", {"action": "check_format"}, "check_format needs name"),
    ("datasets", {"action": "download"}, None),
    ("datasets", {"action": "status"}, None),
    *[
        ("cancel", {"kind": kind}, "needs id")
        for kind in ("training_start", "recipe", "chat", "dataset_download")
    ],
    *[
        ("export_model", {**EXPORT, "save_directory": directory}, None)
        for directory in ("/srv/out", "C:\\out", "~/out", "a/../../etc", "..", "  ")
    ],
]


@pytest.mark.parametrize("tool,args,message", REFUSED)
def test_a_refused_input_never_reaches_a_route(monkeypatch, tmp_path, tool, args, message):
    client = LOCAL if callable(args) else {}
    args = args(tmp_path) if callable(args) else args
    # Every route answers, so any call the tool made would be recorded.
    result, studio = run_tool(monkeypatch, {}, tool, args, fallback = {}, **client)
    assert result["isError"] is True
    if message is not None:
        assert message in result["content"][0]["text"]
    assert studio.state.calls == []


def _detail(detail, status, **headers):
    return JSONResponse({"detail": detail}, status_code = status, headers = headers)


SYSTEM_ONE = ("POST", "/v1/systemone")
IMAGES = ("POST", "/api/inference/images/generate")
STT_LOAD = ("POST", "/api/inference/audio/stt/load")

# (tool, args, the studio's routes, then what the message must be: the whole message, or
# (check, text) pairs).
ROUTE_ERRORS = [
    (
        "chat",
        {"prompt": "hi"},
        {CHAT: openai_error("No model loaded. Call POST /inference/load first.", 400)},
        ("ends", "Load a chat model with load_model first; list_models shows the downloaded ones."),
    ),
    (
        "chat",
        {"prompt": "hi"},
        {
            CHAT: openai_error(
                "Another account is generating on the resident model.",
                409,
                type = "conflict_error",
                param = "model",
                code = "gpu_busy",
                headers = {"Retry-After": "5"},
            )
        },
        "GPU busy: Another account is generating on the resident model. Retry after 5 s.",
    ),
    (
        "embed",
        {"texts": ["x"]},
        {
            ("POST", "/v1/embeddings"): openai_error(
                "The embedding model /srv/models/nomic is not downloaded yet.",
                409,
                type = "conflict_error",
            )
        },
        ("starts", "The embedding model <path>"),
        ("lacks", "/srv/models"),
        ("ends", "load_model(kind='llm') and call embed again."),
    ),
    (
        "system_one",
        {"state": "x", "questions": URGENT},
        {
            SYSTEM_ONE: _detail(
                {
                    "error_type": "api_usage_error",
                    "message": "The Decision API is off. The Studio owner can turn it on in Settings > API.",
                },
                404,
            )
        },
        ("starts", "The Decision API is off."),
        ("ends", "Turn on the Decision API in Settings > API."),
    ),
    (
        "system_one",
        {"state": "x", "questions": URGENT},
        {
            SYSTEM_ONE: _detail(
                {"error_type": "model_loading", "message": "The decision model is loading."},
                503,
                **{"Retry-After": "12"},
            )
        },
        "The decision model is loading. (HTTP 503) Retry after 12 s.",
    ),
    (
        "generate_audio",
        {"workflow": "speak", "text": "Hi"},
        {("POST", "/v1/audio/run"): openai_error("No model loaded.", 400)},
        "No model loaded. (HTTP 400) Load a text-to-speech or music model with load_model(kind='tts') first.",
    ),
    (
        "generate_audio",
        {"workflow": "speak", "text": "Hi", "options": {"speaker_wav": "/srv/x.wav"}},
        {
            ("POST", "/v1/audio/run"): openai_error(
                "options: Value error, Option 'speaker_wav' is not accepted; name audio by id in inputs.",
                400,
                param = "options",
            )
        },
        ("has", "Option 'speaker_wav' is not accepted"),
        ("starts", "Invalid arguments:"),
    ),
    (
        "transcribe",
        {"audio": WAV_INPUT, "timestamps": True},
        {
            ("POST", "/v1/audio/transcriptions"): openai_error(
                "verbose_json reports the language of the audio and the local STT engine cannot detect it.",
                501,
                type = "api_error",
            )
        },
        (
            "ends",
            "(HTTP 501) Pass language with timestamps, or use an audio.cpp speech-to-text model.",
        ),
    ),
    (
        "generate_video",
        {"prompt": "waves"},
        {("POST", "/v1/videos"): openai_error("No video model is loaded.", 503, type = "api_error")},
        "No video model is loaded. (HTTP 503) Load a video model with load_model(kind='video') first.",
    ),
    (
        "generate_image",
        {"prompt": "x"},
        {IMAGES: _detail("No diffusion model is loaded.", 409)},
        "No diffusion model is loaded. (HTTP 409) Load an image model with load_model(kind='image') first.",
    ),
    (
        "generate_image",
        {"prompt": "x"},
        {IMAGES: _detail("Diffusion generation was cancelled.", 409)},
        "Diffusion generation was cancelled. (HTTP 409)",
    ),
    (
        "generate_image",
        {"prompt": "x", "width": 2048, "height": 2048},
        {
            IMAGES: _detail(
                "2048x2048 needs about 30 GB; 12 GB is free.",
                400,
                **{"X-Unsloth-Refusal": "memory-estimate"},
            )
        },
        "2048x2048 needs about 30 GB; 12 GB is free. Pass allow_oversized=true to try anyway.",
    ),
    (
        "run_recipe",
        {"recipe": {"columns": [{"name": "q", "column_type": "sampler"}]}, "mode": "full"},
        {("POST", "/api/data-recipe/jobs"): _detail("A recipe job is already running.", 409)},
        "A recipe job is already running. (HTTP 409)",
    ),
    (
        "start_training",
        {"config": DIFFUSION, "kind": "diffusion"},
        {
            ("POST", "/api/train/diffusion/start"): _detail(
                "An LLM training run is active. Stop it before starting image training.", 409
            )
        },
        "An LLM training run is active. Stop it before starting image training. (HTTP 409)",
    ),
    (
        "start_training",
        {"config": CONFIG},
        {
            ("POST", "/api/train/start"): {
                "job_id": "",
                "status": "error",
                "message": "Training is already in progress. Stop current training before starting a new one.",
                "error": "Training already active",
            }
        },
        "Training is already in progress. Stop current training before starting a new one.",
    ),
    (
        "cancel",
        {"kind": "training", "id": "someone-elses"},
        {
            ("GET", "/api/train/status"): {
                "job_id": "job-7",
                "is_training_running": True,
                "phase": "training",
            },
            ("POST", "/api/train/stop"): _detail(
                "The requested training job is no longer active.", 404
            ),
        },
        "The requested training job is no longer active. (HTTP 404)",
    ),
    (
        "load_model",
        {"model": WHISPER, "kind": "stt"},
        {
            ("GET", "/api/inference/audio/stt/status"): {
                "transformers": stt_state(downloaded = [WHISPER])
            },
            STT_LOAD: _detail("whisper-server is not installed. Run unsloth studio update.", 501),
        },
        ("has", "whisper-server is not installed"),
        ("has", "(HTTP 501)"),
    ),
]
CHECKS = {
    "starts": str.startswith,
    "ends": str.endswith,
    "has": str.__contains__,
    "lacks": lambda message, text: text not in message,
}


@pytest.mark.parametrize("tool,args,routes,expected", [(*row[:3], row[3:]) for row in ROUTE_ERRORS])
def test_a_route_failure_is_a_tool_error_the_agent_can_act_on(
    monkeypatch, tool, args, routes, expected
):
    result, _studio = run_tool(monkeypatch, routes, tool, args)
    assert result["isError"] is True
    message = result["content"][0]["text"]
    for check in expected:
        if isinstance(check, str):
            assert message == check
        else:
            assert CHECKS[check[0]](message, check[1]), (check, message)


COMPLETION = {"model": "m", "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}
DECISION = {"model": "m", "answers": {"q": {"type": "noul", "noul": 0.5}}}
CLIP = {"id": "clip-1", "role": "output", "sample_rate": 24000, "duration_s": 1.0}

# (tool, args for a file path, the studio's routes, the routes a local call reaches in order,
# what else a local call must show). Audio tools get a WAV file, the others a PNG.
PATH_CALLS = [
    (
        "chat",
        lambda path: {"prompt": "What is this?", "images": [{"path": path}]},
        {CHAT: COMPLETION},
        ["/v1/chat/completions"],
        None,
    ),
    (
        "system_one",
        lambda path: {
            "state": "x",
            "questions": {"q": {"type": "noul"}},
            "images": [{"path": path}],
        },
        {SYSTEM_ONE: DECISION},
        ["/v1/systemone"],
        None,
    ),
    (
        "generate_audio",
        lambda path: {"workflow": "separate", "source": {"path": path}},
        {
            ("POST", "/v1/audio/inputs"): JSONResponse({"id": "in-1"}, status_code = 201),
            ("POST", "/v1/audio/run"): {"clips": [CLIP]},
        },
        ["/v1/audio/inputs", "/v1/audio/run", "/v1/audio/gallery/clip-1/file"],
        lambda result, studio: call_to(studio, "/v1/audio/inputs")[3] == WAV,
    ),
    (
        "transcribe",
        lambda path: {"audio": {"path": path}},
        {("POST", "/v1/audio/transcriptions"): TRANSCRIPT},
        ["/v1/audio/transcriptions", "/api/inference/audio/stt/status"],
        lambda result, studio: (
            result["structuredContent"]["text"] == "Hello from Studio."
            and form(studio, "/v1/audio/transcriptions")["file"] == ("mcp-input.wav", WAV)
        ),
    ),
    (
        "generate_image",
        lambda path: {"prompt": "x", "init_image": {"path": path}},
        {IMAGES: {"images": [{"id": "img-1", "width": 8, "height": 8}]}},
        ["/api/inference/images/generate", "/api/inference/images/gallery/img-1/file"],
        None,
    ),
    (
        "generate_video",
        lambda path: {"prompt": "waves", "first_frame": {"path": path}},
        {("POST", "/v1/videos"): {"id": "video-1", "status": "queued"}},
        ["/v1/videos"],
        None,
    ),
]


def _input_file(tmp_path, tool):
    audio = tool in ("generate_audio", "transcribe")
    path = tmp_path / ("mcp-input.wav" if audio else "mcp-input.png")
    path.write_bytes(WAV if audio else PNG)
    return str(path)


@pytest.mark.parametrize("tool,args,routes,reached,check", PATH_CALLS)
def test_a_local_agent_may_send_a_path(
    monkeypatch, file_spy, tmp_path, tool, args, routes, reached, check
):
    path = _input_file(tmp_path, tool)
    # Every route answers, so the list of calls is the whole story.
    result, studio = run_tool(monkeypatch, routes, tool, args(path), fallback = {}, **LOCAL)
    assert result["isError"] is False, result
    assert [kind for kind, touched in file_spy if kind == "read" and touched == path] == ["read"]
    assert [c[1] for c in studio.state.calls] == reached
    assert check is None or check(result, studio)


@pytest.mark.parametrize("tool,args,routes,reached,check", PATH_CALLS)
def test_a_remote_agent_may_not_and_the_file_is_never_opened(
    monkeypatch, file_spy, tmp_path, tool, args, routes, reached, check
):
    path = _input_file(tmp_path, tool)
    result, studio = run_tool(monkeypatch, routes, tool, args(path), fallback = {}, **REMOTE)
    assert result["isError"] is True
    assert result["content"][0]["text"] == PATH_REMOTE
    assert file_spy == []
    assert studio.state.calls == []


def test_a_proxied_loopback_request_counts_as_remote(monkeypatch, file_spy, tmp_path):
    tool, args, routes, _reached, _check = PATH_CALLS[0]
    result, _studio = run_tool(
        monkeypatch,
        routes,
        tool,
        args(_input_file(tmp_path, tool)),
        headers = {"X-Forwarded-For": "203.0.113.9"},
        **LOCAL,
    )
    assert result["content"][0]["text"] == PATH_REMOTE
    assert file_spy == []


class _Row(ToolOutput):
    text: RouteText = "fallback"
    note: Optional[RouteText] = None
    count: Optional[int] = None
    ratio: Optional[float] = None
    flag: bool = False
    maybe: Optional[bool] = None
    names: list[RouteText] = []
    tags: Optional[list[RouteText]] = None
    usage: Optional[Usage] = None


DEFAULTS = {
    "text": "fallback",
    "note": None,
    "count": None,
    "ratio": None,
    "flag": False,
    "maybe": False,
    "names": [],
    "tags": None,
    "usage": None,
}


@pytest.mark.parametrize(
    "payload,expected",
    [
        ({}, DEFAULTS),
        ("not a dict", DEFAULTS),
        (
            {"text": "a", "note": "see /srv/x/y", "count": 3, "ratio": 2, "flag": True},
            {**DEFAULTS, "text": "a", "note": "see <path>", "count": 3, "ratio": 2.0, "flag": True},
        ),
        (
            {
                "text": "",
                "count": True,
                "ratio": False,
                "flag": 1,
                "maybe": True,
                "names": ["a", 1],
            },
            {**DEFAULTS, "maybe": True, "names": ["a"]},
        ),
        (
            {"note": 5, "count": 2.5, "ratio": "1", "names": "a", "tags": ["b", None]},
            {**DEFAULTS, "tags": ["b"]},
        ),
        ({"usage": {"total_tokens": 1}, "unknown": "x"}, DEFAULTS),
    ],
)
def test_from_route_copies_each_field_by_its_type(payload, expected):
    row = _Row.from_route(payload)
    assert row.model_dump(mode = "json") == expected
    assert list(row.model_dump(mode = "json")) == list(_Row.model_fields)


def test_from_route_takes_overrides_as_given():
    row = _Row.from_route({"text": "a", "flag": True}, flag = False, usage = Usage(total_tokens = 3))
    assert row.flag is False
    assert row.text == "a"
    assert row.usage == Usage(total_tokens = 3)
