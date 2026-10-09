# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
import json

import pytest
from fastapi.responses import Response
from fastapi.testclient import TestClient

from mcp_server import create_studio_mcp
from studio_mcp import loading

from .mcp_harness import MCP_HEADERS, TEST_TOKEN, call_tool, fake_studio, served

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

PAYLOADS = {
    ("POST", "/api/inference/load"): LOADED,
    ("GET", "/api/inference/load-progress"): {"phase": "mmap", "fraction": 0.5},
    ("GET", "/api/models/download-progress"): {
        "progress": 0.25,
        "downloaded_bytes": 1,
        "expected_bytes": 4,
    },
    ("POST", "/api/inference/unload"): {"status": "unloaded", "model": LLM},
    ("GET", "/api/inference/status"): STATUS,
}


def _answer(payload):
    return lambda request, body: payload


def _studio(overrides = None):
    routes = {key: _answer(value) for key, value in PAYLOADS.items()}
    routes.update(overrides or {})
    return fake_studio(routes)


def _call(
    monkeypatch,
    studio,
    name,
    args,
    headers = None,
):
    with TestClient(served(create_studio_mcp(), studio, monkeypatch = monkeypatch)) as http:
        return call_tool(http, name, args, headers = headers)


def _bodies(studio, path):
    return [json.loads(body) for _m, p, _h, body in studio.state.calls if p == path]


def test_load_sends_the_body_and_reports_the_result(monkeypatch):
    studio = _studio()
    result = _call(monkeypatch, studio, "load_model", {"model": LLM, "variant": "Q4_K_M"})
    assert result["structuredContent"] == {
        "kind": "llm",
        "model": LLM,
        "loaded": True,
        "display_name": "Llama 3.2 1B",
        "evicted": ["unsloth/Qwen3-0.6B"],
    }
    assert _bodies(studio, "/api/inference/load") == [
        {"model_path": LLM, "load_in_4bit": True, "max_seq_length": 0, "gguf_variant": "Q4_K_M"}
    ]


def test_the_hub_token_goes_in_the_body_not_the_header(monkeypatch):
    studio = _studio()
    _call(
        monkeypatch,
        studio,
        "load_model",
        {"model": LLM, "hf_token": "hf_param", "load_in_4bit": False},
    )
    (_m, _p, headers, body) = next(c for c in studio.state.calls if c[1] == "/api/inference/load")
    assert json.loads(body)["hf_token"] == "hf_param"
    assert json.loads(body)["load_in_4bit"] is False
    assert "x-unsloth-hf-token" not in headers

    studio = _studio()
    _call(
        monkeypatch,
        studio,
        "load_model",
        {"model": LLM},
        headers = {"X-Unsloth-HF-Token": "hf_header"},
    )
    (_m, _p, headers, body) = next(c for c in studio.state.calls if c[1] == "/api/inference/load")
    assert json.loads(body)["hf_token"] == "hf_header"
    assert "x-unsloth-hf-token" not in headers


@pytest.mark.parametrize(
    "deferred",
    [
        {"status_code": 409, "detail": "Another model is loading"},
        {"status_code": 500, "detail": "RuntimeError: CUDA out of memory"},
    ],
)
def test_a_padded_deferred_error_is_a_tool_error(monkeypatch, deferred):
    padded = b" " * 40 + json.dumps({"_deferred_error": deferred}).encode()

    def slow_failure(request, body):
        return Response(padded, media_type = "application/json")

    studio = _studio({("POST", "/api/inference/load"): slow_failure})
    result = _call(monkeypatch, studio, "load_model", {"model": LLM})
    assert result["isError"] is True
    assert deferred["detail"] in result["content"][0]["text"]
    assert f"(HTTP {deferred['status_code']})" in result["content"][0]["text"]


def test_progress_is_reported_while_the_load_runs(monkeypatch):
    monkeypatch.setattr(loading, "POLL_INTERVAL_S", 0.02)

    async def slow_load(request, body):
        await asyncio.sleep(0.3)
        return LOADED

    studio = _studio({("POST", "/api/inference/load"): slow_load})
    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": "load_model",
            "arguments": {"model": LLM},
            "_meta": {"progressToken": "p1"},
        },
    }
    with TestClient(served(create_studio_mcp(), studio, monkeypatch = monkeypatch)) as http:
        response = http.post(
            "/mcp/", json = request, headers = {**MCP_HEADERS, "Authorization": f"Bearer {TEST_TOKEN}"}
        )
    messages = [
        json.loads(line[5:]) for line in response.text.splitlines() if line.startswith("data:")
    ]
    progress = [m["params"] for m in messages if m.get("method") == "notifications/progress"]
    assert progress, response.text
    assert progress[0]["progressToken"] == "p1"
    assert progress[0]["progress"] == 0.5
    assert "mmap" in progress[0]["message"]
    assert messages[-1]["result"]["structuredContent"]["loaded"] is True


def test_download_progress_is_used_before_the_load_starts(monkeypatch):
    monkeypatch.setattr(loading, "POLL_INTERVAL_S", 0.02)

    async def slow_load(request, body):
        await asyncio.sleep(0.2)
        return LOADED

    studio = _studio(
        {
            ("POST", "/api/inference/load"): slow_load,
            ("GET", "/api/inference/load-progress"): _answer({"phase": None, "fraction": 0.0}),
        }
    )
    _call(monkeypatch, studio, "load_model", {"model": LLM})
    downloads = [c for c in studio.state.calls if c[1] == "/api/models/download-progress"]
    assert downloads
    assert all(c[2].get("x-unsloth-hf-token") is None for c in downloads)


def test_unload_picks_the_serving_checkpoint_and_checks_the_result(monkeypatch):
    statuses = [STATUS, EMPTY_STATUS]
    studio = _studio({("GET", "/api/inference/status"): lambda request, body: statuses.pop(0)})
    result = _call(monkeypatch, studio, "unload_model", {})
    assert result["structuredContent"] == {"kind": "llm", "model": LLM, "unloaded": True}
    assert _bodies(studio, "/api/inference/unload") == [{"model_path": "/srv/models/llama.gguf"}]


def test_unload_by_public_id_maps_to_its_checkpoint(monkeypatch):
    statuses = [STATUS, EMPTY_STATUS]
    studio = _studio({("GET", "/api/inference/status"): lambda request, body: statuses.pop(0)})
    result = _call(monkeypatch, studio, "unload_model", {"model": LLM})
    assert result["structuredContent"]["unloaded"] is True
    assert _bodies(studio, "/api/inference/unload") == [{"model_path": "/srv/models/llama.gguf"}]


def test_an_unmatched_unload_reports_unloaded_false(monkeypatch):
    studio = _studio()
    result = _call(monkeypatch, studio, "unload_model", {"model": "unsloth/not-loaded"})
    assert result["structuredContent"] == {
        "kind": "llm",
        "model": "unsloth/not-loaded",
        "unloaded": False,
    }


def test_an_unload_that_leaves_the_model_resident_is_not_success(monkeypatch):
    studio = _studio()
    result = _call(monkeypatch, studio, "unload_model", {"model": LLM})
    assert result["structuredContent"]["unloaded"] is False


def test_unload_with_nothing_loaded(monkeypatch):
    studio = _studio({("GET", "/api/inference/status"): _answer(EMPTY_STATUS)})
    result = _call(monkeypatch, studio, "unload_model", {})
    assert result["structuredContent"] == {"kind": "llm", "model": None, "unloaded": False}
    assert _bodies(studio, "/api/inference/unload") == []


def test_annotations_and_schemas():
    tools = {t.name: t for t in asyncio.run(create_studio_mcp().list_tools())}
    load, unload = tools["load_model"], tools["unload_model"]
    assert load.annotations.readOnlyHint is False
    assert load.annotations.destructiveHint is False
    assert unload.annotations.destructiveHint is True
    assert unload.annotations.readOnlyHint is False
    for tool in (load, unload):
        assert tool.annotations.openWorldHint is False
        assert tool.output_schema is not None
    assert set(load.parameters["properties"]) == {
        "model",
        "kind",
        "variant",
        "max_seq_length",
        "load_in_4bit",
        "hf_token",
    }
    assert load.parameters["properties"]["kind"]["default"] == "llm"


FLUX_GGUF = "city96/FLUX.1-schnell-gguf"
LTX = "Lightricks/LTX-Video"


def _media_studio(
    kind,
    *,
    progress,
    statuses,
    load_answer = None,
    variants = None,
    plan = None,
):
    """``progress`` and ``statuses`` are consumed in order; the last one repeats."""
    progress, statuses = list(progress), list(statuses)

    def next_of(items):
        return lambda request, body: items.pop(0) if len(items) > 1 else items[0]

    prefix = "/api/inference/images" if kind == "image" else "/api/inference/video"
    return fake_studio(
        {
            ("GET", "/api/hub/gguf-variants"): _answer(
                variants
                or {
                    "repo_id": FLUX_GGUF,
                    "default_variant": "Q4_K_S",
                    "variants": [
                        {"filename": "flux1-schnell-Q4_K_S.gguf", "quant": "Q4_K_S"},
                        {"filename": "flux1-schnell-Q8_0.gguf", "quant": "Q8_0"},
                    ],
                }
            ),
            ("POST", f"{prefix}/download-plan"): _answer(
                plan or {"entries": [], "plan_failed": False}
            ),
            ("POST", f"{prefix}/load"): _answer(load_answer or {"loaded": False}),
            ("GET", f"{prefix}/load-progress"): next_of(progress),
            ("GET", f"{prefix}/status"): next_of(statuses),
            ("POST", f"{prefix}/unload"): _answer({"loaded": False}),
        }
    )


@pytest.fixture
def fast_polls(monkeypatch):
    monkeypatch.setattr(loading, "POLL_INTERVAL_S", 0.01)


def test_a_gguf_image_repo_sends_its_gguf_filename(monkeypatch, fast_polls):
    studio = _media_studio(
        "image",
        progress = [
            {"phase": "downloading", "bytes_downloaded": 1, "bytes_total": 2},
            {"phase": "ready"},
        ],
        statuses = [{"loaded": True, "repo_id": FLUX_GGUF, "family": "flux"}],
    )
    result = _call(
        monkeypatch, studio, "load_model", {"model": FLUX_GGUF, "kind": "image", "variant": "Q8_0"}
    )
    assert result["structuredContent"] == {
        "kind": "image",
        "model": FLUX_GGUF,
        "loaded": True,
        "display_name": None,
        "evicted": [],
    }
    expected = {
        "model_path": FLUX_GGUF,
        "gguf_filename": "flux1-schnell-Q8_0.gguf",
        "model_kind": "gguf",
    }
    assert _bodies(studio, "/api/inference/images/download-plan") == [expected]
    assert _bodies(studio, "/api/inference/images/load") == [expected]


def test_the_default_variant_is_used_when_none_is_named(monkeypatch, fast_polls):
    studio = _media_studio(
        "image", progress = [{"phase": "ready"}], statuses = [{"loaded": True, "repo_id": FLUX_GGUF}]
    )
    _call(monkeypatch, studio, "load_model", {"model": FLUX_GGUF, "kind": "image"})
    assert (
        _bodies(studio, "/api/inference/images/load")[0]["gguf_filename"]
        == "flux1-schnell-Q4_K_S.gguf"
    )


def test_an_unknown_variant_is_refused_before_any_load(monkeypatch, fast_polls):
    studio = _media_studio("image", progress = [{"phase": "ready"}], statuses = [{"loaded": False}])
    result = _call(
        monkeypatch, studio, "load_model", {"model": FLUX_GGUF, "kind": "image", "variant": "Q2_K"}
    )
    assert result["isError"] is True
    assert "Q2_K" in result["content"][0]["text"]
    assert _bodies(studio, "/api/inference/images/load") == []


def test_a_plan_entry_supplies_the_checkpoint_file(monkeypatch, fast_polls):
    plan = {
        "entries": [
            {
                "repo_id": "a/b",
                "files": [],
                "bytes": 1,
                "gguf_filename": "model.safetensors",
                "checkpoint": True,
            }
        ]
    }
    studio = _media_studio(
        "image",
        progress = [{"phase": "ready"}],
        statuses = [{"loaded": True, "repo_id": "a/b"}],
        plan = plan,
    )
    _call(monkeypatch, studio, "load_model", {"model": "a/b", "kind": "image"})
    assert _bodies(studio, "/api/inference/images/load") == [
        {"model_path": "a/b", "gguf_filename": "model.safetensors"}
    ]
    assert [c for c in studio.state.calls if c[1] == "/api/hub/gguf-variants"] == []


def test_an_incompatible_plan_is_a_tool_error(monkeypatch, fast_polls):
    studio = _media_studio(
        "image",
        progress = [{"phase": "ready"}],
        statuses = [{"loaded": False}],
        plan = {"entries": [], "incompatible_reason": "Needs a CUDA GPU"},
    )
    result = _call(monkeypatch, studio, "load_model", {"model": "a/b", "kind": "image"})
    assert result["isError"] is True
    assert "Needs a CUDA GPU" in result["content"][0]["text"]


def test_a_load_error_phase_is_a_tool_error(monkeypatch, fast_polls):
    studio = _media_studio(
        "image",
        progress = [{"phase": "downloading"}, {"phase": "error", "error": "Out of disk space"}],
        statuses = [{"loaded": False}],
    )
    result = _call(monkeypatch, studio, "load_model", {"model": "a/b", "kind": "image"})
    assert result["isError"] is True
    assert "Out of disk space" in result["content"][0]["text"]


def test_the_previous_model_in_the_load_answer_is_not_success(monkeypatch, fast_polls):
    previous = {"loaded": True, "repo_id": "old/model"}
    studio = _media_studio(
        "image", progress = [{"phase": "ready"}], statuses = [previous], load_answer = previous
    )
    result = _call(monkeypatch, studio, "load_model", {"model": "new/model", "kind": "image"})
    assert result["isError"] is True
    assert "did not finish loading new/model" in result["content"][0]["text"]

    studio = _media_studio(
        "image",
        progress = [{"phase": None}, {"phase": "downloading"}, {"phase": "ready"}],
        statuses = [previous, {"loaded": True, "repo_id": "new/model"}],
        load_answer = previous,
    )
    result = _call(monkeypatch, studio, "load_model", {"model": "new/model", "kind": "image"})
    assert result["structuredContent"]["model"] == "new/model"


def test_a_load_that_never_starts_gives_up(monkeypatch, fast_polls):
    studio = _media_studio(
        "video", progress = [{"phase": None}], statuses = [{"loaded": True, "repo_id": "old/model"}]
    )
    result = _call(monkeypatch, studio, "load_model", {"model": LTX, "kind": "video"})
    assert result["isError"] is True
    assert "stopped reporting" in result["content"][0]["text"]


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
    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": "load_model",
            "arguments": {"model": LTX, "kind": "video"},
            "_meta": {"progressToken": 7},
        },
    }
    with TestClient(served(create_studio_mcp(), studio, monkeypatch = monkeypatch)) as http:
        response = http.post(
            "/mcp/", json = request, headers = {**MCP_HEADERS, "Authorization": f"Bearer {TEST_TOKEN}"}
        )
    messages = [
        json.loads(line[5:]) for line in response.text.splitlines() if line.startswith("data:")
    ]
    progress = [m["params"] for m in messages if m.get("method") == "notifications/progress"]
    assert [p["progress"] for p in progress] == [0.25, 1.0]
    assert progress[0]["message"] == "Downloading"
    assert messages[-1]["result"]["structuredContent"]["model"] == LTX
    assert _bodies(studio, "/api/inference/video/load") == [{"model_path": LTX}]


@pytest.mark.parametrize("kind", ["image", "video"])
def test_media_unload_takes_no_body(monkeypatch, kind):
    studio = _media_studio(
        kind, progress = [{"phase": None}], statuses = [{"loaded": True, "repo_id": LTX}]
    )
    result = _call(monkeypatch, studio, "unload_model", {"kind": kind})
    assert result["structuredContent"] == {"kind": kind, "model": LTX, "unloaded": True}
    prefix = "/api/inference/images" if kind == "image" else "/api/inference/video"
    (unload,) = [c for c in studio.state.calls if c[1] == f"{prefix}/unload"]
    assert unload[3] == b""


def test_media_unload_with_nothing_loaded(monkeypatch):
    studio = _media_studio("image", progress = [{"phase": None}], statuses = [{"loaded": False}])
    result = _call(monkeypatch, studio, "unload_model", {"kind": "image"})
    assert result["structuredContent"] == {"kind": "image", "model": None, "unloaded": False}
    assert [c for c in studio.state.calls if c[1].endswith("/unload")] == []
