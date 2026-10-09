# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import json

import pytest
from fastapi.responses import Response

from .mcp_harness import (
    DEFERRED,
    WHISPER,
    bodies,
    call_to,
    call_with_progress,
    fake_studio,
    fast_polls,  # noqa: F401  (fixture)
    queries,
    run_tool,
    sequence,
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


def test_load_sends_the_body_and_reports_the_result(monkeypatch):
    result, studio = run_tool(
        monkeypatch, PAYLOADS, "load_model", {"model": LLM, "variant": "Q4_K_M"}
    )
    assert result["structuredContent"] == {
        "kind": "llm",
        "model": LLM,
        "loaded": True,
        "display_name": "Llama 3.2 1B",
        "evicted": ["unsloth/Qwen3-0.6B"],
    }
    assert bodies(studio, "/api/inference/load") == [
        {"model_path": LLM, "load_in_4bit": True, "max_seq_length": 0, "gguf_variant": "Q4_K_M"}
    ]


@pytest.mark.parametrize("kind", ["tts", "audio"])
def test_the_kinds_list_models_reports_for_audio_load_as_llm(monkeypatch, kind):
    result, studio = run_tool(monkeypatch, PAYLOADS, "load_model", {"model": LLM, "kind": kind})
    assert result["isError"] is False
    assert bodies(studio, "/api/inference/load")[0]["model_path"] == LLM
    # The kind asked for is the kind reported.
    assert result["structuredContent"]["kind"] == kind


def test_a_chat_model_unloaded_without_the_route_saying_so_is_evicted(monkeypatch):
    # An audio model's engine switch unloads the chat model, and the route's evicted list misses it.
    routes = {
        **PAYLOADS,
        ("POST", "/api/inference/load"): {**LOADED, "evicted": []},
        ("GET", "/api/inference/status"): sequence(
            {**STATUS, "serving": ["unsloth/Qwen3-0.6B-GGUF"]}, EMPTY_STATUS
        ),
    }
    result, _studio = run_tool(monkeypatch, routes, "load_model", {"model": LLM, "kind": "tts"})
    assert result["structuredContent"]["evicted"] == ["unsloth/Qwen3-0.6B-GGUF"]


def test_what_the_route_names_is_not_listed_again_by_its_serving_id(monkeypatch):
    # The route keys a dropped GGUF as repo:variant while status serves the bare repo id.
    routes = {
        **PAYLOADS,
        ("POST", "/api/inference/load"): {**LOADED, "evicted": ["unsloth/Qwen3-8B-GGUF:Q4_K_M"]},
        ("GET", "/api/inference/status"): sequence(
            {**STATUS, "serving": [LLM, "unsloth/Qwen3-8B-GGUF"]}, STATUS
        ),
    }
    result, _studio = run_tool(monkeypatch, routes, "load_model", {"model": LLM})
    assert result["structuredContent"]["evicted"] == ["unsloth/Qwen3-8B-GGUF:Q4_K_M"]


def test_a_model_still_serving_is_not_evicted(monkeypatch):
    result, _studio = run_tool(monkeypatch, PAYLOADS, "load_model", {"model": LLM})
    # STATUS serves LLM before and after; only the route's own list counts.
    assert result["structuredContent"]["evicted"] == ["unsloth/Qwen3-0.6B"]


def test_the_hub_token_goes_in_the_body_not_the_header(monkeypatch):
    args = {"model": LLM, "hf_token": "hf_param", "load_in_4bit": False}
    _result, studio = run_tool(monkeypatch, PAYLOADS, "load_model", args)
    (_m, _p, headers, body) = call_to(studio, "/api/inference/load")
    assert json.loads(body)["hf_token"] == "hf_param"
    assert json.loads(body)["load_in_4bit"] is False
    assert "x-unsloth-hf-token" not in headers

    _result, studio = run_tool(
        monkeypatch,
        PAYLOADS,
        "load_model",
        {"model": LLM},
        headers = {"X-Unsloth-HF-Token": "hf_header"},
    )
    (_m, _p, headers, body) = call_to(studio, "/api/inference/load")
    assert json.loads(body)["hf_token"] == "hf_header"
    assert "x-unsloth-hf-token" not in headers


@pytest.mark.parametrize("deferred", DEFERRED)
def test_a_padded_deferred_error_is_a_tool_error(monkeypatch, deferred):
    padded = b" " * 40 + json.dumps({"_deferred_error": deferred}).encode()
    slow_failure = Response(padded, media_type = "application/json")
    routes = {**PAYLOADS, ("POST", "/api/inference/load"): slow_failure}
    result, _studio = run_tool(monkeypatch, routes, "load_model", {"model": LLM})
    assert result["isError"] is True
    assert deferred["detail"] in result["content"][0]["text"]
    assert f"(HTTP {deferred['status_code']})" in result["content"][0]["text"]


def test_progress_is_reported_while_the_load_runs(monkeypatch, fast_polls):
    studio = fake_studio({**PAYLOADS, ("POST", "/api/inference/load"): slow(LOADED, 0.3)})
    progress, result = call_with_progress(monkeypatch, studio, "load_model", {"model": LLM}, "p1")
    assert progress, result
    assert progress[0]["progressToken"] == "p1"
    assert progress[0]["progress"] == 0.5
    assert "mmap" in progress[0]["message"]
    assert result["structuredContent"]["loaded"] is True


def test_download_progress_is_used_before_the_load_starts(monkeypatch, fast_polls):
    routes = {
        **PAYLOADS,
        ("POST", "/api/inference/load"): slow(LOADED, 0.2),
        ("GET", "/api/inference/load-progress"): {"phase": None, "fraction": 0.0},
    }
    _result, studio = run_tool(monkeypatch, routes, "load_model", {"model": LLM})
    downloads = [c for c in studio.state.calls if c[1] == "/api/models/download-progress"]
    assert downloads
    assert all(c[2].get("x-unsloth-hf-token") is None for c in downloads)


# Without a model the serving one is unloaded; a public id maps to its checkpoint.
@pytest.mark.parametrize("args", [{}, {"model": LLM}])
def test_unload_picks_the_serving_checkpoint_and_checks_the_result(monkeypatch, args):
    routes = {**PAYLOADS, ("GET", "/api/inference/status"): sequence(STATUS, EMPTY_STATUS)}
    result, studio = run_tool(monkeypatch, routes, "unload_model", args)
    assert result["structuredContent"] == {"kind": "llm", "model": LLM, "unloaded": True}
    assert bodies(studio, "/api/inference/unload") == [{"model_path": "/srv/models/llama.gguf"}]


# An unload that matched nothing, or that left the model resident (STATUS still serves LLM),
# is not success.
@pytest.mark.parametrize("model", ["unsloth/not-loaded", LLM])
def test_an_unload_that_did_not_happen_reports_unloaded_false(monkeypatch, model):
    result, _studio = run_tool(monkeypatch, PAYLOADS, "unload_model", {"model": model})
    assert result["structuredContent"] == {"kind": "llm", "model": model, "unloaded": False}


def test_unload_with_nothing_loaded(monkeypatch):
    routes = {**PAYLOADS, ("GET", "/api/inference/status"): EMPTY_STATUS}
    result, studio = run_tool(monkeypatch, routes, "unload_model", {})
    assert result["structuredContent"] == {"kind": "llm", "model": None, "unloaded": False}
    assert bodies(studio, "/api/inference/unload") == []


FLUX_GGUF = "city96/FLUX.1-schnell-gguf"
LTX = "Lightricks/LTX-Video"


def _media_studio(
    kind,
    *,
    progress = ({"phase": "ready"},),
    statuses = ({"loaded": False},),
    load_answer = None,
    plan = None,
):
    """``progress`` and ``statuses`` are consumed in order; the last one repeats."""
    prefix = "/api/inference/images" if kind == "image" else "/api/inference/video"
    return fake_studio(
        {
            ("GET", "/api/hub/gguf-variants"): {
                "repo_id": FLUX_GGUF,
                "default_variant": "Q4_K_S",
                "variants": [
                    {"filename": "flux1-schnell-Q4_K_S.gguf", "quant": "Q4_K_S"},
                    {"filename": "flux1-schnell-Q8_0.gguf", "quant": "Q8_0"},
                ],
            },
            ("POST", f"{prefix}/download-plan"): plan or {"entries": [], "plan_failed": False},
            ("POST", f"{prefix}/load"): load_answer or {"loaded": False},
            ("GET", f"{prefix}/load-progress"): sequence(*progress),
            ("GET", f"{prefix}/status"): sequence(*statuses),
            ("POST", f"{prefix}/unload"): {"loaded": False},
        }
    )


def test_a_gguf_image_repo_sends_its_gguf_filename(monkeypatch, fast_polls):
    studio = _media_studio(
        "image",
        progress = [
            {"phase": "downloading", "bytes_downloaded": 1, "bytes_total": 2},
            {"phase": "ready"},
        ],
        statuses = [{"loaded": True, "repo_id": FLUX_GGUF, "family": "flux"}],
    )
    result, _studio = run_tool(
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
    assert bodies(studio, "/api/inference/images/download-plan") == [expected]
    assert bodies(studio, "/api/inference/images/load") == [expected]


def test_the_default_variant_is_used_when_none_is_named(monkeypatch, fast_polls):
    studio = _media_studio("image", statuses = [{"loaded": True, "repo_id": FLUX_GGUF}])
    run_tool(monkeypatch, studio, "load_model", {"model": FLUX_GGUF, "kind": "image"})
    assert (
        bodies(studio, "/api/inference/images/load")[0]["gguf_filename"]
        == "flux1-schnell-Q4_K_S.gguf"
    )


def test_an_unknown_variant_is_refused_before_any_load(monkeypatch, fast_polls):
    studio = _media_studio("image")
    result, _studio = run_tool(
        monkeypatch, studio, "load_model", {"model": FLUX_GGUF, "kind": "image", "variant": "Q2_K"}
    )
    assert result["isError"] is True
    assert "Q2_K" in result["content"][0]["text"]
    assert bodies(studio, "/api/inference/images/load") == []


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
    studio = _media_studio("image", statuses = [{"loaded": True, "repo_id": "a/b"}], plan = plan)
    run_tool(monkeypatch, studio, "load_model", {"model": "a/b", "kind": "image"})
    assert bodies(studio, "/api/inference/images/load") == [
        {"model_path": "a/b", "gguf_filename": "model.safetensors"}
    ]
    assert [c for c in studio.state.calls if c[1] == "/api/hub/gguf-variants"] == []


def test_an_incompatible_plan_is_a_tool_error(monkeypatch, fast_polls):
    studio = _media_studio("image", plan = {"entries": [], "incompatible_reason": "Needs a CUDA GPU"})
    result, _studio = run_tool(monkeypatch, studio, "load_model", {"model": "a/b", "kind": "image"})
    assert result["isError"] is True
    assert "Needs a CUDA GPU" in result["content"][0]["text"]


def test_a_load_error_phase_is_a_tool_error(monkeypatch, fast_polls):
    studio = _media_studio(
        "image",
        progress = [{"phase": "downloading"}, {"phase": "error", "error": "Out of disk space"}],
    )
    result, _studio = run_tool(monkeypatch, studio, "load_model", {"model": "a/b", "kind": "image"})
    assert result["isError"] is True
    assert "Out of disk space" in result["content"][0]["text"]


def test_the_previous_model_in_the_load_answer_is_not_success(monkeypatch, fast_polls):
    previous = {"loaded": True, "repo_id": "old/model"}
    studio = _media_studio("image", statuses = [previous], load_answer = previous)
    result, _studio = run_tool(
        monkeypatch, studio, "load_model", {"model": "new/model", "kind": "image"}
    )
    assert result["isError"] is True
    assert "did not finish loading new/model" in result["content"][0]["text"]

    studio = _media_studio(
        "image",
        progress = [{"phase": None}, {"phase": "downloading"}, {"phase": "ready"}],
        statuses = [previous, {"loaded": True, "repo_id": "new/model"}],
        load_answer = previous,
    )
    result, _studio = run_tool(
        monkeypatch, studio, "load_model", {"model": "new/model", "kind": "image"}
    )
    assert result["structuredContent"]["model"] == "new/model"


def test_a_load_that_never_starts_gives_up(monkeypatch, fast_polls):
    studio = _media_studio(
        "video", progress = [{"phase": None}], statuses = [{"loaded": True, "repo_id": "old/model"}]
    )
    result, _studio = run_tool(monkeypatch, studio, "load_model", {"model": LTX, "kind": "video"})
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
    args = {"model": LTX, "kind": "video"}
    progress, result = call_with_progress(monkeypatch, studio, "load_model", args)
    assert [p["progress"] for p in progress] == [0.25, 1.0]
    assert progress[0]["message"] == "Downloading"
    assert result["structuredContent"]["model"] == LTX
    assert bodies(studio, "/api/inference/video/load") == [{"model_path": LTX}]


@pytest.mark.parametrize("kind", ["image", "video"])
def test_media_unload_takes_no_body(monkeypatch, kind):
    studio = _media_studio(kind, statuses = [{"loaded": True, "repo_id": LTX}])
    result, _studio = run_tool(monkeypatch, studio, "unload_model", {"kind": kind})
    assert result["structuredContent"] == {"kind": kind, "model": LTX, "unloaded": True}
    prefix = "/api/inference/images" if kind == "image" else "/api/inference/video"
    (unload,) = [c for c in studio.state.calls if c[1] == f"{prefix}/unload"]
    assert unload[3] == b""


def test_media_unload_with_nothing_loaded(monkeypatch):
    studio = _media_studio("image")
    result, _studio = run_tool(monkeypatch, studio, "unload_model", {"kind": "image"})
    assert result["structuredContent"] == {"kind": "image", "model": None, "unloaded": False}
    assert [c for c in studio.state.calls if c[1].endswith("/unload")] == []


def _stt_studio(statuses, *, download = None):
    return fake_studio(
        {
            ("GET", "/api/inference/audio/stt/status"): sequence(*statuses),
            ("POST", "/api/inference/audio/stt/download"): download
            or {"downloading": True, "model": WHISPER, "download_id": "d1"},
            ("POST", "/api/inference/audio/stt/load"): {"loaded_model": WHISPER, "device": "cuda"},
            ("POST", "/api/inference/audio/stt/unload"): {"loaded_model": None, "device": None},
        }
    )


def test_stt_downloads_then_loads_in_order(monkeypatch, fast_polls):
    studio = _stt_studio(
        [
            {"transformers": stt_state()},
            {
                "transformers": stt_state(
                    download = {
                        "downloading": True,
                        "model": WHISPER,
                        "bytes_done": 1,
                        "bytes_total": 2,
                    }
                )
            },
            {
                "transformers": stt_state(
                    downloaded = [WHISPER],
                    download = {"downloading": False, "completed_download_ids": ["d1"]},
                )
            },
        ]
    )
    result, _studio = run_tool(
        monkeypatch,
        studio,
        "load_model",
        {"model": WHISPER, "kind": "stt"},
        headers = {"X-Unsloth-HF-Token": "hf_h"},
    )
    assert result["structuredContent"]["model"] == WHISPER
    order = [(m, p) for m, p, _h, _b in studio.state.calls]
    assert order == [
        ("GET", "/api/inference/audio/stt/status"),
        ("POST", "/api/inference/audio/stt/download"),
        ("GET", "/api/inference/audio/stt/status"),
        ("GET", "/api/inference/audio/stt/status"),
        ("POST", "/api/inference/audio/stt/load"),
    ]
    assert bodies(studio, "/api/inference/audio/stt/download") == [
        {"model": WHISPER, "engine": "transformers"}
    ]
    assert call_to(studio, "/api/inference/audio/stt/download")[2]["x-unsloth-hf-token"] == "hf_h"
    assert bodies(studio, "/api/inference/audio/stt/load") == [
        {"model": WHISPER, "engine": "transformers"}
    ]


def test_the_hf_token_argument_reaches_the_gguf_lookup(monkeypatch, fast_polls):
    # A gated GGUF repo needs the token on the variant lookup, not only on the load.
    studio = _media_studio(
        "image", statuses = [{"loaded": True, "repo_id": FLUX_GGUF, "family": "flux"}]
    )
    run_tool(
        monkeypatch,
        studio,
        "load_model",
        {"model": FLUX_GGUF, "kind": "image", "hf_token": "hf_arg"},
    )
    lookup = call_to(studio, "/api/hub/gguf-variants")
    assert lookup[2]["x-unsloth-hf-token"] == "hf_arg"


def test_the_hf_token_argument_rides_the_download_header(monkeypatch, fast_polls):
    studio = _stt_studio(
        [{"transformers": stt_state()}, {"transformers": stt_state(downloaded = [WHISPER])}],
        download = {"downloading": True, "model": WHISPER},
    )
    run_tool(
        monkeypatch, studio, "load_model", {"model": WHISPER, "kind": "stt", "hf_token": "hf_arg"}
    )
    download = call_to(studio, "/api/inference/audio/stt/download")
    assert download[2]["x-unsloth-hf-token"] == "hf_arg"
    assert "hf_token" not in json.loads(download[3])


def test_a_downloaded_stt_model_loads_straight_away(monkeypatch):
    studio = _stt_studio(
        [{"gguf": {**stt_state(downloaded = ["whisper-small-q5"]), "models": ["whisper-small-q5"]}}]
    )
    result, _studio = run_tool(
        monkeypatch, studio, "load_model", {"model": "whisper-small-q5", "kind": "stt"}
    )
    assert result["isError"] is False
    assert [c[1] for c in studio.state.calls if c[0] == "POST"] == ["/api/inference/audio/stt/load"]
    assert bodies(studio, "/api/inference/audio/stt/load") == [
        {"model": "whisper-small-q5", "engine": "gguf"}
    ]


def test_a_failed_download_never_loads(monkeypatch, fast_polls):
    studio = _stt_studio(
        [
            {"transformers": stt_state()},
            {"transformers": stt_state(download = {"downloading": False, "error": "401 gated repo"})},
        ]
    )
    result, _studio = run_tool(monkeypatch, studio, "load_model", {"model": WHISPER, "kind": "stt"})
    assert result["isError"] is True
    assert "401 gated repo" in result["content"][0]["text"]
    assert [c for c in studio.state.calls if c[1].endswith("/load")] == []


def test_stt_unload_names_the_engine_and_model_in_the_query(monkeypatch):
    studio = _stt_studio(
        [
            {"transformers": stt_state(), "mtmd": stt_state(loaded = "qwen3-asr")},
            {"transformers": stt_state(), "mtmd": stt_state()},
        ]
    )
    result, _studio = run_tool(monkeypatch, studio, "unload_model", {"kind": "stt"})
    assert result["structuredContent"] == {"kind": "stt", "model": "qwen3-asr", "unloaded": True}
    assert queries(studio, "/api/inference/audio/stt/unload") == [
        {"engine": "mtmd", "model": "qwen3-asr", "wait": "true"}
    ]
    (unload,) = [c for c in studio.state.calls if c[1].endswith("/unload")]
    assert unload[3] == b""


def test_stt_unload_of_a_model_that_is_not_loaded(monkeypatch):
    studio = _stt_studio([{"transformers": stt_state(loaded = WHISPER)}])
    result, _studio = run_tool(
        monkeypatch, studio, "unload_model", {"kind": "stt", "model": "other"}
    )
    assert result["structuredContent"] == {"kind": "stt", "model": "other", "unloaded": False}
    assert [c for c in studio.state.calls if c[1].endswith("/unload")] == []
