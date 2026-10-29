# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import json

import pytest
from fastapi.responses import JSONResponse

from models.inference import AudioRunRequest
from studio_mcp.tools import audio

from .mcp_harness import bodies, fake_studio, run_tool, sequence
from .test_mcp_tools_audio import GALLERY, _clip

RUN = ("POST", "/v1/audio/run")
STATUS = ("GET", "/api/inference/status")
CANCEL = "/api/inference/audio/cancel"
SONG = {"text": "lo-fi", "mode": "song", "lyrics": "la", "variations": 3, "cancel_id": "run-abc12"}


def _status(how):
    mode = {"id": "song", "duration": {"min": 10, "max": 240, "default": 30}}
    return {
        "model_identifier": "ace-step",
        "audio_music": {"modes": [{**mode, "variations": {"max": 4, "how": how}}]},
    }


def _take(
    clip_id,
    seed = 7,
    **fields,
):
    return {"clips": [_clip(clip_id, seed = seed)], "model": "ace-step", **fields}


@pytest.fixture(autouse = True)
def fresh_runs():
    audio._RUNS.clear()
    audio._CANCELLED.clear()
    yield
    audio._RUNS.clear()
    audio._CANCELLED.clear()


def _music(
    monkeypatch,
    run,
    how = "sequential",
    args = SONG,
    studio = None,
):
    studio = studio or fake_studio(
        {RUN: run, STATUS: _status(how), ("POST", CANCEL): {"cancelled": 1}, **GALLERY}
    )
    return run_tool(monkeypatch, studio, "generate_audio", {"workflow": "music", **args})


def test_a_cancel_during_take_two_keeps_take_one_and_sends_no_third(monkeypatch):
    def run(request, body):
        if json.loads(body)["take"] == 1:
            return _take("var-1", seed = 41)
        assert audio.stop_run("owner", "run-abc12") is False
        return JSONResponse({"detail": "Audio generation cancelled"}, status_code = 499)

    result, studio = _music(monkeypatch, run)
    assert result["isError"] is False, result
    out = result["structuredContent"]
    assert [c["id"] for c in out["clips"]] == ["var-1"]
    assert out["warnings"] == ["Stopped after take 1 of 3: cancelled."]
    assert out["cancel_id"] == "run-abc12"
    sent = bodies(studio, "/v1/audio/run")
    assert [b["take"] for b in sent] == [1, 2]
    assert len({b["group_id"] for b in sent}) == 1 == len({out["group_id"], sent[0]["group_id"]})
    # Take 2 reuses the seed take 1 drew; neither asks for variations.
    assert "seed" not in sent[0] and sent[1]["seed"] == 41
    for body in sent:
        assert "variations" not in body and body["cancel_id"] == "run-abc12"
        AudioRunRequest.model_validate(body)


def test_a_cancel_as_a_take_ends_stops_before_the_next(monkeypatch):
    def run(request, body):
        audio.stop_run("owner", "run-abc12")
        return _take("var-1")

    result, studio = _music(monkeypatch, run)
    assert [c["id"] for c in result["structuredContent"]["clips"]] == ["var-1"]
    assert len(bodies(studio, "/v1/audio/run")) == 1

    # The id stays spent as long as the route keeps an early cancel for it.
    again, studio = _music(monkeypatch, _take("var-2"))
    assert again["isError"] is True
    assert "pass a new cancel_id" in again["content"][0]["text"]
    assert bodies(studio, "/v1/audio/run") == []


@pytest.mark.parametrize("how", ["batch", None])
def test_a_batch_model_gets_one_request_with_variations(monkeypatch, how):
    two = {"clips": [_clip("var-1"), _clip("var-2")], "group_id": "grp-9", "model": "ace-step"}
    args = {key: value for key, value in SONG.items() if key != "cancel_id"}
    result, studio = _music(monkeypatch, two, how = how, args = args)
    (body,) = bodies(studio, "/v1/audio/run")
    assert body["variations"] == 3 and "take" not in body
    assert body["cancel_id"].startswith("mcp-")
    assert result["structuredContent"]["cancel_id"] == body["cancel_id"]
    assert result["structuredContent"]["group_id"] == "grp-9"


def test_takes_report_ignored_options_applied_and_warnings_once(monkeypatch):
    ignored = "Ignored option temperature: the model does not take it."
    clamped = "duration_s 75 clamped to 10"
    extra = {"ignored_options": ["temperature"], "applied": {"duration_s": 10.0, "flag": True}}
    answers = [_take(f"var-{n}", warnings = [ignored, clamped], **extra) for n in (1, 2)]
    answers.append(JSONResponse({"detail": "Broke."}, status_code = 500))
    result, _studio = _music(monkeypatch, sequence(*answers))
    out = result["structuredContent"]
    assert [c["id"] for c in out["clips"]] == ["var-1", "var-2"]
    assert out["ignored_options"] == ["temperature"]
    assert out["applied"] == {"duration_s": 10.0}
    assert out["warnings"][:2] == [ignored, clamped]
    assert out["warnings"][2].startswith("Take 3 of 3 failed: Broke.")


def test_more_than_four_variations_is_refused_before_any_take(monkeypatch):
    result, studio = _music(monkeypatch, _take("c1"), args = {**SONG, "variations": 6})
    assert result["isError"] is True
    assert "variations must be 1 to 4" in result["content"][0]["text"]
    assert bodies(studio, "/v1/audio/run") == []


def test_a_failed_first_take_is_the_tool_error(monkeypatch):
    stopped = {"code": "runtime_stopped", "message": "The audio runtime stopped: exit 9."}
    run = JSONResponse({"detail": stopped}, status_code = 500)
    result, studio = _music(monkeypatch, run)
    assert result["isError"] is True
    assert result["content"][0]["text"].startswith("[runtime_stopped] The audio runtime stopped")
    assert len(bodies(studio, "/v1/audio/run")) == 1


def test_a_dead_runtime_answer_tells_the_agent_to_retry_once(monkeypatch):
    stopped = {"code": "runtime_stopped", "message": "The audio runtime stopped: exit 9."}
    failed = JSONResponse({"detail": stopped}, status_code = 500)
    studio = fake_studio({RUN: sequence(failed, _take("clip-1")), **GALLERY})
    args = {"workflow": "speak", "text": "Hi"}
    first, _ = run_tool(monkeypatch, studio, "generate_audio", args)
    assert first["isError"] is True and "Retry once" in first["content"][0]["text"]
    retried, _ = run_tool(monkeypatch, studio, "generate_audio", args)
    assert retried["isError"] is False
    assert [c["id"] for c in retried["structuredContent"]["clips"]] == ["clip-1"]


CANCEL_PAYLOADS = {("POST", CANCEL): {"cancelled": 1}}


def test_cancel_audio_with_no_mcp_run_asks_the_route(monkeypatch):
    args = {"kind": "audio", "id": "run-x"}
    result, studio = run_tool(monkeypatch, CANCEL_PAYLOADS, "cancel", args)
    assert result["structuredContent"]["cancelled"] is True
    assert bodies(studio, CANCEL) == [{"cancel_id": "run-x"}]


def test_cancel_audio_between_takes_stops_the_run_without_the_route(monkeypatch):
    run = audio._RUNS[("owner", "run-abc12")] = audio._Run()
    args = {"kind": "audio", "id": "run-abc12"}
    result, studio = run_tool(monkeypatch, CANCEL_PAYLOADS, "cancel", args)
    assert result["structuredContent"]["cancelled"] is True
    assert "before its next take" in result["structuredContent"]["message"]
    assert run.cancelled is True
    assert studio.state.calls == []


def test_cancel_audio_during_a_take_stops_it_through_the_route(monkeypatch):
    audio._RUNS[("owner", "run-abc12")] = audio._Run(in_flight = True)
    routes = {("POST", CANCEL): {"cancelled": 0}}
    args = {"kind": "audio", "id": "run-abc12"}
    result, studio = run_tool(monkeypatch, routes, "cancel", args)
    assert result["structuredContent"]["cancelled"] is True
    assert bodies(studio, CANCEL) == [{"cancel_id": "run-abc12"}]


def test_cancel_audio_needs_an_id(monkeypatch):
    result, studio = run_tool(monkeypatch, {}, "cancel", {"kind": "audio"}, fallback = {})
    assert result["isError"] is True and "needs id" in result["content"][0]["text"]
    assert studio.state.calls == []


def test_an_unsaved_oversized_later_take_keeps_the_saved_takes(monkeypatch):
    from studio_mcp.media import INLINE_CAP

    from .mcp_harness import WAV, b64

    big = WAV + b"\x00" * INLINE_CAP
    unsaved = {"clips": [], "model": "ace-step", "audio": {"data": b64(big), "format": "wav"}}
    result, studio = _music(monkeypatch, sequence(_take("var-1"), unsaved))
    assert result["isError"] is False, result
    out = result["structuredContent"]
    assert [c["id"] for c in out["clips"]] == ["var-1"] and out["group_id"]
    assert "too large to return inline" in out["warnings"][-1]
    assert len(bodies(studio, "/v1/audio/run")) == 2


def _beside_chat(how):
    """Qwen serves chat; the music model is kept beside it, so the bare status describes Qwen."""
    chat = {
        "active_model": "unsloth/Qwen-GGUF",
        "model_identifier": "unsloth/Qwen-GGUF",
        "serving": ["unsloth/Qwen-GGUF", "ace-step"],
        "serving_checkpoints": ["unsloth/Qwen-GGUF", "ace-step"],
    }
    music = {**_status(how), "active_model": "ace-step"}
    return lambda request, body: music if request.query_params.get("model") == "ace-step" else chat


@pytest.mark.parametrize("named", [False, True])
def test_takes_follow_the_music_model_kept_beside_the_chat_model(monkeypatch, named):
    takes = iter(range(1, 10))
    studio = fake_studio(
        {
            RUN: lambda request, body: _take(f"var-{next(takes)}"),
            STATUS: _beside_chat("sequential"),
            **GALLERY,
        }
    )
    args = {**SONG, "model": "ace-step"} if named else SONG
    result, studio = _music(monkeypatch, None, args = args, studio = studio)
    assert result["isError"] is False, result
    # One request per take, as the kept model makes them, not one batch it would refuse.
    assert [b.get("take") for b in bodies(studio, "/v1/audio/run")] == [1, 2, 3]
