# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""generate_video's controls reach /v1/videos, and the job's seed comes back on generate_video and get_job."""

import base64

import pytest
from fastapi.responses import JSONResponse, Response

from .mcp_harness import PNG, PNG_URL, WEBP, b64, bodies, data_url, fake_studio, run_tool
from .test_mcp_tools_video import _job

_PNG_DATA_URL = data_url(PNG, "image/png")
_WAV = b"RIFF\x24\x00\x00\x00WAVEfmt fake"


def _studio(job = None):
    job = job or _job("video-1", "queued", 0, seed = 1234)
    return fake_studio(
        {
            ("POST", "/v1/videos"): job,
            ("GET", "/v1/videos/video-1"): job,
            ("GET", "/v1/videos/video-1/content"): Response(WEBP, media_type = "image/webp"),
            ("POST", "/v1/audio/inputs"): JSONResponse({"id": "in-1"}, status_code = 201),
        }
    )


def test_every_control_lands_in_the_body(monkeypatch):
    args = {
        "prompt": "a singer",
        "seconds": 4,
        "negative_prompt": "blur",
        "seed": 7,
        "steps": 20,
        "guidance": 4.5,
        "fps": 24,
        "reference_images": [PNG_URL, PNG_URL],
        "reference_audios": [
            {"data_base64": b64(_WAV), "filename": "voice.wav"},
            {"clip_id": "clip-1"},
            {"voice_id": "voice-1"},
        ],
    }
    result, studio = run_tool(monkeypatch, _studio(), "generate_video", args)
    assert result["isError"] is False, result
    (body,) = bodies(studio, "/v1/videos")
    assert body == {
        "prompt": "a singer",
        "seconds": 4,
        "negative_prompt": "blur",
        "seed": 7,
        "steps": 20,
        "guidance": 4.5,
        "fps": 24,
        "reference_images": [_PNG_DATA_URL, _PNG_DATA_URL],
        "reference_audios": [{"input_id": "in-1"}, {"clip_id": "clip-1"}, {"voice_id": "voice-1"}],
    }
    upload = next(c for c in studio.state.calls if c[1] == "/v1/audio/inputs")
    assert upload[3] == _WAV


def test_last_frame_num_frames_and_a_string_seconds_are_forwarded(monkeypatch):
    args = {"prompt": "x", "seconds": "4", "first_frame": PNG_URL, "last_frame": PNG_URL}
    result, studio = run_tool(monkeypatch, _studio(), "generate_video", args)
    assert result["isError"] is False, result
    body = bodies(studio, "/v1/videos")[0]
    assert body["seconds"] == "4"
    assert body["last_frame"] == body["input_reference"] == _PNG_DATA_URL
    result, studio = run_tool(
        monkeypatch, _studio(), "generate_video", {"prompt": "x", "num_frames": 49}
    )
    assert bodies(studio, "/v1/videos") == [{"prompt": "x", "num_frames": 49}]


def test_audio_without_a_reference_image_is_refused_before_any_call(monkeypatch):
    args = {"prompt": "x", "reference_audios": [{"data_base64": b64(_WAV), "filename": "v.wav"}]}
    result, studio = run_tool(monkeypatch, _studio(), "generate_video", args)
    assert result["isError"] is True
    assert "reference_audios need a reference image to go with" in result["content"][0]["text"]
    assert studio.state.calls == []


@pytest.mark.parametrize(
    "args",
    [
        {"reference_images": [PNG_URL] * 10},
        {"reference_images": [PNG_URL], "reference_audios": [{"clip_id": "c"}] * 4},
        {"seed": 2**53},
        {"steps": 0},
        {"fps": 121},
    ],
)
def test_out_of_range_controls_never_reach_a_route(monkeypatch, args):
    result, studio = run_tool(monkeypatch, _studio(), "generate_video", {"prompt": "x", **args})
    assert result["isError"] is True
    assert studio.state.calls == []


def test_the_seed_comes_back_on_the_job_and_on_get_job(monkeypatch):
    result, _studio_app = run_tool(monkeypatch, _studio(), "generate_video", {"prompt": "x"})
    assert result["structuredContent"]["seed"] == 1234
    done = _job("video-1", "completed", 100, seed = 1234)
    result, _studio_app = run_tool(
        monkeypatch, _studio(done), "get_job", {"kind": "video", "id": "video-1"}
    )
    assert result["structuredContent"]["seed"] == 1234
    assert base64.b64decode(result["content"][1]["data"]) == WEBP


def test_images_share_the_one_request_budget_of_video_requests(monkeypatch):
    import studio_mcp.tools.video as video_tool

    # The budget covers first_frame, last_frame and reference_images together.
    monkeypatch.setattr(video_tool, "VIDEO_INPUT_REFERENCE_MAX_BYTES", 2 * len(PNG))
    for ok in (
        {"prompt": "x", "first_frame": PNG_URL, "last_frame": PNG_URL},
        {"prompt": "x", "reference_images": [PNG_URL] * 2},
    ):
        result, studio = run_tool(monkeypatch, _studio(), "generate_video", ok)
        assert result["isError"] is False, result
    monkeypatch.setattr(video_tool, "VIDEO_INPUT_REFERENCE_MAX_BYTES", 2 * len(PNG) - 1)
    for args in (
        {"prompt": "x", "first_frame": PNG_URL, "last_frame": PNG_URL},
        {"prompt": "x", "reference_images": [PNG_URL] * 2},
    ):
        result, studio = run_tool(monkeypatch, _studio(), "generate_video", args)
        assert result["isError"] is True
        assert "larger than the 32 MB one video request takes" in result["content"][0]["text"]
        assert studio.state.calls == []


def test_a_body_over_the_route_cap_is_refused_before_the_call(monkeypatch):
    import studio_mcp.tools.video as video_tool

    # The route counts the encoded body, so a long prompt beside the images counts too.
    monkeypatch.setattr(video_tool, "VIDEO_INPUT_REFERENCE_JSON_MAX_BYTES", 1024)
    args = {"prompt": "x" * 2048, "reference_images": [PNG_URL]}
    result, studio = run_tool(monkeypatch, _studio(), "generate_video", args)
    assert result["isError"] is True
    assert studio.state.calls == []


def test_reference_audio_over_the_video_cap_is_refused_before_any_upload(monkeypatch, tmp_path):
    import studio_mcp.tools.video as video_tool
    from fastmcp.exceptions import ToolError
    from studio_mcp.inputs import AudioInput

    from .mcp_harness import make_caller

    monkeypatch.setattr(video_tool, "MAX_REFERENCE_AUDIO_BYTES", len(_WAV) - 1)
    audio = {"data_base64": b64(_WAV), "filename": "v.wav"}
    args = {"prompt": "x", "reference_images": [PNG_URL], "reference_audios": [audio]}
    result, studio = run_tool(monkeypatch, _studio(), "generate_video", args)
    assert result["isError"] is True
    assert "larger than the 24 MB a video takes" in result["content"][0]["text"]
    assert studio.state.calls == []
    # A local file is sized from disk, before audio_ref reads or uploads it.
    clip = tmp_path / "voice.wav"
    clip.write_bytes(_WAV)
    with pytest.raises(ToolError, match = "24 MB"):
        video_tool._refuse_large_audio(make_caller(), AudioInput(path = str(clip)))


@pytest.mark.parametrize(
    "args, text",
    [
        ({"seconds": 4, "num_frames": 49}, "not both"),
        ({"last_frame": PNG_URL}, "last_frame cannot be combined"),
        ({"first_frame": PNG_URL}, "first_frame cannot be combined"),
    ],
)
def test_combinations_the_route_refuses_never_upload_audio(monkeypatch, args, text):
    audio = {"data_base64": b64(_WAV), "filename": "v.wav"}
    body = {"prompt": "x", "reference_images": [PNG_URL], "reference_audios": [audio], **args}
    result, studio = run_tool(monkeypatch, _studio(), "generate_video", body)
    assert result["isError"] is True
    assert text in result["content"][0]["text"]
    assert studio.state.calls == []


def test_fractional_seconds_reach_the_route(monkeypatch):
    result, studio = run_tool(
        monkeypatch, _studio(), "generate_video", {"prompt": "x", "seconds": 2.5}
    )
    assert result["isError"] is False, result
    assert bodies(studio, "/v1/videos") == [{"prompt": "x", "seconds": 2.5}]
