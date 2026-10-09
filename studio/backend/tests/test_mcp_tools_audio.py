# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
import base64
import json

import pytest
from fastapi.responses import JSONResponse, Response
from fastapi.testclient import TestClient

from mcp_server import create_studio_mcp
from models.inference import AudioRunRequest
from studio_mcp.inputs import PATH_REMOTE

from .mcp_harness import call_tool, fake_studio, served

WAV = b"RIFF\x24\x00\x00\x00WAVEfmt " + b"\x00" * 64


def _clip(
    clip_id,
    role = "output",
    duration = 1.0,
    rate = 24000,
):
    return {
        "id": clip_id,
        "role": role,
        "url": f"/v1/audio/gallery/{clip_id}/file",
        "sample_rate": rate,
        "duration_s": duration,
        "workflow": "speak",
    }


RUN = {"clips": [_clip("clip-1")], "group_id": None, "model": "unsloth/orpheus-3b", "audio": None}
UPLOADED = {
    "id": "in-1",
    "name": "ref.wav",
    "duration_s": 3.0,
    "sample_rate": 24000,
    "channels": 1,
    "url": "/v1/audio/inputs/in-1/file",
    "expires_at": "2026-10-10T00:00:00Z",
}

PAYLOADS = {
    ("POST", "/v1/audio/run"): RUN,
    ("POST", "/v1/audio/inputs"): UPLOADED,
}


def _wav(request, body):
    return Response(WAV, media_type = "audio/wav")


def _studio(overrides = None, upload_status = 201):
    routes = {("POST", "/v1/audio/run"): lambda request, body: RUN}
    routes[("POST", "/v1/audio/inputs")] = lambda request, body: JSONResponse(
        UPLOADED, status_code = upload_status
    )
    for clip_id in ("clip-1", "vocals", "drums", "bass", "other", "var-1", "var-2"):
        routes[("GET", f"/v1/audio/gallery/{clip_id}/file")] = _wav
    routes.update(overrides or {})
    return fake_studio(routes)


def _call(monkeypatch, studio, args, **client):
    with TestClient(served(create_studio_mcp(), studio, monkeypatch = monkeypatch), **client) as http:
        return call_tool(http, "generate_audio", args)


def _run_body(studio):
    return json.loads(next(c for c in studio.state.calls if c[1] == "/v1/audio/run")[3])


def _b64(data):
    return base64.b64encode(data).decode()


WORKFLOWS = [
    ("clone", {"text": "Hello there", "reference": {"clip_id": "ref-1"}, "reference_text": "Hi"}),
    ("speak", {"text": "Hello there", "reference": {"voice_id": "voice-1"}, "language": "en"}),
    (
        "edit",
        {
            "text": "Hello world",
            "source": {"clip_id": "clip-0"},
            "edit": {"mode": "words", "markup": "Hello world"},
        },
    ),
    (
        "convert",
        {
            "source": {"clip_id": "clip-0"},
            "target": {"voice_id": "voice-1"},
            "convert": {"mode": "singing"},
        },
    ),
    (
        "music",
        {
            "text": "lo-fi piano",
            "mode": "song",
            "lyrics": "la la",
            "duration_s": 30,
            "variations": 2,
        },
    ),
    ("separate", {"source": {"input_id": "in-9"}}),
]


@pytest.mark.parametrize("workflow,args", WORKFLOWS)
def test_each_workflow_sends_a_body_the_route_accepts(monkeypatch, workflow, args):
    studio = _studio()
    result = _call(monkeypatch, studio, {"workflow": workflow, **args})
    assert result["isError"] is False, result
    body = _run_body(studio)
    AudioRunRequest.model_validate(body)
    assert body["workflow"] == workflow
    assert "voice_id" not in body


def test_a_saved_voice_is_the_reference(monkeypatch):
    studio = _studio()
    _call(
        monkeypatch,
        studio,
        {"workflow": "speak", "text": "Hi", "reference": {"voice_id": "voice-1"}},
    )
    assert _run_body(studio)["inputs"] == {"reference": {"voice_id": "voice-1"}}


def test_separate_returns_every_stem(monkeypatch):
    stems = {
        "clips": [_clip(stem, role = stem) for stem in ("vocals", "drums", "bass", "other")],
        "group_id": "grp-1",
        "model": "htdemucs",
    }
    studio = _studio({("POST", "/v1/audio/run"): lambda request, body: stems})
    result = _call(monkeypatch, studio, {"workflow": "separate", "source": {"clip_id": "song-1"}})
    out = result["structuredContent"]
    assert [c["role"] for c in out["clips"]] == ["vocals", "drums", "bass", "other"]
    assert out["group_id"] == "grp-1"
    assert [c["type"] for c in result["content"][1:]] == ["audio"] * 4


def test_music_variations_share_a_group(monkeypatch):
    music = {"clips": [_clip("var-1"), _clip("var-2")], "group_id": "grp-9", "model": "ace-step"}
    studio = _studio({("POST", "/v1/audio/run"): lambda request, body: music})
    result = _call(
        monkeypatch, studio, {"workflow": "music", "text": "jazz", "mode": "song", "variations": 2}
    )
    assert result["structuredContent"]["group_id"] == "grp-9"
    assert len(result["structuredContent"]["clips"]) == 2
    assert _run_body(studio)["variations"] == 2


def test_short_clips_are_inline_and_long_ones_are_links_never_fetched(monkeypatch):
    long_clip = _clip("long-1", duration = 600.0, rate = 44100)
    fetched = []
    run = {"clips": [_clip("clip-1"), long_clip], "model": "m"}
    studio = _studio(
        {
            ("POST", "/v1/audio/run"): lambda request, body: run,
            ("GET", "/v1/audio/gallery/long-1/file"): lambda request, body: fetched.append(1)
            or Response(WAV),
        }
    )
    result = _call(
        monkeypatch,
        studio,
        {"workflow": "speak", "text": "Hi"},
        base_url = "http://192.168.1.20:8888",
    )
    content = result["content"]
    assert [c["type"] for c in content[1:]] == ["audio", "resource_link"]
    assert base64.b64decode(content[1]["data"]) == WAV
    assert content[2]["uri"] == "http://192.168.1.20:8888/v1/audio/gallery/long-1/file"
    assert fetched == []
    assert (
        result["structuredContent"]["clips"][0]["url"]
        == "http://192.168.1.20:8888/v1/audio/gallery/clip-1/file"
    )


def test_the_history_fallback_comes_back_inline_and_unsaved(monkeypatch):
    fallback = {
        "clips": [],
        "model": "m",
        "audio": {"data": _b64(WAV), "format": "wav", "sample_rate": 24000},
    }
    studio = _studio({("POST", "/v1/audio/run"): lambda request, body: fallback})
    result = _call(monkeypatch, studio, {"workflow": "speak", "text": "Hi"})
    assert result["structuredContent"]["saved"] is False
    assert result["structuredContent"]["clips"] == []
    assert base64.b64decode(result["content"][1]["data"]) == WAV


@pytest.mark.parametrize("status", [201, 200])
def test_inline_audio_is_uploaded_with_a_content_length(monkeypatch, status):
    studio = _studio(upload_status = status)
    result = _call(
        monkeypatch,
        studio,
        {
            "workflow": "clone",
            "text": "Hi",
            "reference": {"data_base64": _b64(WAV), "filename": "ref.wav"},
        },
    )
    assert result["isError"] is False
    (_m, path, headers, body) = next(c for c in studio.state.calls if c[1] == "/v1/audio/inputs")
    assert headers["content-length"] == str(len(WAV))
    assert body == WAV
    assert _run_body(studio)["inputs"]["reference"] == {"input_id": "in-1"}


def test_oversized_audio_is_refused_before_any_upload(monkeypatch):
    studio = _studio()
    huge = "A" * (4 * (200 * 1024 * 1024 // 3) + 8)
    result = _call(
        monkeypatch,
        studio,
        {
            "workflow": "clone",
            "text": "Hi",
            "reference": {"data_base64": huge, "filename": "a.wav"},
        },
    )
    assert result["isError"] is True
    assert "larger than 200 MiB" in result["content"][0]["text"]
    assert studio.state.calls == []


@pytest.mark.parametrize(
    "reference",
    [{}, {"clip_id": "a", "voice_id": "b"}, {"data_base64": "AAAA"}, {"clip_id": "../etc"}],
)
def test_an_audio_input_takes_exactly_one_valid_source(monkeypatch, reference):
    studio = _studio()
    result = _call(monkeypatch, studio, {"workflow": "clone", "text": "Hi", "reference": reference})
    assert result["isError"] is True
    assert studio.state.calls == []


def test_a_remote_agent_cannot_send_an_audio_path(monkeypatch, tmp_path):
    source = tmp_path / "mcp-input.wav"
    source.write_bytes(WAV)
    opened = []
    monkeypatch.setattr(type(source), "read_bytes", lambda self: opened.append(self) or WAV)
    studio = _studio()
    remote = _call(
        monkeypatch,
        studio,
        {"workflow": "separate", "source": {"path": str(source)}},
        base_url = "http://192.168.1.20:8888",
        client = ("192.0.2.7", 50000),
    )
    assert remote["content"][0]["text"] == PATH_REMOTE
    assert opened == []
    assert studio.state.calls == []
    local = _call(
        monkeypatch,
        studio,
        {"workflow": "separate", "source": {"path": str(source)}},
        base_url = "http://127.0.0.1:8888",
        client = ("127.0.0.1", 50000),
    )
    assert local["isError"] is False
    assert opened == [source]
    upload = next(c for c in studio.state.calls if c[1] == "/v1/audio/inputs")
    assert upload[3] == WAV


def test_file_like_options_are_refused_by_the_route(monkeypatch):
    def refused(request, body):
        return JSONResponse(
            {
                "error": {
                    "message": "options: Value error, Option 'speaker_wav' is not accepted; name audio by id in inputs.",
                    "type": "invalid_request_error",
                    "param": "options",
                    "code": None,
                }
            },
            status_code = 400,
        )

    studio = _studio({("POST", "/v1/audio/run"): refused})
    result = _call(
        monkeypatch,
        studio,
        {"workflow": "speak", "text": "Hi", "options": {"speaker_wav": "/srv/x.wav"}},
    )
    assert result["isError"] is True
    assert "Option 'speaker_wav' is not accepted" in result["content"][0]["text"]
    assert result["content"][0]["text"].startswith("Invalid arguments:")


def test_no_loaded_model_says_what_to_load(monkeypatch):
    def none(request, body):
        return JSONResponse(
            {
                "error": {
                    "message": "No model loaded.",
                    "type": "invalid_request_error",
                    "param": None,
                    "code": None,
                }
            },
            status_code = 400,
        )

    studio = _studio({("POST", "/v1/audio/run"): none})
    result = _call(monkeypatch, studio, {"workflow": "speak", "text": "Hi"})
    assert result["content"][0]["text"] == (
        "No model loaded. (HTTP 400) Load a text-to-speech or music model with load_model(kind='llm') first."
    )


def test_generate_audio_annotations():
    tool = {t.name: t for t in asyncio.run(create_studio_mcp().list_tools())}["generate_audio"]
    assert tool.annotations.readOnlyHint is False
    assert tool.annotations.destructiveHint is False
    assert tool.annotations.openWorldHint is False
    assert "text" in tool.parameters["properties"]
    assert tool.output_schema["additionalProperties"] is False
