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


def test_an_unsaved_clip_over_the_inline_cap_is_not_inlined(monkeypatch):
    from studio_mcp.media import INLINE_CAP

    big = WAV + b"\x00" * INLINE_CAP
    fallback = {"model": "kokoro", "clips": [], "audio": {"data": _b64(big), "format": "wav"}}
    studio = _studio({("POST", "/v1/audio/run"): lambda request, body: fallback})
    result = _call(monkeypatch, studio, {"workflow": "speak", "text": "Hi"})
    assert result["isError"] is True
    assert "too large to return inline" in result["content"][0]["text"]
    assert all(c["type"] != "audio" for c in result["content"])


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


def test_oversized_audio_is_refused_before_any_upload(monkeypatch, tmp_path):
    # Only a file path can carry this much: inline data stops at the 4 MiB request limit.
    huge = tmp_path / "huge.wav"
    with huge.open("wb") as handle:
        handle.truncate(200 * 1024 * 1024 + 1)
    studio = _studio()
    result = _call(
        monkeypatch,
        studio,
        {"workflow": "clone", "text": "Hi", "reference": {"path": str(huge)}},
        base_url = "http://127.0.0.1:8888",
        client = ("127.0.0.1", 50000),
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


# ---------------------------------------------------------------- transcribe

TRANSCRIPT = {"text": "Hello from Studio."}
VERBOSE = {
    "task": "transcribe",
    "language": "en",
    "duration": 2.0,
    "text": "Hello from Studio.",
    "segments": [
        {"id": 0, "start": 0.0, "end": 1.2, "text": "Hello"},
        {"id": 1, "start": 1.2, "end": 2.0, "text": "from Studio."},
    ],
}
TRANSCRIBE_PAYLOADS = {
    ("POST", "/v1/audio/transcriptions"): VERBOSE,
    ("POST", "/v1/audio/translations"): TRANSCRIPT,
}


def _form(request_body: bytes, content_type: str):
    from email.parser import BytesParser
    from email.policy import default

    message = BytesParser(policy = default).parsebytes(
        b"Content-Type: " + content_type.encode() + b"\r\n\r\n" + request_body
    )
    fields = {}
    for part in message.iter_parts():
        name = part.get_param("name", header = "content-disposition")
        fields[name] = (part.get_filename(), part.get_payload(decode = True))
    return fields


def _transcribe(monkeypatch, studio, args, **client):
    with TestClient(served(create_studio_mcp(), studio, monkeypatch = monkeypatch), **client) as http:
        return call_tool(http, "transcribe", args)


def _sent_form(studio, path):
    (_m, _p, headers, body) = next(c for c in studio.state.calls if c[1] == path)
    return _form(body, headers["content-type"])


def test_small_audio_goes_as_multipart_with_openai_field_names(monkeypatch):
    studio = fake_studio({("POST", "/v1/audio/transcriptions"): lambda request, body: TRANSCRIPT})
    result = _transcribe(
        monkeypatch,
        studio,
        {
            "audio": {"data_base64": _b64(WAV), "filename": "memo.wav"},
            "language": "en",
            "model": "openai/whisper-small",
        },
    )
    assert result["structuredContent"] == {
        "text": "Hello from Studio.",
        "language": None,
        "model": "openai/whisper-small",
        "segments": None,
        "saved_to_history": False,
    }
    form = _sent_form(studio, "/v1/audio/transcriptions")
    assert form["file"] == ("memo.wav", WAV)
    assert {k: v[1] for k, v in form.items() if k != "file"} == {
        "response_format": b"json",
        "model": b"openai/whisper-small",
        "language": b"en",
    }


def test_timestamps_ask_for_verbose_json_and_return_segments(monkeypatch):
    studio = fake_studio({("POST", "/v1/audio/transcriptions"): lambda request, body: VERBOSE})
    result = _transcribe(
        monkeypatch,
        studio,
        {
            "audio": {"data_base64": _b64(WAV), "filename": "a.wav"},
            "timestamps": True,
            "language": "en",
        },
    )
    out = result["structuredContent"]
    assert out["language"] == "en"
    assert out["segments"] == [
        {"start": 0.0, "end": 1.2, "text": "Hello"},
        {"start": 1.2, "end": 2.0, "text": "from Studio."},
    ]
    assert _sent_form(studio, "/v1/audio/transcriptions")["response_format"][1] == b"verbose_json"


def test_translate_uses_the_translations_route_without_language(monkeypatch):
    studio = fake_studio({("POST", "/v1/audio/translations"): lambda request, body: TRANSCRIPT})
    _transcribe(
        monkeypatch,
        studio,
        {
            "audio": {"data_base64": _b64(WAV), "filename": "a.wav"},
            "translate": True,
            "language": "de",
        },
    )
    form = _sent_form(studio, "/v1/audio/translations")
    assert "language" not in form
    assert [c[1] for c in studio.state.calls] == ["/v1/audio/translations"]


def test_a_501_for_timestamps_says_what_to_change(monkeypatch):
    def unsupported(request, body):
        return JSONResponse(
            {
                "error": {
                    "message": "verbose_json reports the language of the audio and the local STT engine cannot detect it.",
                    "type": "api_error",
                    "param": None,
                    "code": None,
                }
            },
            status_code = 501,
        )

    studio = fake_studio({("POST", "/v1/audio/transcriptions"): unsupported})
    result = _transcribe(
        monkeypatch,
        studio,
        {"audio": {"data_base64": _b64(WAV), "filename": "a.wav"}, "timestamps": True},
    )
    assert result["isError"] is True
    assert result["content"][0]["text"].endswith(
        "(HTTP 501) Pass language with timestamps, or use an audio.cpp speech-to-text model."
    )


def test_a_missing_model_is_downloaded_then_the_transcription_retried_once(monkeypatch):
    from studio_mcp import loading

    monkeypatch.setattr(loading, "POLL_INTERVAL_S", 0.01)
    answers = [
        JSONResponse(
            {
                "error": {
                    "message": "STT model 'openai/whisper-small' is not downloaded. Download it in Settings, then Voice, before loading it.",
                    "type": "conflict_error",
                    "param": None,
                    "code": None,
                }
            },
            status_code = 409,
        ),
        JSONResponse(TRANSCRIPT),
    ]
    statuses = [
        {
            "transformers": {
                "models": ["openai/whisper-small"],
                "downloaded_models": [],
                "default_model": "openai/whisper-small",
                "download": {"downloading": False},
            }
        },
        {
            "transformers": {
                "models": ["openai/whisper-small"],
                "downloaded_models": ["openai/whisper-small"],
                "download": {"downloading": False, "completed_download_ids": ["d1"]},
            }
        },
    ]
    studio = fake_studio(
        {
            ("POST", "/v1/audio/transcriptions"): lambda request, body: answers.pop(0),
            ("GET", "/api/inference/audio/stt/status"): lambda request, body: statuses.pop(0)
            if len(statuses) > 1
            else statuses[0],
            ("POST", "/api/inference/audio/stt/download"): lambda request, body: {
                "downloading": True,
                "download_id": "d1",
            },
        }
    )
    result = _transcribe(
        monkeypatch, studio, {"audio": {"data_base64": _b64(WAV), "filename": "a.wav"}}
    )
    assert result["structuredContent"]["text"] == "Hello from Studio."
    assert [c[1] for c in studio.state.calls] == [
        "/v1/audio/transcriptions",
        "/api/inference/audio/stt/status",
        "/api/inference/audio/stt/download",
        "/api/inference/audio/stt/status",
        "/v1/audio/transcriptions",
    ]
    download = next(c for c in studio.state.calls if c[1].endswith("/download"))
    assert json.loads(download[3]) == {"model": "openai/whisper-small", "engine": "transformers"}


def test_a_second_refusal_is_not_retried_again(monkeypatch):
    from studio_mcp import loading

    monkeypatch.setattr(loading, "POLL_INTERVAL_S", 0.01)
    refusal = {
        "error": {
            "message": "STT model 'x' is not downloaded.",
            "type": "conflict_error",
            "param": None,
            "code": None,
        }
    }
    studio = fake_studio(
        {
            ("POST", "/v1/audio/transcriptions"): lambda request, body: JSONResponse(
                refusal, status_code = 409
            ),
            ("GET", "/api/inference/audio/stt/status"): lambda request, body: {
                "transformers": {
                    "models": ["x"],
                    "downloaded_models": ["x"],
                    "download": {"downloading": False},
                }
            },
            ("POST", "/api/inference/audio/stt/download"): lambda request, body: {
                "downloading": True
            },
        }
    )
    result = _transcribe(
        monkeypatch,
        studio,
        {"audio": {"data_base64": _b64(WAV), "filename": "a.wav"}, "model": "x"},
    )
    assert result["isError"] is True
    assert [c[1] for c in studio.state.calls].count("/v1/audio/transcriptions") == 2


def test_a_remote_agent_cannot_transcribe_a_path(monkeypatch, tmp_path):
    source = tmp_path / "mcp-input.wav"
    source.write_bytes(WAV)
    opened = []
    monkeypatch.setattr(type(source), "read_bytes", lambda self: opened.append(self) or WAV)
    studio = fake_studio({("POST", "/v1/audio/transcriptions"): lambda request, body: TRANSCRIPT})
    remote = _transcribe(
        monkeypatch,
        studio,
        {"audio": {"path": str(source)}},
        base_url = "http://192.168.1.20:8888",
        client = ("192.0.2.7", 50000),
    )
    assert remote["content"][0]["text"] == PATH_REMOTE
    assert opened == [] and studio.state.calls == []
    local = _transcribe(
        monkeypatch,
        studio,
        {"audio": {"path": str(source)}},
        base_url = "http://127.0.0.1:8888",
        client = ("127.0.0.1", 50000),
    )
    assert local["structuredContent"]["text"] == "Hello from Studio."
    assert _sent_form(studio, "/v1/audio/transcriptions")["file"] == ("mcp-input.wav", WAV)


def test_transcribe_annotations():
    tool = {t.name: t for t in asyncio.run(create_studio_mcp().list_tools())}["transcribe"]
    assert tool.annotations.readOnlyHint is False
    assert tool.annotations.destructiveHint is False
    assert tool.annotations.openWorldHint is False
    assert tool.output_schema is not None


# ---------------------------------------------------------------- large or stored audio

STT_STATUS = {
    "transformers": {
        "loaded_model": None,
        "models": ["openai/whisper-small"],
        "downloaded_models": ["openai/whisper-small"],
        "default_model": "openai/whisper-small",
    },
    "gguf": {
        "loaded_model": "whisper-large-v3-turbo-q5",
        "models": ["whisper-large-v3-turbo-q5"],
        "downloaded_models": ["whisper-large-v3-turbo-q5"],
    },
}
COMPLETE = {
    "type": "complete",
    "text": "A long meeting.",
    "language": "en",
    "segments": [{"start": 0.0, "end": 5.0, "text": "A long meeting."}],
    "record": {"id": "tr-1"},
}


def _ndjson(*events):
    return Response(
        b"".join(json.dumps(e).encode() + b"\n" for e in events), media_type = "application/x-ndjson"
    )


SOURCE_PAYLOADS = {
    ("GET", "/api/inference/audio/stt/status"): STT_STATUS,
    ("POST", "/api/inference/audio/transcribe/source"): COMPLETE,
    ("POST", "/v1/audio/inputs"): UPLOADED,
}


def _source_studio(events = None, status = None):
    return fake_studio(
        {
            ("GET", "/api/inference/audio/stt/status"): lambda request, body: status or STT_STATUS,
            ("POST", "/api/inference/audio/transcribe/source"): lambda request, body: _ndjson(
                *(
                    events
                    or [{"type": "progress", "fraction": 0.5}, {"type": "heartbeat"}, COMPLETE]
                )
            ),
            ("POST", "/v1/audio/inputs"): lambda request, body: JSONResponse(
                UPLOADED, status_code = 201
            ),
        }
    )


def test_a_stored_clip_is_transcribed_by_id_with_the_loaded_model(monkeypatch):
    studio = _source_studio()
    result = _transcribe(monkeypatch, studio, {"audio": {"clip_id": "clip-7"}, "timestamps": True})
    assert result["structuredContent"] == {
        "text": "A long meeting.",
        "language": "en",
        "model": "whisper-large-v3-turbo-q5",
        "segments": [{"start": 0.0, "end": 5.0, "text": "A long meeting."}],
        "saved_to_history": True,
    }
    body = json.loads(next(c for c in studio.state.calls if c[1].endswith("/transcribe/source"))[3])
    assert body == {
        "source": {"clip_id": "clip-7"},
        "model": "whisper-large-v3-turbo-q5",
        "engine": "gguf",
        "timestamps": True,
    }


def test_without_a_loaded_model_the_curated_default_is_used(monkeypatch):
    idle = {**STT_STATUS, "gguf": {**STT_STATUS["gguf"], "loaded_model": None}}
    studio = _source_studio(status = idle)
    _transcribe(monkeypatch, studio, {"audio": {"input_id": "in-3"}, "language": "en"})
    body = json.loads(next(c for c in studio.state.calls if c[1].endswith("/transcribe/source"))[3])
    assert body["model"] == "openai/whisper-small"
    assert body["engine"] == "transformers"
    assert body["language"] == "en"


def test_audio_over_25_mib_is_uploaded_then_transcribed(monkeypatch):
    from studio_mcp.tools import audio

    monkeypatch.setattr(audio, "MULTIPART_LIMIT", len(WAV) - 1)
    studio = _source_studio()
    result = _transcribe(
        monkeypatch, studio, {"audio": {"data_base64": _b64(WAV), "filename": "meeting.wav"}}
    )
    assert result["structuredContent"]["saved_to_history"] is True
    assert [c[1] for c in studio.state.calls] == [
        "/v1/audio/inputs",
        "/api/inference/audio/stt/status",
        "/api/inference/audio/transcribe/source",
    ]
    upload = studio.state.calls[0]
    assert upload[2]["content-length"] == str(len(WAV))
    body = json.loads(studio.state.calls[2][3])
    assert body["source"] == {"input_id": "in-1"}


def test_an_ndjson_error_line_is_a_tool_error(monkeypatch):
    studio = _source_studio(
        events = [
            {"type": "progress"},
            {"type": "error", "message": "Audio is longer than 30 minutes."},
        ]
    )
    result = _transcribe(monkeypatch, studio, {"audio": {"clip_id": "clip-7"}})
    assert result["isError"] is True
    assert result["content"][0]["text"] == "Audio is longer than 30 minutes."


def test_a_stream_that_stops_early_is_an_error(monkeypatch):
    studio = _source_studio(events = [{"type": "progress"}, {"type": "heartbeat"}])
    result = _transcribe(monkeypatch, studio, {"audio": {"clip_id": "clip-7"}})
    assert result["isError"] is True


def test_translation_of_large_or_stored_audio_is_refused(monkeypatch):
    studio = _source_studio()
    result = _transcribe(monkeypatch, studio, {"audio": {"clip_id": "clip-7"}, "translate": True})
    assert result["isError"] is True
    assert "under 25 MB" in result["content"][0]["text"]
    assert studio.state.calls == []
