# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import base64
import json

import pytest
from fastapi.responses import JSONResponse, Response

from models.inference import AudioRunRequest

from .mcp_harness import (
    REMOTE,
    WAV,
    WHISPER,
    b64,
    bodies,
    call_to,
    fast_polls,  # noqa: F401  (fixture)
    form,
    openai_error,
    run_tool,
    sequence,
    stt_state,
)


def _clip(clip_id, **fields):
    return {
        "id": clip_id,
        "role": "output",
        "url": f"/v1/audio/gallery/{clip_id}/file",
        "sample_rate": 24000,
        "duration_s": 1.0,
        "workflow": "speak",
        **fields,
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

PAYLOADS = {("POST", "/v1/audio/run"): RUN, ("POST", "/v1/audio/inputs"): UPLOADED}


def _wav(request, body):
    return Response(WAV, media_type = "audio/wav")


GALLERY = {
    ("GET", f"/v1/audio/gallery/{clip_id}/file"): _wav
    for clip_id in ("clip-1", "vocals", "drums", "bass", "other", "var-1", "var-2")
}


def _generate(
    monkeypatch,
    args,
    run = RUN,
    routes = None,
    upload_status = 201,
    **client,
):
    studio = {
        ("POST", "/v1/audio/run"): run,
        ("POST", "/v1/audio/inputs"): JSONResponse(UPLOADED, status_code = upload_status),
        **GALLERY,
        **(routes or {}),
    }
    return run_tool(monkeypatch, studio, "generate_audio", args, **client)


MUSIC = {"clips": [_clip("var-1"), _clip("var-2")], "group_id": "grp-9", "model": "ace-step"}
# (workflow, args, what else the sent body and the result must show). Music answers with MUSIC.
WORKFLOWS = [
    (
        "clone",
        {"text": "Hello there", "reference": {"clip_id": "ref-1"}, "reference_text": "Hi"},
        None,
    ),
    (
        "speak",
        {"text": "Hello there", "reference": {"voice_id": "voice-1"}, "language": "en"},
        # A saved voice is the reference.
        lambda body, out: body["inputs"] == {"reference": {"voice_id": "voice-1"}},
    ),
    (
        "edit",
        {
            "text": "Hello world",
            "source": {"clip_id": "clip-0"},
            "edit": {"mode": "words", "markup": "Hello world"},
        },
        None,
    ),
    (
        "convert",
        {
            "source": {"clip_id": "clip-0"},
            "target": {"voice_id": "voice-1"},
            "convert": {"mode": "singing"},
        },
        None,
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
        # Variations share a group.
        lambda body, out: (
            body["variations"] == 2 and out["group_id"] == "grp-9" and len(out["clips"]) == 2
        ),
    ),
    ("separate", {"source": {"input_id": "in-9"}}, None),
]


@pytest.mark.parametrize("workflow,args,expect", WORKFLOWS)
def test_each_workflow_sends_a_body_the_route_accepts(monkeypatch, workflow, args, expect):
    run = MUSIC if workflow == "music" else RUN
    result, studio = _generate(monkeypatch, {"workflow": workflow, **args}, run)
    assert result["isError"] is False, result
    (body,) = bodies(studio, "/v1/audio/run")
    AudioRunRequest.model_validate(body)
    assert body["workflow"] == workflow
    assert "voice_id" not in body
    assert expect is None or expect(body, result["structuredContent"])


def test_separate_returns_every_stem(monkeypatch):
    stems = {
        "clips": [_clip(stem, role = stem) for stem in ("vocals", "drums", "bass", "other")],
        "group_id": "grp-1",
        "model": "htdemucs",
    }
    args = {"workflow": "separate", "source": {"clip_id": "song-1"}}
    result, _studio = _generate(monkeypatch, args, stems)
    out = result["structuredContent"]
    assert [c["role"] for c in out["clips"]] == ["vocals", "drums", "bass", "other"]
    assert out["group_id"] == "grp-1"
    assert [c["type"] for c in result["content"][1:]] == ["audio"] * 4


def test_short_clips_are_inline_and_long_ones_are_links_never_fetched(monkeypatch):
    long_clip = _clip("long-1", duration_s = 600.0, sample_rate = 44100)
    fetched = []
    run = {"clips": [_clip("clip-1"), long_clip], "model": "m"}
    long_file = {
        ("GET", "/v1/audio/gallery/long-1/file"): lambda request, body: fetched.append(1)
        or Response(WAV)
    }
    result, _studio = _generate(
        monkeypatch, {"workflow": "speak", "text": "Hi"}, run, long_file, **REMOTE
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
    fallback = {"model": "kokoro", "clips": [], "audio": {"data": b64(big), "format": "wav"}}
    result, _studio = _generate(monkeypatch, {"workflow": "speak", "text": "Hi"}, fallback)
    assert result["isError"] is True
    assert "too large to return inline" in result["content"][0]["text"]
    assert all(c["type"] != "audio" for c in result["content"])


def test_the_history_fallback_comes_back_inline_and_unsaved(monkeypatch):
    fallback = {
        "clips": [],
        "model": "m",
        "audio": {"data": b64(WAV), "format": "wav", "sample_rate": 24000},
    }
    result, _studio = _generate(monkeypatch, {"workflow": "speak", "text": "Hi"}, fallback)
    assert result["structuredContent"]["saved"] is False
    assert result["structuredContent"]["clips"] == []
    assert base64.b64decode(result["content"][1]["data"]) == WAV


@pytest.mark.parametrize("status", [201, 200])
def test_inline_audio_is_uploaded_with_a_content_length(monkeypatch, status):
    args = {
        "workflow": "clone",
        "text": "Hi",
        "reference": {"data_base64": b64(WAV), "filename": "ref.wav"},
    }
    result, studio = _generate(monkeypatch, args, upload_status = status)
    assert result["isError"] is False
    (_m, path, headers, body) = call_to(studio, "/v1/audio/inputs")
    assert headers["content-length"] == str(len(WAV))
    assert body == WAV
    (run,) = bodies(studio, "/v1/audio/run")
    assert run["inputs"]["reference"] == {"input_id": "in-1"}


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
TRANSCRIPTIONS = ("POST", "/v1/audio/transcriptions")
TRANSLATIONS = ("POST", "/v1/audio/translations")
STT_STATUS_ROUTE = ("GET", "/api/inference/audio/stt/status")
STT_DOWNLOAD = ("POST", "/api/inference/audio/stt/download")
# Before and after the missing default model is downloaded.
MISSING_STATE = {"transformers": stt_state(default_model = WHISPER)}
DONE_STATE = {
    "transformers": stt_state(
        downloaded = [WHISPER],
        download = {"downloading": False, "completed_download_ids": ["d1"]},
        default_model = WHISPER,
    )
}


def _memo(filename = "a.wav", **args):
    return {"audio": {"data_base64": b64(WAV), "filename": filename}, **args}


def test_without_a_model_the_resident_one_is_named(monkeypatch):
    routes = {
        TRANSCRIPTIONS: TRANSCRIPT,
        STT_STATUS_ROUTE: {"transformers": {"loaded_model": "small", "loading": False}},
    }
    result, _studio = run_tool(monkeypatch, routes, "transcribe", _memo("memo.wav"))
    assert result["structuredContent"]["model"] == "small"
    assert result["structuredContent"]["language"] is None


def test_small_audio_goes_as_multipart_with_openai_field_names(monkeypatch):
    args = _memo("memo.wav", language = "en", model = "openai/whisper-small")
    result, studio = run_tool(monkeypatch, {TRANSCRIPTIONS: TRANSCRIPT}, "transcribe", args)
    assert result["structuredContent"] == {
        "text": "Hello from Studio.",
        # The route answers with text alone; the language asked for is reported.
        "language": "en",
        "model": "openai/whisper-small",
        "segments": None,
        "saved_to_history": False,
    }
    fields = form(studio, "/v1/audio/transcriptions")
    assert fields["file"] == ("memo.wav", WAV)
    assert {k: v[1] for k, v in fields.items() if k != "file"} == {
        "response_format": b"json",
        "model": b"openai/whisper-small",
        "language": b"en",
    }


def test_timestamps_ask_for_verbose_json_and_return_segments(monkeypatch):
    args = _memo(timestamps = True, language = "en")
    result, studio = run_tool(monkeypatch, {TRANSCRIPTIONS: VERBOSE}, "transcribe", args)
    out = result["structuredContent"]
    assert out["language"] == "en"
    assert out["segments"] == [
        {"start": 0.0, "end": 1.2, "text": "Hello"},
        {"start": 1.2, "end": 2.0, "text": "from Studio."},
    ]
    assert form(studio, "/v1/audio/transcriptions")["response_format"][1] == b"verbose_json"


def test_translate_uses_the_translations_route_without_language(monkeypatch):
    args = _memo(translate = True, language = "de")
    _result, studio = run_tool(monkeypatch, {TRANSLATIONS: TRANSCRIPT}, "transcribe", args)
    assert "language" not in form(studio, "/v1/audio/translations")
    assert [c[1] for c in studio.state.calls] == ["/v1/audio/translations"]


def test_a_translation_reports_english_however_the_route_spells_it(monkeypatch):
    verbose = {**VERBOSE, "task": "translate", "language": "english"}
    args = _memo(translate = True, timestamps = True)
    result, _studio = run_tool(monkeypatch, {TRANSLATIONS: verbose}, "transcribe", args)
    assert result["structuredContent"]["language"] == "en"


def test_a_missing_model_is_downloaded_then_the_transcription_retried_once(monkeypatch, fast_polls):
    refusal = openai_error(
        "STT model 'openai/whisper-small' is not downloaded. Download it in Settings, then Voice, before loading it.",
        409,
        type = "conflict_error",
    )
    routes = {
        TRANSCRIPTIONS: sequence(refusal, JSONResponse(TRANSCRIPT)),
        STT_STATUS_ROUTE: sequence(MISSING_STATE, DONE_STATE),
        STT_DOWNLOAD: {"downloading": True, "download_id": "d1"},
    }
    result, studio = run_tool(monkeypatch, routes, "transcribe", _memo())
    assert result["structuredContent"]["text"] == "Hello from Studio."
    assert [c[1] for c in studio.state.calls] == [
        "/v1/audio/transcriptions",
        "/api/inference/audio/stt/status",
        "/api/inference/audio/stt/download",
        "/api/inference/audio/stt/status",
        "/v1/audio/transcriptions",
        # Names the model that ran, since no model was asked for.
        "/api/inference/audio/stt/status",
    ]
    assert bodies(studio, "/api/inference/audio/stt/download") == [
        {"model": "openai/whisper-small", "engine": "transformers"}
    ]


def test_a_second_refusal_is_not_retried_again(monkeypatch, fast_polls):
    routes = {
        TRANSCRIPTIONS: lambda request, body: openai_error(
            "STT model 'x' is not downloaded.", 409, type = "conflict_error"
        ),
        STT_STATUS_ROUTE: {"transformers": stt_state(downloaded = ["x"], models = ["x"])},
        STT_DOWNLOAD: {"downloading": True},
    }
    result, studio = run_tool(monkeypatch, routes, "transcribe", _memo(model = "x"))
    assert result["isError"] is True
    assert [c[1] for c in studio.state.calls].count("/v1/audio/transcriptions") == 2


# ---------------------------------------------------------------- large or stored audio

TURBO = "whisper-large-v3-turbo-q5"
STT_STATUS = {
    "transformers": stt_state(downloaded = [WHISPER], default_model = WHISPER),
    "gguf": stt_state(loaded = TURBO, downloaded = [TURBO], models = [TURBO]),
}
COMPLETE = {
    "type": "complete",
    "text": "A long meeting.",
    "language": "en",
    "segments": [{"start": 0.0, "end": 5.0, "text": "A long meeting."}],
    "record": {"id": "tr-1"},
}
SOURCE = ("POST", "/api/inference/audio/transcribe/source")


def _ndjson(*events):
    return Response(
        b"".join(json.dumps(e).encode() + b"\n" for e in events), media_type = "application/x-ndjson"
    )


def _from_source(
    monkeypatch,
    args,
    events = None,
    status = None,
):
    events = events or [{"type": "progress", "fraction": 0.5}, {"type": "heartbeat"}, COMPLETE]
    routes = {
        STT_STATUS_ROUTE: status or STT_STATUS,
        SOURCE: _ndjson(*events),
        ("POST", "/v1/audio/inputs"): JSONResponse(UPLOADED, status_code = 201),
    }
    return run_tool(monkeypatch, routes, "transcribe", args)


def test_a_stored_clip_is_transcribed_by_id_with_the_loaded_model(monkeypatch):
    args = {"audio": {"clip_id": "clip-7"}, "timestamps": True}
    result, studio = _from_source(monkeypatch, args)
    assert result["structuredContent"] == {
        "text": "A long meeting.",
        "language": "en",
        "model": "whisper-large-v3-turbo-q5",
        "segments": [{"start": 0.0, "end": 5.0, "text": "A long meeting."}],
        "saved_to_history": True,
    }
    assert bodies(studio, "/api/inference/audio/transcribe/source") == [
        {
            "source": {"clip_id": "clip-7"},
            "model": "whisper-large-v3-turbo-q5",
            "engine": "gguf",
            "timestamps": True,
        }
    ]


def test_without_a_loaded_model_the_curated_default_is_used(monkeypatch):
    idle = {**STT_STATUS, "gguf": {**STT_STATUS["gguf"], "loaded_model": None}}
    args = {"audio": {"input_id": "in-3"}, "language": "en"}
    _result, studio = _from_source(monkeypatch, args, status = idle)
    (body,) = bodies(studio, "/api/inference/audio/transcribe/source")
    assert body["model"] == "openai/whisper-small"
    assert body["engine"] == "transformers"
    assert body["language"] == "en"


def test_audio_over_25_mib_is_uploaded_then_transcribed(monkeypatch):
    from studio_mcp.tools import audio

    monkeypatch.setattr(audio, "MULTIPART_LIMIT", len(WAV) - 1)
    result, studio = _from_source(monkeypatch, _memo("meeting.wav"))
    assert result["structuredContent"]["saved_to_history"] is True
    assert [c[1] for c in studio.state.calls] == [
        "/v1/audio/inputs",
        "/api/inference/audio/stt/status",
        "/api/inference/audio/transcribe/source",
    ]
    upload = studio.state.calls[0]
    assert upload[2]["content-length"] == str(len(WAV))
    assert bodies(studio, "/api/inference/audio/transcribe/source")[0]["source"] == {
        "input_id": "in-1"
    }


MISSING = "STT model 'openai/whisper-small' is not downloaded. Download it in Settings, then Voice, before loading it."


@pytest.mark.parametrize("as_stream", [True, False])
def test_a_missing_model_is_downloaded_before_a_stored_clip_is_retried(
    monkeypatch, fast_polls, as_stream
):
    refusal = (
        _ndjson({"type": "error", "message": MISSING})
        if as_stream
        else JSONResponse({"detail": MISSING}, status_code = 409)
    )
    routes = {
        SOURCE: sequence(refusal, _ndjson(COMPLETE)),
        STT_STATUS_ROUTE: sequence(MISSING_STATE, MISSING_STATE, DONE_STATE),
        STT_DOWNLOAD: {"downloading": True, "download_id": "d1"},
    }
    result, studio = run_tool(monkeypatch, routes, "transcribe", {"audio": {"clip_id": "clip-7"}})
    assert result["structuredContent"]["text"] == "A long meeting."
    paths = [c[1] for c in studio.state.calls]
    assert paths.count("/api/inference/audio/transcribe/source") == 2
    assert "/api/inference/audio/stt/download" in paths


def test_a_transcript_that_mentions_the_phrase_is_not_retried(monkeypatch):
    said = {**COMPLETE, "text": "The file is not downloaded yet, he said."}
    result, studio = _from_source(monkeypatch, {"audio": {"clip_id": "clip-7"}}, [said])
    assert result["structuredContent"]["text"] == "The file is not downloaded yet, he said."
    paths = [c[1] for c in studio.state.calls]
    assert paths.count("/api/inference/audio/transcribe/source") == 1
    assert "/api/inference/audio/stt/download" not in paths


def test_an_ndjson_error_line_is_a_tool_error(monkeypatch):
    events = [
        {"type": "progress"},
        {"type": "error", "message": "Audio is longer than 30 minutes."},
    ]
    result, _studio = _from_source(monkeypatch, {"audio": {"clip_id": "clip-7"}}, events)
    assert result["isError"] is True
    assert result["content"][0]["text"] == "Audio is longer than 30 minutes."


def test_a_stream_that_stops_early_is_an_error(monkeypatch):
    events = [{"type": "progress"}, {"type": "heartbeat"}]
    result, _studio = _from_source(monkeypatch, {"audio": {"clip_id": "clip-7"}}, events)
    assert result["isError"] is True
