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
    call_with_progress,
    fake_studio,
    fast_polls,  # noqa: F401  (fixture)
    form,
    openai_error,
    run_tool,
    sequence,
    slow,
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
STEMS = ("vocals", "drums", "bass", "other")
SEPARATED = {"clips": [_clip(s, role = s) for s in STEMS], "group_id": "grp-1", "model": "htdemucs"}
# (workflow, args, answer, what else the sent body and the result must show).
WORKFLOWS = [
    ("clone", {"text": "Hi", "reference": {"clip_id": "ref-1"}, "reference_text": "Hi"}, RUN, None),
    (
        "speak",
        {"text": "Hi", "reference": {"voice_id": "voice-1"}, "language": "en"},
        RUN,
        lambda body, result: body["inputs"] == {"reference": {"voice_id": "voice-1"}},
    ),
    (
        "edit",
        {"text": "Hi", "source": {"clip_id": "clip-0"}, "edit": {"mode": "words", "markup": "Hi"}},
        RUN,
        None,
    ),
    (
        "convert",
        {
            "source": {"clip_id": "clip-0"},
            "target": {"voice_id": "voice-1"},
            "convert": {"mode": "singing"},
        },
        RUN,
        None,
    ),
    (
        "music",
        {"text": "lo-fi", "mode": "song", "lyrics": "la", "duration_s": 30, "variations": 2},
        MUSIC,
        lambda body, result: (
            body["variations"] == 2
            and result["structuredContent"]["group_id"] == "grp-9"
            and len(result["structuredContent"]["clips"]) == 2
        ),
    ),
    ("separate", {"source": {"input_id": "in-9"}}, RUN, None),
    (
        "separate",
        {"source": {"clip_id": "song-1"}},
        SEPARATED,
        lambda body, result: (
            [c["role"] for c in result["structuredContent"]["clips"]] == list(STEMS)
            and result["structuredContent"]["group_id"] == "grp-1"
            and [c["type"] for c in result["content"][1:]] == ["audio"] * 4
        ),
    ),
]


@pytest.mark.parametrize("workflow,args,run,expect", WORKFLOWS)
def test_each_workflow_sends_a_body_the_route_accepts(monkeypatch, workflow, args, run, expect):
    result, studio = _generate(monkeypatch, {"workflow": workflow, **args}, run)
    assert result["isError"] is False, result
    (body,) = bodies(studio, "/v1/audio/run")
    AudioRunRequest.model_validate(body)
    assert body["workflow"] == workflow
    assert "voice_id" not in body
    assert expect is None or expect(body, result)


VOICE = {"name": "voice", "type": "enum", "values": ["vivian", "ryan"], "default": "vivian"}
CUSTOM_VOICE = {"audio_options": [VOICE, {"name": "temperature", "type": "float"}]}


# The route drops an enum value it does not list, so "Ryan" would speak in the default voice.
@pytest.mark.parametrize(
    "voice,sent,refusal",
    [
        ("Ryan", "ryan", None),
        ("ryan", "ryan", None),
        ("Bob", None, "'Bob' is not one of: vivian, ryan"),
    ],
)
def test_an_enum_option_is_sent_as_the_model_lists_it(monkeypatch, voice, sent, refusal):
    args = {"workflow": "speak", "text": "Hi", "options": {"voice": voice, "temperature": 0.7}}
    routes = {("GET", "/api/inference/status"): CUSTOM_VOICE}
    result, studio = _generate(monkeypatch, args, routes = routes)
    if refusal:
        assert result["isError"] is True and refusal in result["content"][0]["text"]
        assert bodies(studio, "/v1/audio/run") == []
    else:
        (body,) = bodies(studio, "/v1/audio/run")
        assert body["options"] == {"voice": sent, "temperature": 0.7}


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


MISSING = "STT model 'openai/whisper-small' is not downloaded. Download it in Settings, then Voice, before loading it."


def _memo(filename = "a.wav", **args):
    return {"audio": {"data_base64": b64(WAV), "filename": filename}, **args}


def test_without_a_model_the_resident_one_is_named(monkeypatch):
    routes = {
        TRANSCRIPTIONS: TRANSCRIPT,
        STT_STATUS_ROUTE: {"transformers": {"loaded_model": "small", "loading": False}},
    }
    result, studio = run_tool(monkeypatch, routes, "transcribe", _memo("memo.wav"))
    assert result["structuredContent"]["model"] == "small"
    assert result["structuredContent"]["language"] is None
    assert form(studio, "/v1/audio/transcriptions")["model"][1] == b"small"


def test_small_audio_runs_on_the_loaded_non_transformers_model(monkeypatch):
    # Without the id the route would run its Transformers default and the result would name Qwen3-ASR.
    asr = "unsloth/Qwen3-ASR-0.6B-GGUF"
    routes = {
        TRANSCRIPTIONS: TRANSCRIPT,
        STT_STATUS_ROUTE: {"mtmd": {"loaded_model": asr, "loading": False}},
    }
    result, studio = run_tool(monkeypatch, routes, "transcribe", _memo())
    assert result["structuredContent"]["model"] == asr
    assert form(studio, "/v1/audio/transcriptions")["model"][1] == asr.encode()


SEGMENTS = [
    {"start": 0.0, "end": 1.2, "text": "Hello"},
    {"start": 1.2, "end": 2.0, "text": "from Studio."},
]


UNTIMED = {key: value for key, value in VERBOSE.items() if key != "segments"}


# A text-only answer reports the language asked for; timestamps ask for verbose_json and always
# come back as a list, empty when the route timed no speech.
@pytest.mark.parametrize(
    "answer,model,timestamps,segments",
    [
        (TRANSCRIPT, "openai/whisper-small", False, None),
        (VERBOSE, None, True, SEGMENTS),
        (UNTIMED, None, True, []),
    ],
)
def test_small_audio_goes_as_multipart_with_openai_field_names(
    monkeypatch, answer, model, timestamps, segments
):
    args = _memo("memo.wav", language = "en", model = model, timestamps = timestamps)
    result, studio = run_tool(monkeypatch, {TRANSCRIPTIONS: answer}, "transcribe", args)
    out = result["structuredContent"]
    assert (out["text"], out["language"]) == ("Hello from Studio.", "en")
    assert out["segments"] == segments
    assert out["saved_to_history"] is False
    fields = form(studio, "/v1/audio/transcriptions")
    assert fields["file"] == ("memo.wav", WAV)
    assert {k: v[1] for k, v in fields.items() if k != "file"} == {
        "response_format": b"verbose_json" if timestamps else b"json",
        # Without it the route returns verbose_json with no segments at all.
        **({"timestamp_granularities[]": b"segment"} if timestamps else {}),
        **({"model": model.encode()} if model else {}),
        "language": b"en",
    }
    if model:
        assert out["model"] == model


def test_translate_drops_language_and_reports_english_however_spelled(monkeypatch):
    verbose = {**VERBOSE, "task": "translate", "language": "english"}
    args = _memo(translate = True, language = "de")
    result, studio = run_tool(monkeypatch, {TRANSLATIONS: verbose}, "transcribe", args)
    assert result["structuredContent"]["language"] == "en"
    assert "language" not in form(studio, "/v1/audio/translations")
    assert [c[1] for c in studio.state.calls] == ["/v1/audio/translations"]


def test_a_missing_model_is_downloaded_then_the_transcription_retried_once(monkeypatch, fast_polls):
    refusal = openai_error(MISSING, 409, type = "conflict_error")
    routes = {
        TRANSCRIPTIONS: sequence(refusal, JSONResponse(TRANSCRIPT)),
        STT_STATUS_ROUTE: sequence(MISSING_STATE, DONE_STATE),
        STT_DOWNLOAD: {"downloading": True, "download_id": "d1"},
    }
    result, studio = run_tool(monkeypatch, routes, "transcribe", _memo())
    assert result["structuredContent"]["text"] == "Hello from Studio."
    assert [c[1] for c in studio.state.calls] == [
        # Nothing loaded, so the route's default is asked for.
        "/api/inference/audio/stt/status",
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


IDLE = {**STT_STATUS, "gguf": {**STT_STATUS["gguf"], "loaded_model": None}}


# Without a loaded model the curated default is used.
@pytest.mark.parametrize(
    "status,args,model,engine",
    [
        (STT_STATUS, {"audio": {"clip_id": "clip-7"}, "timestamps": True}, TURBO, "gguf"),
        (IDLE, {"audio": {"input_id": "in-3"}, "language": "en"}, WHISPER, "transformers"),
    ],
)
def test_a_stored_clip_is_transcribed_by_id_with_the_loaded_model(
    monkeypatch, status, args, model, engine
):
    result, studio = _from_source(monkeypatch, args, status = status)
    timestamps = args.get("timestamps", False)
    assert result["structuredContent"] == {
        "text": "A long meeting.",
        "language": "en",
        "model": model,
        "segments": [{"start": 0.0, "end": 5.0, "text": "A long meeting."}] if timestamps else None,
        "saved_to_history": True,
    }
    extra = {"language": "en"} if "language" in args else {}
    assert bodies(studio, "/api/inference/audio/transcribe/source") == [
        {
            "source": args["audio"],
            "model": model,
            "engine": engine,
            "timestamps": timestamps,
            **extra,
        }
    ]


def test_small_audio_keeps_a_loaded_whisper_cpp_model_on_its_engine(monkeypatch):
    # The id is shared with Transformers, so multipart would quietly run another engine.
    result, studio = _from_source(monkeypatch, _memo())
    assert result["structuredContent"]["model"] == TURBO
    assert "/v1/audio/transcriptions" not in [c[1] for c in studio.state.calls]
    assert bodies(studio, "/api/inference/audio/transcribe/source")[0]["engine"] == "gguf"


SHARED = {
    "transformers": stt_state(downloaded = [WHISPER], models = [WHISPER]),
    "gguf": stt_state(loaded = WHISPER, downloaded = [WHISPER], models = [WHISPER]),
}


# Stored audio, and a small file naming the model: both stay on whisper.cpp.
@pytest.mark.parametrize("audio", [{"clip_id": "clip-7"}, _memo()["audio"]])
def test_a_shared_whisper_id_runs_on_the_engine_holding_it(monkeypatch, audio):
    args = {"audio": audio, "model": WHISPER}
    result, studio = _from_source(monkeypatch, args, status = SHARED)
    assert result["structuredContent"]["model"] == WHISPER
    assert "/v1/audio/transcriptions" not in [c[1] for c in studio.state.calls]
    assert bodies(studio, "/api/inference/audio/transcribe/source")[0]["engine"] == "gguf"


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


@pytest.mark.parametrize(
    "events,message",
    [
        (
            [
                {"type": "progress"},
                {"type": "error", "message": "Audio is longer than 30 minutes."},
            ],
            "Audio is longer than 30 minutes.",
        ),
        ([{"type": "progress"}, {"type": "heartbeat"}], None),
    ],
)
def test_an_ndjson_error_line_or_early_stop_is_a_tool_error(monkeypatch, events, message):
    result, _studio = _from_source(monkeypatch, {"audio": {"clip_id": "clip-7"}}, events)
    assert result["isError"] is True
    assert message is None or result["content"][0]["text"] == message


def test_a_slow_song_sends_a_heartbeat_and_still_returns_its_clip(monkeypatch):
    from studio_mcp import progress as progress_module

    monkeypatch.setattr(progress_module, "HEARTBEAT_S", 0.05)
    studio = fake_studio({("POST", "/v1/audio/run"): slow(RUN, 0.3), **GALLERY})
    progress, result = call_with_progress(
        monkeypatch, studio, "generate_audio", {"workflow": "speak", "text": "Hi"}
    )
    assert result["isError"] is False
    assert result["structuredContent"]["clips"][0]["id"] == "clip-1"
    assert len(progress) >= 2
    assert all(p["message"].startswith("Generating audio") for p in progress)
