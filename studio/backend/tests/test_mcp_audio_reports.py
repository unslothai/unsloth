# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import pytest
from fastapi.responses import JSONResponse

from .mcp_harness import WAV_INPUT, bodies, run_tool
from .test_mcp_tools_audio import (
    COMPLETE,
    PAYLOADS as AUDIO_PAYLOADS,
    STT_STATUS,
    STT_STATUS_ROUTE,
    TRANSCRIPT,
    TRANSCRIPTIONS,
    UPLOADED,
    _from_source,
    _memo,
)
from .test_mcp_tools_status import PAYLOADS as STATUS_PAYLOADS

ANALYZE = ("POST", "/api/inference/audio/analyze")
ANALYSIS = {
    "duration_s": 2.5,
    "sample_rate": 24000,
    "channels": 1,
    "integrated_lufs": -23.0,
    "true_peak_dbtp": -20.0,
    "clipped_samples": 0,
    "clipping_ratio": 0.0,
    "speech_start_s": 0.5,
    "speech_end_s": 1.5,
    "transcript": "hello",
    "match_score": 1.0,
    "stt_model": "openai/whisper-small",
}
UPLOAD = ("POST", "/v1/audio/inputs")
ANALYZE_PAYLOADS = {ANALYZE: ANALYSIS, UPLOAD: AUDIO_PAYLOADS[UPLOAD]}


def test_status_lists_the_active_music_models_duration_bounds(monkeypatch):
    chat = STATUS_PAYLOADS[("GET", "/api/inference/status")]
    music = {
        "modes": [
            {"id": "song", "duration": {"min": 10, "max": 240, "default": 30}},
            {"id": "edit", "actions": ["repaint"]},
            {"id": "sfx", "duration": {"min": 1, "max": 10.5, "default": True}},
        ]
    }
    routes = {**STATUS_PAYLOADS, ("GET", "/api/inference/status"): {**chat, "audio_music": music}}
    result, _studio = run_tool(monkeypatch, routes, "studio_status")
    active, other = result["structuredContent"]["chat"]["loaded"]
    assert active["audio_music"] == [
        {"id": "song", "min_s": 10.0, "max_s": 240.0, "default_s": 30.0},
        {"id": "sfx", "min_s": 1.0, "max_s": 10.5, "default_s": None},
    ]
    assert other["audio_music"] is None


DIARIZED = {
    **COMPLETE,
    "segments": [
        {"start": 0.0, "end": 2.0, "text": "Hi.", "speaker": "S0"},
        {"start": 2.0, "end": 5.0, "text": "Hello.", "speaker": "S1"},
    ],
    "speakers": [{"id": "S0", "label": "Speaker 1"}, {"id": "S1", "label": "Speaker 2"}],
}


def test_speakers_label_segments_and_small_audio_goes_by_id(monkeypatch):
    result, studio = _from_source(monkeypatch, {**_memo(), "speakers": True}, [DIARIZED])
    out = result["structuredContent"]
    assert [(s["speaker"], s["text"]) for s in out["segments"]] == [
        ("Speaker 1", "Hi."),
        ("Speaker 2", "Hello."),
    ]
    (body,) = bodies(studio, "/api/inference/audio/transcribe/source")
    assert body["source"] == {"input_id": UPLOADED["id"]}
    assert body["speakers"] is True and body["timestamps"] is False
    assert "/v1/audio/transcriptions" not in [c[1] for c in studio.state.calls]


def test_speakers_with_translate_is_refused_before_any_route(monkeypatch):
    args = {**_memo(), "translate": True, "speakers": True}
    result, studio = run_tool(monkeypatch, {}, "transcribe", args, fallback = {})
    assert result["isError"] is True
    assert "leave out timestamps and speakers" in result["content"][0]["text"]
    assert studio.state.calls == []


@pytest.mark.parametrize(
    "expected,score", [("Hello from Studio", 1.0), ("Goodbye from Studio.", 2 / 3)]
)
def test_expected_text_scores_a_multipart_transcript(monkeypatch, expected, score):
    routes = {TRANSCRIPTIONS: TRANSCRIPT, STT_STATUS_ROUTE: {}}
    args = {**_memo(), "model": "openai/whisper-small", "expected_text": expected}
    result, _studio = run_tool(monkeypatch, routes, "transcribe", args)
    assert result["structuredContent"]["match_score"] == pytest.approx(score, abs = 1e-4)


def test_expected_text_scores_a_stored_clip(monkeypatch):
    args = {"audio": {"clip_id": "clip-7"}, "expected_text": "A long meeting"}
    result, _studio = _from_source(monkeypatch, args, status = STT_STATUS)
    assert result["structuredContent"]["match_score"] == 1.0


def test_analyze_audio_forwards_the_clip_and_returns_the_metrics(monkeypatch):
    args = {"audio": {"clip_id": "clip-7"}, "expected_text": "hello"}
    result, studio = run_tool(monkeypatch, ANALYZE_PAYLOADS, "analyze_audio", args)
    assert result["structuredContent"] == ANALYSIS
    assert bodies(studio, ANALYZE[1]) == [{"clip_id": "clip-7", "expected_text": "hello"}]


def test_analyze_audio_uploads_inline_audio_and_leaves_out_unasked_text(monkeypatch):
    silent = {**ANALYSIS, "integrated_lufs": None, "transcript": None, "match_score": None}
    routes = {**ANALYZE_PAYLOADS, ANALYZE: silent}
    result, studio = run_tool(monkeypatch, routes, "analyze_audio", {"audio": WAV_INPUT})
    assert result["structuredContent"]["integrated_lufs"] is None
    assert bodies(studio, ANALYZE[1]) == [{"input_id": UPLOADED["id"]}]


def test_analyze_audio_without_the_stt_model_says_how_to_get_it(monkeypatch):
    missing = JSONResponse(
        {"detail": "STT model 'openai/whisper-small' is not downloaded."}, status_code = 409
    )
    args = {"audio": {"clip_id": "clip-7"}, "expected_text": "hello"}
    result, _studio = run_tool(monkeypatch, {ANALYZE: missing}, "analyze_audio", args)
    assert result["isError"] is True
    assert result["content"][0]["text"].endswith("or leave out expected_text.")
