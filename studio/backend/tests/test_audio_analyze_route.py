# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""POST /audio/analyze: metrics for audio named by id, and a transcript match when asked."""

from __future__ import annotations

import sys
import wave
from pathlib import Path

import numpy as np
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from core.inference import audio_inputs, stt_capabilities
from routes import inference

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

from test_account_media_isolation import ALICE, isolated  # noqa: E402,F401
from test_audio_run_extras import app_for  # noqa: E402

CLIP = "c" * 32
FIELDS = {
    "duration_s", "sample_rate", "channels", "integrated_lufs", "true_peak_dbtp",
    "clipped_samples", "clipping_ratio", "speech_start_s", "speech_end_s", "transcript",
    "match_score", "stt_model",
}  # fmt: skip


def _wav(
    path,
    seconds = 1.0,
    rate = 16000,
    amplitude = 0.1,
):
    t = np.arange(int(seconds * rate)) / rate
    pcm = np.round(amplitude * np.sin(2 * np.pi * 1000 * t) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())
    return path


@pytest.fixture
def source(monkeypatch, tmp_path):
    """The file the request's id resolves to; ``heard`` is what speech-to-text returns."""
    state = {"path": _wav(tmp_path / "clip.wav"), "heard": "hello world", "stt": []}
    monkeypatch.setattr(
        audio_inputs,
        "resolve_source",
        lambda ref: audio_inputs.Source("clip", ref["clip_id"], state["path"], "Clip"),
    )
    monkeypatch.setattr(audio_inputs, "prepared_path", lambda src, rate: src.path)
    monkeypatch.setattr(stt_capabilities, "capabilities_for", lambda *_a: {"family": "whisper"})
    monkeypatch.setattr(inference, "_stt_resident_model", lambda engine: None)

    async def transcribe(
        raw,
        model,
        *args,
        source_path = None,
        **_kwargs,
    ):
        state["stt"].append((model, source_path))
        if isinstance(state["heard"], Exception):
            raise state["heard"]
        return {"text": state["heard"]}

    monkeypatch.setattr(inference, "_transcribe_audio_result", transcribe)
    return state


def _post(body, path = "/api/inference/audio/analyze"):
    with TestClient(app_for(ALICE)) as client:
        return client.post(path, json = body)


@pytest.mark.parametrize("path", ["/api/inference/audio/analyze", "/v1/audio/analyze"])
def test_metrics_come_back_on_both_mounts_without_stt(source, path):
    response = _post({"clip_id": CLIP}, path)
    assert response.status_code == 200, response.text
    body = response.json()
    assert set(body) == FIELDS
    assert (body["duration_s"], body["sample_rate"], body["channels"]) == (1.0, 16000, 1)
    assert body["integrated_lufs"] == pytest.approx(-23.0, abs = 0.1)
    assert (body["transcript"], body["match_score"], body["stt_model"]) == (None, None, None)
    assert source["stt"] == []


def test_expected_text_scores_the_transcript(source):
    from core.inference.stt_sidecar import DEFAULT_STT_MODEL

    exact = _post({"clip_id": CLIP, "expected_text": "Hello, world."}).json()
    assert (exact["transcript"], exact["match_score"]) == ("hello world", 1.0)
    assert exact["stt_model"] == DEFAULT_STT_MODEL
    assert source["stt"][0] == (DEFAULT_STT_MODEL, source["path"])
    wrong = _post({"clip_id": CLIP, "expected_text": "goodbye cruel world"}).json()
    assert wrong["match_score"] < 1.0


def test_a_resident_stt_model_is_used_first(source, monkeypatch):
    monkeypatch.setattr(
        inference, "_stt_resident_model", lambda engine: "m/asr" if engine == "mtmd" else None
    )
    body = _post({"clip_id": CLIP, "expected_text": "hello world"}).json()
    assert body["stt_model"] == "m/asr" and source["stt"][0][0] == "m/asr"


def test_an_stt_refusal_passes_through(source):
    source["heard"] = HTTPException(status_code = 409, detail = "Model is not downloaded.")
    response = _post({"clip_id": CLIP, "expected_text": "hello"})
    assert (response.status_code, response.json()["detail"]) == (409, "Model is not downloaded.")


def test_non_wav_audio_is_decoded_to_a_temporary_wav(source, monkeypatch, tmp_path):
    source["path"] = tmp_path / "clip.webm"
    source["path"].write_bytes(b"\x1aE\xdf\xa3 not a wav")
    made = []

    def transcode(
        src,
        dst,
        *,
        max_seconds = None,
        **_kwargs,
    ):
        made.append((src, dst, max_seconds))
        _wav(dst, seconds = 2.0)

    monkeypatch.setattr(audio_inputs, "transcode", transcode)
    response = _post({"clip_id": CLIP})
    assert response.status_code == 200 and response.json()["duration_s"] == 2.0
    ((src, dst, cap),) = made
    assert (src, cap) == (source["path"], 600.0) and dst.name.startswith(".analyze-")
    assert not dst.exists()


def test_long_audio_and_bad_requests_are_refused(source, tmp_path):
    source["path"] = _wav(tmp_path / "long.wav", seconds = 601, rate = 100)
    response = _post({"clip_id": CLIP})
    assert (response.status_code, response.json()["detail"]) == (
        413,
        "Audio is longer than 10 minutes.",
    )
    assert _post({"clip_id": CLIP, "input_id": CLIP}).status_code == 422
    assert _post({"clip_id": CLIP, "expected_text": "x" * 20001}).status_code == 422
