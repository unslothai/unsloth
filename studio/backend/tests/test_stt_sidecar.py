# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from types import SimpleNamespace

from core.inference.stt_sidecar import WhisperSttSidecar


class _FakeWhisperModel:
    def __init__(self) -> None:
        self.options = None

    def transcribe(self, _audio, **options):
        self.options = options
        return iter([SimpleNamespace(start = 0, text = " hello")]), SimpleNamespace(
            language = "en", duration = 0.5
        )


def test_fast_transcription_uses_low_latency_decode_options(monkeypatch):
    sidecar = WhisperSttSidecar()
    model = _FakeWhisperModel()
    monkeypatch.setattr(sidecar, "load", lambda _model: model)

    result = sidecar.transcribe(b"encoded audio", language = "en", fast = True)

    assert result["text"] == "hello"
    assert model.options == {
        "language": "en",
        "beam_size": 1,
        "vad_filter": True,
        "condition_on_previous_text": False,
        "best_of": 1,
        "temperature": 0.0,
        "without_timestamps": True,
    }


def test_accurate_transcription_keeps_beam_search_default(monkeypatch):
    sidecar = WhisperSttSidecar()
    model = _FakeWhisperModel()
    monkeypatch.setattr(sidecar, "load", lambda _model: model)

    sidecar.transcribe(b"encoded audio")

    assert model.options == {
        "language": None,
        "beam_size": 5,
        "vad_filter": True,
        "condition_on_previous_text": False,
    }
