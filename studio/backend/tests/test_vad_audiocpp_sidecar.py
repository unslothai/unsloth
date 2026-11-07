# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import pytest

from core.inference import audio_cpp_models as acm
from core.inference import vad_audiocpp_sidecar as vad


def test_segments_are_reported_in_seconds_with_the_voiced_total():
    result = vad.segments_from_result(
        {
            "segments": [
                {"start_sample": 1600, "end_sample": 8000, "confidence": 0.9},
                {"start_sample": 9600, "end_sample": 9600},
                {"start_sample": 16000, "end_sample": 24000},
            ],
            "sample_rate": 16000,
        },
        fallback_rate = 8000,
    )
    assert result["sample_rate"] == 16000
    assert [(s["start_s"], s["end_s"]) for s in result["segments"]] == [(0.1, 0.5), (1.0, 1.5)]
    assert result["speech_ms"] == 900.0


def test_a_clip_with_no_segments_has_no_speech():
    result = vad.segments_from_result({"timing": {"wall_ms": 3.0}}, fallback_rate = 16000)
    assert result == {"segments": [], "speech_ms": 0.0, "sample_rate": 16000}


def test_the_default_detector_is_the_pulsevad_folder():
    assert vad.DEFAULT_VAD_MODEL == f"{acm.AUDIO_CPP_REPO}/PulseVAD-GGUF"
    assert acm.family_policy("pulsevad").task == "vad"


def test_only_vad_models_resolve(monkeypatch):
    kokoro = acm.family_policy("kokoro_tts")
    fake = acm.AudioCppModel(
        id = f"{acm.AUDIO_CPP_REPO}/Kokoro-82M-GGUF",
        repo_id = acm.AUDIO_CPP_REPO,
        folder = "Kokoro-82M-GGUF",
        display_name = "Kokoro-82M-GGUF",
        family = kokoro.family,
        task = kokoro.task,
        server_task = kokoro.default_server_task,
        variant = acm.AudioCppVariant("Q8_0", (), "x.gguf"),
        variants = (),
        default_variant = "Q8_0",
    )
    monkeypatch.setattr(vad, "resolve", lambda *a, **k: fake)
    with pytest.raises(vad.VadError, match = "not a voice activity detection model"):
        vad.resolve_vad_model(fake.id)
    with pytest.raises(vad.VadError, match = "not an audio GGUF"):
        vad.resolve_vad_model("not a model id at all!!")


def test_a_detect_keeps_the_server_it_loaded_until_the_run_ends(monkeypatch):
    import io
    import threading
    import wave

    sidecar = vad.AudioCppVadSidecar()
    held_during_load = []

    def lock_free_elsewhere() -> bool:
        box = []
        probe = threading.Thread(target = lambda: box.append(sidecar._lock.acquire(blocking = False)))
        probe.start()
        probe.join()
        if box[0]:
            sidecar._lock.release()
        return box[0]

    class Server:
        model_id = "pulsevad"
        process = type("P", (), {"poll": lambda self: None})()

        def post_json(self, path, body, timeout):
            # Another request's load would need the lock here; it must still be held.
            held_during_load.append(not lock_free_elsewhere())
            return 200, b'{"segments": [], "sample_rate": 16000}'

    def load(model):
        held_during_load.append(not lock_free_elsewhere())
        sidecar._server = Server()
        return "pulsevad"

    monkeypatch.setattr(sidecar, "load", load)
    monkeypatch.setattr(vad, "_debug_dump", lambda *a: None)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * 1600)
    result = sidecar.detect(buf.getvalue(), "pulsevad")
    assert result["model"] == "pulsevad"
    assert held_during_load == [True, True]
