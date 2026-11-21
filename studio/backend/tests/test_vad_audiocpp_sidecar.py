# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import pytest
from types import SimpleNamespace

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


def test_a_download_resolves_the_repo_with_the_callers_token(monkeypatch):
    from core.inference import audio_cpp_backend

    seen: list = []
    entry = SimpleNamespace(id = "org/private-vad-GGUF")

    def _resolve(
        model,
        *,
        network = False,
        hf_token = None,
    ):
        seen.append(("resolve", hf_token))
        return entry

    monkeypatch.setattr(vad, "resolve_vad_model", _resolve)
    monkeypatch.setattr(
        audio_cpp_backend.AudioCppBackend,
        "_download_missing",
        staticmethod(lambda e, token: seen.append(("download", token))),
    )
    monkeypatch.setattr(acm, "forget", lambda identifier = None: seen.append(("forget", identifier)))
    monkeypatch.setattr(vad.AudioCppVadSidecar, "__init__", lambda self: None)
    sidecar = vad.AudioCppVadSidecar()
    assert sidecar.download("org/private-vad-GGUF", "hf_secret") == "org/private-vad-GGUF"
    assert seen == [
        ("resolve", "hf_secret"),
        ("download", "hf_secret"),
        ("forget", "org/private-vad-GGUF"),
    ]


def test_a_detector_refused_during_an_audio_runtime_update_is_retryable():
    from core.inference import audio_cpp_server
    from core.inference.vad_audiocpp_sidecar import VadUnavailableError
    from routes import inference

    audio_cpp_server.UPDATE_IN_PROGRESS.set()
    try:
        assert inference._vad_http_error(VadUnavailableError("updating")).status_code == 503
    finally:
        audio_cpp_server.UPDATE_IN_PROGRESS.clear()
    assert inference._vad_http_error(VadUnavailableError("no runtime")).status_code == 501


def test_the_detector_starts_on_the_cpu(monkeypatch):
    # GPU memory it held would outlive Voice for training and chat: nothing frees the detector there.
    entry = SimpleNamespace(id = "audio-cpp/audio.cpp-gguf/PulseVAD-GGUF", display_name = "PulseVAD")
    calls = []
    monkeypatch.setattr(vad, "resolve_vad_model", lambda *a, **k: entry)
    monkeypatch.setattr(vad.audio_cpp_files, "materialize", lambda e: "/models/pulsevad")
    monkeypatch.setattr(
        vad.AudioCppServer, "start", classmethod(lambda cls, *a, **k: calls.append(k) or object())
    )
    assert vad.AudioCppVadSidecar().load(None) == entry.id
    assert calls == [{"force_cpu": True}]
