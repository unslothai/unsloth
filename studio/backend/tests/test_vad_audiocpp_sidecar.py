# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import time
from types import SimpleNamespace

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


def test_a_download_drops_the_cached_miss_so_the_next_load_sees_it(monkeypatch):
    """The load route downloads on a miss and loads again at once; that load must not reuse the miss."""
    entry = SimpleNamespace(id = vad.DEFAULT_VAD_MODEL)
    monkeypatch.setattr(vad, "resolve_vad_model", lambda *a, **k: entry)
    from core.inference import audio_cpp_backend

    monkeypatch.setattr(
        audio_cpp_backend.AudioCppBackend, "_download_missing", staticmethod(lambda *a: True)
    )
    miss = (vad.DEFAULT_VAD_MODEL.lower(), "", False, "", (), "")
    monkeypatch.setitem(acm._resolve_cache, miss, (time.monotonic(), None))

    assert vad.AudioCppVadSidecar().download(None) == vad.DEFAULT_VAD_MODEL
    assert miss not in acm._resolve_cache


def test_the_detector_runs_on_the_cpu_while_training_holds_the_gpu(monkeypatch, tmp_path):
    """As the STT sidecar does: no second GPU runtime beside training, and back to the GPU after."""
    from core.inference import stt_sidecar

    entry = SimpleNamespace(id = vad.DEFAULT_VAD_MODEL, display_name = "PulseVAD")
    monkeypatch.setattr(vad, "resolve_vad_model", lambda *a, **k: entry)
    monkeypatch.setattr(vad.audio_cpp_files, "materialize", lambda e: tmp_path / "vad.gguf")
    launched = []

    def start(
        model,
        path,
        force_cpu = False,
    ):
        launched.append(force_cpu)
        return SimpleNamespace(process = SimpleNamespace(poll = lambda: None), stop = lambda: None)

    monkeypatch.setattr(vad.AudioCppServer, "start", staticmethod(start))
    training = {"on": True}
    monkeypatch.setattr(stt_sidecar, "_training_active", lambda: training["on"])
    sidecar = vad.AudioCppVadSidecar()

    sidecar.load(None)
    sidecar.load(None)
    training["on"] = False
    sidecar.load(None)
    assert launched == [True, False]
