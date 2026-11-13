# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The MLX audio mirror must not change TTS or whisper classification, only audio_vlm."""

import sys
from types import SimpleNamespace

import pytest

from core.inference import worker
from core.inference.mlx_inference import _classify_mlx_audio_type
from utils.models.model_config import is_audio_input_type


class _Q:
    def __init__(self):
        self.sent = []

    def put(self, item, *a, **k):
        self.sent.append(item)


def _mlx_entry_for(mc):
    """Expected from the real classifier, not hand-written: what MLXInferenceBackend.load_model records."""
    resolved = _classify_mlx_audio_type(
        SimpleNamespace(),
        None,
        mc.is_vision,
        config_audio_type = mc.audio_type,
    )
    return {
        "is_audio": resolved is not None and resolved != "audio_vlm",
        "audio_type": resolved,
        "has_audio_input": is_audio_input_type(resolved),
    }


def _drive_handle_load(
    monkeypatch,
    mc,
    entry = None,
    device = "mlx",
    load_model = lambda **kw: True,
):
    """Run _handle_load against a stub MLX backend, return the emitted model_info."""
    backend = SimpleNamespace(
        device = device,
        active_model_name = mc.identifier,
        models = {mc.identifier: entry or _mlx_entry_for(mc)},
        load_model = load_model,
    )

    monkeypatch.setattr(worker, "_build_model_config", lambda cfg: mc)
    monkeypatch.setattr(worker, "_run_security_gates", lambda *a, **k: True)
    monkeypatch.setattr(worker, "_resolve_lora_4bit", lambda mc_, v: False)
    monkeypatch.setattr(worker, "_needs_nemotron_trust", lambda *a, **k: False)

    fake_xet = type(sys)("utils.hf_xet_fallback")
    fake_xet.start_watchdog = lambda **k: SimpleNamespace(set = lambda: None)
    monkeypatch.setitem(sys.modules, "utils.hf_xet_fallback", fake_xet)

    q = _Q()
    worker._handle_load(backend, {"model_name": mc.identifier}, q)

    loaded = [m for m in q.sent if m.get("type") == "loaded"]
    assert loaded, f"no loaded response: {q.sent}"
    if loaded[0].get("failure"):
        return loaded[0]
    assert loaded[0].get("success"), loaded[0]
    return loaded[0]["model_info"]


def _mc(audio_type, is_audio, has_audio_input):
    return SimpleNamespace(
        identifier = "unsloth/orpheus-3b-0.1-ft",
        display_name = "orpheus",
        is_vision = False,
        is_lora = False,
        base_model = None,
        is_audio = is_audio,
        audio_type = audio_type,
        has_audio_input = has_audio_input,
    )


@pytest.mark.parametrize("codec", ["snac", "csm", "bicodec", "dac"])
def test_tts_classification_survives_the_mlx_post_load_mirror(monkeypatch, codec):
    """A TTS checkpoint is never a vision model, so mirroring the probe's silence drops the TTS redirect."""
    info = _drive_handle_load(monkeypatch, _mc(codec, is_audio = True, has_audio_input = False))

    assert info["audio_type"] == codec, (
        f"MLX load reclassified a {codec} TTS checkpoint as {info['audio_type']!r}; "
        "the chat route's TTS redirect will no longer fire."
    )
    assert info["is_audio"] is True


def test_whisper_classification_survives_the_mlx_post_load_mirror(monkeypatch):
    """Same for ASR: losing audio_type='whisper' drops both the audio-input
    route and the "Whisper models require audio input" guard."""
    info = _drive_handle_load(monkeypatch, _mc("whisper", is_audio = False, has_audio_input = True))

    assert info["audio_type"] == "whisper"
    assert info["has_audio_input"] is True


def test_audio_cpp_workflow_fields_cross_the_worker_and_the_parent(monkeypatch):
    """Dropped at either hop, status falls back to "speak" from audio_type and offers
    Speak for a clone-only model."""
    from core.inference.orchestrator import _mirrored_model_entry

    mc = _mc("audiocpp_tts", is_audio = True, has_audio_input = False)
    fields = {
        "audio_workflows": ["clone"],
        "audio_reference_text": "required",
        "audio_required_inputs": ["instruct"],
        "audio_clone": {"reference_text": "required", "emotion_audio": False},
        "audio_options_by_workflow": {"convert": [{"name": "length_adjust", "type": "float"}]},
        "audio_workflow_tasks": {"clone": "clon", "convert": "vc", "convert:singing": "svc"},
        "audio_server_task": "clon",
        "audio_convert": {"modes": ["speech"], "target": "audio", "source_max_seconds": 300},
        "audio_convert_route": "v2_vc",
        "audio_convert_rules": {"source_rate": 16000, "target_rate": 24000},
        "audio_edit": {"style": "instructions", "delivery": True, "max_changes": 5},
        "audio_music": {"modes": [{"id": "song", "variations": None}]},
        "audio_cpp_backend": "cpu",
        "audio_device_fallback": "cpu",
        "audio_request_inputs": ["language"],
    }
    entry = {"is_audio": True, "audio_type": "audiocpp_tts", "has_audio_input": False, **fields}
    info = _drive_handle_load(monkeypatch, mc, entry, device = "cuda")
    entry = _mirrored_model_entry(info, mc.identifier)
    assert {k: info[k] for k in fields} == {k: entry[k] for k in fields} == fields


def test_a_classified_load_failure_crosses_the_worker_and_the_parent(monkeypatch):
    from core.inference.audio_cpp_failures import AudioCppFailure, AudioCppFailureError, failure_of

    failure = AudioCppFailure("runtime_missing", "Install the runtime.", 503)

    def refuse(**_kw):
        raise AudioCppFailureError("The audio runtime is not installed.", failure)

    mc = _mc("audiocpp_tts", is_audio = True, has_audio_input = False)
    loaded = _drive_handle_load(monkeypatch, mc, device = "cuda", load_model = refuse)
    assert loaded["success"] is False and AudioCppFailure.from_payload(loaded["failure"]) == failure
    assert failure_of(AudioCppFailureError(loaded["error"], failure)) is failure


@pytest.mark.parametrize(
    "device, config_vision, served_as_vision, published",
    [
        ("mlx", True, False, False),
        ("mlx", True, True, True),
        ("cuda", True, False, True),
        ("mlx", False, True, True),
        ("cuda", False, True, False),
    ],
)
def test_an_mlx_vision_config_served_as_text_stops_accepting_images(
    monkeypatch, device, config_vision, served_as_vision, published
):
    """MLX serves a vision config mlx-lm built (Muse Spark) as text: the parent must not offer images.
    And a tower only mlx-vlm knows (OXR's vit_config) reads them although the config detector said no."""
    from core.inference.orchestrator import _mirrored_model_entry

    mc = _mc(None, is_audio = False, has_audio_input = False)
    mc.is_vision = config_vision
    entry = {"is_audio": False, "audio_type": None, "has_audio_input": False}
    entry["is_vision"] = served_as_vision
    info = _drive_handle_load(monkeypatch, mc, entry, device = device)
    assert info["is_vision"] is published
    assert _mirrored_model_entry(info, mc.identifier)["is_vision"] is published
