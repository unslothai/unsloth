# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The MLX post-load audio mirror must not strip a classification the pre-load
config already earned.

`_handle_load` mirrors the backend entry's is_audio/audio_type/has_audio_input
over the pre-load ModelConfig values. The MLX probe only ever speaks for
"audio_vlm", so for every other audio family (snac/csm/bicodec/dac TTS, whisper
ASR) the mirror has to be a no-op. Otherwise the chat route's TTS redirect
(`model_info.get("is_audio") and audio_type != "whisper"`) and the whisper guard
stop firing on Apple Silicon, and the checkpoint is served as a plain text model
that streams raw codec tokens into chat.
"""

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
    """The entry MLXInferenceBackend.load_model records for a non-vision
    checkpoint, built from the real classifier rather than a hand-written
    expectation."""
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


def _drive_handle_load(monkeypatch, mc):
    """Run _handle_load against a stub MLX backend, return the emitted model_info."""
    backend = SimpleNamespace(
        device = "mlx",
        active_model_name = mc.identifier,
        models = {mc.identifier: _mlx_entry_for(mc)},
        load_model = lambda **kw: True,
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
    """A TTS checkpoint is never a vision model, so the audio_vlm probe never
    ran. Mirroring its silence over the config would drop the TTS redirect and
    serve raw codec tokens as chat text."""
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
