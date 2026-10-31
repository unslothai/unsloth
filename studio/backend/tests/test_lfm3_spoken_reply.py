# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""LFM3-Audio / -Omni on the transformers backend: spoken replies (the MLX contract from #15889)
and reading text aloud through the chat model's TTS mode."""

import asyncio
import base64
import importlib
import io
import sys
import threading
import types
import wave
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from fastapi import HTTPException

_STUBBED: list[str] = []


def _stub_if_missing(name, attrs):
    """Stubs a dep the backend pytest job lacks, so collection succeeds; real installs are left alone."""
    if name in sys.modules:
        return
    try:
        importlib.import_module(name)
        return
    except Exception:  # noqa: BLE001 - any import failure means "not usable here", so stub it
        pass
    _STUBBED.append(name)
    mod = types.ModuleType(name)
    mod.__spec__ = None
    for attr in attrs:
        setattr(mod, attr, MagicMock())
    sys.modules[name] = mod
    parent, _, child = name.rpartition(".")
    if parent and parent in sys.modules:
        setattr(sys.modules[parent], child, mod)


_stub_if_missing("unsloth", ("FastLanguageModel", "FastVisionModel", "is_bfloat16_supported"))
_stub_if_missing("unsloth.chat_templates", ("get_chat_template",))

import core.inference.inference as inf  # noqa: E402

# Drop the stubs once inf is bound: leftover stubs make _shared_policy return None.
for _name in reversed(_STUBBED):
    sys.modules.pop(_name, None)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_audio_chat_load import _config, backend  # noqa: E402, F401 - fixture


class _Codec(torch.nn.Module):
    config = types.SimpleNamespace(sampling_rate = 24000)

    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(1))
        self.decoded = None

    def decode(self, codes):
        self.decoded = codes
        # 1920 samples per frame, as LiquidCodec
        samples = codes.shape[-1] * 1920
        return types.SimpleNamespace(audio_values = torch.full((1, 1, samples), 0.25))


class _Tokenizer:
    def decode(
        self,
        ids,
        skip_special_tokens = False,
    ):
        assert skip_special_tokens
        return " Paris is the capital. "


class _Inputs(dict):
    def to(self, device):
        return self


class _Processor:
    feature_extractor = object()

    def __init__(self, codec = None):
        self.tokenizer = _Tokenizer()
        self.audio_tokenizer = codec
        self.rendered = None

    def apply_chat_template(self, conversation, **kwargs):
        self.rendered = (conversation, kwargs)
        return _Inputs(input_ids = torch.zeros((1, 7), dtype = torch.long))


class _Model(torch.nn.Module):
    config = types.SimpleNamespace(interleaved_n_audio = 12)
    device = "cpu"

    def __init__(
        self,
        frames = 3,
        cancel_during = None,
    ):
        super().__init__()
        self.p = torch.nn.Parameter(torch.zeros(1))
        self.frames, self.cancel_during, self.kwargs = frames, cancel_during, None

    def generate(self, **kwargs):
        self.kwargs = kwargs
        streamers = (kwargs.get("streamer"), kwargs.get("audio_streamer"))
        for streamer in {id(s): s for s in streamers if s}.values():
            if self.cancel_during is not None:
                self.cancel_during.set()
            streamer.put(torch.zeros(1))
        codes = torch.full((1, 12, max(self.frames, 1)), -1)
        return types.SimpleNamespace(
            sequences = torch.zeros((1, 7 + 5), dtype = torch.long),
            audio_codes = codes,
            audio_lengths = torch.tensor([self.frames]),
        )


def _backend(
    model,
    processor,
    audio_type = "audio_vlm",
):
    b = inf.InferenceBackend.__new__(inf.InferenceBackend)
    b.active_model_name = "LiquidAI/LFM3-Audio-2B"
    b.models = {
        b.active_model_name: {
            "model": model,
            "processor": processor,
            "tokenizer": processor,
            "audio_type": audio_type,
            "speaks": inf._speech_codec(model, processor) is not None,
        }
    }
    b._generation_lock = threading.Lock()
    b._apply_adapter_state = lambda *_a, **_k: None
    b.last_generation_stats = None
    return b


def _spoken(
    b,
    cancel_event = None,
    **kwargs,
):
    return list(
        inf.InferenceBackend.generate_audio_input_response(
            b,
            messages = [{"role": "user", "content": "Answer this."}],
            system_prompt = "Be brief.",
            audio_array = np.zeros(1600, dtype = np.float32),
            temperature = 0.7,
            top_p = 0.9,
            top_k = 40,
            min_p = 0.0,
            max_new_tokens = 256,
            repetition_penalty = 1.0,
            cancel_event = cancel_event,
            **kwargs,
        )
    )


def _wav(b64):
    with wave.open(io.BytesIO(base64.b64decode(b64))) as w:
        return w.getframerate(), w.getnframes()


def test_speech_codec_needs_the_bundled_codec_and_speech_frames():
    assert inf._speech_codec(_Model(), _Processor(_Codec())) is not None
    assert inf._speech_codec(_Model(), _Processor(None)) is None  # Gemma 3n, Qwen-Omni text-only
    plain = _Model()
    plain.config = types.SimpleNamespace()
    assert inf._speech_codec(plain, _Processor(_Codec())) is None


@pytest.mark.parametrize("codec, speaks", [(True, True), (False, False)])
def test_load_reports_speaks(backend, codec, speaks):  # noqa: F811
    b, _ = backend
    model, processor = _Model(), _Processor(_Codec() if codec else None)
    # The backend fixture put a per-test unsloth module in sys.modules; load_model imports from it.
    sys.modules["unsloth"].FastModel = types.SimpleNamespace(
        from_pretrained = lambda **_k: (model, processor), for_inference = lambda *_a: None
    )
    assert inf.InferenceBackend.load_model(b, _config(), trust_remote_code = True) is True
    assert b.models["LiquidAI/LFM3-Audio-2B"]["speaks"] is speaks


def test_spoken_reply_yields_text_then_one_wav():
    codec = _Codec()
    model, processor = _Model(frames = 3), _Processor(codec)
    b = _backend(model, processor)

    text, speech = _spoken(b, speak = True)

    assert text == "Paris is the capital."
    rate, frames = _wav(speech["speech"])
    assert (rate, frames, speech["sample_rate"]) == (24000, 3 * 1920, 24000)
    assert codec.decoded.shape == (1, 12, 3)
    conversation, kwargs = processor.rendered
    assert conversation[0]["role"] == "system"
    assert "Respond with interleaved text and audio." in str(conversation[0]["content"])
    assert "Be brief." in str(conversation[0]["content"])
    assert conversation[-1] == {"role": "assistant", "content": "<|mixed_start|>"}
    assert kwargs["continue_final_message"] is True
    assert model.kwargs["return_dict_in_generate"] is True
    # Each checkpoint's generation_config carries its own audio sampling (Audio and Omni differ).
    assert not [k for k in model.kwargs if k.startswith("audio_")]
    assert b.last_generation_stats["usage"]["completion_tokens"] == 5


def test_spoken_reply_without_speech_frames_returns_text_alone():
    b = _backend(_Model(frames = 0), _Processor(_Codec()))
    assert _spoken(b, speak = True) == ["Paris is the capital."]


def test_stopping_a_spoken_reply_yields_nothing():
    cancel = threading.Event()
    b = _backend(_Model(cancel_during = cancel), _Processor(_Codec()))
    assert _spoken(b, cancel_event = cancel, speak = True) == []


def test_spoken_reply_refuses_a_model_that_cannot_speak():
    b = _backend(_Model(), _Processor(None))
    with pytest.raises(RuntimeError, match = "cannot answer in speech"):
        _spoken(b, speak = True)


def test_read_aloud_uses_the_chat_models_tts_mode():
    model, processor = _Model(frames = 2), _Processor(_Codec())
    b = _backend(model, processor)

    wav, rate = inf.InferenceBackend.generate_audio_response(
        b,
        text = "Unsloth makes fine tuning fast.",
        language = "en-GB",
        audio_options = {"voice": "male"},
    )

    assert rate == 24000 and wav[:4] == b"RIFF"
    conversation, _ = processor.rendered
    assert conversation[0]["content"] == "Perform TTS in British English using a male voice."
    assert conversation[1]["content"] == "Unsloth makes fine tuning fast."
    assert conversation[-1] == {"role": "assistant", "content": "<|audio_start|>"}


def test_read_aloud_with_no_speech_is_an_error_and_a_stop_is_a_cancel():
    from core.inference.audio_errors import AudioGenerationCancelledError

    with pytest.raises(RuntimeError, match = "no speech"):
        inf.InferenceBackend.generate_audio_response(
            _backend(_Model(frames = 0), _Processor(_Codec())), text = "hi"
        )
    cancel = threading.Event()
    with pytest.raises(AudioGenerationCancelledError):
        inf.InferenceBackend.generate_audio_response(
            _backend(_Model(cancel_during = cancel), _Processor(_Codec())),
            text = "hi",
            cancel_event = cancel,
        )


@pytest.mark.parametrize(
    "language, voice, expected",
    [
        (None, None, "Perform TTS in American English using a female voice."),
        ("fr", "female", "Perform TTS in French using a female voice."),
        ("Japanese", "male", "Perform TTS in Japanese using a male voice."),
        ("xx", None, "Perform TTS in American English using a female voice."),
    ],
)
def test_tts_system_prompt(language, voice, expected):
    assert inf._tts_system_prompt(language, {"voice": voice} if voice else None) == expected


def test_tts_route_serves_a_speaking_chat_model_and_refuses_a_silent_one(monkeypatch):
    import routes.inference as route
    from models.inference import ChatCompletionRequest

    class _Llama:
        is_loaded = False
        _is_audio = False

    def backend_for(speaks):
        class _Orchestrator:
            active_model_name = "LiquidAI/LFM3-Audio-2B"
            models = {
                active_model_name: {"is_audio": False, "audio_type": "audio_vlm", "speaks": speaks}
            }

            def generate_audio_response(self, **_kwargs):
                return b"RIFFwav", 24000

        return _Orchestrator()

    async def _no_switch(*_a, **_k):
        return None

    monkeypatch.setattr(route, "get_llama_cpp_backend", lambda: _Llama())
    monkeypatch.setattr(route, "_maybe_auto_switch_model", _no_switch)
    payload = ChatCompletionRequest(messages = [{"role": "user", "content": "hello"}])

    monkeypatch.setattr(route, "get_inference_backend", lambda: backend_for(True))
    wav, rate, _, audio_type = asyncio.run(route._generate_tts_wav("hello", payload, None, "t"))
    assert (wav, rate, audio_type) == (b"RIFFwav", 24000, "audio_vlm")

    monkeypatch.setattr(route, "get_inference_backend", lambda: backend_for(False))
    with pytest.raises(HTTPException) as exc:
        asyncio.run(route._generate_tts_wav("hello", payload, None, "t"))
    assert exc.value.detail == "Active model is not an audio model."
