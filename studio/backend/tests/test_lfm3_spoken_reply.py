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


_stub_if_missing(
    "unsloth", ("FastLanguageModel", "FastVisionModel", "FastModel", "is_bfloat16_supported")
)
_stub_if_missing("unsloth.chat_templates", ("get_chat_template",))

import core.inference.inference as inf  # noqa: E402

# Drop the stubs once inf is bound: leftover stubs make _shared_policy return None.
for _name in reversed(_STUBBED):
    sys.modules.pop(_name, None)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_audio_chat_load import _config, backend, multimodal_auto_class  # noqa: E402, F401 - fixtures


class _Codec(torch.nn.Module):
    config = types.SimpleNamespace(sampling_rate = 24000)

    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(1))
        self.decoded = None

    def decode(self, codes):
        self.decoded = codes
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
    assert not [k for k in model.kwargs if k.startswith("audio_") and k != "audio_streamer"]
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

    seen = []

    def backend_for(speaks, is_mlx = False):
        class _Orchestrator:
            active_model_name = "LiquidAI/LFM3-Audio-2B"
            models = {
                active_model_name: {
                    "is_audio": False,
                    "audio_type": "audio_vlm",
                    "speaks": speaks,
                    "is_mlx": is_mlx,
                }
            }

            def generate_audio_response(self, **kwargs):
                seen.append(kwargs)
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
    asyncio.run(route._generate_tts_wav("hello", payload, None, "t", voice = "male"))
    assert seen[-1]["audio_options"] == {"voice": "male"}

    for silent in (backend_for(False), backend_for(True, is_mlx = True)):
        monkeypatch.setattr(route, "get_inference_backend", lambda silent = silent: silent)
        with pytest.raises(HTTPException) as exc:
            asyncio.run(route._generate_tts_wav("hello", payload, None, "t"))
        assert exc.value.detail == "Active model is not an audio model."


@pytest.mark.parametrize("window, expected", [(20, 13), (100_000, 2048), (None, 2048)])
def test_unset_spoken_budget_fits_the_served_window(monkeypatch, window, expected):
    import core.inference.runtime_context as rc

    monkeypatch.setattr(rc, "runtime_context_length", lambda _m, *_a: window)
    model = _Model()
    b = _backend(model, _Processor(_Codec()))
    list(
        inf.InferenceBackend._speak_audio_reply(
            b, [{"role": "user", "content": "Hi"}], "", np.zeros(1600, np.float32), None, None, None
        )
    )
    assert model.kwargs["max_new_tokens"] == expected


class LiquidCodecStreamingCache:
    """Stands in for the codec's own cache: holds nothing back, counts chunks."""

    def __init__(self, config):
        self.chunks = 0
        self.finished = False


class _StreamingCodec(_Codec):
    config = types.SimpleNamespace(sampling_rate = 24000, n_q = 12)

    def decode(
        self,
        codes,
        streaming_cache = None,
        finalize = True,
    ):
        if streaming_cache is not None:
            streaming_cache.chunks += 1
            streaming_cache.finished = finalize
        return types.SimpleNamespace(audio_values = torch.full((1, codes.shape[-1] * 1920), 0.25))


# _SpeechChunker finds the cache class in the codec's module, as with the remote LiquidCodec.
_StreamingCodec.__module__ = __name__


class _TokenTokenizer:
    words = {1: "Paris", 2: " is", 3: " the", 4: " capital."}

    def decode(
        self,
        ids,
        skip_special_tokens = False,
    ):
        ids = ids.tolist() if hasattr(ids, "tolist") else ids
        return "".join(self.words.get(int(i), "") for i in ids)


class _StreamingModel(torch.nn.Module):
    """Interleaves text tokens and speech frames through the streamers, as LFM3's generate does,
    and waits after the first chunk's frames until the caller lets it finish."""

    config = types.SimpleNamespace(interleaved_n_audio = 12)
    device = "cpu"

    def __init__(self, frames = 10):
        super().__init__()
        self.p = torch.nn.Parameter(torch.zeros(1))
        self.frames = frames
        self.release = threading.Event()
        self.finished = threading.Event()

    def generate(self, streamer, audio_streamer, **kwargs):
        streamer.put(torch.zeros((1, 7), dtype = torch.long))  # the prompt
        for token in (1, 2):
            streamer.put(torch.tensor([[token]]))
        for i in range(self.frames):
            if i == inf._SPEECH_FIRST_CHUNK_FRAMES:
                assert self.release.wait(5)
            audio_streamer.put(torch.zeros(12, dtype = torch.long))
            if i == 5:
                for token in (3, 4):
                    streamer.put(torch.tensor([[token]]))
        self.finished.set()
        return types.SimpleNamespace(
            sequences = torch.tensor([[0] * 7 + [1, 2, 3, 4]]),
            audio_codes = torch.zeros((1, 12, self.frames), dtype = torch.long),
            audio_lengths = torch.tensor([self.frames]),
        )


def _streaming_backend(model):
    processor = _Processor(_StreamingCodec())
    processor.tokenizer = _TokenTokenizer()
    return _backend(model, processor)


def _spoken_iter(b, cancel_event = None):
    return inf.InferenceBackend.generate_audio_input_response(
        b,
        messages = [{"role": "user", "content": "Answer this."}],
        system_prompt = None,
        audio_array = np.zeros(1600, dtype = np.float32),
        temperature = 0.7,
        top_p = 0.9,
        top_k = 40,
        min_p = 0.0,
        max_new_tokens = 256,
        repetition_penalty = 1.0,
        cancel_event = cancel_event,
        speak = True,
    )


def test_spoken_reply_streams_speech_before_generation_ends():
    model = _StreamingModel(frames = 10)
    b = _streaming_backend(model)
    gen = _spoken_iter(b)

    early = []
    for item in gen:
        early.append(item)
        if isinstance(item, dict):
            break
    # The first chunk is out while generate still waits to make the rest.
    assert not model.finished.is_set()
    assert early == ["Paris", " is", early[-1]]
    assert _wav(early[-1]["speech"]) == (24000, inf._SPEECH_FIRST_CHUNK_FRAMES * 1920)
    model.release.set()
    rest = list(gen)

    text = "".join(x for x in early + rest if isinstance(x, str))
    speech = [x for x in early + rest if isinstance(x, dict)]
    assert text == "Paris is the capital."
    assert len(speech) == 2
    assert sum(_wav(x["speech"])[1] for x in speech) == 10 * 1920
    assert b.last_generation_stats["usage"]["completion_tokens"] == 4


def test_stopping_a_streamed_spoken_reply_ends_generate():
    model = _StreamingModel(frames = 10)
    cancel = threading.Event()
    gen = _spoken_iter(_streaming_backend(model), cancel)
    for item in gen:
        if isinstance(item, dict):
            break
    cancel.set()
    model.release.set()
    assert list(gen) == []
    assert not model.finished.is_set()


def test_closing_a_streamed_spoken_reply_frees_the_generation_lock():
    model = _StreamingModel(frames = 10)
    b = _streaming_backend(model)
    gen = _spoken_iter(b)
    for item in gen:
        if isinstance(item, dict):
            break
    model.release.set()
    gen.close()
    assert b._generation_lock.acquire(timeout = 5)
    b._generation_lock.release()


def test_non_streamed_route_joins_speech_chunks_into_one_wav():
    import routes.inference as route
    from core.inference.inference import _speech_item

    chunks = [_speech_item(np.full(n, 0.25, dtype = np.float32), 24000) for n in (1920, 3840)]
    joined = route._joined_speech_reply(chunks)
    assert joined["format"] == "wav" and joined["sample_rate"] == 24000
    assert _wav(joined["data"]) == (24000, 5760)
    assert route._joined_speech_reply(chunks[:1])["data"] == chunks[0]["speech"]
    assert route._joined_speech_reply([]) == {}


def test_a_worker_still_alive_after_finishing_keeps_the_reply(monkeypatch):
    # The worker queues `done` from its finally block and can still be alive when it is read.
    real_put = inf.queue.Queue.put if hasattr(inf, "queue") else None
    import queue

    class _SlowExit(queue.Queue):
        def put(self, item, *a, **k):
            super().put(item, *a, **k)
            if not isinstance(item, tuple):
                threading.Event().wait(0.2)

    monkeypatch.setattr(queue, "Queue", _SlowExit)
    b = _backend(_Model(frames = 0), _Processor(_Codec()))
    assert _spoken(b, speak = True) == ["Paris is the capital."]


class _ByteTokenizer:
    """Byte-level ids, as LFM3's BPE splits an emoji: a prefix ending mid-character decodes to U+FFFD."""

    def decode(
        self,
        ids,
        skip_special_tokens = False,
    ):
        ids = ids.tolist() if hasattr(ids, "tolist") else ids
        return bytes(int(i) for i in ids if int(i) < 256).decode("utf-8", errors = "replace")


class _ByteModel(_StreamingModel):
    reply = "Hi \N{SMILING FACE WITH SMILING EYES} there.".encode()

    def generate(self, streamer, audio_streamer, **kwargs):
        streamer.put(torch.full((1, 7), 300, dtype = torch.long))
        for byte in self.reply:
            streamer.put(torch.tensor([[byte]]))
        return types.SimpleNamespace(
            sequences = torch.tensor([[300] * 7 + list(self.reply)]),
            audio_codes = torch.zeros((1, 12, 0), dtype = torch.long),
            audio_lengths = torch.tensor([0]),
        )


def test_a_character_split_over_tokens_streams_whole_and_keeps_the_rest_of_the_reply():
    processor = _Processor(_StreamingCodec())
    processor.tokenizer = _ByteTokenizer()
    b = _backend(_ByteModel(), processor)
    text = [x for x in _spoken_iter(b) if isinstance(x, str)]
    assert "".join(text) == _ByteModel.reply.decode()
    assert not any("\ufffd" in t for t in text)
