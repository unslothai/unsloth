# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import os
import struct
import sys
from types import SimpleNamespace

import pytest

from utils.audio_tokens import SNAC_PROBE_TOKEN_IDS
from utils.models import gguf_metadata
from utils.models.gguf_metadata import classify_gguf_tts_audio_prefix, read_gguf_tts_audio_type


def test_outetts_v3_speakerless_prompt_stops_before_generated_features():
    from core.inference.chat_template_helpers import build_dac_tts_prompt
    from core.inference.llama_cpp import LlamaCppBackend

    expected = "<|im_start|>\n<|text_start|>Read this aloud.<|text_end|>\n<|audio_start|>\n"

    assert build_dac_tts_prompt("Read this aloud.") == expected
    assert LlamaCppBackend._TTS_PROMPTS["dac"][0].format(text = "Read this aloud.") == expected
    assert "<|global_features_start|>" not in expected


def test_native_outetts_uses_the_same_speakerless_prompt(monkeypatch):
    pytest.importorskip("peft")
    import torch

    from core.inference.inference import InferenceBackend

    captured = {}

    class _Inputs(dict):
        def to(self, _device):
            return self

    class _Tokenizer:
        def __call__(self, prompts, **_kwargs):
            captured["prompt"] = prompts[0]
            return _Inputs(input_ids = torch.tensor([[1]]))

        def batch_decode(self, _generated, **_kwargs):
            return ["<|c1_1|><|c2_2|>"]

    class _Model:
        device = torch.device("cpu")
        dtype = torch.float32

        def generate(self, **_kwargs):
            return torch.tensor([[1, 2]])

    class _CodecManager:
        def decode_dac(self, generated, device):
            captured["generated"] = generated
            captured["device"] = device
            return b"RIFFfake", 44100

    backend = InferenceBackend.__new__(InferenceBackend)
    backend._audio_codec_manager = _CodecManager()
    monkeypatch.setattr(backend, "_patch_repetition_penalty_processor", lambda: None)

    assert backend._generate_dac(
        _Model(),
        _Tokenizer(),
        "Read this aloud.",
        temperature = 0.6,
        top_k = 50,
        top_p = 0.95,
        min_p = 0.0,
        max_new_tokens = 64,
        repetition_penalty = 1.1,
    ) == (b"RIFFfake", 44100)
    assert captured == {
        "prompt": "<|im_start|>\n<|text_start|>Read this aloud.<|text_end|>\n<|audio_start|>\n",
        "generated": "<|c1_1|><|c2_2|>",
        "device": "cpu",
    }


def _string(value: str) -> bytes:
    data = value.encode()
    return struct.pack("<Q", len(data)) + data


def test_snac_codec_load_uses_the_preflighted_snapshot(monkeypatch):
    from core.inference.audio_codecs import AudioCodecManager

    calls = []

    class _Model:
        def to(self, device):
            calls.append(("device", device))
            return self

        def eval(self):
            return self

    class _Snac:
        @staticmethod
        def from_pretrained(source, **kwargs):
            calls.append((source, kwargs))
            return _Model()

    monkeypatch.setitem(sys.modules, "snac", SimpleNamespace(SNAC = _Snac))
    manager = AudioCodecManager()
    manager.load_codec("snac", "cpu", model_repo_path = "/staged/snac-snapshot")

    assert calls[0][0] == "/staged/snac-snapshot"
    assert calls[1] == ("device", "cpu")


def test_dac_codec_load_uses_the_preflighted_weights(monkeypatch):
    from core.inference import audio_codecs

    calls = []

    class _Config:
        def __init__(self, **kwargs):
            calls.append(("config", kwargs))

    class _Processor:
        def __init__(self, config):
            self.audio_codec = object()

    monkeypatch.setattr(audio_codecs, "ensure_outetts_source", lambda: "/pinned/outetts")
    monkeypatch.setattr(
        audio_codecs,
        "ensure_dac_speech_weights",
        lambda: pytest.fail("staged DAC weights were reacquired"),
    )
    monkeypatch.setattr(
        audio_codecs,
        "import_outetts_module",
        lambda name, _source: SimpleNamespace(
            AudioProcessor = _Processor,
            ModelConfig = _Config,
        ),
    )

    manager = audio_codecs.AudioCodecManager()
    manager.load_codec("dac", "cpu", model_repo_path = "/captured/dac.pth")

    assert calls == [
        (
            "config",
            {
                "tokenizer_path": None,
                "device": "cpu",
                "audio_codec_path": "/captured/dac.pth",
            },
        )
    ]


def test_gguf_bicodec_load_uses_the_preflighted_repository(monkeypatch):
    from core.inference import audio_codecs, llama_cpp

    calls = []

    class _Manager:
        def load_codec(self, *args, **kwargs):
            calls.append((args, kwargs))

    monkeypatch.setattr(audio_codecs, "AudioCodecManager", _Manager)
    monkeypatch.setattr(
        audio_codecs,
        "resolve_bicodec_repo_path",
        lambda *_a, **_k: pytest.fail("staged BiCodec repository was reacquired"),
    )
    backend = llama_cpp.LlamaCppBackend.__new__(llama_cpp.LlamaCppBackend)
    backend._arch_gate_forced_cpu = True
    llama_cpp.LlamaCppBackend._codec_mgr = None

    backend.init_audio_codec("bicodec", "/captured/spark")

    assert calls == [(("bicodec", "cpu"), {"model_repo_path": "/captured/spark"})]
    llama_cpp.LlamaCppBackend._codec_mgr = None


def _write_gguf(
    path,
    tokens,
    token_types = None,
):
    metadata = _string("general.architecture") + struct.pack("<I", 8) + _string("llama")
    if tokens is not None:
        array = struct.pack("<IQ", 8, len(tokens)) + b"".join(_string(t) for t in tokens)
        metadata += _string("tokenizer.ggml.tokens") + struct.pack("<I", 9) + array
        token_types = token_types or [3] * len(tokens)
        types = struct.pack("<IQ", 5, len(token_types)) + struct.pack(
            f"<{len(token_types)}i", *token_types
        )
        metadata += _string("tokenizer.ggml.token_type") + struct.pack("<I", 9) + types
    kv_count = 1 + (2 if tokens is not None else 0)
    path.write_bytes(struct.pack("<IIQQ", 0x46554747, 3, 0, kv_count) + metadata)
    return str(path)


def _snac_vocab():
    """Orpheus's shape: the codes start where the base vocabulary ends, so the two ids
    the serving detector detokenizes land on codes."""
    base = [f"t{i}" for i in range(SNAC_PROBE_TOKEN_IDS[0])]
    return [*base, *(f"<custom_token_{i}>" for i in range(10_002))]


@pytest.fixture(autouse = True)
def _clear_cache():
    gguf_metadata._TTS_AUDIO_TYPE_CACHE.clear()


def test_orpheus_snac_codes_are_a_speech_model(tmp_path):
    assert read_gguf_tts_audio_type(_write_gguf(tmp_path / "orpheus.gguf", _snac_vocab())) == "snac"


def test_snac_codes_the_serving_probe_would_miss_are_not_a_speech_model(tmp_path):
    # llama.cpp asks what two fixed ids detokenize to, so codes that do not reach them
    # are not served as SNAC. Accepting them here would evict for a target that fails.
    tokens = ["hello", "<|eot_id|>", *(f"<custom_token_{i}>" for i in range(10_002))]
    assert read_gguf_tts_audio_type(_write_gguf(tmp_path / "shifted.gguf", tokens)) is None


def test_spark_bicodec_tokens_are_a_speech_model(tmp_path):
    tokens = ["hi", "<|bicodec_semantic_0|>", "<|bicodec_global_0|>"]
    assert read_gguf_tts_audio_type(_write_gguf(tmp_path / "spark.gguf", tokens)) == "bicodec"


def test_a_complete_remote_prefix_uses_the_same_speech_classifier(tmp_path):
    _write_gguf(
        tmp_path / "spark.gguf",
        ["hi", "<|bicodec_semantic_0|>", "<|bicodec_global_0|>"],
    )
    assert classify_gguf_tts_audio_prefix((tmp_path / "spark.gguf").read_bytes()) == (
        "bicodec",
        True,
    )


def test_a_truncated_remote_prefix_is_inconclusive(tmp_path):
    _write_gguf(
        tmp_path / "spark.gguf",
        ["hi", "<|bicodec_semantic_0|>", "<|bicodec_global_0|>"],
    )
    data = (tmp_path / "spark.gguf").read_bytes()
    assert classify_gguf_tts_audio_prefix(data[:-1]) == (None, False)


def test_normal_vocab_membership_is_not_a_runtime_special_token(tmp_path):
    tokens = ["hi", "<|bicodec_semantic_0|>", "<|bicodec_global_0|>"]
    assert (
        read_gguf_tts_audio_type(
            _write_gguf(tmp_path / "normal.gguf", tokens, token_types = [1, 1, 1])
        )
        is None
    )


def test_a_partial_bicodec_vocabulary_is_not_a_speech_model(tmp_path):
    # The serving detector wants both token-zero markers, not any <|bicodec_ token.
    tokens = ["hi", "<|bicodec_semantic_0|>"]
    assert read_gguf_tts_audio_type(_write_gguf(tmp_path / "half.gguf", tokens)) is None


def test_outetts_dac_tokens_are_a_speech_model(tmp_path):
    tokens = ["<|audio_start|>", "<|audio_end|>", "<|c1_0|>", "<|c2_0|>"]
    assert read_gguf_tts_audio_type(_write_gguf(tmp_path / "oute.gguf", tokens)) == "dac"


def test_outetts_0_2_delimiters_without_a_codebook_are_not_a_speech_model(tmp_path):
    # Real regression: OuteTTS 0.2 ships these four delimiters but none of DAC's codebook
    # tokens, so llama.cpp reports it as not audio. The delimiters alone used to pass this
    # gate, which evicted the resident speech model for a target the loader then refused.
    tokens = ["<|audio_start|>", "<|audio_end|>", "<|text_start|>", "<|text_end|>"]
    assert read_gguf_tts_audio_type(_write_gguf(tmp_path / "oute02.gguf", tokens)) is None


def test_an_audio_input_marker_outranks_codec_tokens(tmp_path):
    # Ordering matches the serving detector: it checks the audio-in markers before the
    # codecs, so a vocabulary carrying both is audio_vlm and must not be switched to.
    tokens = ["<|audio|>", "<|bicodec_semantic_0|>", "<|bicodec_global_0|>"]
    assert read_gguf_tts_audio_type(_write_gguf(tmp_path / "both.gguf", tokens)) is None


def test_a_chat_vocabulary_is_not_a_speech_model(tmp_path):
    tokens = ["hello", "<|im_start|>", "<|im_end|>", "<custom_token_1>"]
    assert read_gguf_tts_audio_type(_write_gguf(tmp_path / "chat.gguf", tokens)) is None


def test_an_audio_input_model_is_not_a_speech_model(tmp_path):
    tokens = ["<|startoftranscript|>", "<|audio|>"]
    assert read_gguf_tts_audio_type(_write_gguf(tmp_path / "whisper.gguf", tokens)) is None


def test_a_header_without_a_vocabulary_is_not_a_speech_model(tmp_path):
    assert read_gguf_tts_audio_type(_write_gguf(tmp_path / "bare.gguf", None)) is None


def test_a_non_gguf_file_is_not_a_speech_model(tmp_path):
    path = tmp_path / "weights.bin"
    path.write_bytes(b"\0" * 64)
    assert read_gguf_tts_audio_type(str(path)) is None
    assert read_gguf_tts_audio_type(str(tmp_path / "missing.gguf")) is None


def test_the_verdict_is_cached_per_file(tmp_path, monkeypatch):
    path = _write_gguf(tmp_path / "spark.gguf", ["<|bicodec_semantic_0|>", "<|bicodec_global_0|>"])
    assert read_gguf_tts_audio_type(path) == "bicodec"
    monkeypatch.setattr(
        gguf_metadata, "_parse_gguf_marker_tokens", lambda _p: pytest.fail("reparsed")
    )
    assert read_gguf_tts_audio_type(path) == "bicodec"


def test_atomic_replacement_invalidates_a_same_size_same_mtime_verdict(tmp_path):
    path = tmp_path / "model.gguf"
    replacement = tmp_path / "replacement.gguf"
    _write_gguf(path, ["<|bicodec_semantic_0|>", "<|bicodec_global_0|>"])
    _write_gguf(replacement, ["x" * 22, "y" * 20])
    assert path.stat().st_size == replacement.stat().st_size
    assert read_gguf_tts_audio_type(str(path)) == "bicodec"
    original = path.stat()
    os.replace(replacement, path)
    os.utime(path, ns = (path.stat().st_atime_ns, original.st_mtime_ns))
    assert path.stat().st_size == original.st_size
    assert path.stat().st_mtime_ns == original.st_mtime_ns
    assert read_gguf_tts_audio_type(str(path)) is None


def test_a_short_final_token_body_is_not_a_speech_model(tmp_path):
    path = tmp_path / "truncated.gguf"
    tokens = ["<|bicodec_semantic_0|>", "<|bicodec_global_0|>"]
    metadata = _string("general.architecture") + struct.pack("<I", 8) + _string("llama")
    token_array = struct.pack("<IQ", 8, len(tokens)) + _string(tokens[0])
    final = tokens[1].encode()
    token_array += struct.pack("<Q", len(final) + 4096) + final
    metadata += _string("tokenizer.ggml.tokens") + struct.pack("<I", 9) + token_array
    path.write_bytes(struct.pack("<IIQQ", 0x46554747, 3, 0, 2) + metadata)
    assert read_gguf_tts_audio_type(str(path)) is None


@pytest.mark.parametrize(
    ("key", "element_type"),
    (("tokenizer.ggml.tokens", 8), ("tokenizer.ggml.token_type", 5)),
)
def test_oversized_vocabulary_arrays_are_rejected_before_allocation(tmp_path, key, element_type):
    path = tmp_path / f"oversized-{element_type}.gguf"
    metadata = (
        _string(key)
        + struct.pack("<I", 9)
        + struct.pack("<IQ", element_type, gguf_metadata._MAX_GGUF_VOCAB_ENTRIES + 1)
    )
    path.write_bytes(struct.pack("<IIQQ", 0x46554747, 3, 0, 1) + metadata)
    assert gguf_metadata._parse_gguf_marker_tokens(str(path)) is None


def test_generic_angle_bracket_tokens_are_not_accumulated_as_audio_markers(tmp_path):
    path = _write_gguf(tmp_path / "generic-markers.gguf", ["<"] * 50_000)
    markers, snac_probe = gguf_metadata._parse_gguf_marker_tokens(path)
    assert markers == []
    assert snac_probe is False


def test_repeated_audio_markers_are_rejected_without_accumulation(tmp_path):
    path = _write_gguf(tmp_path / "repeated-markers.gguf", ["<|AUDIO|>"] * 50_000)
    assert gguf_metadata._parse_gguf_marker_tokens(path) is None


def test_the_switch_probe_reads_the_variant_the_load_will_open(tmp_path, monkeypatch):
    import routes.inference as inference_route
    from utils.models import model_config

    speech = _write_gguf(
        tmp_path / "model-Q8_0.gguf", ["<|bicodec_semantic_0|>", "<|bicodec_global_0|>"]
    )
    text = _write_gguf(tmp_path / "model-Q4_K_M.gguf", ["<|im_start|>"])
    monkeypatch.setattr(
        model_config,
        "_find_local_gguf_by_variant",
        lambda _d, variant, *_a: speech if variant == "Q8_0" else text,
    )
    assert inference_route._target_speech_audio_type(str(tmp_path), True, "Q8_0") == "bicodec"
    assert inference_route._target_speech_audio_type(str(tmp_path), True, "Q4_K_M") is None
    monkeypatch.setattr(model_config, "detect_gguf_model", lambda _p, *_a: speech)
    assert inference_route._target_speech_audio_type(speech, True) == "bicodec"


def test_the_switch_probe_asks_a_checkpoint_for_an_output_codec(monkeypatch):
    import routes.inference as inference_route
    from utils.models import model_config

    seen = {}

    def _detect(path, **kwargs):
        seen.update(kwargs)
        return {"/srv/tts": "csm", "/srv/higgs": "higgs_tts2", "/srv/asr": "whisper"}.get(path)

    monkeypatch.setattr(model_config, "detect_audio_type", _detect)
    assert inference_route._target_speech_audio_type("/srv/tts", False) == "csm"
    assert inference_route._target_speech_audio_type("/srv/higgs", False) == "higgs_tts2"
    assert inference_route._target_speech_audio_type("/srv/asr", False) is None
    assert inference_route._target_speech_audio_type("/srv/text", False) is None
    assert seen["local_files_only"] is True


def test_a_failing_probe_refuses_rather_than_evicts(monkeypatch):
    import routes.inference as inference_route
    from utils.models import model_config

    def _boom(*_a, **_kw):
        raise RuntimeError("unreadable")

    monkeypatch.setattr(model_config, "detect_audio_type", _boom)
    assert inference_route._target_speech_audio_type("/srv/tts", False) is None


def test_an_ordinary_codec_checkpoint_is_refused_on_an_mlx_host(monkeypatch):
    # A checkpoint the worker would hand to MLX cannot speak: MLX answers generate_audio
    # with "not supported". Refused before detect_audio_type, which never runs.
    import routes.inference as inference_route
    from core.inference import local_model_resolver, native_audio
    from utils.models import model_config

    monkeypatch.setattr(local_model_resolver, "_host_serves_mlx", lambda: True)
    monkeypatch.setattr(native_audio, "is_native_audio_model", lambda _p: False)
    monkeypatch.setattr(
        model_config, "detect_audio_type", lambda *_a, **_kw: pytest.fail("probed on MLX")
    )
    assert inference_route._target_speech_audio_type("/srv/orpheus", False) is None


def test_a_native_audio_checkpoint_still_switches_on_an_mlx_host(monkeypatch):
    # The worker picks the native-audio backend before the MLX fast path, and that
    # backend has an MPS device path, so Higgs must not be refused on Apple Silicon.
    import routes.inference as inference_route
    from core.inference import local_model_resolver, native_audio
    from utils.models import model_config

    monkeypatch.setattr(local_model_resolver, "_host_serves_mlx", lambda: True)
    monkeypatch.setattr(native_audio, "is_native_audio_model", lambda _p: True)
    monkeypatch.setattr(model_config, "detect_audio_type", lambda *_a, **_kw: "higgs_tts2")
    assert inference_route._target_speech_audio_type("/srv/higgs", False) == "higgs_tts2"


def test_a_non_gguf_speech_target_is_allowed_off_mlx(monkeypatch):
    import routes.inference as inference_route
    from core.inference import local_model_resolver
    from utils.models import model_config

    monkeypatch.setattr(local_model_resolver, "_host_serves_mlx", lambda: False)
    monkeypatch.setattr(model_config, "detect_audio_type", lambda *_a, **_kw: "snac")
    assert inference_route._target_speech_audio_type("/srv/orpheus", False) == "snac"
