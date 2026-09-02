# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import struct

import pytest

from utils.audio_tokens import SNAC_PROBE_TOKEN_IDS
from utils.models import gguf_metadata
from utils.models.gguf_metadata import read_gguf_tts_audio_type


def _string(value: str) -> bytes:
    data = value.encode()
    return struct.pack("<Q", len(data)) + data


def _write_gguf(path, tokens):
    metadata = _string("general.architecture") + struct.pack("<I", 8) + _string("llama")
    if tokens is not None:
        array = struct.pack("<IQ", 8, len(tokens)) + b"".join(_string(t) for t in tokens)
        metadata += _string("tokenizer.ggml.tokens") + struct.pack("<I", 9) + array
    kv_count = 1 + (tokens is not None)
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
