# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import struct

import pytest

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


@pytest.fixture(autouse = True)
def _clear_cache():
    gguf_metadata._TTS_AUDIO_TYPE_CACHE.clear()


def test_orpheus_snac_codes_are_a_speech_model(tmp_path):
    tokens = ["hello", "<|eot_id|>", *(f"<custom_token_{i}>" for i in range(10_002))]
    assert read_gguf_tts_audio_type(_write_gguf(tmp_path / "orpheus.gguf", tokens)) == "snac"


def test_spark_bicodec_tokens_are_a_speech_model(tmp_path):
    tokens = ["hi", "<|bicodec_semantic_0|>", "<|bicodec_global_0|>"]
    assert read_gguf_tts_audio_type(_write_gguf(tmp_path / "spark.gguf", tokens)) == "bicodec"


def test_outetts_dac_tokens_are_a_speech_model(tmp_path):
    tokens = ["<|audio_start|>", "<|audio_end|>", "<|text_start|>", "<|text_end|>"]
    assert read_gguf_tts_audio_type(_write_gguf(tmp_path / "oute.gguf", tokens)) == "dac"


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
    assert inference_route._target_speaks(str(tmp_path), True, "Q8_0") is True
    assert inference_route._target_speaks(str(tmp_path), True, "Q4_K_M") is False
    monkeypatch.setattr(model_config, "detect_gguf_model", lambda _p, *_a: speech)
    assert inference_route._target_speaks(speech, True) is True


def test_the_switch_probe_asks_a_checkpoint_for_an_output_codec(monkeypatch):
    import routes.inference as inference_route
    from utils.models import model_config

    seen = {}

    def _detect(path, **kwargs):
        seen.update(kwargs)
        return {"/srv/tts": "csm", "/srv/higgs": "higgs_tts2", "/srv/asr": "whisper"}.get(path)

    monkeypatch.setattr(model_config, "detect_audio_type", _detect)
    assert inference_route._target_speaks("/srv/tts", False) is True
    assert inference_route._target_speaks("/srv/higgs", False) is True
    assert inference_route._target_speaks("/srv/asr", False) is False
    assert inference_route._target_speaks("/srv/text", False) is False
    assert seen["local_files_only"] is True


def test_a_failing_probe_refuses_rather_than_evicts(monkeypatch):
    import routes.inference as inference_route
    from utils.models import model_config

    def _boom(*_a, **_kw):
        raise RuntimeError("unreadable")

    monkeypatch.setattr(model_config, "detect_audio_type", _boom)
    assert inference_route._target_speaks("/srv/tts", False) is False
