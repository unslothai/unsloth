# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import os
import struct
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

    captured = []

    def tokenizer(prompts, **_kwargs):
        captured.extend(prompts)
        raise RuntimeError("prompt captured")

    backend = InferenceBackend.__new__(InferenceBackend)
    backend._audio_codec_manager = SimpleNamespace()
    monkeypatch.setattr(backend, "_patch_repetition_penalty_processor", lambda: None)
    with pytest.raises(RuntimeError, match = "prompt captured"):
        backend._generate_dac(
            SimpleNamespace(device = "cpu", dtype = torch.float32),
            tokenizer,
            "Read this aloud.",
            temperature = 0.6,
            top_k = 50,
            top_p = 0.95,
            min_p = 0.0,
            max_new_tokens = 64,
            repetition_penalty = 1.1,
        )
    assert captured == [
        "<|im_start|>\n<|text_start|>Read this aloud.<|text_end|>\n<|audio_start|>\n"
    ]


def _string(value: str) -> bytes:
    data = value.encode()
    return struct.pack("<Q", len(data)) + data


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


@pytest.mark.parametrize(
    "tokens,types,expected",
    [
        (["<|bicodec_semantic_0|>", "<|bicodec_global_0|>"], None, "bicodec"),
        (["<|c1_0|>", "<|c2_0|>"], None, "dac"),
        (["<|c1_0|>", "<|c2_0|>"], [1, 1], None),
        (["<|bicodec_semantic_0|>"], None, None),
        (["<|audio|>", "<|c1_0|>", "<|c2_0|>"], None, None),
        (["<|audio_start|>", "<|audio_end|>"], None, None),
        (["hello", "<|im_start|>"], None, None),
        (None, None, None),
    ],
)
def test_gguf_speech_vocabulary(tmp_path, tokens, types, expected):
    path = _write_gguf(tmp_path / "model.gguf", tokens, types)
    assert read_gguf_tts_audio_type(path) == expected


@pytest.mark.parametrize("offset,expected", [(0, "snac"), (8, None)])
def test_snac_requires_runtime_probe_positions(tmp_path, offset, expected):
    assert (
        read_gguf_tts_audio_type(
            _write_gguf(tmp_path / "snac.gguf", ["pad"] * offset + _snac_vocab())
        )
        == expected
    )


def test_remote_prefix_rejects_truncation(tmp_path):
    from pathlib import Path

    path = _write_gguf(tmp_path / "model.gguf", ["<|c1_0|>", "<|c2_0|>"])
    data = Path(path).read_bytes()
    assert classify_gguf_tts_audio_prefix(data) == ("dac", True)
    assert classify_gguf_tts_audio_prefix(data[:-1]) == (None, False)
