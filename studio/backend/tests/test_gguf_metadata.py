# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for :mod:`utils.models.gguf_metadata`. Synthesise small GGUF headers
in tmp dirs so we never depend on real model files."""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Iterable, Mapping

from utils.models.gguf_metadata import (
    detect_gguf_audio_type,
    is_mmproj_by_metadata,
    pairing_score,
    read_gguf_context_length,
    read_gguf_general_metadata,
    read_gguf_staged_dims,
    read_mmproj_audio_capability,
    read_mmproj_vision_capability,
)


_GGUF_MAGIC = 0x46554747
_VTYPE_STRING = 8
_VTYPE_UINT32 = 4
_VTYPE_UINT64 = 10
_VTYPE_ARRAY = 9
_VTYPE_BOOL = 7


def _enc_string(s: str) -> bytes:
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def _enc_kv_string(key: str, value: str) -> bytes:
    return _enc_string(key) + struct.pack("<I", _VTYPE_STRING) + _enc_string(value)


def _enc_kv_uint32(key: str, value: int) -> bytes:
    return _enc_string(key) + struct.pack("<I", _VTYPE_UINT32) + struct.pack("<I", value)


def _enc_kv_uint64(key: str, value: int) -> bytes:
    return _enc_string(key) + struct.pack("<I", _VTYPE_UINT64) + struct.pack("<Q", value)


def _enc_kv_bool(key: str, value: bool) -> bytes:
    return _enc_string(key) + struct.pack("<I", _VTYPE_BOOL) + struct.pack("<B", 1 if value else 0)


def _enc_kv_string_array(key: str, values: Iterable[str]) -> bytes:
    vals = list(values)
    out = _enc_string(key) + struct.pack("<I", _VTYPE_ARRAY)
    out += struct.pack("<I", _VTYPE_STRING) + struct.pack("<Q", len(vals))
    for v in vals:
        out += _enc_string(v)
    return out


def _write_synthetic_gguf(
    path: Path,
    general_strings: Mapping[str, str],
    *,
    extra_uint32: Mapping[str, int] | None = None,
    extra_uint64: Mapping[str, int] | None = None,
    extra_string_arrays: Mapping[str, Iterable[str]] | None = None,
    extra_bools: Mapping[str, bool] | None = None,
) -> Path:
    """Minimal GGUF: header + KV body, no tensors."""
    extra_uint32 = extra_uint32 or {}
    extra_uint64 = extra_uint64 or {}
    extra_string_arrays = extra_string_arrays or {}
    extra_bools = extra_bools or {}
    kv_count = (
        len(general_strings)
        + len(extra_uint32)
        + len(extra_uint64)
        + len(extra_string_arrays)
        + len(extra_bools)
    )
    body = b""
    for k, v in general_strings.items():
        body += _enc_kv_string(k, v)
    for k, v in extra_uint32.items():
        body += _enc_kv_uint32(k, v)
    for k, v in extra_uint64.items():
        body += _enc_kv_uint64(k, v)
    for k, v in extra_string_arrays.items():
        body += _enc_kv_string_array(k, v)
    for k, v in extra_bools.items():
        body += _enc_kv_bool(k, v)
    header = struct.pack(
        "<IIQQ",
        _GGUF_MAGIC,
        3,  # version
        0,  # tensor_count
        kv_count,
    )
    path.parent.mkdir(parents = True, exist_ok = True)
    path.write_bytes(header + body)
    return path


# --- read_gguf_general_metadata ----------------------------------------


def test_returns_none_for_missing_file(tmp_path: Path):
    assert read_gguf_general_metadata(str(tmp_path / "nope.gguf")) is None


def test_returns_none_for_non_gguf(tmp_path: Path):
    p = tmp_path / "garbage.gguf"
    p.write_bytes(b"not a gguf file at all, just bytes")
    assert read_gguf_general_metadata(str(p)) is None


def test_context_length_none_for_missing_file(tmp_path: Path):
    assert read_gguf_context_length(str(tmp_path / "nope.gguf")) is None


def test_context_length_none_for_non_gguf(tmp_path: Path):
    p = tmp_path / "garbage.gguf"
    p.write_bytes(b"not a gguf file at all, just bytes")
    assert read_gguf_context_length(str(p)) is None


def test_context_length_read_from_arch_namespaced_key(tmp_path: Path):
    p = _write_synthetic_gguf(
        tmp_path / "model.gguf",
        {"general.architecture": "llama"},
        extra_uint32 = {"llama.context_length": 4096, "llama.block_count": 32},
    )
    assert read_gguf_context_length(str(p)) == 4096


def test_context_length_none_when_absent(tmp_path: Path):
    # Architecture present but no <arch>.context_length key.
    p = _write_synthetic_gguf(
        tmp_path / "model.gguf",
        {"general.architecture": "llama"},
        extra_uint32 = {"llama.block_count": 32},
    )
    assert read_gguf_context_length(str(p)) is None


def test_context_length_ignores_foreign_arch_key(tmp_path: Path):
    # A context_length under a different arch namespace must not match.
    p = _write_synthetic_gguf(
        tmp_path / "model.gguf",
        {"general.architecture": "llama"},
        extra_uint32 = {"qwen2.context_length": 8192},
    )
    assert read_gguf_context_length(str(p)) is None


# --- read_gguf_staged_dims (one pass: context + layer + moe counts) ----


def test_staged_dims_none_for_missing_or_non_gguf(tmp_path: Path):
    assert read_gguf_staged_dims(str(tmp_path / "nope.gguf")) is None
    p = tmp_path / "garbage.gguf"
    p.write_bytes(b"not a gguf at all")
    assert read_gguf_staged_dims(str(p)) is None


def test_staged_dims_moe_with_leading_dense(tmp_path: Path):
    # GLM-4.7-Flash shape: context + total layers + MoE layers in one read.
    p = _write_synthetic_gguf(
        tmp_path / "glm.gguf",
        {"general.architecture": "deepseek2"},
        extra_uint32 = {
            "deepseek2.context_length": 202752,
            "deepseek2.block_count": 47,
            "deepseek2.expert_count": 64,
            "deepseek2.leading_dense_block_count": 1,
        },
    )
    assert read_gguf_staged_dims(str(p)) == {
        "context_length": 202752,
        "layer_count": 47,
        "moe_layer_count": 46,
    }


def test_staged_dims_dense_model(tmp_path: Path):
    # Dense: layer_count present, moe_layer_count 0 (slider hidden).
    p = _write_synthetic_gguf(
        tmp_path / "dense.gguf",
        {"general.architecture": "qwen3"},
        extra_uint32 = {"qwen3.context_length": 40960, "qwen3.block_count": 36},
    )
    assert read_gguf_staged_dims(str(p)) == {
        "context_length": 40960,
        "layer_count": 36,
        "moe_layer_count": 0,
    }


def test_staged_dims_all_moe_no_leading_dense(tmp_path: Path):
    # Experts present, no leading_dense key -> every block is a MoE layer.
    p = _write_synthetic_gguf(
        tmp_path / "moe.gguf",
        {"general.architecture": "qwen35moe"},
        extra_uint32 = {"qwen35moe.block_count": 40, "qwen35moe.expert_count": 256},
    )
    assert read_gguf_staged_dims(str(p)) == {
        "context_length": None,
        "layer_count": 40,
        "moe_layer_count": 40,
    }


def test_staged_dims_uint64_block_count(tmp_path: Path):
    # block_count stored as uint64 (vtype 10) still parses; moe == block_count.
    p = _write_synthetic_gguf(
        tmp_path / "moe64.gguf",
        {"general.architecture": "gpt-oss"},
        extra_uint32 = {"gpt-oss.expert_count": 32},
        extra_uint64 = {"gpt-oss.block_count": 24},
    )
    assert read_gguf_staged_dims(str(p)) == {
        "context_length": None,
        "layer_count": 24,
        "moe_layer_count": 24,
    }


def test_context_length_read_from_uint64(tmp_path: Path):
    # Some models store context_length as a uint64 (vtype 10).
    p = _write_synthetic_gguf(
        tmp_path / "model.gguf",
        {"general.architecture": "qwen3"},
        extra_uint64 = {"qwen3.context_length": 262144},
    )
    assert read_gguf_context_length(str(p)) == 262144


def test_context_length_zero_treated_as_absent(tmp_path: Path):
    # A zero/garbage ceiling must read as None so the UI can't build a slider
    # with max < min.
    p = _write_synthetic_gguf(
        tmp_path / "model.gguf",
        {"general.architecture": "llama"},
        extra_uint32 = {"llama.context_length": 0},
    )
    assert read_gguf_context_length(str(p)) is None


def test_extracts_general_string_fields(tmp_path: Path):
    p = _write_synthetic_gguf(
        tmp_path / "model.gguf",
        {
            "general.architecture": "qwen2vl",
            "general.type": "model",
            "general.basename": "Qwen3.5",
            "general.organization": "Qwen",
            "general.base_model.0.repo_url": "https://huggingface.co/Qwen/Qwen3.5-9B",
            "general.base_model.0.name": "Qwen3.5 9B",
            "general.base_model.0.organization": "Qwen",
        },
    )
    meta = read_gguf_general_metadata(str(p))
    assert meta is not None
    assert meta["general.architecture"] == "qwen2vl"
    assert meta["general.basename"] == "Qwen3.5"
    assert meta["general.base_model.0.repo_url"] == "https://huggingface.co/Qwen/Qwen3.5-9B"


def test_skips_unrelated_fields_without_breaking(tmp_path: Path):
    """Skip unwanted arrays and uint32s without losing position."""
    p = _write_synthetic_gguf(
        tmp_path / "model.gguf",
        {"general.basename": "Foo"},
        extra_uint32 = {"qwen2vl.context_length": 32768},
        extra_string_arrays = {"tokenizer.ggml.tokens": ["a", "bc", "def"]},
    )
    meta = read_gguf_general_metadata(str(p))
    assert meta == {"general.basename": "Foo"}


def test_metadata_is_cached(tmp_path: Path):
    """Cache invalidates on size change."""
    p = _write_synthetic_gguf(
        tmp_path / "model.gguf",
        {"general.basename": "First"},
    )
    first = read_gguf_general_metadata(str(p))
    assert first == {"general.basename": "First"}
    # Change size so the (path, mtime, size) cache key invalidates.
    _write_synthetic_gguf(
        tmp_path / "model.gguf",
        {"general.basename": "Second", "general.organization": "X"},
    )
    second = read_gguf_general_metadata(str(p))
    assert second == {"general.basename": "Second", "general.organization": "X"}


# --- is_mmproj_by_metadata --------------------------------------------


def test_is_mmproj_by_metadata_signals():
    assert is_mmproj_by_metadata({"general.type": "mmproj"}) is True
    assert is_mmproj_by_metadata({"general.type": "MMProj"}) is True
    assert is_mmproj_by_metadata({"general.type": "model"}) is False
    assert is_mmproj_by_metadata({"general.basename": "foo"}) is None
    assert is_mmproj_by_metadata({}) is None
    assert is_mmproj_by_metadata(None) is None


# --- pairing_score -----------------------------------------------------


def test_pairing_score_base_model_url_match():
    weight = {
        "general.base_model.0.repo_url": "https://huggingface.co/Qwen/Qwen3.5-9B",
    }
    mmproj = {
        "general.base_model.0.repo_url": "https://huggingface.co/Qwen/Qwen3.5-9B",
    }
    assert pairing_score(weight, mmproj) == 100


def test_pairing_score_base_model_url_mismatch():
    weight = {
        "general.base_model.0.repo_url": "https://huggingface.co/Qwen/Qwen3.5-9B",
    }
    mmproj = {
        "general.base_model.0.repo_url": "https://huggingface.co/google/gemma-3-9B",
    }
    assert pairing_score(weight, mmproj) == -1


def test_pairing_score_base_model_url_trailing_slash_normalised():
    weight = {
        "general.base_model.0.repo_url": "https://huggingface.co/Qwen/Qwen3.5-9B/",
    }
    mmproj = {
        "general.base_model.0.repo_url": "https://huggingface.co/Qwen/Qwen3.5-9B",
    }
    assert pairing_score(weight, mmproj) == 100


def test_pairing_score_basename_plus_org_fallback():
    weight = {
        "general.basename": "Nanonets-Ocr-S",
        "general.base_model.0.organization": "Nanonets",
    }
    mmproj = {
        "general.basename": "Nanonets-Ocr-S",
        "general.base_model.0.organization": "Nanonets",
    }
    assert pairing_score(weight, mmproj) == 80


def test_pairing_score_basename_only_fallback():
    assert (
        pairing_score(
            {"general.basename": "Nanonets-Ocr-S"},
            {"general.basename": "Nanonets-Ocr-S"},
        )
        == 60
    )


def test_pairing_score_no_overlap_returns_zero():
    """One side empty: scorer punts to filename fallback."""
    assert pairing_score({"general.basename": "Foo"}, {}) == 0
    assert pairing_score({}, {"general.basename": "Foo"}) == 0
    assert pairing_score(None, {"general.basename": "Foo"}) == 0


# --- read_mmproj_audio_capability --------------------------------------


def test_mmproj_audio_capability_true(tmp_path: Path):
    """clip.has_audio_encoder=True (e.g. Gemma 4's gemma4ua projector)."""
    p = _write_synthetic_gguf(
        tmp_path / "mmproj.gguf",
        {"general.type": "mmproj"},
        extra_bools = {
            "clip.has_vision_encoder": True,
            "clip.has_audio_encoder": True,
        },
    )
    assert read_mmproj_audio_capability(str(p)) is True


def test_mmproj_audio_capability_false(tmp_path: Path):
    """Vision-only projector: key present but false."""
    p = _write_synthetic_gguf(
        tmp_path / "mmproj.gguf",
        {"general.type": "mmproj"},
        extra_bools = {
            "clip.has_vision_encoder": True,
            "clip.has_audio_encoder": False,
        },
    )
    assert read_mmproj_audio_capability(str(p)) is False


def test_mmproj_audio_capability_absent_returns_none(tmp_path: Path):
    """Key absent (older/vision-only mmproj): None, not False."""
    p = _write_synthetic_gguf(
        tmp_path / "mmproj.gguf",
        {"general.type": "mmproj"},
        extra_bools = {"clip.has_vision_encoder": True},
    )
    assert read_mmproj_audio_capability(str(p)) is None


def test_mmproj_audio_capability_missing_or_non_gguf(tmp_path: Path):
    assert read_mmproj_audio_capability(str(tmp_path / "nope.gguf")) is None
    junk = tmp_path / "garbage.gguf"
    junk.write_bytes(b"not a gguf header at all")
    assert read_mmproj_audio_capability(str(junk)) is None


def test_mmproj_vision_capability_distinguishes_audio_only(tmp_path: Path):
    p = _write_synthetic_gguf(
        tmp_path / "audio-mmproj.gguf",
        {"general.type": "mmproj"},
        extra_bools = {
            "clip.has_vision_encoder": False,
            "clip.has_audio_encoder": True,
        },
    )
    assert read_mmproj_vision_capability(str(p)) is False


def test_shared_csm_tokens_default_to_chat_audio_without_name_identity(tmp_path: Path):
    p = _write_synthetic_gguf(
        tmp_path / "generic.gguf",
        {"general.architecture": "llama"},
        extra_string_arrays = {
            "tokenizer.ggml.tokens": ["<s>", "<|AUDIO|>", "<|audio_eos|>"],
        },
    )
    assert detect_gguf_audio_type(str(p)) == "audio_vlm"


def test_csm_tokens_require_corroborating_name_identity(tmp_path: Path):
    p = _write_synthetic_gguf(
        tmp_path / "csm.gguf",
        {"general.architecture": "llama", "general.name": "sesame-csm-1b"},
        extra_string_arrays = {
            "tokenizer.ggml.tokens": ["<s>", "<|AUDIO|>", "<|audio_eos|>"],
        },
    )
    assert detect_gguf_audio_type(str(p)) == "csm"


def test_dac_runtime_tokens_match_without_wrapper_markers(tmp_path: Path):
    p = _write_synthetic_gguf(
        tmp_path / "generic.gguf",
        {"general.architecture": "llama"},
        extra_string_arrays = {
            "tokenizer.ggml.tokens": ["<s>", "<|c1_0|>", "<|c2_0|>"],
        },
    )
    assert detect_gguf_audio_type(str(p)) == "dac"


def test_qwen3_asr_projector_metadata_is_non_chat_audio(tmp_path: Path):
    p = _write_synthetic_gguf(
        tmp_path / "mmproj-qwen3-asr.gguf",
        {
            "general.type": "mmproj",
            "clip.audio.projector_type": "qwen3a",
        },
        extra_bools = {
            "clip.has_audio_encoder": True,
            "clip.has_vision_encoder": False,
        },
    )
    assert detect_gguf_audio_type(str(p)) == "asr"


def test_vocab_sized_fixed_array_before_tokens_is_skipped(tmp_path: Path):
    # Regression: the previous 2**18 entry cap rejected valid metadata arrays
    # even though the token vocabulary itself allowed up to 2**20 entries.
    fixed_count = (1 << 18) + 1
    fixed_array = (
        _enc_string("tokenizer.ggml.token_type")
        + struct.pack("<I", _VTYPE_ARRAY)
        + struct.pack("<IQ", _VTYPE_UINT32, fixed_count)
        + (b"\x00" * (fixed_count * 4))
    )
    tokens = _enc_kv_string_array(
        "tokenizer.ggml.tokens",
        ["<|AUDIO|>", "<|audio_eos|>"],
    )
    p = tmp_path / "large-metadata-array.gguf"
    p.write_bytes(struct.pack("<IIQQ", _GGUF_MAGIC, 3, 0, 2) + fixed_array + tokens)
    assert detect_gguf_audio_type(str(p)) == "audio_vlm"
