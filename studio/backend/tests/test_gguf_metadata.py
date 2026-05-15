# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for :mod:`utils.models.gguf_metadata`. Synthesise small GGUF
headers in tmp dirs so we never depend on real model files."""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Iterable, Mapping

from utils.models.gguf_metadata import (
    is_mmproj_by_metadata,
    pairing_score,
    read_gguf_general_metadata,
)


_GGUF_MAGIC = 0x46554747
_VTYPE_STRING = 8
_VTYPE_UINT32 = 4
_VTYPE_ARRAY = 9


def _enc_string(s: str) -> bytes:
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def _enc_kv_string(key: str, value: str) -> bytes:
    return _enc_string(key) + struct.pack("<I", _VTYPE_STRING) + _enc_string(value)


def _enc_kv_uint32(key: str, value: int) -> bytes:
    return (
        _enc_string(key) + struct.pack("<I", _VTYPE_UINT32) + struct.pack("<I", value)
    )


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
    extra_string_arrays: Mapping[str, Iterable[str]] | None = None,
) -> Path:
    """Minimal GGUF: header + KV body, no tensors."""
    extra_uint32 = extra_uint32 or {}
    extra_string_arrays = extra_string_arrays or {}
    kv_count = len(general_strings) + len(extra_uint32) + len(extra_string_arrays)
    body = b""
    for k, v in general_strings.items():
        body += _enc_kv_string(k, v)
    for k, v in extra_uint32.items():
        body += _enc_kv_uint32(k, v)
    for k, v in extra_string_arrays.items():
        body += _enc_kv_string_array(k, v)
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
    assert (
        meta["general.base_model.0.repo_url"]
        == "https://huggingface.co/Qwen/Qwen3.5-9B"
    )


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
    # Force size change so the (path, mtime, size) key invalidates.
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
