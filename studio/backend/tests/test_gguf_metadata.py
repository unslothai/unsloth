# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for :mod:`utils.models.gguf_metadata`. Synthesise small GGUF headers
in tmp dirs so we never depend on real model files."""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Iterable, Mapping
from unittest.mock import patch

from utils.models.gguf_metadata import (
    is_gguf_embedding_architecture,
    is_gguf_embedding_model,
    is_mmproj_by_metadata,
    mmproj_accepts_image,
    pairing_score,
    read_gguf_architecture,
    read_gguf_context_length,
    read_gguf_general_metadata,
    read_gguf_staged_dims,
    read_mmproj_audio_capability,
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
    tensor_names: Iterable[str] | None = None,
    extra_uint32: Mapping[str, int] | None = None,
    extra_uint64: Mapping[str, int] | None = None,
    extra_string_arrays: Mapping[str, Iterable[str]] | None = None,
    extra_bools: Mapping[str, bool] | None = None,
) -> Path:
    """Minimal GGUF header with optional tensor-table entries and no tensor data."""
    tensor_names = list(tensor_names or ())
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
    tensor_info = b""
    for name in tensor_names:
        tensor_info += _enc_string(name)
        tensor_info += struct.pack("<I", 1)  # n_dimensions
        tensor_info += struct.pack("<Q", 1)  # dimensions
        tensor_info += struct.pack("<I", 0)  # GGML_TYPE_F32
        tensor_info += struct.pack("<Q", 0)  # data offset
    header = struct.pack(
        "<IIQQ",
        _GGUF_MAGIC,
        3,  # version
        len(tensor_names),
        kv_count,
    )
    path.parent.mkdir(parents = True, exist_ok = True)
    path.write_bytes(header + body + tensor_info)
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


# read_gguf_architecture


class _CountingFile:
    """A file handle that records how many reads and seeks a parser performs on it."""

    def __init__(self, handle) -> None:
        self._handle = handle
        self.operations = 0

    def read(self, size = -1):
        self.operations += 1
        return self._handle.read(size)

    def seek(
        self,
        offset,
        whence = 0,
    ):
        self.operations += 1
        return self._handle.seek(offset, whence)

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return self._handle.__exit__(*exc_info)


def _count_header_operations(monkeypatch, path: Path, read) -> int:
    """File operations ``read`` performs against ``path``. Other files are untouched."""
    import builtins

    real_open = builtins.open
    counters: list[_CountingFile] = []

    def counting_open(file, *args, **kwargs):
        handle = real_open(file, *args, **kwargs)
        if str(file) != str(path):
            return handle
        counter = _CountingFile(handle)
        counters.append(counter)
        return counter

    monkeypatch.setattr(builtins, "open", counting_open)
    try:
        read(str(path))
    finally:
        monkeypatch.undo()
    return sum(counter.operations for counter in counters)


def test_architecture_read_stops_before_a_large_tokenizer_array(tmp_path: Path, monkeypatch):
    """Reading one key must not scan a large tokenizer array."""
    p = _write_synthetic_gguf(
        tmp_path / "model.gguf",
        {"general.architecture": "llama", "general.name": "Test"},
        extra_string_arrays = {"tokenizer.ggml.tokens": [f"tok{i}" for i in range(20000)]},
    )

    assert read_gguf_architecture(str(p)) == "llama"
    assert read_gguf_general_metadata(str(p)) == {
        "general.architecture": "llama",
        "general.name": "Test",
    }

    # Use separate paths to avoid the readers' file-stat caches.
    targeted = tmp_path / "targeted.gguf"
    targeted.write_bytes(p.read_bytes())
    whole = tmp_path / "whole.gguf"
    whole.write_bytes(p.read_bytes())

    targeted_ops = _count_header_operations(monkeypatch, targeted, read_gguf_architecture)
    whole_ops = _count_header_operations(monkeypatch, whole, read_gguf_general_metadata)
    assert targeted_ops < 10, targeted_ops
    assert whole_ops > 20000, whole_ops


def test_architecture_is_stripped_and_absent_values_are_none(tmp_path: Path):
    stripped = _write_synthetic_gguf(
        tmp_path / "padded.gguf", {"general.architecture": "  ltxv \n"}
    )
    assert read_gguf_architecture(str(stripped)) == "ltxv"

    blank = _write_synthetic_gguf(tmp_path / "blank.gguf", {"general.architecture": "   "})
    assert read_gguf_architecture(str(blank)) is None

    empty = _write_synthetic_gguf(tmp_path / "empty.gguf", {})
    assert read_gguf_architecture(str(empty)) is None

    assert read_gguf_architecture(str(tmp_path / "nope.gguf")) is None
    junk = tmp_path / "garbage.gguf"
    junk.write_bytes(b"not a gguf file at all, just bytes")
    assert read_gguf_architecture(str(junk)) is None


def test_architecture_matches_the_general_metadata_reader(tmp_path: Path):
    """The targeted and general readers must return the same architecture."""
    for index, arch in enumerate(("llama", "flux2", "ltxv", "dflash", "qwen3vl")):
        p = _write_synthetic_gguf(
            tmp_path / f"model{index}.gguf",
            {"general.architecture": arch, "general.name": "n", "general.type": "model"},
            extra_uint32 = {f"{arch}.block_count": 32},
        )
        expected = (read_gguf_general_metadata(str(p)) or {}).get("general.architecture")
        assert read_gguf_architecture(str(p)) == expected == arch


# --- mmproj_accepts_image ----------------------------------------------


def _projector(tmp_path: Path, **bools) -> str:
    return str(
        _write_synthetic_gguf(
            tmp_path / "mmproj.gguf", {"general.type": "mmproj"}, extra_bools = bools
        )
    )


def test_accepts_image_when_vision_declared(tmp_path: Path):
    """A projector declaring vision accepts images, whatever it says about audio."""
    assert mmproj_accepts_image(_projector(tmp_path, **{"clip.has_vision_encoder": True})) is True


def test_audio_only_projector_is_not_a_vision_tower(tmp_path: Path):
    """ultravox / Voxtral / Qwen3-ASR: audio declared, vision key absent."""
    assert mmproj_accepts_image(_projector(tmp_path, **{"clip.has_audio_encoder": True})) is False


def test_dual_projector_still_accepts_images(tmp_path: Path):
    """Qwen2.5-Omni declares both, which is what makes a lone audio claim evidence."""
    p = _projector(tmp_path, **{"clip.has_vision_encoder": True, "clip.has_audio_encoder": True})
    assert mmproj_accepts_image(p) is True


def test_projector_declaring_neither_stays_image_capable(tmp_path: Path):
    """An unreadable or older convert must not be refused: the load decides."""
    assert mmproj_accepts_image(_projector(tmp_path)) is True
    assert mmproj_accepts_image(str(tmp_path / "nope.gguf")) is True


def test_declared_audio_false_is_not_an_audio_claim(tmp_path: Path):
    """A vision projector writing audio=False is still a vision tower."""
    p = _projector(tmp_path, **{"clip.has_vision_encoder": True, "clip.has_audio_encoder": False})
    assert mmproj_accepts_image(p) is True


def test_is_gguf_embedding_architecture_recognises_encoder_arches():
    assert is_gguf_embedding_architecture("nomic-bert")
    assert is_gguf_embedding_architecture("NOMIC-BERT-MOE")
    assert not is_gguf_embedding_architecture("bert")
    assert not is_gguf_embedding_architecture("llama")
    assert not is_gguf_embedding_architecture(None)


def test_is_gguf_embedding_model_from_architecture(tmp_path: Path):
    p = _write_synthetic_gguf(
        tmp_path / "nomic.gguf",
        {"general.architecture": "nomic-bert"},
    )
    assert is_gguf_embedding_model(str(p)) is True


def test_is_gguf_embedding_model_from_name_hint(tmp_path: Path):
    p = _write_synthetic_gguf(
        tmp_path / "Qwen3-Embedding-4B-Q4_K_M.gguf",
        {"general.architecture": "qwen3"},
    )
    with patch("utils.models.model_config.is_embedding_model") as remote_classifier:
        assert (
            is_gguf_embedding_model(str(p), model_identifier = "unsloth/Qwen3-Embedding-4B") is True
        )
    remote_classifier.assert_not_called()


def test_is_gguf_embedding_model_ignores_owner_name_hint(tmp_path: Path):
    p = _write_synthetic_gguf(
        tmp_path / "Llama-3-Q4_K_M.gguf",
        {"general.architecture": "llama"},
    )
    with patch("utils.models.model_config.is_embedding_model") as remote_classifier:
        assert (
            is_gguf_embedding_model(str(p), model_identifier = "embedding-lab/Llama-3-GGUF") is False
        )
    remote_classifier.assert_not_called()


def test_is_gguf_embedding_model_from_intrinsic_name_hints(tmp_path: Path):
    for index, (key, value) in enumerate(
        (
            ("general.name", "Qwen3 Embedding 4B"),
            ("general.basename", "Qwen3-Embedding"),
        )
    ):
        p = _write_synthetic_gguf(
            tmp_path / f"model-{index}.gguf",
            {"general.architecture": "qwen3", key: value},
        )
        assert is_gguf_embedding_model(str(p), model_identifier = "local/model") is True


def test_is_gguf_embedding_model_excludes_reranker_without_pooling(tmp_path: Path):
    p = _write_synthetic_gguf(
        tmp_path / "bge-reranker-v2-m3-Q4_K_M.gguf",
        {"general.architecture": "bert", "general.name": "Bge M3"},
    )
    assert (
        is_gguf_embedding_model(str(p), model_identifier = "gpustack/bge-reranker-v2-m3-GGUF")
        is False
    )


def test_is_gguf_embedding_model_rejects_generic_bert_without_pooling(tmp_path: Path):
    p = _write_synthetic_gguf(
        tmp_path / "bge-small-en-v1.5.gguf",
        {"general.architecture": "bert", "general.name": "Bge Small Encoder"},
    )
    # No classifier head proves this is not a reranker, but it cannot tell us
    # whether the missing pooling strategy should be CLS or MEAN.
    assert is_gguf_embedding_model(str(p), model_identifier = "local/bge-small") is False


def test_is_gguf_embedding_model_excludes_unnamed_encoder_classifier_heads(tmp_path: Path):
    for index, classifier_tensor in enumerate(("cls.weight", "cls.output.weight")):
        p = _write_synthetic_gguf(
            tmp_path / f"model-{index}.gguf",
            {"general.architecture": "modern-bert", "general.name": "MS MARCO Encoder"},
            tensor_names = (classifier_tensor,),
        )
        assert is_gguf_embedding_model(str(p), model_identifier = "local/model") is False


def test_is_gguf_embedding_model_checks_every_split_for_classifier_head(tmp_path: Path):
    first = _write_synthetic_gguf(
        tmp_path / "model-00001-of-00002.gguf",
        {"general.architecture": "modern-bert"},
    )
    _write_synthetic_gguf(
        tmp_path / "model-00002-of-00002.gguf",
        {"general.architecture": "modern-bert"},
        tensor_names = ("cls.weight",),
    )
    assert is_gguf_embedding_model(str(first), model_identifier = "local/model") is False
