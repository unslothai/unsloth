# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Tests for :func:`utils.models.model_config.detect_mmproj_file`.

Covers the disambiguation rules introduced for #5347, where a flat local
GGUF directory containing several unrelated models was attaching the wrong
projector to llama-server.
"""

from __future__ import annotations

from pathlib import Path

from utils.models.model_config import detect_mmproj_file


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return path


def test_returns_none_when_no_mmproj(tmp_path: Path):
    model = _touch(tmp_path / "Qwen3.5-9B-Q4_K_M.gguf")
    assert detect_mmproj_file(str(model)) is None


def test_single_matching_family_mmproj_picked(tmp_path: Path):
    """One same-family projector → return it (preserves historical behaviour)."""
    model = _touch(tmp_path / "Qwen3.5-9B-Q4_K_M.gguf")
    mmproj = _touch(tmp_path / "Qwen3.5-9B-BF16-mmproj.gguf")
    assert detect_mmproj_file(str(model)) == str(mmproj.resolve())


def test_hf_style_unprefixed_mmproj_still_works(tmp_path: Path):
    """HF convention: weight + ``mmproj-F16.gguf`` sibling, no other GGUFs."""
    model = _touch(tmp_path / "model.gguf")
    mmproj = _touch(tmp_path / "mmproj-F16.gguf")
    assert detect_mmproj_file(str(model)) == str(mmproj.resolve())


def test_blocks_single_cross_family_projector(tmp_path: Path):
    """
    #5347 (core case): a Qwen model in a flat dir with a single Gemma
    mmproj must NOT silently attach the Gemma projector — the load would
    fail and the user would see a confusing crash.
    """
    model = _touch(tmp_path / "Qwen3.5-9B-Q4_K_M.gguf")
    _touch(tmp_path / "gemma-4-26B-A4B-it.mmproj-q8_0.gguf")
    assert detect_mmproj_file(str(model)) is None


def test_picks_matching_family_among_mixed_candidates(tmp_path: Path):
    """
    Flat dir with Qwen, Gemma, and a Qwen-matching projector: pick the
    Qwen projector, drop the Gemma one.
    """
    model = _touch(tmp_path / "Qwen3.5-9B-Q4_K_M.gguf")
    qwen_mm = _touch(tmp_path / "Qwen3.5-9B-BF16-mmproj.gguf")
    _touch(tmp_path / "gemma-4-26B-A4B-it.mmproj-q8_0.gguf")
    assert detect_mmproj_file(str(model)) == str(qwen_mm.resolve())


def test_prefers_longest_prefix_within_same_family(tmp_path: Path):
    """
    Two same-family projectors (e.g. 9B and 35B Qwen) → pick the one
    whose stem shares the longest prefix with the model.
    """
    model = _touch(tmp_path / "Qwen3.5-35B-A3B-UD-Q4_K_L.gguf")
    _touch(tmp_path / "Qwen3.5-9B-BF16-mmproj.gguf")
    big_mm = _touch(tmp_path / "Qwen3.5-35B-A3B-BF16-mmproj.gguf")
    assert detect_mmproj_file(str(model)) == str(big_mm.resolve())


def test_unrecognised_family_does_not_break_detection(tmp_path: Path):
    """
    If the target model has no recognised family token, fall back to
    returning the first candidate — old behaviour. We never want to
    return None just because we don't recognise the family.
    """
    model = _touch(tmp_path / "MyCustomBrand-7B-Q4_K_M.gguf")
    mmproj = _touch(tmp_path / "MyCustomBrand-7B-BF16-mmproj.gguf")
    assert detect_mmproj_file(str(model)) == str(mmproj.resolve())


def test_directory_path_returns_first_candidate(tmp_path: Path):
    """
    When ``path`` is a directory (not a .gguf file), we have no model
    stem to compare against — return the first candidate, matching the
    pre-#5347 behaviour for that code path.
    """
    _touch(tmp_path / "Qwen3.5-9B-BF16-mmproj.gguf")
    _touch(tmp_path / "gemma-4-26B-A4B-it.mmproj-q8_0.gguf")
    result = detect_mmproj_file(str(tmp_path))
    assert result is not None
    assert "mmproj" in Path(result).name.lower()


def test_search_root_walk_still_works(tmp_path: Path):
    """
    Snapshot layout: weight in a quant subdir, mmproj at the root.
    The ``search_root`` walk must still find the projector after #5347.
    """
    snapshot = tmp_path / "snapshot"
    weight = _touch(snapshot / "BF16" / "Qwen3.5-9B-BF16.gguf")
    mmproj = _touch(snapshot / "Qwen3.5-9B-BF16-mmproj.gguf")
    result = detect_mmproj_file(str(weight), search_root=str(snapshot))
    assert result == str(mmproj.resolve())
