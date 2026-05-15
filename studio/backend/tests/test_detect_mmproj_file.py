# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for :func:`utils.models.model_config.detect_mmproj_file` (#5347)."""

from __future__ import annotations

from pathlib import Path

import struct

from utils.models.model_config import (
    _detect_family_token,
    detect_mmproj_file,
    mmproj_matches_model_family,
)


_GGUF_MAGIC = 0x46554747


def _gguf_with_general(path: Path, fields: dict) -> Path:
    """Write a minimal GGUF with only ``general.*`` string KVs."""
    body = b""
    for k, v in fields.items():
        kb = k.encode("utf-8")
        vb = v.encode("utf-8")
        body += struct.pack("<Q", len(kb)) + kb
        body += struct.pack("<I", 8)  # STRING vtype
        body += struct.pack("<Q", len(vb)) + vb
    header = struct.pack("<IIQQ", _GGUF_MAGIC, 3, 0, len(fields))
    path.parent.mkdir(parents = True, exist_ok = True)
    path.write_bytes(header + body)
    return path


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents = True, exist_ok = True)
    path.write_bytes(b"")
    return path


def test_returns_none_when_no_mmproj(tmp_path: Path):
    model = _touch(tmp_path / "Qwen3.5-9B-Q4_K_M.gguf")
    assert detect_mmproj_file(str(model)) is None


def test_single_matching_family_mmproj_picked(tmp_path: Path):
    """Single same-family projector: returned (historical behaviour)."""
    model = _touch(tmp_path / "Qwen3.5-9B-Q4_K_M.gguf")
    mmproj = _touch(tmp_path / "Qwen3.5-9B-BF16-mmproj.gguf")
    assert detect_mmproj_file(str(model)) == str(mmproj.resolve())


def test_hf_style_unprefixed_mmproj_still_works(tmp_path: Path):
    """HF convention: weight + ``mmproj-F16.gguf`` sibling."""
    model = _touch(tmp_path / "model.gguf")
    mmproj = _touch(tmp_path / "mmproj-F16.gguf")
    assert detect_mmproj_file(str(model)) == str(mmproj.resolve())


def test_blocks_single_cross_family_projector(tmp_path: Path):
    """#5347 core: Qwen weight + lone Gemma mmproj returns None."""
    model = _touch(tmp_path / "Qwen3.5-9B-Q4_K_M.gguf")
    _touch(tmp_path / "gemma-4-26B-A4B-it.mmproj-q8_0.gguf")
    assert detect_mmproj_file(str(model)) is None


def test_picks_matching_family_among_mixed_candidates(tmp_path: Path):
    """Mixed Qwen + Gemma projectors: pick Qwen, drop Gemma."""
    model = _touch(tmp_path / "Qwen3.5-9B-Q4_K_M.gguf")
    qwen_mm = _touch(tmp_path / "Qwen3.5-9B-BF16-mmproj.gguf")
    _touch(tmp_path / "gemma-4-26B-A4B-it.mmproj-q8_0.gguf")
    assert detect_mmproj_file(str(model)) == str(qwen_mm.resolve())


def test_prefers_longest_prefix_within_same_family(tmp_path: Path):
    """Same family, different sizes: longest shared stem prefix wins."""
    model = _touch(tmp_path / "Qwen3.5-35B-A3B-UD-Q4_K_L.gguf")
    _touch(tmp_path / "Qwen3.5-9B-BF16-mmproj.gguf")
    big_mm = _touch(tmp_path / "Qwen3.5-35B-A3B-BF16-mmproj.gguf")
    assert detect_mmproj_file(str(model)) == str(big_mm.resolve())


def test_unrecognised_family_does_not_break_detection(tmp_path: Path):
    """Unknown model family must not return None on a sole candidate."""
    model = _touch(tmp_path / "MyCustomBrand-7B-Q4_K_M.gguf")
    mmproj = _touch(tmp_path / "MyCustomBrand-7B-BF16-mmproj.gguf")
    assert detect_mmproj_file(str(model)) == str(mmproj.resolve())


def test_directory_path_returns_first_candidate(tmp_path: Path):
    """Directory path: no model stem to compare; legacy first-candidate."""
    _touch(tmp_path / "Qwen3.5-9B-BF16-mmproj.gguf")
    _touch(tmp_path / "gemma-4-26B-A4B-it.mmproj-q8_0.gguf")
    result = detect_mmproj_file(str(tmp_path))
    assert result is not None
    assert "mmproj" in Path(result).name.lower()


def test_search_root_walk_still_works(tmp_path: Path):
    """Snapshot layout: weight in quant subdir, mmproj at snapshot root."""
    snapshot = tmp_path / "snapshot"
    weight = _touch(snapshot / "BF16" / "Qwen3.5-9B-BF16.gguf")
    mmproj = _touch(snapshot / "Qwen3.5-9B-BF16-mmproj.gguf")
    result = detect_mmproj_file(str(weight), search_root = str(snapshot))
    assert result == str(mmproj.resolve())


# -- Family token detection: word-bounded matching ----------------------


def test_family_token_phi_does_not_match_sapphire():
    """``phi`` substring inside ``sapphire`` must not tag Phi."""
    assert _detect_family_token("sapphire-7b-q4_k_m.gguf") is None


def test_family_token_yi_does_not_match_tinyish_names():
    """``yi`` must not cross letter boundaries (``yip``)."""
    assert _detect_family_token("yip-7b.gguf") is None
    assert _detect_family_token("yi-vl-6b.gguf") == "yi"


def test_family_token_mimo_does_not_match_mimosa():
    """``mimo`` must not tag ``mimosa``."""
    assert _detect_family_token("mimosa-rosa-7b.gguf") is None
    assert _detect_family_token("MiMo-VL-7B-RL-BF16.gguf") == "mimo"


def test_family_token_mistral_does_not_match_ministral():
    """Pin Mistral-derivative tagging."""
    assert _detect_family_token("Ministral-3-8B-Instruct-2512-BF16.gguf") == "ministral"
    assert _detect_family_token("Mistral-7B-Instruct-v0.3.gguf") == "mistral"
    assert _detect_family_token("Magistral-Small-2506-BF16.gguf") == "magistral"
    assert (
        _detect_family_token("Devstral-Small-2-24B-Instruct-2512-BF16.gguf")
        == "devstral"
    )


def test_family_token_picks_leftmost_when_multiple_present():
    """Leftmost family token wins, not tuple order."""
    assert _detect_family_token("llama-phi-merge.gguf") == "llama"
    assert _detect_family_token("phi-llama-merge.gguf") == "phi"
    assert _detect_family_token("llama3-3b-instruct.gguf") == "llama"


def test_family_token_new_families_recognised():
    """Catalogue-audit additions tag correctly."""
    assert _detect_family_token("NVIDIA-Nemotron-3-Nano-Omni-30B.gguf") == "nemotron"
    assert _detect_family_token("Kimi-K2.6-BF16.gguf") == "kimi"
    assert _detect_family_token("Nanonets-OCR-s-BF16.gguf") == "nanonets"
    assert _detect_family_token("Cosmos-Reason1-7B-BF16.gguf") == "cosmos"
    assert _detect_family_token("Apriel-1.5-15b-Thinker-BF16.gguf") == "apriel"
    assert _detect_family_token("LFM2.5-VL-1.6B-BF16.gguf") == "lfm"


# -- Cross-family rejection with the expanded token list ----------------


def test_blocks_cross_family_for_new_token_pair(tmp_path: Path):
    """Nemotron weight + lone Gemma projector returns None."""
    model = _touch(
        tmp_path / "NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-MXFP4_MOE.gguf"
    )
    _touch(tmp_path / "gemma-4-26B-A4B-it.mmproj-q8_0.gguf")
    assert detect_mmproj_file(str(model)) is None


def test_picks_devstral_mmproj_in_mixed_dir(tmp_path: Path):
    """Devstral weight + Devstral mmproj + a Qwen mmproj: pick Devstral."""
    model = _touch(tmp_path / "Devstral-Small-2-24B-Instruct-2512-BF16.gguf")
    dev_mm = _touch(tmp_path / "Devstral-Small-2-mmproj-bf16.gguf")
    _touch(tmp_path / "Qwen3.5-9B-BF16-mmproj.gguf")
    assert detect_mmproj_file(str(model)) == str(dev_mm.resolve())


# -- Launcher-level family guard ----------------------------------------


def test_mmproj_family_guard_blocks_cross_family():
    assert (
        mmproj_matches_model_family(
            "/models/Qwen3.5-9B-Q4_K_M.gguf",
            "/models/gemma-4-26B-A4B-it.mmproj-q8_0.gguf",
        )
        is False
    )


def test_mmproj_family_guard_allows_same_family():
    assert (
        mmproj_matches_model_family(
            "/models/Qwen3.5-9B-Q4_K_M.gguf",
            "/models/Qwen3.5-9B-BF16-mmproj.gguf",
        )
        is True
    )


def test_mmproj_family_guard_allows_generic_hf_mmproj():
    """No family token on the projector: wildcard."""
    assert (
        mmproj_matches_model_family(
            "/models/Qwen3.5-9B-Q4_K_M.gguf",
            "/models/mmproj-F16.gguf",
        )
        is True
    )


def test_mmproj_family_guard_allows_unrecognised_model_family():
    """No family token on the model: wildcard."""
    assert (
        mmproj_matches_model_family(
            "/models/Apriel-1.5-15b-Thinker-BF16.gguf",
            "/models/mmproj-F16.gguf",
        )
        is True
    )


# -- Metadata-primary pairing in detect_mmproj_file ---------------------


def test_metadata_url_match_picked_over_filename_lookalike(tmp_path: Path):
    """URL match beats a longer-prefix sibling."""
    weight = _gguf_with_general(
        tmp_path / "Qwen3.5-9B-Q4_K_M.gguf",
        {
            "general.architecture": "qwen2vl",
            "general.type": "model",
            "general.basename": "Qwen3.5",
            "general.base_model.0.repo_url": "https://huggingface.co/Qwen/Qwen3.5-9B",
        },
    )
    # Closer filename prefix, wrong upstream.
    _gguf_with_general(
        tmp_path / "Qwen3.5-9B-mmproj-bf16.gguf",
        {
            "general.architecture": "clip",
            "general.type": "mmproj",
            "general.basename": "Qwen3.5",
            "general.base_model.0.repo_url": "https://huggingface.co/Qwen/Qwen3.5-1.5B",
        },
    )
    # Matching upstream.
    correct = _gguf_with_general(
        tmp_path / "mmproj-BF16.gguf",
        {
            "general.architecture": "clip",
            "general.type": "mmproj",
            "general.basename": "Qwen3.5",
            "general.base_model.0.repo_url": "https://huggingface.co/Qwen/Qwen3.5-9B",
        },
    )
    assert detect_mmproj_file(str(weight)) == str(correct.resolve())


def test_metadata_url_mismatch_dropped(tmp_path: Path):
    """Filenames match family but metadata disagrees: returns None."""
    weight = _gguf_with_general(
        tmp_path / "qwen-9b.gguf",
        {
            "general.architecture": "qwen2vl",
            "general.type": "model",
            "general.base_model.0.repo_url": "https://huggingface.co/Qwen/Qwen3.5-9B",
        },
    )
    _gguf_with_general(
        tmp_path / "qwen-9b-mmproj.gguf",
        {
            "general.architecture": "clip",
            "general.type": "mmproj",
            "general.base_model.0.repo_url": "https://huggingface.co/google/gemma-3-9B",
        },
    )
    assert detect_mmproj_file(str(weight)) is None


def test_metadata_identifies_mmproj_without_filename_hint(tmp_path: Path):
    """Projector named ``vision-projector.gguf`` discovered via header."""
    weight = _gguf_with_general(
        tmp_path / "Qwen3.5-9B.gguf",
        {
            "general.architecture": "qwen2vl",
            "general.type": "model",
            "general.basename": "Qwen3.5",
            "general.base_model.0.repo_url": "https://huggingface.co/Qwen/Qwen3.5-9B",
        },
    )
    projector = _gguf_with_general(
        tmp_path / "vision-projector.gguf",
        {
            "general.architecture": "clip",
            "general.type": "mmproj",
            "general.basename": "Qwen3.5",
            "general.base_model.0.repo_url": "https://huggingface.co/Qwen/Qwen3.5-9B",
        },
    )
    assert detect_mmproj_file(str(weight)) == str(projector.resolve())


def test_metadata_score_outranks_filename_prefix(tmp_path: Path):
    """Score 100 (URL match) beats score 0 (long filename prefix)."""
    weight = _gguf_with_general(
        tmp_path / "Qwen3.5-9B-Q4_K_M.gguf",
        {
            "general.architecture": "qwen2vl",
            "general.type": "model",
            "general.basename": "Qwen3.5",
            "general.base_model.0.repo_url": "https://huggingface.co/Qwen/Qwen3.5-9B",
        },
    )
    # Headerless: long shared stem, score 0.
    _touch(tmp_path / "Qwen3.5-9B-Q4_K_M-mmproj.gguf")
    # Headered: generic name, score 100.
    correct = _gguf_with_general(
        tmp_path / "mmproj-BF16.gguf",
        {
            "general.architecture": "clip",
            "general.type": "mmproj",
            "general.base_model.0.repo_url": "https://huggingface.co/Qwen/Qwen3.5-9B",
        },
    )
    assert detect_mmproj_file(str(weight)) == str(correct.resolve())
