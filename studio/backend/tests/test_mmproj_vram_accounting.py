# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for mmproj VRAM accounting in GGUF fit budgeting (#5825)."""

from __future__ import annotations

from pathlib import Path

from core.inference.llama_cpp import LlamaCppBackend


def _write(path: Path, n_bytes: int) -> Path:
    path.parent.mkdir(parents = True, exist_ok = True)
    path.write_bytes(b"\x00" * n_bytes)
    return path


def _backend() -> LlamaCppBackend:
    return LlamaCppBackend.__new__(LlamaCppBackend)


def test_counts_resolved_projector_size(tmp_path: Path):
    mmproj = _write(tmp_path / "Qwen3.5-9B-BF16-mmproj.gguf", 1024)

    got = _backend()._mmproj_vram_bytes(str(mmproj))

    assert got == 1024


def test_zero_when_no_projector_resolved(tmp_path: Path):
    assert _backend()._mmproj_vram_bytes(None) == 0


def test_zero_when_projector_missing_on_disk(tmp_path: Path):
    missing = tmp_path / "Qwen3.5-9B-BF16-mmproj.gguf"  # never created

    got = _backend()._mmproj_vram_bytes(str(missing))

    assert got == 0
