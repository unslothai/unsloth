# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for vision-projector (mmproj) VRAM accounting in GPU fit (#5825).

When a vision-capable GGUF is launched, llama-server loads the mmproj
(vision projector) onto the GPU alongside the weights via ``--mmproj``.
Studio's context auto-sizing and GPU selection size their VRAM budget off
``_get_gguf_size_bytes(model_path)``, which counts only the weight file(s).
The projector was never added, so the budget was too optimistic: context
got mis-estimated and tightly-fitting loads spilled to system RAM / OOM'd.

``LlamaCppBackend._mmproj_vram_bytes`` returns the extra VRAM a resolved
projector contributes (0 when none is resolved or the file is unreadable),
so ``load_model`` can fold it into the fit budget. The projector is resolved
once (``_resolve_launch_mmproj_path``) and the same path feeds both the fit
budget and the actual ``--mmproj`` launch flag, so the two cannot disagree.

Requires no GPU, network, or subprocess. Cross-platform.
"""

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
    """A resolved projector path is counted at its full on-disk size."""
    mmproj = _write(tmp_path / "Qwen3.5-9B-BF16-mmproj.gguf", 1024)

    got = _backend()._mmproj_vram_bytes(str(mmproj))

    assert got == 1024


def test_zero_when_no_projector_resolved(tmp_path: Path):
    """None (no projector will launch) contributes nothing to the budget."""
    assert _backend()._mmproj_vram_bytes(None) == 0


def test_zero_when_projector_missing_on_disk(tmp_path: Path):
    """An unreadable / vanished path is swallowed, not raised, and counts 0."""
    missing = tmp_path / "Qwen3.5-9B-BF16-mmproj.gguf"  # never created

    got = _backend()._mmproj_vram_bytes(str(missing))

    assert got == 0
