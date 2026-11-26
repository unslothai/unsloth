# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The multi-GPU module states what it guarantees: one computer, bit-identical for the guidance split only."""

from __future__ import annotations

import ast
from pathlib import Path

_DOC = ast.get_docstring(
    ast.parse(
        (Path(__file__).resolve().parents[1] / "core/inference/diffusion_multigpu.py").read_text(
            encoding = "utf-8"
        )
    )
)


def test_bit_identity_is_scoped_to_the_guidance_split():
    assert "outputs bit-identical;" not in _DOC
    assert "guidance (CFG) split keep outputs bit-identical" in " ".join(_DOC.split())
    assert "batch split can differ" in " ".join(_DOC.split())


def test_multi_gpu_means_one_computer():
    flat = " ".join(_DOC.split())
    assert "ONE computer" in flat and "no multi-node" in flat
