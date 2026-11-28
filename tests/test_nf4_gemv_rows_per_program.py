# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""CPU check of how many decode rows one NF4 GEMV program takes: up to 4 rows share a program everywhere; above 4,
CUDA groups by the largest of 4, 3, 2 dividing M, ROCm runs one row per program (gfx1151 got 3-row groups wrong)."""

import pytest
import torch

pytest.importorskip("triton")
nf4_gemv = pytest.importorskip("unsloth.kernels.nf4_gemv")


@pytest.mark.parametrize("hip", [None, "7.13.0"], ids = ["cuda", "rocm"])
def test_rows_per_program(hip, monkeypatch):
    monkeypatch.setattr(torch.version, "hip", hip)
    for M in (1, 2, 3, 4):
        assert nf4_gemv._rows_per_program(M) == M
    want = (
        {5: 1, 6: 3, 8: 4, 9: 3, 10: 2, 12: 4}
        if hip is None
        else {M: 1 for M in (5, 6, 8, 9, 10, 12)}
    )
    for M, rows in want.items():
        assert nf4_gemv._rows_per_program(M) == rows, M
        assert M % rows == 0
