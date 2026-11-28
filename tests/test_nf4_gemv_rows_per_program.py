# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""CPU check of how many decode rows one NF4 GEMV program takes: up to 4 rows share a program everywhere; above 4,
CUDA groups by the largest of 4, 3, 2 dividing M, ROCm runs one row per program (gfx1151 got 3-row groups wrong)."""

import pytest

pytest.importorskip("triton")
nf4_gemv = pytest.importorskip("unsloth.kernels.nf4_gemv")


@pytest.mark.parametrize("rocm", [False, True], ids = ["cuda", "rocm"])
def test_rows_per_program(rocm, monkeypatch):
    monkeypatch.setattr(nf4_gemv, "_ROCM", rocm)
    for M in (1, 2, 3, 4):
        assert nf4_gemv._rows_per_program(M) == M
    want = (
        {5: 1, 6: 3, 8: 4, 9: 3, 10: 2, 12: 4} if not rocm else {M: 1 for M in (5, 6, 8, 9, 10, 12)}
    )
    for M, rows in want.items():
        assert nf4_gemv._rows_per_program(M) == rows, M
        assert M % rows == 0


def test_cuda_rows_unchanged(monkeypatch):
    # The expression _launch used before this route existed, for every row count a row cap can reach.
    monkeypatch.setattr(nf4_gemv, "_ROCM", False)
    for M in range(0, 129):
        old = max(M, 1) if M <= 4 else next(r for r in (4, 3, 2, 1) if M % r == 0)
        assert nf4_gemv._rows_per_program(M) == old, M
