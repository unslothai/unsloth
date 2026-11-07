# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""Per-architecture NF4 GEMV row cap and the T4 handoff to bitsandbytes gemm_4bit."""

import pytest
import torch

U = pytest.importorskip("unsloth.kernels.utils")


def _fresh(monkeypatch, arch):
    monkeypatch.setattr(U, "_GEMV_ROWS_CAP", {})
    monkeypatch.setattr(U, "_GEMV_ROWS_BNB_FROM", {})
    monkeypatch.setattr(U, "_GEMV_MAX_ROWS", 4)
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: arch)


@pytest.mark.parametrize(
    "arch, cap, bnb_from",
    [
        ((7, 5), 4, 4),
        ((8, 0), 2, 3),
        ((8, 9), 4, 5),
        ((12, 0), 4, 5),
        ((9, 0), 1, 2),
        ((10, 0), 1, 2),
    ],
)
def test_gemv_rows_cap_per_architecture(arch, cap, bnb_from, monkeypatch):
    _fresh(monkeypatch, arch)
    monkeypatch.delenv("UNSLOTH_GEMV_MAX_ROWS", raising = False)
    assert U._gemv_rows_cap(torch.device("cuda", 0)) == cap
    assert U._GEMV_ROWS_BNB_FROM[0] == bnb_from


def test_gemv_rows_env_override_keeps_t4_batch_4_on_the_gemv(monkeypatch):
    _fresh(monkeypatch, (7, 5))
    monkeypatch.setenv("UNSLOTH_GEMV_MAX_ROWS", "4")
    assert U._gemv_rows_cap(torch.device("cuda", 0)) == 4
    assert U._GEMV_ROWS_BNB_FROM[0] == 5


def test_t4_batch_4_steps_aside_when_the_bnb_gemm_applies(monkeypatch):
    _fresh(monkeypatch, (7, 5))
    monkeypatch.delenv("UNSLOTH_GEMV_MAX_ROWS", raising = False)
    asked = []
    monkeypatch.setattr(U, "_bnb_gemm_4bit_ok", lambda X, W, q: asked.append(X.shape[0]) or True)
    monkeypatch.setattr(U, "_GEMV_ROWS_CAP", {None: 4})
    monkeypatch.setattr(U, "_GEMV_ROWS_BNB_FROM", {None: 4})
    X = torch.randn(4, 1, 64, dtype = torch.float16)
    assert U._gemv_rows(X, torch.empty(1, dtype = torch.uint8), object(), None) is None
    assert asked == [4]
