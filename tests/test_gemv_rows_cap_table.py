# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""Decode rows per architecture that take the Triton NF4 GEMV; T4 hands batch 4 to the bitsandbytes gemm."""

import pytest
import torch

from unsloth.kernels import utils as U


@pytest.mark.parametrize(
    "arch, cap",
    [((7, 5), 3), ((8, 0), 2), ((8, 9), 4), ((12, 0), 4), ((9, 0), 1), ((10, 0), 1)],
)
def test_gemv_rows_cap_per_architecture(arch, cap, monkeypatch):
    monkeypatch.setattr(U, "_GEMV_ROWS_CAP", {})
    monkeypatch.setattr(U, "_GEMV_MAX_ROWS", 4)
    monkeypatch.delenv("UNSLOTH_GEMV_MAX_ROWS", raising = False)
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: arch)
    assert U._gemv_rows_cap(torch.device("cuda", 0)) == cap


def test_gemv_rows_env_override_still_wins_on_t4(monkeypatch):
    monkeypatch.setattr(U, "_GEMV_ROWS_CAP", {})
    monkeypatch.setattr(U, "_GEMV_MAX_ROWS", 4)
    monkeypatch.setenv("UNSLOTH_GEMV_MAX_ROWS", "4")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: (7, 5))
    assert U._gemv_rows_cap(torch.device("cuda", 0)) == 4
