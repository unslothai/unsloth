# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""A fine-grained FP8 checkpoint loads on ROCm (Unsloth's FP8 kernels route per device); FBGEMM FP8 does not."""

import types

import pytest

U = pytest.importorskip("unsloth.models._utils")


def _config(method):
    return types.SimpleNamespace(quantization_config = {"quant_method": method})


def test_rocm_accepts_a_fine_grained_fp8_checkpoint(monkeypatch):
    monkeypatch.setattr(U, "DEVICE_TYPE", "hip")
    U.verify_fp8_support_if_applicable(_config("fp8"))
    U.verify_fp8_support_if_applicable(_config("fp8"), fast_inference = True)


def test_rocm_refuses_fbgemm_fp8(monkeypatch):
    monkeypatch.setattr(U, "DEVICE_TYPE", "hip")
    with pytest.raises(ValueError, match = "FBGEMM"):
        U.verify_fp8_support_if_applicable(_config("fbgemm_fp8"))


@pytest.mark.parametrize("device", ["mps", "cpu", "npu"])
def test_other_devices_still_refuse_fp8(monkeypatch, device):
    monkeypatch.setattr(U, "DEVICE_TYPE", device)
    for method in ("fp8", "fbgemm_fp8"):
        with pytest.raises(ValueError, match = "only supported on"):
            U.verify_fp8_support_if_applicable(_config(method))


def test_non_fp8_checkpoints_pass_everywhere(monkeypatch):
    monkeypatch.setattr(U, "DEVICE_TYPE", "hip")
    U.verify_fp8_support_if_applicable(_config("bitsandbytes"))
    U.verify_fp8_support_if_applicable(types.SimpleNamespace())
