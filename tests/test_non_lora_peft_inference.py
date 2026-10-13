# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""fast_linear_forward must run non-LoRA PEFT adapters (VeRA, LoHa, ...) through their own forward (#1385)."""

from __future__ import annotations

import pytest
import torch
import unsloth  # noqa: F401

from real_accelerator import has_real_cuda

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not has_real_cuda(), reason = "needs a real CUDA device"),
]

H = 64


class _Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = torch.nn.Linear(H, H, bias = False)


def _config(method):
    import peft

    if method == "lora":
        return peft.LoraConfig(
            r = 4, lora_alpha = 8, target_modules = ["q_proj"], init_lora_weights = False
        )
    if method == "vera":
        return peft.VeraConfig(r = 8, target_modules = ["q_proj"], init_weights = False)
    if method == "loha":
        return peft.LoHaConfig(r = 4, alpha = 8, target_modules = ["q_proj"], init_weights = False)
    if method == "lokr":
        return peft.LoKrConfig(r = 4, alpha = 8, target_modules = ["q_proj"], init_weights = False)
    if method == "ia3":
        return peft.IA3Config(
            target_modules = ["q_proj"], feedforward_modules = [], init_ia3_weights = False
        )
    if method == "adalora":
        return peft.AdaLoraConfig(init_r = 4, target_r = 2, total_step = 10, target_modules = ["q_proj"])
    raise ValueError(method)


def _q_proj(method):
    from peft import get_peft_model

    torch.manual_seed(3407)
    model = get_peft_model(_Block(), _config(method)).to("cuda", torch.bfloat16)
    proj = model.base_model.model.q_proj
    if method == "adalora":
        with torch.no_grad():
            # lora_E starts at zero, so the adapter would be a no-op
            proj.lora_E["default"].normal_(std = 10.0)
    return proj


@pytest.mark.parametrize("method", ["vera", "loha", "lokr", "ia3", "adalora"])
@pytest.mark.parametrize("q_len", [1, 5])
def test_fast_linear_forward_runs_non_lora_adapter(method, q_len):
    from unsloth.kernels import fast_linear_forward
    from unsloth.kernels.utils import _has_non_lora_adapter

    proj = _q_proj(method)
    assert _has_non_lora_adapter(proj)
    X = torch.randn(1, q_len, H, device = "cuda", dtype = torch.bfloat16)
    with torch.no_grad():
        expected = proj(X)
        # a no-op adapter would make this test vacuous
        assert not torch.allclose(expected, proj.base_layer(X), rtol = 2e-2, atol = 2e-2)
        out = torch.empty(1, q_len, H, device = "cuda", dtype = torch.bfloat16)
        got = fast_linear_forward(proj, X, out = out)
        torch.testing.assert_close(got, expected, rtol = 2e-2, atol = 2e-2)
        assert got.data_ptr() == out.data_ptr()


def test_lora_and_plain_linear_keep_the_fast_path():
    from unsloth.kernels.utils import _has_non_lora_adapter
    assert not _has_non_lora_adapter(_q_proj("lora"))
    assert not _has_non_lora_adapter(torch.nn.Linear(H, H))
