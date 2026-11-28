# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""Cross entropy launches fit 1024 threads per block: 16 warps on wave64 AMD, 32 on NVIDIA and RDNA."""

import types

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("unsloth.kernels needs a CUDA or ROCm device", allow_module_level = True)

from unsloth.kernels import cross_entropy_loss as ce_mod
from unsloth.kernels import utils as utils_mod
from unsloth.kernels.utils import MAX_FUSED_SIZE


def _fake_target(
    monkeypatch,
    hip,
    warp_size = None,
):
    target = types.SimpleNamespace(arch = "gfx906", warp_size = warp_size)
    driver = types.SimpleNamespace(active = types.SimpleNamespace(get_current_target = lambda: target))
    monkeypatch.setattr(utils_mod, "is_hip", lambda: hip)
    monkeypatch.setattr(
        utils_mod, "triton", types.SimpleNamespace(runtime = types.SimpleNamespace(driver = driver))
    )


@pytest.mark.parametrize(
    "hip, warp_size, expected",
    [(False, None, 32), (True, 64, 16), (True, 32, 32), (True, None, 16)],
)
def test_max_num_warps_is_1024_threads_over_the_warp_size(hip, warp_size, expected, monkeypatch):
    _fake_target(monkeypatch, hip, warp_size)
    utils_mod.max_num_warps.cache_clear()
    try:
        assert utils_mod.max_num_warps() == expected
    finally:
        utils_mod.max_num_warps.cache_clear()


@pytest.mark.parametrize("warps", [16, 32])
@pytest.mark.parametrize("vocab", [65537, 128256, 262144])
def test_chunked_forward_on_hip_uses_the_warp_limit(warps, vocab, monkeypatch):
    monkeypatch.setattr(torch.version, "hip", "6.3")
    monkeypatch.setattr(ce_mod, "max_num_warps", lambda: warps)
    ce_mod._chunked_forward_config_at.cache_clear()
    try:
        assert ce_mod._chunked_forward_config_at("cuda", 0, vocab) == (MAX_FUSED_SIZE, warps)
    finally:
        ce_mod._chunked_forward_config_at.cache_clear()


# 40000: single block kernel (BLOCK_SIZE 65536, 32 warps before the cap); the others: chunked.
@pytest.mark.parametrize("vocab", [40000, 128256, 262144])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_loss_and_grads_match_torch(vocab, dtype):
    rows = 16
    generator = torch.Generator(device = "cuda").manual_seed(vocab)
    logits = (torch.randn(rows, vocab, device = "cuda", generator = generator) * 3).to(dtype)
    labels = torch.randint(0, vocab, (rows,), device = "cuda", generator = generator)
    labels[::5] = -100
    labels[1:3] = torch.tensor([0, vocab - 1], device = "cuda")
    logits.requires_grad_(True)
    loss = ce_mod.fast_cross_entropy_loss(logits.view(1, rows, vocab), labels.view(1, rows))
    loss.backward()

    reference = logits.detach().float().requires_grad_(True)
    expected = torch.nn.functional.cross_entropy(reference, labels, ignore_index = -100)
    expected.backward()
    torch.testing.assert_close(loss.detach().float(), expected.detach(), rtol = 1e-4, atol = 1e-4)
    torch.testing.assert_close(logits.grad.float(), reference.grad, rtol = 2e-2, atol = 2e-3)
