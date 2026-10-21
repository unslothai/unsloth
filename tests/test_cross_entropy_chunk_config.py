# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""The chunked (vocab > 65536) cross entropy forward picks (BLOCK_SIZE, num_warps) per GPU arch:
16384 / 8 on T4 (sm_75) and A100 (sm_80), MAX_FUSED_SIZE / 32 everywhere else. The launch has to use
the routed values, and the loss and gradients must not depend on the chunk width."""

import pytest
import torch

if not torch.cuda.is_available() or torch.version.hip is not None:
    pytest.skip("needs an NVIDIA CUDA device", allow_module_level = True)

from unsloth.kernels import cross_entropy_loss as ce_mod
from unsloth.kernels.utils import MAX_FUSED_SIZE


def _route(capability, monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index = None: capability)
    ce_mod._chunked_forward_config_at.cache_clear()
    try:
        return ce_mod._chunked_forward_config(torch.device("cuda", 0))
    finally:
        ce_mod._chunked_forward_config_at.cache_clear()


@pytest.mark.parametrize("capability", [(7, 5), (8, 0)])
def test_t4_and_a100_use_16k_blocks_with_8_warps(capability, monkeypatch):
    assert _route(capability, monkeypatch) == (16384, 8)


@pytest.mark.parametrize(
    "capability", [(6, 1), (7, 0), (7, 2), (8, 6), (8, 7), (8, 9), (9, 0), (10, 0), (10, 3), (12, 0)]
)
def test_other_archs_keep_the_previous_config(capability, monkeypatch):
    assert _route(capability, monkeypatch) == (MAX_FUSED_SIZE, 32)


def test_device_without_index_uses_the_current_device(monkeypatch):
    seen = []

    def capability(index = None):
        seen.append(index)
        return (7, 5)

    monkeypatch.setattr(torch.cuda, "get_device_capability", capability)
    ce_mod._chunked_forward_config_at.cache_clear()
    try:
        assert ce_mod._chunked_forward_config(torch.device("cuda")) == (16384, 8)
    finally:
        ce_mod._chunked_forward_config_at.cache_clear()
    assert seen == [torch.cuda.current_device()]


def test_non_cuda_devices_keep_the_previous_config():
    assert ce_mod._chunked_forward_config_at("xpu", 0) == (MAX_FUSED_SIZE, 32)


def _launch(vocab, config, monkeypatch, rows = 64, softcap = 0.0, scale = 0.0):
    launches = []
    kernel = ce_mod._chunked_cross_entropy_forward

    class Spy:
        def __getitem__(self, grid):
            def run(*args, **kwargs):
                launches.append(
                    (tuple(grid), kwargs["BLOCK_SIZE"], kwargs["N_CHUNKS"], kwargs["num_warps"])
                )
                return kernel[grid](*args, **kwargs)

            return run

    monkeypatch.setattr(ce_mod, "_chunked_cross_entropy_forward", Spy())
    monkeypatch.setattr(ce_mod, "_chunked_forward_config", lambda device: config)
    generator = torch.Generator(device = "cuda").manual_seed(vocab)
    logits = (torch.randn(rows, vocab, device = "cuda", generator = generator) * 3).to(torch.bfloat16)
    labels = torch.randint(0, vocab, (rows,), device = "cuda", generator = generator)
    labels[::5] = -100
    labels[1:4] = torch.tensor([0, vocab - 1, 16383], device = "cuda")
    logits.requires_grad_(True)
    loss = ce_mod.fast_cross_entropy_loss(
        logits.view(1, rows, vocab), labels.view(1, rows), logit_softcapping = softcap, logit_scaling = scale
    )
    loss.backward()
    return launches, loss.detach(), logits.grad.detach()


@pytest.mark.parametrize("vocab", [128256, 151936, 262144])
def test_launch_uses_the_routed_config(vocab, monkeypatch):
    launches, _, _ = _launch(vocab, (16384, 8), monkeypatch)
    n_chunks = -(-vocab // 16384)
    assert launches == [((64, n_chunks), 16384, n_chunks, 8)]


@pytest.mark.parametrize(
    "vocab, softcap, scale",
    [
        (128256, 0.0, 0.0),
        (151936, 0.0, 0.0),
        (262144, 0.0, 0.0),
        (256000, 30.0, 0.0),
        (256000, 0.0, 0.0625),
    ],
)
def test_loss_and_grads_match_across_chunk_widths(vocab, softcap, scale, monkeypatch):
    _, loss_old, grad_old = _launch(vocab, (MAX_FUSED_SIZE, 32), monkeypatch, softcap = softcap, scale = scale)
    _, loss_new, grad_new = _launch(vocab, (16384, 8), monkeypatch, softcap = softcap, scale = scale)
    torch.testing.assert_close(loss_new, loss_old, rtol = 1e-5, atol = 1e-5)
    torch.testing.assert_close(grad_new, grad_old, rtol = 2e-2, atol = 2e-3)


def test_vocab_at_most_65536_stays_on_the_single_block_kernel(monkeypatch):
    launches, _, _ = _launch(32768, (16384, 8), monkeypatch)
    assert launches == []
