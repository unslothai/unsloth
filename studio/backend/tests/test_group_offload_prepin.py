# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Chunked pinning ahead of a diffusers stream group offload, on a real card."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a CUDA device")


def _blocks():
    torch.manual_seed(0)

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList(
                torch.nn.Sequential(torch.nn.Linear(256, 384), torch.nn.GELU(), torch.nn.Linear(384, 256))
                for _ in range(4)
            )
            self.register_buffer("gain", torch.full((256,), 0.5))

        def forward(self, x):
            for block in self.blocks:
                x = x + block(x)
            return x * self.gain

    return Net().to(torch.bfloat16)


def _chunk_bases(tensors):
    return {t.untyped_storage().data_ptr() for t in tensors}


def test_packed_weights_are_pinned_in_shared_chunks_with_values_unchanged(monkeypatch):
    import core.inference.diffusion_memory as mem

    monkeypatch.setenv(mem.OFFLOAD_PIN_ENV, "1")
    net = _blocks()
    before = {n: p.detach().clone() for n, p in net.named_parameters()}
    pinned = mem._pin_streamed_weights(net)
    assert pinned > 0
    params = list(net.parameters())
    assert all(p.is_pinned() for p in params)
    assert net.gain.is_pinned()
    # one chunk, not one allocation per tensor
    assert len(_chunk_bases(params + [net.gain])) == 1
    for n, p in net.named_parameters():
        assert torch.equal(p.detach(), before[n]), n


def test_the_stream_apply_reuses_the_packed_chunks_and_renders_the_same(monkeypatch):
    import core.inference.diffusion_memory as mem
    from diffusers.hooks import apply_group_offloading

    monkeypatch.setenv(mem.OFFLOAD_PIN_ENV, "1")
    x = torch.randn(8, 256, dtype = torch.bfloat16, device = "cuda")
    ref_net = _blocks()
    apply_group_offloading(
        ref_net, onload_device = torch.device("cuda"), offload_device = torch.device("cpu"),
        offload_type = "block_level", num_blocks_per_group = 1, use_stream = True,
    )
    with torch.no_grad():
        ref = ref_net(x)
    net = _blocks()
    mem._pin_streamed_weights(net)
    packed = {id(p): p.data.data_ptr() for p in net.parameters()}
    apply_group_offloading(
        net, onload_device = torch.device("cuda"), offload_device = torch.device("cpu"),
        offload_type = "block_level", num_blocks_per_group = 1, use_stream = True,
    )
    with torch.no_grad():
        out = net(x)
        again = net(x)
    torch.cuda.synchronize()
    assert torch.equal(out, ref) and torch.equal(again, ref)
    # after an offload every parameter points back at its packed chunk: diffusers made no second host copy
    assert all(p.device.type == "cpu" and p.data.data_ptr() == packed[id(p)] for p in net.parameters())
