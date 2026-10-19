# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Background pinning of diffusers stream group offload, on a real card."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a CUDA device")


class _Pipe:
    pass


def _net():
    torch.manual_seed(0)

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList(
                torch.nn.Sequential(
                    torch.nn.Linear(256, 384), torch.nn.GELU(), torch.nn.Linear(384, 256)
                )
                for _ in range(6)
            )
            self.register_buffer("gain", torch.full((256,), 0.5))

        def forward(self, x):
            for block in self.blocks:
                x = x + block(x)
            return x * self.gain

    return Net().to(torch.bfloat16)


def _offload(net, pinned: bool):
    from diffusers.hooks import apply_group_offloading
    apply_group_offloading(
        net,
        onload_device = torch.device("cuda"),
        offload_device = torch.device("cpu"),
        offload_type = "block_level",
        num_blocks_per_group = 1,
        use_stream = True,
        low_cpu_mem_usage = not pinned,
    )
    return net


def _reference(x):
    with torch.no_grad():
        return _offload(_net(), pinned = True)(x)


def _groups_pinned(net):
    import core.inference.diffusion_memory as mem
    groups = mem._offload_groups(net)
    return groups, all(t.is_pinned() for g in groups for t in g.cpu_param_dict.values())


def test_a_deferred_pin_renders_identically_and_ends_with_every_group_pinned():
    import core.inference.diffusion_memory as mem

    mem.install_group_pin_wait()
    x = torch.randn(8, 256, dtype = torch.bfloat16, device = "cuda")
    ref = _reference(x)
    pipe, net = _Pipe(), _offload(_net(), pinned = False)
    before = {id(p): p.detach().clone() for p in net.parameters()}
    assert mem._defer_pinning(pipe, net, torch.device("cuda"), None)
    groups, pinned = _groups_pinned(net)
    assert groups and not pinned
    assert mem.start_background_pins(pipe) == 1
    with torch.no_grad():
        out = net(x)
        again = net(x)
    torch.cuda.synchronize()
    mem.stop_background_pins(pipe)
    assert torch.equal(out, ref) and torch.equal(again, ref)
    groups, pinned = _groups_pinned(net)
    assert pinned
    # the host copies were replaced, not duplicated: every offloaded parameter now IS its pinned chunk view
    for g in groups:
        for t, host in g.cpu_param_dict.items():
            if isinstance(t, torch.nn.Parameter):
                assert t.device.type == "cpu" and t.data_ptr() == host.data_ptr()
    for p in net.parameters():
        assert torch.equal(p.detach(), before[id(p)])


def test_an_onload_before_the_pinner_starts_waits_for_its_group():
    import core.inference.diffusion_memory as mem

    mem.install_group_pin_wait()
    x = torch.randn(8, 256, dtype = torch.bfloat16, device = "cuda")
    ref = _reference(x)
    pipe, net = _Pipe(), _offload(_net(), pinned = False)
    mem._defer_pinning(pipe, net, torch.device("cuda"), None)
    # never started by the caller: the first onload starts it and waits for its own group
    with torch.no_grad():
        out = net(x)
    torch.cuda.synchronize()
    mem.stop_background_pins(pipe)
    assert torch.equal(out, ref)
    assert _groups_pinned(net)[1]


def test_a_stopped_pinner_leaves_the_rest_to_diffusers_and_never_blocks():
    import core.inference.diffusion_memory as mem

    mem.install_group_pin_wait()
    x = torch.randn(8, 256, dtype = torch.bfloat16, device = "cuda")
    ref = _reference(x)
    pipe, net = _Pipe(), _offload(_net(), pinned = False)
    mem._defer_pinning(pipe, net, torch.device("cuda"), None)
    mem.stop_background_pins(pipe)
    with torch.no_grad():
        out = net(x)
    torch.cuda.synchronize()
    assert torch.equal(out, ref)
    assert not _groups_pinned(net)[1]
