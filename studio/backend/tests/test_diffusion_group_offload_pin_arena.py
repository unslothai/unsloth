# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Image / encoder group-offload tiers pin up front through the slab arena, and the pin plan prices the slabs."""

import inspect

import pytest
import torch

from core.inference import diffusion_memory as dm
from core.inference import diffusion_pinned_arena as arena_mod


@pytest.fixture
def unpinned_slabs(monkeypatch):
    requests = []

    def alloc(nbytes):
        requests.append(nbytes)
        return torch.empty(nbytes, dtype = torch.uint8)

    monkeypatch.setattr(arena_mod, "_alloc_slab", alloc)
    return requests


def _weights():
    # Flux-like shapes: per-tensor pinning rounds 3072x12288 bf16 (72 MiB) up to 128 MiB
    return [
        torch.zeros(3072, 12288, dtype = torch.bfloat16),
        torch.zeros(12288, 3072, dtype = torch.bfloat16),
        torch.zeros(3072, dtype = torch.bfloat16),
        torch.zeros(9216, 3072, dtype = torch.bfloat16),
        torch.zeros(7, dtype = torch.float32),
    ]


def test_estimate_matches_the_arena_it_replays(unpinned_slabs):
    weights = _weights() * 3
    payload = sum(w.numel() * w.element_size() for w in weights)
    assert arena_mod.packable_bytes(weights) == payload
    arena = arena_mod.PinnedArena(slab_bytes = 1 << 28, expected_bytes = payload)
    for w in weights:
        arena.pin(w)
    assert arena_mod.arena_reserved_bytes(weights, slab_bytes = 1 << 28) == arena.reserved_bytes
    per_tensor = sum(arena_mod._pow2_ceil(w.numel() * w.element_size()) for w in weights)
    assert arena.reserved_bytes < per_tensor
    assert arena.reserved_bytes < payload * 1.15
    assert all(n & (n - 1) == 0 for n in unpinned_slabs)  # the host allocator rounds no slab


def test_a_small_module_does_not_reserve_a_whole_slab(unpinned_slabs):
    weights = [torch.zeros(1000, 1000) for _ in range(9)]  # 36 MB, per-tensor 4 MiB each
    payload = arena_mod.packable_bytes(weights)
    arena = arena_mod.PinnedArena(expected_bytes = payload)
    for w in weights:
        arena.pin(w)
    assert payload <= arena.reserved_bytes <= payload * 1.25
    assert arena.reserved_bytes == arena_mod.arena_reserved_bytes(weights)
    # without a size hint the whole first slab is taken, as before
    assert (
        arena_mod.PinnedArena()._take(16).untyped_storage().nbytes() == arena_mod.DEFAULT_SLAB_BYTES
    )


def test_estimate_prices_unpackable_tensors_per_tensor():
    t = torch.zeros(300, 500).t()  # not contiguous: the arena hands it to pin_memory()
    assert arena_mod.arena_reserved_bytes([t]) == arena_mod._pow2_ceil(t.numel() * 4)
    assert arena_mod.arena_reserved_bytes([torch.zeros(0)]) == 0


def test_device_resident_tensors_are_priced_per_tensor():
    # _to_cpu hands a tensor that is not on the CPU (a resident denoiser) to the stock per-tensor pin
    t = torch.empty(300, 500, device = "meta")
    assert arena_mod.packable_bytes([t]) == 0
    assert arena_mod.arena_reserved_bytes([t]) == arena_mod._pow2_ceil(t.numel() * 4)


def test_module_pin_mib_follows_the_kill_switch(monkeypatch):
    net = torch.nn.Sequential(*(torch.nn.Linear(3072, 3000, bias = False) for _ in range(8)))
    per_tensor = dm._module_host_mib(net)
    packed = dm._module_pin_mib(net)
    payload_mib = sum(p.numel() * p.element_size() for p in net.parameters()) / 2**20
    assert payload_mib <= packed < per_tensor
    monkeypatch.setenv(arena_mod.PIN_ARENA_ENV, "0")
    assert dm._module_pin_mib(net) == per_tensor


def _recording_apply(calls):
    from diffusers.hooks import group_offloading as go
    def apply(module, **kwargs):
        calls.append(kwargs)
        module._copies = {
            p: go.ModuleGroup._to_cpu(p, kwargs.get("low_cpu_mem_usage", False))
            for p in module.parameters()
        }

    return apply


@pytest.mark.parametrize(
    "kwargs, packed",
    [
        ({"use_stream": True, "low_cpu_mem_usage": False}, True),
        ({"use_stream": True}, True),  # diffusers' default pins up front
        ({"use_stream": True, "low_cpu_mem_usage": True}, False),
        ({"use_stream": False, "low_cpu_mem_usage": False}, False),
    ],
)
def test_packed_apply_packs_only_an_up_front_pin(monkeypatch, unpinned_slabs, kwargs, packed):
    pytest.importorskip("diffusers.hooks.group_offloading")
    from diffusers.hooks import group_offloading as go

    stock = go.ModuleGroup.__dict__["_to_cpu"]
    monkeypatch.setattr(dm, "host_pinning_allowed", lambda *a, **k: True)
    if not packed:
        # an unpacked apply must reach the stock host copy; stub its pin so no CUDA is needed
        monkeypatch.setattr(torch.Tensor, "pin_memory", lambda self: self.clone())
    calls = []
    net = torch.nn.Linear(64, 32)
    dm._apply_group_offloading_packed(_recording_apply(calls), net, dict(kwargs))
    assert calls == [kwargs]
    assert go.ModuleGroup.__dict__["_to_cpu"] is stock
    if packed:
        payload = sum(p.numel() * p.element_size() for p in net.parameters())
        assert net._unsloth_pin_arena_bytes[0] == payload
        for p, copy in net._copies.items():
            assert p.data_ptr() == copy.data_ptr()  # re-pointed at the slab view
    else:
        assert not hasattr(net, "_unsloth_pin_arena_bytes")
        assert not unpinned_slabs


def test_packed_apply_respects_integrated_gpus(monkeypatch, unpinned_slabs):
    pytest.importorskip("diffusers.hooks.group_offloading")
    monkeypatch.setattr(dm, "host_pinning_allowed", lambda *a, **k: False)
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda self: self.clone())
    calls = []
    net = torch.nn.Linear(16, 16)
    dm._apply_group_offloading_packed(
        _recording_apply(calls), net, {"use_stream": True, "low_cpu_mem_usage": False}
    )
    assert not unpinned_slabs
    assert not hasattr(net, "_unsloth_pin_arena_bytes")


def test_every_up_front_group_offload_apply_goes_through_the_arena():
    """The tiers that pin a whole module up front call the packed apply, and size the plan from the slabs."""
    sources = {
        name: inspect.getsource(getattr(dm, name))
        for name in (
            "_apply_group_offload",
            "_apply_streaming_offload",
            "stream_text_encoders_onto",
        )
    }
    for name, src in sources.items():
        assert "_apply_group_offloading_packed(" in src, name
        assert "_streamed_pin_plan(" in src, name
        plan = src[src.index("_streamed_pin_plan(") :][:400]
        assert "_module_host_mib(m)" not in plan, name
    from core.inference import diffusion as d

    assert "_apply_group_offloading_packed(" in inspect.getsource(d._offload_controlnet_module)
