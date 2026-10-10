# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""MiniMax-H3 host-RAM guard and the streamed denoiser's host footprint. CPU only, no weights.

The reported failure: on a 94 GB host the Diffusers INT8 path rendered once and then refused the second
render ("needs about 85 GB available system RAM ... 82.5 GB is available"). Measured on a 48 GB tier:
the streamed int8 denoiser's 19.45 GB of weights held 33.8 GB of pinned host memory (torch's host
allocator rounds every pin up to a power of two), and the guard priced the denoiser twice."""

import types

import pytest
import torch

from core.inference import video_minimax_h3 as vmh3


def _arena_mod():
    # Imported per test so each one reports on a tree that lacks the module, rather than one collection error.
    from core.inference import diffusion_pinned_arena
    return diffusion_pinned_arena


@pytest.fixture
def unpinned_slabs(monkeypatch):
    """Slabs from pageable memory (no CUDA here); records every slab request."""
    arena_mod = _arena_mod()
    requests = []

    def alloc(nbytes):
        requests.append(nbytes)
        return torch.empty(nbytes, dtype = torch.uint8)

    monkeypatch.setattr(arena_mod, "_alloc_slab", alloc)
    return requests


def _storage_ptrs(tensors):
    return {t.untyped_storage().data_ptr() for t in tensors}


def test_arena_packs_weights_without_power_of_two_rounding(unpinned_slabs):
    arena_mod = _arena_mod()
    arena = arena_mod.PinnedArena(slab_bytes = 1 << 20)
    # 200 tensors of ~30 KB: per-tensor pinning would round each to 32 KiB.
    weights = [torch.randn(7_500 + i) for i in range(200)]
    pinned = [arena.pin(w) for w in weights]
    payload = sum(w.numel() * w.element_size() for w in weights)
    per_tensor = sum(arena_mod._pow2_ceil(w.numel() * w.element_size()) for w in weights)
    assert arena.payload_bytes == payload
    assert all(n & (n - 1) == 0 for n in unpinned_slabs)  # every slab request is a power of two
    assert arena.reserved_bytes <= payload * 1.05 + (1 << 20)
    assert arena.reserved_bytes < per_tensor
    assert len(_storage_ptrs(pinned)) == len(unpinned_slabs) < len(weights)
    for w, p in zip(weights, pinned):
        assert p.dtype == w.dtype and p.shape == w.shape and torch.equal(p, w)


def test_arena_rebuilds_a_torchao_weight_from_its_inner_tensors(unpinned_slabs):
    arena_mod = _arena_mod()
    pytest.importorskip("torchao")
    try:
        from torchao.quantization.granularity import PerRow
        from torchao.quantization.quantize_.workflows import Int8Tensor
    except ImportError:
        pytest.skip("this torchao has no Int8Tensor")
    src = Int8Tensor.from_hp(torch.randn(64, 32, dtype = torch.bfloat16), granularity = PerRow())
    arena = arena_mod.PinnedArena(slab_bytes = 1 << 20)
    out = arena.pin(src)
    assert type(out) is type(src)
    names, _ = src.__tensor_flatten__()
    for name in names:
        assert torch.equal(getattr(out, name), getattr(src, name))
    assert _storage_ptrs([getattr(out, n) for n in names]) == _storage_ptrs(
        [s for s, _ in arena._slabs]
    )


def test_group_offload_pins_through_the_arena_and_drops_the_pageable_source(unpinned_slabs):
    arena_mod = _arena_mod()
    go = pytest.importorskip("diffusers.hooks.group_offloading")
    stock = go.ModuleGroup.__dict__["_to_cpu"]
    lin = torch.nn.Linear(256, 256)
    source_ptr = lin.weight.data_ptr()
    with arena_mod.pinned_arena_for_group_offload(enabled = True) as arena:
        assert arena is not None
        copy = go.ModuleGroup._to_cpu(lin.weight, False)
    # The parameter now reads the single copy: the pageable source is no longer referenced by it.
    assert lin.weight.data_ptr() == copy.data_ptr() != source_ptr
    assert arena.payload_bytes == 256 * 256 * 4
    assert go.ModuleGroup.__dict__["_to_cpu"] is stock


def test_pin_arena_kill_switch_leaves_group_offload_untouched(monkeypatch):
    arena_mod = _arena_mod()
    go = pytest.importorskip("diffusers.hooks.group_offloading")
    stock = go.ModuleGroup.__dict__["_to_cpu"]
    monkeypatch.setenv(arena_mod.PIN_ARENA_ENV, "0")
    with arena_mod.pinned_arena_for_group_offload() as arena:
        assert arena is None
        assert go.ModuleGroup.__dict__["_to_cpu"] is stock


def test_streamed_prequant_denoiser_keeps_one_host_copy(monkeypatch, unpinned_slabs):
    pytest.importorskip("diffusers.hooks.group_offloading")
    import diffusers.hooks as hooks

    from core.inference import diffusion_memory as dm
    from core.inference import diffusion_prequant as dp

    monkeypatch.setattr(dp, "torchao_group_offload_supported", lambda: True)
    monkeypatch.setattr(dp, "_unhook_from_manager", lambda *a, **k: True)
    monkeypatch.setattr(dp, "_weights_pinnable", lambda m: True)
    monkeypatch.setattr(dp, "_move_groups_outside_inference_mode", lambda m: None)
    monkeypatch.setattr(dp, "_evict_rotation_hook", lambda *a: (lambda m, a: None))
    monkeypatch.setattr(dm, "install_group_offload_buffer_restore", lambda: None)
    monkeypatch.setattr(dm, "_streamed_pin_plan", lambda *a, **k: (True, True))

    from diffusers.hooks import group_offloading as go

    def fake_apply(
        module,
        *,
        onload_device,
        offload_device,
        offload_type,
        num_blocks_per_group,
        use_stream,
        non_blocking = False,
        record_stream = False,
        low_cpu_mem_usage = False,
    ):
        # What diffusers does up front with use_stream: one host copy per tensor through ModuleGroup._to_cpu.
        module._fake_cpu_param_dict = {
            p: go.ModuleGroup._to_cpu(p, low_cpu_mem_usage) for p in module.parameters()
        }

    monkeypatch.setattr(hooks, "apply_group_offloading", fake_apply)
    net = torch.nn.Sequential(torch.nn.Linear(128, 128), torch.nn.Linear(128, 64))
    mode = dp.stream_prequantized_module(object(), net, "cuda")
    assert mode == "stream"
    payload = sum(p.numel() * p.element_size() for p in net.parameters())
    assert net._unsloth_pin_arena_bytes[0] == payload
    for p, copy in net._fake_cpu_param_dict.items():
        assert p.data_ptr() == copy.data_ptr()


def _host(
    monkeypatch,
    *,
    available_gb,
    anon_gb,
    shmem_gb,
    file_gb = 0.0,
):
    gb = 1_000_000_000
    # A host with no enforcing cgroup: the CI runner's own limit must not leak into these numbers.
    from utils import host_memory

    monkeypatch.setattr(host_memory, "cgroup_memory_budgets", lambda *a, **k: [])
    monkeypatch.setattr(
        vmh3,
        "_proc_status_kb",
        lambda fields: {"RssAnon": int(anon_gb * gb / 1024), "RssShmem": int(shmem_gb * gb / 1024)},
    )
    import psutil

    monkeypatch.setattr(
        psutil, "virtual_memory", lambda: types.SimpleNamespace(available = int(available_gb * gb))
    )
    monkeypatch.setattr(
        psutil,
        "Process",
        lambda: types.SimpleNamespace(
            memory_info = lambda: types.SimpleNamespace(rss = int((anon_gb + shmem_gb + file_gb) * gb))
        ),
    )


def test_capacity_counts_held_memory_once_and_not_mmapped_page_cache(monkeypatch):
    # A checkpoint still mmap'd is in MemAvailable as page cache AND in RSS as RssFile: count it once.
    _host(monkeypatch, available_gb = 30.0, anon_gb = 36.0, shmem_gb = 20.0, file_gb = 10.8)
    assert vmh3.h3_host_capacity_bytes() / 1e9 == pytest.approx(86.0, abs = 0.01)
    monkeypatch.setenv(vmh3.H3_HOST_GUARD_HELD_ENV, "0")
    assert vmh3.h3_host_capacity_bytes() / 1e9 == pytest.approx(96.8, abs = 0.01)


def test_repeat_render_on_the_reported_94gb_host_is_admitted_with_one_host_copy(monkeypatch):
    # The reported second render: 82.5 GB capacity at a 47 GB VRAM tier, int8 conditioner + int8 denoiser.
    _host(monkeypatch, available_gb = 25.6, anon_gb = 36.7, shmem_gb = 20.2)
    sizes = dict(text_encoder_gb = 27.2, transformer_gb = 20.3)
    assert vmh3.h3_host_ram_shortfall(47.2, transformer_streamed = False, **sizes) is None
    # The same host with the denoiser priced twice (pageable source + pinned copy) is what refused it.
    message = vmh3.h3_host_ram_shortfall(47.2, transformer_streamed = True, **sizes)
    assert message is not None and "85 GB" in message


def test_a_real_host_shortfall_is_still_refused(monkeypatch):
    _host(monkeypatch, available_gb = 3.0, anon_gb = 36.7, shmem_gb = 20.2)
    message = vmh3.h3_host_ram_shortfall(
        47.2, text_encoder_gb = 27.2, transformer_gb = 20.3, transformer_streamed = False
    )
    assert message is not None
    assert "Load the GGUF artifact instead" in message


def test_the_load_records_a_single_host_copy_for_an_arena_pinned_denoiser():
    import inspect

    from core.inference import video as vid

    load = inspect.getsource(vid)
    assert (
        'denoiser_host_copy = denoiser_streamed == "stream" and not denoiser_single_host_copy'
        in load
    )
    assert '"_unsloth_pin_arena_bytes"' in load
