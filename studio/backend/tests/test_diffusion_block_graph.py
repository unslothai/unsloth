# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Per-block CUDA graphs for offloaded denoisers (``diffusion_block_graph.py``) and the prefetcher's slot ring.

CPU cases cover the call-tree / key / refusal logic, the weight-placement read-back, installation below the
offload hooks, the arming decision per placement and the slot assignment. CUDA cases record real blocks under
Studio's ``_apply_group_offload`` (streamed through the event-fenced prefetcher, partially resident, released and
restored) and check every replay against the ungraphed forward bit for bit.
"""

from __future__ import annotations

import copy
import types
import warnings

import pytest
import torch

import core.inference.diffusion_block_graph as bg
import core.inference.diffusion_cuda_graph as cg
import core.inference.diffusion_memory as dm
import core.inference.diffusion_offload_prefetch as op


@pytest.fixture(autouse = True)
def _clean_env(monkeypatch):
    for name in (
        bg.BLOCK_GRAPHS_ENV,
        cg.CUDA_GRAPHS_ENV,
        cg.CUDA_GRAPH_DISABLE_ENV,
        op.ASYNC_PREFETCH_ENV,
        op.PREFETCH_DEPTH_ENV,
        "UNSLOTH_DIFFUSION_PARTIAL_RESIDENT",
        "UNSLOTH_DIFFUSION_GROUP_OFFLOAD_PIN",
        "UNSLOTH_DIFFUSION_PIN_TOP_GROUP",
    ):
        monkeypatch.delenv(name, raising = False)


class QwenImage21KVLayerCache:  # the diffusers class's shape: k / v set by store(), read by get()
    def __init__(self):
        self.k = None
        self.v = None

    def store(self, k, v):
        self.k, self.v = k, v

    def get(self):
        return self.k, self.v


class Block(torch.nn.Module):
    def __init__(self, width = 64):
        super().__init__()
        self.lin = torch.nn.Linear(width, width)

    def forward(
        self,
        x,
        temb = None,
        layer_cache = None,
        kv_cache_mode = None,
    ):
        y = torch.nn.functional.gelu(self.lin(x))
        if temb is not None:
            y = y + temb
        if layer_cache is not None and kv_cache_mode == "cached":
            k, v = layer_cache.get()
            y = y + k.sum() * 0 + v.mean(dim = 0, keepdim = True)
        return y


class Net(torch.nn.Module):
    _repeated_blocks = ["Block"]

    def __init__(
        self,
        width = 64,
        blocks = 6,
    ):
        super().__init__()
        self.proj_in = torch.nn.Linear(16, width)
        self.blocks = torch.nn.ModuleList(Block(width) for _ in range(blocks))
        self.proj_out = torch.nn.Linear(width, 16)

    def forward(
        self,
        x,
        temb = None,
        caches = None,
    ):
        x = self.proj_in(x)
        for i, block in enumerate(self.blocks):
            if caches is not None:
                x = block(x, temb = temb, layer_cache = caches[i], kv_cache_mode = "cached")
            else:
                x = block(x, temb = temb)
        return self.proj_out(x)


# --- call trees and keys ---------------------------------------------------------------------------------------


def test_kv_layer_cache_flattens_to_its_two_tensors_and_rebuilds_a_fresh_cache():
    cache = QwenImage21KVLayerCache()
    cache.store(torch.ones(2, 3), torch.zeros(2, 3))
    live: list = []
    key = bg._walk(((torch.ones(1),), {"layer_cache": cache, "kv_cache_mode": "cached"}), live)
    assert [t.shape for t in live] == [torch.Size([1]), torch.Size([2, 3]), torch.Size([2, 3])]
    hash(key)
    statics = [torch.full_like(t, 7.0) for t in live]
    args, kwargs = bg._unwalk(key, iter(statics))
    assert kwargs["kv_cache_mode"] == "cached"
    rebuilt = kwargs["layer_cache"]
    assert type(rebuilt) is QwenImage21KVLayerCache and rebuilt is not cache
    assert rebuilt.k is statics[1] and rebuilt.v is statics[2]
    assert cache.k is not statics[1]  # the caller's cache is never touched


def test_kv_cache_is_recordable_only_on_a_cached_step():
    cache = QwenImage21KVLayerCache()
    cache.store(torch.ones(2, 3), torch.zeros(2, 3))
    cached = {"layer_cache": cache, "kv_cache_mode": "cached"}
    extract = {"layer_cache": cache, "kv_cache_mode": "extract"}
    assert bg._refusal(bg.graph_key(((), cached)), cached) is None
    assert bg._refusal(bg.graph_key(((), extract)), extract) == "kv_write"


def test_key_separates_inference_mode_shape_and_scalars_and_refuses_floats_and_objects():
    a = torch.zeros(2, 3)
    with torch.inference_mode():
        b = torch.zeros(2, 3)
    assert bg.graph_key(a) != bg.graph_key(b)
    assert bg.graph_key(a) != bg.graph_key(torch.zeros(3, 2))
    assert bg._refusal(bg.graph_key(((a,), {"scale": 0.5})), {}) == "float"
    assert bg._refusal(bg.graph_key(((a,), {"thing": object()})), {}) == "object"
    assert bg._refusal(bg.graph_key(((a,), {"n": 3, "mode": "x", "none": None})), {}) is None


@pytest.mark.parametrize("version", [None, 2])
def test_placement_reads_every_level_of_torchao_weights(version):
    torchao = pytest.importorskip("torchao")
    from torchao.quantization import quantize_

    try:
        from torchao.quantization import Int8WeightOnlyConfig as Cfg
        cfg = Cfg() if version is None else Cfg(version = version)
    except (ImportError, TypeError):
        pytest.skip(f"torchao {torchao.__version__}: no such int8 config")
    block = Block().to(torch.bfloat16)
    quantize_(block, cfg)
    leaves = bg._leaves(block.lin.weight)
    # torchao 0.17's default nests two wrappers whose own data_ptr is 0: the key must reach the plain payload
    assert leaves and all(not bg._is_wrapper_subclass(x) for x in leaves)
    assert all(x.data_ptr() != 0 for x in leaves)


def test_a_host_tensor_input_is_never_recorded():
    _cuda()
    net = _net(blocks = 1).cuda()
    handle, _ = bg.install_block_graphs(net, device = "cuda", slots = False)
    graph = handle.graphs[0]
    width = graph.block.lin.in_features
    x = torch.randn(4, width, device = "cuda")
    with torch.inference_mode():
        for i in range(4):
            temb = torch.tensor(float(i))  # a 0-dim host scalar a CUDA op reads at launch
            assert torch.equal(graph(x, temb = temb), graph.compute(x, temb = temb))
    assert graph.stats["refused_host_input"] == 4 and graph.stats["captures"] == 0
    handle.free()


def test_weight_placement_is_none_while_a_weight_is_on_the_host():
    view = bg._WeightView(Block())
    assert view.placement(None) is None


def test_block_on_the_host_runs_its_compute_and_never_records():
    block = Block()
    calls = []

    def compute(*a, **k):
        calls.append(1)
        return Block.forward(block, *a, **k)

    graph = bg.BlockGraph(block, compute, bg._Shared(None))
    with torch.no_grad():
        for _ in range(3):
            out = graph(torch.randn(2, 64))
    assert out.shape == (2, 64) and len(calls) == 3
    assert graph.stats["refused_host_weight"] == 3 and graph.stats["captures"] == 0


def test_grad_mode_runs_the_compute():
    block = Block()
    graph = bg.BlockGraph(block, block.forward, bg._Shared(None))
    graph(torch.randn(2, 64))
    assert graph.stats["refused_grad"] == 1


# --- installation ----------------------------------------------------------------------------------------------


def _hooked_net(blocks = 4):
    pytest.importorskip("diffusers.hooks")
    from diffusers.hooks import apply_group_offloading

    net = Net(blocks = blocks)
    try:
        apply_group_offloading(
            net,
            onload_device = torch.device("cpu"),
            offload_type = "block_level",
            num_blocks_per_group = 1,
        )
    except (
        RuntimeError
    ) as exc:  # torch < 2.7: diffusers' group offload needs an accelerator even onto the CPU
        pytest.skip(f"diffusers group offload unavailable here: {exc}")
    return net


def test_install_sits_below_the_group_offload_hook_and_free_restores_it():
    net = _hooked_net()
    hook = bg._group_offload_hook(net.blocks[0])
    assert hook is not None
    handle, reason = bg.install_block_graphs(net, slots = False)
    assert reason == "armed" and handle is not None and len(handle.graphs) == 4
    refs = net.blocks[0]._diffusers_hook._fn_refs
    assert any(isinstance(r.forward, bg.BlockGraph) for r in refs)
    x = torch.randn(3, 16)
    with torch.no_grad():
        got = net(x)
    # the hook chain still runs (onload / offload) and the compute is the block's own forward
    assert all(g.stats["refused_host_weight"] == 1 for g in handle.graphs)
    handle.free()
    assert not any(isinstance(r.forward, bg.BlockGraph) for r in refs)
    with torch.no_grad():
        assert torch.equal(net(x), got)


def test_compile_moves_below_the_offload_hook_and_the_kill_switch_keeps_it_traced(monkeypatch):
    net = _hooked_net(blocks = 2)
    net._unsloth_regional_compile_kwargs = {"fullgraph": False, "dynamic": True}
    marker = lambda *a, **k: None  # noqa: E731
    for b in net.blocks:
        b._compiled_call_impl = marker
    monkeypatch.setenv(bg.COMPILE_BELOW_HOOKS_ENV, "0")
    assert bg.compile_below_offload_hooks(net) == 0
    assert all(b._compiled_call_impl is marker for b in net.blocks)
    monkeypatch.delenv(bg.COMPILE_BELOW_HOOKS_ENV)
    assert bg.compile_below_offload_hooks(net) == 2
    for b in net.blocks:
        assert b._compiled_call_impl is None
        refs = b._diffusers_hook._fn_refs
        assert not any(bg._is_original_forward(r.forward, b) for r in refs)
    # idempotent, and the block graph layer still finds the (now compiled) compute below the hook
    assert bg.compile_below_offload_hooks(net) == 0
    handle, reason = bg.install_block_graphs(net, slots = False)
    assert reason == "armed" and len(handle.graphs) == 2
    for b, g in zip(net.blocks, handle.graphs):
        assert b._unsloth_below_hook_ref.forward is g
    handle.free()
    assert not any(isinstance(b._unsloth_below_hook_ref.forward, bg.BlockGraph) for b in net.blocks)


def test_compile_below_the_hook_keeps_the_block_restride():
    from core.inference import diffusion_block_restride as restride

    net = _hooked_net(blocks = 2)
    net._unsloth_regional_compile_kwargs = {"fullgraph": False, "dynamic": True}
    marker = lambda *a, **k: None  # noqa: E731
    net.blocks[0]._compiled_call_impl = restride.wrap(marker)
    net.blocks[1]._compiled_call_impl = marker
    assert bg.compile_below_offload_hooks(net) == 2
    assert restride.is_wrapped(net.blocks[0]._unsloth_below_hook_ref.forward)
    assert not restride.is_wrapped(net.blocks[1]._unsloth_below_hook_ref.forward)


def test_static_buffers_keep_the_storage_offset_of_a_view():
    full = torch.arange(24.0).view(2, 12)
    view = full[:, 4:]
    static = bg._static_like(view)
    assert static.shape == view.shape and static.stride() == view.stride()
    assert static.storage_offset() == view.storage_offset() == 4
    static.copy_(view)
    assert torch.equal(static, view)


def test_install_on_an_unhooked_compiled_block_takes_the_compiled_call_slot():
    net = Net(blocks = 2)
    marker = lambda *a, **k: "compiled"  # noqa: E731
    for b in net.blocks:
        b._compiled_call_impl = marker
    handle, _ = bg.install_block_graphs(net, slots = False)
    assert all(isinstance(b._compiled_call_impl, bg.BlockGraph) for b in net.blocks)
    assert handle.graphs[0].compute is marker
    handle.free()
    assert all(b._compiled_call_impl is marker for b in net.blocks)


def test_per_layer_offload_hooks_inside_a_block_are_refused():
    net = Net(blocks = 2)
    for b in net.blocks:
        b.lin._hf_hook = object()
    handle, reason = bg.install_block_graphs(net, slots = False)
    assert handle is None and "per-layer offload hooks" in reason


def test_kill_switches(monkeypatch):
    net = Net(blocks = 2)
    monkeypatch.setenv(bg.BLOCK_GRAPHS_ENV, "0")
    handle, reason = bg.install_block_graphs(net, slots = False)
    assert handle is None and bg.BLOCK_GRAPHS_ENV in reason
    monkeypatch.delenv(bg.BLOCK_GRAPHS_ENV)
    monkeypatch.setenv(cg.CUDA_GRAPHS_ENV, "0")
    assert cg.cuda_graph_disabled()
    ok, why = cg.graph_eligible(
        types.SimpleNamespace(device = "cuda", backend = "cuda"),
        family = None,
        pipe = None,
        offload_active = False,
        cache_active = False,
        speed_mode = "default",
    )
    assert not ok and cg.CUDA_GRAPHS_ENV in why


# --- arming per placement --------------------------------------------------------------------------------------


def _pipe_with(net):
    return types.SimpleNamespace(transformer = net, components = {"transformer": net})


def _arm(
    pipe,
    monkeypatch,
    *,
    hooked,
    pinned = False,
    backend = "cuda",
    cache = False,
    mode = "default",
    cuda = True,
    opt_in = True,
):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    if opt_in:
        monkeypatch.setenv(bg.BLOCK_GRAPHS_ENV, "1")
    applied = {"cuda_graph": False}
    target = types.SimpleNamespace(device = "cuda", backend = backend, torch_device = "cuda")
    handles = cg.arm_block_graphs(
        pipe,
        applied,
        target = target,
        family = types.SimpleNamespace(),
        hooked = hooked,
        pinned = pinned,
        cache_engaged = cache,
        speed_mode = mode,
    )
    return handles, applied


def test_an_offloaded_denoiser_is_recorded_per_block(monkeypatch):
    pipe = _pipe_with(Net(blocks = 3))
    pipe._unsloth_cuda_graph_reason = "offload active"
    handles, applied = _arm(pipe, monkeypatch, hooked = True)
    assert applied["cuda_graph"] and len(handles) == 1 and len(handles[0].graphs) == 3
    assert "recorded per block" in cg.status_reason(pipe, True)
    cg.uninstall_all(handles)


def test_per_block_graphs_are_opt_in_and_say_so(monkeypatch):
    pipe = _pipe_with(Net(blocks = 3))
    pipe._unsloth_cuda_graph_reason = "offload active"
    handles, applied = _arm(pipe, monkeypatch, hooked = True, opt_in = False)
    assert handles == () and not applied["cuda_graph"]
    assert bg.BLOCK_GRAPHS_ENV + "=1" in cg.status_reason(pipe, False)
    assert not any(
        isinstance(b.__dict__.get("forward"), bg.BlockGraph) for b in pipe.transformer.blocks
    )
    monkeypatch.setenv(bg.BLOCK_GRAPHS_ENV, "0")
    pipe2 = _pipe_with(Net(blocks = 3))
    handles, applied = _arm(pipe2, monkeypatch, hooked = True, opt_in = False)
    assert handles == () and "BLOCK_GRAPHS=0" in cg.status_reason(pipe2, False)


def test_blocks_that_stay_on_the_device_record_by_default(monkeypatch):
    monkeypatch.delenv(bg.BLOCK_GRAPHS_ENV, raising = False)
    pipe = _pipe_with(Net(blocks = 3))
    pipe._unsloth_cuda_graph_reason = "offload active"
    handles, applied = _arm(pipe, monkeypatch, hooked = True, pinned = True, opt_in = False)
    assert applied["cuda_graph"] and len(handles[0].graphs) == 3
    assert cg.status_reason(pipe, True).startswith("pinned denoiser recorded per block")
    cg.uninstall_all(handles)
    pipe = _pipe_with(Net(blocks = 2))
    pipe._unsloth_cuda_graph_reason = (
        "QwenImage21Transformer2DModel forward is not capture-safe (prefix KV cache)"
    )
    handles, applied = _arm(pipe, monkeypatch, hooked = False, opt_in = False)
    assert applied["cuda_graph"] and handles
    cg.uninstall_all(handles)
    monkeypatch.setenv(bg.BLOCK_GRAPHS_ENV, "0")
    pipe = _pipe_with(Net(blocks = 2))
    handles, applied = _arm(pipe, monkeypatch, hooked = True, pinned = True, opt_in = False)
    assert handles == () and "BLOCK_GRAPHS=0" in cg.status_reason(pipe, False)


def test_model_offload_stays_opt_in_and_says_why(monkeypatch):
    monkeypatch.delenv(bg.BLOCK_GRAPHS_ENV, raising = False)
    net = Net(blocks = 2)
    net._hf_hook = object()
    pipe = _pipe_with(net)
    handles, applied = _arm(pipe, monkeypatch, hooked = True, opt_in = False)
    assert handles == () and not applied["cuda_graph"]
    assert "model offload re-uploads" in cg.status_reason(pipe, False)
    handles, applied = _arm(_pipe_with(net), monkeypatch, hooked = True, opt_in = True)
    assert applied["cuda_graph"] and handles
    cg.uninstall_all(handles)


def test_a_resident_whole_forward_recording_is_kept(monkeypatch):
    net = Net(blocks = 2)
    pipe = _pipe_with(net)
    whole = cg.GraphedForward(net)
    pipe._unsloth_cuda_graphs = (whole,)
    handles, applied = _arm(pipe, monkeypatch, hooked = False)
    assert handles == (whole,)
    assert cg.status_reason(pipe, True) == cg.WHOLE_REASON


def test_a_forward_that_is_not_capture_safe_is_recorded_per_block(monkeypatch):
    pipe = _pipe_with(Net(blocks = 2))
    pipe._unsloth_cuda_graph_reason = (
        "QwenImage21Transformer2DModel forward is not capture-safe (prefix KV cache object)"
    )
    handles, applied = _arm(pipe, monkeypatch, hooked = False)
    assert applied["cuda_graph"] and handles
    cg.uninstall_all(handles)


def test_unrelated_refusals_stay_off_with_their_reason(monkeypatch):
    pipe = _pipe_with(Net(blocks = 2))
    pipe._unsloth_cuda_graph_reason = "speed tier off"
    handles, applied = _arm(pipe, monkeypatch, hooked = False, mode = "off")
    assert handles == () and not applied["cuda_graph"]
    for kw, want in (
        ({"backend": "rocm"}, "backend is rocm"),
        ({"cache": True}, "step cache active"),
        ({"mode": "off"}, "speed tier off"),
    ):
        pipe = _pipe_with(Net(blocks = 2))
        handles, applied = _arm(pipe, monkeypatch, hooked = True, **kw)
        assert handles == () and not applied["cuda_graph"]
        assert cg.status_reason(pipe, False) == want


def test_a_sage_denoiser_is_never_recorded_per_block(monkeypatch):
    net = Net(blocks = 2)
    net._unsloth_attention_backend = "sage"
    pipe = _pipe_with(net)
    handles, applied = _arm(pipe, monkeypatch, hooked = True, pinned = True)
    assert handles == () and not applied["cuda_graph"]
    assert "SageAttention" in cg.status_reason(pipe, False)


def test_the_master_switch_turns_block_graphs_off_with_its_name(monkeypatch):
    monkeypatch.setenv(cg.CUDA_GRAPHS_ENV, "0")
    pipe = _pipe_with(Net(blocks = 2))
    handles, applied = _arm(pipe, monkeypatch, hooked = True)
    assert handles == () and cg.CUDA_GRAPHS_ENV in cg.status_reason(pipe, False)


def test_status_says_why_armed_blocks_never_replayed():
    net = Net(blocks = 2)
    handle, _ = bg.install_block_graphs(net, slots = False)
    assert cg.never_engaged((handle,)) is None  # nothing ran yet
    with torch.no_grad():
        net(torch.randn(2, 16))
    why = cg.never_engaged((handle,))
    assert why is not None and "weights not on the GPU" in why
    resolved, optims = cg.live_status({"cuda_graph": {"value": "on"}}, ["cuda_graph"], (handle,))
    assert resolved["cuda_graph"]["value"] == "off" and "cuda_graph" not in optims
    handle.graphs[0].churned = True
    assert "new address every call" in handle.why_off()
    handle.free()


# --- slot ring -------------------------------------------------------------------------------------------------


class _G:
    def __init__(self, name, shape):
        self.name = name
        self.modules = []
        self.parameters = [torch.zeros(shape)]
        self.buffers = []
        self.cpu_param_dict = {}
        self.offload_leader = None


def _slot_prefetcher(shapes, depth = 2):
    groups = [_G(str(i), s) for i, s in enumerate(shapes)]
    pf = op.GroupPrefetcher.__new__(op.GroupPrefetcher)
    pf.module = object()
    pf.depth = depth
    pf.groups = groups
    pf.by_id = {id(g): g for g in groups}
    pf.nbytes = {id(g): 10 for g in groups}
    pf.ready = {}
    pf.active = False
    pf.stats = {"dropped": 0}
    pf.slot_of, pf.slot_buffers, pf.slot_raw, pf.slot_owner = {}, {}, {}, {}
    pf.slot_size, pf.slot_bytes, pf.slot_bytes_planned = 0, 0, 0
    released = []
    pf._release = lambda g, e, counted = True: released.append(g.name)
    return pf, groups, released


def test_slots_are_shared_round_robin_and_sized_like_the_prefetch_window():
    pf, groups, _ = _slot_prefetcher([(4, 4)] * 5 + [(8, 4)] * 2, depth = 2)
    pf.enable_slots()
    # every layout shares the same depth + 1 slots, in block order
    assert [pf.slot_of[id(g)] for g in groups] == [0, 1, 2, 0, 1, 2, 0]
    # each slot holds the largest packed group (an (8, 4) float32 = 128 bytes): the ring is the prefetch window
    assert pf.slot_size == 128 and pf.slot_bytes_planned == 3 * 128
    assert pf.slot_streamed == 7 and pf.slot_bytes == 0  # planned, allocated on first fill


def test_a_pinned_denoiser_counts_no_streamed_slot_groups():
    pf, groups, _ = _slot_prefetcher([(4, 4)] * 3, depth = 2)
    for g in groups:
        g._unsloth_resident = True
    pf.enable_slots()
    assert pf.slot_streamed == 0 and pf.slot_bytes == 0


def test_slot_views_pack_leaves_aligned_and_mirror_the_sources():
    raw = torch.zeros(4096, dtype = torch.uint8)
    srcs = [torch.arange(6, dtype = torch.float32).view(2, 3), torch.ones(5, dtype = torch.bfloat16)]
    assert op._packed_bytes(srcs) == op._SLOT_ALIGN + 10
    views = op._slot_views(raw, srcs, torch.device("cpu"))
    assert [v.shape for v in views] == [torch.Size([2, 3]), torch.Size([5])]
    assert views[1].data_ptr() - views[0].data_ptr() == op._SLOT_ALIGN
    for v, src in zip(views, srcs):
        op._copy_into(v, src)
        assert torch.equal(v, src)
    assert op._packed_bytes([torch.zeros(4, 4).t()]) is None  # a strided leaf cannot be viewed in


def test_a_slot_held_by_a_group_on_the_device_is_never_refilled_ahead():
    pf, groups, released = _slot_prefetcher([(4, 4)] * 4, depth = 2)
    pf.enable_slots()
    a, d = groups[0], groups[3]  # same slot
    pf.slot_owner[pf.slot_of[id(a)]] = id(a)
    pf.ready[id(a)] = None  # onloaded, compute in flight
    assert pf._slot_free(d, must = False) is False
    # a forced onload of d copies to fresh memory instead (the block then runs ungraphed once)
    assert pf._slot_free(d, must = True) is None and pf.stats["slot_fallbacks"] == 1
    # copied ahead but not run yet: a forced onload drops it and takes the slot
    pf.ready[id(a)] = object()
    assert pf._slot_free(d, must = True) is True and released == ["0"]
    # released (no longer on the device): free
    pf.ready.clear()
    assert pf._slot_free(d, must = False) is True


def test_the_top_level_group_keeps_fresh_copies():
    pf, groups, _ = _slot_prefetcher([(4, 4)] * 3)
    groups[0].offload_leader = pf.module
    pf.enable_slots()
    assert id(groups[0]) not in pf.slot_of and len(pf.slot_of) == 2


# --- memory guard ----------------------------------------------------------------------------------------------


def test_pool_bytes_count_only_the_target_card(monkeypatch):
    here, there = bg._Shared(0), bg._Shared(1)
    here.pool_bytes, there.pool_bytes = 100, 900
    monkeypatch.setattr(bg, "_LIVE", {here, there})
    assert bg.pool_bytes(0) == 100 and bg.pool_bytes() == 1000


def test_block_graph_pools_are_not_credited_as_reclaimable(monkeypatch):
    target = types.SimpleNamespace(device = "cuda", backend = "cuda")
    snap = dm.DeviceMemory("cuda", "cuda", "dedicated", 1000, 8000)
    monkeypatch.setattr(dm, "snapshot_device_memory", lambda t: snap)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda *a, **k: 600 << 20)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda *a, **k: 100 << 20)
    monkeypatch.setattr(bg, "pool_bytes", lambda device = None: 0)
    assert dm.reclaimable_snapshot_device_memory(target).free_mib == 1500
    monkeypatch.setattr(bg, "pool_bytes", lambda device = None: 300 << 20)
    assert dm.reclaimable_snapshot_device_memory(target).free_mib == 1200


# --- CUDA ------------------------------------------------------------------------------------------------------


def _cuda():
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    pytest.importorskip("diffusers.hooks")


def _streamed(net, resident_mib = None):
    pipe = _pipe_with(net)
    kwargs = {}
    if resident_mib:
        kwargs["resident_transformer_mib"] = resident_mib
    assert dm._apply_group_offload(pipe, "cuda", None, **kwargs)
    return pipe


def _no_syncs(fn):
    prev = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("warn")
    try:
        with warnings.catch_warnings(record = True) as caught:
            warnings.simplefilter("always")
            out = fn()
    finally:
        torch.cuda.set_sync_debug_mode(prev)
    return out, sum(1 for w in caught if "synchroniz" in str(w.message).lower())


def _net(
    width = 256,
    blocks = 8,
    dtype = torch.float32,
):
    torch.manual_seed(0)
    return Net(width = width, blocks = blocks).to(dtype)


@pytest.mark.parametrize("resident_mib", [None, 1])
def test_streamed_blocks_replay_from_the_slot_ring_bit_identically(resident_mib):
    _cuda()
    net = _net()
    ref = copy.deepcopy(net).cuda()
    _streamed(net, resident_mib)
    pf = op.module_prefetcher(net)
    assert pf is not None
    handle, reason = bg.install_block_graphs(net, device = "cuda")
    assert reason == "armed" and handle.slots_mib >= 0
    x = torch.randn(4, 16, device = "cuda")
    temb = torch.randn(1, 256, device = "cuda")
    with torch.no_grad():
        want = ref(x, temb = temb)
        for i in range(4):
            got, syncs = _no_syncs(lambda: net(x, temb = temb))
            torch.cuda.synchronize()
            assert torch.equal(got, want), i
            if i >= 2:
                assert syncs == 0
        # a different input is copied into the same static buffers and replays the same recordings
        x2 = torch.randn(4, 16, device = "cuda")
        captures = handle.stats["captures"]
        assert torch.equal(net(x2, temb = temb), ref(x2, temb = temb))
        assert handle.stats["captures"] == captures
    s = handle.stats
    assert s["replays"] >= 8 and s["fallbacks"] == 0 and s["refused_host_weight"] == 0
    # one recording per block (weights at their slot), each made once
    assert s["captures"] == 8 and s["recaptures"] == 0
    assert pf.stats["slot_fills"] > 0 and pf.stats.get("slot_fallbacks", 0) == 0
    assert net.blocks[-1].lin.weight.device.type == "cpu"
    handle.free()
    with torch.no_grad():
        assert torch.equal(net(x, temb = temb), want)


class SlowBlock(Block):
    def forward(
        self,
        x,
        temb = None,
        layer_cache = None,
        kv_cache_mode = None,
    ):
        torch.cuda._sleep(
            200_000
        )  # a slow GPU: the host queues later copies long before this block reads
        return super().forward(x, temb = temb)


def test_a_slow_gpu_never_reads_a_slot_the_copy_stream_is_refilling(monkeypatch):
    _cuda()
    monkeypatch.setenv(op.PREFETCH_DEPTH_ENV, "3")
    torch.manual_seed(0)
    net = Net(width = 512, blocks = 10)
    net.blocks = torch.nn.ModuleList(SlowBlock(512) for _ in range(10))
    Net._repeated_blocks = ["Block", "SlowBlock"]
    try:
        ref = copy.deepcopy(net).cuda()
        _streamed(net)
        handle, _ = bg.install_block_graphs(net, device = "cuda")
        x = torch.randn(4, 16, device = "cuda")
        with torch.no_grad():
            want = ref(x)
            for _ in range(5):
                got = net(x)
            torch.cuda.synchronize()
            assert torch.equal(got, want)
        assert handle.stats["replays"] > 0
        handle.free()
    finally:
        Net._repeated_blocks = ["Block"]


def test_compiled_streamed_blocks_record_below_their_hooks_with_no_graph_break():
    _cuda()
    import torch._dynamo.utils as du

    net = _net(blocks = 6)
    ref = copy.deepcopy(net).cuda()
    kwargs = {"fullgraph": False, "dynamic": None}
    for b in net.blocks:
        b.compile(**kwargs)
    net._unsloth_regional_compile_kwargs = kwargs
    _streamed(net)
    assert bg.compile_below_offload_hooks(net) == 6
    handle, reason = bg.install_block_graphs(net, device = "cuda")
    assert reason == "armed" and len(handle.graphs) == 6
    breaks = sum(du.counters["graph_break"].values())
    x = torch.randn(4, 16, device = "cuda")
    with torch.no_grad():
        want = ref(x)
        for _ in range(4):
            got = net(x)
            torch.cuda.synchronize()
            torch.testing.assert_close(got, want, rtol = 1e-4, atol = 1e-4)
        first = got.clone()
        for _ in range(2):
            assert torch.equal(net(x), first)  # replays are deterministic
    assert (
        sum(du.counters["graph_break"].values()) == breaks
    )  # the hooks stay outside the compiled region
    s = handle.stats
    assert s["replays"] > 0 and s["fallbacks"] == 0 and s["captures"] == 6
    handle.free()


def test_alternating_layouts_both_record():
    """True CFG calls each block with two text lengths in turn; each layout must still record and replay."""
    _cuda()
    net = _net(blocks = 2).cuda()
    ref = copy.deepcopy(net)
    handle, _ = bg.install_block_graphs(net, device = "cuda", slots = False)
    xs = [torch.randn(7, 16, device = "cuda"), torch.randn(5, 16, device = "cuda")]
    with torch.inference_mode():
        for _ in range(4):
            for x in xs:
                assert torch.equal(net(x), ref(x))
    s = handle.stats
    assert s["captures"] == 4 and s["replays"] >= 8 and s["fallbacks"] == 0
    handle.free()


def test_evicted_layouts_free_their_static_buffers():
    _cuda()
    net = _net(blocks = 2).cuda()
    ref = copy.deepcopy(net)
    handle, _ = bg.install_block_graphs(net, device = "cuda", slots = False)
    for g in handle.graphs:
        g.max_graphs = 2
    shared = handle.graphs[0].shared
    with torch.inference_mode():
        for rows in range(3, 11):  # a new layout per render, as a new prompt length is
            x = torch.randn(rows, 16, device = "cuda")
            for _ in range(3):
                assert torch.equal(net(x), ref(x))
    assert handle.stats["evictions"] > 0
    assert len(shared.static_in) == 2 and len(shared.static_out) == 2
    handle.free()


def test_each_nvfp4_precision_branch_records_its_own_graph():
    _cuda()
    net = _net(blocks = 2).cuda()
    ref = copy.deepcopy(net)
    handle, _ = bg.install_block_graphs(net, device = "cuda", slots = False)
    ctl = types.SimpleNamespace(armed = True, protected = False)
    for g in handle.graphs:
        g.protect = ctl
    x = torch.randn(4, 16, device = "cuda")
    with torch.inference_mode():
        for protected in (False, True, False, True, False, True):
            ctl.protected = protected
            assert torch.equal(net(x), ref(x))
    assert handle.stats["captures"] == 2 * len(handle.graphs)
    for g in handle.graphs:
        assert {k[1][1] for k in g.cache} == {
            (("nvfp4_protect", False),),
            (("nvfp4_protect", True),),
        }
    handle.free()


def test_a_block_whose_weights_moved_records_again_at_the_new_addresses():
    _cuda()
    net = _net(blocks = 3).cuda()
    ref = copy.deepcopy(net)
    handle, _ = bg.install_block_graphs(net, device = "cuda", slots = False)
    x = torch.randn(4, 16, device = "cuda")
    with torch.inference_mode():
        want = ref(x)
        for _ in range(3):
            assert torch.equal(net(x), want)
        assert handle.stats["captures"] == 3
        # the 12 GB tier's encode release / restore and model offload's re-upload land weights elsewhere
        for block in net.blocks:
            block.lin.weight.data = block.lin.weight.data.clone()
        for _ in range(3):
            assert torch.equal(net(x), want)
        # the moved weights change the result if a stale recording replayed them: perturb and compare
        net.blocks[1].lin.weight.data = net.blocks[1].lin.weight.data * 2
        ref.blocks[1].lin.weight.data = ref.blocks[1].lin.weight.data * 2
        for _ in range(3):
            assert torch.equal(net(x), ref(x))
    assert handle.stats["recaptures"] >= 3 and handle.stats["fallbacks"] == 0
    handle.free()


def test_prefix_kv_cache_blocks_replay_on_cached_steps():
    _cuda()
    net = _net(blocks = 3).cuda()
    ref = copy.deepcopy(net)
    caches = []
    for _ in range(3):
        c = QwenImage21KVLayerCache()
        c.store(torch.randn(5, 256, device = "cuda"), torch.randn(5, 256, device = "cuda"))
        caches.append(c)
    handle, _ = bg.install_block_graphs(net, device = "cuda", slots = False)
    x = torch.randn(4, 16, device = "cuda")
    with torch.inference_mode():
        want = ref(x, caches = caches)
        for _ in range(4):
            assert torch.equal(net(x, caches = caches), want)
        # new prefix values (a new prompt) are copied in; the recordings stay
        for c in caches:
            c.store(torch.randn(5, 256, device = "cuda"), torch.randn(5, 256, device = "cuda"))
        captures = handle.stats["captures"]
        assert torch.equal(net(x, caches = caches), ref(x, caches = caches))
        assert handle.stats["captures"] == captures
    assert handle.stats["replays"] > 0 and handle.stats["fallbacks"] == 0
    handle.free()


def test_torchao_int8_weights_stream_through_the_slot_ring_with_graphs():
    _cuda()
    torchao = pytest.importorskip("torchao")
    from torchao.quantization import quantize_

    try:
        from torchao.quantization import Int8WeightOnlyConfig as Cfg
    except ImportError:
        pytest.skip(f"torchao {torchao.__version__} has no Int8WeightOnlyConfig")
    net = _net(dtype = torch.bfloat16)
    quantize_(net, Cfg())
    ref = copy.deepcopy(net).cuda()
    _streamed(net)
    pf = op.module_prefetcher(net)
    assert pf is not None
    handle, _ = bg.install_block_graphs(net, device = "cuda")
    x = torch.randn(4, 16, device = "cuda", dtype = torch.bfloat16)
    with torch.no_grad():
        want = ref(x)
        for _ in range(4):
            got = net(x)
            torch.cuda.synchronize()
            assert torch.equal(got, want)
    assert handle.stats["replays"] > 0 and handle.stats["fallbacks"] == 0
    assert pf.stats["slot_fills"] > 0
    handle.free()
