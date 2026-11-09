# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Whole-component offload onloads pinned weights one chunk at a time, not one tensor at a time.

The offload chain empties the device cache between components, so a per-tensor onload pays one cudaMalloc per
weight (and one cudaFree on the way out). Under host load that ran at 57-120 MiB/s instead of the link's 50 GiB/s.
CPU tests fake the pinning and onload to ``meta``: a per-chunk onload leaves every weight a view of one shared
allocation, a per-tensor onload gives every weight its own.
"""

from __future__ import annotations

import sys
import types

import pytest

torch = pytest.importorskip("torch")
hooks = pytest.importorskip("accelerate.hooks")

from core.inference import diffusion_memory as dm  # noqa: E402

N_LAYERS = 6


def _stack():
    torch.manual_seed(0)
    return torch.nn.Sequential(*(torch.nn.Linear(16, 16) for _ in range(N_LAYERS)))


def _pipe(device, *names):
    comps = {name: _stack() for name in names}
    pipe = types.SimpleNamespace(components = comps)

    def enable_model_cpu_offload(device = device):
        prev = None
        for module in comps.values():
            hooks.remove_hook_from_module(module)
            module.to("cpu")
            hook = hooks.CpuOffload(execution_device = device, prev_module_hook = prev)
            hooks.add_hook_to_module(module, hook)
            prev = hooks.UserCpuOffloadHook(module, hook)

    pipe.enable_model_cpu_offload = enable_model_cpu_offload
    for name, module in comps.items():
        setattr(pipe, name, module)
    return pipe


@pytest.fixture
def fake_pin(monkeypatch):
    """Pinning without CUDA: plain host chunks, small enough that one module spans several."""
    real_empty = torch.empty
    device_allocs: list = []

    def empty(*args, **kwargs):
        kwargs.pop("pin_memory", None)
        out = real_empty(*args, **kwargs)
        if out.device.type != "cpu":
            device_allocs.append(out.nbytes)
        return out

    monkeypatch.setenv(dm.OFFLOAD_PIN_ENV, "1")
    monkeypatch.setattr(dm, "_PIN_CHUNK_BYTES", 4096)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch, "empty", empty)
    return device_allocs


def _storages(module):
    return {p.untyped_storage().nbytes() for p in module.parameters()}


def _onload(pipe):
    module = pipe.transformer
    module._hf_hook.pre_forward(module, torch.zeros(1, 16))
    return module


def test_pinning_records_the_chunk_layout(fake_pin):
    module = _stack()
    host = {n: p.data for n, p in module.named_parameters()}
    layout: list = []
    assert dm._pin_host_weights(module, host, layout = layout) > 0
    assert 1 < len(layout) < len(host)
    names = [name for _, entries in layout for name, _, _ in entries]
    assert sorted(names) == sorted(host)
    for buf, entries in layout:
        for name, start, view in entries:
            assert view is host[name]
            assert view.data_ptr() == buf.data_ptr() + start


def test_a_device_that_cannot_back_host_parameters_takes_the_stock_copy(fake_pin):
    # meta tensors cannot be set as a CPU Parameter's data; the stock move hands back new Parameters there.
    pipe = _pipe("meta", "transformer")
    pipe.enable_model_cpu_offload()
    dm.keep_cpu_weights_on_offload(pipe)
    module = _onload(pipe)
    assert module.__dict__[dm._KEEP_ATTR]["pin_layout"]
    assert all(p.device.type == "meta" for p in module.parameters())
    assert all(p.untyped_storage().nbytes() == p.nbytes for p in module.parameters())


@pytest.mark.parametrize(
    "value,on", [("", True), ("1", True), ("0", False), ("off", False), ("False", False)]
)
def test_fast_upload_switch(monkeypatch, value, on):
    monkeypatch.setenv(dm.OFFLOAD_FAST_UPLOAD_ENV, value)
    assert dm._fast_upload_enabled() is on


def _fake_pipeline_utils(monkeypatch):
    calls: list = []
    module = types.ModuleType("diffusers.pipelines.pipeline_utils")
    module.empty_device_cache = lambda *a, **k: calls.append(a)
    monkeypatch.setitem(sys.modules, "diffusers.pipelines.pipeline_utils", module)
    return module, calls


def test_the_re_enable_keeps_the_device_cache_only_once_chunks_are_in_use(monkeypatch):
    module, calls = _fake_pipeline_utils(monkeypatch)
    lin = torch.nn.Linear(2, 2)
    pipe = types.SimpleNamespace(components = {"transformer": lin})
    with dm._keep_device_cache(pipe):
        module.empty_device_cache("cuda")
    assert calls == [("cuda",)]  # nothing pinned yet: the stock empty_cache runs
    lin.__dict__[dm._KEEP_ATTR] = {"pin_layout": [object()]}
    with dm._keep_device_cache(pipe):
        module.empty_device_cache("cuda")
    assert calls == [("cuda",)]
    module.empty_device_cache("cuda")  # restored on exit
    assert len(calls) == 2
    monkeypatch.setenv(dm.OFFLOAD_FAST_UPLOAD_ENV, "0")
    with dm._keep_device_cache(pipe):
        module.empty_device_cache("cuda")
    assert len(calls) == 3


def test_the_device_cache_is_restored_when_the_re_enable_raises(monkeypatch):
    module, calls = _fake_pipeline_utils(monkeypatch)
    original = module.empty_device_cache
    lin = torch.nn.Linear(2, 2)
    lin.__dict__[dm._KEEP_ATTR] = {"pin_layout": [object()]}
    with pytest.raises(RuntimeError):
        with dm._keep_device_cache(types.SimpleNamespace(components = {"t": lin})):
            raise RuntimeError("boom")
    assert module.empty_device_cache is original


cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a CUDA device")


def _render(
    pipe,
    x,
    ctx = torch.no_grad,
):
    out = x
    with ctx():
        for module in pipe.components.values():
            out = module(out.to(next(module.parameters()).device))
    out = out.detach().cpu()
    pipe.transformer._hf_hook.init_hook(pipe.transformer)
    pipe.enable_model_cpu_offload()
    return out


def _device_allocs():
    return torch.cuda.memory_stats().get("num_device_alloc", 0)


@cuda
@pytest.mark.parametrize("mode", ["no_grad", "inference_mode"])
def test_chunk_onload_is_bit_identical_and_stops_allocating(monkeypatch, mode):
    monkeypatch.setenv(dm.OFFLOAD_PIN_ENV, "1")
    monkeypatch.setattr(dm, "_PIN_CHUNK_BYTES", 4096)
    ctx = torch.inference_mode if mode == "inference_mode" else torch.no_grad
    x = torch.randn(4, 16)
    stock = _pipe("cuda", "text_encoder", "transformer")
    stock.enable_model_cpu_offload()
    ref = _render(stock, x, ctx)
    pipe = _pipe("cuda", "text_encoder", "transformer")
    pipe.enable_model_cpu_offload()
    dm.keep_cpu_weights_on_offload(pipe)
    assert torch.equal(_render(pipe, x, ctx), ref)  # the first render pins
    layouts = {n: m.__dict__[dm._KEEP_ATTR]["pin_layout"] for n, m in pipe.components.items()}
    assert all(1 < len(layout) < N_LAYERS * 2 for layout in layouts.values())
    seen = []

    def spy(module, *args):
        # A post-hook: accelerate onloads inside forward, after any pre-hook.
        assert all(p.device.type == "cuda" for p in module.parameters())
        seen.append({p.untyped_storage().data_ptr() for p in module.parameters()})

    for module in pipe.components.values():
        module.register_forward_hook(spy)
    allocs = _device_allocs()
    for _ in range(3):
        assert torch.equal(_render(pipe, x, ctx), ref)
    # Onloaded weights are views of one allocation per chunk (per tensor would be one storage per weight) ...
    assert [len(s) for s in seen] == [len(layouts["text_encoder"]), len(layouts["transformer"])] * 3
    # ... and the chunks come back out of the caching allocator: no cudaMalloc per swap.
    assert _device_allocs() - allocs <= 2
    host = pipe.transformer.__dict__[dm._KEEP_ATTR]["host"]
    assert all(
        p.data_ptr() == host[n].data_ptr() and p.is_pinned()
        for n, p in pipe.transformer.named_parameters()
    )


@cuda
def test_kill_switch_restores_the_per_tensor_onload(monkeypatch):
    monkeypatch.setenv(dm.OFFLOAD_PIN_ENV, "1")
    monkeypatch.setenv(dm.OFFLOAD_FAST_UPLOAD_ENV, "0")
    pipe = _pipe("cuda", "transformer")
    pipe.enable_model_cpu_offload()
    dm.keep_cpu_weights_on_offload(pipe)
    module = pipe.transformer
    module._hf_hook.pre_forward(module, torch.zeros(1, 16))
    assert all(p.device.type == "cuda" for p in module.parameters())
    assert all(p.untyped_storage().nbytes() == p.nbytes for p in module.parameters())


@cuda
def test_a_weight_replaced_while_offloaded_takes_the_stock_copy(monkeypatch):
    monkeypatch.setenv(dm.OFFLOAD_PIN_ENV, "1")
    monkeypatch.setattr(dm, "_PIN_CHUNK_BYTES", 4096)
    pipe = _pipe("cuda", "transformer")
    pipe.enable_model_cpu_offload()
    dm.keep_cpu_weights_on_offload(pipe)
    module = pipe.transformer
    _render(pipe, torch.randn(4, 16))
    fresh = torch.nn.Parameter(torch.full((16, 16), 0.5))
    module[0].weight = fresh
    module._hf_hook.pre_forward(module, torch.zeros(1, 16))
    assert all(p.device.type == "cuda" for p in module.parameters())
    assert module[0].weight.untyped_storage().nbytes() == fresh.nbytes
    assert torch.equal(module[0].weight.cpu(), torch.full((16, 16), 0.5))
    assert module[1].weight.untyped_storage().nbytes() > module[1].weight.nbytes


@cuda
def test_a_weight_written_on_the_device_after_a_chunk_onload_is_copied_back(monkeypatch):
    monkeypatch.setenv(dm.OFFLOAD_PIN_ENV, "1")
    pipe = _pipe("cuda", "transformer")
    pipe.enable_model_cpu_offload()
    dm.keep_cpu_weights_on_offload(pipe)
    module = pipe.transformer
    _render(pipe, torch.randn(4, 16))
    before = module[2].weight.detach().clone()
    module._hf_hook.pre_forward(module, torch.zeros(1, 16))
    with torch.no_grad():
        module[2].weight.add_(1)
    module._hf_hook.init_hook(module)
    assert module[2].weight.device.type == "cpu"
    assert torch.equal(module[2].weight.detach(), before + 1)


def test_overlapping_re_enables_restore_the_real_device_cache(monkeypatch):
    module, calls = _fake_pipeline_utils(monkeypatch)
    original = module.empty_device_cache
    lin = torch.nn.Linear(2, 2)
    lin.__dict__[dm._KEEP_ATTR] = {"pin_layout": [object()]}
    pipe = types.SimpleNamespace(components = {"t": lin})
    first, second = dm._keep_device_cache(pipe), dm._keep_device_cache(pipe)
    first.__enter__()
    second.__enter__()
    first.__exit__(None, None, None)  # the first out must not end the second's patch ...
    module.empty_device_cache("cuda")
    assert calls == []
    second.__exit__(None, None, None)  # ... nor may the last out leave a no-op behind
    assert module.empty_device_cache is original


def test_a_replaced_weight_leaves_no_stale_hole_in_the_chunk_upload(fake_pin):
    module = _stack()
    host = {n: p.data for n, p in module.named_parameters()}
    owner = dict(module.named_parameters())
    layout: list = []
    assert dm._pin_host_weights(module, host, layout = layout) > 0
    buf, entries = layout[0]
    assert len(entries) > 1
    first, last = entries[0][0], entries[-1][0]
    expect = {n: p.detach().clone() for n, p in module.named_parameters()}
    prefix, _, leaf = first.rpartition(".")
    setattr(module.get_submodule(prefix), leaf, torch.nn.Parameter(torch.zeros_like(expect[first])))
    copied = dm._upload_pinned_chunks(module, layout[:1], host, owner, {}, torch.device("cpu"))
    live = [view for name, _, view in entries if name != first]
    assert copied < max(start + view.nbytes for _, start, view in entries)
    assert copied <= sum(-(-v.nbytes // dm._PIN_ALIGN) * dm._PIN_ALIGN for v in live)
    params = dict(module.named_parameters())
    assert params[last].data_ptr() != host[last].data_ptr()  # re-pointed at the device chunk
    for name, _, _ in entries[1:]:
        assert torch.equal(params[name].detach(), expect[name])


def test_an_instance_level_mover_is_kept_and_used(fake_pin):
    pipe = _pipe("meta", "transformer")
    pipe.enable_model_cpu_offload()
    dm.keep_cpu_weights_on_offload(pipe)
    module = pipe.transformer
    calls: list = []
    stock = type(module).to

    def mover(*args, **kwargs):
        calls.append(args)
        return stock(module, *args, **kwargs)

    module.__dict__["to"] = mover  # e.g. a stable arena's placement
    _onload(pipe)
    assert module.__dict__[dm._KEEP_ATTR]["pin_layout"]  # the chunk path was armed ...
    assert calls  # ... yet the onload went through the installed mover
    assert module.__dict__.get("to") is mover


def test_tensors_left_out_of_the_layout_are_not_uploaded_with_the_chunk(fake_pin):
    # A torchao wrapper weight is pinned into the chunk but left out of its entries (the stock module.to moves it).
    module = _stack()
    host = {n: p.data for n, p in module.named_parameters()}
    owner = dict(module.named_parameters())
    layout: list = []
    assert dm._pin_host_weights(module, host, layout = layout) > 0
    buf, entries = layout[0]
    assert len(entries) > 1 and entries[0][1] == 0
    span = max(start + view.nbytes for _, start, view in entries)
    kept = entries[1:]
    expect = {name: view.clone() for name, _, view in kept}
    copied = dm._upload_pinned_chunks(module, [(buf, kept)], host, owner, {}, torch.device("cpu"))
    assert copied <= span - entries[1][1]
    params = dict(module.named_parameters())
    for name, _, view in kept:
        assert params[name].data_ptr() != view.data_ptr()
        assert torch.equal(params[name].detach(), expect[name])


def _layout_chunk_ptrs(module):
    layout = module.__dict__[dm._KEEP_ATTR].get("pin_layout") or []
    return {buf.data_ptr() for buf, _ in layout}


@pytest.mark.parametrize("fast_upload", ["1", "0"])
@pytest.mark.parametrize("cast_on", ["host", "device"])
def test_a_dtype_change_releases_the_obsolete_pinned_chunks(
    fake_pin, monkeypatch, fast_upload, cast_on
):
    # The SDXL fp32 VAE decode casts an offloaded component and back: its kept weights are new host tensors, so the
    # chunks pinned for the old ones must not stay referenced by the layout (fast upload on or off).
    monkeypatch.setenv(dm.OFFLOAD_FAST_UPLOAD_ENV, fast_upload)
    pipe = _pipe("cpu", "transformer")
    pipe.enable_model_cpu_offload()
    dm.keep_cpu_weights_on_offload(pipe)
    module = _onload(pipe)
    old = _layout_chunk_ptrs(module)
    assert old
    if cast_on == "device":
        module.half()  # cast while onloaded: the offload brings back new host tensors
        module._hf_hook.init_hook(module)
    else:
        module._hf_hook.init_hook(module)
        module.half()  # cast while offloaded: the next onload finds new host tensors
        _onload(pipe)
    assert not old & _layout_chunk_ptrs(module)
    state = module.__dict__[dm._KEEP_ATTR]
    live = {v.untyped_storage().data_ptr() for v in state["host"].values()}
    assert not old & live


def test_a_replaced_weight_drops_out_of_the_layout_and_the_rest_stays(fake_pin):
    pipe = _pipe("cpu", "transformer")
    pipe.enable_model_cpu_offload()
    dm.keep_cpu_weights_on_offload(pipe)
    module = _onload(pipe)
    module._hf_hook.init_hook(module)
    before = module.__dict__[dm._KEEP_ATTR]["pin_layout"]
    names = {name for _, entries in before for name, _, _ in entries}
    module[0].weight = torch.nn.Parameter(torch.zeros(16, 16))
    _onload(pipe)
    after = module.__dict__[dm._KEEP_ATTR]["pin_layout"]
    left = {name for _, entries in after for name, _, _ in entries}
    assert left == names - {"0.weight"}


def _quantised(scheme, n = N_LAYERS):
    """A Linear stack with torchao int8 / fp8 wrapper weights (Studio's quantised denoisers under low_vram)."""
    quant = pytest.importorskip("torchao.quantization")
    torch.manual_seed(0)
    module = torch.nn.Sequential(
        *(torch.nn.Linear(64, 64, bias = True, dtype = torch.bfloat16) for _ in range(n))
    )
    if scheme == "int8":
        try:
            config = quant.Int8DynamicActivationInt8WeightConfig(version = 2)
        except TypeError:
            config = quant.Int8DynamicActivationInt8WeightConfig()
    else:
        config = quant.Float8DynamicActivationFloat8WeightConfig(granularity = quant.PerRow())
    try:
        quant.quantize_(module, config)
    except Exception as exc:  # noqa: BLE001 - e.g. an fp8 build that needs a capable GPU
        pytest.skip(f"torchao {scheme} quantize unavailable here: {exc}")
    if dm._wrapper_inner(module[0].weight.data) is None:
        pytest.skip("this torchao build stores the weights without a traceable wrapper")
    return module


def _inner_state(t):
    return {attr: x.detach().clone() for attr, x in dm._wrapper_inner(t)}


@pytest.mark.parametrize("scheme", ["int8", "fp8"])
def test_wrapper_weights_ride_the_chunk_and_are_rebuilt_over_its_slices(fake_pin, scheme):
    module = _quantised(scheme)
    host = {n: p.data for n, p in module.named_parameters()}
    owner = dict(module.named_parameters())
    expect = {
        n: _inner_state(p.data) for n, p in module.named_parameters() if dm._wrapper_inner(p.data)
    }
    layout: list = []
    assert dm._pin_host_weights(module, host, layout = layout) > 0
    listed = [(name, view) for _, entries in layout for name, _, view in entries]
    for name in expect:
        # one layout entry per inner tensor, each at the pinned inner tensor the host weight now reads
        views = [view for n, view in listed if n == name]
        assert sorted(v.data_ptr() for v in views) == sorted(
            x.data_ptr() for _, x in dm._wrapper_inner(host[name])
        )
    params = dict(module.named_parameters())
    ids = {n: id(p) for n, p in params.items()}
    copied = dm._upload_pinned_chunks(module, layout, host, owner, {}, torch.device("cpu"))
    assert copied >= sum(x.nbytes for n in expect for _, x in dm._wrapper_inner(host[n]))
    chunk_ptrs = {buf.data_ptr() for buf, _ in layout}
    for name, inner in expect.items():
        p = dict(module.named_parameters())[name]
        assert id(p) == ids[name] and type(p) is type(
            host[name]
        )  # swapped in place, same Parameter object
        now = dict(dm._wrapper_inner(p.data))
        assert set(now) == set(inner)
        for attr, x in now.items():
            assert torch.equal(x, inner[attr])
            assert (
                x.data_ptr() not in chunk_ptrs
                and x.data_ptr() != dict(dm._wrapper_inner(host[name]))[attr].data_ptr()
            )
        assert p.shape == host[name].shape and p.dtype == host[name].dtype
    storages = {
        x.untyped_storage().data_ptr()
        for n in expect
        for _, x in dm._wrapper_inner(dict(module.named_parameters())[n].data)
    }
    assert len(storages) <= len(layout)


def test_a_wrapper_with_an_inner_tensor_outside_the_layout_takes_the_stock_copy(fake_pin):
    module = _quantised("int8", n = 2)
    host = {n: p.data for n, p in module.named_parameters()}
    owner = dict(module.named_parameters())
    layout: list = []
    assert dm._pin_host_weights(module, host, layout = layout) > 0
    name = "0.weight"
    drop = dm._wrapper_inner(host[name])[1][1]  # its scale
    trimmed = [
        (buf, [e for e in entries if not (e[0] == name and e[2].data_ptr() == drop.data_ptr())])
        for buf, entries in layout
    ]
    before = [x.data_ptr() for _, x in dm._wrapper_inner(module[0].weight.data)]
    dm._upload_pinned_chunks(module, trimmed, host, owner, {}, torch.device("cpu"))
    assert [
        x.data_ptr() for _, x in dm._wrapper_inner(module[0].weight.data)
    ] == before  # left on the host
    assert [x.data_ptr() for _, x in dm._wrapper_inner(module[1].weight.data)] != [
        x.data_ptr() for _, x in dm._wrapper_inner(host["1.weight"])
    ]


@pytest.mark.parametrize("scheme", ["int8", "fp8"])
def test_the_layout_prune_keeps_unchanged_wrapper_weights_and_drops_replaced_ones(fake_pin, scheme):
    # A wrapper weight's entries are its inner tensors: the prune must match those, not the wrapper itself.
    module = _quantised(scheme)
    host = {n: p.data for n, p in module.named_parameters()}
    layout: list = []
    assert dm._pin_host_weights(module, host, layout = layout) > 0
    host = {n: p.data for n, p in module.named_parameters()}
    names = [name for _, entries in layout for name, _, _ in entries]
    wrapped = {n for n in names if dm._wrapper_inner(host.get(n)) is not None}
    assert wrapped
    dm._prune_pin_layout(layout, host, {})
    assert [name for _, entries in layout for name, _, _ in entries] == names
    # a dequantised (dtype-changed) or re-quantised weight is a new host tensor: its entries go, the rest stay
    host["0.weight"] = host["0.weight"].dequantize()
    host["1.weight"] = _quantised(scheme, n = 2)[1].weight.data
    dm._prune_pin_layout(layout, host, {})
    left = [name for _, entries in layout for name, _, _ in entries]
    assert left == [n for n in names if n not in ("0.weight", "1.weight")]


def _quantised_pipe(scheme):
    pipe = _pipe("cuda", "text_encoder", "transformer")
    pipe.components["text_encoder"] = torch.nn.Linear(8, 64, dtype = torch.bfloat16)
    pipe.components["transformer"] = _quantised(scheme)
    pipe.text_encoder, pipe.transformer = (
        pipe.components["text_encoder"],
        pipe.components["transformer"],
    )
    return pipe


@cuda
@pytest.mark.parametrize("scheme", ["int8", "fp8"])
def test_quantised_chunk_onload_is_bit_identical_and_skips_the_per_tensor_copy(monkeypatch, scheme):
    # no_grad only: torchao weights cannot move under inference_mode, so Studio renders offloaded quant under no_grad.
    monkeypatch.setenv(dm.OFFLOAD_PIN_ENV, "1")
    monkeypatch.setattr(dm, "_PIN_CHUNK_BYTES", 4096)
    if scheme == "fp8" and torch.cuda.get_device_capability() < (8, 9):
        pytest.skip("fp8 matmul needs sm89+")
    ctx = torch.no_grad
    x = torch.randn(4, 8, dtype = torch.bfloat16)
    stock = _quantised_pipe(scheme)
    stock.enable_model_cpu_offload()
    ref = _render(stock, x, ctx)
    pipe = _quantised_pipe(scheme)
    pipe.enable_model_cpu_offload()
    dm.keep_cpu_weights_on_offload(pipe)
    assert torch.equal(_render(pipe, x, ctx), ref)  # the first render pins
    module = pipe.transformer
    layout = module.__dict__[dm._KEEP_ATTR]["pin_layout"]
    assert layout
    seen = []

    def spy(mod, *args):
        inner = [
            x
            for p in mod.parameters()
            if dm._wrapper_inner(p.data)
            for _, x in dm._wrapper_inner(p.data)
        ]
        assert inner and all(x.device.type == "cuda" for x in inner)
        seen.append({x.untyped_storage().data_ptr() for x in inner})

    module.register_forward_hook(spy)

    allocs = _device_allocs()
    for _ in range(3):
        assert torch.equal(_render(pipe, x, ctx), ref)
    # every inner tensor is a slice of one of the chunk allocations (per tensor: one storage each) ...
    assert all(len(s) <= len(layout) for s in seen) and len(seen) == 3
    # ... the chunks come back out of the caching allocator ...
    assert _device_allocs() - allocs <= 2
    # ... and every offload re-points the weights at their pinned host tensors
    host = module.__dict__[dm._KEEP_ATTR]["host"]
    for n, p in module.named_parameters():
        assert p.device.type == "cpu" and dm._tensor_ptr(p.data) == dm._tensor_ptr(host[n])
